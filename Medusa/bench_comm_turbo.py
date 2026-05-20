import argparse
import csv
import gc
import math
import os
import re
import sys
import time
from collections import defaultdict

import torch
from transformers import BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from medusa.model.medusa_model import MedusaModel, infer_model_context_window
from medusa.model.kv_cache import (
    OutlierAwareTurboVQKVCache,
    PolarQuantizedKVCache,
    TurboQuantizedKVCache,
    extract_outlier_calibration_indices,
    initialize_outlier_calibration_past_key_values,
    turbo_vq_attention_with_qjl_residual,
)


DEFAULT_MODEL_DIR = "Medusa/TinyLlama-1.1B-Chat-v1.0-4heads"


def profile_defaults(profile):
    profile = str(profile or "manual").lower()
    if profile == "tinyllama":
        return {
            "model_dir": DEFAULT_MODEL_DIR,
            "kv_max_length": 2048,
        }
    if profile == "llama32-long":
        return {
            "model_dir": os.environ.get(
                "LLAMA32_MEDUSA_MODEL_DIR",
                "Medusa/Llama-3.2-1B-Instruct-4heads",
            ),
            "kv_max_length": int(os.environ.get("LLAMA32_KV_MAX_LENGTH", "32768")),
            "long_context_tokens": int(os.environ.get("LLAMA32_LONG_CONTEXT_TOKENS", "8192")),
            "long_only": True,
            "target_new_tokens": 96,
            "max_steps": 64,
        }
    if profile == "llama32-128k":
        return {
            "model_dir": os.environ.get(
                "LLAMA32_MEDUSA_MODEL_DIR",
                "Medusa/Llama-3.2-1B-Instruct-4heads",
            ),
            "kv_max_length": int(os.environ.get("LLAMA32_KV_MAX_LENGTH", "131072")),
            "long_context_tokens": int(os.environ.get("LLAMA32_LONG_CONTEXT_TOKENS", "32768")),
            "long_only": True,
            "target_new_tokens": 128,
            "max_steps": 80,
            "use_model_context_kv": True,
        }
    return {}


def arg_was_provided(flag):
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def apply_profile_defaults(args):
    defaults = profile_defaults(args.profile)
    for dest, value in defaults.items():
        flag = f"--{dest.replace('_', '-')}"
        if not arg_was_provided(flag):
            setattr(args, dest, value)

BASE_PROMPTS = [
    (
        "hpc",
        "Write a concise C++ MPI+OpenMP blocked GEMM example and explain the overlap strategy.",
    ),
    (
        "systems",
        "Explain strong scaling versus weak scaling for distributed systems in practical terms.",
    ),
]

GENERAL_PROMPTS = [
    (
        "chat",
        "Give practical advice to a student who keeps procrastinating on a programming assignment.",
    ),
    (
        "summarization",
        "Summarize why renewable energy storage matters for electric grids in one compact paragraph.",
    ),
    (
        "reasoning",
        "A train leaves at 3 PM traveling 60 mph. Another leaves at 4 PM traveling 80 mph on the same route. When does the second catch the first?",
    ),
    (
        "code",
        "Write a small Python function that groups strings by their first letter and explain the edge cases.",
    ),
    (
        "creative",
        "Write a short atmospheric opening paragraph for a science fiction story set on a quiet moon base.",
    ),
    (
        "instruction",
        "Explain how to make a simple weekly study plan for learning machine learning while working part time.",
    ),
    (
        "technical_qa",
        "Explain the difference between latency and throughput using examples from web servers.",
    ),
    (
        "comparison",
        "Compare SQLite and PostgreSQL for a small analytics dashboard in practical terms.",
    ),
]

CODING_PROMPTS = [
    (
        "python_grouping",
        "Write a Python function that groups strings by their first letter. Include type hints and handle empty strings.",
    ),
    (
        "cuda_kernel",
        "Write a minimal CUDA C++ vector addition kernel and explain the grid-stride loop.",
    ),
    (
        "debugging",
        "A Python function mutates its default list argument across calls. Explain the bug and show a fixed implementation.",
    ),
    (
        "systems_code",
        "Write a concise C++ example that uses std::thread to split work across workers and joins them safely.",
    ),
]

LONG_CONTEXT_SEED = (
    "Cache locality, memory bandwidth, kernel launch overhead, branch prediction, "
    "NUMA placement, PCIe transfer, KV cache reuse, and asynchronous prefetching "
    "all affect CPU and GPU program performance. "
)

CODE_CONTEXT_SEED = (
    "Consider a codebase with Python data loaders, CUDA kernels, C++ worker pools, "
    "unit tests, benchmarks, error handling, memory ownership, and profiling notes. "
    "The implementation should prioritize correctness, readable control flow, "
    "stable APIs, and predictable performance. "
)


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def prefix_match(a, b):
    a = clean(a)
    b = clean(b)
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx / max(1, limit)


def sync():
    torch.cuda.synchronize()


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def estimate_fp16_kv_mb(config, kv_len):
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    bytes_total = layers * 2 * kv_heads * int(kv_len) * head_dim * 2
    return bytes_total / (1024**2)


def estimate_ideal_turbo_vq_kv_mb(
    config,
    kv_len,
    bits=8,
    key_bits=None,
    residual_dim=128,
    outlier_channels=0,
    outlier_bits=None,
    key_outlier_bits=None,
):
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    key_bits = int(bits if key_bits is None else key_bits)
    bits = int(bits)
    outlier_channels = int(max(0, outlier_channels))
    outlier_bits = bits if outlier_bits is None else int(outlier_bits)
    key_outlier_bits = key_bits if key_outlier_bits is None else int(key_outlier_bits)
    residual_dim = int(head_dim if int(residual_dim) < 0 else max(0, residual_dim))

    n_outlier = int(min(outlier_channels, max(0, head_dim - 1)))
    n_regular = head_dim - n_outlier
    key_bytes_per_vec = ((n_regular * key_bits) + (n_outlier * key_outlier_bits)) / 8.0
    key_bytes_per_vec += 2.0
    if residual_dim > 0:
        key_bytes_per_vec += (residual_dim / 8.0) + 2.0
    value_bytes_per_vec = ((n_regular * bits) + (n_outlier * outlier_bits)) / 8.0
    value_bytes_per_vec += 2.0
    bytes_total = layers * kv_heads * int(kv_len) * (key_bytes_per_vec + value_bytes_per_vec)
    return bytes_total / (1024**2)


def estimate_turbo_vq_storage(
    config,
    kv_len,
    bits=8,
    key_bits=None,
    residual_dim=128,
    outlier_channels=0,
    outlier_bits=None,
    key_outlier_bits=None,
):
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    kv_len = int(max(0, kv_len))
    bits = int(max(1, min(8, bits)))
    key_bits = bits if key_bits is None else int(max(1, min(8, key_bits)))
    outlier_channels = int(max(0, outlier_channels))
    outlier_bits = bits if outlier_bits is None else int(max(1, min(8, outlier_bits)))
    key_outlier_bits = (
        key_bits if key_outlier_bits is None else int(max(1, min(8, key_outlier_bits)))
    )
    residual_dim = int(head_dim if int(residual_dim) < 0 else max(0, residual_dim))

    n_outlier = int(min(outlier_channels, max(0, head_dim - 1)))
    n_regular = head_dim - n_outlier
    weighted_key_bits = (
        (n_regular * key_bits) + (n_outlier * key_outlier_bits)
    ) / max(1, head_dim)
    weighted_value_bits = (
        (n_regular * bits) + (n_outlier * outlier_bits)
    ) / max(1, head_dim)
    key_qidx_bytes = math.ceil(((n_regular * key_bits) + (n_outlier * key_outlier_bits)) / 8.0)
    value_qidx_bytes = math.ceil(((n_regular * bits) + (n_outlier * outlier_bits)) / 8.0)
    key_bytes_per_vec = key_qidx_bytes + 2.0
    if residual_dim > 0:
        key_bytes_per_vec += math.ceil(residual_dim / 8.0) + 2.0
    value_bytes_per_vec = value_qidx_bytes + 2.0

    compressed_bytes = layers * kv_heads * kv_len * (key_bytes_per_vec + value_bytes_per_vec)
    total_bytes = compressed_bytes
    fp16_bytes = layers * 2 * kv_heads * max(1, kv_len) * head_dim * 2
    residual_algorithmic_bits = 1.0 if residual_dim >= head_dim and residual_dim > 0 else (
        residual_dim / max(1, head_dim)
    )
    algorithmic_key_bits = weighted_key_bits + residual_algorithmic_bits
    algorithmic_value_bits = weighted_value_bits
    actual_bits_per_channel = (total_bytes * 8.0) / (
        layers * 2 * kv_heads * max(1, kv_len) * head_dim
    )
    return {
        "turbo_vq_kv_mb_est": total_bytes / (1024**2),
        "turbo_vq_compressed_kv_mb_est": compressed_bytes / (1024**2),
        "turbo_vq_transfer_reduction": fp16_bytes / max(1.0, total_bytes),
        "turbo_vq_key_algorithmic_bits": algorithmic_key_bits,
        "turbo_vq_value_bits": algorithmic_value_bits,
        "turbo_vq_actual_bits_per_channel_est": actual_bits_per_channel,
        "turbo_vq_key_qidx_bytes_per_token_head": key_qidx_bytes,
        "turbo_vq_value_qidx_bytes_per_token_head": value_qidx_bytes,
    }


def estimate_polar_paper_storage(config, kv_len, first_bits=4, other_bits=2, polar_levels=4):
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    polar_levels = int(max(1, min(int(polar_levels), int(math.log2(head_dim)))))
    final_dim = head_dim // (1 << polar_levels)
    level_bits = [int(first_bits)] + [int(other_bits)] * (polar_levels - 1)
    level_dims = [head_dim // (1 << level) for level in range(1, polar_levels + 1)]
    angle_bytes = sum(math.ceil(dim * bits / 8.0) for dim, bits in zip(level_dims, level_bits))
    radius_bytes = final_dim * 2
    bytes_per_vec = angle_bytes + radius_bytes
    total_bytes = layers * 2 * kv_heads * int(kv_len) * bytes_per_vec
    fp16_bytes = layers * 2 * kv_heads * max(1, int(kv_len)) * head_dim * 2
    algorithmic_bits = (
        final_dim * 16.0 + sum(dim * bits for dim, bits in zip(level_dims, level_bits))
    ) / head_dim
    return {
        "polar_kv_mb_est": total_bytes / (1024**2),
        "polar_transfer_reduction": fp16_bytes / max(1.0, total_bytes),
        "polar_algorithmic_bits_per_channel": algorithmic_bits,
        "polar_angle_bytes_per_token_head": angle_bytes,
        "polar_radius_bytes_per_token_head": radius_bytes,
    }


def make_token_sized_long_prompt(tokenizer, target_tokens, seed=LONG_CONTEXT_SEED, suffix=None):
    if suffix is None:
        suffix = "Now summarize the most important optimization bottlenecks in five bullets."
    if target_tokens <= 0:
        return None
    if tokenizer is None:
        repeat = max(1, int(target_tokens) // 24)
        return (seed * repeat) + suffix

    seed_tokens = tokenizer(
        seed,
        add_special_tokens=False,
    ).input_ids
    suffix_tokens = tokenizer(suffix, add_special_tokens=False).input_ids
    repeat = max(
        1,
        math.ceil(max(1, int(target_tokens) - len(suffix_tokens)) / max(1, len(seed_tokens))),
    )
    return (seed * repeat) + suffix


def build_prompts(
    long_repeat,
    long_only=False,
    prompt_suite="technical",
    tokenizer=None,
    long_context_tokens=0,
):
    if long_only:
        prompts = []
    elif prompt_suite == "general":
        prompts = list(GENERAL_PROMPTS)
    elif prompt_suite == "mixed":
        prompts = list(BASE_PROMPTS) + list(GENERAL_PROMPTS)
    elif prompt_suite == "coding":
        prompts = list(CODING_PROMPTS)
    else:
        prompts = list(BASE_PROMPTS)
    if int(long_context_tokens) > 0:
        seed = LONG_CONTEXT_SEED
        suffix = "Now summarize the most important optimization bottlenecks in five bullets."
        if prompt_suite == "coding":
            seed = CODE_CONTEXT_SEED
            suffix = (
                "Now write a compact Python module that implements a benchmark timer, "
                "validates inputs, and reports the fastest implementation."
            )
        prompts.append(
            (
                f"long_context_{int(long_context_tokens)}t",
                make_token_sized_long_prompt(
                    tokenizer,
                    int(long_context_tokens),
                    seed=seed,
                    suffix=suffix,
                ),
            )
        )
    elif long_repeat > 0:
        prompts.append(
            (
                "long_context",
                (LONG_CONTEXT_SEED * int(long_repeat))
                + "Now summarize the most important optimization bottlenecks in five bullets.",
            )
        )
    return prompts


def parse_int_csv(value):
    if not value:
        return []
    parsed = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(int(item))
    seen = set()
    unique = []
    for item in parsed:
        if item <= 0 or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def parse_adaptive_pairs(value):
    if not value:
        return []
    pairs = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        for sep in (":", "->", "-", "/"):
            if sep in item:
                left, right = item.split(sep, 1)
                break
        else:
            raise ValueError(
                f"Adaptive sweep item '{item}' must look like 16:24 or 16->32."
            )
        base_limit = int(left.strip())
        balanced_limit = int(right.strip())
        if base_limit <= 0 or balanced_limit <= 0:
            continue
        if balanced_limit <= base_limit:
            raise ValueError(
                f"Adaptive sweep item '{item}' must expand to a larger tree."
            )
        pairs.append((base_limit, balanced_limit))
    seen = set()
    unique = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        unique.append(pair)
    return unique


def build_static_tree_sweep_modes(args):
    choice_limits = parse_int_csv(args.choice_sweep)
    adaptive_pairs = parse_adaptive_pairs(args.adaptive_sweep)
    if not choice_limits and not adaptive_pairs:
        return None

    modes = [("medusa_base", {})]
    for limit in choice_limits:
        modes.append(
            (
                f"turbo_fast_{limit}",
                {
                    "turbo_fast_preset": True,
                    "medusa_choice_limit": int(limit),
                    "_use_model_choice_resolution": True,
                },
            )
        )
    for base_limit, balanced_limit in adaptive_pairs:
        modes.append(
            (
                f"turbo_adaptive_{base_limit}_{balanced_limit}",
                {
                    "turbo_fast_preset": True,
                    "medusa_choice_limit": int(base_limit),
                    "turbo_adaptive_tree": True,
                    "turbo_adaptive_tree_balanced_limit": int(balanced_limit),
                    "turbo_adaptive_tree_confidence_threshold": args.adaptive_tree_confidence_threshold,
                    "turbo_adaptive_tree_check_interval": args.adaptive_tree_check_interval,
                    "turbo_adaptive_tree_accept_threshold": args.adaptive_tree_accept_threshold,
                    "_use_model_choice_resolution": True,
                },
            )
        )
    return modes


def build_modes(args):
    static_tree_sweep_modes = build_static_tree_sweep_modes(args)
    if static_tree_sweep_modes is not None:
        modes = static_tree_sweep_modes
        if args.only:
            wanted = {item.strip() for item in args.only.split(",") if item.strip()}
            modes = [(name, kwargs) for name, kwargs in modes if name in wanted]
        return modes

    node_budget_name = f"nb{args.node_budget}" if args.node_budget > 0 else "nball"
    vq_bits = int(max(1, min(8, args.vq_bits)))
    vq_key_total_bits = int(args.vq_key_bits) if int(args.vq_key_bits) > 0 else vq_bits
    vq_key_total_bits = int(max(2, min(8, vq_key_total_bits)))
    vq_key_index_bits = max(1, vq_key_total_bits - 1)
    vq_bit_name = (
        f"k{vq_key_total_bits}v{vq_bits}"
        if vq_key_total_bits != vq_bits
        else f"b{vq_bits}"
    )
    vq_kwargs = {
        # TurboQuantprod Algorithm 2: a b-bit key quantizer uses a (b-1)-bit
        # TurboQuantmse stage plus a 1-bit QJL residual over the full head dim.
        "turbo_vq_key_bits": vq_key_index_bits,
        "turbo_vq_residual_dim": -1,
        "turbo_vq_residual_scale": float(args.residual_scale),
        "turbo_quant_seed": int(args.quant_seed),
    }
    outlier_channels = int(max(0, args.vq_outlier_channels))
    outlier_value_bits = int(max(1, min(8, args.vq_outlier_bits)))
    outlier_key_total_bits = (
        int(args.vq_key_outlier_bits)
        if int(args.vq_key_outlier_bits) > 0
        else outlier_value_bits
    )
    outlier_key_total_bits = int(max(2, min(8, outlier_key_total_bits)))
    outlier_key_index_bits = max(1, outlier_key_total_bits - 1)

    modes = [
        ("medusa_base", {}),
        (
            f"turboquant_prod_{vq_bit_name}_full_tree",
            {
                "turbo_quant": True,
                "turbo_kv_compression": True,
                "turbo_force_full_tree_fast_verifier": True,
                "turbo_kv_quant_mode": "turbo_vq",
                "turbo_vq_bits": vq_bits,
                "turbo_runtime_dequant_cache": False,
                **vq_kwargs,
            },
        ),
        (
            f"turboquant_prod_outlier_k{outlier_key_total_bits}v{outlier_value_bits}_c{outlier_channels}_full_tree",
            {
                "turbo_quant": True,
                "turbo_kv_compression": True,
                "turbo_force_full_tree_fast_verifier": True,
                "turbo_kv_quant_mode": "turbo_vq",
                # Paper KV recipe: regular channels use one fewer bit than
                # identified outlier channels. Keys still reserve one advertised
                # bit for the QJL residual stage.
                "turbo_vq_bits": max(1, outlier_value_bits - 1),
                "turbo_vq_key_bits": max(1, outlier_key_total_bits - 2),
                "turbo_vq_outlier_bits": outlier_value_bits,
                "turbo_vq_key_outlier_bits": outlier_key_index_bits,
                "turbo_vq_outlier_channels": outlier_channels,
                "turbo_vq_residual_dim": -1,
                "turbo_vq_residual_scale": float(args.residual_scale),
                "turbo_runtime_dequant_cache": False,
                "turbo_quant_seed": int(args.quant_seed),
            },
        ) if outlier_channels > 0 else None,
        (
            f"polarquant_l{args.polar_levels}_f{args.polar_first_bits}o{args.polar_other_bits}_full_tree",
            {
                "turbo_quant": True,
                "turbo_kv_compression": True,
                "turbo_force_full_tree_fast_verifier": True,
                "turbo_kv_quant_mode": "polar",
                "turbo_theta_bits": int(args.polar_first_bits),
                "turbo_radius_bits": int(args.polar_other_bits),
                "turbo_polar_levels": int(args.polar_levels),
                "turbo_runtime_dequant_cache": False,
                "turbo_quant_seed": int(args.quant_seed),
            },
        ),
        (
            f"qjl_prune_{node_budget_name}",
            {
                "turbo_quant": True,
                "turbo_kv_compression": False,
                "turbo_prune_node_budget": args.node_budget,
                "turbo_prune_keep": args.prune_keep,
                "turbo_prune_min": args.prune_min,
                "turbo_prune_max": args.prune_max,
                "turbo_prune_confidence_margin": args.prune_confidence_margin,
                "turbo_prune_prescreen_margin": args.prune_prescreen_margin,
                "turbo_prune_min_fraction": args.prune_min_fraction,
                "turbo_prune_min_node_fraction": args.prune_min_node_fraction,
                "turbo_prune_decisive_margin": args.prune_decisive_margin,
                "turbo_prune_decisive_keep": args.prune_decisive_keep,
                "turbo_fallback_accept_threshold": args.fallback_accept_threshold,
                "turbo_prune_acceptance_prune_threshold": args.prune_acceptance_prune_threshold,
                "turbo_prune_acceptance_keep_threshold": args.prune_acceptance_keep_threshold,
                "turbo_prune_acceptance_dynamic": args.prune_acceptance_dynamic,
                "turbo_prune_acceptance_dynamic_prune_min": args.prune_acceptance_dynamic_prune_min,
                "turbo_prune_acceptance_dynamic_prune_max": args.prune_acceptance_dynamic_prune_max,
                "turbo_prune_acceptance_dynamic_keep_min": args.prune_acceptance_dynamic_keep_min,
                "turbo_prune_acceptance_dynamic_keep_max": args.prune_acceptance_dynamic_keep_max,
                "turbo_prune_use_qjl": True,
                "turbo_qjl_dim": args.qjl_dim,
                "turbo_quant_seed": int(args.quant_seed),
            },
        ),
    ]
    modes = [mode for mode in modes if mode is not None]
    if args.quick:
        modes = modes[:3]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        modes = [(name, kwargs) for name, kwargs in modes if name in wanted]
    return modes


def safe_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def print_mode_summary(rows):
    by_mode = defaultdict(list)
    for row in rows:
        by_mode[row["mode"]].append(row)
    if not by_mode:
        return

    print("")
    print("summary by mode:")
    summary_rows = []
    for mode, mode_rows in by_mode.items():
        n = max(1, len(mode_rows))
        avg_tps = sum(safe_float(row.get("tps")) for row in mode_rows) / n
        avg_speedup = sum(safe_float(row.get("speedup_vs_base")) for row in mode_rows) / n
        avg_accept = sum(safe_float(row.get("accepted_tokens_per_step")) for row in mode_rows) / n
        avg_nodes = sum(safe_float(row.get("verified_nodes_per_step")) for row in mode_rows) / n
        avg_prefix = sum(safe_float(row.get("prefix_match_vs_base")) for row in mode_rows) / n
        adaptive_steps = sum(safe_float(row.get("stat_adaptive_tree_steps")) for row in mode_rows)
        balanced_steps = sum(safe_float(row.get("stat_adaptive_tree_balanced_steps")) for row in mode_rows)
        balanced_ratio = balanced_steps / max(1.0, adaptive_steps)
        summary_rows.append(
            {
                "mode": mode,
                "avg_tps": avg_tps,
                "avg_speedup": avg_speedup,
                "avg_accept": avg_accept,
                "avg_nodes": avg_nodes,
                "avg_prefix": avg_prefix,
                "balanced_ratio": balanced_ratio,
            }
        )
        print(
            mode,
            f"avg_tps={avg_tps:.2f}",
            f"speedup={avg_speedup:.3f}",
            f"accept/step={avg_accept:.3f}",
            f"nodes/step={avg_nodes:.1f}",
            f"prefix={avg_prefix:.3f}",
            f"balanced={balanced_ratio:.2f}",
        )

    candidates = [
        row
        for row in summary_rows
        if row["mode"] != "medusa_base" and row["avg_prefix"] >= 0.999
    ]
    if candidates:
        winner = max(candidates, key=lambda row: row["avg_tps"])
        print(
            "winner",
            winner["mode"],
            f"avg_tps={winner['avg_tps']:.2f}",
            f"accept/step={winner['avg_accept']:.3f}",
            f"nodes/step={winner['avg_nodes']:.1f}",
        )


def build_outlier_calibration_prompts(args, tokenizer):
    suite = args.prompt_suite
    if suite == "general":
        base = list(GENERAL_PROMPTS)
    elif suite == "mixed":
        base = list(BASE_PROMPTS) + list(GENERAL_PROMPTS)
    elif suite == "coding":
        base = list(CODING_PROMPTS)
    else:
        base = list(BASE_PROMPTS)

    texts = [prompt for _, prompt in base]
    if int(args.outlier_calibration_tokens) > 0:
        seed = CODE_CONTEXT_SEED if suite == "coding" else LONG_CONTEXT_SEED
        texts.append(
            make_token_sized_long_prompt(
                tokenizer,
                int(args.outlier_calibration_tokens),
                seed=seed,
                suffix="Summarize the implementation and performance implications.",
            )
        )
    return texts[: max(1, int(args.outlier_calibration_prompts))]


def calibrate_outlier_indices(model, args):
    n_outlier = int(max(0, args.vq_outlier_channels))
    if n_outlier <= 0 or int(args.outlier_calibration_prompts) <= 0:
        return None

    tokenizer = getattr(model, "tokenizer", None)
    texts = build_outlier_calibration_prompts(args, tokenizer)
    if not texts:
        return None

    max_tokens = int(max(16, args.outlier_calibration_tokens))
    safe_max_length = min(int(args.kv_max_length), max_tokens + 8)
    past_key_values, current_length_data = initialize_outlier_calibration_past_key_values(
        model.base_model,
        safe_max_length=safe_max_length,
    )
    old_mask = getattr(model.base_model.model, "medusa_mask", None)
    model.base_model.model.medusa_mask = None
    try:
        with torch.inference_mode():
            for text in texts:
                full_prompt = f"<|user|>\n{text}\n<|assistant|>\n"
                inputs = tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=safe_max_length,
                ).to("cuda")
                current_length_data.zero_()
                model.base_model.model(
                    input_ids=inputs.input_ids,
                    past_key_values=past_key_values,
                )
                sync()
    finally:
        model.base_model.model.medusa_mask = old_mask

    calibrated = extract_outlier_calibration_indices(past_key_values, n_outlier)
    gc.collect()
    torch.cuda.empty_cache()
    return calibrated


def _first_calibrated_idx(call_kwargs, cache_idx: int, head_dim: int):
    calibrated = call_kwargs.get("turbo_vq_outlier_indices")
    if calibrated is None:
        return None
    try:
        idx = calibrated[0][cache_idx]
    except (IndexError, TypeError):
        return None
    if idx is None:
        return None
    if torch.is_tensor(idx):
        idx = idx.detach().cpu().to(torch.long)
    else:
        idx = torch.tensor(list(idx), dtype=torch.long)
    valid = (idx >= 0) & (idx < int(head_dim))
    idx = torch.unique(idx[valid], sorted=True)
    return idx if idx.numel() > 0 else None


def compute_paper_quant_metrics(config, call_kwargs, args):
    if not bool(args.paper_metrics):
        return {}
    quant_mode = str(call_kwargs.get("turbo_kv_quant_mode", "")).lower()
    if quant_mode not in {"turbo_vq", "polar"}:
        return {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    tokens = int(max(16, args.paper_metric_tokens))
    queries_n = int(max(4, min(16, tokens // 2)))
    seed = int(call_kwargs.get("turbo_quant_seed", args.quant_seed))
    gen = torch.Generator(device="cpu")
    gen.manual_seed((20260512 + seed * 1_000_003 + head_dim * 31 + tokens) % (2**63 - 1))
    keys = torch.randn(1, 1, tokens, head_dim, generator=gen, dtype=torch.float32).to(device)
    values = torch.randn(1, 1, tokens, head_dim, generator=gen, dtype=torch.float32).to(device)
    queries = torch.randn(1, 1, queries_n, head_dim, generator=gen, dtype=torch.float32).to(device)
    exact_scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(float(head_dim))
    exact_attention = torch.softmax(exact_scores, dim=-1) @ values
    metrics = {"paper_metric_seed": seed, "paper_metric_tokens": tokens}

    if quant_mode == "turbo_vq":
        value_bits = int(call_kwargs.get("turbo_vq_bits", args.vq_bits))
        key_bits = int(call_kwargs.get("turbo_vq_key_bits", max(1, value_bits - 1)))
        outlier_channels = int(call_kwargs.get("turbo_vq_outlier_channels", 0))
        residual_dim = int(call_kwargs.get("turbo_vq_residual_dim", args.residual_dim))
        residual_scale = float(call_kwargs.get("turbo_vq_residual_scale", args.residual_scale))
        if outlier_channels > 0:
            key_cache = OutlierAwareTurboVQKVCache(
                batch_size=1,
                num_heads=1,
                max_length=tokens,
                head_dim=head_dim,
                device=device,
                dtype=torch.float32,
                current_length=torch.zeros((), dtype=torch.long),
                regular_bits=key_bits,
                outlier_bits=int(call_kwargs.get("turbo_vq_key_outlier_bits", key_bits + 1)),
                n_outlier=outlier_channels,
                residual_dim=residual_dim,
                residual_scale=residual_scale,
                runtime_dequant_cache=False,
                outlier_idx=_first_calibrated_idx(call_kwargs, 0, head_dim),
                quant_seed=seed,
            )
            value_cache = OutlierAwareTurboVQKVCache(
                batch_size=1,
                num_heads=1,
                max_length=tokens,
                head_dim=head_dim,
                device=device,
                dtype=torch.float32,
                current_length=torch.zeros((), dtype=torch.long),
                regular_bits=value_bits,
                outlier_bits=int(call_kwargs.get("turbo_vq_outlier_bits", value_bits + 1)),
                n_outlier=outlier_channels,
                residual_dim=0,
                runtime_dequant_cache=False,
                outlier_idx=_first_calibrated_idx(call_kwargs, 1, head_dim),
                quant_seed=seed,
            )
        else:
            key_cache = TurboQuantizedKVCache(
                batch_size=1,
                num_heads=1,
                max_length=tokens,
                head_dim=head_dim,
                device=device,
                dtype=torch.float32,
                current_length=torch.zeros((), dtype=torch.long),
                bits=key_bits,
                residual_dim=residual_dim,
                residual_scale=residual_scale,
                runtime_dequant_cache=False,
                quant_seed=seed,
            )
            value_cache = TurboQuantizedKVCache(
                batch_size=1,
                num_heads=1,
                max_length=tokens,
                head_dim=head_dim,
                device=device,
                dtype=torch.float32,
                current_length=torch.zeros((), dtype=torch.long),
                bits=value_bits,
                residual_dim=0,
                runtime_dequant_cache=False,
                quant_seed=seed,
            )

        key_cache.cat(keys)
        value_cache.cat(values)
        decoded_keys = key_cache._decode_range(0, tokens).to(torch.float32)
        decoded_values = value_cache._decode_range(0, tokens).to(torch.float32)
        true_ip = torch.matmul(queries, keys.transpose(2, 3))
        mse_ip = torch.matmul(queries, decoded_keys.transpose(2, 3))
        qjl_attention = turbo_vq_attention_with_qjl_residual(
            queries,
            key_cache,
            value_cache,
            attention_mask=None,
            num_key_value_groups=1,
            head_dim=head_dim,
        )
        if qjl_attention is None:
            qjl_attention = torch.softmax(mse_ip / math.sqrt(float(head_dim)), dim=-1) @ decoded_values

        metrics.update(
            {
                "paper_turbo_mse_rel": float(
                    ((keys - decoded_keys).pow(2).sum(dim=-1) / keys.pow(2).sum(dim=-1).clamp_min(1e-8)).mean().item()
                ),
                "paper_turbo_prod_ip_mse_mse": float((mse_ip - true_ip).pow(2).mean().item()),
                "paper_turbo_attention_mse": float((qjl_attention.to(torch.float32) - exact_attention).pow(2).mean().item()),
            }
        )
        if isinstance(key_cache, TurboQuantizedKVCache) and key_cache.residual_proj is not None:
            residual_sign = key_cache._unpack_residual_sign_range(0, tokens)
            residual_norm = key_cache.residual_norm[:, :, :tokens, 0].to(torch.float32)
            residual_inner = torch.einsum(
                "bhqm,bhkm->bhqk",
                queries @ key_cache.residual_proj,
                residual_sign,
            )
            qjl_ip = mse_ip + key_cache.residual_coeff * residual_inner * residual_norm.unsqueeze(2)
            metrics["paper_turbo_prod_ip_mse_qjl"] = float((qjl_ip - true_ip).pow(2).mean().item())
            metrics["paper_turbo_prod_ip_bias"] = float((qjl_ip - true_ip).mean().item())

    if quant_mode == "polar":
        key_cache = PolarQuantizedKVCache(
            batch_size=1,
            num_heads=1,
            max_length=tokens,
            head_dim=head_dim,
            device=device,
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            first_level_bits=int(call_kwargs.get("turbo_theta_bits", args.polar_first_bits)),
            other_level_bits=int(call_kwargs.get("turbo_radius_bits", args.polar_other_bits)),
            polar_levels=int(call_kwargs.get("turbo_polar_levels", args.polar_levels)),
            runtime_dequant_cache=False,
            quant_seed=seed,
        )
        value_cache = PolarQuantizedKVCache(
            batch_size=1,
            num_heads=1,
            max_length=tokens,
            head_dim=head_dim,
            device=device,
            dtype=torch.float32,
            current_length=torch.zeros((), dtype=torch.long),
            first_level_bits=int(call_kwargs.get("turbo_theta_bits", args.polar_first_bits)),
            other_level_bits=int(call_kwargs.get("turbo_radius_bits", args.polar_other_bits)),
            polar_levels=int(call_kwargs.get("turbo_polar_levels", args.polar_levels)),
            runtime_dequant_cache=False,
            quant_seed=seed,
        )
        key_cache.cat(keys)
        value_cache.cat(values)
        decoded_keys = key_cache._decode_range(0, tokens).to(torch.float32)
        decoded_values = value_cache._decode_range(0, tokens).to(torch.float32)
        approx_scores = torch.matmul(queries, decoded_keys.transpose(2, 3)) / math.sqrt(float(head_dim))
        approx_attention = torch.softmax(approx_scores, dim=-1) @ decoded_values
        metrics.update(
            {
                "paper_polar_recon_rel": float(
                    ((keys - decoded_keys).pow(2).sum(dim=-1) / keys.pow(2).sum(dim=-1).clamp_min(1e-8)).mean().item()
                ),
                "paper_polar_attention_mse": float((approx_attention - exact_attention).pow(2).mean().item()),
            }
        )

    return metrics


def run_one(model, prompt, medusa_choices, args, mode, kwargs):
    reset_memory()
    sync()
    full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = model.tokenizer(full_prompt, return_tensors="pt").to("cuda")
    text = ""
    stats = {}
    first = None
    call_kwargs = dict(kwargs)
    use_model_choice_resolution = bool(call_kwargs.pop("_use_model_choice_resolution", False))
    call_kwargs["stream"] = args.stream
    call_kwargs["collect_stats"] = args.collect_stats
    call_kwargs.setdefault("draft_head_type", args.draft_head_type)
    call_kwargs.setdefault("tree_policy", args.tree_policy)
    call_kwargs.setdefault("tree_calibration_path", args.tree_calibration_path)
    call_kwargs.setdefault("turbo_kv_use_model_context", args.use_model_context_kv)
    call_medusa_choices = None if use_model_choice_resolution else medusa_choices

    start = time.perf_counter()
    with torch.inference_mode():
        for out in model.medusa_generate(
            inputs.input_ids,
            medusa_choices=call_medusa_choices,
            temperature=0.0,
            max_steps=args.max_steps,
            max_new_tokens=args.target_new_tokens,
            sampling="typical",
            fast=True,
            turbo_kv_max_length=args.kv_max_length,
            **call_kwargs,
        ):
            if first is None:
                sync()
                first = time.perf_counter()
            text = out["text"]
            stats = out.get("stats", stats)
    sync()
    end = time.perf_counter()

    exact_generated_tokens = int(stats.get("generated_tokens", 0) or 0)
    if exact_generated_tokens > 0:
        tokens = exact_generated_tokens
    else:
        tokens = max(1, len(model.tokenizer(text, add_special_tokens=False).input_ids))
    prompt_tokens = int(inputs.input_ids.shape[1])
    kv_len = prompt_tokens + tokens
    fp16_kv_mb = estimate_fp16_kv_mb(model.config, kv_len)
    ideal_vq_kv_mb = estimate_ideal_turbo_vq_kv_mb(
        model.config,
        kv_len,
        bits=int(call_kwargs.get("turbo_vq_bits", 8)),
        key_bits=call_kwargs.get("turbo_vq_key_bits"),
        residual_dim=int(call_kwargs.get("turbo_vq_residual_dim", args.residual_dim)),
        outlier_channels=int(call_kwargs.get("turbo_vq_outlier_channels", 0)),
        outlier_bits=call_kwargs.get("turbo_vq_outlier_bits"),
        key_outlier_bits=call_kwargs.get("turbo_vq_key_outlier_bits"),
    )
    turbo_vq_storage = estimate_turbo_vq_storage(
        model.config,
        kv_len,
        bits=int(call_kwargs.get("turbo_vq_bits", 8)),
        key_bits=call_kwargs.get("turbo_vq_key_bits"),
        residual_dim=int(call_kwargs.get("turbo_vq_residual_dim", args.residual_dim)),
        outlier_channels=int(call_kwargs.get("turbo_vq_outlier_channels", 0)),
        outlier_bits=call_kwargs.get("turbo_vq_outlier_bits"),
        key_outlier_bits=call_kwargs.get("turbo_vq_key_outlier_bits"),
    )
    polar_storage = estimate_polar_paper_storage(
        model.config,
        kv_len,
        first_bits=int(call_kwargs.get("turbo_theta_bits", args.polar_first_bits)),
        other_bits=int(call_kwargs.get("turbo_radius_bits", args.polar_other_bits)),
        polar_levels=int(call_kwargs.get("turbo_polar_levels", args.polar_levels)),
    )
    model_context_window = infer_model_context_window(
        model.config,
        tokenizer=getattr(model, "tokenizer", None),
    )

    row = {
        "mode": mode,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "tokens_per_step_cap": tokens / max(1, args.max_steps),
        "total_s": end - start,
        "ttft_s": (first or end) - start,
        "tps": tokens / max(1e-6, end - start),
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        "model_context_window": model_context_window,
        "context_utilization": kv_len / max(1, model_context_window),
        "fp16_kv_mb_est": fp16_kv_mb,
        "ideal_turbo_vq_kv_mb_est": ideal_vq_kv_mb,
        "ideal_turbo_vq_transfer_reduction": fp16_kv_mb / max(1e-6, ideal_vq_kv_mb),
        **turbo_vq_storage,
        **polar_storage,
        "text": text,
    }
    for key, value in stats.items():
        row[f"stat_{key}"] = value
    if stats.get("decode_steps"):
        row["accepted_tokens_per_step"] = tokens / max(1, int(stats["decode_steps"]))
        row["verified_nodes_per_step"] = int(stats.get("verified_tree_nodes", 0)) / max(
            1, int(stats["decode_steps"])
        )
    else:
        row["accepted_tokens_per_step"] = ""
        row["verified_nodes_per_step"] = ""
    row.update(compute_paper_quant_metrics(model.config, call_kwargs, args))
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dense KV, true TurboQuantprod, recursive PolarQuant, and QJL pruning."
    )
    parser.add_argument(
        "--profile",
        choices=("manual", "tinyllama", "llama32-long", "llama32-128k"),
        default="manual",
        help=(
            "Apply benchmark defaults. llama32-* profiles use longer prompts and "
            "larger KV allocations so TurboQuantprod and PolarQuant are exercised."
        ),
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-csv", default="Medusa/comm_turbo_benchmark.csv")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument(
        "--target-new-tokens",
        type=int,
        default=0,
        help=(
            "Stop each generation after this many generated token IDs. "
            "Use this for fair tree-size sweeps where accepted tokens/step differs."
        ),
    )
    parser.add_argument("--kv-max-length", type=int, default=2048)
    parser.add_argument(
        "--use-model-context-kv",
        action="store_true",
        help="Preallocate KV cache to at least the model/tokenizer context window.",
    )
    parser.add_argument("--long-repeat", type=int, default=0)
    parser.add_argument(
        "--long-context-tokens",
        type=int,
        default=0,
        help="Append a tokenizer-sized long-context prompt of roughly this many prompt tokens.",
    )
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--prompt-suite", choices=("technical", "general", "mixed", "coding"), default="technical")
    parser.add_argument("--choice-max-depth", type=int, default=0)
    parser.add_argument("--choice-limit", type=int, default=0)
    parser.add_argument(
        "--choice-sweep",
        default="",
        help=(
            "Comma-separated static choice limits to benchmark as turbo_fast_N. "
            "When set, only medusa_base plus these static/adaptive tree modes run."
        ),
    )
    parser.add_argument(
        "--adaptive-sweep",
        default="",
        help=(
            "Comma-separated adaptive static tree pairs like 16:24,16:32. "
            "The first limit is the default tree; the second is used on low-confidence steps."
        ),
    )
    parser.add_argument("--adaptive-tree-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--adaptive-tree-check-interval", type=int, default=4)
    parser.add_argument("--adaptive-tree-accept-threshold", type=float, default=0.0)
    parser.add_argument("--node-budget", type=int, default=40)
    parser.add_argument("--prune-keep", type=int, default=16)
    parser.add_argument("--prune-min", type=int, default=12)
    parser.add_argument("--prune-max", type=int, default=24)
    parser.add_argument("--prune-confidence-margin", type=float, default=0.50)
    parser.add_argument("--prune-prescreen-margin", type=float, default=-1.0)
    parser.add_argument("--prune-min-fraction", type=float, default=0.0)
    parser.add_argument("--prune-min-node-fraction", type=float, default=0.15)
    parser.add_argument("--prune-decisive-margin", type=float, default=1.5)
    parser.add_argument("--prune-decisive-keep", type=int, default=8)
    parser.add_argument("--fallback-accept-threshold", type=int, default=0)
    parser.add_argument("--prune-acceptance-prune-threshold", type=float, default=0.0)
    parser.add_argument("--prune-acceptance-keep-threshold", type=float, default=0.0)
    parser.add_argument("--prune-acceptance-dynamic", action="store_true")
    parser.add_argument("--prune-acceptance-dynamic-prune-min", type=float, default=0.10)
    parser.add_argument("--prune-acceptance-dynamic-prune-max", type=float, default=0.45)
    parser.add_argument("--prune-acceptance-dynamic-keep-min", type=float, default=0.45)
    parser.add_argument("--prune-acceptance-dynamic-keep-max", type=float, default=0.70)
    parser.add_argument("--qjl-dim", type=int, default=64)
    parser.add_argument("--residual-dim", type=int, default=128)
    parser.add_argument("--polar-levels", type=int, default=4)
    parser.add_argument("--polar-first-bits", type=int, default=4)
    parser.add_argument("--polar-other-bits", type=int, default=2)
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=1.0,
        help="Multiplier for TurboQuantprod QJL residual correction; 1.0 is the unbiased paper estimator.",
    )
    parser.add_argument(
        "--vq-bits",
        type=int,
        default=4,
        help="TurboQuant value bit width and default total key bit width.",
    )
    parser.add_argument(
        "--vq-key-bits",
        type=int,
        default=0,
        help="Optional total TurboQuantprod key bit width; 0 means use --vq-bits. Keys use b-1 MSE bits plus a full-head 1-bit QJL residual.",
    )
    parser.add_argument(
        "--vq-outlier-channels",
        type=int,
        default=16,
        help="Outlier channels for the paper TurboQuant KV mode; 0 disables that extra mode.",
    )
    parser.add_argument(
        "--vq-outlier-bits",
        type=int,
        default=4,
        help="High-precision outlier value bit width for the paper TurboQuant KV mode.",
    )
    parser.add_argument(
        "--vq-key-outlier-bits",
        type=int,
        default=0,
        help="Total high-precision outlier key bit width; 0 means use --vq-outlier-bits.",
    )
    parser.add_argument(
        "--outlier-calibration-prompts",
        type=int,
        default=4,
        help="Untimed calibration prompts used to freeze TurboQuant outlier channels; 0 disables separate calibration.",
    )
    parser.add_argument(
        "--outlier-calibration-tokens",
        type=int,
        default=512,
        help="Maximum tokens per outlier-calibration prompt.",
    )
    parser.add_argument(
        "--quant-seed",
        type=int,
        default=0,
        help="Seed for TurboQuant rotations and QJL projections. Use with --fresh-quant-randomness for per-run random matrices.",
    )
    parser.add_argument(
        "--fresh-quant-randomness",
        action="store_true",
        help="Replace --quant-seed with a fresh per-run seed before building caches.",
    )
    parser.add_argument(
        "--paper-metrics",
        dest="paper_metrics",
        action="store_true",
        default=True,
        help="Add small paper-style reconstruction, inner-product, and attention-error metrics to the CSV.",
    )
    parser.add_argument("--no-paper-metrics", dest="paper_metrics", action="store_false")
    parser.add_argument("--paper-metric-tokens", type=int, default=64)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--device-map",
        default="",
        help="Optional transformers device_map. Use 'auto' for quantized Kaggle runs.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-stats", dest="collect_stats", action="store_false")
    parser.set_defaults(collect_stats=True)
    parser.add_argument("--draft-head-type", choices=("medusa", "hydra"), default="medusa")
    parser.add_argument("--tree-policy", choices=("fixed", "adaptive_calibrated"), default="fixed")
    parser.add_argument("--tree-calibration-path", default="")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    apply_profile_defaults(args)
    if args.fresh_quant_randomness:
        args.quant_seed = int(time.time_ns() % (2**31 - 1))

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run outside the sandbox or on a GPU machine.")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    load_kwargs = {"torch_dtype": torch.float16}
    if args.load_in_8bit and args.load_in_4bit:
        raise SystemExit("Choose only one of --load-in-8bit or --load-in-4bit.")
    if args.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["medusa_head"],
        )
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["medusa_head"],
        )
    if args.device_map or args.load_in_8bit or args.load_in_4bit:
        load_kwargs["device_map"] = args.device_map or "auto"
        load_kwargs["low_cpu_mem_usage"] = True

    model = MedusaModel.from_pretrained(args.model_dir, **load_kwargs)
    if not (args.device_map or args.load_in_8bit or args.load_in_4bit):
        model = model.to("cuda")
    elif hasattr(model, "medusa_head"):
        # Keep custom Medusa heads as regular fp16 CUDA modules. Quantizing them
        # with bitsandbytes can produce device-map mismatches on Colab/T4.
        model.medusa_head.to(device="cuda", dtype=torch.float16)
    model = model.eval()
    raw_choices = model.get_medusa_choice(model.base_model_name_or_path)
    max_choice_depth = int(args.choice_max_depth) if int(args.choice_max_depth) > 0 else int(getattr(model, "medusa", 1))
    medusa_choices = [
        tuple(path) for path in raw_choices if len(path) <= max_choice_depth
    ]
    if int(args.choice_limit) > 0:
        medusa_choices = medusa_choices[: int(args.choice_limit)]
    print(
        "model",
        args.model_dir,
        "profile",
        args.profile,
        "choices",
        len(medusa_choices),
        "max_depth",
        max(len(path) for path in medusa_choices),
        "max_steps",
        args.max_steps,
        "kv_max_length",
        args.kv_max_length,
        "use_model_context_kv",
        int(bool(args.use_model_context_kv)),
        "long_context_tokens",
        args.long_context_tokens,
        "quant_seed",
        args.quant_seed,
    )

    prompts = build_prompts(
        args.long_repeat,
        long_only=args.long_only,
        prompt_suite=args.prompt_suite,
        tokenizer=getattr(model, "tokenizer", None),
        long_context_tokens=args.long_context_tokens,
    )
    modes = build_modes(args)
    if not modes:
        raise SystemExit("No modes selected.")

    outlier_indices = None
    if any(int(kwargs.get("turbo_vq_outlier_channels", 0)) > 0 for _, kwargs in modes):
        outlier_indices = calibrate_outlier_indices(model, args)
        if outlier_indices is not None:
            for _, kwargs in modes:
                if int(kwargs.get("turbo_vq_outlier_channels", 0)) > 0:
                    kwargs["turbo_vq_outlier_indices"] = outlier_indices
            sample = outlier_indices[0][0].tolist() if outlier_indices and outlier_indices[0][0] is not None else []
            print(
                "outlier calibration",
                f"prompts={int(args.outlier_calibration_prompts)}",
                f"tokens={int(args.outlier_calibration_tokens)}",
                f"channels={int(args.vq_outlier_channels)}",
                "layer0_key",
                sample,
            )
        else:
            print("outlier calibration disabled; outlier cache will select channels from first prefill batch")

    warm_modes = modes
    for _, kwargs in warm_modes:
        run_one(model, "Say hello in one sentence.", medusa_choices, args, "warmup", kwargs)

    rows = []
    base_by_category = {}
    for category, prompt in prompts:
        base_tps = None
        for mode, kwargs in modes:
            row = run_one(model, prompt, medusa_choices, args, mode, kwargs)
            row["category"] = category
            if mode == "medusa_base":
                base_by_category[category] = row["text"]
                base_tps = float(row["tps"])
                row["prefix_match_vs_base"] = 1.0
                row["speedup_vs_base"] = 1.0
            else:
                row["prefix_match_vs_base"] = prefix_match(base_by_category.get(category, ""), row["text"])
                row["speedup_vs_base"] = (
                    float(row["tps"]) / max(1e-6, base_tps)
                    if base_tps is not None
                    else ""
                )
            rows.append(row)
            print(
                category,
                mode,
                f"{float(row['tps']):.2f} TPS",
                "speedup",
                f"{float(row['speedup_vs_base'] or 0):.3f}",
                "prefix",
                f"{float(row['prefix_match_vs_base']):.3f}",
                "alloc",
                f"{float(row['peak_alloc_mb']):.1f} MB",
                "nodes/step",
                row.get("verified_nodes_per_step", ""),
            )

    preferred = [
        "category",
        "mode",
        "tokens",
        "prompt_tokens",
        "tokens_per_step_cap",
        "accepted_tokens_per_step",
        "verified_nodes_per_step",
        "total_s",
        "ttft_s",
        "tps",
        "speedup_vs_base",
        "prefix_match_vs_base",
        "peak_alloc_mb",
        "peak_reserved_mb",
        "model_context_window",
        "context_utilization",
        "fp16_kv_mb_est",
        "ideal_turbo_vq_kv_mb_est",
        "ideal_turbo_vq_transfer_reduction",
        "turbo_vq_kv_mb_est",
        "turbo_vq_compressed_kv_mb_est",
        "turbo_vq_transfer_reduction",
        "turbo_vq_key_algorithmic_bits",
        "turbo_vq_value_bits",
        "turbo_vq_actual_bits_per_channel_est",
        "turbo_vq_key_qidx_bytes_per_token_head",
        "turbo_vq_value_qidx_bytes_per_token_head",
        "polar_kv_mb_est",
        "polar_transfer_reduction",
        "polar_algorithmic_bits_per_channel",
        "polar_angle_bytes_per_token_head",
        "polar_radius_bytes_per_token_head",
        "paper_metric_seed",
        "paper_metric_tokens",
        "paper_turbo_mse_rel",
        "paper_turbo_prod_ip_mse_mse",
        "paper_turbo_prod_ip_mse_qjl",
        "paper_turbo_prod_ip_bias",
        "paper_turbo_attention_mse",
        "paper_polar_recon_rel",
        "paper_polar_attention_mse",
    ]
    stat_keys = sorted({key for row in rows for key in row if key.startswith("stat_")})
    fields = preferred + stat_keys + ["text"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print_mode_summary(rows)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
