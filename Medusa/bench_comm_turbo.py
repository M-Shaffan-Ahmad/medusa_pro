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


DEFAULT_MODEL_DIR = "Medusa/TinyLlama-1.1B-Chat-v1.0-4heads"


def profile_defaults(profile):
    profile = str(profile or "manual").lower()
    if profile == "tinyllama":
        return {
            "model_dir": DEFAULT_MODEL_DIR,
            "kv_max_length": 2048,
            "kv_qjl_min_kv_len": 2048,
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
            "kv_qjl_min_kv_len": 4096,
            "hot_window": 1024,
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
            "kv_qjl_min_kv_len": 8192,
            "hot_window": 2048,
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


def estimate_ideal_turbo_vq_kv_mb(config, kv_len, bits=8, key_bits=None, residual_dim=128):
    layers = int(config.num_hidden_layers)
    kv_heads = int(config.num_key_value_heads)
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    key_bits = int(bits if key_bits is None else key_bits)
    bits = int(bits)
    residual_dim = int(max(0, residual_dim))

    key_bytes_per_vec = (head_dim * key_bits / 8.0) + 2.0
    if residual_dim > 0:
        key_bytes_per_vec += (residual_dim / 8.0) + 2.0
    value_bytes_per_vec = (head_dim * bits / 8.0) + 2.0
    bytes_total = layers * kv_heads * int(kv_len) * (key_bytes_per_vec + value_bytes_per_vec)
    return bytes_total / (1024**2)


def estimate_packed_kv_qjl_sidecar_mb(config, kv_len, sketch_dim):
    kv_heads = int(config.num_key_value_heads)
    sketch_dim = int(max(0, sketch_dim))
    if sketch_dim <= 0:
        return 0.0
    # Packed KV-QJL sketches are enabled for one key-cache layer in the current
    # planner: kv_heads * sequence length * sketch_dim bits.
    bytes_total = kv_heads * int(kv_len) * (sketch_dim / 8.0)
    return bytes_total / (1024**2)


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
    modes = [
        ("medusa_base", {}),
        (
            "turbo_best_full_tree",
            {
                "turbo_quant": True,
                "turbo_kv_compression": False,
                "turbo_force_full_tree_fast_verifier": True,
            },
        ),
        (
            "turbo_best_full_tree_fused",
            {
                "turbo_quant": True,
                "turbo_kv_compression": False,
                "turbo_force_full_tree_fast_verifier": True,
                "turbo_fused_lm_head_argmax": True,
            },
        ),
        (
            "turbo_fast_24",
            {
                "turbo_fast_preset": True,
                "_use_model_choice_resolution": True,
            },
        ),
        (
            "turbo_auto",
            {
                "turbo_auto": True,
                "turbo_adaptive_tree_confidence_threshold": args.adaptive_tree_confidence_threshold,
                "turbo_adaptive_tree_check_interval": args.adaptive_tree_check_interval,
                "turbo_adaptive_tree_accept_threshold": args.adaptive_tree_accept_threshold,
                "_use_model_choice_resolution": True,
            },
        ),
        (
            "turbo_fast_24_fused",
            {
                "turbo_fast_preset": True,
                "turbo_fused_lm_head_argmax": True,
                "_use_model_choice_resolution": True,
            },
        ),
        (
            "turbo_adaptive_24_32",
            {
                "turbo_fast_preset": True,
                "turbo_adaptive_tree": True,
                "turbo_adaptive_tree_balanced_limit": 32,
                "turbo_adaptive_tree_confidence_threshold": args.adaptive_tree_confidence_threshold,
                "turbo_adaptive_tree_check_interval": args.adaptive_tree_check_interval,
                "_use_model_choice_resolution": True,
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
            },
        ),
        (
            f"packed_kv_qjl_{node_budget_name}",
            {
                "turbo_quant": True,
                "turbo_kv_compression": False,
                "turbo_prune_use_kv_qjl": True,
                "turbo_prune_use_qjl": False,
                "turbo_kv_qjl_dim": args.kv_qjl_dim,
                "turbo_kv_qjl_layer": args.kv_qjl_layer,
                "turbo_kv_qjl_keep_fraction": args.kv_qjl_keep_fraction,
                "turbo_kv_qjl_weight": args.kv_qjl_weight,
                "turbo_kv_qjl_min_kv_len": args.kv_qjl_min_kv_len,
                "turbo_kv_qjl_medusa_pool_fraction": args.kv_qjl_medusa_pool_fraction,
                "turbo_kv_qjl_medusa_anchor_keep": args.kv_qjl_medusa_anchor_keep,
                "turbo_packed_kv_qjl_auto_disable_after": args.kv_qjl_auto_disable_after,
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
            },
        ),
        (
            f"packed_kv_qjl_strict_{node_budget_name}",
            {
                "turbo_quant": True,
                "turbo_kv_compression": False,
                "turbo_prune_use_kv_qjl": True,
                "turbo_prune_use_qjl": False,
                "turbo_fallback_full_tree": False,
                "turbo_kv_qjl_dim": args.kv_qjl_dim,
                "turbo_kv_qjl_layer": args.kv_qjl_layer,
                "turbo_kv_qjl_keep_fraction": args.kv_qjl_keep_fraction,
                "turbo_kv_qjl_weight": args.kv_qjl_weight,
                "turbo_kv_qjl_min_kv_len": args.kv_qjl_min_kv_len,
                "turbo_kv_qjl_medusa_pool_fraction": args.kv_qjl_medusa_pool_fraction,
                "turbo_kv_qjl_medusa_anchor_keep": args.kv_qjl_medusa_anchor_keep,
                "turbo_packed_kv_qjl_auto_disable_after": args.kv_qjl_auto_disable_after,
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
            },
        ),
        (
            "turbo_vq_shadow_b8_full_tree",
            {
                "turbo_quant": True,
                "turbo_kv_compression": True,
                "turbo_force_full_tree_fast_verifier": True,
                "turbo_kv_quant_mode": "turbo_vq",
                "turbo_vq_bits": 8,
                "turbo_vq_residual_dim": args.residual_dim,
                "turbo_runtime_dequant_cache": True,
            },
        ),
        (
            "turbo_vq_strict_b8_full_tree",
            {
                "turbo_quant": True,
                "turbo_kv_compression": True,
                "turbo_force_full_tree_fast_verifier": True,
                "turbo_kv_quant_mode": "turbo_vq",
                "turbo_vq_bits": 8,
                "turbo_vq_residual_dim": args.residual_dim,
                "turbo_runtime_dequant_cache": False,
            },
        ),
        (
            f"hybrid_vq_h{args.hot_window}_full_tree",
            {
                "turbo_quant": True,
                "turbo_kv_compression": True,
                "turbo_force_full_tree_fast_verifier": True,
                "turbo_kv_quant_mode": "hybrid_turbo_vq",
                "turbo_vq_bits": 8,
                "turbo_vq_residual_dim": args.residual_dim,
                "turbo_hybrid_hot_window": args.hot_window,
                "turbo_runtime_dequant_cache": False,
            },
        ),
        (
            f"hybrid_vq_h{args.hot_window}_qjl_{node_budget_name}",
            {
                "turbo_quant": True,
                "turbo_kv_compression": True,
                "turbo_kv_quant_mode": "hybrid_turbo_vq",
                "turbo_vq_bits": 8,
                "turbo_vq_residual_dim": args.residual_dim,
                "turbo_hybrid_hot_window": args.hot_window,
                "turbo_runtime_dequant_cache": False,
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
            },
        ),
    ]
    if args.quick:
        modes = modes[:4]
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
    )
    qjl_sidecar_mb = estimate_packed_kv_qjl_sidecar_mb(
        model.config,
        kv_len,
        int(call_kwargs.get("turbo_kv_qjl_dim", args.kv_qjl_dim)),
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
        "packed_kv_qjl_sidecar_mb_est": qjl_sidecar_mb,
        "packed_kv_qjl_sidecar_pct_of_fp16_kv": qjl_sidecar_mb / max(1e-6, fp16_kv_mb),
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
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dense KV, QJL pruning, strict TurboVQ, and hybrid TurboVQ modes."
    )
    parser.add_argument(
        "--profile",
        choices=("manual", "tinyllama", "llama32-long", "llama32-128k"),
        default="manual",
        help=(
            "Apply benchmark defaults. llama32-* profiles use longer prompts and "
            "larger KV allocations so TurboVQ and packed KV-QJL are exercised."
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
    parser.add_argument("--hot-window", type=int, default=512)
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
    parser.add_argument("--kv-qjl-dim", type=int, default=128)
    parser.add_argument("--kv-qjl-layer", type=int, default=-1)
    parser.add_argument("--kv-qjl-keep-fraction", type=float, default=0.55)
    parser.add_argument("--kv-qjl-weight", type=float, default=0.05)
    parser.add_argument("--kv-qjl-min-kv-len", type=int, default=16384)
    parser.add_argument("--kv-qjl-medusa-pool-fraction", type=float, default=0.80)
    parser.add_argument("--kv-qjl-medusa-anchor-keep", type=int, default=2)
    parser.add_argument("--kv-qjl-auto-disable-after", type=int, default=2)
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
        "packed_kv_qjl_sidecar_mb_est",
        "packed_kv_qjl_sidecar_pct_of_fp16_kv",
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
