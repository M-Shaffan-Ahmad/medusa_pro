import argparse
import csv
import gc
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from medusa.model.medusa_model import MedusaModel


DEFAULT_MODEL_DIR = "Medusa/TinyLlama-1.1B-Chat-v1.0-4heads"

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

LONG_CONTEXT_SEED = (
    "Cache locality, memory bandwidth, kernel launch overhead, branch prediction, "
    "NUMA placement, PCIe transfer, KV cache reuse, and asynchronous prefetching "
    "all affect CPU and GPU program performance. "
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


def build_prompts(long_repeat, long_only=False):
    prompts = [] if long_only else list(BASE_PROMPTS)
    if long_repeat > 0:
        prompts.append(
            (
                "long_context",
                (LONG_CONTEXT_SEED * int(long_repeat))
                + "Now summarize the most important optimization bottlenecks in five bullets.",
            )
        )
    return prompts


def build_modes(args):
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
            f"qjl_prune_{node_budget_name}",
            {
                "turbo_quant": True,
                "turbo_kv_compression": False,
                "turbo_prune_node_budget": args.node_budget,
                "turbo_prune_keep": args.prune_keep,
                "turbo_prune_min": args.prune_min,
                "turbo_prune_max": args.prune_max,
                "turbo_prune_use_qjl": True,
            },
        ),
        (
            f"packed_kv_qjl_{node_budget_name}",
            {
                "turbo_quant": True,
                "turbo_kv_compression": False,
                "turbo_prune_use_kv_qjl": True,
                "turbo_prune_use_qjl": False,
                "turbo_kv_qjl_dim": 128,
                "turbo_kv_qjl_layer": args.kv_qjl_layer,
                "turbo_kv_qjl_keep_fraction": args.kv_qjl_keep_fraction,
                "turbo_kv_qjl_weight": args.kv_qjl_weight,
                "turbo_kv_qjl_min_kv_len": args.kv_qjl_min_kv_len,
                "turbo_kv_qjl_medusa_pool_fraction": args.kv_qjl_medusa_pool_fraction,
                "turbo_kv_qjl_medusa_anchor_keep": args.kv_qjl_medusa_anchor_keep,
                "turbo_prune_node_budget": args.node_budget,
                "turbo_prune_keep": args.prune_keep,
                "turbo_prune_min": args.prune_min,
                "turbo_prune_max": args.prune_max,
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
                "turbo_kv_qjl_dim": 128,
                "turbo_kv_qjl_layer": args.kv_qjl_layer,
                "turbo_kv_qjl_keep_fraction": args.kv_qjl_keep_fraction,
                "turbo_kv_qjl_weight": args.kv_qjl_weight,
                "turbo_kv_qjl_min_kv_len": args.kv_qjl_min_kv_len,
                "turbo_kv_qjl_medusa_pool_fraction": args.kv_qjl_medusa_pool_fraction,
                "turbo_kv_qjl_medusa_anchor_keep": args.kv_qjl_medusa_anchor_keep,
                "turbo_prune_node_budget": args.node_budget,
                "turbo_prune_keep": args.prune_keep,
                "turbo_prune_min": args.prune_min,
                "turbo_prune_max": args.prune_max,
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
                "turbo_prune_use_qjl": True,
            },
        ),
    ]
    if args.quick:
        modes = modes[:4]
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        modes = [(name, kwargs) for name, kwargs in modes if name in wanted]
    return modes


def run_one(model, prompt, medusa_choices, args, mode, kwargs):
    reset_memory()
    sync()
    full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = model.tokenizer(full_prompt, return_tensors="pt").to("cuda")
    text = ""
    stats = {}
    first = None
    call_kwargs = dict(kwargs)
    call_kwargs["stream"] = args.stream
    call_kwargs["collect_stats"] = args.collect_stats

    start = time.perf_counter()
    with torch.inference_mode():
        for out in model.medusa_generate(
            inputs.input_ids,
            medusa_choices=medusa_choices,
            temperature=0.0,
            max_steps=args.max_steps,
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
        "fp16_kv_mb_est": fp16_kv_mb,
        "ideal_turbo_vq_kv_mb_est": ideal_vq_kv_mb,
        "ideal_turbo_vq_transfer_reduction": fp16_kv_mb / max(1e-6, ideal_vq_kv_mb),
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
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-csv", default="Medusa/comm_turbo_benchmark.csv")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--kv-max-length", type=int, default=2048)
    parser.add_argument("--long-repeat", type=int, default=0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--hot-window", type=int, default=512)
    parser.add_argument("--node-budget", type=int, default=32)
    parser.add_argument("--prune-keep", type=int, default=12)
    parser.add_argument("--prune-min", type=int, default=10)
    parser.add_argument("--prune-max", type=int, default=15)
    parser.add_argument("--residual-dim", type=int, default=128)
    parser.add_argument("--kv-qjl-layer", type=int, default=-1)
    parser.add_argument("--kv-qjl-keep-fraction", type=float, default=0.30)
    parser.add_argument("--kv-qjl-weight", type=float, default=0.5)
    parser.add_argument("--kv-qjl-min-kv-len", type=int, default=16384)
    parser.add_argument("--kv-qjl-medusa-pool-fraction", type=float, default=0.70)
    parser.add_argument("--kv-qjl-medusa-anchor-keep", type=int, default=2)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--no-stats", dest="collect_stats", action="store_false")
    parser.set_defaults(collect_stats=True)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run outside the sandbox or on a GPU machine.")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = MedusaModel.from_pretrained(args.model_dir, torch_dtype=torch.float16).to("cuda").eval()
    raw_choices = model.get_medusa_choice(model.base_model_name_or_path)
    medusa_choices = [
        tuple(path) for path in raw_choices if len(path) <= int(getattr(model, "medusa", 1))
    ]
    print(
        "model",
        args.model_dir,
        "choices",
        len(medusa_choices),
        "max_depth",
        max(len(path) for path in medusa_choices),
        "max_steps",
        args.max_steps,
        "kv_max_length",
        args.kv_max_length,
    )

    prompts = build_prompts(args.long_repeat, long_only=args.long_only)
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
        "fp16_kv_mb_est",
        "ideal_turbo_vq_kv_mb_est",
        "ideal_turbo_vq_transfer_reduction",
    ]
    stat_keys = sorted({key for row in rows for key in row if key.startswith("stat_")})
    fields = preferred + stat_keys + ["text"]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
