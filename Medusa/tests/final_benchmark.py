#!/usr/bin/env python3
"""Final fast-24 benchmark matrix for TinyLlama and Llama 3.2 Medusa models.

This runner compares the regular Medusa verifier against the local fastest
reduced-tree profile, `turbo_fast_24`, across prompt types and context sizes.
It writes both detailed rows and a compact speedup summary under
`tests/final_benchmark/` by default.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench_comm_turbo import prefix_match, run_one  # noqa: E402
from medusa.model.medusa_model import MedusaModel  # noqa: E402


MODEL_SPECS = {
    "tinyllama": {
        "display_name": "TinyLlama",
        "env": "TINYLLAMA_MEDUSA_MODEL_DIR",
        "model_dir": "TinyLlama-1.1B-Chat-v1.0-4heads",
        "contexts": (0, 512, 1024, 1536),
        "kv_max_length": 2048,
    },
    "llama32": {
        "display_name": "Llama-3.2-1B",
        "env": "LLAMA32_MEDUSA_MODEL_DIR",
        "model_dir": "llama32_1b_medusa_heads_code",
        "contexts": (0, 1024, 4096, 8192),
        "kv_max_length": 16384,
    },
}


PROMPT_TYPES = {
    "technical": {
        "seed": (
            "Distributed systems performance depends on cache locality, network "
            "latency, memory bandwidth, synchronization, scheduling, and profiling "
            "discipline. "
        ),
        "short": (
            "Explain strong scaling versus weak scaling for distributed systems in "
            "practical terms."
        ),
        "suffix": (
            "Now explain the most important performance bottlenecks and give a "
            "clear optimization checklist."
        ),
    },
    "general": {
        "seed": (
            "A student is balancing lectures, project deadlines, sleep, exercise, "
            "family responsibilities, finances, and long-term career planning. "
        ),
        "short": (
            "Give practical advice to a student who keeps procrastinating on a "
            "programming assignment."
        ),
        "suffix": (
            "Now give practical, encouraging advice that turns this situation into "
            "a weekly plan."
        ),
    },
    "coding": {
        "seed": (
            "Consider a codebase with Python data loaders, CUDA kernels, C++ worker "
            "pools, tests, benchmarks, error handling, memory ownership, and "
            "profiling notes. "
        ),
        "short": (
            "Write a Python function that groups strings by their first letter. "
            "Include type hints and handle empty strings."
        ),
        "suffix": (
            "Now write a compact Python module that validates inputs, times two "
            "implementations, and reports the faster one."
        ),
    },
}


MODES = [
    ("medusa_base", {}),
    (
        "turbo_fast_24",
        {
            "turbo_fast_preset": True,
            "medusa_choice_limit": 24,
            "_use_model_choice_resolution": True,
        },
    ),
]


DETAIL_FIELDS = [
    "model",
    "model_dir",
    "prompt_type",
    "requested_context_tokens",
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
    "turbo_vq_kv_mb_est",
    "polar_kv_mb_est",
]


SUMMARY_FIELDS = [
    "model",
    "prompt_type",
    "requested_context_tokens",
    "base_tps",
    "turbo_fast_24_tps",
    "speedup_vs_base",
    "prefix_match_vs_base",
    "base_accept_tokens_per_step",
    "turbo_accept_tokens_per_step",
    "turbo_verified_nodes_per_step",
    "base_peak_alloc_mb",
    "turbo_peak_alloc_mb",
    "tokens",
    "prompt_tokens",
]


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_int_csv(value: str) -> list[int]:
    parsed = []
    for item in parse_csv(value):
        parsed.append(int(item))
    return parsed


def resolve_model_dir(model_key: str) -> str:
    spec = MODEL_SPECS[model_key]
    return os.environ.get(spec["env"], spec["model_dir"])


def make_context_prompt(tokenizer, prompt_type: str, context_tokens: int) -> str:
    prompt_spec = PROMPT_TYPES[prompt_type]
    if int(context_tokens) <= 0:
        return prompt_spec["short"]

    seed = prompt_spec["seed"]
    suffix = prompt_spec["suffix"]
    seed_tokens = tokenizer(seed, add_special_tokens=False).input_ids
    suffix_tokens = tokenizer(suffix, add_special_tokens=False).input_ids
    repeat = max(
        1,
        math.ceil((int(context_tokens) - len(suffix_tokens)) / max(1, len(seed_tokens))),
    )
    return (seed * repeat) + suffix


def make_args(model_key: str, kv_max_length: int, cli_args) -> SimpleNamespace:
    return SimpleNamespace(
        max_steps=int(cli_args.max_steps),
        target_new_tokens=int(cli_args.target_new_tokens),
        kv_max_length=int(kv_max_length),
        use_model_context_kv=bool(cli_args.use_model_context_kv),
        stream=bool(cli_args.stream),
        collect_stats=True,
        draft_head_type=cli_args.draft_head_type,
        tree_policy="fixed",
        tree_calibration_path="",
        residual_dim=128,
        polar_first_bits=4,
        polar_other_bits=2,
        polar_levels=4,
        model_key=model_key,
    )


def load_model(model_dir: str):
    model = MedusaModel.from_pretrained(model_dir, torch_dtype=torch.float16)
    return model.to("cuda").eval()


def unload_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def get_medusa_choices(model):
    raw_choices = model.get_medusa_choice(model.base_model_name_or_path)
    max_depth = int(getattr(model, "medusa", 1))
    return [tuple(path) for path in raw_choices if len(path) <= max_depth]


def write_csv(path: Path, rows: list[dict], preferred_fields: list[str]) -> None:
    stat_fields = sorted({key for row in rows for key in row if key.startswith("stat_")})
    extra_fields = sorted(
        {key for row in rows for key in row}
        - set(preferred_fields)
        - set(stat_fields)
        - {"text"}
    )
    fields = preferred_fields + stat_fields + extra_fields
    if any("text" in row for row in rows):
        fields.append("text")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def to_float(value, default=0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def run_model_matrix(model_key: str, contexts: list[int], prompt_types: list[str], cli_args):
    model_dir = resolve_model_dir(model_key)
    spec = MODEL_SPECS[model_key]
    kv_max_length = int(cli_args.kv_max_length or spec["kv_max_length"])
    bench_args = make_args(model_key, kv_max_length, cli_args)

    print(
        "loading",
        spec["display_name"],
        "from",
        model_dir,
        "contexts",
        ",".join(str(item) for item in contexts),
        "prompt_types",
        ",".join(prompt_types),
    )
    model = load_model(model_dir)
    medusa_choices = get_medusa_choices(model)
    print(
        "model",
        spec["display_name"],
        "choices",
        len(medusa_choices),
        "max_depth",
        max(len(path) for path in medusa_choices),
        "kv_max_length",
        kv_max_length,
    )

    for _, kwargs in MODES:
        run_one(model, "Say hello in one sentence.", medusa_choices, bench_args, "warmup", kwargs)

    rows = []
    summary_rows = []
    try:
        for context_tokens in contexts:
            for prompt_type in prompt_types:
                prompt = make_context_prompt(model.tokenizer, prompt_type, int(context_tokens))
                category = f"{prompt_type}_ctx{int(context_tokens)}"
                base_row = None
                base_text = ""
                for mode, kwargs in MODES:
                    row = run_one(model, prompt, medusa_choices, bench_args, mode, kwargs)
                    row["model"] = spec["display_name"]
                    row["model_dir"] = model_dir
                    row["prompt_type"] = prompt_type
                    row["requested_context_tokens"] = int(context_tokens)
                    row["category"] = category
                    if mode == "medusa_base":
                        base_row = row
                        base_text = row["text"]
                        row["prefix_match_vs_base"] = 1.0
                        row["speedup_vs_base"] = 1.0
                    else:
                        row["prefix_match_vs_base"] = prefix_match(base_text, row["text"])
                        row["speedup_vs_base"] = to_float(row["tps"]) / max(
                            1e-6,
                            to_float(base_row["tps"] if base_row else 0.0),
                        )
                    rows.append(row)
                    print(
                        spec["display_name"],
                        category,
                        mode,
                        f"{to_float(row['tps']):.2f} TPS",
                        "speedup",
                        f"{to_float(row['speedup_vs_base']):.3f}",
                        "prefix",
                        f"{to_float(row['prefix_match_vs_base']):.3f}",
                        "accept",
                        row.get("accepted_tokens_per_step", ""),
                    )

                turbo_row = next(
                    row
                    for row in rows
                    if row["model"] == spec["display_name"]
                    and row["category"] == category
                    and row["mode"] == "turbo_fast_24"
                )
                summary_rows.append(
                    {
                        "model": spec["display_name"],
                        "prompt_type": prompt_type,
                        "requested_context_tokens": int(context_tokens),
                        "base_tps": to_float(base_row["tps"] if base_row else 0.0),
                        "turbo_fast_24_tps": to_float(turbo_row["tps"]),
                        "speedup_vs_base": to_float(turbo_row["speedup_vs_base"]),
                        "prefix_match_vs_base": to_float(turbo_row["prefix_match_vs_base"]),
                        "base_accept_tokens_per_step": base_row.get(
                            "accepted_tokens_per_step", ""
                        )
                        if base_row
                        else "",
                        "turbo_accept_tokens_per_step": turbo_row.get(
                            "accepted_tokens_per_step", ""
                        ),
                        "turbo_verified_nodes_per_step": turbo_row.get(
                            "verified_nodes_per_step", ""
                        ),
                        "base_peak_alloc_mb": to_float(
                            base_row["peak_alloc_mb"] if base_row else 0.0
                        ),
                        "turbo_peak_alloc_mb": to_float(turbo_row["peak_alloc_mb"]),
                        "tokens": turbo_row["tokens"],
                        "prompt_tokens": turbo_row["prompt_tokens"],
                    }
                )
    finally:
        unload_model(model)

    return rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final fast-24 benchmark across TinyLlama and Llama 3.2 "
            "prompt/context combinations."
        )
    )
    parser.add_argument(
        "--models",
        default="tinyllama,llama32",
        help="Comma-separated model keys: tinyllama,llama32.",
    )
    parser.add_argument(
        "--prompt-types",
        default="technical,general,coding",
        help="Comma-separated prompt types: technical,general,coding.",
    )
    parser.add_argument(
        "--contexts",
        default="",
        help=(
            "Optional comma-separated context-token targets used for every model. "
            "By default each model uses its own context grid."
        ),
    )
    parser.add_argument(
        "--tinyllama-contexts",
        default="",
        help="Override TinyLlama context-token grid.",
    )
    parser.add_argument(
        "--llama32-contexts",
        default="",
        help="Override Llama 3.2 context-token grid.",
    )
    parser.add_argument("--out-dir", default="tests/final_benchmark")
    parser.add_argument("--max-steps", type=int, default=48)
    parser.add_argument("--target-new-tokens", type=int, default=64)
    parser.add_argument(
        "--kv-max-length",
        type=int,
        default=0,
        help="Override KV preallocation length for every model.",
    )
    parser.add_argument(
        "--use-model-context-kv",
        action="store_true",
        help="Preallocate KV to the model context window when supported.",
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--draft-head-type", choices=("medusa", "hydra"), default="medusa")
    args = parser.parse_args()

    models = parse_csv(args.models)
    prompt_types = parse_csv(args.prompt_types)
    unknown_models = sorted(set(models) - set(MODEL_SPECS))
    unknown_prompts = sorted(set(prompt_types) - set(PROMPT_TYPES))
    if unknown_models:
        raise SystemExit(f"Unknown model key(s): {', '.join(unknown_models)}")
    if unknown_prompts:
        raise SystemExit(f"Unknown prompt type(s): {', '.join(unknown_prompts)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the final benchmark.")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_summary_rows = []
    shared_contexts = parse_int_csv(args.contexts) if args.contexts else []
    for model_key in models:
        override = getattr(args, f"{model_key}_contexts")
        contexts = (
            parse_int_csv(override)
            if override
            else shared_contexts
            if shared_contexts
            else list(MODEL_SPECS[model_key]["contexts"])
        )
        rows, summary_rows = run_model_matrix(model_key, contexts, prompt_types, args)
        all_rows.extend(rows)
        all_summary_rows.extend(summary_rows)
        write_csv(
            out_dir / f"final_benchmark_{model_key}_rows.csv",
            rows,
            DETAIL_FIELDS,
        )
        write_csv(
            out_dir / f"final_benchmark_{model_key}_summary.csv",
            summary_rows,
            SUMMARY_FIELDS,
        )

    write_csv(out_dir / "final_benchmark_rows.csv", all_rows, DETAIL_FIELDS)
    write_csv(out_dir / "final_benchmark_summary.csv", all_summary_rows, SUMMARY_FIELDS)

    if all_summary_rows:
        best = max(all_summary_rows, key=lambda row: to_float(row["speedup_vs_base"]))
        print(
            "best",
            best["model"],
            best["prompt_type"],
            f"ctx={best['requested_context_tokens']}",
            f"speedup={to_float(best['speedup_vs_base']):.3f}",
            f"prefix={to_float(best['prefix_match_vs_base']):.3f}",
        )
    print("wrote", out_dir / "final_benchmark_summary.csv")


if __name__ == "__main__":
    main()
