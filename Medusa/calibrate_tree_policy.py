#!/usr/bin/env python3
"""Calibrate exact-safe Medusa tree presets for adaptive runtime selection."""

import argparse
import csv
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from bench_comm_turbo import build_prompts, prefix_match, reset_memory, sync
from medusa.model.medusa_model import MedusaModel


DEFAULT_MODEL_DIR = "Medusa/TinyLlama-1.1B-Chat-v1.0-4heads"


def parse_choice_sweep(raw: str) -> list[int]:
    values = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if 0 not in values:
        values.insert(0, 0)
    seen = set()
    ordered = []
    for value in values:
        if value < 0 or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def run_one(model, prompt, args, choice_limit: int) -> dict:
    reset_memory()
    sync()
    full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = model.tokenizer(full_prompt, return_tensors="pt").to("cuda")
    start = time.perf_counter()
    final = {}
    with torch.inference_mode():
        for out in model.medusa_generate(
            inputs.input_ids,
            temperature=0.0,
            max_steps=args.max_steps,
            max_new_tokens=args.target_new_tokens,
            medusa_choice_limit=int(choice_limit),
            medusa_choice_max_depth=int(args.choice_max_depth),
            stream=False,
            collect_stats=True,
            draft_head_type=args.draft_head_type,
            tree_policy="fixed",
            turbo_kv_max_length=args.kv_max_length,
        ):
            final = out
    sync()
    end = time.perf_counter()

    stats = final.get("stats", {})
    tokens = int(stats.get("generated_tokens", 0) or 0)
    text = final.get("text", "")
    if tokens <= 0:
        tokens = max(1, len(model.tokenizer(text, add_special_tokens=False).input_ids))
    decode_steps = max(1, int(stats.get("decode_steps", 0) or 0))
    actual_depth = int(args.choice_max_depth or stats.get("medusa_choice_max_depth", 0) or 0)
    return {
        "mode": "full_tree" if int(choice_limit) == 0 else f"choice_{int(choice_limit)}",
        "choice_limit": int(choice_limit),
        "max_depth": actual_depth,
        "tokens": tokens,
        "total_s": end - start,
        "tps": tokens / max(1e-6, end - start),
        "accepted_tokens_per_step": tokens / decode_steps,
        "verified_nodes_per_step": float(stats.get("verified_tree_nodes", 0) or 0) / decode_steps,
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Medusa tree sizes and write an exact-safe calibration CSV."
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-csv", default="tree_calibration.csv")
    parser.add_argument("--choice-sweep", default="0,8,16,24,32,48")
    parser.add_argument("--choice-max-depth", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--target-new-tokens", type=int, default=160)
    parser.add_argument("--kv-max-length", type=int, default=2048)
    parser.add_argument("--prompt-suite", choices=("technical", "general", "mixed"), default="mixed")
    parser.add_argument("--long-repeat", type=int, default=0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--draft-head-type", choices=("medusa", "hydra"), default="medusa")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for calibration.")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = MedusaModel.from_pretrained(args.model_dir, torch_dtype=torch.float16).to("cuda")
    model = model.eval()
    prompts = build_prompts(args.long_repeat, args.long_only, args.prompt_suite)
    choices = parse_choice_sweep(args.choice_sweep)

    rows = []
    fields = [
        "mode",
        "choice_limit",
        "max_depth",
        "prompt_category",
        "tokens",
        "total_s",
        "tps",
        "accepted_tokens_per_step",
        "verified_nodes_per_step",
        "prefix_match_vs_base",
        "peak_alloc_mb",
    ]
    for category, prompt in prompts:
        reference_text = None
        for choice_limit in choices:
            row = run_one(model, prompt, args, choice_limit)
            row["prompt_category"] = category
            if choice_limit == 0:
                reference_text = row.pop("text")
                row["prefix_match_vs_base"] = 1.0
            else:
                text = row.pop("text")
                row["prefix_match_vs_base"] = prefix_match(reference_text or "", text)
            rows.append({key: row.get(key, "") for key in fields})
            print(
                category,
                row["mode"],
                f"{row['tps']:.2f} TPS",
                "accept/step",
                f"{row['accepted_tokens_per_step']:.3f}",
                "prefix",
                f"{row['prefix_match_vs_base']:.3f}",
            )

    with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
