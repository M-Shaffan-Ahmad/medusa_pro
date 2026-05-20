import csv
import gc
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from medusa.model.medusa_model import MedusaModel


MODEL_DIR = os.environ.get("MEDUSA_MODEL_DIR", "Medusa/TinyLlama-1.1B-Chat-v1.0-4heads")
OUT_CSV = os.environ.get("OUT_CSV", "Medusa/tinyllama_turbo_best_old_heads_benchmark.csv")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "35"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "0"))

PROMPTS = [
    ("hpc", "Write a concise C++ MPI+OpenMP blocked GEMM example and explain the overlap strategy."),
    ("systems", "Explain strong scaling versus weak scaling for distributed systems in practical terms."),
    (
        "medium_context",
        "Summarize cache locality, memory bandwidth, kernel launch overhead, and branch prediction when optimizing CPU and GPU programs.",
    ),
]

MODES = [
    ("medusa_base_stream", {}),
    (
        "turbo_best_stream",
        {
            "turbo_quant": True,
            "turbo_kv_compression": False,
            "turbo_force_full_tree_fast_verifier": True,
        },
    ),
    ("medusa_base_nonstream", {"stream": False}),
    (
        "turbo_best_nonstream",
        {
            "stream": False,
            "turbo_quant": True,
            "turbo_kv_compression": False,
            "turbo_force_full_tree_fast_verifier": True,
        },
    ),
]


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


def run_one(model, prompt, medusa_choices, mode, kwargs):
    reset_memory()
    sync()
    full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
    inputs = model.tokenizer(full_prompt, return_tensors="pt").to("cuda")
    first = None
    text = ""
    stats = {}
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("collect_stats", True)
    call_kwargs.setdefault("max_new_tokens", MAX_NEW_TOKENS)
    start = time.perf_counter()
    with torch.inference_mode():
        for out in model.medusa_generate(
            inputs.input_ids,
            medusa_choices=medusa_choices,
            temperature=0.0,
            max_steps=MAX_STEPS,
            sampling="typical",
            fast=True,
            **call_kwargs,
        ):
            if first is None:
                sync()
                first = time.perf_counter()
            text = out["text"]
            stats = out.get("stats", stats)
    sync()
    end = time.perf_counter()
    tokens = int(stats.get("generated_tokens", 0) or 0)
    if tokens <= 0:
        tokens = max(1, len(model.tokenizer(text, add_special_tokens=False).input_ids))
    return {
        "mode": mode,
        "tokens": tokens,
        "prompt_tokens": int(inputs.input_ids.shape[1]),
        "tokens_per_step_cap": tokens / max(1, MAX_STEPS),
        "total_s": end - start,
        "ttft_s": (first or end) - start,
        "tps": tokens / max(1e-6, end - start),
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        "text": text,
    }


def main():
    assert torch.cuda.is_available(), "CUDA is required for this benchmark."
    torch.set_grad_enabled(False)
    model = MedusaModel.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to("cuda").eval()
    raw_choices = model.get_medusa_choice(model.base_model_name_or_path)
    medusa_choices = [tuple(path) for path in raw_choices if len(path) <= int(getattr(model, "medusa", 1))]
    print(
        "model",
        MODEL_DIR,
        "choices",
        len(medusa_choices),
        "max_depth",
        max(len(path) for path in medusa_choices),
        "max_steps",
        MAX_STEPS,
    )

    # Warm both major execution paths before timing.
    for _, kwargs in MODES[:2]:
        run_one(model, "Say hello in one sentence.", medusa_choices, "warmup", kwargs)

    rows = []
    base_text = {}
    for category, prompt in PROMPTS:
        for mode, kwargs in MODES:
            row = run_one(model, prompt, medusa_choices, mode, kwargs)
            row["category"] = category
            stream_key = "nonstream" if "nonstream" in mode else "stream"
            if mode.startswith("medusa_base"):
                base_text[(category, stream_key)] = row["text"]
                row["prefix_match_vs_base"] = 1.0
            else:
                row["prefix_match_vs_base"] = prefix_match(base_text[(category, stream_key)], row["text"])
            rows.append(row)
            print(
                category,
                mode,
                f"{row['tps']:.2f} TPS",
                "tokens",
                row["tokens"],
                "tok/step",
                f"{row['tokens_per_step_cap']:.2f}",
                "prefix",
                f"{row['prefix_match_vs_base']:.3f}",
                "alloc",
                f"{row['peak_alloc_mb']:.1f} MB",
            )

    fields = [
        "category",
        "mode",
        "tokens",
        "prompt_tokens",
        "tokens_per_step_cap",
        "total_s",
        "ttft_s",
        "tps",
        "peak_alloc_mb",
        "peak_reserved_mb",
        "prefix_match_vs_base",
        "text",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT_CSV)
    print("summary")
    pairs = [
        ("medusa_base_stream", "turbo_best_stream"),
        ("medusa_base_nonstream", "turbo_best_nonstream"),
    ]
    for base_mode, turbo_mode in pairs:
        base = sum(float(r["tps"]) for r in rows if r["mode"] == base_mode) / len(PROMPTS)
        turbo = sum(float(r["tps"]) for r in rows if r["mode"] == turbo_mode) / len(PROMPTS)
        prefix = sum(float(r["prefix_match_vs_base"]) for r in rows if r["mode"] == turbo_mode) / len(PROMPTS)
        print(base_mode, f"tps={base:.3f}", turbo_mode, f"tps={turbo:.3f}", f"speedup={turbo/base:.3f}", f"prefix={prefix:.3f}")


if __name__ == "__main__":
    main()
