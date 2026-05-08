import argparse
import csv
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from bench_comm_turbo import build_prompts, reset_memory, sync


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark plain autoregressive HF generation on the same prompt suites as bench_comm_turbo.py."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-csv", default="base_transformers_benchmark.csv")
    parser.add_argument("--target-new-tokens", type=int, default=160)
    parser.add_argument("--prompt-suite", choices=("technical", "general", "mixed"), default="mixed")
    parser.add_argument("--long-repeat", type=int, default=0)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--device-map",
        default="",
        help="Optional transformers device_map. Use 'auto' for quantized Kaggle runs.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required. Run this on a GPU machine.")
    if args.load_in_8bit and args.load_in_4bit:
        raise SystemExit("Choose only one of --load-in-8bit or --load-in-4bit.")

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"torch_dtype": torch.float16}
    if args.load_in_8bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    if args.device_map or args.load_in_8bit or args.load_in_4bit:
        load_kwargs["device_map"] = args.device_map or "auto"
        load_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, **load_kwargs)
    if not (args.device_map or args.load_in_8bit or args.load_in_4bit):
        model = model.to("cuda")
    model = model.eval()

    prompts = build_prompts(
        args.long_repeat,
        long_only=args.long_only,
        prompt_suite=args.prompt_suite,
    )
    rows = []
    for category, prompt in prompts:
        reset_memory()
        sync()
        full_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.target_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        sync()
        end = time.perf_counter()

        prompt_tokens = int(inputs.input_ids.shape[1])
        generated_tokens = int(output_ids.shape[1] - prompt_tokens)
        text = tokenizer.decode(output_ids[0, prompt_tokens:], skip_special_tokens=True)
        row = {
            "category": category,
            "mode": "base_transformers",
            "tokens": generated_tokens,
            "prompt_tokens": prompt_tokens,
            "total_s": end - start,
            "tps": generated_tokens / max(1e-6, end - start),
            "peak_alloc_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
            "text": text,
        }
        rows.append(row)
        print(category, f"{row['tps']:.2f} TPS", "alloc", f"{row['peak_alloc_mb']:.1f} MB")

    fields = [
        "category",
        "mode",
        "tokens",
        "prompt_tokens",
        "total_s",
        "tps",
        "peak_alloc_mb",
        "peak_reserved_mb",
        "text",
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
