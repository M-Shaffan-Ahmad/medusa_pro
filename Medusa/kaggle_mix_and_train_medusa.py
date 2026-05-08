#!/usr/bin/env python3
"""Kaggle one-shot data mix builder and Medusa self-distillation launcher.

This script is meant to be run inside a Kaggle notebook with Internet enabled.
It builds a local JSONL mixture:

    70% UltraChat, 15% SlimOrca, 10% CodeAlpaca, 5% GSM8K

Then it launches train_tinyllama_medusa_heads.py to fine-tune only the Medusa
heads.  Upload your current good medusa_tinyllama_heads folder as a Kaggle input
dataset, or pass --init-medusa-dir explicitly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import subprocess
import sys
import zipfile
from itertools import islice
from pathlib import Path
from typing import Any, Callable


DatasetFormatter = Callable[[dict[str, Any]], dict[str, Any] | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Kaggle Medusa data mix and train heads.")
    parser.add_argument(
        "--init-medusa-dir",
        default="auto",
        help="Existing good Medusa head folder. Use auto to search /kaggle/input.",
    )
    parser.add_argument(
        "--output-dir",
        default="/kaggle/working/medusa_tinyllama_heads_selfdistill",
        help="Output folder for trained heads.",
    )
    parser.add_argument(
        "--work-dir",
        default="/kaggle/working/medusa_training",
        help="Working directory for the mixed JSONL.",
    )
    parser.add_argument("--mixed-jsonl", default="", help="Reuse/write this JSONL path.")
    parser.add_argument("--total-examples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-package-install", action="store_true")

    parser.add_argument("--device", default="cuda:0", help="Kaggle T4 x2 should use cuda:0.")
    parser.add_argument("--seq-len", type=int, default=768)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Kaggle-friendly cap. Set 0 to run a full epoch over the mixed data.",
    )
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=96)
    parser.add_argument(
        "--checkpoint-dir",
        default="/kaggle/working/medusa_training/checkpoints",
        help="Resumable checkpoint directory.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Use 'latest' to resume the newest checkpoint from --checkpoint-dir.",
    )
    parser.add_argument("--loss-token-chunk-size", type=int, default=128)
    parser.add_argument("--argmax-token-chunk-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument(
        "--gt-loss-weight",
        type=float,
        default=0.0,
        help="Keep 0.0 for acceptance-focused self-distillation.",
    )
    return parser.parse_args()


def ensure_packages(skip_install: bool) -> None:
    required = {
        "datasets": "datasets",
        "safetensors": "safetensors",
        "sentencepiece": "sentencepiece",
        "tqdm": "tqdm",
        "transformers": "transformers",
    }
    missing = []
    for module_name, package_name in required.items():
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(package_name)
    if missing and skip_install:
        raise RuntimeError(f"Missing packages: {missing}. Re-run without --skip-package-install.")
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


def first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def format_ultrachat(row: dict[str, Any]) -> dict[str, Any] | None:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return {"messages": messages}
    text = first_text(row.get("text"), row.get("prompt"))
    return {"text": text} if text else None


def format_slimorca(row: dict[str, Any]) -> dict[str, Any] | None:
    conversations = row.get("conversations")
    system = first_text(row.get("system"))
    if isinstance(conversations, list) and conversations:
        if system and not (
            isinstance(conversations[0], dict)
            and conversations[0].get("from") in {"system", "system_prompt"}
        ):
            conversations = [{"from": "system", "value": system}] + conversations
        return {"conversations": conversations}
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return {"messages": messages}
    text = first_text(row.get("text"), row.get("response"), row.get("answer"))
    return {"text": text} if text else None


def format_codealpaca(row: dict[str, Any]) -> dict[str, Any] | None:
    instruction = first_text(row.get("instruction"), row.get("prompt"))
    input_text = first_text(row.get("input"), row.get("context"))
    output = first_text(row.get("output"), row.get("response"), row.get("answer"))
    if not instruction or not output:
        return None
    if input_text:
        text = f"User: {instruction}\n{input_text}\nAssistant: {output}"
    else:
        text = f"User: {instruction}\nAssistant: {output}"
    return {"text": text}


def format_gsm8k(row: dict[str, Any]) -> dict[str, Any] | None:
    question = first_text(row.get("question"))
    answer = first_text(row.get("answer"))
    if not question or not answer:
        return None
    return {"text": f"User: {question}\nAssistant: {answer}"}


def count_plan(total_examples: int) -> dict[str, int]:
    counts = {
        "ultrachat": int(total_examples * 0.70),
        "slimorca": int(total_examples * 0.15),
        "codealpaca": int(total_examples * 0.10),
        "gsm8k": int(total_examples * 0.05),
    }
    counts["ultrachat"] += total_examples - sum(counts.values())
    return counts


def load_streaming_dataset(source: tuple[str, str | None, str]):
    from datasets import load_dataset

    path, name, split = source
    if name is None:
        return load_dataset(path, split=split, streaming=True)
    return load_dataset(path, name, split=split, streaming=True)


def collect_rows(
    label: str,
    sources: list[tuple[str, str | None, str]],
    count: int,
    formatter: DatasetFormatter,
    seed: int,
    shuffle_buffer: int,
) -> list[str]:
    from tqdm.auto import tqdm

    last_error: Exception | None = None
    for source in sources:
        try:
            dataset = load_streaming_dataset(source)
            if shuffle_buffer > 0:
                dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
            lines: list[str] = []
            progress = tqdm(total=count, desc=f"{label} {source[0]}", leave=False)
            for row in islice(dataset, count * 4):
                item = formatter(row)
                if not item:
                    continue
                lines.append(json.dumps(item, ensure_ascii=False))
                progress.update(1)
                if len(lines) >= count:
                    break
            progress.close()
            if len(lines) < count:
                raise RuntimeError(f"{source[0]} produced only {len(lines)} usable rows, need {count}.")
            return lines
        except Exception as exc:
            last_error = exc
            print(f"WARNING: failed {label} source {source}: {exc}")
    raise RuntimeError(f"All sources failed for {label}: {last_error}")


def build_mixed_jsonl(args: argparse.Namespace, output_path: Path) -> Path:
    plan = count_plan(args.total_examples)
    print(f"building mix at {output_path}")
    print("counts:", plan)

    sources: dict[str, tuple[list[tuple[str, str | None, str]], DatasetFormatter]] = {
        "ultrachat": (
            [("HuggingFaceH4/ultrachat_200k", None, "train_sft")],
            format_ultrachat,
        ),
        "slimorca": (
            [("Open-Orca/SlimOrca", None, "train")],
            format_slimorca,
        ),
        "codealpaca": (
            [
                ("flwrlabs/code-alpaca-20k", None, "train"),
                ("sahil2801/CodeAlpaca-20k", None, "train"),
            ],
            format_codealpaca,
        ),
        "gsm8k": (
            [("openai/gsm8k", "main", "train")],
            format_gsm8k,
        ),
    }

    all_lines: list[str] = []
    for label, needed in plan.items():
        dataset_sources, formatter = sources[label]
        all_lines.extend(
            collect_rows(
                label,
                dataset_sources,
                needed,
                formatter,
                args.seed,
                args.shuffle_buffer,
            )
        )

    rng = random.Random(args.seed)
    rng.shuffle(all_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    print(f"wrote {len(all_lines)} rows to {output_path}")
    return output_path


def find_medusa_dir(value: str) -> Path:
    def extract_zip(zip_path: Path) -> Path:
        extract_root = Path("/kaggle/working/medusa_head_inputs") / zip_path.stem
        extract_root.mkdir(parents=True, exist_ok=True)
        marker = extract_root / ".extracted"
        if not marker.exists():
            print(f"extracting head zip {zip_path} -> {extract_root}")
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_root)
            marker.write_text("ok\n", encoding="utf-8")
        matches = list(extract_root.glob("**/medusa_lm_head.safetensors"))
        matches.extend(extract_root.glob("**/medusa_lm_head.pt"))
        if not matches:
            raise FileNotFoundError(f"Extracted {zip_path}, but no Medusa head file was found.")
        return sorted(matches, key=lambda path: len(str(path)))[0].parent.resolve()

    if value != "auto":
        path = Path(value).expanduser().resolve()
        if path.suffix == ".zip":
            return extract_zip(path)
        if not (path / "medusa_lm_head.safetensors").exists() and not (path / "medusa_lm_head.pt").exists():
            raise FileNotFoundError(f"No medusa_lm_head.safetensors or medusa_lm_head.pt in {path}")
        return path

    candidates: list[Path] = []
    zip_candidates: list[Path] = []
    for root in (Path("/kaggle/input"), Path.cwd(), Path.cwd().parent):
        if root.exists():
            candidates.extend(root.glob("**/medusa_lm_head.safetensors"))
            candidates.extend(root.glob("**/medusa_lm_head.pt"))
            zip_candidates.extend(root.glob("**/*.zip"))
    if not candidates:
        for zip_path in sorted(zip_candidates, key=lambda path: len(str(path))):
            try:
                with zipfile.ZipFile(zip_path) as archive:
                    names = archive.namelist()
                if any(name.endswith(("medusa_lm_head.safetensors", "medusa_lm_head.pt")) for name in names):
                    return extract_zip(zip_path)
            except Exception as exc:
                print(f"WARNING: could not inspect zip {zip_path}: {exc}")
        raise FileNotFoundError(
            "Could not find medusa_lm_head.safetensors or medusa_lm_head.pt. "
            "Upload your good Medusa head folder as a Kaggle input dataset, then pass "
            "--init-medusa-dir /kaggle/input/<dataset>/<folder> if auto cannot find it."
        )
    candidates = sorted(candidates, key=lambda path: len(str(path)))
    return candidates[0].parent.resolve()


def run_training(args: argparse.Namespace, mixed_jsonl: Path, init_medusa_dir: Path) -> None:
    repo_dir = Path(__file__).resolve().parent
    trainer = repo_dir / "train_tinyllama_medusa_heads.py"
    if not trainer.exists():
        raise FileNotFoundError(f"Missing trainer: {trainer}")

    cmd = [
        sys.executable,
        str(trainer),
        "--init-medusa-dir",
        str(init_medusa_dir),
        "--output-dir",
        str(Path(args.output_dir).resolve()),
        "--data-path",
        str(mixed_jsonl),
        "--seq-len",
        str(args.seq_len),
        "--micro-batch-size",
        str(args.micro_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--learning-rate",
        str(args.learning_rate),
        "--epochs",
        str(args.epochs),
        "--max-steps",
        str(args.max_steps),
        "--eval-every",
        str(args.eval_every),
        "--eval-batches",
        str(args.eval_batches),
        "--checkpoint-dir",
        str(Path(args.checkpoint_dir).resolve()),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--keep-checkpoints",
        str(args.keep_checkpoints),
        "--loss-token-chunk-size",
        str(args.loss_token_chunk_size),
        "--argmax-token-chunk-size",
        str(args.argmax_token_chunk_size),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--gt-loss-weight",
        str(args.gt_loss_weight),
    ]
    if args.resume_checkpoint:
        cmd.extend(["--resume-checkpoint", args.resume_checkpoint])
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print("launching trainer:")
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=str(repo_dir), env=env)


def main() -> None:
    args = parse_args()
    ensure_packages(args.skip_package_install)
    work_dir = Path(args.work_dir)
    mixed_jsonl = Path(args.mixed_jsonl) if args.mixed_jsonl else work_dir / "medusa_train_mix_70_15_10_5.jsonl"
    mixed_jsonl = mixed_jsonl.expanduser().resolve()

    if not args.train_only:
        build_mixed_jsonl(args, mixed_jsonl)
    elif not mixed_jsonl.exists():
        raise FileNotFoundError(f"--train-only requested, but {mixed_jsonl} does not exist.")

    if args.download_only:
        print("download-only complete")
        return

    init_medusa_dir = find_medusa_dir(args.init_medusa_dir)
    print(f"using init heads: {init_medusa_dir}")
    run_training(args, mixed_jsonl, init_medusa_dir)


if __name__ == "__main__":
    main()
