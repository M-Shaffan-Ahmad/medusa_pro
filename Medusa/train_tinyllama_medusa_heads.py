#!/usr/bin/env python3
"""Self-distill TinyLlama Medusa heads without touching the base model.

This trainer is built for the local TinyLlama 4-head setup used by the
benchmarks in this repository.  It freezes the base model and LM head, computes
base-model greedy future tokens with no gradients, and trains only the Medusa
residual heads to match those future greedy tokens.  That objective is aligned
with greedy speculative acceptance: a proposed Medusa token is accepted only
when it matches the verifier/base model token.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from medusa.model.medusa_model import MedusaConfig, MedusaModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/fine-tune TinyLlama Medusa heads by self-distillation."
    )
    parser.add_argument(
        "--init-medusa-dir",
        default="../medusa_tinyllama_heads",
        help="Existing Medusa head folder to initialize from.",
    )
    parser.add_argument(
        "--output-dir",
        default="../medusa_tinyllama_heads_selfdistill",
        help="Folder where the best validated heads will be written.",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Local .json, .jsonl, or .txt training data.",
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--min-seq-len", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.03)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--no-pack", action="store_true", help="Do not pack samples into fixed token windows.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0, help="Override epochs when > 0.")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--head-loss-weights",
        default="1.0,0.85,0.70,0.55",
        help="Comma-separated loss weights for Medusa heads.",
    )
    parser.add_argument(
        "--gt-loss-weight",
        type=float,
        default=0.0,
        help="Optional fraction of ground-truth future-token CE mixed into the self-distill loss.",
    )
    parser.add_argument(
        "--loss-token-chunk-size",
        type=int,
        default=128,
        help="Token chunk size for head CE. Lower this for 6GB GPUs.",
    )
    parser.add_argument(
        "--argmax-token-chunk-size",
        type=int,
        default=128,
        help="Token chunk size for base greedy argmax. Lower this for 6GB GPUs.",
    )

    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=48)
    parser.add_argument(
        "--min-save-improvement",
        type=float,
        default=1e-4,
        help="Validation score margin required to overwrite the initial-head floor.",
    )
    parser.add_argument(
        "--keep-initial-if-worse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the initial heads as the floor and only overwrite with a better validation score.",
    )

    parser.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--save-dtype", choices=("train", "fp16", "bf16", "fp32"), default="train")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if device.type == "cuda":
            return torch.float16
        return torch.float32
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def parse_head_weights(raw: str, num_heads: int) -> list[float]:
    weights = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not weights:
        raise ValueError("--head-loss-weights cannot be empty.")
    if len(weights) < num_heads:
        weights.extend([weights[-1]] * (num_heads - len(weights)))
    return weights[:num_heads]


def read_json_records(path: Path) -> list[Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "train", "examples", "samples"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Could not find a list of records in {path}.")


def read_head_config(init_medusa_dir: str | Path) -> dict[str, Any]:
    config_path = Path(init_medusa_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_training_tokenizer(init_medusa_dir: str | Path) -> AutoTokenizer:
    init_medusa_dir = Path(init_medusa_dir)
    head_config = read_head_config(init_medusa_dir)
    sources = [str(init_medusa_dir)]
    base_source = head_config.get("base_model_name_or_path")
    if base_source and str(base_source) not in sources:
        sources.append(str(base_source))

    errors = []
    for source in sources:
        try:
            return AutoTokenizer.from_pretrained(source, use_fast=True)
        except Exception as exc:
            errors.append(f"{source}: {exc}")
        try:
            return AutoTokenizer.from_pretrained(source, use_fast=False)
        except Exception as exc:
            errors.append(f"{source} slow: {exc}")

    detail = "\n".join(errors)
    raise RuntimeError(f"Could not load a tokenizer for {init_medusa_dir}.\n{detail}")


def read_jsonl_records(path: Path) -> list[Any]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def message_text(messages: Iterable[Any], tokenizer: AutoTokenizer) -> str:
    normalized = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("from") or "user"
        content = item.get("content")
        if content is None:
            content = item.get("value")
        if content is None:
            continue
        role = "assistant" if role in {"gpt", "assistant", "bot"} else role
        role = "user" if role in {"human", "user"} else role
        normalized.append({"role": str(role), "content": str(content)})
    if not normalized:
        return ""
    try:
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        return "\n".join(f"{msg['role']}: {msg['content']}" for msg in normalized)


def record_to_text(record: Any, tokenizer: AutoTokenizer) -> str:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return ""
    for key in ("text", "content", "completion", "response", "output"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            if key in {"response", "output"}:
                prefix = record.get("prompt") or record.get("instruction") or record.get("input") or ""
                return f"{prefix}\n{value}".strip()
            return value
    for key in ("messages", "conversations"):
        value = record.get(key)
        if isinstance(value, list):
            return message_text(value, tokenizer)
    prompt = record.get("prompt") or record.get("instruction")
    output = record.get("output") or record.get("response")
    if prompt or output:
        return f"{prompt or ''}\n{output or ''}".strip()
    return ""


def load_texts(path: Path, tokenizer: AutoTokenizer, max_samples: int, seed: int) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = read_jsonl_records(path)
    elif suffix == ".json":
        records = read_json_records(path)
    else:
        raw = path.read_text(encoding="utf-8")
        records = [block.strip() for block in raw.split("\n\n") if block.strip()]
        if len(records) < 8:
            records = [line.strip() for line in raw.splitlines() if line.strip()]

    rng = random.Random(seed)
    rng.shuffle(records)
    if max_samples > 0:
        records = records[:max_samples]
    texts = [record_to_text(record, tokenizer).strip() for record in records]
    return [text for text in texts if text]


class TokenWindowDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        tokenizer: AutoTokenizer,
        seq_len: int,
        min_seq_len: int,
        pack: bool,
    ) -> None:
        self.examples: list[list[int]] = []
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            eos_id = tokenizer.pad_token_id
        if eos_id is None:
            raise ValueError("Tokenizer needs eos_token_id or pad_token_id.")

        if pack:
            stream: list[int] = []
            for text in texts:
                ids = tokenizer.encode(text, add_special_tokens=False)
                if ids:
                    stream.extend(ids)
                    stream.append(eos_id)
            usable = (len(stream) // seq_len) * seq_len
            self.examples = [
                stream[offset : offset + seq_len]
                for offset in range(0, usable, seq_len)
                if len(stream[offset : offset + seq_len]) >= min_seq_len
            ]
        else:
            for text in texts:
                ids = tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=seq_len,
                )
                if len(ids) >= min_seq_len:
                    self.examples.append(ids)

        if not self.examples:
            raise ValueError("No usable token windows were produced from the data.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> list[int]:
        return self.examples[index]


class PadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[list[int]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for i, row in enumerate(rows):
            length = len(row)
            input_ids[i, :length] = torch.tensor(row, dtype=torch.long)
            attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class EvalMetrics:
    loss: float
    score: float
    tokens: int
    head_top1: list[float]
    head_top5: list[float]
    head_top10: list[float]


def split_texts(texts: list[str], val_ratio: float, val_samples: int) -> tuple[list[str], list[str]]:
    if len(texts) < 2:
        raise ValueError("Need at least two text samples so validation is separate from training.")
    val_count = max(1, int(round(len(texts) * val_ratio)))
    if val_samples > 0:
        val_count = min(val_count, val_samples)
    val_count = min(max(1, val_count), len(texts) - 1)
    return texts[val_count:], texts[:val_count]


def prepare_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> MedusaModel:
    model = MedusaModel.from_pretrained(args.init_medusa_dir, torch_dtype=dtype)
    model.to(device)
    # Keep the frozen verifier cheap, but train the small Medusa heads as fp32
    # master weights.  This avoids fp16-gradient unscale failures and is more
    # stable for gentle fine-tuning from an already-good head folder.
    model.medusa_head.to(device=device, dtype=torch.float32)
    model.config.use_cache = False
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.medusa_head.parameters():
        param.requires_grad_(True)
    model.medusa_head.train()
    return model


@torch.no_grad()
def base_greedy_tokens(
    model: MedusaModel,
    hidden: torch.Tensor,
    token_chunk_size: int,
) -> torch.Tensor:
    chunks = []
    chunk_size = max(1, int(token_chunk_size))
    for start in range(0, hidden.shape[1], chunk_size):
        piece = hidden[:, start : start + chunk_size, :]
        logits = model.lm_head(piece)
        chunks.append(torch.argmax(logits, dim=-1))
    return torch.cat(chunks, dim=1)


@torch.no_grad()
def frozen_hidden_and_targets(
    model: MedusaModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    argmax_token_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state.detach()
    greedy = base_greedy_tokens(model, hidden, argmax_token_chunk_size)
    return hidden, greedy


def cosine_lr(
    base_lr: float,
    min_lr_ratio: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def valid_head_length(seq_len: int, head_idx: int) -> int:
    return max(0, seq_len - head_idx - 2)


def head_valid_mask(attention_mask: torch.Tensor, head_idx: int, length: int) -> torch.Tensor:
    if length <= 0:
        return attention_mask.new_zeros((attention_mask.shape[0], 0), dtype=torch.bool)
    current_ok = attention_mask[:, :length].bool()
    future_ok = attention_mask[:, head_idx + 2 : head_idx + 2 + length].bool()
    return current_ok & future_ok


def backward_train_batch(
    model: MedusaModel,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    train_dtype: torch.dtype,
    head_weights: list[float],
    gt_loss_weight: float,
    loss_token_chunk_size: int,
    argmax_token_chunk_size: int,
    loss_scale: float,
) -> dict[str, float]:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    hidden, greedy = frozen_hidden_and_targets(
        model,
        input_ids,
        attention_mask,
        argmax_token_chunk_size,
    )

    seq_len = input_ids.shape[1]
    total_loss = 0.0
    total_tokens = 0
    per_head_loss = []
    autocast_enabled = device.type == "cuda" and train_dtype in {torch.float16, torch.bfloat16}

    for head_idx in range(model.medusa):
        length = valid_head_length(seq_len, head_idx)
        if length <= 0:
            per_head_loss.append(0.0)
            continue

        mask = head_valid_mask(attention_mask, head_idx, length)
        valid_count = int(mask.sum().item())
        if valid_count == 0:
            per_head_loss.append(0.0)
            continue

        total_tokens += valid_count
        head_loss_sum = 0.0
        weight = float(head_weights[head_idx])
        for start in range(0, length, max(1, loss_token_chunk_size)):
            end = min(length, start + max(1, loss_token_chunk_size))
            chunk_mask = mask[:, start:end]
            chunk_valid = int(chunk_mask.sum().item())
            if chunk_valid == 0:
                continue
            target_distill = greedy[:, head_idx + 1 + start : head_idx + 1 + end]
            target_gt = input_ids[:, head_idx + 2 + start : head_idx + 2 + end]
            hidden_chunk = hidden[:, start:end, :]
            with torch.autocast("cuda", dtype=train_dtype, enabled=autocast_enabled):
                medusa_hidden = model.medusa_head[head_idx](hidden_chunk)
                logits = model.lm_head(medusa_hidden)
                flat_logits = logits[chunk_mask]
                distill_sum = F.cross_entropy(
                    flat_logits.float(),
                    target_distill[chunk_mask],
                    reduction="sum",
                )
                if gt_loss_weight > 0.0:
                    gt_sum = F.cross_entropy(
                        flat_logits.float(),
                        target_gt[chunk_mask],
                        reduction="sum",
                    )
                    chunk_loss_sum = (1.0 - gt_loss_weight) * distill_sum + gt_loss_weight * gt_sum
                else:
                    chunk_loss_sum = distill_sum
                normalized = weight * chunk_loss_sum / float(valid_count)
                backward_loss = normalized * loss_scale
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()
            head_loss_sum += float((chunk_loss_sum.detach() / float(chunk_valid)).item()) * chunk_valid

        per_head_loss.append(head_loss_sum / float(valid_count))
        total_loss += weight * head_loss_sum / float(valid_count)

    return {
        "loss": total_loss / max(1.0, sum(head_weights[: model.medusa])),
        "tokens": float(total_tokens),
        **{f"head{idx}_loss": value for idx, value in enumerate(per_head_loss)},
    }


@torch.no_grad()
def evaluate(
    model: MedusaModel,
    loader: DataLoader,
    device: torch.device,
    train_dtype: torch.dtype,
    head_weights: list[float],
    gt_loss_weight: float,
    loss_token_chunk_size: int,
    argmax_token_chunk_size: int,
    max_batches: int,
) -> EvalMetrics:
    model.eval()
    model.medusa_head.eval()
    autocast_enabled = device.type == "cuda" and train_dtype in {torch.float16, torch.bfloat16}
    loss_sum = 0.0
    token_count = 0
    top1 = [0 for _ in range(model.medusa)]
    top5 = [0 for _ in range(model.medusa)]
    top10 = [0 for _ in range(model.medusa)]
    totals = [0 for _ in range(model.medusa)]

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        hidden, greedy = frozen_hidden_and_targets(
            model,
            input_ids,
            attention_mask,
            argmax_token_chunk_size,
        )
        seq_len = input_ids.shape[1]

        for head_idx in range(model.medusa):
            length = valid_head_length(seq_len, head_idx)
            if length <= 0:
                continue
            mask = head_valid_mask(attention_mask, head_idx, length)
            valid_count = int(mask.sum().item())
            if valid_count == 0:
                continue
            totals[head_idx] += valid_count
            token_count += valid_count
            weight = float(head_weights[head_idx])
            for start in range(0, length, max(1, loss_token_chunk_size)):
                end = min(length, start + max(1, loss_token_chunk_size))
                chunk_mask = mask[:, start:end]
                chunk_valid = int(chunk_mask.sum().item())
                if chunk_valid == 0:
                    continue
                target_distill = greedy[:, head_idx + 1 + start : head_idx + 1 + end]
                target_gt = input_ids[:, head_idx + 2 + start : head_idx + 2 + end]
                hidden_chunk = hidden[:, start:end, :]
                with torch.autocast("cuda", dtype=train_dtype, enabled=autocast_enabled):
                    medusa_hidden = model.medusa_head[head_idx](hidden_chunk)
                    logits = model.lm_head(medusa_hidden)
                flat_logits = logits[chunk_mask].float()
                targets = target_distill[chunk_mask]
                distill_sum = F.cross_entropy(flat_logits, targets, reduction="sum")
                if gt_loss_weight > 0.0:
                    gt_sum = F.cross_entropy(
                        flat_logits,
                        target_gt[chunk_mask],
                        reduction="sum",
                    )
                    chunk_loss_sum = (1.0 - gt_loss_weight) * distill_sum + gt_loss_weight * gt_sum
                else:
                    chunk_loss_sum = distill_sum
                loss_sum += weight * float(chunk_loss_sum.item())

                k = min(10, flat_logits.shape[-1])
                pred = torch.topk(flat_logits, k=k, dim=-1).indices
                top1[head_idx] += int((pred[:, :1] == targets[:, None]).any(dim=-1).sum().item())
                top5[head_idx] += int((pred[:, : min(5, k)] == targets[:, None]).any(dim=-1).sum().item())
                top10[head_idx] += int((pred == targets[:, None]).any(dim=-1).sum().item())

    model.train()
    model.model.eval()
    model.lm_head.eval()
    model.medusa_head.train()

    head_top1 = [top1[i] / totals[i] if totals[i] else 0.0 for i in range(model.medusa)]
    head_top5 = [top5[i] / totals[i] if totals[i] else 0.0 for i in range(model.medusa)]
    head_top10 = [top10[i] / totals[i] if totals[i] else 0.0 for i in range(model.medusa)]
    weighted_score = 0.0
    weight_sum = 0.0
    for i in range(model.medusa):
        w = float(head_weights[i])
        weighted_score += w * (head_top1[i] + 0.25 * head_top5[i] + 0.10 * head_top10[i])
        weight_sum += w
    score = weighted_score / max(weight_sum, 1e-9)
    loss = loss_sum / max(float(token_count), 1.0)
    return EvalMetrics(
        loss=loss,
        score=score,
        tokens=token_count,
        head_top1=head_top1,
        head_top5=head_top5,
        head_top10=head_top10,
    )


def make_output_config(model: MedusaModel, args: argparse.Namespace) -> MedusaConfig:
    config = MedusaConfig(
        medusa_num_heads=int(model.medusa),
        medusa_num_layers=int(model.medusa_num_layers),
        base_model_name_or_path=getattr(model.config, "_name_or_path", None)
        or getattr(model, "base_model_name_or_path", None),
        version="2",
    )
    config.medusa_head_uses_base_lm_head = True
    return config


def convert_state_dtype(state: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().to(dtype=dtype) for key, value in state.items()}


def save_head_folder(
    output_dir: Path,
    model: MedusaModel,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    metrics: EvalMetrics,
    step: int,
    reason: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_output_config(model, args)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    state = convert_state_dtype(model.medusa_head.state_dict(), save_dtype)
    save_file(state, str(output_dir / "medusa_lm_head.safetensors"))
    metadata = {
        "step": step,
        "reason": reason,
        "metrics": asdict(metrics),
        "args": vars(args),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "training_metrics.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def copy_tokenizer_sidecars_if_needed(init_dir: Path, output_dir: Path) -> None:
    for name in ("chat_template.jinja",):
        src = init_dir / name
        dst = output_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def format_metrics(metrics: EvalMetrics) -> str:
    top1 = ",".join(f"{value:.3f}" for value in metrics.head_top1)
    top5 = ",".join(f"{value:.3f}" for value in metrics.head_top5)
    top10 = ",".join(f"{value:.3f}" for value in metrics.head_top10)
    return (
        f"loss={metrics.loss:.4f} score={metrics.score:.4f} "
        f"top1=[{top1}] top5=[{top5}] top10=[{top10}]"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    init_dir = Path(args.init_medusa_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if init_dir == output_dir:
        raise ValueError("--output-dir must be different from --init-medusa-dir.")

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This will be extremely slow on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dtype = resolve_dtype(args.dtype, device)
    save_dtype = train_dtype if args.save_dtype == "train" else resolve_dtype(args.save_dtype, device)

    tokenizer = load_training_tokenizer(args.init_medusa_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer needs a pad/eos/unk token for batching.")

    texts = load_texts(Path(args.data_path), tokenizer, args.max_samples, args.seed)
    train_texts, val_texts = split_texts(texts, args.val_ratio, args.val_samples)
    train_dataset = TokenWindowDataset(
        train_texts,
        tokenizer,
        seq_len=args.seq_len,
        min_seq_len=args.min_seq_len,
        pack=not args.no_pack,
    )
    val_dataset = TokenWindowDataset(
        val_texts,
        tokenizer,
        seq_len=args.seq_len,
        min_seq_len=args.min_seq_len,
        pack=not args.no_pack,
    )
    collator = PadCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = prepare_model(args, device, train_dtype)
    head_weights = parse_head_weights(args.head_loss_weights, int(model.medusa))
    trainable_params = sum(param.numel() for param in model.medusa_head.parameters() if param.requires_grad)
    print(f"device={device} dtype={train_dtype} trainable_head_params={trainable_params:,}")
    print(f"train_windows={len(train_dataset)} val_windows={len(val_dataset)} seq_len={args.seq_len}")

    optimizer = torch.optim.AdamW(
        [param for param in model.medusa_head.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_scaler = device.type == "cuda" and train_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler) if use_scaler else None

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = int(round(total_steps * args.warmup_ratio))

    initial_metrics = evaluate(
        model,
        val_loader,
        device,
        train_dtype,
        head_weights,
        args.gt_loss_weight,
        args.loss_token_chunk_size,
        args.argmax_token_chunk_size,
        args.eval_batches,
    )
    print(f"initial: {format_metrics(initial_metrics)}")
    best_score = initial_metrics.score if args.keep_initial_if_worse else -float("inf")
    best_step = 0
    if args.keep_initial_if_worse:
        save_head_folder(
            output_dir,
            model,
            tokenizer,
            args,
            save_dtype,
            initial_metrics,
            step=0,
            reason="initial_floor",
        )
        copy_tokenizer_sidecars_if_needed(init_dir, output_dir)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_steps, desc="training", dynamic_ncols=True)
    rolling_loss = 0.0
    rolling_batches = 0

    while global_step < total_steps:
        for micro_idx, batch in enumerate(train_loader):
            lr = cosine_lr(
                args.learning_rate,
                args.min_learning_rate_ratio,
                global_step,
                total_steps,
                warmup_steps,
            )
            set_optimizer_lr(optimizer, lr)
            stats = backward_train_batch(
                model,
                batch,
                device,
                scaler,
                train_dtype,
                head_weights,
                args.gt_loss_weight,
                args.loss_token_chunk_size,
                args.argmax_token_chunk_size,
                loss_scale=1.0 / float(args.grad_accum),
            )
            rolling_loss += stats["loss"]
            rolling_batches += 1

            should_step = ((micro_idx + 1) % args.grad_accum == 0) or (micro_idx + 1 == len(train_loader))
            if not should_step:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.medusa_head.parameters(), args.grad_clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            progress.update(1)
            if args.log_every > 0 and global_step % args.log_every == 0:
                avg_loss = rolling_loss / max(1, rolling_batches)
                progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                rolling_loss = 0.0
                rolling_batches = 0

            should_eval = args.eval_every > 0 and global_step % args.eval_every == 0
            if should_eval or global_step >= total_steps:
                metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    train_dtype,
                    head_weights,
                    args.gt_loss_weight,
                    args.loss_token_chunk_size,
                    args.argmax_token_chunk_size,
                    args.eval_batches,
                )
                print(f"\nstep {global_step}: {format_metrics(metrics)}")
                if metrics.score > best_score + args.min_save_improvement:
                    best_score = metrics.score
                    best_step = global_step
                    save_head_folder(
                        output_dir,
                        model,
                        tokenizer,
                        args,
                        save_dtype,
                        metrics,
                        step=global_step,
                        reason="validation_improved",
                    )
                    copy_tokenizer_sidecars_if_needed(init_dir, output_dir)
                    print(f"saved new best to {output_dir}")
                elif args.keep_initial_if_worse:
                    print(f"kept existing best score={best_score:.4f} from step {best_step}")

            if global_step >= total_steps:
                break

    progress.close()
    print(f"done. best_score={best_score:.4f} best_step={best_step} output_dir={output_dir}")


if __name__ == "__main__":
    main()
