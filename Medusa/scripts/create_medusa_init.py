#!/usr/bin/env python3
"""Create a config-only Medusa head folder for scratch head training."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer

from medusa.model.medusa_model import MedusaConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="HF repo id or local base model path.")
    parser.add_argument("--output-dir", required=True, help="Folder to create.")
    parser.add_argument("--medusa-num-heads", type=int, default=4)
    parser.add_argument("--medusa-num-layers", type=int, default=1)
    parser.add_argument("--draft-head-type", choices=("medusa", "hydra"), default="medusa")
    parser.add_argument(
        "--version",
        default="2",
        help="Use '2' to train lightweight heads that reuse the base LM head.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MedusaConfig(
        medusa_num_heads=args.medusa_num_heads,
        medusa_num_layers=args.medusa_num_layers,
        base_model_name_or_path=args.base_model,
        version=args.version,
        draft_head_type=args.draft_head_type,
    )
    if str(args.version).lower() in {"2", "medusa2", "medusa-2"}:
        config.medusa_head_uses_base_lm_head = True
    config.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.save_pretrained(output_dir)
    print(f"wrote config-only Medusa init folder: {output_dir}")


if __name__ == "__main__":
    main()
