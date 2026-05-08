#!/usr/bin/env python3
"""Filter a mixed instruction JSONL into code/technical examples.

The Kaggle mix used earlier does not store source labels, so this script uses a
conservative text classifier to keep programming, systems, math, and debugging
examples while dropping obvious creative/essay/marketing/travel rows.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


CODE_TERMS = {
    "algorithm",
    "api",
    "array",
    "bash",
    "bug",
    "c++",
    "class",
    "code",
    "compiler",
    "complexity",
    "cuda",
    "debug",
    "dictionary",
    "docker",
    "function",
    "git",
    "implement",
    "java",
    "javascript",
    "kernel",
    "leetcode",
    "linux",
    "method",
    "mpi",
    "numpy",
    "openmp",
    "pandas",
    "program",
    "python",
    "regex",
    "runtime",
    "script",
    "sql",
    "stack trace",
    "tensor",
    "test case",
    "unit test",
}

STRONG_CODE_TERMS = {
    "api",
    "array",
    "bash",
    "bug",
    "c++",
    "class",
    "code",
    "compiler",
    "cuda",
    "debug",
    "dictionary",
    "docker",
    "function",
    "git",
    "implement",
    "java",
    "javascript",
    "kernel",
    "linux",
    "method",
    "mpi",
    "numpy",
    "openmp",
    "pandas",
    "program",
    "python",
    "regex",
    "script",
    "sql",
    "stack trace",
    "tensor",
    "test case",
    "unit test",
}

TECH_TERMS = {
    "cache",
    "database",
    "distributed",
    "gpu",
    "latency",
    "memory",
    "network",
    "operating system",
    "parallel",
    "performance",
    "server",
    "throughput",
}

MATH_TERMS = {
    "equation",
    "geometry",
    "probability",
    "solve",
    "theorem",
}

NEGATIVE_TERMS = {
    "article",
    "atmospheric",
    "brand awareness",
    "classroom",
    "contest",
    "creative story",
    "customer",
    "essay",
    "fiction",
    "film",
    "food",
    "healthcare",
    "hotel",
    "marketing",
    "membrane",
    "patient",
    "poem",
    "restaurant",
    "screenplay",
    "social media",
    "story",
    "summarize",
    "television",
    "travel",
    "vacation",
    "whitepaper",
    "writer",
}

CODE_SYNTAX_PATTERNS = [
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"```",
        r"#include\s*<",
        r"\bdef\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
        r"\bfunction\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*\(",
        r"\bSELECT\b.+\bFROM\b",
        r"\bpublic\s+static\b",
        r"\bclass\s+[a-zA-Z_][a-zA-Z0-9_]*",
        r"\bimport\s+[a-zA-Z_][a-zA-Z0-9_.]*",
        r"\breturn\s+[^.\n]+;",
        r"console\.log\s*\(",
        r"\bfor\s*\(",
        r"\bwhile\s*\(",
        r"\bif\s*\(",
        r"=>\s*[{(]",
        r"\bvar\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*=",
        r"\blet\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*=",
        r"\bconst\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*=",
        r"\bprint\s*\(",
        r"\bcout\s*<<",
        r"\bcin\s*>>",
    )
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a code-focused Medusa training JSONL.")
    parser.add_argument("--input", required=True, help="Input mixed JSONL.")
    parser.add_argument("--output", required=True, help="Output filtered JSONL.")
    parser.add_argument("--max-rows", type=int, default=0, help="Cap output rows after filtering/shuffle.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score", type=int, default=2)
    parser.add_argument(
        "--strict-code",
        action="store_true",
        help="Require strong programming signals, not just general technical writing.",
    )
    parser.add_argument(
        "--code-syntax-only",
        action="store_true",
        help="Keep only rows that contain code-like syntax in the prompt/answer.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Drop rows whose flattened text is longer than this many chars. 0 disables.",
    )
    return parser.parse_args()


def collect_text(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, str):
        parts.append(value)
    elif isinstance(value, dict):
        for key in ("content", "value", "text", "instruction", "input", "output", "prompt", "response"):
            if isinstance(value.get(key), str):
                parts.append(value[key])
        for key in ("messages", "conversations"):
            if isinstance(value.get(key), list):
                parts.append(collect_text(value[key]))
    elif isinstance(value, list):
        for item in value:
            parts.append(collect_text(item))
    return "\n".join(part for part in parts if part)


def term_hits(text: str, terms: set[str]) -> int:
    return sum(1 for term in terms if re.search(rf"(?<![a-z0-9_]){re.escape(term)}(?![a-z0-9_])", text))


def score_row(row: dict[str, Any]) -> tuple[int, str, int, int, bool]:
    text = collect_text(row).lower()
    code_hits = term_hits(text, CODE_TERMS)
    strong_code_hits = term_hits(text, STRONG_CODE_TERMS)
    tech_hits = term_hits(text, TECH_TERMS)
    math_hits = term_hits(text, MATH_TERMS)
    negative_hits = term_hits(text, NEGATIVE_TERMS)
    has_fenced_code = "```" in text
    fenced_code = 2 if has_fenced_code else 0
    assignment_shape = 1 if re.search(r"\b(write|implement|debug|fix|optimi[sz]e)\b", text) else 0
    score = code_hits * 3 + tech_hits * 2 + math_hits + fenced_code + assignment_shape - negative_hits * 3
    return score, text, code_hits, strong_code_hits, has_fenced_code


def has_code_syntax(text: str) -> bool:
    return any(pattern.search(text) for pattern in CODE_SYNTAX_PATTERNS)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    kept: list[str] = []
    seen = 0
    scored = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            seen += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            score, _, code_hits, strong_code_hits, has_fenced_code = score_row(row)
            text = collect_text(row)
            if args.max_chars > 0 and len(text) > args.max_chars:
                continue
            if args.code_syntax_only and not has_code_syntax(text):
                continue
            strict_ok = (
                not args.strict_code
                or has_fenced_code
                or strong_code_hits >= 2
                or (strong_code_hits >= 1 and code_hits >= 2)
            )
            if strict_ok and score >= args.min_score:
                scored.append((score, line))

    rng.shuffle(scored)
    if args.max_rows > 0:
        scored = scored[: args.max_rows]
    kept = [line for _, line in scored]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    scores = [score for score, _ in scored]
    avg_score = sum(scores) / max(1, len(scores))
    print(f"read {seen} rows")
    print(f"kept {len(kept)} rows at min_score>={args.min_score}, avg_score={avg_score:.2f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
