#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./medusa_env/bin/python}"

if [[ -z "${MODEL_DIR:-}" ]]; then
  if [[ -d "./llama32_1b_medusa_heads_code" ]]; then
    MODEL_DIR="./llama32_1b_medusa_heads_code"
  elif [[ -d "./TinyLlama-1.1B-Chat-v1.0-4heads" ]]; then
    MODEL_DIR="./TinyLlama-1.1B-Chat-v1.0-4heads"
  else
    echo "Set MODEL_DIR to a local Medusa model directory." >&2
    exit 2
  fi
fi

CONTEXTS="${CONTEXTS:-1024 2048 4096 8192 16384}"
TARGET_NEW_TOKENS="${TARGET_NEW_TOKENS:-48}"
MAX_STEPS="${MAX_STEPS:-32}"
KV_MARGIN="${KV_MARGIN:-1024}"
PROMPT_SUITE="${PROMPT_SUITE:-coding}"
OUT_DIR="${OUT_DIR:-artifacts/benchmarks/medusa/local_turbo_context_sweep_$(date +%Y%m%d_%H%M%S)}"
MODES="${MODES:-medusa_base,turboquant_prod_b4_full_tree,turboquant_prod_outlier_k4v4_c16_full_tree}"

mkdir -p "$OUT_DIR"

for ctx in $CONTEXTS; do
  kv_max_length=$((ctx + TARGET_NEW_TOKENS + KV_MARGIN))
  out_csv="$OUT_DIR/context_${ctx}.csv"
  echo "running context=${ctx} kv_max_length=${kv_max_length} -> ${out_csv}"
  "$PYTHON_BIN" bench_comm_turbo.py \
    --model-dir "$MODEL_DIR" \
    --out-csv "$out_csv" \
    --long-context-tokens "$ctx" \
    --long-only \
    --prompt-suite "$PROMPT_SUITE" \
    --target-new-tokens "$TARGET_NEW_TOKENS" \
    --max-steps "$MAX_STEPS" \
    --kv-max-length "$kv_max_length" \
    --only "$MODES" \
    --no-paper-metrics \
    ${EXTRA_ARGS:-}
done

"$PYTHON_BIN" - "$OUT_DIR"/context_*.csv "$OUT_DIR/summary.csv" <<'PY'
import csv
import sys

inputs = sys.argv[1:-1]
out_path = sys.argv[-1]
fields = [
    "category",
    "mode",
    "prompt_tokens",
    "tokens",
    "total_s",
    "ttft_s",
    "tps",
    "speedup_vs_base",
    "prefix_match_vs_base",
    "peak_alloc_mb",
    "peak_reserved_mb",
    "context_utilization",
    "fp16_kv_mb_est",
    "turbo_vq_kv_mb_est",
    "turbo_vq_transfer_reduction",
    "accepted_tokens_per_step",
    "verified_nodes_per_step",
]

rows = []
for path in inputs:
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({field: row.get(field, "") for field in fields})

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"wrote {out_path}")
for row in rows:
    print(
        row["category"],
        row["mode"],
        f"tps={row['tps']}",
        f"speedup={row['speedup_vs_base']}",
        f"alloc_mb={row['peak_alloc_mb']}",
    )
PY
