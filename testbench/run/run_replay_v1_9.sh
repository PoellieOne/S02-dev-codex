#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TB_ROOT="${REPO_ROOT}/testbench"

# Make repo-root importable for "import sym_cycles"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

INPUT_DEFAULT="${TB_ROOT}/data/core0_events.jsonl"
INPUT_PATH="${1:-$INPUT_DEFAULT}"

OUT_DIR="${TB_ROOT}/out"
LOG_DIR="${TB_ROOT}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_OUT="${LOG_DIR}/replay_v1_9_${STAMP}.out.log"
LOG_ERR="${LOG_DIR}/replay_v1_9_${STAMP}.err.log"

OUT_CSV="${OUT_DIR}/core0_events_v1_9_bench.csv"

cd "$REPO_ROOT"

echo "[i] REPO_ROOT: $REPO_ROOT" | tee "$LOG_OUT"
echo "[i] PYTHONPATH: $PYTHONPATH" | tee -a "$LOG_OUT"
echo "[i] INPUT:     $INPUT_PATH" | tee -a "$LOG_OUT"
echo "[i] OUTPUT:    $OUT_CSV" | tee -a "$LOG_OUT"
echo "[i] LOGS:      $LOG_OUT / $LOG_ERR" | tee -a "$LOG_OUT"

python3 "${REPO_ROOT}/scripts/replay_core0_events_v1_9.py" \
  "$INPUT_PATH" "$OUT_CSV" --profile bench \
  > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

echo "[✓] replay done" | tee -a "$LOG_OUT"
echo "[✓] wrote: $OUT_CSV" | tee -a "$LOG_OUT"
