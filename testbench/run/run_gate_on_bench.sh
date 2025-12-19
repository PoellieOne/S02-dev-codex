#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TB_ROOT="${REPO_ROOT}/testbench"

# Make repo-root importable for "import sym_cycles"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${TB_ROOT}/out"
LOG_DIR="${TB_ROOT}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

IN_CSV_DEFAULT="${OUT_DIR}/core0_events_v1_9_bench.csv"
IN_CSV="${1:-$IN_CSV_DEFAULT}"
OUT_CSV="${2:-${OUT_DIR}/core0_events_v1_9_bench__gate.csv}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_OUT="${LOG_DIR}/gate_on_bench_${STAMP}.out.log"
LOG_ERR="${LOG_DIR}/gate_on_bench_${STAMP}.err.log"

cd "$REPO_ROOT"

echo "[i] REPO_ROOT: $REPO_ROOT" | tee "$LOG_OUT"
echo "[i] PYTHONPATH: $PYTHONPATH" | tee -a "$LOG_OUT"
echo "[i] INPUT:     $IN_CSV" | tee -a "$LOG_OUT"
echo "[i] OUTPUT:    $OUT_CSV" | tee -a "$LOG_OUT"

set +e
python3 "${TB_ROOT}/run/gate_on_bench_csv.py" "$IN_CSV" "$OUT_CSV" \
  > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)
EC=$?
set -e

echo "[i] EXIT_CODE=$EC" | tee -a "$LOG_OUT"
echo "[i] stdout: $LOG_OUT" | tee -a "$LOG_OUT"
echo "[i] stderr: $LOG_ERR" | tee -a "$LOG_OUT"

if [ "$EC" -ne 0 ]; then
  echo "[!] gate failed; see $LOG_ERR" | tee -a "$LOG_OUT"
  exit "$EC"
fi

echo "[✓] gate done" | tee -a "$LOG_OUT"
echo "[✓] wrote: $OUT_CSV" | tee -a "$LOG_OUT"
