#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TB_ROOT="${REPO_ROOT}/testbench"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export S02_TESTBENCH_ROOT="${TB_ROOT}"
export S02_LOG_DIR="${TB_ROOT}/logs"
export S02_OUT_DIR="${TB_ROOT}/out"
export S02_DATA_DIR="${TB_ROOT}/data"

mkdir -p "$S02_LOG_DIR" "$S02_OUT_DIR" "$S02_DATA_DIR"
