#!/usr/bin/env bash
# Run the requested chronological prepare/train/validate/test workflow.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/run.py" "$@"
