#!/bin/bash
# Evo2 → vortex imports flash_attn_2_cuda (flash-attn). Pip's default isolated build hides torch;
# install with --no-build-isolation. Needs nvcc + a GCC version CUDA supports (not gcc/15).
#
# Usage (after env.sh venv exists):
#   cd /path/to/BioReason && source "$SCRATCH/envs/bio/bin/activate"
#   bash install_evo2_stack.sh
#
# Or from a short GPU interactive job if login-node build is too slow or OOMs:
#   salloc -p kempner_h100 -t 1:00:00 --gpus=1 ...
#
# If pip logs HTTP 404 then a long nvcc/ninja trace ending in "Killed" + cicc: the 404 is
# normal (no prebuilt wheel). "Killed" is usually the OOM killer — lower MAX_JOBS or build
# on a compute node with more RAM. Optional: only A100 → export TORCH_CUDA_ARCH_LIST=8.0
set -euo pipefail

: "${SCRATCH:?SCRATCH must be set}"
VENV="${VENV:-${SCRATCH}/envs/bio}"
PIP="${VENV}/bin/pip"
PY="${VENV}/bin/python"

# Match env.sh: uv venv at $SCRATCH/envs/bio (not conda `activate bio`).
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

if [[ ! -x "$PIP" ]]; then
  echo "No venv at $VENV — run env.sh first." >&2
  exit 1
fi

module purge
module load gcc/12.2.0-fasrc01 cuda/12.4 cudnn/9.10.2.21_cuda12
export CPATH="${CUDNN_HOME}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${CUDNN_HOME}/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${LD_LIBRARY_PATH:-}"

# Cap ninja/nvcc parallelism: flash-attn defaults MAX_JOBS from free RAM, which is too high
# on shared login nodes and leads to cicc being OOM-killed (exit 255). Override on fat nodes.
export MAX_JOBS="${MAX_JOBS:-2}"

echo "Installing flash-attn (CUDA build, often 15–40+ min; nvcc is mostly silent without -v) ..."
# Verbose so batch/interactive logs show nvcc progress (uv pip can look \"stuck\" here).
"$PIP" install -v 'flash-attn==2.8.0.post2' --no-build-isolation

py_minor="$("$PY" -c 'import sys; print(sys.version_info[1])')"
if (( py_minor < 11 )); then
  echo "Installing evo2 for Python 3.10 ..."
  "$PIP" install 'evo2>=0.3.0,<0.4.0'
else
  echo "Installing evo2 (Python 3.11+) ..."
  "$PIP" install 'evo2>=0.5.0'
fi

echo "Smoke test ..."
"$PY" -c 'from evo2 import Evo2; print("evo2 import OK")'
