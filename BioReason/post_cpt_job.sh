#!/bin/bash
# post_cpt_job.sh — after cpt_job.sh finishes, launch Base-SFT/RL and CPT-SFT/RL.
# Usage:
#   bash post_cpt_job.sh [CPT_SWEEP_ID]          # waits for any running cpt-* jobs
#   CPT_SWEEP_ID defaults to the newest cpt_ffw_* under CHECKPOINT_DIR/drive/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/n/holylfs06/LABS/mzitnik_lab/Lab/${USER}/evo_tfm/BioReason/checkpoints}"
CPT_SWEEP_ID="${1:-$(ls -1dt "${CHECKPOINT_DIR}"/drive/cpt_ffw_[0-9]*/ 2>/dev/null | head -1 | xargs -r -n1 basename)}"
[[ -n "${CPT_SWEEP_ID}" ]] || { echo "no CPT sweep id (pass as arg, or ensure cpt_ffw_* exists)" >&2; exit 1; }

CPT_JIDS=$(squeue -h -u "${USER}" -o '%i %j' | awk '$2 ~ /^cpt-/{print $1}' | paste -sd: -)
DEP=""; [[ -n "${CPT_JIDS}" ]] && DEP="--dependency=afterany:${CPT_JIDS}"
mkdir -p "${SCRIPT_DIR}/logs/post_cpt"
echo "CPT sweep: ${CPT_SWEEP_ID} ; gating jids: ${CPT_JIDS:-<none; cpt already done>}"

WRAP=$(cat <<'EOF'
set -euo pipefail
ROOT="${CHECKPOINT_DIR}/drive/${CPT_SWEEP_ID}"
pick() {
  python3 - "$ROOT" "$1" "$2" <<'PY'
import glob, json, os, sys
root, needle, tag = sys.argv[1], sys.argv[2], sys.argv[3]
best = None
for f in glob.glob(os.path.join(root, f"*{needle}*", "final_eval_metrics.json")):
    final_dir = os.path.join(os.path.dirname(f), "final")
    if not os.path.isdir(final_dir):
        continue
    el = json.load(open(f)).get("eval_loss")
    if el is None:
        continue
    if best is None or el < best[1]:
        best = (final_dir, el)
if best:
    print(f"{best[0]}:{tag}")
PY
}
M1=$(pick "Qwen3-1.7B" "cpt_qwen3_1p7b")
M4=$(pick "Qwen3-4B"   "cpt_qwen3_4b")
SPEC=""
[[ -n "$M1" ]] && SPEC="$M1"
[[ -n "$M4" ]] && SPEC="${SPEC:+$SPEC }$M4"
[[ -n "$SPEC" ]] || { echo "[post-cpt] no CPT final/ found under $ROOT" >&2; exit 1; }
echo "[post-cpt] MODELS_OVERRIDE=$SPEC"
STAMP=$(date +%Y%m%d_%H%M%S)
cd "${SCRIPT_DIR}"
MODELS_OVERRIDE="$SPEC" SWEEP_ID="post_cpt_${STAMP}" bash submit_drive_genomics_sweeps.sh
EOF
)

sbatch --parsable ${DEP} \
  --partition="${POST_PARTITION:-shared}" --account="${POST_ACCOUNT:-barak_lab}" \
  --time="${POST_TIME:-0-01:00}" --mem=4G --cpus-per-task=2 --job-name=post-cpt \
  --output="${SCRIPT_DIR}/logs/post_cpt/%j.out" --error="${SCRIPT_DIR}/logs/post_cpt/%j.err" \
  --export=ALL,SCRIPT_DIR="${SCRIPT_DIR}",CHECKPOINT_DIR="${CHECKPOINT_DIR}",CPT_SWEEP_ID="${CPT_SWEEP_ID}" \
  --wrap="${WRAP}"
