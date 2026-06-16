#!/bin/bash
set -euo pipefail

# Submit KEGG SFT jobs with local genomics CSV data.
# Uses:
#   train_network_split.csv
#   id_test_network_split.csv
#   ood_test_network_split.csv
#
# Default configs follow run.sh mapping:
#   1,2 -> Qwen3-1.7B (kegg, 5/10 epochs)
#   7,8 -> Qwen3-4B   (kegg, 5/10 epochs)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKING_DIR="${WORKING_DIR:-${SCRIPT_DIR}}"
USER_NAME="${USER_NAME:-$(whoami)}"

PARTITION="${PARTITION:-kempner_h100}"
ACCOUNT="${ACCOUNT:-kempner_mzitnik_lab}"
TIME_LIMIT="${TIME_LIMIT:-3-00:00}"
CPUS="${CPUS:-24}"
MEM="${MEM:-60G}"
GPUS="${GPUS:-1}"
CONSTRAINT="${CONSTRAINT:-}"

PYTHON="${PYTHON_BIN:-/n/netscratch/kempner_sham_lab/Lab/hanlinzhang/envs/bio/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python interpreter not found or not executable: ${PYTHON}" >&2
  exit 1
fi

SCRATCH="${SCRATCH:-${BIOREASON_DATA_ROOT:-/n/holylfs06/LABS/mzitnik_lab/Everyone/data/bioreason}}"
GENOMICS_DIR="${GENOMICS_DIR:-${SCRATCH}/genomics}"
TRAIN_CSV="${GENOMICS_DIR}/train_network_split.csv"
ID_CSV="${GENOMICS_DIR}/id_test_network_split.csv"
OOD_CSV="${GENOMICS_DIR}/ood_test_network_split.csv"

for f in "${TRAIN_CSV}" "${ID_CSV}" "${OOD_CSV}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing required data file: ${f}" >&2
    exit 1
  fi
done

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found in PATH. Please run on a Slurm login node." >&2
  exit 1
fi

mkdir -p "${WORKING_DIR}/logs/drive"

CONFIG_IDS=(${CONFIG_IDS:-1 2 7 8})

echo "Submitting KEGG jobs with local genomics data: ${GENOMICS_DIR}"
echo "Test will include both ID and OOD via merge_val_test_set=True in run.sh."

for cfg in "${CONFIG_IDS[@]}"; do
  job_name="drive-kegg-c${cfg}"
  active_job_id=$(squeue -h -u "${USER_NAME}" --states=PD,R,CF,S,RS --name "${job_name}" -o "%i" | head -n 1 || true)
  if [[ -n "${active_job_id}" ]]; then
    echo "SKIP config=${cfg}: active job exists (${active_job_id})"
    continue
  fi

  sbatch_args=(
    --job-name="${job_name}"
    --partition="${PARTITION}"
    --account="${ACCOUNT}"
    --time="${TIME_LIMIT}"
    --nodes=1
    --ntasks=1
    --cpus-per-task="${CPUS}"
    --gpus="${GPUS}"
    --mem="${MEM}"
    --output="${WORKING_DIR}/logs/drive/${job_name}_%j.out"
    --error="${WORKING_DIR}/logs/drive/${job_name}_%j.err"
    --export=ALL,CONFIG_ID="${cfg}",KEGG_DATA_DIR_LOCAL="${GENOMICS_DIR}",PYTHON="${PYTHON}",WORKING_DIR="${WORKING_DIR}"
    --wrap='set -euo pipefail; cd "$WORKING_DIR"; bash run.sh'
  )

  if [[ -n "${CONSTRAINT}" ]]; then
    sbatch_args+=(--constraint="${CONSTRAINT}")
  fi

  submit_out=$(sbatch "${sbatch_args[@]}")
  echo "SUBMITTED config=${cfg}: ${submit_out}"
done

echo "Done."
