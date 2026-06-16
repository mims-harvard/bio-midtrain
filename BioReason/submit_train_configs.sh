#!/bin/bash
# Submit train configs while applying OOM-safe overrides to non-exempt job IDs.
cd "$(dirname "$0")"

# Keep these IDs unchanged; apply OOM-safe micro-batch to all others.
# EXEMPT_IDS=(2 5 6)
EXEMPT_IDS=(3 4 5 6 9 10 11 12) # KEGG - CONFIG_ID - 1, 2, 7, 8

is_exempt_id() {
  local id="$1"
  local exempt
  for exempt in "${EXEMPT_IDS[@]}"; do
    [[ "$id" == "$exempt" ]] && return 0
  done
  return 1
}

for i in $(seq 1 12); do
  if is_exempt_id "$i"; then
    continue
  fi
  # FORCE_BATCH_SIZE=1 lowers peak GPU memory to reduce CUDA OOM risk.
  sbatch --job-name="bio-${i}" --export=ALL,CONFIG_ID="$i",FORCE_BATCH_SIZE=1 job.sh
done
