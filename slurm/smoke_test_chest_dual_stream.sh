#!/bin/bash -l
#SBATCH --job-name=vimed_chest_dual_smoke
#SBATCH --output=/home/thahoa/PET/ViMed/finetune_logs/%x-%j.out
#SBATCH --error=/home/thahoa/PET/ViMed/finetune_logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=standard

set -euo pipefail

log() {
  echo "[$(date)] $*"
}

log "Starting chest dual-stream smoke test"
log "Job ID: ${SLURM_JOB_ID:-unknown}"
log "Host: $(hostname)"

module load miniforge3
eval "$(mamba shell hook --shell bash)"
mamba activate runai

export PLAN_ROOT=/home/thahoa/PET/ViMed/vimed_petct_atlas_finetune_plan
export OUT_DIR=/scratch/thahoa/PET/ViMed_prep_v2

mkdir -p /home/thahoa/PET/ViMed/finetune_logs

python "${PLAN_ROOT}/scripts/smoke_test_chest_dual_stream.py"   --manifest "${OUT_DIR}/manifest_splits.csv"   --split train   --index 0   --include-raw

log "Finished chest dual-stream smoke test"
