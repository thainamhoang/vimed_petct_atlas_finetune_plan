#!/bin/bash -l
#SBATCH --job-name=vimed_qc_v2
#SBATCH --output=/home/thahoa/PET/ViMed/finetune_logs/%x-%j.out
#SBATCH --error=/home/thahoa/PET/ViMed/finetune_logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard

set -euo pipefail

log() {
  echo "[$(date)] $*"
}

log "Starting ViMED QC montage generation"
log "Job ID: ${SLURM_JOB_ID:-unknown}"
log "Host: $(hostname)"

module load miniforge3
eval "$(mamba shell hook --shell bash)"
mamba activate runai

export PLAN_ROOT=/home/thahoa/PET/ViMed/vimed_petct_atlas_finetune_plan
export OUT_DIR=/scratch/thahoa/PET/ViMed_prep_v2

mkdir -p /home/thahoa/PET/ViMed/finetune_logs "${OUT_DIR}/qc"

python "${PLAN_ROOT}/scripts/make_qc_montages.py"   --manifest "${OUT_DIR}/manifest_splits.csv"   --out-dir "${OUT_DIR}/qc"   --num-samples 50

log "Finished ViMED QC montage generation"
