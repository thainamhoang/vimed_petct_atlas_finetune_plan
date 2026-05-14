#!/bin/bash -l
#SBATCH --job-name=vimed_prep_v2
#SBATCH --output=/home/thahoa/PET/ViMed/finetune_logs/%x-%A_%a.out
#SBATCH --error=/home/thahoa/PET/ViMed/finetune_logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --partition=standard

set -euo pipefail

log() {
  echo "[$(date)] $*"
}

log "Starting ViMED preprocessing"
log "Job ID: ${SLURM_JOB_ID:-unknown}"
log "Array task ID: ${SLURM_ARRAY_TASK_ID:-0}"
log "Array task count: ${SLURM_ARRAY_TASK_COUNT:-1}"
log "Host: $(hostname)"

module load miniforge3
eval "$(mamba shell hook --shell bash)"
mamba activate runai

export PLAN_ROOT=/home/thahoa/PET/ViMed/vimed_petct_atlas_finetune_plan
export DATA_ROOT=/scratch/thahoa/PET/ViMed
export OUT_DIR=/scratch/thahoa/PET/ViMed_prep_v2

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${OUT_DIR}" /home/thahoa/PET/ViMed/finetune_logs

python "${PLAN_ROOT}/scripts/preprocess_vimed_petct.py"   --data-root "${DATA_ROOT}"   --out-dir "${OUT_DIR}"   --label-rules "${PLAN_ROOT}/configs/label_rules.json"   --num-shards "${SLURM_ARRAY_TASK_COUNT}"   --shard-index "${SLURM_ARRAY_TASK_ID}"   --overwrite

log "Finished shard ${SLURM_ARRAY_TASK_ID}"
