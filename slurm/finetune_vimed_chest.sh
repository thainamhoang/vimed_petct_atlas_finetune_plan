#!/bin/bash -l
#SBATCH --job-name=vimed_chest_ft
#SBATCH --output=/home/thahoa/PET/ViMed/finetune_logs/%x-%j.out
#SBATCH --error=/home/thahoa/PET/ViMed/finetune_logs/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=H100:1

set -euo pipefail

log() {
  echo "[$(date)] $*"
}

module load cuda
module load miniforge3 dev2025a cmake cuda h100

eval "$(mamba shell hook --shell bash)"
mamba activate runai

export PLAN_ROOT=/home/thahoa/PET/ViMed/vimed_petct_atlas_finetune_plan
export PILLAR_ROOT=/home/thahoa/PET/Pillar-0/pillar-finetune
export ENV_FILE="${PLAN_ROOT}/.env"
export HF_HOME=/scratch/thahoa/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/thahoa/.cache/huggingface/hub
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" /home/thahoa/PET/ViMed/finetune_logs

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

cd "${PILLAR_ROOT}"

log "Starting ViMED chest PET/CT fine-tuning"
log "PLAN_ROOT=${PLAN_ROOT}"
log "PILLAR_ROOT=${PILLAR_ROOT}"
log "HF_HOME=${HF_HOME}"
log "ENV_FILE=${ENV_FILE}"

python scripts/train.py "${PLAN_ROOT}/pillar_finetune_patch/configs/vimed_chest_petct_atlas.yaml"
