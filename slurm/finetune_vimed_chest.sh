#!/bin/bash -l
#SBATCH --job-name=vimed_chest_ft
#SBATCH --output=/home/thahoa/random/%x-%j.out
#SBATCH --error=/home/thahoa/random/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1

set -euo pipefail

log() {
  echo "[$(date)] $*"
}

module load cuda
module load miniforge3

eval "$(mamba shell hook --shell bash)"
mamba activate runai

export PLAN_ROOT=/home/thahoa/PET/ViMed/vimed_petct_atlas_finetune_plan
export PILLAR_ROOT=/home/thahoa/PET/Pillar-0/pillar-finetune
export HF_HOME=/scratch/thahoa/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/thahoa/.cache/huggingface/hub
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"

cd "${PILLAR_ROOT}"

log "Starting ViMED chest PET/CT fine-tuning"
log "PLAN_ROOT=${PLAN_ROOT}"
log "PILLAR_ROOT=${PILLAR_ROOT}"
log "HF_HOME=${HF_HOME}"

python scripts/train.py "${PLAN_ROOT}/pillar_finetune_patch/configs/vimed_chest_petct_atlas.yaml"
