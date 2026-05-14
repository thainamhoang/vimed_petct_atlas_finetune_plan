#!/bin/bash
set -euo pipefail

PLAN_ROOT=/home/thahoa/PET/ViMed/vimed_petct_atlas_finetune_plan
DATA_ROOT=/scratch/thahoa/PET/ViMed
OUT_DIR=/scratch/thahoa/PET/ViMed_prep_v2
LOG_DIR=/home/thahoa/PET/ViMed/finetune_logs

mkdir -p "${LOG_DIR}"

prep_jobid=$(sbatch --parsable <<EOF
#!/bin/bash -l
#SBATCH --job-name=vimed_prep_v2
#SBATCH --output=${LOG_DIR}/%x-%A_%a.out
#SBATCH --error=${LOG_DIR}/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-7
#SBATCH --partition=standard

set -euo pipefail

module load miniforge3
eval "\$(mamba shell hook --shell bash)"
mamba activate runai

export PLAN_ROOT=${PLAN_ROOT}
export DATA_ROOT=${DATA_ROOT}
export OUT_DIR=${OUT_DIR}
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=\${SLURM_CPUS_PER_TASK}

mkdir -p "\${OUT_DIR}" "${LOG_DIR}"

python "\${PLAN_ROOT}/scripts/preprocess_vimed_petct.py"   --data-root "\${DATA_ROOT}"   --out-dir "\${OUT_DIR}"   --label-rules "\${PLAN_ROOT}/configs/label_rules.json"   --num-shards "\${SLURM_ARRAY_TASK_COUNT}"   --shard-index "\${SLURM_ARRAY_TASK_ID}"   --overwrite
EOF
)

echo "Submitted preprocessing array job: ${prep_jobid}"

post_jobid=$(sbatch --parsable --dependency=afterok:${prep_jobid} <<EOF
#!/bin/bash -l
#SBATCH --job-name=vimed_post_v2
#SBATCH --output=${LOG_DIR}/%x-%j.out
#SBATCH --error=${LOG_DIR}/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=standard

set -euo pipefail

module load miniforge3
eval "\$(mamba shell hook --shell bash)"
mamba activate runai

export PLAN_ROOT=${PLAN_ROOT}
export OUT_DIR=${OUT_DIR}

mkdir -p "\${OUT_DIR}/qc" "${LOG_DIR}"

python "\${PLAN_ROOT}/scripts/combine_manifests.py"   --inputs "\${OUT_DIR}"/manifest_shard*.csv   --output "\${OUT_DIR}/manifest.csv"

python "\${PLAN_ROOT}/scripts/combine_manifests.py"   --inputs "\${OUT_DIR}"/skipped_shard*.csv   --output "\${OUT_DIR}/skipped_all.csv"

python "\${PLAN_ROOT}/scripts/split_manifest.py"   --manifest "\${OUT_DIR}/manifest.csv"   --output "\${OUT_DIR}/manifest_splits.csv"

python "\${PLAN_ROOT}/scripts/make_qc_montages.py"   --manifest "\${OUT_DIR}/manifest_splits.csv"   --out-dir "\${OUT_DIR}/qc"   --num-samples 50
EOF
)

echo "Submitted postprocess/QC job: ${post_jobid}"
echo "Pipeline submitted successfully."
