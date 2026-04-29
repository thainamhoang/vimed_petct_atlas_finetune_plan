#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <plan_root> <pillar_root>"
  exit 1
fi

PLAN_ROOT="$1"
PILLAR_ROOT="$2"

cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/datasets/vimed_petct.py" "${PILLAR_ROOT}/pillar/datasets/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/datasets/__init__.py" "${PILLAR_ROOT}/pillar/datasets/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/engines/base.py" "${PILLAR_ROOT}/pillar/engines/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/engines/classifier.py" "${PILLAR_ROOT}/pillar/engines/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/losses/multilabel.py" "${PILLAR_ROOT}/pillar/losses/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/losses/__init__.py" "${PILLAR_ROOT}/pillar/losses/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/models/multi_stage.py" "${PILLAR_ROOT}/pillar/models/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/models/backbones/mmatlas.py" "${PILLAR_ROOT}/pillar/models/backbones/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/metrics/multilabel.py" "${PILLAR_ROOT}/pillar/metrics/"
cp "${PLAN_ROOT}/pillar_finetune_patch/pillar/metrics/__init__.py" "${PILLAR_ROOT}/pillar/metrics/"
cp "${PLAN_ROOT}/pillar_finetune_patch/configs/vimed_chest_petct_atlas.yaml" "${PILLAR_ROOT}/configs/"

echo "Applied ViMED chest PET/CT patch into ${PILLAR_ROOT}"
