#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <plan_root> <pillar_root>"
  exit 1
fi

PLAN_ROOT="$1"
PILLAR_ROOT="$2"

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required but not found in PATH" >&2
  exit 1
fi

mkdir -p \
  "${PILLAR_ROOT}/pillar/datasets" \
  "${PILLAR_ROOT}/pillar/engines" \
  "${PILLAR_ROOT}/pillar/losses" \
  "${PILLAR_ROOT}/pillar/models" \
  "${PILLAR_ROOT}/pillar/models/backbones" \
  "${PILLAR_ROOT}/pillar/metrics" \
  "${PILLAR_ROOT}/configs" \
  "${PILLAR_ROOT}/scripts"

rsync -a "${PLAN_ROOT}/pillar_finetune_patch/pillar/datasets/" "${PILLAR_ROOT}/pillar/datasets/"
rsync -a "${PLAN_ROOT}/pillar_finetune_patch/pillar/engines/" "${PILLAR_ROOT}/pillar/engines/"
rsync -a "${PLAN_ROOT}/pillar_finetune_patch/pillar/losses/" "${PILLAR_ROOT}/pillar/losses/"
rsync -a "${PLAN_ROOT}/pillar_finetune_patch/pillar/models/" "${PILLAR_ROOT}/pillar/models/"
rsync -a "${PLAN_ROOT}/pillar_finetune_patch/pillar/metrics/" "${PILLAR_ROOT}/pillar/metrics/"
rsync -a "${PLAN_ROOT}/pillar_finetune_patch/configs/" "${PILLAR_ROOT}/configs/"
rsync -a "${PLAN_ROOT}/pillar_finetune_patch/scripts/" "${PILLAR_ROOT}/scripts/"

echo "Applied ViMED chest PET/CT patch into ${PILLAR_ROOT}"
