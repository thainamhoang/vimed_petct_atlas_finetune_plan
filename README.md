# ViMED PET/CT Atlas Finetune Plan

This folder contains a self-contained implementation kit for preparing ViMED
PET/CT NPZ data for Atlas/Pillar fine-tuning.

## What The Plan Implements

- ViMED region samples: head-neck, chest, abdomen-pelvis.
- Output cache format: Torch `.pt` files with `x_raw` shaped `(2, D, 256, 256)`.
- Channel 0 is CT clipped to `[-1024, 3071]`.
- Channel 1 is PET robust-normalized to `[0, 1]` with percentile clipping plus `log1p`.
- Training input conversion: `11 CT windows + 1 PET channel = 12 channels`.
- Weak report-derived labels from English regional report sections.
- Study-level train/val/test splitting to avoid leakage across regions.

Chest note: the ViMED paper gives exactly `8,271 = 2,757 x 3` region samples,
so chest is `33.33%` of region samples. It does not publish an exact raw
chest-slice percentage.

## Recommended Cluster Workflow

Install dependencies in your cluster environment:

```bash
pip install -r requirements.txt
```

Run preprocessing as an array job. Example with 16 shards:

```bash
python scripts/preprocess_vimed_petct.py \
  --data-root /path/to/ViMed/data \
  --out-dir /path/to/outputs/vimed_petct_atlas_256 \
  --label-rules configs/label_rules.json \
  --num-shards 16 \
  --shard-index ${SLURM_ARRAY_TASK_ID}
```

Combine shard manifests:

```bash
python scripts/combine_manifests.py \
  --inputs /path/to/outputs/vimed_petct_atlas_256/manifest_shard*.csv \
  --output /path/to/outputs/vimed_petct_atlas_256/manifest.csv
```

Assign study-level splits:

```bash
python scripts/split_manifest.py \
  --manifest /path/to/outputs/vimed_petct_atlas_256/manifest.csv \
  --output /path/to/outputs/vimed_petct_atlas_256/manifest_splits.csv
```

Generate QC montages:

```bash
python scripts/make_qc_montages.py \
  --manifest /path/to/outputs/vimed_petct_atlas_256/manifest_splits.csv \
  --out-dir /path/to/outputs/vimed_petct_atlas_256/qc \
  --num-samples 50
```

## Pillar Integration

The `pillar_adapters` directory contains code to use inside `pillar-finetune`:

- `vimed_petct_dataset.py`: reads the manifest and Torch tensors.
- `petct_windowing.py`: converts `(2, D, H, W)` raw PET/CT to `(12, D, H, W)`.
- `atlas_pet_channel.py`: expands an 11-channel CT patch embedding to 12 channels.

The intended model path is:

1. Load the original Atlas CT checkpoint.
2. Call `expand_first_ct_patch_embed_to_petct(model)`.
3. Use `make_petct_atlas_input(x_raw)` before the Atlas backbone.
4. Train a multi-label classifier with `BCEWithLogitsLoss`.

For the first run, train separate region models:

- `head_neck`: target depth 128.
- `chest`: target depth 256, best match for `Pillar0-ChestCT`.
- `abdomen_pelvis`: target depth 256 for the practical baseline cache.

## Finetune Start

The `pillar_finetune_patch` directory contains the files that need to exist in
your `pillar-finetune` checkout for ViMED PET/CT training:

- `pillar/datasets/vimed_petct.py`
- `pillar/datasets/__init__.py`
- `pillar/engines/classifier.py`
- `pillar/models/backbones/mmatlas.py`
- `pillar/metrics/multilabel.py`
- `pillar/metrics/__init__.py`
- `configs/vimed_chest_petct_atlas.yaml`

Cluster-side copy example:

```bash
export PLAN_ROOT=/home/thahoa/PET/ViMed/vimed_petct_atlas_finetune_plan
export PILLAR_ROOT=/home/thahoa/PET/Pillar-0/pillar-finetune

bash "${PLAN_ROOT}/scripts/apply_pillar_finetune_patch.sh" "${PLAN_ROOT}" "${PILLAR_ROOT}"
```

Set a shared Hugging Face cache on scratch so the Atlas checkpoint does not go
to home quota:

```bash
export HF_HOME=/scratch/thahoa/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/scratch/thahoa/.cache/huggingface/hub
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}"
```

Run a dry check before training. This will:

- load the config
- instantiate the dataset
- fetch one sample
- instantiate the model
- trigger the first Hugging Face download if the Atlas checkpoint is not cached

```bash
python "${PLAN_ROOT}/scripts/dry_check_vimed_chest_finetune.py" \
  --pillar-root "${PILLAR_ROOT}" \
  --config "${PILLAR_ROOT}/configs/vimed_chest_petct_atlas.yaml"
```

Then launch chest fine-tuning from the patched `pillar-finetune` checkout:

```bash
cd "${PILLAR_ROOT}"
python scripts/train.py configs/vimed_chest_petct_atlas.yaml
```

If you want label reweighting before the first run, compute suggested
`pos_weight` values on the training split:

```bash
python scripts/compute_label_stats.py \
  --manifest /scratch/thahoa/PET/ViMed_prep/manifest_splits.csv \
  --region chest \
  --split train
```

If the dry check or first train step fails only because the checkpoint cannot
be downloaded, the fallback is to pre-download it once into `HF_HOME` on a node
with internet access and reuse that shared cache for later jobs.

## Label Caution

The labels are weak report-derived labels. They are useful to start supervised
experiments, but they are not clinician-reviewed ground truth. Audit label
prevalence and sample reports before using results as clinical claims.
