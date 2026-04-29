#!/usr/bin/env python3
"""Dry-check the ViMED chest PET/CT finetune path without launching training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pillar-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--sample-index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.pillar_root))

    from pillar import datasets, models  # noqa: WPS433
    from pillar.utils.parsing import load_config  # noqa: WPS433

    cfg = load_config(str(args.config))
    cfg.main.multi_gpu = False
    cfg.main.global_rank = 0
    cfg.main.world_size = 1
    cfg.main.local_rank = 0
    cfg.main.disable_wandb = True

    dataset_type = cfg.dataset.type
    dataset_kwargs = dict(cfg.dataset.shared_dataset_kwargs)
    dataset = datasets.__dict__[dataset_type](
        args=cfg,
        augmentations=[],
        split_group=args.split,
        **dataset_kwargs,
    )

    print(f"dataset_type={dataset_type}")
    print(f"dataset_len={len(dataset)}")
    sample = dataset[args.sample_index]
    print(f"x_shape_before_windowing={tuple(sample['x'].shape)}")
    print(f"y_shape={tuple(sample['y'].shape)}")
    print(f"region={sample['region']}")
    print(f"accession={sample['accession']}")

    model = models.__dict__[cfg.model.type](args=cfg, **dict(cfg.model.kwargs))
    print(f"model_type={cfg.model.type}")

    first_conv = None
    first_conv_name = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d):
            first_conv = module
            first_conv_name = name
            break

    if first_conv is None:
        raise RuntimeError("No Conv3d found in model")

    print(f"first_conv_name={first_conv_name}")
    print(f"first_conv_in_channels={first_conv.in_channels}")
    print("dry_check=ok")


if __name__ == "__main__":
    main()
