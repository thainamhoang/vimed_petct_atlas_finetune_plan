#!/usr/bin/env python3
"""Smoke-test the chest-only dual-stream ViMED data path.

This script works with the current x_raw preprocessing cache. It derives CT and
PET windows on the fly and prints shapes/ranges/report text for quick sanity
checks before model integration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PLAN_ROOT = Path(__file__).resolve().parents[1]
if str(PLAN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLAN_ROOT))

from pillar_adapters.vimed_chest_report_dataset import ViMedChestReportDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--include-raw', action='store_true')
    return p.parse_args()


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    tensor = tensor.float()
    print(f'{name}_shape={tuple(tensor.shape)}')
    print(f'{name}_min={float(tensor.min()):.6f}')
    print(f'{name}_max={float(tensor.max()):.6f}')


def main() -> None:
    args = parse_args()
    ds = ViMedChestReportDataset(args.manifest, split=args.split, include_raw=args.include_raw)
    if len(ds) == 0:
        raise SystemExit(f'No rows found for split={args.split!r} in {args.manifest}')
    sample = ds[args.index]

    print(f'dataset_len={len(ds)}')
    print(f'study_id={sample["study_id"]}')
    print(f'region={sample["region"]}')
    summarize_tensor('ct_windows', sample['ct_windows'])
    summarize_tensor('pet_windows', sample['pet_windows'])
    if args.include_raw:
        summarize_tensor('x_raw', sample['x_raw'])
    if 'labels' in sample:
        print(f'labels_shape={tuple(sample["labels"].shape)}')
        print(f'labels={sample["labels"].tolist()}')
    report = sample['report_text'] or ''
    report_preview = report.replace('\n', ' ')[:300]
    print(f'report_chars={len(report)}')
    print(f'report_preview={report_preview}')


if __name__ == '__main__':
    main()
