#!/usr/bin/env python3
"""Compute region-specific label prevalence and suggested BCE pos_weight values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--region", type=str, default="chest")
    p.add_argument("--split", type=str, default="train")
    p.add_argument(
        "--label-columns",
        nargs="+",
        default=[
            "abnormal_fdg_uptake",
            "lymph_node",
            "mass_or_nodule",
            "metastasis_or_recurrence",
            "inflammation_or_hyperplasia",
            "suv_mentioned",
            "effusion_or_fluid",
            "no_abnormal_fdg",
        ],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest)
    df = df[(df["region"] == args.region) & (df["split"] == args.split)].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows matched the requested region/split")

    pos_weights = []
    summary = []
    for col in args.label_columns:
        positives = int(df[col].sum())
        total = int(len(df))
        negatives = total - positives
        prevalence = positives / total
        pos_weight = negatives / positives if positives > 0 else 1.0
        pos_weights.append(pos_weight)
        summary.append(
            {
                "label": col,
                "positives": positives,
                "negatives": negatives,
                "prevalence": prevalence,
                "suggested_pos_weight": pos_weight,
            }
        )

    print(json.dumps({"region": args.region, "split": args.split, "summary": summary, "pos_weight": pos_weights}, indent=2))


if __name__ == "__main__":
    main()

