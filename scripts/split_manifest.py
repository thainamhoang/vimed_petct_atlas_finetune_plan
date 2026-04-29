#!/usr/bin/env python3
"""Assign deterministic study-level train/val/test splits to a manifest."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total = args.train_frac + args.val_frac + args.test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test fractions must sum to 1.0")

    rows = list(csv.DictReader(args.manifest.open()))
    studies = sorted({r["study_id"] for r in rows})
    rng = random.Random(args.seed)
    rng.shuffle(studies)

    n = len(studies)
    n_train = int(round(n * args.train_frac))
    n_val = int(round(n * args.val_frac))
    train = set(studies[:n_train])
    val = set(studies[n_train : n_train + n_val])
    test = set(studies[n_train + n_val :])

    split_by_study = {s: "train" for s in train}
    split_by_study.update({s: "val" for s in val})
    split_by_study.update({s: "test" for s in test})

    for row in rows:
        row["split"] = split_by_study[row["study_id"]]

    output = args.output or args.manifest.with_name(args.manifest.stem + "_splits.csv")
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"studies: train={len(train)} val={len(val)} test={len(test)}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

