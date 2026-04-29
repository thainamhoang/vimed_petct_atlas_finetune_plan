#!/usr/bin/env python3
"""Create quick CT/PET montage PNGs from a preprocessed ViMED manifest."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--region", choices=["head_neck", "chest", "abdomen_pelvis"], default=None)
    return p.parse_args()


def to_u8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = np.clip((x.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def make_row(volume: np.ndarray, label: str, lo: float, hi: float, tile: int = 160) -> Image.Image:
    depth = volume.shape[0]
    indices = np.linspace(0, depth - 1, 5).round().astype(int)
    tiles = []
    for idx in indices:
        img = Image.fromarray(to_u8(volume[idx], lo, hi), mode="L").resize((tile, tile))
        tiles.append(img.convert("RGB"))
    row = Image.new("RGB", (tile * len(tiles), tile + 24), "white")
    draw = ImageDraw.Draw(row)
    draw.text((4, 4), label, fill=(0, 0, 0))
    for i, img in enumerate(tiles):
        row.paste(img, (i * tile, 24))
    return row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(args.manifest.open()))
    if args.region is not None:
        rows = [r for r in rows if r["region"] == args.region]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    rows = rows[: args.num_samples]

    for i, row in enumerate(rows):
        item = torch.load(row["tensor_path"], map_location="cpu")
        x = item["x_raw"].float().numpy()
        ct = x[0]
        pet = x[1]
        ct_row = make_row(ct, f"{row['study_id']} {row['region']} CT", lo=-1024.0, hi=600.0)
        pet_row = make_row(pet, "PET normalized", lo=0.0, hi=1.0)
        canvas = Image.new("RGB", (ct_row.width, ct_row.height + pet_row.height), "white")
        canvas.paste(ct_row, (0, 0))
        canvas.paste(pet_row, (0, ct_row.height))
        out = args.out_dir / f"{i:04d}_{row['study_id']}_{row['region']}.png"
        canvas.save(out)

    print(f"wrote {len(rows)} QC montages to {args.out_dir}")


if __name__ == "__main__":
    main()

