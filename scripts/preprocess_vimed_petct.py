#!/usr/bin/env python3
"""Preprocess ViMED PET/CT NPZ studies into Atlas-friendly Torch tensors.

This script is cluster-friendly: run one shard per job with --num-shards and
--shard-index, then concatenate or keep the generated manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


REGION_KEYS = ("head_neck", "chest", "abdomen_pelvis")
REGION_DEPTHS = {"head_neck": 64, "chest": 64, "abdomen_pelvis": 192}
REGION_REPORT_KEYS = {
    "head_neck": "head_neck",
    "chest": "chest",
    "abdomen_pelvis": "abdomen_pelvis",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True, help="ViMED data root containing metadata.csv")
    p.add_argument("--metadata", default="metadata_with_ct_pet_minmax.csv")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--label-rules", type=Path, required=True)
    p.add_argument("--regions", nargs="+", default=list(REGION_KEYS), choices=REGION_KEYS)
    p.add_argument("--target-size", type=int, default=256)
    p.add_argument("--head-neck-depth", type=int, default=64)
    p.add_argument("--chest-depth", type=int, default=64)
    p.add_argument("--abdomen-pelvis-depth", type=int, default=192)
    p.add_argument("--overlap-slices", type=int, default=20)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max-studies", type=int, default=None, help="Debug limit after sharding")
    return p.parse_args()


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"


def read_rows(data_root: Path, metadata: str) -> list[dict[str, str]]:
    with (data_root / metadata).open(newline="") as f:
        return list(csv.DictReader(f))


def load_label_rules(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    rules = json.loads(path.read_text())
    label_rules = rules["labels"]
    return [r["name"] for r in label_rules], label_rules


def regex_any(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def make_labels(text: str, rules: list[dict[str, Any]]) -> list[int]:
    labels = []
    for rule in rules:
        positive = regex_any(rule.get("positive", []), text)
        negative = regex_any(rule.get("negative", []), text)
        mode = rule.get("mode", "any_positive")

        if mode == "any_positive":
            labels.append(int(positive))
        elif mode == "positive_unless_only_negative":
            if not positive:
                labels.append(0)
            elif negative and not regex_any(
                ["\\bsuvmax\\b", "\\bfdg[- ]avid\\b", "\\babnormal\\b.{0,40}\\bfdg\\b", "\\bfocal\\b.{0,40}\\bfdg\\b"],
                text,
            ):
                labels.append(0)
            else:
                labels.append(1)
        else:
            raise ValueError(f"Unknown label rule mode: {mode}")
    return labels


def load_npz_data(path: Path) -> np.ndarray:
    with np.load(path, mmap_mode="r") as z:
        if "data" not in z.files:
            raise KeyError(f"{path} does not contain key 'data'; found {z.files}")
        return np.asarray(z["data"])


def resize_inplane(volume: np.ndarray, target_size: int, mode: str) -> np.ndarray:
    if volume.shape[-2:] == (target_size, target_size):
        return np.ascontiguousarray(volume)
    tensor = torch.from_numpy(np.ascontiguousarray(volume)).float().unsqueeze(1)
    # Area mode is preferred for the exact 512->256 CT downsampling because it
    # behaves like anti-aliased block averaging during pure downsampling.
    if mode == "area":
        resized = F.interpolate(tensor, size=(target_size, target_size), mode=mode)
    else:
        resized = F.interpolate(tensor, size=(target_size, target_size), mode=mode, align_corners=False)
    return resized.squeeze(1).cpu().numpy()


def normalize_ct(ct: np.ndarray) -> np.ndarray:
    return np.clip(ct.astype(np.float32), -1024.0, 3071.0)


def normalize_pet(pet: np.ndarray) -> np.ndarray:
    pet = pet.astype(np.float32)
    positive = pet[pet > 0]
    if positive.size == 0:
        return np.zeros_like(pet, dtype=np.float32)
    p995 = float(np.percentile(positive, 99.5))
    if not math.isfinite(p995) or p995 <= 0:
        p995 = float(positive.max()) if positive.max() > 0 else 1.0
    clipped = np.clip(pet, 0.0, p995)
    return np.clip(np.log1p(clipped) / np.log1p(p995), 0.0, 1.0).astype(np.float32)


def region_bounds(depth: int, region: str, overlap: int) -> tuple[int, int]:
    hn_end = int(round(depth * 0.20))
    chest_end = int(round(depth * 0.55))
    before = overlap // 2
    after = overlap - before
    if region == "head_neck":
        start, end = 0, hn_end + after
    elif region == "chest":
        start, end = hn_end - before, chest_end + after
    elif region == "abdomen_pelvis":
        start, end = chest_end - before, depth
    else:
        raise ValueError(f"Unknown region: {region}")
    return max(0, start), min(depth, max(start + 1, end))


def fit_depth(volume: np.ndarray, target_depth: int, pad_value: float) -> np.ndarray:
    depth = volume.shape[0]
    if depth >= target_depth:
        start = (depth - target_depth) // 2
        return np.ascontiguousarray(volume[start : start + target_depth])

    before = (target_depth - depth) // 2
    after = target_depth - depth - before
    return np.pad(
        volume,
        ((before, after), (0, 0), (0, 0)),
        mode="constant",
        constant_values=pad_value,
    )


def load_report_text(report_path: Path, region: str) -> str:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    desc = report.get("image_description", {})
    return str(desc.get(REGION_REPORT_KEYS[region], ""))


def process_study(
    row_index: int,
    row: dict[str, str],
    args: argparse.Namespace,
    label_names: list[str],
    label_rules: list[dict[str, Any]],
    depths: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data_root = args.data_root
    study_name = safe_name(row.get("name", f"row_{row_index}"))
    study_id = f"{row_index:06d}_{study_name}_{safe_name(row.get('year', 'unknown'))}"

    ct_path = data_root / row["ct_path"]
    pet_path = data_root / row["pet_path"]
    report_path = data_root / row["report_en_path"]
    missing = [str(p) for p in (ct_path, pet_path, report_path) if not p.exists()]
    if missing:
        return [], [{"study_id": study_id, "reason": "missing_file", "details": "|".join(missing)}]

    try:
        ct = load_npz_data(ct_path)
        pet = load_npz_data(pet_path)
        if ct.ndim != 3 or pet.ndim != 3:
            raise ValueError(f"Expected 3D arrays, got CT {ct.shape}, PET {pet.shape}")

        depth = min(ct.shape[0], pet.shape[0])
        ct = ct[:depth]
        pet = pet[:depth]
        ct = normalize_ct(resize_inplane(ct, args.target_size, mode="area"))
        pet = normalize_pet(resize_inplane(pet, args.target_size, mode="bilinear"))

        manifest_rows = []
        skipped = []
        for region in args.regions:
            z_start, z_end = region_bounds(depth, region, args.overlap_slices)
            target_depth = depths[region]
            ct_region = fit_depth(ct[z_start:z_end], target_depth, pad_value=-1024.0)
            pet_region = fit_depth(pet[z_start:z_end], target_depth, pad_value=0.0)
            x_raw = torch.stack(
                [torch.from_numpy(ct_region), torch.from_numpy(pet_region)],
                dim=0,
            ).to(torch.float16)

            report_text = load_report_text(report_path, region)
            labels = make_labels(report_text, label_rules)
            tensor_dir = args.out_dir / "tensors" / region
            tensor_dir.mkdir(parents=True, exist_ok=True)
            tensor_path = tensor_dir / f"{study_id}_{region}.pt"

            if args.overwrite or not tensor_path.exists():
                torch.save(
                    {
                        "x_raw": x_raw,
                        "labels": torch.tensor(labels, dtype=torch.float32),
                        "label_names": label_names,
                        "metadata": {
                            "study_id": study_id,
                            "name": row.get("name", ""),
                            "region": region,
                            "z_start": z_start,
                            "z_end": z_end,
                            "raw_depth": depth,
                            "ct_path": str(ct_path),
                            "pet_path": str(pet_path),
                            "report_en_path": str(report_path),
                            "n_slices": row.get("n_slices", ""),
                            "height": row.get("height", ""),
                            "weight": row.get("weight", ""),
                            "year": row.get("year", ""),
                            "report_text": report_text,
                        },
                    },
                    tensor_path,
                )

            out_row = {
                "study_id": study_id,
                "split": "",
                "region": region,
                "tensor_path": str(tensor_path),
                "ct_path": str(ct_path),
                "pet_path": str(pet_path),
                "report_en_path": str(report_path),
                "z_start": z_start,
                "z_end": z_end,
                "raw_depth": depth,
                "height": row.get("height", ""),
                "weight": row.get("weight", ""),
                "year": row.get("year", ""),
                "report_text": report_text,
                "n_slices": row.get("n_slices", ""),
            }
            out_row.update({name: labels[i] for i, name in enumerate(label_names)})
            manifest_rows.append(out_row)
        return manifest_rows, skipped
    except Exception as exc:
        return [], [
            {
                "study_id": study_id,
                "reason": type(exc).__name__,
                "details": str(exc),
                "traceback": traceback.format_exc(limit=3),
            }
        ]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    label_names, label_rules = load_label_rules(args.label_rules)
    depths = {
        "head_neck": args.head_neck_depth,
        "chest": args.chest_depth,
        "abdomen_pelvis": args.abdomen_pelvis_depth,
    }

    rows = read_rows(args.data_root, args.metadata)
    shard_rows = [(i, r) for i, r in enumerate(rows) if i % args.num_shards == args.shard_index]
    if args.max_studies is not None:
        shard_rows = shard_rows[: args.max_studies]

    manifest_path = args.out_dir / f"manifest_shard{args.shard_index:04d}_of_{args.num_shards:04d}.csv"
    skipped_path = args.out_dir / f"skipped_shard{args.shard_index:04d}_of_{args.num_shards:04d}.csv"

    manifest_fieldnames = [
        "study_id",
        "split",
        "region",
        "tensor_path",
        "ct_path",
        "pet_path",
        "report_en_path",
        "z_start",
        "z_end",
        "raw_depth",
        "height",
        "weight",
        "year",
        "report_text",
        "n_slices",
        *label_names,
    ]
    skipped_fieldnames = ["study_id", "reason", "details", "traceback"]

    with manifest_path.open("w", newline="") as mf, skipped_path.open("w", newline="") as sf:
        manifest_writer = csv.DictWriter(mf, fieldnames=manifest_fieldnames)
        skipped_writer = csv.DictWriter(sf, fieldnames=skipped_fieldnames)
        manifest_writer.writeheader()
        skipped_writer.writeheader()

        for n, (row_index, row) in enumerate(shard_rows, start=1):
            manifest_rows, skipped_rows = process_study(row_index, row, args, label_names, label_rules, depths)
            for r in manifest_rows:
                manifest_writer.writerow(r)
            for r in skipped_rows:
                skipped_writer.writerow({k: r.get(k, "") for k in skipped_fieldnames})
            if n % 25 == 0:
                print(f"processed {n}/{len(shard_rows)} studies for shard {args.shard_index}", flush=True)

    print(f"wrote {manifest_path}")
    print(f"wrote {skipped_path}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
