"""Torch dataset for the ViMED PET/CT Atlas preprocessing cache."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset


class ViMedPETCTDataset(Dataset):
    """Read preprocessed ViMED PET/CT tensors from a manifest CSV.

    Each tensor file is expected to contain:
        x_raw: Tensor (2, D, H, W)
        labels: Tensor (num_labels,)
        label_names: list[str]
        metadata: dict
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        region: Optional[str] = None,
        label_names: Optional[Iterable[str]] = None,
        atlas_ready: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = list(csv.DictReader(self.manifest_path.open()))
        if split is not None:
            self.rows = [r for r in self.rows if r.get("split") == split]
        if region is not None:
            self.rows = [r for r in self.rows if r.get("region") == region]
        self.label_names = list(label_names) if label_names is not None else None
        self.atlas_ready = atlas_ready

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        item = torch.load(row["tensor_path"], map_location="cpu")
        x = item["x_raw"].float()
        labels = item["labels"].float()
        names = list(item["label_names"])

        if self.label_names is not None:
            index_by_name = {name: i for i, name in enumerate(names)}
            labels = torch.stack([labels[index_by_name[name]] for name in self.label_names])
            names = self.label_names

        if self.atlas_ready:
            try:
                from .petct_windowing import make_petct_atlas_input
            except ImportError:
                from petct_windowing import make_petct_atlas_input

            x = make_petct_atlas_input(x)

        return {
            "x": x,
            "y": labels,
            "label_names": names,
            "accession": row["study_id"],
            "study_id": row["study_id"],
            "region": row["region"],
            "metadata": item.get("metadata", {}),
        }
