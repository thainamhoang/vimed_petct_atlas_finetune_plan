"""Chest-only ViMED dataset for dual-stream PET/CT report-generation experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from .petct_windowing import make_dual_stream_window_inputs


class ViMedChestReportDataset(Dataset):
    """Load existing ViMED x_raw tensors and derive CT/PET windows on the fly.

    This reuses the current preprocessing cache, so you do not need a full
    reprocessing pass just to get started on the dual-stream chest pipeline.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        region: str = "chest",
        include_raw: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = list(csv.DictReader(self.manifest_path.open()))
        if split is not None:
            self.rows = [r for r in self.rows if r.get("split") == split]
        self.rows = [r for r in self.rows if r.get("region") == region]
        self.include_raw = include_raw

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        item = torch.load(row["tensor_path"], map_location="cpu", weights_only=False)
        x_raw = item["x_raw"].float()
        dual = make_dual_stream_window_inputs(x_raw)
        metadata = item.get("metadata", {})
        report_text = metadata.get("report_text", row.get("report_text", ""))
        out = {
            "ct_windows": dual["ct_windows"],
            "pet_windows": dual["pet_windows"],
            "report_text": report_text,
            "study_id": row["study_id"],
            "accession": row["study_id"],
            "region": row["region"],
            "metadata": metadata,
        }
        if "labels" in item:
            out["labels"] = item["labels"].float()
            out["label_names"] = list(item.get("label_names", []))
        if self.include_raw:
            out["x_raw"] = x_raw
        return out
