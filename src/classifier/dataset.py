"""Torch Dataset for the individual-tile classifier, reading the CSV splits
produced by scripts/prepare_classifier_splits.py."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from src.common.labels import CANONICAL_TO_IDX


class TileDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = CANONICAL_TO_IDX[row["canonical_label"]]
        return img, label
