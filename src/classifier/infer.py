"""Load a trained classifier checkpoint and label a single cropped tile image."""
from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.classifier.train import build_model, build_transforms


class TileClassifier:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.classes = ckpt["classes"]
        self.model = build_model(len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.transform = build_transforms(train=False)

    @torch.no_grad()
    def predict(self, crop: Image.Image) -> tuple[str, float]:
        """Return (canonical_class_name, confidence) for a single cropped tile."""
        x = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top_idx = int(probs.argmax())
        return self.classes[top_idx], float(probs[top_idx])

    @torch.no_grad()
    def predict_topk(self, crop: Image.Image, k: int = 3) -> list[tuple[str, float]]:
        """Top-k predictions -- useful for the review screen where users can
        correct a mis-identified tile from a short list rather than typing."""
        x = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(x), dim=1)[0]
        top = torch.topk(probs, k)
        return [(self.classes[i], float(p)) for p, i in zip(top.values, top.indices)]
