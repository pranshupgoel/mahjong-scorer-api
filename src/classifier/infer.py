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

    @torch.no_grad()
    def predict_topk_batch(self, crops: list[Image.Image], k: int = 3) -> list[list[tuple[str, float]]]:
        """Same as predict_topk but for many crops in a single forward pass.
        On CPU (especially the very limited CPU of a free-tier host) this is
        dramatically faster than calling predict_topk in a loop -- one hand
        photo can have 15+ tiles, and per-call Python/PyTorch overhead adds
        up fast when each tile is its own separate forward pass."""
        if not crops:
            return []
        batch = torch.stack([self.transform(c.convert("RGB")) for c in crops]).to(self.device)
        probs = torch.softmax(self.model(batch), dim=1)
        top = torch.topk(probs, k, dim=1)
        results: list[list[tuple[str, float]]] = []
        for row_values, row_indices in zip(top.values, top.indices):
            results.append([(self.classes[i], float(p)) for p, i in zip(row_values, row_indices)])
        return results
