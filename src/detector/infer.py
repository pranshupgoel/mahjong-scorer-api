"""Run the trained YOLO detector on a full scene photo and return tile crops."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from ultralytics import YOLO


class TileDetector:
    def __init__(self, weights_path: str | Path, conf: float = 0.25):
        self.model = YOLO(str(weights_path))
        self.conf = conf

    def detect_crops(self, image_path: str | Path) -> list[Image.Image]:
        """Return a list of cropped PIL images, one per detected tile,
        left-to-right (a reasonable default reading order for a laid-out hand;
        revisit once we see real photos -- hands aren't always in one row)."""
        results = self.model.predict(str(image_path), conf=self.conf, verbose=False)
        img = Image.open(image_path).convert("RGB")
        boxes = sorted(results[0].boxes.xyxy.tolist(), key=lambda box: box[0])
        return [img.crop(tuple(box)) for box in boxes]

    def detect_with_boxes(self, image_path: str | Path) -> list[tuple[tuple[float, float, float, float], Image.Image]]:
        """Like detect_crops, but also returns each tile's (x1,y1,x2,y2) box
        in original-image pixel coordinates -- the review screen overlays
        these on the photo so the user can see which crop is which."""
        results = self.model.predict(str(image_path), conf=self.conf, verbose=False)
        img = Image.open(image_path).convert("RGB")
        boxes = sorted(results[0].boxes.xyxy.tolist(), key=lambda box: box[0])
        return [(tuple(box), img.crop(tuple(box))) for box in boxes]
