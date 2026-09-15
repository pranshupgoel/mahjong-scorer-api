"""
Train the YOLO tile detector on Datasets/yolo dataset (Roboflow export,
already in YOLO format with its own data.yaml).

The detector is class-agnostic by design: it only localizes tile-shaped
boxes (single class "tile", data.yaml has nc: 1). Which tile each box is
gets decided later by the classifier (trained separately on
mahjong-dataset-master, which covers all 42 classes including flower/bonus
tiles). The original per-face labels were collapsed to one class with
scripts/make_detector_class_agnostic.py -- see that script's docstring if
the dataset is ever re-exported from Roboflow and needs re-collapsing.

Usage:
    python -m src.detector.train --data "Datasets/yolo dataset/data.yaml" \
        --model yolov8n.pt --epochs 100 --imgsz 640
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Datasets/yolo dataset/data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="starting checkpoint; 'n' (nano) fits on-device best")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--project", default="runs/detect")
    ap.add_argument("--name", default="tile-detector")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
