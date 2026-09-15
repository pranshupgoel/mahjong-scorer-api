"""
End-to-end glue: scene photo -> detector -> per-tile classifier -> rule
engine. This is the shape described for the app: detect tiles, classify
each one, let the user confirm/correct on a review screen, then score.

Usage (once both models are trained):
    from src.pipeline import identify_hand
    result = identify_hand("photo.jpg", detector_weights="checkpoints/detector_best.pt",
                            classifier_ckpt="checkpoints/classifier.pt")
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.classifier.infer import TileClassifier
from src.detector.infer import TileDetector


@dataclass
class TileGuess:
    label: str
    confidence: float
    alternatives: list[tuple[str, float]]  # top-k, for the review screen's correction UI
    bbox: tuple[float, float, float, float] | None = None  # (x1,y1,x2,y2) in source-photo pixels


def identify_hand(
    image_path: str,
    detector_weights: str,
    classifier_ckpt: str,
    topk: int = 3,
) -> list[TileGuess]:
    detector = TileDetector(detector_weights)
    classifier = TileClassifier(classifier_ckpt)

    detections = detector.detect_with_boxes(image_path)
    guesses = []
    for bbox, crop in detections:
        alternatives = classifier.predict_topk(crop, k=topk)
        best_label, best_conf = alternatives[0]
        guesses.append(TileGuess(label=best_label, confidence=best_conf, alternatives=alternatives, bbox=bbox))
    return guesses
