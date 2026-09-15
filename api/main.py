"""
FastAPI backend wrapping the mahjong pipeline (detector -> classifier) and
the rule engine (src/pipeline_scoring.py), for the phone app to call.

Two endpoints:
    POST /detect  -- multipart photo upload -> per-tile detections for the
                      review screen (label, confidence, top-k alternatives,
                      bounding box).
    POST /score   -- reviewed tiles + WinContext questionnaire (JSON) ->
                      scored hand (see src/pipeline_scoring.py for the exact
                      request/response shape).

Model weights are loaded once at startup (not per-request) from
DETECTOR_WEIGHTS / CLASSIFIER_CKPT env vars, defaulting to
checkpoints/detector_best.pt and checkpoints/classifier.pt relative to the
repo root -- see claude/mahjong-pipeline-status.md for where those come
from (trained on sirlserver03, scp'd down manually).

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Deployed on Hugging Face Spaces (Docker SDK), see Dockerfile in repo root.
"""
from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mahjong_api")

REPO_ROOT = Path(__file__).resolve().parent.parent
DETECTOR_WEIGHTS = os.environ.get("DETECTOR_WEIGHTS", str(REPO_ROOT / "checkpoints" / "detector_best.pt"))
CLASSIFIER_CKPT = os.environ.get("CLASSIFIER_CKPT", str(REPO_ROOT / "checkpoints" / "classifier.pt"))

app = FastAPI(title="Mahjong Scorer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo/testing phase -- the app has no auth yet, tighten before real release
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector = None
_classifier = None
_models_error: Optional[str] = None


@app.on_event("startup")
def load_models() -> None:
    """Load once at process startup so /detect requests don't re-load
    weights every call. If weights are missing, don't crash the whole
    server -- /health and /score (which doesn't need the vision models)
    should still work; /detect will return a clear 503 instead."""
    global _detector, _classifier, _models_error
    try:
        from src.detector.infer import TileDetector
        from src.classifier.infer import TileClassifier

        if not Path(DETECTOR_WEIGHTS).exists():
            raise FileNotFoundError(f"detector weights not found at {DETECTOR_WEIGHTS}")
        if not Path(CLASSIFIER_CKPT).exists():
            raise FileNotFoundError(f"classifier checkpoint not found at {CLASSIFIER_CKPT}")

        _detector = TileDetector(DETECTOR_WEIGHTS)
        _classifier = TileClassifier(CLASSIFIER_CKPT)
        logger.info("Loaded detector (%s) and classifier (%s)", DETECTOR_WEIGHTS, CLASSIFIER_CKPT)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: log and degrade, don't crash startup
        _models_error = str(exc)
        logger.warning("Vision models not loaded: %s. /detect will 503 until weights are present.", exc)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": _detector is not None and _classifier is not None,
        "models_error": _models_error,
    }


class TileDetection(BaseModel):
    id: str
    label: str
    confidence: float
    alternatives: list[tuple[str, float]]
    bbox: tuple[float, float, float, float]


class DetectResponse(BaseModel):
    tiles: list[TileDetection]
    image_width: int
    image_height: int


@app.post("/detect", response_model=DetectResponse)
async def detect(image: UploadFile = File(...)) -> DetectResponse:
    if _detector is None or _classifier is None:
        raise HTTPException(
            status_code=503,
            detail=f"Vision models not loaded on the server yet ({_models_error}). "
            "Detector/classifier weights need to be deployed alongside the API.",
        )

    suffix = Path(image.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"{uuid.uuid4().hex}{suffix}"
        tmp_path.write_bytes(await image.read())

        from PIL import Image as PILImage

        with PILImage.open(tmp_path) as im:
            width, height = im.size

        detections = _detector.detect_with_boxes(str(tmp_path))
        crops = [crop for _bbox, crop in detections]
        # Classify all detected tiles in a single batched forward pass rather
        # than one model call per tile -- much faster on CPU, which matters a
        # lot on a resource-constrained host.
        all_alternatives = _classifier.predict_topk_batch(crops, k=3)
        tiles: list[TileDetection] = []
        for (bbox, _crop), alternatives in zip(detections, all_alternatives):
            best_label, best_conf = alternatives[0]
            tiles.append(
                TileDetection(
                    id=uuid.uuid4().hex[:8],
                    label=best_label,
                    confidence=best_conf,
                    alternatives=alternatives,
                    bbox=bbox,
                )
            )

    return DetectResponse(tiles=tiles, image_width=width, image_height=height)


class MeldPayload(BaseModel):
    type: str  # "sheung" | "pong" | "gong"
    tiles: list[str]
    concealed: bool = False
    gong_kind: str = "none"  # "none" | "open" | "concealed"


class WinContextPayload(BaseModel):
    seat_wind: str
    round_wind: str
    self_draw: bool
    win_from_flower_wall: bool = False
    win_from_robbing_gong: bool = False
    gong_gong_win: bool = False
    tiles_on_table: Optional[int] = None
    is_seabed: bool = False
    is_earthly: bool = False
    is_heavenly: bool = False
    wait_type: str = "none"
    closed_hand_declared: bool = False
    triple_fan_dice: Optional[int] = None
    dealer_win_streak: int = 0


class ScoreRequest(BaseModel):
    melds: list[MeldPayload] = Field(default_factory=list)
    concealed_tiles: list[str] = Field(default_factory=list)
    flowers: list[str] = Field(default_factory=list)
    winning_tile: Optional[str] = None
    context: WinContextPayload
    ruleset: str = "taiwanese"


@app.post("/score")
def score(req: ScoreRequest) -> dict[str, Any]:
    from src.pipeline_scoring import score_reviewed_hand

    try:
        return score_reviewed_hand(req.model_dump())
    except ValueError as exc:
        # The tiles the user confirmed don't form a valid winning hand --
        # that's a client-input problem (422), not a server bug (500).
        raise HTTPException(status_code=422, detail=str(exc)) from exc
