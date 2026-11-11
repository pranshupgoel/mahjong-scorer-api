# Mahjong Scorer API

Take a photo of a winning Taiwanese Mahjong hand and get it scored. The backend finds every tile in the image, identifies each one, and runs the hand through a rule engine that works out which scoring patterns apply and how many points the hand is worth.

**Stack:** Python, PyTorch, YOLOv8 (Ultralytics), FastAPI, Docker

## How it works

```
 photo ──► tile detector ──► tile classifier ──► user review ──► rule engine ──► score
           (YOLOv8n)         (MobileNetV3-Small,   (app confirms     (hand decomposition
                              42 tile classes)      tiles + answers   + Taiwanese rules)
                                                    a short quiz)
```

1. **Detector.** A YOLOv8n model trained to localise tiles. It is deliberately class-agnostic (a single "tile" class), so it only has to find tiles, not tell them apart.
2. **Classifier.** A fine-tuned MobileNetV3-Small assigns each cropped tile one of 42 classes: the three number suits, the honour tiles and the eight flower and season bonus tiles. The labelled data is small (about 600 images), so training uses class weighting, strong augmentation without horizontal flips (which would produce mirrored tile faces that do not exist) and early stopping.
3. **Review step.** The app shows detections with top-k alternatives so the player can correct mistakes, then asks a short questionnaire for things a photo cannot show: seat wind, round wind, self-draw or discard, and so on.
4. **Rule engine.** Decomposes the hand into every valid arrangement of melds and pair, detects special hands, and scores the best-scoring reading using the Taiwanese ruleset. Rulesets are pluggable.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Service and model-loading status |
| `POST /detect` | Upload a photo and receive per-tile detections: label, confidence, top-k alternatives and bounding box |
| `POST /score` | Send the reviewed tiles plus the questionnaire answers and receive the scored hand |

Interactive docs are available at `/docs` when the server is running.

## Project layout

```
api/                  FastAPI app (endpoints, request and response models)
src/detector/         YOLO training and inference
src/classifier/       Dataset, training, evaluation and inference for the tile classifier
src/rule_engine/      Hand decomposition, special hands, scoring engine and tests
  rulesets/           Base interface and the Taiwanese ruleset
src/pipeline.py       Detector to classifier pipeline
src/pipeline_scoring.py  Connects reviewed tiles to the rule engine
checkpoints/          Trained model weights
Dockerfile            CPU-only image, deployable to Hugging Face Spaces or Render
```

## Run it

```bash
pip install -r requirements-api.txt        # plus torch and torchvision (CPU build is fine)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or with Docker:

```bash
docker build -t mahjong-scorer-api .
docker run -p 7860:7860 mahjong-scorer-api
```

Run the rule-engine tests:

```bash
python -m pytest src/rule_engine/test_engine.py
```

## Known limitations

Scoring a hand is a rules-interpretation problem as much as an engineering one, so the gaps are listed openly:

- The "four in 2, 3 or 4 ways" pattern and the extra Nico Nico gong bonuses are not implemented yet.
- Multi-round game state (penalties, dealer payments and so on) is out of scope; the engine scores one completed winning hand.
- The API has no authentication and allows any origin, which is fine for a demo but should be tightened before a real release.

## Roadmap

- Quantise and export the classifier for on-device inference (TFLite or Core ML) once accuracy is validated.
- Close the scoring gaps above using worked example hands.
