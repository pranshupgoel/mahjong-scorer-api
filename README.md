# Mahjong Scorer API

FastAPI backend (YOLO detector + tile classifier + Taiwanese Mahjong rule
engine) for the Mahjong Scorer app. Deployed on Render as a Docker web
service.

Model weights (`checkpoints/detector_best.pt`, `checkpoints/classifier.pt`)
are not committed here yet (large binary files) -- see the Mahjong project
docs (`mahjong-app-and-backend.md`) for how they're supplied to the deployed
service.

See `api/main.py` for endpoints (`/health`, `/detect`, `/score`).
