# Hugging Face Spaces (Docker SDK) image for the Mahjong Scorer API.
# Spaces expects the app to listen on port 7860 (Render also honors $PORT --
# see the CMD below).
FROM python:3.10-slim

WORKDIR /app

# opencv/ultralytics need these system libs even in headless/CPU mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch/torchvision FIRST, from PyTorch's own CPU wheel index --
# the default PyPI wheels pull ~2GB of CUDA packages we don't need and
# won't fit on a free-tier host. Installing these first means the later
# `pip install -r requirements-api.txt` (which pulls in ultralytics, whose
# own torch dependency would otherwise resolve to the CUDA build) finds
# torch/torchvision already satisfied and leaves them alone.
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY checkpoints/ checkpoints/

ENV DETECTOR_WEIGHTS=/app/checkpoints/detector_best.pt
ENV CLASSIFIER_CKPT=/app/checkpoints/classifier.pt

EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
