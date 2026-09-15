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
# won't fit on a free-tier host. --no-deps + a single-purpose index here is
# deliberate: pytorch.org's /whl/cpu index only mirrors torch/torchvision
# themselves, not the full PyPI catalog, so asking it to also resolve their
# transitive deps (typing-extensions, sympy, etc.) fails to build some of
# them from source. Those deps are plain, GPU-agnostic Python packages, so
# they're installed normally from PyPI via requirements-api.txt instead
# (torch-runtime-deps.txt below).
RUN pip install --no-cache-dir --no-deps torch==2.4.1 torchvision==0.19.1 \
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
