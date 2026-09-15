# Hugging Face Spaces (Docker SDK) image for the Mahjong Scorer API.
# Spaces expects the app to listen on port 7860.
FROM python:3.10-slim

WORKDIR /app

# opencv/ultralytics need these system libs even in headless/CPU mode
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY api/ api/
COPY checkpoints/ checkpoints/

ENV DETECTOR_WEIGHTS=/app/checkpoints/detector_best.pt
ENV CLASSIFIER_CKPT=/app/checkpoints/classifier.pt

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
