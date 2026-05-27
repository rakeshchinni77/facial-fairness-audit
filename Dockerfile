FROM python:3.10-slim

# Keep the container lean and deterministic for CPU-only execution.
WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OpenCV on slim images needs these shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade packaging tools before installing project dependencies.
RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full repository so the container matches the local project layout.
COPY . ./

# Create common output directories if they are missing from the mounted source tree.
RUN mkdir -p data/raw data/processed data/interim data/audit artifacts/checkpoints artifacts/embeddings artifacts/plots results submission

CMD ["python", "main.py"]