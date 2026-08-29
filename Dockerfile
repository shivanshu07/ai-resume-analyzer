# syntax=docker/dockerfile:1

FROM python:3.12-slim

WORKDIR /app

# System deps: some packages (e.g. PyMuPDF/sentence-transformers'
# own dependencies) occasionally need a compiler to build a wheel
# on slim images. Installed then cleaned up in the same layer to
# keep image size down.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch's CPU-only build FIRST, explicitly, before
# sentence-transformers pulls it in as a transitive dependency.
# A plain `pip install sentence-transformers` installs the full
# CUDA-enabled torch build by default -- hundreds of MB of GPU
# libraries that are never used on CPU-only hosting (like
# Render's free tier), but that still bloat both the image and
# runtime memory. This is very likely what pushed the container
# over Render's 512MB limit.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies first, separately from app code, so
# Docker's layer cache skips this (slow) step on every rebuild
# unless requirements-docker.txt actually changed.
#
# NOTE: this intentionally installs requirements-docker.txt, a
# minimal runtime-only list, rather than the full requirements.txt.
# requirements.txt reflects your whole Windows dev environment
# (dev/test tools like deepeval included) and has version
# conflicts that have nothing to do with what the API actually
# needs to run.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Pre-download the sentence-transformer model at BUILD time, not
# at container startup. Without this, every container start (and
# every reload during development) re-downloads ~90MB from
# Hugging Face, which is slow and fails outright in any network-
# restricted deployment environment. This step NEEDS network
# access, so it must run before offline mode is turned on below.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Only now, AFTER the model is already downloaded and cached in
# this image layer, turn off Hugging Face Hub network calls for
# runtime. This stops the app from pinging HF Hub on every
# container start (removes the "unauthenticated requests" warning
# you saw, and avoids any startup network dependency).
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Now copy the rest of the application code.
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Lightweight healthcheck against the /health endpoint using only
# Python's stdlib, since curl isn't installed in the slim image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]