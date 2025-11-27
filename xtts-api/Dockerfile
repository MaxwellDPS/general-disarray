# XTTS2 Streaming Server Dockerfile
# ==================================
# High-performance TTS server using Coqui XTTS v2
#
# Features:
#   - GPU-accelerated synthesis
#   - Streaming support via SSE
#   - Voice cloning from reference audio
#   - REST API for integration

FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ============================================================================
# 1. Install System Dependencies
# ============================================================================
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Audio libraries (Required for Coqui/SoundFile)
    libsndfile1 \
    ffmpeg \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# 2. Install Latest Rust (Fix for indexmap/rustc 1.82+ error)
# ============================================================================
# We use rustup to get the latest stable version (apt version is too old)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# ============================================================================
# 3. Install Conda and Create Environment
# ============================================================================
# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-`uname -m`.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Put conda in path so we can use 'conda' command
ENV PATH=$CONDA_DIR/bin:$PATH

# Acceopt Anaconda TOS to avoid interactive prompt
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create Conda environment with Python 3.11
RUN conda create -n xtts_env python=3.11 -y

# IMPORTANT: Set the path to the NEW environment.
# This makes the shell use the conda python/pip by default for all following commands.
ENV PATH=/opt/conda/envs/xtts_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=xtts_env

# ============================================================================
# 4. Install Python Dependencies
# ============================================================================
WORKDIR /app

# Install Python dependencies (TTS, Pytorch, FastAPI, Uvicorn, etc.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# 5. Pre-download XTTS Model
# ============================================================================
# This downloads the model during build to speed up first start
# Because we set ENV PATH above, this uses the Conda python automatically
ENV COQUI_TOS_AGREED=1

RUN python -c "\
from TTS.api import TTS; \
print('Downloading XTTS v2 model...'); \
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False); \
print('Model downloaded successfully')"
# ============================================================================
# 6. Application Setup
# ============================================================================

# Copy server code
COPY xtts_server.py .

# Create data directories
RUN mkdir -p /app/data/voices

# Environment variables
ENV XTTS_HOST=0.0.0.0
ENV XTTS_PORT=8001
ENV VOICES_DIR=/app/data/voices

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Entry point
# We use shell form or explicit python path to ensure env usage, 
# though ENV PATH handles this generally.
CMD ["python", "xtts_server.py"]