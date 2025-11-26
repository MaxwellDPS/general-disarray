# Dockerfile for SIP AI Assistant
# ================================
# Multi-stage build for optimized image

# -------------------------------------------------------------------
# STAGE 1: PJSIP Builder
# Compiles PJSIP and generates the Python Wheel
# -------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS pjsip-builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    libasound2-dev \
    libssl-dev \
    libopus-dev \
    libv4l-dev \
    libsdl2-dev \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Build PJSIP (Version 2.14)
WORKDIR /tmp
RUN wget -q https://github.com/pjsip/pjproject/archive/refs/tags/2.14.tar.gz && \
    tar xzf 2.14.tar.gz && \
    cd pjproject-2.14 && \
    # Configure with shared libraries and Opus support
    ./configure --enable-shared --with-opus && \
    make dep && \
    make -j$(nproc) && \
    make install

# Build Python Bindings (Wheel)
RUN pip install --upgrade setuptools wheel
RUN cd /tmp/pjproject-2.14/pjsip-apps/src/swig && \
    make python && \
    cd python && \
    make wheel

# -------------------------------------------------------------------
# STAGE 2: Piper Downloader
# Downloads the standalone Piper binary and Voice Model
# -------------------------------------------------------------------
FROM ubuntu:22.04 AS piper-downloader

RUN apt-get update && apt-get install -y wget tar

WORKDIR /opt/piper

# 1. Download Piper Binary (amd64)
# We use the specific 1.2.0 release which is stable
RUN wget -q "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz" && \
    tar xzf "piper_amd64.tar.gz" -C . && \
    rm "piper_amd64.tar.gz" 
    # # Move content up one level so /opt/piper/piper exists
    # cp -r piper/* . && \
    # rm -rf piper

# 2. Download Voice Model (Amy Medium)
RUN mkdir -p /opt/piper/models
WORKDIR /opt/piper/models
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx" && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"

# -------------------------------------------------------------------
# STAGE 3: Final Runtime Image
# -------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 1. Install System Runtime Dependencies
# Note: libasound2 and libportaudio2 are critical for SIP audio
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    libasound2 \
    libssl3 \
    libopus0 \
    curl \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy PJSIP Artifacts from Stage 1
# Copy shared libraries
COPY --from=pjsip-builder /usr/local/lib/libpj* /usr/local/lib/
# Copy the generated Python wheel
COPY --from=pjsip-builder /tmp/pjproject-2.14/pjsip-apps/src/swig/python/dist/*.whl /tmp/pjsua_dist/

# 3. Copy Piper Artifacts from Stage 2
COPY --from=piper-downloader /opt/piper /opt/piper

# 4. Setup Environment Variables
# LD_LIBRARY_PATH: Essential for Python to find PJSIP .so files
# PATH: Adds piper to the path so subprocess calls work
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"
ENV PATH="/opt/piper:${PATH}"
ENV PIPER_MODEL_PATH="/opt/piper/models"

# 5. Update Shared Library Cache
RUN ldconfig

# 6. Install Python Dependencies
WORKDIR /app
COPY requirements.txt .

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools

# Install Torch (CUDA 12.1 specific)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PJSIP Wheel (From Stage 1)
RUN pip3 install --no-cache-dir /tmp/pjsua_dist/*.whl

# Install other requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# --- CRITICAL FIX FOR WHISPER CRASH ---
# faster-whisper/ctranslate2 requires cuDNN 9 for CUDA 12
# The base image only provides CUDA runtime, not the specific cuDNN libs needed by CTranslate2
RUN pip3 install --no-cache-dir nvidia-cudnn-cu12==9.1.0.70

# Add NVIDIA libs to LD_LIBRARY_PATH so faster-whisper can find libcudnn_ops.so.9
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# 7. Copy Application Code
COPY src/ .

# 8. Create data directories
RUN mkdir -p /app/data/recordings /app/data/logs

# 9. Expose Ports
# SIP Signaling
EXPOSE 5060/udp
EXPOSE 5060/tcp
# RTP Audio (Range matching your config)
EXPOSE 10000-10100/udp

# 10. Run
CMD ["python3", "main.py"]