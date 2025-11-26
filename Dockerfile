FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Install Build & Runtime Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    tar \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    # Audio Libraries (Critical for PJSIP)
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
    libportaudio2 \
    libsndfile1 \
    # Tools
    swig \
    curl \
    iproute2 \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# 2. Build PJSIP (Source)
WORKDIR /tmp
RUN wget -q https://github.com/pjsip/pjproject/archive/refs/tags/2.14.tar.gz && \
    tar xzf 2.14.tar.gz && \
    cd pjproject-2.14 && \
    # Configure: Enable shared libs, link to Opus
    ./configure --enable-shared --with-opus && \
    make dep && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# 3. Build PJSIP Python Bindings
RUN pip3 install --upgrade pip setuptools wheel
RUN cd /tmp/pjproject-2.14/pjsip-apps/src/swig && \
    make python && \
    cd python && \
    python3 setup.py install

# 4. Install Piper TTS (CLI Binary)
WORKDIR /opt/piper
# Download Piper 1.2.0 binary
RUN wget -q "https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz" && \
    tar xzf "piper_amd64.tar.gz" -C . && \
    rm "piper_amd64.tar.gz"


# Download Voice Model (Amy Medium)
RUN mkdir -p /opt/piper/models && \
    cd /opt/piper/models && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx" && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"

# Set up Piper Path
ENV PATH="/opt/piper:${PATH}"
ENV PIPER_MODEL_PATH="/opt/piper/models"

# 5. Application Setup
WORKDIR /app
COPY requirements.txt .

# Install Python Dependencies
# Install Torch first (CUDA 12.1)
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# --- CRITICAL FIX: Install cuDNN 9 for faster-whisper ---
# This fixes the "Unable to load symbol cudnnCreateTensorDescriptor" crash
RUN pip3 install --no-cache-dir nvidia-cudnn-cu12>=9.1.0.70

# Install remaining requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Add NVIDIA cuDNN libraries to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Copy Source Code
COPY src/ .
RUN mkdir -p /app/data/recordings /app/data/logs

# Expose Ports (Signaling + Audio)
EXPOSE 5060/udp 5060/tcp 10000-10100/udp

CMD ["python3", "main.py"]