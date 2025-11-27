# ============================================================================
# STAGE 1: Builder
# ============================================================================
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install Build Tools
# Note: We removed libpcre2-dev as it is useless for SWIG 4.0.2
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    tar \
    pkg-config \
    python3-dev \
    # Audio dev libs
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
    && rm -rf /var/lib/apt/lists/*

# 2. Build SWIG 4.0.2 (With Manual PCRE Build)
# Ubuntu 24.04 only has PCRE2. SWIG 4.0.2 needs PCRE1.
# We use the internal script to build PCRE1 statically.
# 2. Build SWIG 4.0.2 (With Manual PCRE Download)
WORKDIR /tmp
RUN wget -q http://prdownloads.sourceforge.net/swig/swig-4.0.2.tar.gz && \
    tar -xf swig-4.0.2.tar.gz && \
    cd swig-4.0.2 && \
    # FIX: Manually download PCRE 8.45 so pcre-build.sh finds it
    wget -q https://sourceforge.net/projects/pcre/files/pcre/8.45/pcre-8.45.tar.gz && \
    ./Tools/pcre-build.sh && \
    ./configure --prefix=/usr && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf swig-4.0.2*

# 2. Setup Conda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-`uname -m`.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n xtts_env python=3.11 -y

ENV PATH=/opt/conda/envs/xtts_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=xtts_env

# 3. Build PJSIP (STATIC with fPIC)
WORKDIR /tmp
RUN wget -q https://github.com/pjsip/pjproject/archive/refs/tags/2.14.tar.gz && \
    tar xzf 2.14.tar.gz && \
    cd pjproject-2.14 && \
    # IMPORTANT: Force -fPIC so these static libs can be linked into a Python shared module
    export CFLAGS="$CFLAGS -fPIC" && \
    export CXXFLAGS="$CXXFLAGS -fPIC" && \
    # REMOVED --enable-shared
    ./configure --with-opus && \
    make dep && \
    make -j$(nproc) && \
    make install

# 4. Build Python Wheel (Self-Contained)
RUN pip install --upgrade pip setuptools wheel && \
    cd /tmp/pjproject-2.14/pjsip-apps/src/swig && \
    make python && \
    cd python && \
    python setup.py bdist_wheel && \
    mkdir -p /build_artifacts && \
    cp dist/*.whl /build_artifacts/

# ============================================================================
# STAGE 2: Final Runtime Image
# ============================================================================
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 1. Runtime System Deps
# Note: No need for build tools, but we still need runtime audio libs
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    libasound2t64 \
    libssl-dev \
    libopus0 \
    libv4l-0 \
    libsdl2-2.0-0 \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    espeak-ng \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Conda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-`uname -m`.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n xtts_env python=3.11 -y

ENV PATH=/opt/conda/envs/xtts_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=xtts_env

# 3. Install Application & Wheel
WORKDIR /app

# Copy ONLY the wheel. No raw .so files needed!
COPY --from=builder /build_artifacts/*.whl /tmp/

# The wheel now contains the PJSIP code statically linked
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# 4. Python Dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu129

RUN pip3 install --no-cache-dir -r requirements.txt

# 5. Final Setup
COPY src/ ./
RUN mkdir -p /app/data/recordings /app/data/logs /app/data/voices

EXPOSE 5060/udp 5060/tcp 10000-10100/udp

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

CMD ["python3", "main.py"]