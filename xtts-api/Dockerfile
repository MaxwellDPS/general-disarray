# =============================================================================
# XTTS2 Server for NVIDIA DGX Spark (GB10 / sm_121)
# =============================================================================
FROM nvcr.io/nvidia/pytorch:25.10-py3

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV COQUI_TOS_AGREED=1

# System deps
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg libsox-dev sox curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create Python 3.11 environment with conda (TTS requires <3.12)
RUN conda create -n tts python=3.11 -y

# Install everything in the conda environment
RUN source activate tts && \
    pip install --no-cache-dir \
    TTS==0.22.0 \
    fastapi uvicorn[standard] python-multipart httpx \
    transformers==4.40.2

# Download XTTS model
RUN source activate tts && python << 'PYEOF'
import torch
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, 'weights_only': False})
from TTS.utils.manage import ModelManager
ModelManager().download_model('tts_models/multilingual/multi-dataset/xtts_v2')
print('Model downloaded')
PYEOF

# Application
COPY xtts_server.py /app/
RUN mkdir -p /app/data/voices

ENV XTTS_HOST=0.0.0.0 XTTS_PORT=8001 VOICES_DIR=/app/data/voices
EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:8001/health || exit 1

CMD ["conda", "run", "-n", "tts", "--no-capture-output", "python", "/app/xtts_server.py"]