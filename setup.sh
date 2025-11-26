#!/bin/bash
# =============================================================================
# SIP AI Assistant - Setup Script for NVIDIA GB10 (Grace Blackwell)
# =============================================================================
# This script sets up all components for the local SIP AI assistant.
# Designed for Ubuntu 22.04/24.04 with NVIDIA drivers already installed.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${INSTALL_DIR:-/opt/sip-ai-assistant}"
PIPER_DIR="/opt/piper"
VLLM_PORT=8000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Check NVIDIA GPU
check_gpu() {
    log_info "Checking GPU..."
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. Please install NVIDIA drivers first."
        exit 1
    fi
    
    nvidia-smi --query-gpu=name,memory.total --format=csv
    
    # Check for GB10/Grace Blackwell
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    
    log_info "Detected GPU: $GPU_NAME with ${GPU_MEM}MB memory"
    
    if [[ $GPU_MEM -lt 90000 ]]; then
        log_warn "Less than 90GB VRAM detected. You may need to use smaller models."
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    apt-get update
    apt-get install -y \
        build-essential \
        cmake \
        git \
        curl \
        wget \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        ffmpeg \
        libsndfile1 \
        portaudio19-dev \
        espeak-ng \
        libopus-dev \
        libspeexdsp-dev \
        libssl-dev \
        libasound2-dev \
        sox \
        libsox-dev
        
    # PJSIP dependencies
    apt-get install -y \
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
        libx264-dev \
        libvpx-dev
}

# Install PJSIP with Python bindings
install_pjsip() {
    log_info "Installing PJSIP..."
    
    PJSIP_VERSION="2.14"
    PJSIP_DIR="/opt/pjproject"
    
    if [[ -d "$PJSIP_DIR" ]]; then
        log_info "PJSIP already installed at $PJSIP_DIR"
        return
    fi
    
    cd /tmp
    wget -q "https://github.com/pjsip/pjproject/archive/refs/tags/${PJSIP_VERSION}.tar.gz"
    tar xzf "${PJSIP_VERSION}.tar.gz"
    mv "pjproject-${PJSIP_VERSION}" "$PJSIP_DIR"
    cd "$PJSIP_DIR"
    
    # Configure with audio/video support
    ./configure --enable-shared --with-opus --with-sdl
    make dep
    make -j$(nproc)
    make install
    
    # Build Python bindings
    cd pjsip-apps/src/swig
    make python
    
    # Install Python module
    cd python
    python3 setup.py install
    
    log_info "PJSIP installed successfully"
}

# Install Piper TTS
install_piper() {
    log_info "Installing Piper TTS..."
    
    PIPER_VERSION="v1.2.0"
    
    mkdir -p "$PIPER_DIR"
    cd "$PIPER_DIR"
    
    # Download Piper binary
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
        PIPER_ARCH="amd64"
    elif [[ "$ARCH" == "aarch64" ]]; then
        PIPER_ARCH="arm64"  # For Grace Blackwell
    else
        log_error "Unsupported architecture: $ARCH"
        exit 1
    fi
    
    wget -q "https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_${PIPER_ARCH}.tar.gz"
    tar xzf "piper_linux_${PIPER_ARCH}.tar.gz"
    rm "piper_linux_${PIPER_ARCH}.tar.gz"
    
    # Download voice model
    mkdir -p models
    cd models
    
    # Download Amy (medium quality, good for phone)
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json"
    
    # Create symlink for easy access
    ln -sf "$PIPER_DIR/piper/piper" /usr/local/bin/piper
    
    log_info "Piper TTS installed with Amy voice"
}

# Setup Python virtual environment
setup_python_env() {
    log_info "Setting up Python environment..."
    
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    
    # Install PyTorch with CUDA support
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install faster-whisper
    pip install faster-whisper
    
    # Install vLLM for LLM serving
    pip install vllm
    
    # Install application requirements
    pip install -r "$SCRIPT_DIR/requirements.txt"
    
    log_info "Python environment ready"
}

# Download Whisper model
download_whisper_model() {
    log_info "Downloading Whisper large-v3 model..."
    
    source "$INSTALL_DIR/venv/bin/activate"
    
    python3 -c "
from faster_whisper import WhisperModel
print('Downloading Whisper large-v3...')
model = WhisperModel('large-v3', device='cuda', compute_type='float16')
print('Model downloaded and cached')
"
    
    log_info "Whisper model ready"
}

# Download LLM model
download_llm_model() {
    log_info "Downloading LLM model..."
    
    # For GB10 with 100GB VRAM, we can run Llama 3.1 70B
    # This will be served via vLLM
    
    MODEL_NAME="${LLM_MODEL:-meta-llama/Llama-3.1-70B-Instruct}"
    
    # Check if HuggingFace token is set (needed for Llama)
    if [[ -z "$HF_TOKEN" ]]; then
        log_warn "HF_TOKEN not set. You may need to authenticate with HuggingFace."
        log_warn "Run: huggingface-cli login"
    fi
    
    source "$INSTALL_DIR/venv/bin/activate"
    
    # Pre-download the model
    python3 -c "
from huggingface_hub import snapshot_download
import os

model = '${MODEL_NAME}'
print(f'Downloading {model}...')
try:
    snapshot_download(repo_id=model)
    print('Model downloaded successfully')
except Exception as e:
    print(f'Note: {e}')
    print('Model will be downloaded when vLLM starts')
"
    
    log_info "LLM model preparation complete"
}

# Create systemd services
create_services() {
    log_info "Creating systemd services..."
    
    # vLLM service
    cat > /etc/systemd/system/vllm.service << EOF
[Unit]
Description=vLLM Server for SIP AI Assistant
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin"
Environment="HF_TOKEN=${HF_TOKEN:-}"
ExecStart=$INSTALL_DIR/venv/bin/python -m vllm.entrypoints.openai.api_server \\
    --model ${LLM_MODEL:-meta-llama/Llama-3.1-70B-Instruct} \\
    --port $VLLM_PORT \\
    --tensor-parallel-size 1 \\
    --gpu-memory-utilization 0.85 \\
    --max-model-len 8192
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    # SIP AI Assistant service
    cat > /etc/systemd/system/sip-ai-assistant.service << EOF
[Unit]
Description=SIP AI Assistant
After=network.target vllm.service
Wants=vllm.service

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin"
Environment="LLM_BASE_URL=http://localhost:$VLLM_PORT/v1"
Environment="PIPER_MODEL_PATH=$PIPER_DIR/models"
EnvironmentFile=-/etc/sip-ai-assistant/config.env
ExecStart=$INSTALL_DIR/venv/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    # Create config directory
    mkdir -p /etc/sip-ai-assistant
    
    # Create default config
    cat > /etc/sip-ai-assistant/config.env << EOF
# SIP Configuration
SIP_USER=ai-assistant
SIP_PASSWORD=
SIP_DOMAIN=localhost
SIP_PORT=5060
# SIP_REGISTRAR=sip.yourprovider.com

# LLM Configuration
LLM_BACKEND=vllm
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
LLM_BASE_URL=http://localhost:8000/v1

# Whisper Configuration
WHISPER_MODEL=large-v3

# Piper TTS Configuration
PIPER_MODEL=en_US-amy-medium
PIPER_MODEL_PATH=/opt/piper/models

# Logging
LOG_LEVEL=INFO
EOF

    systemctl daemon-reload
    
    log_info "Systemd services created"
}

# Copy application files
install_application() {
    log_info "Installing application files..."
    
    mkdir -p "$INSTALL_DIR"
    
    cp "$SCRIPT_DIR"/*.py "$INSTALL_DIR/"
    cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"
    
    # Create data directory
    mkdir -p "$INSTALL_DIR/data"
    mkdir -p "$INSTALL_DIR/data/recordings"
    mkdir -p "$INSTALL_DIR/data/logs"
    
    log_info "Application installed to $INSTALL_DIR"
}

# Print usage instructions
print_usage() {
    log_info "Installation complete!"
    echo ""
    echo "=============================================="
    echo "SIP AI Assistant - Setup Complete"
    echo "=============================================="
    echo ""
    echo "Configuration file: /etc/sip-ai-assistant/config.env"
    echo ""
    echo "To start services:"
    echo "  sudo systemctl start vllm"
    echo "  sudo systemctl start sip-ai-assistant"
    echo ""
    echo "To enable on boot:"
    echo "  sudo systemctl enable vllm"
    echo "  sudo systemctl enable sip-ai-assistant"
    echo ""
    echo "To view logs:"
    echo "  journalctl -u sip-ai-assistant -f"
    echo "  journalctl -u vllm -f"
    echo ""
    echo "SIP Endpoint: sip:ai-assistant@localhost:5060"
    echo ""
    echo "For SIP provider integration, update config.env with:"
    echo "  - SIP_REGISTRAR"
    echo "  - SIP_USER"
    echo "  - SIP_PASSWORD"
    echo "  - SIP_DOMAIN"
    echo ""
    echo "=============================================="
}

# Main installation
main() {
    log_info "Starting SIP AI Assistant setup..."
    
    check_root
    check_gpu
    install_system_deps
    install_pjsip
    install_piper
    setup_python_env
    download_whisper_model
    download_llm_model
    install_application
    create_services
    print_usage
}

# Run installation
main "$@"
