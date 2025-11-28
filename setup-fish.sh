#!/bin/bash
# ============================================================================
# Setup Script for SIP AI Assistant with Fish Speech
# ============================================================================

set -e

echo "==================================="
echo "SIP AI Assistant Setup with Fish Speech"
echo "==================================="

# Create necessary directories
echo "[1/5] Creating directories..."
mkdir -p cache/fish-speech-checkpoints
mkdir -p cache/huggingface-cache
mkdir -p cache/whisper-cache
mkdir -p data/voices
mkdir -p data/recordings
mkdir -p data/logs

# Copy .env if not exists
if [ ! -f .env ]; then
    echo "[2/5] Creating .env from template..."
    cp .env.example .env
    echo "  -> Created .env - please edit with your settings"
else
    echo "[2/5] .env already exists, skipping..."
fi

# Clone Fish Speech repository for Docker build
echo "[3/5] Cloning Fish Speech repository..."
if [ ! -d "fish-speech" ]; then
    git clone https://github.com/fishaudio/fish-speech.git
    echo "  -> Cloned fish-speech repository"
else
    echo "  -> fish-speech directory exists, pulling latest..."
    cd fish-speech && git pull && cd ..
fi

# Download Fish Speech / OpenAudio model
echo "[4/5] Downloading Fish Speech model (OpenAudio S1-mini)..."
echo "  This may take a while depending on your internet connection..."

# Check if huggingface-cli is available
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download fishaudio/openaudio-s1-mini \
        --local-dir cache/fish-speech-checkpoints/openaudio-s1-mini 
elif command -v hf &> /dev/null; then
    hf download fishaudio/openaudio-s1-mini \
        --local-dir cache/fish-speech-checkpoints/openaudio-s1-mini 
else
    echo "  huggingface-cli not found. Installing..."
    pip install "huggingface_hub[cli]"
    huggingface-cli download fishaudio/openaudio-s1-mini \
        --local-dir cache/fish-speech-checkpoints/openaudio-s1-mini 
fi

echo "[5/5] Setup complete!"
echo ""
echo "==================================="
echo "Next steps:"
echo "==================================="
echo ""
echo "1. Edit .env with your configuration:"
echo "   - Set HF_TOKEN if using gated models"
echo "   - Configure SIP settings (SIP_USER, SIP_PASSWORD, SIP_DOMAIN)"
echo "   - Optionally set FISH_SPEECH_REFERENCE_ID for voice cloning"
echo ""
echo "2. (Optional) Add a reference voice for cloning:"
echo "   - Place a WAV file in data/voices/"
echo "   - Set FISH_SPEECH_REFERENCE_AUDIO in .env"
echo ""
echo "3. Build and start the services:"
echo "   docker compose build"
echo "   docker compose up -d"
echo ""
echo "   Note: First build will take 10-20 minutes to compile Fish Speech."
echo ""
echo "4. Check logs:"
echo "   docker compose logs -f"
echo ""
echo "5. Test by calling your SIP endpoint:"
echo "   sip:ai-assistant@localhost:5060"
echo ""