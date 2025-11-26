# SIP AI Assistant

A fully local AI voice assistant that communicates via SIP (Session Initiation Protocol). Designed for NVIDIA GB10/Grace Blackwell with ≤100GB VRAM, supporting single-user real-time conversations.

## Features

- **Fully Local**: No cloud dependencies - all processing happens on your hardware
- **SIP Integration**: Standard telephony protocol, works with any SIP client/trunk
- **Real-time STT**: Whisper large-v3 for accurate speech recognition
- **Local LLM**: Llama 3.1 70B via vLLM for intelligent responses
- **Natural TTS**: Piper TTS for high-quality voice synthesis
- **Tool Calling**: Built-in support for timers, callbacks, and extensible tools
- **Low Latency**: Optimized for conversational response times

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SIP AI Assistant                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────┐    ┌──────────────┐    ┌─────────────────────┐   │
│   │   SIP   │◄──►│    Audio     │◄──►│       LLM           │   │
│   │ Handler │    │   Pipeline   │    │      Engine         │   │
│   │(PJSUA2) │    │              │    │                     │   │
│   └─────────┘    │ ┌──────────┐ │    │ ┌─────────────────┐ │   │
│                  │ │   VAD    │ │    │ │   vLLM Server   │ │   │
│   ┌─────────┐    │ │(WebRTC)  │ │    │ │  (Llama 70B)    │ │   │
│   │  Tool   │◄───┤ ├──────────┤ │    │ └─────────────────┘ │   │
│   │ Manager │    │ │   STT    │ │    │                     │   │
│   │         │    │ │(Whisper) │ │    │ ┌─────────────────┐ │   │
│   │ Timers  │    │ ├──────────┤ │    │ │  Tool Parser    │ │   │
│   │Callbacks│    │ │   TTS    │ │    │ └─────────────────┘ │   │
│   └─────────┘    │ │ (Piper)  │ │    └─────────────────────┘   │
│                  │ └──────────┘ │                               │
│                  └──────────────┘                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

### Hardware
- NVIDIA GPU with ≤100GB VRAM (designed for GB10/Grace Blackwell)
- 64GB+ system RAM recommended
- Fast NVMe storage for model loading

### Software
- Ubuntu 22.04 or 24.04
- NVIDIA Driver 535+
- CUDA 12.1+
- Python 3.10+
- Docker (optional, for containerized deployment)

## Quick Start

### Option 1: Docker Deployment (Recommended)

```bash
# Clone or copy the project
cd sip-ai-assistant

# Create environment file
cat > .env << EOF
HF_TOKEN=your_huggingface_token
LLM_MODEL=meta-llama/Llama-3.1-70B-Instruct
WHISPER_MODEL=large-v3
SIP_USER=ai-assistant
SIP_DOMAIN=localhost
EOF

# Start services
docker compose up -d

# View logs
docker compose logs -f sip-assistant
```

### Option 2: Native Installation

```bash
# Make setup script executable
chmod +x setup.sh

# Run installation (as root)
sudo ./setup.sh

# Configure SIP settings
sudo nano /etc/sip-ai-assistant/config.env

# Start services
sudo systemctl start vllm
sudo systemctl start sip-ai-assistant

# Enable on boot
sudo systemctl enable vllm sip-ai-assistant
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIP_USER` | ai-assistant | SIP username |
| `SIP_PASSWORD` | (empty) | SIP password |
| `SIP_DOMAIN` | localhost | SIP domain |
| `SIP_PORT` | 5060 | SIP port |
| `SIP_REGISTRAR` | (empty) | SIP registrar server |
| `LLM_MODEL` | meta-llama/Llama-3.1-70B-Instruct | LLM model name |
| `LLM_BASE_URL` | http://localhost:8000/v1 | vLLM API endpoint |
| `WHISPER_MODEL` | large-v3 | Whisper model size |
| `PIPER_MODEL` | en_US-amy-medium | Piper voice model |
| `LOG_LEVEL` | INFO | Logging verbosity |

### SIP Provider Integration

To connect to a SIP provider or PBX:

```bash
# Edit configuration
sudo nano /etc/sip-ai-assistant/config.env

# Add your provider details:
SIP_USER=your_extension
SIP_PASSWORD=your_password
SIP_DOMAIN=sip.provider.com
SIP_REGISTRAR=sip.provider.com
```

### Asterisk Integration

For robust telephony, integrate with Asterisk:

```ini
; /etc/asterisk/pjsip.conf
[ai-assistant]
type=endpoint
context=from-internal
disallow=all
allow=ulaw
allow=alaw
auth=ai-assistant-auth
aors=ai-assistant-aor

[ai-assistant-auth]
type=auth
auth_type=userpass
username=ai-assistant
password=your_password

[ai-assistant-aor]
type=aor
contact=sip:ai-assistant@localhost:5060
```

## Usage

### Making a Call

Connect any SIP client to:
```
sip:ai-assistant@your-server:5060
```

### Voice Commands

The assistant responds to natural conversation. Built-in capabilities include:

**Timers:**
- "Set a timer for 5 minutes"
- "Remind me in 30 seconds"
- "Set a 1 hour timer"

**Callbacks:**
- "Call me back in 10 minutes"
- "Give me a wake-up call in 2 hours"

**Status:**
- "What timers do I have?"
- "Check my callbacks"

**Control:**
- "Cancel all timers"
- "Goodbye" (ends call)

## Extending with Custom Tools

Add custom tools by creating a new tool class:

```python
from tool_manager import BaseTool, ToolResult, ToolStatus

class WeatherTool(BaseTool):
    name = "WEATHER"
    description = "Get weather information"
    
    async def execute(self, params):
        city = params.get('city', 'unknown')
        # Your weather API logic here
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message=f"The weather in {city} is sunny and 72 degrees"
        )

# Register in main.py
assistant.tool_manager.register_tool(WeatherTool(assistant))
```

Update the system prompt in `config.py` to inform the LLM about the new tool.

## Voice Cloning

The assistant supports voice cloning to use a custom voice for TTS. This allows you to create a unique voice identity from just 5-30 seconds of reference audio.

### Supported Backends

| Backend | Quality | Speed | VRAM | Notes |
|---------|---------|-------|------|-------|
| XTTS v2 | ⭐⭐⭐⭐⭐ | Medium | ~6GB | Best quality, multilingual |
| Chatterbox | ⭐⭐⭐⭐ | Fast | ~4GB | Good balance |
| OpenVoice | ⭐⭐⭐⭐ | Medium | ~4GB | Style transfer |
| Fish Speech | ⭐⭐⭐ | Fast | ~3GB | Streaming support |

### Quick Start with Voice Cloning

```bash
# 1. Enable voice cloning in .env
VOICE_CLONING_ENABLED=true
VOICE_CLONING_BACKEND=xtts

# 2. Create a voice profile from your recording
python voice_manager.py create my_voice recording.wav --test

# 3. Set as default (or specify in .env)
python voice_manager.py set-default my_voice
```

### Creating Voice Profiles

Use the `voice_manager.py` CLI tool:

```bash
# Create from a single recording
python voice_manager.py create assistant_voice reference.wav

# Create from multiple samples (better quality)
python voice_manager.py create assistant_voice sample1.wav sample2.wav sample3.wav

# Create with options
python voice_manager.py create my_voice recording.wav \
    --language en \
    --description "Main assistant voice" \
    --default \
    --test

# List all profiles
python voice_manager.py list

# Test a profile
python voice_manager.py test my_voice "Hello, this is a test!" --play

# Delete a profile
python voice_manager.py delete old_voice
```

### Programmatic Voice Cloning

```python
from audio_pipeline import AudioPipeline
from config import Config

config = Config()
config.voice_cloning_enabled = True
config.voice_cloning_backend = "xtts"

pipeline = AudioPipeline(config)
await pipeline.start()

# Create a voice profile
await pipeline.create_voice_profile(
    name="my_voice",
    reference_audio="/path/to/recording.wav",
    description="My custom voice",
    language="en",
    set_as_default=True
)

# List available profiles
profiles = pipeline.list_voice_profiles()
print(f"Available voices: {profiles}")

# Synthesize with cloned voice
audio = await pipeline.synthesize("Hello world!", voice_profile="my_voice")

# Switch default voice
pipeline.set_voice_profile("other_voice")
```

### Tips for Best Results

**Recording Reference Audio:**
- Use 5-30 seconds of clean speech
- Minimize background noise
- Speak naturally in your normal voice
- Record in a quiet room
- Use consistent volume and tone
- Multiple short samples often work better than one long one

**Supported Audio Formats:**
- WAV (recommended)
- MP3
- FLAC
- OGG

**VRAM Considerations:**
With voice cloning enabled, total VRAM usage:
- vLLM (Llama 70B): ~80GB
- Whisper large-v3: ~5GB
- XTTS v2: ~6GB
- Total: ~91GB (fits in 100GB)

For tighter memory, use a smaller LLM or Whisper model.

## Performance Tuning

### GPU Memory Optimization

For 100GB VRAM allocation (with voice cloning):
- vLLM: ~80GB (Llama 70B with KV cache)
- Whisper large-v3: ~5GB
- Voice Cloning (XTTS): ~6GB
- System overhead: ~9GB

Without voice cloning:
- vLLM: ~80GB
- Whisper large-v3: ~5GB
- Piper TTS: ~0.1GB
- System overhead: ~15GB

Adjust `gpu-memory-utilization` in vLLM if needed:
```bash
# In docker-compose.yml or vllm.service
--gpu-memory-utilization 0.80  # 80% of available memory
```

### Latency Optimization

1. **Reduce max_model_len** for faster inference:
   ```bash
   --max-model-len 4096  # Instead of 8192
   ```

2. **Use faster Whisper model** for lower latency:
   ```bash
   WHISPER_MODEL=medium  # Instead of large-v3
   ```

3. **Adjust TTS speed**:
   ```python
   # In config.py
   piper_length_scale: float = 0.9  # Faster speech
   ```

### Smaller Models for Lower VRAM

If you have less than 100GB VRAM:

| VRAM | Recommended LLM | Whisper |
|------|-----------------|---------|
| 80GB | Llama 3.1 70B (4-bit) | large-v3 |
| 48GB | Llama 3.1 70B (4-bit) | medium |
| 24GB | Llama 3.1 8B | medium |
| 16GB | Llama 3.1 8B (4-bit) | small |

## Troubleshooting

### SIP Registration Failed

```bash
# Check network connectivity
telnet sip.provider.com 5060

# Verify credentials
journalctl -u sip-ai-assistant | grep -i "registration"

# Enable debug logging
LOG_LEVEL=DEBUG
```

### No Audio

```bash
# Check audio codecs
journalctl -u sip-ai-assistant | grep -i "codec"

# Verify RTP ports are open
sudo ss -ulnp | grep -E "1000[0-9]"
```

### LLM Timeout

```bash
# Check vLLM status
curl http://localhost:8000/health

# View vLLM logs
journalctl -u vllm -f

# Reduce model size if OOM
LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
```

### Whisper Errors

```bash
# Verify CUDA is available
python3 -c "import torch; print(torch.cuda.is_available())"

# Check Whisper model loading
python3 -c "from faster_whisper import WhisperModel; m = WhisperModel('large-v3', device='cuda')"
```

## API Reference

### REST API (if enabled)

```bash
# Health check
GET /health

# List active calls
GET /calls

# Make outbound call
POST /calls
{
  "uri": "sip:user@domain.com",
  "message": "Hello, this is a test call"
}

# Cancel scheduled tasks
DELETE /tasks/{task_id}
```

## Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Project Structure

```
sip-ai-assistant/
├── main.py              # Application entry point
├── config.py            # Configuration management
├── sip_handler.py       # SIP communication (PJSUA2)
├── audio_pipeline.py    # VAD, STT, TTS processing
├── llm_engine.py        # LLM inference with tool calling
├── tool_manager.py      # Tool system and scheduling
├── requirements.txt     # Python dependencies
├── setup.sh             # Installation script
├── Dockerfile           # Container build
├── docker-compose.yml   # Container orchestration
└── README.md            # This file
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [PJSIP](https://www.pjsip.org/) - SIP stack
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - Optimized Whisper inference
- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM serving
- [Piper](https://github.com/rhasspy/piper) - Fast neural TTS
- [WebRTC VAD](https://github.com/wiseman/py-webrtcvad) - Voice activity detection
