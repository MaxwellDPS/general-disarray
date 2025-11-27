# SIP AI Assistant

A fully local voice AI assistant that communicates via SIP phone calls. Designed for NVIDIA GB10/Grace Blackwell with 100GB+ VRAM.

## Features

- **Speech-to-Text**: OpenAI Whisper with CUDA acceleration (large-v3 by default)
- **LLM**: vLLM serving Llama 3.1 70B (or other models)
- **Text-to-Speech**: XTTS v2 with voice cloning and streaming support
- **SIP Integration**: Full PJSIP implementation for phone calls
- **Smart Barge-in**: Interrupt the AI mid-speech
- **Tools**: Timers, callbacks, and extensible tool framework

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   vLLM      │  │   XTTS      │  │   SIP Assistant     │  │
│  │  (LLM)      │  │  (TTS API)  │  │  (Main App)         │  │
│  │  Port 8000  │  │  Port 8001  │  │  Port 5060          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│        ↑               ↑                    │               │
│        └───────────────┴────────────────────┘               │
└─────────────────────────────────────────────────────────────┘
                         │
                         │ SIP/RTP
                         ↓
                  ┌─────────────┐
                  │  Asterisk   │
                  │  PBX        │
                  └─────────────┘
```

## Quick Start

### 1. Prerequisites

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with 32GB+ VRAM (GB10 recommended)
- Asterisk PBX or SIP server

### 2. Configuration

```bash
# Copy example config
cp .env.example .env

# Edit with your settings
nano .env
```

Key settings:
- `HF_TOKEN`: HuggingFace token for model downloads
- `SIP_USER`, `SIP_PASSWORD`, `SIP_DOMAIN`: SIP credentials
- `WHISPER_MODEL`: STT model size (large-v3 recommended)
- `LLM_MODEL`: LLM to use (default: Llama 3.1 70B)

### 3. Launch

```bash
# Build and start all services
docker-compose up -d

# Watch logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 4. Create Voice Profile (Optional)

```bash
# Record 10-30 seconds of clear speech
# Then create a voice profile:
python voice_manager.py create my_voice reference.wav --default --test
```

## GPU Memory Usage

With GB10 (100GB+ unified memory):

| Component | Memory Usage |
|-----------|-------------|
| vLLM (70B) | ~50-60 GB |
| XTTS v2 | ~6 GB |
| Whisper large-v3 | ~3 GB |
| **Total** | ~60-70 GB |

For smaller GPUs, adjust `LLM_MODEL` to use smaller models.

## Voice Cloning

The XTTS server supports voice cloning from reference audio:

```bash
# Create a profile
python voice_manager.py create assistant reference.wav

# List profiles
python voice_manager.py list

# Test synthesis
python voice_manager.py test assistant "Hello, this is a test!"

# Delete profile
python voice_manager.py delete assistant
```

### Reference Audio Tips

- **Duration**: 5-30 seconds of clear speech
- **Quality**: Clean recording, minimal background noise
- **Content**: Natural conversational speech
- **Format**: WAV, MP3, FLAC, or OGG

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | HuggingFace token |
| `LLM_MODEL` | `meta-llama/Llama-3.1-70B-Instruct` | LLM model |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | STT device |
| `SIP_USER` | `ai-assistant` | SIP username |
| `SIP_PASSWORD` | - | SIP password |
| `SIP_DOMAIN` | `localhost` | SIP domain |
| `XTTS_DEFAULT_VOICE` | - | Default voice profile |
| `BARGE_IN_MIN_DURATION` | `700` | Barge-in sensitivity (ms) |
| `BARGE_IN_ENERGY_THRESHOLD` | `2500` | Barge-in volume threshold |

### Asterisk Configuration

Example `pjsip.conf`:

```ini
[ai-assistant]
type=endpoint
context=internal
disallow=all
allow=ulaw
allow=alaw
auth=ai-assistant-auth
aors=ai-assistant-aor

[ai-assistant-auth]
type=auth
auth_type=userpass
username=ai-assistant
password=your-password-here

[ai-assistant-aor]
type=aor
max_contacts=1
```

Example `extensions.conf`:

```ini
[internal]
exten => 100,1,Dial(PJSIP/ai-assistant)
```

## API Reference

### XTTS Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/tts` | POST | Synthesize speech |
| `/v1/tts/stream` | POST | Stream synthesis (SSE) |
| `/v1/tts/raw` | POST | Raw PCM output |
| `/v1/voices` | GET | List voices |
| `/v1/voices` | POST | Create voice |
| `/v1/voices/{name}` | DELETE | Delete voice |

### Example: Synthesize Speech

```bash
curl -X POST http://localhost:8001/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "default"}' \
  --output speech.wav
```

## Troubleshooting

### XTTS Not Loading

```bash
# Check XTTS logs
docker-compose logs xtts

# Verify GPU access
docker-compose exec xtts nvidia-smi
```

### Whisper CUDA Issues

The app uses standard OpenAI Whisper instead of faster-whisper for better GB10 compatibility. If you see CUDA errors:

```bash
# Check CUDA setup
docker-compose exec sip-assistant python3 -c "import torch; print(torch.cuda.is_available())"
```

### SIP Registration Failed

1. Check credentials in `.env`
2. Verify Asterisk is reachable
3. Check firewall (ports 5060/udp, 10000-10100/udp)

```bash
# Test SIP connectivity
docker-compose exec sip-assistant python3 -c "
from sip_handler import SIPHandler
# ...test code...
"
```

### Audio Quality Issues

- Increase `BARGE_IN_ENERGY_THRESHOLD` for noisy environments
- Adjust `WHISPER_MODEL` for accuracy vs speed
- Try different XTTS temperature settings

## Development

### Local Testing (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run XTTS server
python xtts_server.py &

# Run main app
python main.py
```

### Adding Custom Tools

Edit `tool_manager.py`:

```python
class MyCustomTool(BaseTool):
    name = "MY_TOOL"
    description = "Description for LLM"
    
    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        # Your logic here
        return ToolResult(
            status=ToolStatus.SUCCESS,
            message="Done!"
        )
```

## License

MIT License - See LICENSE file for details.

## Contributing

Pull requests welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a PR with clear description