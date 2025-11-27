"""
Low-Latency Configuration
=========================
Optimized settings for minimum response latency.

Changes from standard config:
- Shorter silence timeout (500ms vs 1000ms)
- More aggressive VAD
- Reduced barge-in threshold
- Smaller Whisper beam size
- Lower XTTS temperature/top_k for faster sampling
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """Low-latency optimized configuration."""
    
    # ===================
    # SIP Configuration
    # ===================
    sip_user: str = field(default_factory=lambda: os.getenv("SIP_USER", "ai-assistant"))
    sip_password: str = field(default_factory=lambda: os.getenv("SIP_PASSWORD", ""))
    sip_domain: str = field(default_factory=lambda: os.getenv("SIP_DOMAIN", "localhost"))
    sip_port: int = field(default_factory=lambda: int(os.getenv("SIP_PORT", "5060")))
    sip_transport: str = field(default_factory=lambda: os.getenv("SIP_TRANSPORT", "udp"))
    sip_registrar: Optional[str] = field(default_factory=lambda: os.getenv("SIP_REGISTRAR"))
    audio_codecs: list = field(default_factory=lambda: ["PCMU", "PCMA", "opus"])
    
    # ===================
    # Audio Configuration (Optimized)
    # ===================
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 20  # Reduced from 30ms for faster processing
    
    # Voice Activity Detection (Aggressive)
    vad_aggressiveness: int = 3  # Maximum (was 0) - faster speech detection
    
    # Barge-in (More sensitive)
    barge_in_min_duration_ms: int = field(default_factory=lambda: int(os.getenv("BARGE_IN_MIN_DURATION", "400")))  # Reduced from 700
    barge_in_energy_threshold: int = field(default_factory=lambda: int(os.getenv("BARGE_IN_ENERGY_THRESHOLD", "2000")))  # Reduced from 2500
    
    # Speech detection (Faster)
    speech_pad_ms: int = 200  # Reduced from 300
    min_speech_duration_ms: int = 200  # Reduced from 250
    max_speech_duration_s: float = 8.0  # Reduced from 10
    silence_duration_ms: int = field(default_factory=lambda: int(os.getenv("SILENCE_TIMEOUT_MS", "500")))  # Reduced from 1000
    
    # ===================
    # STT Configuration (Speed optimized)
    # ===================
    # Try faster-whisper first, fall back to standard
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "medium"))  # Medium faster than large
    whisper_device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cuda"))
    whisper_language: str = field(default_factory=lambda: os.getenv("WHISPER_LANGUAGE", "en"))
    whisper_beam_size: int = field(default_factory=lambda: int(os.getenv("WHISPER_BEAM_SIZE", "3")))  # Reduced from 5
    
    # ===================
    # LLM Configuration
    # ===================
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "vllm"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "not-needed"))
    
    # Generation (Faster)
    llm_max_tokens: int = 150  # Reduced from 2000 - keep responses short for voice
    llm_temperature: float = 0.6  # Lower for more predictable/faster responses
    llm_top_p: float = 0.85
    
    max_conversation_turns: int = 10  # Reduced from 20 for less context
    
    # ===================
    # TTS Configuration (Speed optimized)
    # ===================
    xtts_api_url: str = field(default_factory=lambda: os.getenv("XTTS_API_URL", "http://localhost:8001"))
    xtts_default_voice: str = field(default_factory=lambda: os.getenv("XTTS_DEFAULT_VOICE", "tts_models/multilingual/multi-dataset/xtts_v2"))
    
    # Lower values = faster synthesis
    xtts_temperature: float = field(default_factory=lambda: float(os.getenv("XTTS_TEMPERATURE", "0.65")))
    xtts_top_k: int = field(default_factory=lambda: int(os.getenv("XTTS_TOP_K", "30")))
    xtts_top_p: float = field(default_factory=lambda: float(os.getenv("XTTS_TOP_P", "0.8")))
    xtts_repetition_penalty: float = field(default_factory=lambda: float(os.getenv("XTTS_REPETITION_PENALTY", "2.0")))
    
    # ===================
    # Voice Cloning
    # ===================
    voices_dir: Path = field(default_factory=lambda: Path(os.getenv("VOICES_DIR", "./data/voices")))
    
    # ===================
    # Tools
    # ===================
    enable_timer_tool: bool = True
    enable_callback_tool: bool = True
    enable_search_tool: bool = False
    enable_calendar_tool: bool = False
    max_timer_duration_hours: int = 24
    callback_retry_attempts: int = 3
    callback_retry_delay_s: int = 60
    
    # ===================
    # System
    # ===================
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_gpu_memory_gb: float = 90.0
    
    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "recordings").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        
    @property
    def system_prompt(self) -> str:
        """Optimized system prompt - shorter for faster processing."""
        return """You are a voice assistant on a phone call.

RULES:
- Keep responses SHORT (1-2 sentences max)
- Be conversational, not formal
- No markdown or formatting
- If confused, ask briefly

TOOLS (format: [TOOL:NAME:params]):
- SET_TIMER: [TOOL:SET_TIMER:duration=SECONDS,message=TEXT]
- CALLBACK: [TOOL:CALLBACK:delay=SECONDS,destination=NUMBER]
- HANGUP: [TOOL:HANGUP]

Be helpful and concise."""


# Singleton
_config: Optional[Config] = None

def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config