"""
Configuration for SIP AI Assistant
===================================
All configurable parameters in one place.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class Config:
    """Application configuration."""
    
    # ===================
    # SIP Configuration
    # ===================
    sip_user: str = field(default_factory=lambda: os.getenv("SIP_USER", "ai-assistant"))
    sip_password: str = field(default_factory=lambda: os.getenv("SIP_PASSWORD", ""))
    sip_domain: str = field(default_factory=lambda: os.getenv("SIP_DOMAIN", "localhost"))
    sip_port: int = field(default_factory=lambda: int(os.getenv("SIP_PORT", "5060")))
    sip_transport: str = field(default_factory=lambda: os.getenv("SIP_TRANSPORT", "udp"))  # udp, tcp, tls
    sip_registrar: Optional[str] = field(default_factory=lambda: os.getenv("SIP_REGISTRAR"))
    
    # Audio codec preferences (in order of preference)
    audio_codecs: list = field(default_factory=lambda: ["PCMU", "PCMA", "opus"])
    
    # ===================
    # Audio Configuration  
    # ===================
    sample_rate: int = 16000  # Hz - optimal for Whisper
    channels: int = 1  # Mono
    chunk_duration_ms: int = 30  # VAD frame size
    
    # Voice Activity Detection
    vad_aggressiveness: int = 0  # 0-3, higher = more aggressive
    barge_in_min_duration_ms: int = field(default_factory=lambda: int(os.getenv("BARGE_IN_MIN_DURATION", "700")))
    barge_in_energy_threshold: int = field(default_factory=lambda: int(os.getenv("BARGE_IN_ENERGY_THRESHOLD", "2500")))
    speech_pad_ms: int = 300  # Padding around speech
    min_speech_duration_ms: int = 250  # Minimum speech to process
    max_speech_duration_s: float = 10.0  # Maximum single utterance
    silence_duration_ms: int = 1000  # Silence to end utterance
    
    # ===================
    # STT Configuration (faster-whisper)
    # ===================
    whisper_model: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "large-v3"))
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"  # float16, int8, int8_float16
    whisper_language: str = "en"
    whisper_beam_size: int = 5
    whisper_vad_filter: bool = True
    
    # ===================
    # LLM Configuration
    # ===================
    llm_backend: str = field(default_factory=lambda: os.getenv("LLM_BACKEND", "vllm"))  # vllm, ollama, lmstudio
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-70B-Instruct"))
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "not-needed"))
    
    # Generation parameters
    llm_max_tokens: int = 2000  # Keep responses concise for voice
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    
    # Context management
    max_conversation_turns: int = 20  # Limit context length
    
    # ===================
    # TTS Configuration (Piper - fallback)
    # ===================
    piper_model: str = field(default_factory=lambda: os.getenv("PIPER_MODEL", "en_US-amy-medium"))
    piper_model_path: Path = field(default_factory=lambda: Path(os.getenv(
        "PIPER_MODEL_PATH", 
        "/opt/piper/models"
    )))
    piper_speaker_id: int = 0
    piper_length_scale: float = 1.0  # Speech speed (lower = faster)
    piper_noise_scale: float = 0.667
    piper_noise_w: float = 0.8
    tts_sample_rate: int = 22050  # Piper native rate
    
    # ===================
    # Voice Cloning Configuration
    # ===================
    voice_cloning_enabled: bool = field(default_factory=lambda: os.getenv("VOICE_CLONING_ENABLED", "false").lower() == "true")
    voice_cloning_backend: str = field(default_factory=lambda: os.getenv("VOICE_CLONING_BACKEND", "xtts"))  # xtts, chatterbox, openvoice, fish_speech
    voice_cloning_profile: str = field(default_factory=lambda: os.getenv("VOICE_CLONING_PROFILE", ""))  # Default voice profile name
    voices_dir: Path = field(default_factory=lambda: Path(os.getenv("VOICES_DIR", "./data/voices")))
    
    # XTTS specific settings
    xtts_temperature: float = field(default_factory=lambda: float(os.getenv("XTTS_TEMPERATURE", "0.7")))
    xtts_top_k: int = field(default_factory=lambda: int(os.getenv("XTTS_TOP_K", "50")))
    xtts_top_p: float = field(default_factory=lambda: float(os.getenv("XTTS_TOP_P", "0.85")))
    xtts_repetition_penalty: float = field(default_factory=lambda: float(os.getenv("XTTS_REPETITION_PENALTY", "2.0")))
    
    # Chatterbox specific settings
    chatterbox_exaggeration: float = field(default_factory=lambda: float(os.getenv("CHATTERBOX_EXAGGERATION", "0.5")))
    chatterbox_cfg_weight: float = field(default_factory=lambda: float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5")))
    
    # ===================
    # Tool Configuration
    # ===================
    enable_timer_tool: bool = True
    enable_callback_tool: bool = True
    enable_search_tool: bool = False  # Requires internet
    enable_calendar_tool: bool = False  # Requires integration
    
    max_timer_duration_hours: int = 24
    callback_retry_attempts: int = 3
    callback_retry_delay_s: int = 60
    
    # ===================
    # System Configuration
    # ===================
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Memory limits (for GB10 optimization)
    max_gpu_memory_gb: float = 90.0  # Leave headroom
    
    def __post_init__(self):
        """Validate and setup paths."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "recordings").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        
    @property
    def system_prompt(self) -> str:
        """System prompt for the LLM."""
        return """You are a CHAOS-AI voice assistant communicating over a phone call. 

IMPORTANT GUIDELINES:
- Keep responses concise and conversational - this is a phone call, not a text chat
- Speak naturally as if having a phone conversation
- Avoid using markdown, bullet points, or formatting - just speak plainly
- If you don't understand, ask for clarification
- Be friendly and helpful

AVAILABLE TOOLS:
1. SET_TIMER: Set a timer/reminder
   - User says: "Set a timer for 5 minutes"
   - Output: [TOOL:SET_TIMER:duration=300,message=Timer done]

2. CALLBACK: Schedule a callback
   - User says: "Call me back in 30 minutes"
     Output: [TOOL:CALLBACK:delay=1800,message=Calling you back]
   
   - User says: "Call me back at 405" or "Call 405 in 1 minute"
     Output: [TOOL:CALLBACK:delay=60,destination=405,message=Callback for extension 405]

3. HANGUP: End the call
   - User says: "Goodbye"
   - Output: [TOOL:HANGUP]

FORMAT:
[TOOL:tool_name:param1=value1,param2=value2]

Example: "I'll set a timer for 5 minutes. I'll let you know when it goes off. [TOOL:SET_TIMER:duration=300,message=Your 5 minute timer is up]"

Remember: You're on a phone call. Be natural, be helpful, be concise."""


# Singleton instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get or create config singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config
