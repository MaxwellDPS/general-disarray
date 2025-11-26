"""
Voice Cloning TTS Module
========================
Supports multiple voice cloning backends:
- Coqui XTTS v2 (recommended)
- Chatterbox TTS
- OpenVoice
- Fish Speech

All run locally on GPU for privacy and low latency.
"""

import asyncio
import logging
import io
import os
import subprocess
import tempfile
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """A cloned voice profile."""
    name: str
    reference_audio_paths: List[Path]
    description: str = ""
    language: str = "en"
    
    # Cached embeddings/conditioning (backend-specific)
    _cached_conditioning: Any = field(default=None, repr=False)
    _cache_hash: str = field(default="", repr=False)
    
    def get_cache_key(self) -> str:
        """Generate cache key based on reference audio files."""
        hasher = hashlib.md5()
        for path in sorted(self.reference_audio_paths):
            if path.exists():
                hasher.update(path.read_bytes())
        return hasher.hexdigest()
        
    @property
    def is_cached(self) -> bool:
        return self._cached_conditioning is not None and self._cache_hash == self.get_cache_key()


class VoiceCloningBackend(ABC):
    """Base class for voice cloning backends."""
    
    name: str = "base"
    supports_streaming: bool = False
    min_reference_seconds: float = 3.0
    max_reference_seconds: float = 30.0
    
    def __init__(self, config: 'VoiceCloningConfig'):
        self.config = config
        self.model = None
        self.device = config.device
        
    @abstractmethod
    async def load_model(self) -> bool:
        """Load the TTS model. Returns True on success."""
        pass
        
    @abstractmethod
    async def clone_voice(self, profile: VoiceProfile) -> bool:
        """Process reference audio and cache voice conditioning."""
        pass
        
    @abstractmethod
    async def synthesize(self, text: str, profile: VoiceProfile) -> bytes:
        """Synthesize speech using the cloned voice."""
        pass
        
    async def unload_model(self):
        """Unload model to free memory."""
        self.model = None
        

class XTTSBackend(VoiceCloningBackend):
    """
    Coqui XTTS v2 Backend
    =====================
    High-quality multilingual voice cloning.
    Requires ~6GB VRAM.
    
    Install: pip install TTS
    """
    
    name = "xtts"
    supports_streaming = True
    min_reference_seconds = 3.0
    max_reference_seconds = 30.0
    
    async def load_model(self) -> bool:
        try:
            from TTS.api import TTS
            
            logger.info("Loading XTTS v2 model...")
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            )
            
            logger.info("XTTS v2 model loaded successfully")
            return True
            
        except ImportError:
            logger.error("TTS library not installed. Run: pip install TTS")
            return False
        except Exception as e:
            logger.error(f"Failed to load XTTS: {e}")
            return False
            
    async def clone_voice(self, profile: VoiceProfile) -> bool:
        """Pre-compute speaker conditioning from reference audio."""
        if not self.model:
            return False
            
        try:
            # XTTS can use multiple reference files
            reference_files = [str(p) for p in profile.reference_audio_paths if p.exists()]
            
            if not reference_files:
                logger.error(f"No valid reference files for profile: {profile.name}")
                return False
                
            loop = asyncio.get_event_loop()
            
            # Compute speaker latents (conditioning)
            gpt_cond_latent, speaker_embedding = await loop.run_in_executor(
                None,
                lambda: self.model.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=reference_files,
                    gpt_cond_len=self.config.xtts_gpt_cond_len,
                    max_ref_length=self.config.xtts_max_ref_len
                )
            )
            
            profile._cached_conditioning = {
                "gpt_cond_latent": gpt_cond_latent,
                "speaker_embedding": speaker_embedding
            }
            profile._cache_hash = profile.get_cache_key()
            
            logger.info(f"Voice profile '{profile.name}' cloned successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clone voice: {e}")
            return False
            
    async def synthesize(self, text: str, profile: VoiceProfile) -> bytes:
        if not self.model:
            raise RuntimeError("XTTS model not loaded")
            
        # Ensure voice is cloned
        if not profile.is_cached:
            await self.clone_voice(profile)
            
        if not profile._cached_conditioning:
            raise RuntimeError(f"Voice profile '{profile.name}' not properly initialized")
            
        try:
            loop = asyncio.get_event_loop()
            
            # Generate audio using cached conditioning
            conditioning = profile._cached_conditioning
            
            audio = await loop.run_in_executor(
                None,
                lambda: self.model.synthesizer.tts_model.inference(
                    text=text,
                    language=profile.language,
                    gpt_cond_latent=conditioning["gpt_cond_latent"],
                    speaker_embedding=conditioning["speaker_embedding"],
                    temperature=self.config.xtts_temperature,
                    repetition_penalty=self.config.xtts_repetition_penalty,
                    top_k=self.config.xtts_top_k,
                    top_p=self.config.xtts_top_p
                )
            )
            
            # Convert to bytes (16-bit PCM)
            audio_np = np.array(audio["wav"])
            audio_np = (audio_np * 32767).astype(np.int16)
            
            return audio_np.tobytes()
            
        except Exception as e:
            logger.error(f"XTTS synthesis failed: {e}")
            raise


class ChatterboxBackend(VoiceCloningBackend):
    """
    Chatterbox TTS Backend
    ======================
    Resemble AI's open-source voice cloning model.
    Good balance of quality and speed.
    
    Install: pip install chatterbox-tts
    """
    
    name = "chatterbox"
    supports_streaming = False
    min_reference_seconds = 5.0
    max_reference_seconds = 60.0
    
    async def load_model(self) -> bool:
        try:
            from chatterbox.tts import ChatterboxTTS
            
            logger.info("Loading Chatterbox TTS model...")
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: ChatterboxTTS.from_pretrained(device=self.device)
            )
            
            logger.info("Chatterbox TTS loaded successfully")
            return True
            
        except ImportError:
            logger.error("Chatterbox not installed. Run: pip install chatterbox-tts")
            return False
        except Exception as e:
            logger.error(f"Failed to load Chatterbox: {e}")
            return False
            
    async def clone_voice(self, profile: VoiceProfile) -> bool:
        """Chatterbox uses reference audio directly, but we can preload it."""
        if not self.model:
            return False
            
        try:
            import torchaudio
            
            # Load and combine reference audio
            all_audio = []
            sample_rate = None
            
            for path in profile.reference_audio_paths:
                if path.exists():
                    audio, sr = torchaudio.load(str(path))
                    if sample_rate is None:
                        sample_rate = sr
                    elif sr != sample_rate:
                        # Resample if needed
                        audio = torchaudio.functional.resample(audio, sr, sample_rate)
                    all_audio.append(audio)
                    
            if not all_audio:
                logger.error(f"No valid reference files for profile: {profile.name}")
                return False
                
            # Concatenate all reference audio
            import torch
            combined_audio = torch.cat(all_audio, dim=1)
            
            profile._cached_conditioning = {
                "audio": combined_audio,
                "sample_rate": sample_rate
            }
            profile._cache_hash = profile.get_cache_key()
            
            logger.info(f"Voice profile '{profile.name}' prepared for Chatterbox")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare voice: {e}")
            return False
            
    async def synthesize(self, text: str, profile: VoiceProfile) -> bytes:
        if not self.model:
            raise RuntimeError("Chatterbox model not loaded")
            
        if not profile.is_cached:
            await self.clone_voice(profile)
            
        if not profile._cached_conditioning:
            raise RuntimeError(f"Voice profile '{profile.name}' not properly initialized")
            
        try:
            loop = asyncio.get_event_loop()
            conditioning = profile._cached_conditioning
            
            # Synthesize with Chatterbox
            audio = await loop.run_in_executor(
                None,
                lambda: self.model.generate(
                    text=text,
                    audio_prompt=conditioning["audio"],
                    exaggeration=self.config.chatterbox_exaggeration,
                    cfg_weight=self.config.chatterbox_cfg_weight
                )
            )
            
            # Convert to bytes
            audio_np = audio.cpu().numpy()
            if audio_np.ndim > 1:
                audio_np = audio_np.squeeze()
            audio_np = (audio_np * 32767).astype(np.int16)
            
            return audio_np.tobytes()
            
        except Exception as e:
            logger.error(f"Chatterbox synthesis failed: {e}")
            raise


class OpenVoiceBackend(VoiceCloningBackend):
    """
    OpenVoice Backend
    =================
    MyShell's voice cloning with tone color conversion.
    Separates content and style for flexible cloning.
    
    Install: pip install openvoice-cli
    """
    
    name = "openvoice"
    supports_streaming = False
    min_reference_seconds = 3.0
    max_reference_seconds = 30.0
    
    async def load_model(self) -> bool:
        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            from melo.api import TTS as MeloTTS
            
            logger.info("Loading OpenVoice models...")
            
            loop = asyncio.get_event_loop()
            
            # Load base TTS (MeloTTS)
            self.base_tts = await loop.run_in_executor(
                None,
                lambda: MeloTTS(language="EN", device=self.device)
            )
            
            # Load tone color converter
            ckpt_path = self.config.openvoice_checkpoint_path
            self.tone_converter = await loop.run_in_executor(
                None,
                lambda: ToneColorConverter(f"{ckpt_path}/converter", device=self.device)
            )
            
            self.se_extractor = se_extractor
            self.model = True  # Flag that models are loaded
            
            logger.info("OpenVoice loaded successfully")
            return True
            
        except ImportError:
            logger.error("OpenVoice not installed. See: https://github.com/myshell-ai/OpenVoice")
            return False
        except Exception as e:
            logger.error(f"Failed to load OpenVoice: {e}")
            return False
            
    async def clone_voice(self, profile: VoiceProfile) -> bool:
        if not self.model:
            return False
            
        try:
            loop = asyncio.get_event_loop()
            
            # Extract speaker embedding from reference audio
            reference_path = str(profile.reference_audio_paths[0])
            
            target_se = await loop.run_in_executor(
                None,
                lambda: self.se_extractor.get_se(
                    reference_path,
                    self.tone_converter,
                    vad=True
                )
            )
            
            profile._cached_conditioning = {
                "speaker_embedding": target_se,
                "reference_path": reference_path
            }
            profile._cache_hash = profile.get_cache_key()
            
            logger.info(f"Voice profile '{profile.name}' cloned with OpenVoice")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clone voice: {e}")
            return False
            
    async def synthesize(self, text: str, profile: VoiceProfile) -> bytes:
        if not self.model:
            raise RuntimeError("OpenVoice not loaded")
            
        if not profile.is_cached:
            await self.clone_voice(profile)
            
        if not profile._cached_conditioning:
            raise RuntimeError(f"Voice profile '{profile.name}' not initialized")
            
        try:
            loop = asyncio.get_event_loop()
            conditioning = profile._cached_conditioning
            
            # Create temp files for intermediate audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_src:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
                    src_path = tmp_src.name
                    out_path = tmp_out.name
                    
            try:
                # Step 1: Generate base speech
                await loop.run_in_executor(
                    None,
                    lambda: self.base_tts.tts_to_file(
                        text, 
                        self.base_tts.hps.data.spk2id["EN-US"],
                        src_path,
                        speed=self.config.openvoice_speed
                    )
                )
                
                # Step 2: Get source speaker embedding
                source_se = await loop.run_in_executor(
                    None,
                    lambda: self.se_extractor.get_se(
                        src_path,
                        self.tone_converter,
                        vad=False
                    )
                )
                
                # Step 3: Convert tone color to target voice
                await loop.run_in_executor(
                    None,
                    lambda: self.tone_converter.convert(
                        audio_src_path=src_path,
                        src_se=source_se,
                        tgt_se=conditioning["speaker_embedding"],
                        output_path=out_path
                    )
                )
                
                # Read output audio
                import soundfile as sf
                audio, sr = sf.read(out_path)
                audio = (audio * 32767).astype(np.int16)
                
                return audio.tobytes()
                
            finally:
                # Cleanup temp files
                for path in [src_path, out_path]:
                    try:
                        os.unlink(path)
                    except:
                        pass
                        
        except Exception as e:
            logger.error(f"OpenVoice synthesis failed: {e}")
            raise


class FishSpeechBackend(VoiceCloningBackend):
    """
    Fish Speech Backend
    ===================
    Fast voice cloning with good quality.
    Supports streaming output.
    
    Install: pip install fish-speech
    """
    
    name = "fish_speech"
    supports_streaming = True
    min_reference_seconds = 5.0
    max_reference_seconds = 60.0
    
    async def load_model(self) -> bool:
        try:
            from fish_speech.inference import TTSInference
            
            logger.info("Loading Fish Speech model...")
            
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None,
                lambda: TTSInference(
                    checkpoint_path=self.config.fish_speech_checkpoint,
                    device=self.device
                )
            )
            
            logger.info("Fish Speech loaded successfully")
            return True
            
        except ImportError:
            logger.error("Fish Speech not installed. See: https://github.com/fishaudio/fish-speech")
            return False
        except Exception as e:
            logger.error(f"Failed to load Fish Speech: {e}")
            return False
            
    async def clone_voice(self, profile: VoiceProfile) -> bool:
        if not self.model:
            return False
            
        try:
            loop = asyncio.get_event_loop()
            
            # Extract voice features
            reference_path = str(profile.reference_audio_paths[0])
            
            speaker_embedding = await loop.run_in_executor(
                None,
                lambda: self.model.extract_speaker(reference_path)
            )
            
            profile._cached_conditioning = {
                "speaker_embedding": speaker_embedding
            }
            profile._cache_hash = profile.get_cache_key()
            
            logger.info(f"Voice profile '{profile.name}' prepared for Fish Speech")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clone voice: {e}")
            return False
            
    async def synthesize(self, text: str, profile: VoiceProfile) -> bytes:
        if not self.model:
            raise RuntimeError("Fish Speech not loaded")
            
        if not profile.is_cached:
            await self.clone_voice(profile)
            
        if not profile._cached_conditioning:
            raise RuntimeError(f"Voice profile '{profile.name}' not initialized")
            
        try:
            loop = asyncio.get_event_loop()
            conditioning = profile._cached_conditioning
            
            audio = await loop.run_in_executor(
                None,
                lambda: self.model.synthesize(
                    text=text,
                    speaker_embedding=conditioning["speaker_embedding"],
                    speed=self.config.fish_speech_speed
                )
            )
            
            audio = (audio * 32767).astype(np.int16)
            return audio.tobytes()
            
        except Exception as e:
            logger.error(f"Fish Speech synthesis failed: {e}")
            raise


@dataclass
class VoiceCloningConfig:
    """Configuration for voice cloning."""
    
    # General settings
    enabled: bool = True
    backend: str = "xtts"  # xtts, chatterbox, openvoice, fish_speech
    device: str = "cuda"
    
    # Voice profiles directory
    voices_dir: Path = field(default_factory=lambda: Path("./data/voices"))
    
    # Output settings
    output_sample_rate: int = 22050
    
    # XTTS settings
    xtts_temperature: float = 0.7
    xtts_top_k: int = 50
    xtts_top_p: float = 0.85
    xtts_repetition_penalty: float = 2.0
    xtts_gpt_cond_len: int = 30
    xtts_max_ref_len: int = 60
    
    # Chatterbox settings
    chatterbox_exaggeration: float = 0.5
    chatterbox_cfg_weight: float = 0.5
    
    # OpenVoice settings
    openvoice_checkpoint_path: str = "/opt/openvoice/checkpoints"
    openvoice_speed: float = 1.0
    
    # Fish Speech settings
    fish_speech_checkpoint: str = "/opt/fish-speech/checkpoints"
    fish_speech_speed: float = 1.0
    
    def __post_init__(self):
        self.voices_dir = Path(self.voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)


class VoiceCloningTTS:
    """
    Main voice cloning TTS interface.
    Manages voice profiles and synthesis.
    """
    
    BACKENDS = {
        "xtts": XTTSBackend,
        "chatterbox": ChatterboxBackend,
        "openvoice": OpenVoiceBackend,
        "fish_speech": FishSpeechBackend
    }
    
    def __init__(self, config: VoiceCloningConfig):
        self.config = config
        self.backend: Optional[VoiceCloningBackend] = None
        self.profiles: Dict[str, VoiceProfile] = {}
        self.default_profile: Optional[str] = None
        
    async def initialize(self) -> bool:
        """Initialize the voice cloning system."""
        if not self.config.enabled:
            logger.info("Voice cloning is disabled")
            return True
            
        # Create backend
        backend_class = self.BACKENDS.get(self.config.backend)
        if not backend_class:
            logger.error(f"Unknown backend: {self.config.backend}")
            return False
            
        self.backend = backend_class(self.config)
        
        # Load model
        success = await self.backend.load_model()
        if not success:
            logger.error(f"Failed to load {self.config.backend} backend")
            self.backend = None
            return False
            
        # Load saved voice profiles
        await self._load_profiles()
        
        logger.info(f"Voice cloning initialized with {self.config.backend}")
        return True
        
    async def shutdown(self):
        """Shutdown the voice cloning system."""
        if self.backend:
            await self.backend.unload_model()
            self.backend = None
            
    async def _load_profiles(self):
        """Load voice profiles from disk."""
        import json
        
        profiles_file = self.config.voices_dir / "profiles.json"
        if not profiles_file.exists():
            return
            
        try:
            with open(profiles_file) as f:
                data = json.load(f)
                
            for name, profile_data in data.get("profiles", {}).items():
                profile = VoiceProfile(
                    name=name,
                    reference_audio_paths=[
                        Path(p) for p in profile_data.get("reference_files", [])
                    ],
                    description=profile_data.get("description", ""),
                    language=profile_data.get("language", "en")
                )
                self.profiles[name] = profile
                
            self.default_profile = data.get("default")
            logger.info(f"Loaded {len(self.profiles)} voice profiles")
            
        except Exception as e:
            logger.error(f"Failed to load profiles: {e}")
            
    async def _save_profiles(self):
        """Save voice profiles to disk."""
        import json
        
        profiles_file = self.config.voices_dir / "profiles.json"
        
        data = {
            "default": self.default_profile,
            "profiles": {
                name: {
                    "reference_files": [str(p) for p in profile.reference_audio_paths],
                    "description": profile.description,
                    "language": profile.language
                }
                for name, profile in self.profiles.items()
            }
        }
        
        with open(profiles_file, "w") as f:
            json.dump(data, f, indent=2)
            
    async def create_profile(
        self,
        name: str,
        reference_audio: Union[str, Path, List[Union[str, Path]]],
        description: str = "",
        language: str = "en",
        set_as_default: bool = False
    ) -> VoiceProfile:
        """
        Create a new voice profile from reference audio.
        
        Args:
            name: Profile name
            reference_audio: Path(s) to reference audio file(s)
            description: Optional description
            language: Language code (en, es, fr, etc.)
            set_as_default: Whether to set as default voice
            
        Returns:
            Created VoiceProfile
        """
        # Normalize paths
        if isinstance(reference_audio, (str, Path)):
            reference_paths = [Path(reference_audio)]
        else:
            reference_paths = [Path(p) for p in reference_audio]
            
        # Copy reference files to voices directory
        profile_dir = self.config.voices_dir / name
        profile_dir.mkdir(parents=True, exist_ok=True)
        
        copied_paths = []
        for i, src_path in enumerate(reference_paths):
            if src_path.exists():
                dst_path = profile_dir / f"reference_{i}{src_path.suffix}"
                import shutil
                shutil.copy2(src_path, dst_path)
                copied_paths.append(dst_path)
                
        if not copied_paths:
            raise ValueError("No valid reference audio files provided")
            
        # Create profile
        profile = VoiceProfile(
            name=name,
            reference_audio_paths=copied_paths,
            description=description,
            language=language
        )
        
        # Clone voice
        if self.backend:
            success = await self.backend.clone_voice(profile)
            if not success:
                raise RuntimeError("Failed to clone voice")
                
        # Save profile
        self.profiles[name] = profile
        if set_as_default or self.default_profile is None:
            self.default_profile = name
            
        await self._save_profiles()
        
        logger.info(f"Created voice profile: {name}")
        return profile
        
    async def delete_profile(self, name: str) -> bool:
        """Delete a voice profile."""
        if name not in self.profiles:
            return False
            
        # Remove files
        profile_dir = self.config.voices_dir / name
        if profile_dir.exists():
            import shutil
            shutil.rmtree(profile_dir)
            
        del self.profiles[name]
        
        if self.default_profile == name:
            self.default_profile = next(iter(self.profiles), None)
            
        await self._save_profiles()
        
        logger.info(f"Deleted voice profile: {name}")
        return True
        
    def get_profile(self, name: Optional[str] = None) -> Optional[VoiceProfile]:
        """Get a voice profile by name, or the default."""
        if name:
            return self.profiles.get(name)
        elif self.default_profile:
            return self.profiles.get(self.default_profile)
        return None
        
    def list_profiles(self) -> List[str]:
        """List all profile names."""
        return list(self.profiles.keys())
        
    async def synthesize(
        self,
        text: str,
        profile_name: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech using a cloned voice.
        
        Args:
            text: Text to synthesize
            profile_name: Voice profile name (uses default if None)
            
        Returns:
            Audio bytes (16-bit PCM)
        """
        if not self.backend:
            raise RuntimeError("Voice cloning not initialized")
            
        profile = self.get_profile(profile_name)
        if not profile:
            raise ValueError(f"Voice profile not found: {profile_name or 'default'}")
            
        # Clean text for speech
        text = self._prepare_text(text)
        if not text:
            return b''
            
        return await self.backend.synthesize(text, profile)
        
    def _prepare_text(self, text: str) -> str:
        """Prepare text for synthesis."""
        import re
        
        # Remove tool markers
        text = re.sub(r'\[TOOL:[^\]]+\]', '', text)
        
        # Clean whitespace
        text = ' '.join(text.split())
        
        # Remove markdown
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        return text.strip()


# Convenience function to create with config
def create_voice_cloning_tts(
    backend: str = "xtts",
    voices_dir: str = "./data/voices",
    device: str = "cuda",
    **kwargs
) -> VoiceCloningTTS:
    """
    Factory function to create VoiceCloningTTS.
    
    Example:
        tts = create_voice_cloning_tts(
            backend="xtts",
            voices_dir="./voices"
        )
        await tts.initialize()
        
        # Create voice profile from 10 seconds of audio
        await tts.create_profile(
            "my_voice",
            "path/to/reference.wav",
            description="My cloned voice"
        )
        
        # Synthesize
        audio = await tts.synthesize("Hello world!", "my_voice")
    """
    config = VoiceCloningConfig(
        backend=backend,
        voices_dir=Path(voices_dir),
        device=device,
        **kwargs
    )
    return VoiceCloningTTS(config)
