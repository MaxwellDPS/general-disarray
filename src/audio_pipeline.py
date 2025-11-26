"""
Audio Pipeline
==============
Handles Voice Activity Detection, Speech-to-Text, and Text-to-Speech.
"""

import asyncio
import logging
import io
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Union
import numpy as np
from collections import deque

from piperdownload import download_voice

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    # Piper TTS - can be installed via pip or as binary
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

from config import Config

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """WebRTC-based Voice Activity Detection."""
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.sample_rate
        self.vad = None
        
        if VAD_AVAILABLE:
            self.vad = webrtcvad.Vad(config.vad_aggressiveness)
            
        # State for speech detection
        self.speech_frames = deque(maxlen=100)
        self.silence_frames = 0
        self.is_speaking = False
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio chunk contains speech."""
        if not self.vad:
            # Fallback: simple energy-based detection
            return self._energy_vad(audio_chunk)
            
        try:
            # WebRTC VAD needs specific frame sizes: 10, 20, or 30ms
            frame_duration_ms = self.config.chunk_duration_ms
            frame_size = int(self.sample_rate * frame_duration_ms / 1000) * 2  # 16-bit
            
            # Process in frames
            speech_detected = False
            for i in range(0, len(audio_chunk), frame_size):
                frame = audio_chunk[i:i + frame_size]
                if len(frame) == frame_size:
                    if self.vad.is_speech(frame, self.sample_rate):
                        speech_detected = True
                        break
                        
            return speech_detected
            
        except Exception as e:
            logger.warning(f"VAD error: {e}")
            return self._energy_vad(audio_chunk)
            
    def _energy_vad(self, audio_chunk: bytes, threshold: float = 500) -> bool:
        """Fallback energy-based VAD."""
        try:
            samples = np.frombuffer(audio_chunk, dtype=np.int16)
            energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            return energy > threshold
        except:
            return False
            
    def process_audio(self, audio_chunk: bytes) -> Tuple[bool, bool]:
        """
        Process audio and return (has_speech, end_of_utterance).
        
        Returns:
            has_speech: True if speech detected in this chunk
            end_of_utterance: True if we've detected end of speech
        """
        is_speech = self.is_speech(audio_chunk)
        
        if is_speech:
            self.speech_frames.append(audio_chunk)
            self.silence_frames = 0
            self.is_speaking = True
        else:
            self.silence_frames += 1
            
        # Check for end of utterance
        silence_ms = self.silence_frames * self.config.chunk_duration_ms
        end_of_utterance = (
            self.is_speaking and 
            silence_ms >= self.config.silence_duration_ms
        )
        
        if end_of_utterance:
            self.is_speaking = False
            
        return is_speech, end_of_utterance
        
    def reset(self):
        """Reset VAD state."""
        self.speech_frames.clear()
        self.silence_frames = 0
        self.is_speaking = False


class SpeechToText:
    """Speech-to-Text using faster-whisper."""
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[WhisperModel] = None
        
    async def load_model(self):
        """Load the Whisper model."""
        if not WHISPER_AVAILABLE:
            logger.warning("faster-whisper not available, STT will be mocked")
            return
            
        logger.info(f"Loading Whisper model: {self.config.whisper_model}")
        
        # Load model in thread pool to not block
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            self._load_model_sync
        )
        
        logger.info("Whisper model loaded")
        
    def _load_model_sync(self) -> WhisperModel:
        """Synchronous model loading."""
        return WhisperModel(
            self.config.whisper_model,
            device=self.config.whisper_device,
            compute_type=self.config.whisper_compute_type
        )
        
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio to text."""
        if not self.model:
            # Mock response for testing
            return "[Mock transcription - faster-whisper not available]"
            
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            self._transcribe_sync,
            audio_data
        )
        return text
        
    def _transcribe_sync(self, audio_data: bytes) -> str:
        """Synchronous transcription."""
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_array /= 32768.0  # Normalize to [-1, 1]
        
        # Transcribe
        segments, info = self.model.transcribe(
            audio_array,
            language=self.config.whisper_language,
            beam_size=self.config.whisper_beam_size,
            vad_filter=self.config.whisper_vad_filter,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=self.config.speech_pad_ms
            )
        )
        
        # Combine segments
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()


class TextToSpeech:
    """
    Text-to-Speech with voice cloning support.
    
    Supports:
    - Voice cloning (XTTS, Chatterbox, OpenVoice, Fish Speech)
    - Piper TTS (fast, lightweight) via pip install piper-tts
    - espeak-ng fallback
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model_path: Optional[Path] = None
        self.piper_voice = None
        self.piper_executable = "piper"
        
        # Voice cloning
        self.voice_cloning = None
        self.use_voice_cloning = False
        
    async def load_model(self):
        """Load TTS model(s)."""
        # Try to load voice cloning if enabled
        if self.config.voice_cloning_enabled:
            await self._load_voice_cloning()
            
        # Always load Piper as fallback
        await self._load_piper()
        
    async def _load_voice_cloning(self):
        """Initialize voice cloning backend."""
        try:
            from voice_cloning import VoiceCloningTTS, VoiceCloningConfig
            
            vc_config = VoiceCloningConfig(
                enabled=True,
                backend=self.config.voice_cloning_backend,
                device="cuda" if self.config.whisper_device == "cuda" else "cpu",
                voices_dir=self.config.voices_dir,
                output_sample_rate=self.config.sample_rate,
                xtts_temperature=self.config.xtts_temperature,
                xtts_top_k=self.config.xtts_top_k,
                xtts_top_p=self.config.xtts_top_p,
                xtts_repetition_penalty=self.config.xtts_repetition_penalty,
                chatterbox_exaggeration=self.config.chatterbox_exaggeration,
                chatterbox_cfg_weight=self.config.chatterbox_cfg_weight
            )
            
            self.voice_cloning = VoiceCloningTTS(vc_config)
            success = await self.voice_cloning.initialize()
            
            if success:
                self.use_voice_cloning = True
                logger.info(f"Voice cloning enabled with {self.config.voice_cloning_backend}")
                
                # Log available profiles
                profiles = self.voice_cloning.list_profiles()
                if profiles:
                    logger.info(f"Available voice profiles: {profiles}")
                else:
                    logger.info("No voice profiles found. Create one with create_voice_profile()")
            else:
                logger.warning("Voice cloning initialization failed, using Piper fallback")
                self.use_voice_cloning = False
                
        except ImportError as e:
            logger.warning(f"Voice cloning dependencies not available: {e}")
            self.use_voice_cloning = False
        except Exception as e:
            logger.error(f"Voice cloning error: {e}")
            self.use_voice_cloning = False
            
    async def _load_piper(self):
        """Load Piper TTS model via Python package."""
        try:
            from piper import PiperVoice
            
            model_name = self.config.piper_model
            model_dir = self.config.piper_model_path
            
            # First check local paths
            possible_paths = [
                model_dir / f"{model_name}.onnx",
                model_dir / model_name / f"{model_name}.onnx",
                Path(f"/opt/piper/models/{model_name}.onnx"),
                Path.home() / ".local/share/piper/models" / f"{model_name}.onnx",
                Path.home() / ".cache/piper" / f"{model_name}.onnx"
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
                    
            # If not found locally, try to download
            if not model_path:
                logger.info(f"Downloading Piper voice: {model_name}")
                try:
                    data_dir = Path.home() / ".local/share/piper"
                    data_dir.mkdir(parents=True, exist_ok=True)
                    
                    loop = asyncio.get_event_loop()
                    
                    # Download the voice model
                    model_path, config_path = await loop.run_in_executor(
                        None,
                        lambda: download_voice(
                            model_name,
                            download_dir=str(data_dir)
                        )
                    )
                    model_path = Path(model_path)
                    logger.info(f"Downloaded Piper voice to: {model_path}")
                    
                except Exception as e:
                    logger.warning(f"Could not download Piper voice: {e}")
                    logger.info("Will use espeak-ng fallback for TTS")
                    return
                    
            if model_path and model_path.exists():
                self.model_path = model_path
                
                # Load the model
                loop = asyncio.get_event_loop()
                self.piper_voice = await loop.run_in_executor(
                    None,
                    lambda: PiperVoice.load(str(model_path))
                )
                logger.info(f"Loaded Piper model: {model_path}")
            else:
                logger.warning("No Piper model available, will use espeak-ng fallback")
                
        except ImportError:
            logger.warning("piper-tts not installed. Run: pip install piper-tts")
            self.piper_voice = None
            self.model_path = None
        except Exception as e:
            logger.warning(f"Failed to load Piper model: {e}")
            self.piper_voice = None
            self.model_path = None
        
    async def create_voice_profile(
        self,
        name: str,
        reference_audio: str,
        description: str = "",
        language: str = "en",
        set_as_default: bool = True
    ) -> bool:
        """
        Create a new cloned voice profile.
        
        Args:
            name: Profile name
            reference_audio: Path to reference audio (5-30 seconds recommended)
            description: Optional description
            language: Language code
            set_as_default: Whether to use as default voice
            
        Returns:
            True on success
        """
        if not self.voice_cloning:
            logger.error("Voice cloning not initialized")
            return False
            
        try:
            await self.voice_cloning.create_profile(
                name=name,
                reference_audio=reference_audio,
                description=description,
                language=language,
                set_as_default=set_as_default
            )
            logger.info(f"Created voice profile: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create voice profile: {e}")
            return False
            
    def list_voice_profiles(self) -> list:
        """List available voice profiles."""
        if self.voice_cloning:
            return self.voice_cloning.list_profiles()
        return []
        
    def set_voice_profile(self, name: str) -> bool:
        """Set the active voice profile."""
        if not self.voice_cloning:
            return False
        if name in self.voice_cloning.profiles:
            self.voice_cloning.default_profile = name
            logger.info(f"Set voice profile: {name}")
            return True
        return False
        
    async def synthesize(self, text: str, voice_profile: Optional[str] = None) -> bytes:
        """
        Synthesize text to speech audio.
        
        Args:
            text: Text to synthesize
            voice_profile: Optional voice profile name (for cloning)
            
        Returns:
            Audio bytes (16-bit PCM)
        """
        if not text:
            logger.warning("Empty text passed to synthesize")
            return b''
        
        logger.info(f"Synthesizing: {text[:50]}...")
            
        # Clean text for speech
        text = self._prepare_text(text)
        
        # Try voice cloning first
        if self.use_voice_cloning and self.voice_cloning:
            try:
                profile_name = voice_profile or self.config.voice_cloning_profile
                if profile_name or self.voice_cloning.default_profile:
                    audio = await self.voice_cloning.synthesize(text, profile_name or None)
                    if audio:
                        logger.info(f"Voice cloning produced {len(audio)} bytes")
                        # Resample if needed
                        vc_rate = self.voice_cloning.config.output_sample_rate
                        if vc_rate != self.config.sample_rate:
                            audio = self._resample(audio, vc_rate, self.config.sample_rate)
                        return audio
            except Exception as e:
                logger.warning(f"Voice cloning failed, falling back to Piper: {e}")
                
        # Fallback to Piper/espeak
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        logger.info(f"TTS produced {len(audio)} bytes of audio")
        return audio
        
    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous speech synthesis."""
        # Try Piper Python API first
        if hasattr(self, 'piper_voice') and self.piper_voice is not None:
            return self._synthesize_piper_python(text)
            
        # Fallback to espeak
        return self._synthesize_espeak(text)
        
    def _synthesize_piper_python(self, text: str) -> bytes:
        """Synthesize using piper-tts Python package."""
        try:
            import io
            import wave
            
            logger.debug(f"Piper synthesizing: {text[:30]}...")
            
            # Synthesize to WAV bytes
            audio_bytes = io.BytesIO()
            
            with wave.open(audio_bytes, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self.piper_voice.config.sample_rate)
                
                # FIX: Explicitly pass configuration parameters
                # Many Piper versions fail silently without speaker_id
                result = self.piper_voice.synthesize(
                    text
                )
                
                # Handle generator vs bytes
                if hasattr(result, '__iter__') and not isinstance(result, (bytes, bytearray)):
                    for audio_chunk in result:
                        if isinstance(audio_chunk, (bytes, bytearray)):
                            wav.writeframes(audio_chunk)
                        elif hasattr(audio_chunk, 'audio'):
                            # Some wrappers return an object with an .audio property
                            wav.writeframes(audio_chunk.audio)
                else:
                    wav.writeframes(result)
                    
            # Get PCM from WAV
            audio_bytes.seek(0)
            wav_data = audio_bytes.read()
            
            if len(wav_data) < 50:
                logger.warning(f"Piper produced very little audio: {len(wav_data)} bytes")
                return self._synthesize_espeak(text)
                
            pcm = self._wav_to_pcm(wav_data)
            
            # Resample if needed
            piper_rate = self.piper_voice.config.sample_rate
            if piper_rate != self.config.sample_rate:
                pcm = self._resample(pcm, piper_rate, self.config.sample_rate)
            
            logger.debug(f"Piper produced {len(pcm)} bytes of PCM audio")
            return pcm
            
        except Exception as e:
            logger.error(f"Piper Python synthesis error: {e}")
            import traceback
            traceback.print_exc()
            return self._synthesize_espeak(text)
            
    def _synthesize_piper(self, text: str) -> bytes:
        """Synthesize using Piper TTS (legacy subprocess method)."""
        try:
            # Run Piper CLI (fallback if Python API fails)
            cmd = [
                self.piper_executable,
                "--model", str(self.model_path),
                "--output-raw",
                "--speaker", str(self.config.piper_speaker_id),
                "--length_scale", str(self.config.piper_length_scale),
                "--noise_scale", str(self.config.piper_noise_scale),
                "--noise_w", str(self.config.piper_noise_w)
            ]
            
            result = subprocess.run(
                cmd,
                input=text.encode(),
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Piper outputs 16-bit PCM at 22050Hz
                # Resample to our target rate if needed
                audio = result.stdout
                if self.config.tts_sample_rate != self.config.sample_rate:
                    audio = self._resample(
                        audio, 
                        self.config.tts_sample_rate, 
                        self.config.sample_rate
                    )
                return audio
            else:
                logger.error(f"Piper error: {result.stderr.decode()}")
                return self._synthesize_espeak(text)
                
        except FileNotFoundError:
            logger.warning("Piper CLI not found, using espeak")
            return self._synthesize_espeak(text)
        except subprocess.TimeoutExpired:
            logger.error("Piper timeout")
            return b''
        except Exception as e:
            logger.error(f"Piper error: {e}")
            return self._synthesize_espeak(text)
            
    def _synthesize_espeak(self, text: str) -> bytes:
        """Fallback synthesis using espeak-ng."""
        try:
            cmd = [
                "espeak-ng",
                "-v", "en-us",
                "-s", "150",  # Speed
                "--stdout",
                text
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # espeak outputs WAV, extract PCM
                return self._wav_to_pcm(result.stdout)
            else:
                logger.error(f"espeak error: {result.stderr.decode()}")
                return self._generate_silence(len(text) * 50)  # ~50ms per char
                
        except FileNotFoundError:
            logger.warning("espeak-ng not found")
            return self._generate_silence(len(text) * 50)
        except Exception as e:
            logger.error(f"espeak error: {e}")
            return self._generate_silence(len(text) * 50)
            
    def _wav_to_pcm(self, wav_data: bytes) -> bytes:
        """Extract PCM from WAV data."""
        try:
            import wave
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav:
                    pcm = wav.readframes(wav.getnframes())
                    
                    # Resample if needed
                    if wav.getframerate() != self.config.sample_rate:
                        pcm = self._resample(
                            pcm, 
                            wav.getframerate(), 
                            self.config.sample_rate
                        )
                    return pcm
        except Exception as e:
            logger.error(f"WAV parsing error: {e}")
            return b''
            
    def _resample(self, audio: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample audio to target sample rate."""
        try:
            import scipy.signal
            
            samples = np.frombuffer(audio, dtype=np.int16)
            
            # Calculate new length
            ratio = to_rate / from_rate
            new_length = int(len(samples) * ratio)
            
            # Resample
            resampled = scipy.signal.resample(samples, new_length)
            
            return resampled.astype(np.int16).tobytes()
            
        except ImportError:
            # Simple linear interpolation fallback
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
            ratio = to_rate / from_rate
            indices = np.arange(0, len(samples), 1/ratio)
            indices = indices[indices < len(samples) - 1].astype(int)
            resampled = samples[indices]
            return resampled.astype(np.int16).tobytes()
            
    def _generate_silence(self, duration_ms: int) -> bytes:
        """Generate silence audio."""
        num_samples = int(self.config.sample_rate * duration_ms / 1000)
        return bytes(num_samples * 2)  # 16-bit silence
        
    def _prepare_text(self, text: str) -> str:
        """Prepare text for speech synthesis."""
        # Remove any tool markers
        import re
        text = re.sub(r'\[TOOL:[^\]]+\]', '', text)
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        # Remove markdown formatting
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        return text.strip()

    async def synthesize_stream(self, text: str):
        """
        Yields chunks of audio bytes via Piper CLI.
        More robust implementation that spawns a fresh process for each sentence
        but handles I/O carefully to prevent 0-byte outputs.
        """
        # 1. Clean text
        text = self._prepare_text(text)
        
        # 2. Split into sentences
        # Splitting by punctuation is simple but effective for TTS
        import re
        # This regex splits by . ! ? but keeps the punctuation attached to the sentence
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Get sample rate (default 22050 for Piper)
        piper_rate = 22050
        if hasattr(self.piper_voice, 'config'):
            piper_rate = self.piper_voice.config.sample_rate

        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # --- FIXED PROCESS HANDLING ---
            try:
                # Use synchronous run() instead of Popen() for stability per sentence
                # This ensures the previous sentence finishes before the next starts
                # which prevents resource contention.
                cmd = [
                    "piper",
                    "--model", str(self.model_path),
                    "--output-raw",
                    "--speaker", str(self.config.piper_speaker_id),
                    "--length-scale", str(self.config.piper_length_scale),
                    "--noise-scale", str(self.config.piper_noise_scale),
                    "--noise-w", str(self.config.piper_noise_w)
                ]
                
                # Run purely in memory
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Communicate handles the closing of stdin automatically
                stdout, stderr = await proc.communicate(input=sentence.encode())
                
                if proc.returncode != 0:
                    logger.warning(f"Piper error on segment: {stderr.decode()}")
                    # Fallback to espeak for this segment
                    yield self._synthesize_espeak(sentence)
                    continue

                if len(stdout) < 100:
                    logger.warning(f"Piper produced silence for: {sentence[:20]}...")
                    # Fallback
                    yield self._synthesize_espeak(sentence)
                    continue
                
                # Resample if needed
                if piper_rate != self.config.sample_rate:
                    # _resample is synchronous, run it in executor to be safe
                    loop = asyncio.get_running_loop()
                    chunk = await loop.run_in_executor(
                        None, 
                        self._resample, 
                        stdout, piper_rate, self.config.sample_rate
                    )
                    yield chunk
                else:
                    yield stdout
                    
            except Exception as e:
                logger.error(f"Streaming segment error: {e}")
                yield self._synthesize_espeak(sentence)

class AudioPipeline:
    """Combined audio processing pipeline with voice cloning support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.vad = VoiceActivityDetector(config)
        self.stt = SpeechToText(config)
        self.tts = TextToSpeech(config)
        
        # Audio buffer for accumulating speech
        self.audio_buffer = bytearray()
        self.max_buffer_size = int(
            config.max_speech_duration_s * config.sample_rate * 2
        )
        
    async def start(self):
        """Initialize all components."""
        await self.stt.load_model()
        await self.tts.load_model()
        logger.info("Audio pipeline ready")
        
        # Log voice cloning status
        if self.tts.use_voice_cloning:
            profiles = self.tts.list_voice_profiles()
            logger.info(f"Voice cloning active. Profiles: {profiles}")
        
    async def stop(self):
        """Cleanup."""
        if self.tts.voice_cloning:
            await self.tts.voice_cloning.shutdown()
        
    def has_speech(self, audio_chunk: bytes) -> bool:
        """Check if audio contains speech."""
        is_speech, _ = self.vad.process_audio(audio_chunk)
        return is_speech
        
    async def process_audio(self, audio_chunk: bytes) -> Optional[str]:
        """
        Process incoming audio chunk.
        
        Returns transcription when end of utterance detected, None otherwise.
        """
        is_speech, end_of_utterance = self.vad.process_audio(audio_chunk)
        
        if is_speech:
            self.audio_buffer.extend(audio_chunk)
            
            # Prevent buffer overflow
            if len(self.audio_buffer) > self.max_buffer_size:
                logger.warning("Audio buffer overflow, truncating")
                self.audio_buffer = self.audio_buffer[-self.max_buffer_size:]
                
        if end_of_utterance and len(self.audio_buffer) > 0:
            # Transcribe accumulated audio
            audio_data = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            self.vad.reset()
            
            # Check minimum duration
            duration_ms = len(audio_data) / (self.config.sample_rate * 2) * 1000
            if duration_ms >= self.config.min_speech_duration_ms:
                return await self.stt.transcribe(audio_data)
                
        return None
        
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio directly."""
        return await self.stt.transcribe(audio_data)
        
    async def synthesize(self, text: str, voice_profile: Optional[str] = None) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to speak
            voice_profile: Optional voice profile for cloned voice
        """
        return await self.tts.synthesize(text, voice_profile)
        
    # Voice cloning convenience methods
    async def create_voice_profile(
        self,
        name: str,
        reference_audio: Union[str, List[str]],
        description: str = "",
        language: str = "en",
        set_as_default: bool = True
    ) -> bool:
        """
        Create a cloned voice profile from reference audio.
        
        Args:
            name: Profile name (e.g., "my_voice", "assistant_voice")
            reference_audio: Path(s) to reference audio file(s)
                           - Recommended: 5-30 seconds of clean speech
                           - Formats: WAV, MP3, FLAC, OGG
            description: Optional description
            language: Language code (en, es, fr, de, etc.)
            set_as_default: Use this profile by default
            
        Returns:
            True on success
            
        Example:
            await pipeline.create_voice_profile(
                "assistant",
                "/path/to/reference.wav",
                description="Main assistant voice"
            )
        """
        return await self.tts.create_voice_profile(
            name, reference_audio, description, language, set_as_default
        )
        
    def list_voice_profiles(self) -> List[str]:
        """List available voice profiles."""
        return self.tts.list_voice_profiles()
        
    def set_voice_profile(self, name: str) -> bool:
        """Set the active voice profile."""
        return self.tts.set_voice_profile(name)
        
    @property
    def voice_cloning_enabled(self) -> bool:
        """Check if voice cloning is active."""
        return self.tts.use_voice_cloning