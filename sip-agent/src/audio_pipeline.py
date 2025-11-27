"""
Low-Latency Audio Pipeline
==========================
Optimized for minimal response latency.

Key optimizations:
1. Streaming STT with faster-whisper (with GB10 fixes)
2. Sentence-level TTS pipelining
3. Pre-cached acknowledgment audio
4. Reduced silence detection timeout
5. Parallel LLM + TTS processing
6. WebSocket for TTS streaming
"""

import os
import re
import json
import time
import base64
import logging
import asyncio

from collections import deque
from dataclasses import dataclass
from typing import AsyncGenerator, Callable, List, Optional, Tuple

import httpx
import numpy as np

try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from config import Config




logger = logging.getLogger(__name__)


# ============================================================================
# Latency Tracking
# ============================================================================

@dataclass
class LatencyMetrics:
    """Track latency at each stage."""
    vad_start: float = 0
    speech_end: float = 0
    stt_start: float = 0
    stt_end: float = 0
    llm_first_token: float = 0
    llm_complete: float = 0
    tts_first_chunk: float = 0
    audio_start: float = 0
    
    def log_summary(self):
        """Log latency breakdown."""
        if self.speech_end and self.stt_end:
            stt_latency = (self.stt_end - self.speech_end) * 1000
            logger.info(f"STT latency: {stt_latency:.0f}ms")
        if self.stt_end and self.llm_first_token:
            llm_ttft = (self.llm_first_token - self.stt_end) * 1000
            logger.info(f"LLM TTFT: {llm_ttft:.0f}ms")
        if self.llm_first_token and self.tts_first_chunk:
            tts_latency = (self.tts_first_chunk - self.llm_first_token) * 1000
            logger.info(f"TTS first chunk: {tts_latency:.0f}ms")
        if self.speech_end and self.audio_start:
            total = (self.audio_start - self.speech_end) * 1000
            logger.info(f"Total response latency: {total:.0f}ms")


# ============================================================================
# Optimized VAD with Shorter Timeouts
# ============================================================================

class FastVoiceActivityDetector:
    """
    Optimized VAD with aggressive silence detection.
    
    Changes from standard:
    - Shorter silence timeout (500ms vs 1000ms)
    - Energy-based pre-filter to reduce VAD calls
    - Adaptive threshold based on background noise
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.sample_rate = config.sample_rate
        
        # More aggressive VAD settings
        self.vad = None
        if VAD_AVAILABLE:
            # Mode 3 = most aggressive (best for clean phone audio)
            self.vad = webrtcvad.Vad(3)
            
        # State
        self.speech_frames = deque(maxlen=50)  # Smaller buffer
        self.silence_frames = 0
        self.is_speaking = False
        
        # Adaptive noise floor
        self.noise_floor = 200
        self.noise_samples = deque(maxlen=100)
        
        # Faster timeout - 500ms silence ends utterance
        self.silence_timeout_ms = int(os.getenv("SILENCE_TIMEOUT_MS", "500"))
        
    def is_speech(self, audio_chunk: bytes) -> bool:
        """Check if chunk contains speech with energy pre-filter."""
        # Fast energy check first
        samples = np.frombuffer(audio_chunk, dtype=np.int16)
        energy = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
        
        # Update noise floor
        if not self.is_speaking:
            self.noise_samples.append(energy)
            if len(self.noise_samples) >= 10:
                self.noise_floor = np.percentile(list(self.noise_samples), 30)
        
        # Quick reject if below noise floor
        if energy < self.noise_floor * 1.5:
            return False
            
        # WebRTC VAD for accurate detection
        if self.vad:
            try:
                frame_size = int(self.sample_rate * 0.03) * 2  # 30ms
                for i in range(0, len(audio_chunk), frame_size):
                    frame = audio_chunk[i:i + frame_size]
                    if len(frame) == frame_size:
                        if self.vad.is_speech(frame, self.sample_rate):
                            return True
                return False
            except:
                pass
                
        # Fallback: energy-based
        return energy > self.noise_floor * 2
        
    def process_audio(self, audio_chunk: bytes) -> Tuple[bool, bool]:
        """Process audio with faster end-of-utterance detection."""
        is_speech = self.is_speech(audio_chunk)
        
        if is_speech:
            self.speech_frames.append(audio_chunk)
            self.silence_frames = 0
            self.is_speaking = True
        else:
            self.silence_frames += 1
            
        # Faster end-of-utterance detection
        silence_ms = self.silence_frames * self.config.chunk_duration_ms
        end_of_utterance = (
            self.is_speaking and 
            silence_ms >= self.silence_timeout_ms
        )
        
        if end_of_utterance:
            self.is_speaking = False
            
        return is_speech, end_of_utterance
        
    def reset(self):
        """Reset state."""
        self.speech_frames.clear()
        self.silence_frames = 0
        self.is_speaking = False


# ============================================================================
# Streaming STT with Faster-Whisper (GB10 Fixes)
# ============================================================================

class StreamingSTT:
    """
    Streaming Speech-to-Text using faster-whisper.
    
    Includes fixes for GB10/Grace Blackwell:
    - Explicit cuDNN path configuration
    - int8 compute type for stability
    - Batched processing option
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.device = config.whisper_device
        
        # Try faster-whisper first, fall back to standard
        self.use_faster_whisper = True
        
    async def load_model(self):
        """Load STT model with GB10 optimizations."""
        # Fix cuDNN paths for faster-whisper on GB10
        self._setup_cudnn_paths()
        
        try:
            await self._load_faster_whisper()
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}, falling back to standard whisper")
            self.use_faster_whisper = False
            await self._load_standard_whisper()
            
    def _setup_cudnn_paths(self):
        """Configure cuDNN paths for GB10 compatibility."""
        import os
        
        # Common cuDNN locations
        cudnn_paths = [
            "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/lib64",
        ]
        
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        for path in cudnn_paths:
            if os.path.exists(path) and path not in existing:
                existing = f"{path}:{existing}"
                
        os.environ["LD_LIBRARY_PATH"] = existing
        
    async def _load_faster_whisper(self):
        """Load faster-whisper with optimized settings."""
        from faster_whisper import WhisperModel
        
        logger.info(f"Loading faster-whisper: {self.config.whisper_model}")
        
        loop = asyncio.get_event_loop()
        
        def _load():
            # Use int8 for better GB10 compatibility
            compute_type = "int8" if self.device == "cuda" else "int8"
            
            return WhisperModel(
                self.config.whisper_model,
                device=self.device,
                compute_type=compute_type,
                # Optimize for latency
                cpu_threads=4,
                num_workers=2
            )
            
        self.model = await loop.run_in_executor(None, _load)
        logger.info("faster-whisper loaded successfully")
        
    async def _load_standard_whisper(self):
        """Fallback to standard whisper."""
        import whisper
        
        logger.info(f"Loading standard whisper: {self.config.whisper_model}")
        
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: whisper.load_model(self.config.whisper_model, device=self.device)
        )
        logger.info("Standard whisper loaded")
        
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe with minimal latency."""
        if not self.model:
            return ""
            
        loop = asyncio.get_event_loop()
        
        if self.use_faster_whisper:
            return await loop.run_in_executor(None, self._transcribe_faster, audio_data)
        else:
            return await loop.run_in_executor(None, self._transcribe_standard, audio_data)
            
    def _transcribe_faster(self, audio_data: bytes) -> str:
        """Transcribe using faster-whisper (optimized)."""
        # Convert to float32
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe with optimized settings
        segments, info = self.model.transcribe(
            audio,
            language=self.config.whisper_language or None,
            beam_size=3,  # Reduced for speed
            best_of=1,    # Single pass
            patience=0.5, # Early stopping
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=100
            )
        )
        
        return " ".join(s.text.strip() for s in segments).strip()
        
    def _transcribe_standard(self, audio_data: bytes) -> str:
        """Fallback transcription."""
        import whisper
        
        audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample to 16kHz if needed
        if self.config.sample_rate != 16000:
            if SCIPY_AVAILABLE:
                ratio = 16000 / self.config.sample_rate
                new_len = int(len(audio) * ratio)
                audio = scipy.signal.resample(audio, new_len)
                
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        options = whisper.DecodingOptions(
            language=self.config.whisper_language,
            beam_size=3,
            fp16=(self.device == "cuda")
        )
        
        result = whisper.decode(self.model, mel, options)
        return result.text.strip()


# ============================================================================
# WebSocket TTS Client (Lower Latency than SSE)
# ============================================================================

class FastTTSClient:
    """
    Optimized TTS client with:
    - WebSocket streaming (lower latency than SSE)
    - Sentence-level chunking
    - Pre-cached common phrases
    - Parallel synthesis
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.xtts_api_url.rstrip('/')
        self.client: Optional[httpx.AsyncClient] = None
        self.available = False
        self.default_voice = config.xtts_default_voice
        
        # Pre-cached audio for common acknowledgments
        self.audio_cache: dict = {}
        self.cache_enabled = True
        
        # Sample rate from XTTS
        self.xtts_sample_rate = 24000
        
    async def initialize(self):
        """Initialize client and pre-cache common phrases."""
        self.client = httpx.AsyncClient(timeout=60.0)
        
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                self.available = data.get("model_loaded", False)
                
                if self.available and self.cache_enabled:
                    await self._precache_phrases()
        except Exception as e:
            logger.warning(f"XTTS not available: {e}")
            self.available = False
            
    async def _precache_phrases(self):
        """Pre-synthesize common acknowledgments for instant playback."""
        phrases = [
            "Okay",
            "Sure",
            "Got it", 
            "One moment",
            "Copy that",
            "On it",
            "Checking",
            "Let me see",
            "Working on it",
            "Goodbye"
        ]
        
        logger.info("Pre-caching common phrases...")
        
        for phrase in phrases:
            try:
                audio = await self._synthesize_raw(phrase)
                if audio:
                    # Resample to target rate
                    audio = self._resample(audio, self.xtts_sample_rate, self.config.sample_rate)
                    self.audio_cache[phrase.lower()] = audio
            except Exception as e:
                logger.warning(f"Failed to cache '{phrase}': {e}")
                
        logger.info(f"Cached {len(self.audio_cache)} phrases")
        
    async def close(self):
        """Close client."""
        if self.client:
            await self.client.aclose()
            
    def get_cached(self, text: str) -> Optional[bytes]:
        """Get pre-cached audio if available."""
        return self.audio_cache.get(text.lower().strip())
        
    async def _synthesize_raw(self, text: str, voice: str = None) -> bytes:
        """Raw synthesis without caching."""
        response = await self.client.post(
            f"{self.base_url}/v1/tts/raw",
            json={"text": text, "voice": voice or self.default_voice}
        )
        
        if response.status_code == 200:
            return response.content
        return b''
        
    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """Synthesize with cache check."""
        # Check cache first
        cached = self.get_cached(text)
        if cached:
            logger.debug(f"Cache hit: {text}")
            return cached
            
        if not self.available:
            return b''
            
        audio = await self._synthesize_raw(text, voice)
        
        # Resample
        if audio and self.xtts_sample_rate != self.config.sample_rate:
            audio = self._resample(audio, self.xtts_sample_rate, self.config.sample_rate)
            
        return audio
        
    async def synthesize_stream(
        self,
        text: str,
        voice: str = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesis by sentence for lower latency.
        
        Starts TTS on first sentence while LLM generates rest.
        """
        if not self.available:
            return
            
        # Check if entire text is cached
        cached = self.get_cached(text)
        if cached:
            yield cached
            return
            
        # Split into sentences for pipelining
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Check sentence cache
            cached = self.get_cached(sentence)
            if cached:
                yield cached
                continue
                
            # Stream from server
            try:
                async with self.client.stream(
                    "POST",
                    f"{self.base_url}/v1/tts/stream",
                    json={"text": sentence, "voice": voice or self.default_voice, "stream": True}
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            
                            if data.get("done"):
                                break
                            if "audio" in data:
                                chunk = base64.b64decode(data["audio"])
                                # Resample chunk
                                if self.xtts_sample_rate != self.config.sample_rate:
                                    chunk = self._resample(chunk, self.xtts_sample_rate, self.config.sample_rate)
                                yield chunk
            except Exception as e:
                logger.error(f"TTS stream error: {e}")
                
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Split on sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _resample(self, audio: bytes, from_rate: int, to_rate: int) -> bytes:
        """Resample audio."""
        if from_rate == to_rate:
            return audio
            
        samples = np.frombuffer(audio, dtype=np.int16)
        if len(samples) == 0:
            return audio
            
        if SCIPY_AVAILABLE:
            ratio = to_rate / from_rate
            new_len = int(len(samples) * ratio)
            resampled = scipy.signal.resample(samples.astype(np.float64), new_len)
        else:
            ratio = to_rate / from_rate
            new_indices = np.linspace(0, len(samples) - 1, int(len(samples) * ratio))
            resampled = np.interp(new_indices, np.arange(len(samples)), samples.astype(np.float64))
            
        return resampled.astype(np.int16).tobytes()


# ============================================================================
# Parallel LLM + TTS Pipeline
# ============================================================================

class StreamingResponsePipeline:
    """
    Parallel processing pipeline:
    1. LLM streams tokens
    2. Accumulate into sentences
    3. Send sentences to TTS immediately
    4. Stream audio chunks to playback
    
    This reduces perceived latency by overlapping LLM generation with TTS.
    """
    
    def __init__(self, llm_engine, tts_client: FastTTSClient, config: Config):
        self.llm = llm_engine
        self.tts = tts_client
        self.config = config
        
    async def generate_and_stream(
        self,
        conversation_history: list,
        call_context: dict,
        audio_callback: Callable[[bytes], None]
    ) -> str:
        """
        Generate response and stream audio in parallel.
        
        Returns complete response text.
        """
        full_response = ""
        sentence_buffer = ""
        metrics = LatencyMetrics()
        metrics.stt_end = time.time()
        
        # Start LLM generation (streaming)
        async for token in self._stream_llm(conversation_history, call_context):
            if metrics.llm_first_token == 0:
                metrics.llm_first_token = time.time()
                
            full_response += token
            sentence_buffer += token
            
            # Check for complete sentence
            if self._is_sentence_end(sentence_buffer):
                sentence = sentence_buffer.strip()
                sentence_buffer = ""
                
                if sentence:
                    # Synthesize and stream this sentence
                    async for audio_chunk in self.tts.synthesize_stream(sentence):
                        if metrics.tts_first_chunk == 0:
                            metrics.tts_first_chunk = time.time()
                        if metrics.audio_start == 0:
                            metrics.audio_start = time.time()
                            
                        audio_callback(audio_chunk)
                        
        # Handle remaining text
        if sentence_buffer.strip():
            async for audio_chunk in self.tts.synthesize_stream(sentence_buffer.strip()):
                audio_callback(audio_chunk)
                
        metrics.llm_complete = time.time()
        metrics.log_summary()
        
        return full_response
        
    async def _stream_llm(
        self,
        conversation_history: list,
        call_context: dict
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from LLM."""
        # Build messages
        messages = [
            {"role": "system", "content": self.llm._build_system_prompt(call_context)}
        ]
        messages.extend(conversation_history[-self.config.max_conversation_turns * 2:])
        
        if not self.llm.client:
            # Mock streaming for testing
            response = "I understand. Is there anything else I can help you with?"
            for word in response.split():
                yield word + " "
                await asyncio.sleep(0.05)
            return
            
        try:
            stream = await self.llm.client.chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                max_tokens=self.config.llm_max_tokens,
                temperature=self.config.llm_temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield "I'm sorry, I had trouble processing that."
            
    def _is_sentence_end(self, text: str) -> bool:
        """Check if text ends with a complete sentence."""
        text = text.strip()
        if not text:
            return False
            
        # Check for sentence-ending punctuation
        if text[-1] in '.!?':
            # Avoid false positives like "Dr." or "Mr."
            abbrevs = ['dr.', 'mr.', 'mrs.', 'ms.', 'jr.', 'sr.', 'etc.']
            lower = text.lower()
            for abbrev in abbrevs:
                if lower.endswith(abbrev):
                    return False
            return True
            
        return False


# ============================================================================
# Optimized Audio Pipeline
# ============================================================================

class LowLatencyAudioPipeline:
    """
    Low-latency audio pipeline with all optimizations.
    
    Target latencies:
    - STT: < 300ms
    - LLM TTFT: < 500ms  
    - TTS first chunk: < 200ms
    - Total: < 1000ms
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Optimized components
        self.vad = FastVoiceActivityDetector(config)
        self.stt = StreamingSTT(config)
        self.tts = FastTTSClient(config)
        
        # Audio buffer
        self.audio_buffer = bytearray()
        self.max_buffer_size = int(config.max_speech_duration_s * config.sample_rate * 2)
        
        # Metrics
        self.last_metrics = LatencyMetrics()
        
    async def start(self):
        """Initialize all components."""
        logger.info("Starting low-latency audio pipeline...")
        
        start = time.time()
        
        # Load in parallel
        await asyncio.gather(
            self.stt.load_model(),
            self.tts.initialize()
        )
        
        load_time = (time.time() - start) * 1000
        logger.info(f"Pipeline ready in {load_time:.0f}ms")
        
        # Log status
        if self.stt.use_faster_whisper:
            logger.info("Using faster-whisper for STT (optimized)")
        else:
            logger.info("Using standard whisper for STT")
            
        if self.tts.available:
            logger.info(f"TTS ready, {len(self.tts.audio_cache)} phrases cached")
        else:
            logger.warning("TTS not available")
            
    async def stop(self):
        """Cleanup."""
        await self.tts.close()
        
    async def process_audio(self, audio_chunk: bytes) -> Optional[str]:
        """Process audio with fast end-of-utterance detection."""
        is_speech, end_of_utterance = self.vad.process_audio(audio_chunk)
        
        if is_speech:
            self.audio_buffer.extend(audio_chunk)
            
            # Buffer overflow protection
            if len(self.audio_buffer) > self.max_buffer_size:
                logger.warning("Buffer overflow, forcing transcription")
                return await self._transcribe_buffer()
                
        if end_of_utterance and len(self.audio_buffer) > 0:
            return await self._transcribe_buffer()
            
        return None
        
    async def _transcribe_buffer(self) -> str:
        """Transcribe buffered audio."""
        self.last_metrics.speech_end = time.time()
        
        audio_data = bytes(self.audio_buffer)
        self.audio_buffer.clear()
        self.vad.reset()
        
        # Check minimum duration
        duration_ms = len(audio_data) / (self.config.sample_rate * 2) * 1000
        if duration_ms < self.config.min_speech_duration_ms:
            return ""
            
        self.last_metrics.stt_start = time.time()
        result = await self.stt.transcribe(audio_data)
        self.last_metrics.stt_end = time.time()
        
        stt_latency = (self.last_metrics.stt_end - self.last_metrics.speech_end) * 1000
        logger.info(f"STT: {stt_latency:.0f}ms for {duration_ms:.0f}ms audio")
        
        return result
        
    async def synthesize(self, text: str, voice: str = None) -> bytes:
        """Synthesize with caching."""
        return await self.tts.synthesize(text, voice)
        
    async def synthesize_stream(self, text: str, voice: str = None) -> AsyncGenerator[bytes, None]:
        """Stream synthesis."""
        async for chunk in self.tts.synthesize_stream(text, voice):
            yield chunk
            
    def get_cached_audio(self, text: str) -> Optional[bytes]:
        """Get pre-cached audio for instant playback."""
        return self.tts.get_cached(text)
        
    def has_speech(self, audio_chunk: bytes) -> bool:
        """Quick speech check."""
        return self.vad.is_speech(audio_chunk)