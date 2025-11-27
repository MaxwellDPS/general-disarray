#!/usr/bin/env python3
"""
XTTS2 Streaming TTS Server
==========================
A FastAPI server providing text-to-speech using Coqui XTTS v2.

Features:
- Voice cloning from reference audio
- Streaming synthesis via SSE
- Voice profile management
- Built-in default voice (no profile required)

API Endpoints:
- GET  /health              - Health check
- POST /v1/tts              - Synthesize (returns WAV)
- POST /v1/tts/stream       - Streaming synthesis (SSE)
- POST /v1/tts/raw          - Raw PCM bytes
- GET  /v1/voices           - List voice profiles
- POST /v1/voices           - Create voice profile
- DELETE /v1/voices/{name}  - Delete profile
- POST /v1/voices/{name}/default - Set default
"""

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class XTTSConfig:
    """XTTS server configuration."""
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    voices_dir: Path = field(default_factory=lambda: Path(os.getenv("VOICES_DIR", "/app/data/voices")))
    
    # Synthesis parameters
    temperature: float = float(os.getenv("XTTS_TEMPERATURE", "0.7"))
    top_k: int = int(os.getenv("XTTS_TOP_K", "50"))
    top_p: float = float(os.getenv("XTTS_TOP_P", "0.85"))
    repetition_penalty: float = float(os.getenv("XTTS_REPETITION_PENALTY", "2.0"))
    
    # Streaming
    stream_chunk_size: int = int(os.getenv("XTTS_CHUNK_SIZE", "20"))
    
    # Conditioning
    gpt_cond_len: int = int(os.getenv("XTTS_GPT_COND_LEN", "30"))
    max_ref_len: int = int(os.getenv("XTTS_MAX_REF_LEN", "60"))
    
    # Output
    output_sample_rate: int = 24000
    
    # Server
    host: str = os.getenv("XTTS_HOST", "0.0.0.0")
    port: int = int(os.getenv("XTTS_PORT", "8001"))
    
    # Default language
    default_language: str = os.getenv("XTTS_DEFAULT_LANGUAGE", "en")
    
    def __post_init__(self):
        self.voices_dir = Path(self.voices_dir)
        self.voices_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Voice Profile
# ============================================================================

@dataclass
class VoiceProfile:
    """A voice profile with reference audio."""
    name: str
    reference_files: List[Path]
    language: str = "en"
    description: str = ""
    
    # Cached conditioning (computed on first use)
    gpt_cond_latent: Optional[torch.Tensor] = None
    speaker_embedding: Optional[torch.Tensor] = None
    cache_hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        """Compute hash of reference files for cache invalidation."""
        h = hashlib.md5()
        for f in sorted(self.reference_files):
            if f.exists():
                h.update(f.read_bytes())
        return h.hexdigest()
        
    @property
    def is_cached(self) -> bool:
        """Check if conditioning is cached and valid."""
        if self.gpt_cond_latent is None or self.speaker_embedding is None:
            return False
        return self.cache_hash == self.compute_hash()
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "reference_files": [str(p) for p in self.reference_files],
            "language": self.language,
            "description": self.description
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "VoiceProfile":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            reference_files=[Path(p) for p in data.get("reference_files", [])],
            language=data.get("language", "en"),
            description=data.get("description", "")
        )


class VoiceProfileManager:
    """Manages voice profiles with persistence."""
    
    def __init__(self, config: XTTSConfig):
        self.config = config
        self.profiles: Dict[str, VoiceProfile] = {}
        self.default_profile: Optional[str] = None
        self._save_path = config.voices_dir / "profiles.json"
        
    def load(self):
        """Load profiles from disk."""
        if not self._save_path.exists():
            logger.info("No saved profiles found")
            return
            
        try:
            with open(self._save_path) as f:
                data = json.load(f)
                
            self.default_profile = data.get("default")
            
            for name, profile_data in data.get("profiles", {}).items():
                self.profiles[name] = VoiceProfile.from_dict(profile_data)
                
            logger.info(f"Loaded {len(self.profiles)} voice profiles")
            if self.default_profile:
                logger.info(f"Default profile: {self.default_profile}")
                
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            
    def save(self):
        """Save profiles to disk."""
        data = {
            "default": self.default_profile,
            "profiles": {
                name: profile.to_dict() 
                for name, profile in self.profiles.items()
            }
        }
        
        with open(self._save_path, "w") as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved {len(self.profiles)} profiles")
        
    def get_profile(self, name: Optional[str] = None) -> Optional[VoiceProfile]:
        """Get profile by name or default."""
        if name:
            return self.profiles.get(name)
        elif self.default_profile:
            return self.profiles.get(self.default_profile)
        return None
        
    def add_profile(self, profile: VoiceProfile, set_default: bool = False):
        """Add a voice profile."""
        self.profiles[profile.name] = profile
        if set_default or not self.default_profile:
            self.default_profile = profile.name
        self.save()
        
    def remove_profile(self, name: str):
        """Remove a voice profile."""
        if name in self.profiles:
            del self.profiles[name]
            if self.default_profile == name:
                self.default_profile = next(iter(self.profiles), None)
            self.save()


# ============================================================================
# XTTS Engine
# ============================================================================

class XTTSEngine:
    """XTTS synthesis engine with optional voice cloning."""
    
    def __init__(self, config: XTTSConfig, profile_manager: VoiceProfileManager):
        self.config = config
        self.profile_manager = profile_manager
        self.model = None
        self.model_lock = asyncio.Lock()
        self._loaded = False
        
        # Default speaker conditioning (for use without custom profiles)
        self._default_gpt_cond: Optional[torch.Tensor] = None
        self._default_speaker_emb: Optional[torch.Tensor] = None
        
    async def load_model(self):
        """Load the XTTS model."""
        if self._loaded:
            return
            
        logger.info(f"Loading XTTS model on {self.config.device}...")
        start_time = time.time()
        
        try:
            from TTS.api import TTS
            
            loop = asyncio.get_event_loop()
            
            def _load():
                return TTS(
                    model_name=self.config.model_name,
                    gpu=(self.config.device == "cuda")
                )
                
            self.model = await loop.run_in_executor(None, _load)
            self._loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"XTTS model loaded in {load_time:.1f}s")
            
            # Initialize default speaker
            await self._init_default_speaker()
            
            # Pre-compute conditioning for existing profiles
            for profile in self.profile_manager.profiles.values():
                try:
                    await self.ensure_profile_conditioned(profile)
                except Exception as e:
                    logger.warning(f"Failed to condition profile {profile.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            raise
            
    async def _init_default_speaker(self):
        """Initialize default speaker from built-in or sample."""
        logger.info("Initializing default speaker...")
        
        # Check for a default reference audio
        default_ref = self.config.voices_dir / "default_reference.wav"
        
        if default_ref.exists():
            # Use provided default reference
            logger.info(f"Using default reference: {default_ref}")
            await self._compute_default_conditioning([str(default_ref)])
        else:
            # Create a simple reference using the model's built-in capability
            # XTTS can work without conditioning by using tts() directly
            logger.info("No default reference found, will use model's tts() method for default voice")
            
    async def _compute_default_conditioning(self, reference_files: List[str]):
        """Compute conditioning for default speaker."""
        loop = asyncio.get_event_loop()
        
        def _compute():
            return self.model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=reference_files,
                gpt_cond_len=self.config.gpt_cond_len,
                max_ref_length=self.config.max_ref_len
            )
            
        self._default_gpt_cond, self._default_speaker_emb = await loop.run_in_executor(
            None, _compute
        )
        logger.info("Default speaker conditioning computed")
            
    async def ensure_profile_conditioned(self, profile: VoiceProfile):
        """Ensure voice profile has cached conditioning."""
        if profile.is_cached:
            return
            
        logger.info(f"Computing conditioning for profile: {profile.name}")
        
        loop = asyncio.get_event_loop()
        reference_files = [str(p) for p in profile.reference_files if p.exists()]
        
        if not reference_files:
            raise ValueError(f"No valid reference files for profile: {profile.name}")
            
        def _compute_conditioning():
            return self.model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=reference_files,
                gpt_cond_len=self.config.gpt_cond_len,
                max_ref_length=self.config.max_ref_len
            )
            
        gpt_cond_latent, speaker_embedding = await loop.run_in_executor(
            None, _compute_conditioning
        )
        
        profile.gpt_cond_latent = gpt_cond_latent
        profile.speaker_embedding = speaker_embedding
        profile.cache_hash = profile.compute_hash()
        
        logger.info(f"Conditioning computed for profile: {profile.name}")
        
    async def synthesize(
        self,
        text: str,
        profile_name: Optional[str] = None,
        language: Optional[str] = None
    ) -> bytes:
        """
        Synthesize speech (non-streaming).
        
        If no profile is specified and no default exists, uses the model's
        built-in default voice.
        """
        async with self.model_lock:
            profile = self.profile_manager.get_profile(profile_name)
            lang = language or (profile.language if profile else self.config.default_language)
            
            loop = asyncio.get_event_loop()
            
            if profile:
                # Use custom voice profile
                await self.ensure_profile_conditioned(profile)
                
                def _synthesize():
                    return self.model.synthesizer.tts_model.inference(
                        text=text,
                        language=lang,
                        gpt_cond_latent=profile.gpt_cond_latent,
                        speaker_embedding=profile.speaker_embedding,
                        temperature=self.config.temperature,
                        repetition_penalty=self.config.repetition_penalty,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p
                    )
                    
                result = await loop.run_in_executor(None, _synthesize)
                audio = np.array(result["wav"])
                
            elif self._default_gpt_cond is not None:
                # Use default speaker conditioning
                def _synthesize_default():
                    return self.model.synthesizer.tts_model.inference(
                        text=text,
                        language=lang,
                        gpt_cond_latent=self._default_gpt_cond,
                        speaker_embedding=self._default_speaker_emb,
                        temperature=self.config.temperature,
                        repetition_penalty=self.config.repetition_penalty,
                        top_k=self.config.top_k,
                        top_p=self.config.top_p
                    )
                    
                result = await loop.run_in_executor(None, _synthesize_default)
                audio = np.array(result["wav"])
                
            else:
                # Use model's high-level tts() method (built-in default voice)
                def _synthesize_builtin():
                    # Use tts() which handles speaker internally
                    wav = self.model.tts(
                        text=text,
                        language=lang
                    )
                    return np.array(wav)
                    
                audio = await loop.run_in_executor(None, _synthesize_builtin)
            
            # Convert to 16-bit PCM bytes
            audio = (audio * 32767).astype(np.int16)
            return audio.tobytes()
            
    async def synthesize_stream(
        self,
        text: str,
        profile_name: Optional[str] = None,
        language: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream synthesis using XTTS streaming inference.
        
        Note: Streaming requires voice conditioning. If no profile exists,
        falls back to non-streaming synthesis.
        """
        async with self.model_lock:
            profile = self.profile_manager.get_profile(profile_name)
            lang = language or (profile.language if profile else self.config.default_language)
            
            # Get conditioning (profile, default, or none)
            gpt_cond = None
            speaker_emb = None
            
            if profile:
                await self.ensure_profile_conditioned(profile)
                gpt_cond = profile.gpt_cond_latent
                speaker_emb = profile.speaker_embedding
            elif self._default_gpt_cond is not None:
                gpt_cond = self._default_gpt_cond
                speaker_emb = self._default_speaker_emb
                
            if gpt_cond is None:
                # No conditioning available - fall back to non-streaming
                logger.warning("No voice conditioning available, using non-streaming synthesis")
                audio = await self.synthesize(text, profile_name, language)
                yield audio
                return
                
            # Streaming synthesis
            try:
                chunks = self.model.synthesizer.tts_model.inference_stream(
                    text=text,
                    language=lang,
                    gpt_cond_latent=gpt_cond,
                    speaker_embedding=speaker_emb,
                    temperature=self.config.temperature,
                    repetition_penalty=self.config.repetition_penalty,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    stream_chunk_size=self.config.stream_chunk_size,
                    enable_text_splitting=True
                )
                
                for chunk in chunks:
                    if chunk is not None:
                        audio = chunk.cpu().numpy()
                        if audio.ndim > 1:
                            audio = audio.squeeze()
                        audio = (audio * 32767).astype(np.int16)
                        yield audio.tobytes()
                        
            except Exception as e:
                logger.error(f"Streaming error: {e}, falling back to non-streaming")
                audio = await self.synthesize(text, profile_name, language)
                yield audio


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="XTTS2 Streaming TTS Server",
    description="Text-to-speech with voice cloning using Coqui XTTS v2",
    version="1.0.0"
)

# Global instances
config: XTTSConfig = None
profile_manager: VoiceProfileManager = None
engine: XTTSEngine = None


class TTSRequest(BaseModel):
    """TTS synthesis request."""
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    stream: bool = False


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    global config, profile_manager, engine
    
    config = XTTSConfig()
    profile_manager = VoiceProfileManager(config)
    profile_manager.load()
    
    engine = XTTSEngine(config, profile_manager)
    await engine.load_model()
    
    logger.info(f"XTTS server ready on {config.host}:{config.port}")
    logger.info(f"Voices directory: {config.voices_dir}")
    logger.info(f"Loaded profiles: {list(profile_manager.profiles.keys())}")
    logger.info(f"Default profile: {profile_manager.default_profile or '(built-in)'}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": engine._loaded if engine else False,
        "device": config.device if config else "unknown",
        "profiles": list(profile_manager.profiles.keys()) if profile_manager else [],
        "default_profile": profile_manager.default_profile if profile_manager else None,
        "has_default_speaker": engine._default_gpt_cond is not None if engine else False
    }


@app.post("/v1/tts")
async def synthesize_speech(request: TTSRequest):
    """Synthesize speech and return WAV file."""
    if not engine or not engine._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
        
    try:
        if request.stream:
            # Redirect to streaming endpoint
            return await stream_synthesis_sse(request)
            
        audio_bytes = await engine.synthesize(
            text=request.text,
            profile_name=request.voice,
            language=request.language
        )
        
        # Create WAV file
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(config.output_sample_rate)
            wav_file.writeframes(audio_bytes)
            
        wav_buffer.seek(0)
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/stream")
async def stream_synthesis_sse(request: TTSRequest):
    """Stream synthesis via Server-Sent Events."""
    if not engine or not engine._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
        
    async def generate():
        try:
            async for chunk in engine.synthesize_stream(
                text=request.text,
                profile_name=request.voice,
                language=request.language
            ):
                encoded = base64.b64encode(chunk).decode('ascii')
                yield f"data: {json.dumps({'audio': encoded})}\n\n"
                
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.post("/v1/tts/raw")
async def synthesize_raw(request: TTSRequest):
    """Synthesize and return raw PCM bytes."""
    if not engine or not engine._loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
        
    try:
        audio_bytes = await engine.synthesize(
            text=request.text,
            profile_name=request.voice,
            language=request.language
        )
        
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(config.output_sample_rate),
                "X-Channels": "1",
                "X-Sample-Width": "2"
            }
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Voice Profile Management
# ============================================================================

@app.get("/v1/voices")
async def list_voices():
    """List all voice profiles."""
    profiles = []
    
    for name, profile in profile_manager.profiles.items():
        profiles.append({
            "name": name,
            "language": profile.language,
            "description": profile.description,
            "reference_count": len(profile.reference_files),
            "is_default": name == profile_manager.default_profile,
            "is_cached": profile.is_cached
        })
        
    return {
        "voices": profiles,
        "default": profile_manager.default_profile,
        "has_builtin_default": engine._default_gpt_cond is not None if engine else False
    }


@app.post("/v1/voices")
async def create_voice(
    name: str = Form(...),
    language: str = Form("en"),
    description: str = Form(""),
    set_default: bool = Form(False),
    reference_audio: List[UploadFile] = File(...)
):
    """Create a new voice profile from reference audio."""
    if not reference_audio:
        raise HTTPException(status_code=400, detail="At least one reference audio file required")
        
    if name in profile_manager.profiles:
        raise HTTPException(status_code=400, detail=f"Profile '{name}' already exists")
        
    # Create profile directory
    profile_dir = config.voices_dir / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    # Save reference files
    reference_files = []
    for i, upload in enumerate(reference_audio):
        ext = Path(upload.filename).suffix or ".wav"
        file_path = profile_dir / f"reference_{i}{ext}"
        
        with open(file_path, "wb") as f:
            content = await upload.read()
            f.write(content)
            
        reference_files.append(file_path)
        
    # Create profile
    profile = VoiceProfile(
        name=name,
        reference_files=reference_files,
        language=language,
        description=description
    )
    
    # Pre-compute conditioning
    try:
        await engine.ensure_profile_conditioned(profile)
    except Exception as e:
        # Clean up on failure
        shutil.rmtree(profile_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")
        
    # Save profile
    profile_manager.add_profile(profile, set_default=set_default)
    
    return {
        "status": "created",
        "name": name,
        "is_default": profile_manager.default_profile == name
    }


@app.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    """Delete a voice profile."""
    if name not in profile_manager.profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
        
    # Remove profile directory
    profile_dir = config.voices_dir / name
    if profile_dir.exists():
        shutil.rmtree(profile_dir)
        
    profile_manager.remove_profile(name)
    
    return {"status": "deleted", "name": name}


@app.post("/v1/voices/{name}/default")
async def set_default_voice(name: str):
    """Set a voice profile as default."""
    if name not in profile_manager.profiles:
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
        
    profile_manager.default_profile = name
    profile_manager.save()
    
    return {"status": "updated", "default": name}


@app.post("/v1/voices/default-reference")
async def upload_default_reference(
    reference_audio: UploadFile = File(...)
):
    """
    Upload a default reference audio file.
    This will be used when no voice profile is specified.
    """
    # Save as default reference
    default_path = config.voices_dir / "default_reference.wav"
    
    with open(default_path, "wb") as f:
        content = await reference_audio.read()
        f.write(content)
        
    # Recompute default conditioning
    try:
        await engine._compute_default_conditioning([str(default_path)])
        return {"status": "updated", "default_reference": str(default_path)}
    except Exception as e:
        default_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to process audio: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the server."""
    cfg = XTTSConfig()
    uvicorn.run(
        "xtts_server:app",
        host=cfg.host,
        port=cfg.port,
        log_level="info",
        workers=1  # XTTS is not thread-safe
    )


if __name__ == "__main__":
    main()