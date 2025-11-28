#!/usr/bin/env python3
"""
XTTS2 Streaming TTS Server - DGX SPARK GB10 EDITION
====================================================
Forces CUDA to work on GB10 by patching PyTorch's capability check.

API Endpoints:
- GET  /health              - Health check
- POST /v1/tts              - Synthesize (returns WAV)
- POST /v1/tts/stream       - Streaming synthesis (SSE)
- POST /v1/tts/raw          - Raw PCM bytes
- GET  /v1/voices           - List voice profiles
- POST /v1/voices           - Create voice profile
- DELETE /v1/voices/{name}  - Delete profile
- POST /v1/voices/{name}/default - Set default
- POST /v1/voices/default-reference - Upload default reference
- POST /v1/bootstrap        - Force re-bootstrap
- GET  /v1/debug/cuda       - Debug CUDA status
"""

import os
import sys

# =============================================================================
# CRITICAL: CUDA FIXES FOR GB10 - MUST BE BEFORE ANY OTHER IMPORTS
# =============================================================================
# GB10 has compute capability sm_121 which PyTorch doesn't officially recognize
# But CUDA 13.0 + PyTorch 2.9 DOES work - we just need to bypass the checks

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "all") 
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import torch

# Store original functions
_original_is_available = torch.cuda.is_available
_original_torch_load = torch.load

# Track if we've successfully initialized CUDA
_cuda_actually_works = None

def _test_cuda_actually_works():
    """Test if CUDA actually works by creating a tensor."""
    global _cuda_actually_works
    if _cuda_actually_works is not None:
        return _cuda_actually_works
    
    try:
        # Try to create a CUDA tensor
        t = torch.zeros(1, device='cuda:0')
        del t
        _cuda_actually_works = True
        print(f"[GB10 CUDA FIX] CUDA works! Device: {torch.cuda.get_device_name(0)}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[GB10 CUDA FIX] CUDA test failed: {e}", file=sys.stderr)
        _cuda_actually_works = False
        return False

def _patched_is_available():
    """
    Patched torch.cuda.is_available() that returns True on GB10.
    The original returns False because PyTorch doesn't recognize sm_121,
    but CUDA actually works fine.
    """
    # First check if original says yes
    if _original_is_available():
        return True
    
    # Original says no, but let's test if CUDA actually works
    return _test_cuda_actually_works()

def _patched_torch_load(*args, **kwargs):
    """Patched torch.load with weights_only=False for TTS compatibility."""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Apply patches BEFORE any other imports
torch.cuda.is_available = _patched_is_available
torch.load = _patched_torch_load

# Force CUDA initialization now
print("[GB10 CUDA FIX] Testing CUDA...", file=sys.stderr)
_test_cuda_actually_works()

# =============================================================================
# Now import everything else
# =============================================================================

import asyncio
import base64
import glob
import hashlib
import io
import json
import logging
import shutil
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA version: {torch.version.cuda}")
logger.info(f"CUDA available (patched): {torch.cuda.is_available()}")
logger.info(f"CUDA actually works: {_cuda_actually_works}")
if _cuda_actually_works:
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class XTTSConfig:
    """XTTS server configuration."""
    model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    device: str = field(default_factory=lambda: "cuda" if _cuda_actually_works else "cpu")
    voices_dir: Path = field(default_factory=lambda: Path(os.getenv("VOICES_DIR", "/app/data/voices")))
    
    temperature: float = float(os.getenv("XTTS_TEMPERATURE", "0.7"))
    top_k: int = int(os.getenv("XTTS_TOP_K", "50"))
    top_p: float = float(os.getenv("XTTS_TOP_P", "0.85"))
    repetition_penalty: float = float(os.getenv("XTTS_REPETITION_PENALTY", "2.0"))
    stream_chunk_size: int = int(os.getenv("XTTS_CHUNK_SIZE", "20"))
    gpt_cond_len: int = int(os.getenv("XTTS_GPT_COND_LEN", "30"))
    max_ref_len: int = int(os.getenv("XTTS_MAX_REF_LEN", "60"))
    output_sample_rate: int = 24000
    host: str = os.getenv("XTTS_HOST", "0.0.0.0")
    port: int = int(os.getenv("XTTS_PORT", "8001"))
    default_language: str = os.getenv("XTTS_DEFAULT_LANGUAGE", "en")
    auto_bootstrap: bool = os.getenv("XTTS_AUTO_BOOTSTRAP", "true").lower() == "true"
    
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
    gpt_cond_latent: Optional[torch.Tensor] = None
    speaker_embedding: Optional[torch.Tensor] = None
    cache_hash: Optional[str] = None
    
    def compute_hash(self) -> str:
        h = hashlib.md5()
        for f in sorted(self.reference_files):
            if f.exists():
                h.update(f.read_bytes())
        return h.hexdigest()
        
    @property
    def is_cached(self) -> bool:
        if self.gpt_cond_latent is None or self.speaker_embedding is None:
            return False
        return self.cache_hash == self.compute_hash()
        
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "reference_files": [str(p) for p in self.reference_files],
            "language": self.language,
            "description": self.description
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "VoiceProfile":
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
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            
    def save(self):
        data = {
            "default": self.default_profile,
            "profiles": {name: profile.to_dict() for name, profile in self.profiles.items()}
        }
        with open(self._save_path, "w") as f:
            json.dump(data, f, indent=2)
            
    def get_profile(self, name: Optional[str] = None) -> Optional[VoiceProfile]:
        if name:
            return self.profiles.get(name)
        elif self.default_profile:
            return self.profiles.get(self.default_profile)
        return None
        
    def add_profile(self, profile: VoiceProfile, set_default: bool = False):
        self.profiles[profile.name] = profile
        if set_default or not self.default_profile:
            self.default_profile = profile.name
        self.save()
        
    def remove_profile(self, name: str):
        if name in self.profiles:
            del self.profiles[name]
            if self.default_profile == name:
                self.default_profile = next(iter(self.profiles), None)
            self.save()


# ============================================================================
# XTTS Engine
# ============================================================================

class XTTSEngine:
    """XTTS synthesis engine."""
    
    def __init__(self, config: XTTSConfig, profile_manager: VoiceProfileManager):
        self.config = config
        self.profile_manager = profile_manager
        self.model = None
        self.model_lock = asyncio.Lock()
        self._loaded = False
        self._default_gpt_cond: Optional[torch.Tensor] = None
        self._default_speaker_emb: Optional[torch.Tensor] = None
        self._bootstrapped = False
        
    async def load_model(self):
        """Load the XTTS model."""
        if self._loaded:
            return
            
        use_gpu = _cuda_actually_works and self.config.device == "cuda"
        logger.info(f"Loading XTTS model (GPU: {use_gpu})...")
        start_time = time.time()
        
        try:
            from TTS.api import TTS
            
            loop = asyncio.get_event_loop()
            
            def _load():
                # TTS checks torch.cuda.is_available() which we've patched
                return TTS(
                    model_name=self.config.model_name,
                    gpu=use_gpu
                )
                
            self.model = await loop.run_in_executor(None, _load)
            self._loaded = True
            
            load_time = time.time() - start_time
            logger.info(f"XTTS model loaded in {load_time:.1f}s")
            
            # Check what device the model is actually on
            if hasattr(self.model, 'synthesizer') and hasattr(self.model.synthesizer, 'tts_model'):
                try:
                    param = next(self.model.synthesizer.tts_model.parameters())
                    logger.info(f"Model is on device: {param.device}")
                except:
                    pass
            
            await self._init_default_speaker()
            
            for profile in self.profile_manager.profiles.values():
                try:
                    await self.ensure_profile_conditioned(profile)
                except Exception as e:
                    logger.warning(f"Failed to condition profile {profile.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load XTTS model: {e}")
            import traceback
            traceback.print_exc()
            raise
            
    async def _init_default_speaker(self):
        logger.info("Initializing default speaker...")
        default_ref = self.config.voices_dir / "default_reference.wav"
        
        if default_ref.exists():
            logger.info(f"Using default reference: {default_ref}")
            await self._compute_default_conditioning([str(default_ref)])
        elif self.config.auto_bootstrap:
            logger.info("Auto-bootstrapping voice conditioning...")
            await self._bootstrap_default_conditioning()
        else:
            logger.info("No default reference, streaming unavailable")
            
    async def _bootstrap_default_conditioning(self):
        logger.info("Bootstrapping voice conditioning...")
        start_time = time.time()
        loop = asyncio.get_event_loop()
        
        def _find_or_create_reference():
            voices_dir = self.config.voices_dir
            for ext in ['*.wav', '*.mp3', '*.flac']:
                files = glob.glob(str(voices_dir / '**' / ext), recursive=True)
                if files:
                    return files[0]
            
            # Generate synthetic audio
            duration = 5.0
            sample_rate = self.config.output_sample_rate
            num_samples = int(sample_rate * duration)
            t = np.linspace(0, duration, num_samples)
            
            f0 = 150 + 10 * np.sin(2 * np.pi * 5 * t)
            phase = np.cumsum(2 * np.pi * f0 / sample_rate)
            audio = np.zeros(num_samples)
            
            for harmonic in range(1, 8):
                audio += (1.0 / harmonic) * np.sin(harmonic * phase)
            
            window = 50
            if len(audio) > window:
                audio = np.convolve(audio, np.ones(window)/window, mode='same')
            
            envelope = np.ones(num_samples)
            fade = int(0.1 * sample_rate)
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            envelope *= 0.7 + 0.3 * np.sin(2 * np.pi * 3 * t)
            
            audio = audio * envelope
            audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.8
            audio_int16 = (audio * 32767).astype(np.int16)
            
            ref_path = str(voices_dir / "bootstrap_reference.wav")
            with wave.open(ref_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
            
            return ref_path
        
        def _do_bootstrap():
            ref_path = _find_or_create_reference()
            logger.info(f"Using reference: {ref_path}")
            return self.model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[ref_path],
                gpt_cond_len=self.config.gpt_cond_len,
                max_ref_length=self.config.max_ref_len
            )
            
        try:
            self._default_gpt_cond, self._default_speaker_emb = await loop.run_in_executor(None, _do_bootstrap)
            self._bootstrapped = True
            logger.info(f"Bootstrapped in {time.time() - start_time:.1f}s")
        except Exception as e:
            logger.error(f"Bootstrap failed: {e}")
            import traceback
            traceback.print_exc()
            
    async def _compute_default_conditioning(self, reference_files: List[str]):
        loop = asyncio.get_event_loop()
        def _compute():
            return self.model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=reference_files,
                gpt_cond_len=self.config.gpt_cond_len,
                max_ref_length=self.config.max_ref_len
            )
        self._default_gpt_cond, self._default_speaker_emb = await loop.run_in_executor(None, _compute)
        logger.info("Default conditioning computed")
            
    async def ensure_profile_conditioned(self, profile: VoiceProfile):
        if profile.is_cached:
            return
        logger.info(f"Computing conditioning for: {profile.name}")
        loop = asyncio.get_event_loop()
        reference_files = [str(p) for p in profile.reference_files if p.exists()]
        if not reference_files:
            raise ValueError(f"No valid reference files for: {profile.name}")
            
        def _compute():
            return self.model.synthesizer.tts_model.get_conditioning_latents(
                audio_path=reference_files,
                gpt_cond_len=self.config.gpt_cond_len,
                max_ref_length=self.config.max_ref_len
            )
        gpt_cond, speaker_emb = await loop.run_in_executor(None, _compute)
        profile.gpt_cond_latent = gpt_cond
        profile.speaker_embedding = speaker_emb
        profile.cache_hash = profile.compute_hash()
        
    async def synthesize(self, text: str, profile_name: Optional[str] = None, language: Optional[str] = None) -> bytes:
        async with self.model_lock:
            profile = self.profile_manager.get_profile(profile_name)
            lang = language or (profile.language if profile else self.config.default_language)
            loop = asyncio.get_event_loop()
            
            if profile:
                await self.ensure_profile_conditioned(profile)
                def _synth():
                    return self.model.synthesizer.tts_model.inference(
                        text=text, language=lang,
                        gpt_cond_latent=profile.gpt_cond_latent,
                        speaker_embedding=profile.speaker_embedding,
                        temperature=self.config.temperature,
                        repetition_penalty=self.config.repetition_penalty,
                        top_k=self.config.top_k, top_p=self.config.top_p
                    )
                result = await loop.run_in_executor(None, _synth)
                audio = np.array(result["wav"])
            elif self._default_gpt_cond is not None:
                def _synth():
                    return self.model.synthesizer.tts_model.inference(
                        text=text, language=lang,
                        gpt_cond_latent=self._default_gpt_cond,
                        speaker_embedding=self._default_speaker_emb,
                        temperature=self.config.temperature,
                        repetition_penalty=self.config.repetition_penalty,
                        top_k=self.config.top_k, top_p=self.config.top_p
                    )
                result = await loop.run_in_executor(None, _synth)
                audio = np.array(result["wav"])
            else:
                def _synth():
                    return np.array(self.model.tts(text=text, language=lang))
                audio = await loop.run_in_executor(None, _synth)
            
            return (audio * 32767).astype(np.int16).tobytes()
            
    async def synthesize_stream(self, text: str, profile_name: Optional[str] = None, language: Optional[str] = None) -> AsyncGenerator[bytes, None]:
        async with self.model_lock:
            profile = self.profile_manager.get_profile(profile_name)
            lang = language or (profile.language if profile else self.config.default_language)
            
            gpt_cond, speaker_emb = None, None
            if profile:
                await self.ensure_profile_conditioned(profile)
                gpt_cond = profile.gpt_cond_latent
                speaker_emb = profile.speaker_embedding
            elif self._default_gpt_cond is not None:
                gpt_cond = self._default_gpt_cond
                speaker_emb = self._default_speaker_emb
                
            if gpt_cond is None:
                audio = await self.synthesize(text, profile_name, language)
                yield audio
                return
                
            loop = asyncio.get_event_loop()
            try:
                def _stream():
                    return list(self.model.synthesizer.tts_model.inference_stream(
                        text=text, language=lang,
                        gpt_cond_latent=gpt_cond, speaker_embedding=speaker_emb,
                        temperature=self.config.temperature,
                        repetition_penalty=self.config.repetition_penalty,
                        top_k=self.config.top_k, top_p=self.config.top_p,
                        stream_chunk_size=self.config.stream_chunk_size,
                        enable_text_splitting=True
                    ))
                chunks = await loop.run_in_executor(None, _stream)
                for chunk in chunks:
                    if chunk is not None:
                        audio = chunk.cpu().numpy()
                        if audio.ndim > 1:
                            audio = audio.squeeze()
                        yield (audio * 32767).astype(np.int16).tobytes()
            except Exception as e:
                logger.error(f"Stream error: {e}")
                audio = await self.synthesize(text, profile_name, language)
                yield audio


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(title="XTTS2 Server", version="2.0.0-gb10")

config: XTTSConfig = None
profile_manager: VoiceProfileManager = None
engine: XTTSEngine = None


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = None
    stream: bool = False


@app.on_event("startup")
async def startup():
    global config, profile_manager, engine
    config = XTTSConfig()
    profile_manager = VoiceProfileManager(config)
    profile_manager.load()
    engine = XTTSEngine(config, profile_manager)
    await engine.load_model()
    
    logger.info(f"Server ready on {config.host}:{config.port}")
    logger.info(f"Device: {config.device}")
    logger.info(f"CUDA works: {_cuda_actually_works}")
    logger.info(f"Streaming: {engine._default_gpt_cond is not None}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": engine._loaded if engine else False,
        "device": config.device if config else "unknown",
        "cuda_works": _cuda_actually_works,
        "cuda_patched": torch.cuda.is_available(),
        "streaming": engine._default_gpt_cond is not None if engine else False,
        "bootstrapped": engine._bootstrapped if engine else False,
        "profiles": list(profile_manager.profiles.keys()) if profile_manager else [],
        "default_profile": profile_manager.default_profile if profile_manager else None,
        "sample_rate": config.output_sample_rate if config else 24000,
        "pytorch": torch.__version__,
        "cuda_version": torch.version.cuda
    }


@app.post("/v1/tts")
async def synthesize_speech(request: TTSRequest):
    if not engine or not engine._loaded:
        raise HTTPException(503, "Model not loaded")
    if not request.text.strip():
        raise HTTPException(400, "Text required")
        
    try:
        if request.stream:
            return await stream_sse(request)
        audio = await engine.synthesize(request.text, request.voice, request.language)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.output_sample_rate)
            wf.writeframes(audio)
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"})
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(500, str(e))


@app.post("/v1/tts/stream")
async def stream_sse(request: TTSRequest):
    if not engine or not engine._loaded:
        raise HTTPException(503, "Model not loaded")
    if not request.text.strip():
        raise HTTPException(400, "Text required")
        
    async def generate():
        try:
            async for chunk in engine.synthesize_stream(request.text, request.voice, request.language):
                if chunk:
                    yield f"data: {json.dumps({'audio': base64.b64encode(chunk).decode()})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.post("/v1/tts/raw")
async def synthesize_raw(request: TTSRequest):
    """Synthesize and return raw PCM bytes."""
    if not engine or not engine._loaded:
        raise HTTPException(503, "Model not loaded")
    if not request.text.strip():
        raise HTTPException(400, "Text required")
        
    try:
        audio = await engine.synthesize(request.text, request.voice, request.language)
        return StreamingResponse(io.BytesIO(audio), media_type="application/octet-stream",
            headers={
                "X-Sample-Rate": str(config.output_sample_rate),
                "X-Channels": "1",
                "X-Sample-Width": "2"
            })
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(500, str(e))


# ============================================================================
# Voice Profile Management
# ============================================================================

@app.get("/v1/voices")
async def list_voices():
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
        "has_builtin_default": engine._default_gpt_cond is not None if engine else False,
        "bootstrapped": engine._bootstrapped if engine else False
    }


@app.post("/v1/voices")
async def create_voice(name: str = Form(...), language: str = Form("en"), 
                       description: str = Form(""), set_default: bool = Form(False),
                       reference_audio: List[UploadFile] = File(...)):
    """Create a new voice profile from reference audio."""
    if not reference_audio:
        raise HTTPException(400, "At least one reference audio file required")
    if name in profile_manager.profiles:
        raise HTTPException(400, f"Profile '{name}' already exists")
        
    profile_dir = config.voices_dir / name
    profile_dir.mkdir(parents=True, exist_ok=True)
    
    refs = []
    for i, upload in enumerate(reference_audio):
        ext = Path(upload.filename).suffix or ".wav"
        path = profile_dir / f"ref_{i}{ext}"
        with open(path, "wb") as f:
            f.write(await upload.read())
        refs.append(path)
        
    profile = VoiceProfile(name=name, reference_files=refs, language=language, description=description)
    try:
        await engine.ensure_profile_conditioned(profile)
    except Exception as e:
        shutil.rmtree(profile_dir, ignore_errors=True)
        raise HTTPException(400, f"Failed to process audio: {e}")
    profile_manager.add_profile(profile, set_default)
    return {"status": "created", "name": name, "is_default": profile_manager.default_profile == name}


@app.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    """Delete a voice profile."""
    if name not in profile_manager.profiles:
        raise HTTPException(404, f"Profile '{name}' not found")
    shutil.rmtree(config.voices_dir / name, ignore_errors=True)
    profile_manager.remove_profile(name)
    return {"status": "deleted", "name": name}


@app.post("/v1/voices/{name}/default")
async def set_default(name: str):
    """Set a voice profile as default."""
    if name not in profile_manager.profiles:
        raise HTTPException(404, f"Profile '{name}' not found")
    profile_manager.default_profile = name
    profile_manager.save()
    return {"status": "updated", "default": name}


@app.post("/v1/voices/default-reference")
async def upload_default_ref(reference_audio: UploadFile = File(...)):
    """Upload a default reference audio file."""
    path = config.voices_dir / "default_reference.wav"
    with open(path, "wb") as f:
        f.write(await reference_audio.read())
    try:
        await engine._compute_default_conditioning([str(path)])
        engine._bootstrapped = False
        return {"status": "updated", "default_reference": str(path)}
    except Exception as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Failed to process audio: {e}")


@app.post("/v1/bootstrap")
async def force_bootstrap():
    """Force re-bootstrap of voice conditioning."""
    if not engine or not engine._loaded:
        raise HTTPException(503, "Model not loaded")
    await engine._bootstrap_default_conditioning()
    return {
        "status": "bootstrapped",
        "streaming_available": engine._default_gpt_cond is not None
    }


@app.get("/v1/debug/cuda")
async def debug_cuda():
    """Debug endpoint for CUDA status."""
    info = {
        "torch_version": torch.__version__,
        "cuda_available_original": _original_is_available(),
        "cuda_available_patched": torch.cuda.is_available(),
        "cuda_actually_works": _cuda_actually_works,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": 0,
        "devices": [],
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
            "CUDA_MODULE_LOADING": os.environ.get("CUDA_MODULE_LOADING"),
        }
    }
    
    try:
        info["cuda_device_count"] = torch.cuda.device_count()
        for i in range(torch.cuda.device_count()):
            info["devices"].append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory,
            })
    except Exception as e:
        info["cuda_error"] = str(e)
        
    # Try forced init
    try:
        test = torch.zeros(1).cuda()
        info["cuda_test_tensor"] = "success"
        del test
    except Exception as e:
        info["cuda_test_tensor"] = f"failed: {e}"
        
    return info


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the server."""
    cfg = XTTSConfig()
    uvicorn.run("xtts_server:app", host=cfg.host, port=cfg.port, log_level="info", workers=1)


if __name__ == "__main__":
    main()