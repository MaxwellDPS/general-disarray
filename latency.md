# Latency Optimization Guide

This document explains the latency optimizations implemented and how to tune them.

## Latency Breakdown

A typical voice AI interaction has these stages:

```
User speaks → VAD detects end → STT → LLM → TTS → Audio playback
              |<-- Silence -->|  |<------ Processing ------>|
```

### Target Latencies

| Stage | Standard | Optimized | Notes |
|-------|----------|-----------|-------|
| Silence detection | 1000ms | 500ms | Time after user stops speaking |
| STT (Whisper) | 500-1500ms | 200-500ms | Depends on audio length |
| LLM TTFT | 200-500ms | 150-300ms | Time to first token |
| TTS first chunk | 300-500ms | 100-200ms | Time to first audio |
| **Total** | **2-3.5s** | **< 1s** | End of speech → first audio |

## Key Optimizations

### 1. Faster Silence Detection (500ms → 300ms possible)

**File**: `audio_pipeline_fast.py`, `config_fast.py`

```python
# Standard
silence_duration_ms: int = 1000  # Wait 1 second after speech

# Optimized  
silence_duration_ms: int = 500   # Wait 500ms (can go lower with clean audio)
```

**Trade-off**: Lower values may cut off slow speakers mid-sentence.

**Tuning**:
```bash
SILENCE_TIMEOUT_MS=400  # Very aggressive
SILENCE_TIMEOUT_MS=600  # Balanced
SILENCE_TIMEOUT_MS=800  # Conservative
```

### 2. Faster Whisper Settings

**File**: `audio_pipeline_fast.py`

```python
# Standard
beam_size=5, best_of=2

# Optimized
beam_size=3, best_of=1, patience=0.5
```

**Model size trade-off**:

| Model | Speed | Accuracy | VRAM |
|-------|-------|----------|------|
| tiny | 10x | 60% | 1GB |
| base | 7x | 75% | 1GB |
| small | 4x | 85% | 2GB |
| medium | 2x | 92% | 5GB |
| large-v3 | 1x | 95% | 10GB |

**Recommendation**: Use `medium` for best speed/accuracy balance.

```bash
WHISPER_MODEL=medium
```

### 3. LLM Token Streaming

**File**: `main_fast.py`

Instead of waiting for complete response, we:
1. Stream tokens from LLM
2. Buffer into sentences
3. Send each sentence to TTS immediately

```python
# Stream tokens
async for token in self._stream_llm_tokens(context):
    sentence_buffer += token
    
    # Complete sentence? Send to TTS immediately
    if self._is_sentence_complete(sentence_buffer):
        async for audio in self.tts.synthesize_stream(sentence):
            play(audio)
```

**Result**: First audio starts while LLM is still generating.

### 4. Shorter LLM Responses

**File**: `config_fast.py`

```python
# Standard
llm_max_tokens: int = 2000
system_prompt = "..." # Long prompt

# Optimized
llm_max_tokens: int = 150  # Force short responses
system_prompt = "Keep responses SHORT (1-2 sentences max)"
```

### 5. TTS Optimizations

**File**: `xtts_server_fast.py`

```python
# Standard XTTS
temperature=0.7, top_k=50, stream_chunk_size=20

# Optimized
temperature=0.65, top_k=30, stream_chunk_size=10
```

**Chunk size**: Smaller = faster first audio, but more overhead.

### 6. Audio Caching

**File**: `xtts_server_fast.py`, `audio_pipeline_fast.py`

Pre-synthesize common phrases:

```python
cached_phrases = [
    "Okay", "Sure", "Got it", "One moment",
    "Copy that", "On it", "Checking", "Goodbye"
]
```

**Result**: Acknowledgments play instantly (0ms latency).

### 7. Faster Barge-in

**File**: `config_fast.py`, `main_fast.py`

```python
# Standard
barge_in_min_duration_ms: int = 700  # 700ms of speech to interrupt

# Optimized
barge_in_min_duration_ms: int = 400  # 400ms
```

## Configuration Presets

### Ultra-Low Latency (< 800ms)
```bash
SILENCE_TIMEOUT_MS=400
WHISPER_MODEL=small
WHISPER_BEAM_SIZE=1
XTTS_TEMPERATURE=0.6
XTTS_TOP_K=20
XTTS_CHUNK_SIZE=8
BARGE_IN_MIN_DURATION=300
```

**Trade-offs**: May miss some speech, less accurate transcription, faster TTS but slightly robotic.

### Balanced (< 1200ms)
```bash
SILENCE_TIMEOUT_MS=500
WHISPER_MODEL=medium
WHISPER_BEAM_SIZE=3
XTTS_TEMPERATURE=0.65
XTTS_TOP_K=30
XTTS_CHUNK_SIZE=10
BARGE_IN_MIN_DURATION=400
```

**Recommended for most use cases.**

### High Quality (< 2000ms)
```bash
SILENCE_TIMEOUT_MS=800
WHISPER_MODEL=large-v3
WHISPER_BEAM_SIZE=5
XTTS_TEMPERATURE=0.7
XTTS_TOP_K=50
XTTS_CHUNK_SIZE=20
BARGE_IN_MIN_DURATION=700
```

**Best accuracy and voice quality.**

## Measurement

Add timing logs to measure latency:

```python
from audio_pipeline_fast import LatencyMetrics

metrics = LatencyMetrics()
metrics.speech_end = time.time()
# ... STT ...
metrics.stt_end = time.time()
# ... LLM ...
metrics.llm_first_token = time.time()
# ... TTS ...
metrics.tts_first_chunk = time.time()
metrics.log_summary()
```

Output:
```
STT latency: 250ms
LLM TTFT: 180ms
TTS first chunk: 120ms
Total response latency: 550ms
```

## Hardware Considerations

### GPU Memory Allocation

For GB10 with 100GB+ VRAM:
```
vLLM (70B model): ~50-60GB
XTTS: ~6GB
Whisper large: ~10GB
---
Total: ~70GB (leaves headroom)
```

### CPU Considerations

- Whisper can use CPU for preprocessing
- Audio resampling happens on CPU
- Consider `cpu_threads=4` for Whisper

## Network Latency

If running services on different machines:

| Connection | Latency Impact |
|------------|---------------|
| Same machine | ~0ms |
| Same LAN | 1-5ms |
| Cloud (same region) | 10-50ms |
| Cross-region | 50-200ms |

**Recommendation**: Run all services on same machine for lowest latency.

## Troubleshooting

### STT Too Slow
1. Use smaller Whisper model
2. Reduce beam_size
3. Check GPU utilization

### LLM Too Slow  
1. Reduce max_tokens
2. Use smaller model
3. Enable speculative decoding in vLLM

### TTS Too Slow
1. Lower temperature and top_k
2. Reduce stream_chunk_size
3. Enable caching
4. Check GPU isn't memory-bound

### Audio Cutting Off
1. Increase silence_duration_ms
2. Reduce vad_aggressiveness
3. Check audio quality/noise

### Barge-in Not Working
1. Lower barge_in_energy_threshold
2. Reduce barge_in_min_duration_ms
3. Check for echo/feedback issues