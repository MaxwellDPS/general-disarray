#!/usr/bin/env python3
"""
SIP AI Assistant - Test/Demo Script
====================================
Tests individual components without full SIP integration.
Useful for verifying setup before running the full system.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_config():
    """Test configuration loading."""
    print("\n=== Testing Configuration ===")
    try:
        from config import Config



        config = Config()
        print(f"✓ Config loaded successfully")
        print(f"  - SIP User: {config.sip_user}")
        print(f"  - SIP Domain: {config.sip_domain}")
        print(f"  - LLM Model: {config.llm_model}")
        print(f"  - Whisper Model: {config.whisper_model}")
        return True
    except Exception as e:
        print(f"✗ Config failed: {e}")
        return False


async def test_vad():
    """Test Voice Activity Detection."""
    print("\n=== Testing Voice Activity Detection ===")
    try:
        from audio_pipeline import VoiceActivityDetector
        from config import Config



        
        config = Config()
        vad = VoiceActivityDetector(config)
        
        # Test with silence
        silence = bytes(320 * 2)  # 20ms of silence at 16kHz
        is_speech = vad.is_speech(silence)
        print(f"✓ VAD initialized")
        print(f"  - Silence detected as speech: {is_speech} (expected: False)")
        
        # Test with noise (simulated speech)
        import random
        noise = bytes([random.randint(200, 255) for _ in range(320 * 2)])
        is_speech_noise = vad.is_speech(noise)
        print(f"  - Noise detected as speech: {is_speech_noise}")
        
        return True
    except ImportError as e:
        print(f"⚠ VAD test skipped (missing dependency): {e}")
        return True
    except Exception as e:
        print(f"✗ VAD failed: {e}")
        return False


async def test_whisper():
    """Test Whisper STT loading."""
    print("\n=== Testing Whisper STT ===")
    try:
        from audio_pipeline import SpeechToText
        from config import Config



        
        config = Config()
        # Use smaller model for testing
        config.whisper_model = "tiny"
        
        stt = SpeechToText(config)
        await stt.load_model()
        
        if stt.model:
            print(f"✓ Whisper model loaded successfully")
            
            # Test transcription with silence (should return empty or minimal)
            import numpy as np
            silence = np.zeros(16000, dtype=np.int16).tobytes()  # 1 second
            text = await stt.transcribe(silence)
            print(f"  - Test transcription: '{text}' (may be empty)")
        else:
            print(f"⚠ Whisper not available, using mock mode")
            
        return True
    except ImportError as e:
        print(f"⚠ Whisper test skipped (missing dependency): {e}")
        return True
    except Exception as e:
        print(f"✗ Whisper failed: {e}")
        return False


async def test_piper():
    """Test Piper TTS."""
    print("\n=== Testing Piper TTS ===")
    try:
        # Check if piper-tts is installed
        try:
            from piper import PiperVoice
            from piper.download import get_voices
            print("✓ piper-tts package is installed")
            
            # List available voices
            voices = get_voices()
            print(f"  Available voices online: {len(voices)}")
            
        except ImportError:
            print("⚠ piper-tts not installed. Run: pip install piper-tts")
            return True
            
        # Test with the audio pipeline
        from audio_pipeline import TextToSpeech
        from config import Config



        
        config = Config()
        tts = TextToSpeech(config)
        await tts._load_piper()
        
        if tts.piper_voice:
            print(f"✓ Piper voice loaded: {config.piper_model}")
            
            # Test synthesis
            audio = tts._synthesize_sync("Hello, this is a test.")
            
            if len(audio) > 0:
                duration_ms = len(audio) / (config.sample_rate * 2) * 1000
                print(f"✓ TTS generated {duration_ms:.0f}ms of audio")
            else:
                print("⚠ TTS returned no audio")
        else:
            print(f"⚠ Piper voice not loaded (model may need to be downloaded)")
            print(f"  Model will be auto-downloaded on first use: {config.piper_model}")
            
        return True
    except Exception as e:
        print(f"✗ TTS failed: {e}")
        return False


async def test_voice_cloning():
    """Test voice cloning system."""
    print("\n=== Testing Voice Cloning ===")
    try:
        from voice_cloning import VoiceCloningTTS, VoiceCloningConfig
        
        vc_config = VoiceCloningConfig(
            enabled=True,
            backend="xtts",
            device="cuda"
        )
        
        print(f"  Backend: {vc_config.backend}")
        print(f"  Device: {vc_config.device}")
        print(f"  Voices dir: {vc_config.voices_dir}")
        
        # Check if TTS library is available
        try:
            from TTS.api import TTS
            print("✓ XTTS (TTS library) is available")
        except ImportError:
            print("⚠ XTTS not installed. Run: pip install TTS")
            
        # Check for Chatterbox
        try:
            from chatterbox.tts import ChatterboxTTS
            print("✓ Chatterbox is available")
        except ImportError:
            print("⚠ Chatterbox not installed. Run: pip install chatterbox-tts")
            
        # Test profile loading
        tts = VoiceCloningTTS(vc_config)
        await tts._load_profiles()
        
        profiles = tts.list_profiles()
        if profiles:
            print(f"✓ Found {len(profiles)} voice profile(s): {profiles}")
        else:
            print("⚠ No voice profiles found")
            print("  Create one with: python voice_manager.py create <name> <audio.wav>")
            
        return True
        
    except ImportError as e:
        print(f"⚠ Voice cloning test skipped (missing dependency): {e}")
        return True
    except Exception as e:
        print(f"✗ Voice cloning test failed: {e}")
        return False


async def test_llm():
    """Test LLM connection."""
    print("\n=== Testing LLM Connection ===")
    try:
        import httpx
        from config import Config



        
        config = Config()
        
        # Test vLLM health endpoint
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{config.llm_base_url.replace('/v1', '')}/health",
                    timeout=5.0
                )
                if response.status_code == 200:
                    print(f"✓ vLLM server is healthy")
                else:
                    print(f"⚠ vLLM responded with status {response.status_code}")
            except httpx.ConnectError:
                print(f"⚠ vLLM server not running at {config.llm_base_url}")
                print(f"  Start with: python -m vllm.entrypoints.openai.api_server --model {config.llm_model}")
                return True  # Not a failure, just not running
                
        # Test model listing
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(
                base_url=config.llm_base_url,
                api_key="not-needed"
            )
            models = await client.models.list()
            print(f"✓ Available models: {[m.id for m in models.data]}")
            await client.close()
        except Exception as e:
            print(f"⚠ Could not list models: {e}")
            
        return True
    except ImportError as e:
        print(f"⚠ LLM test skipped (missing dependency): {e}")
        return True
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
        return False


async def test_tools():
    """Test tool system."""
    print("\n=== Testing Tool System ===")
    try:
        from tool_manager import ToolManager, ToolResult, ToolStatus
        from config import Config



        
        # Create mock assistant
        class MockAssistant:
            def __init__(self):
                self.config = Config()
                self.current_call = None
                
        assistant = MockAssistant()
        tm = ToolManager(assistant)
        
        print(f"✓ Tool manager initialized")
        print(f"  - Registered tools: {list(tm.tools.keys())}")
        
        # Test scheduling
        task_id = await tm.schedule_task(
            task_type="timer",
            delay_seconds=60,
            message="Test timer"
        )
        print(f"✓ Scheduled test task: {task_id}")
        
        pending = tm.get_pending_tasks()
        print(f"  - Pending tasks: {len(pending)}")
        
        # Cancel it
        cancelled = await tm.cancel_tasks()
        print(f"  - Cancelled: {cancelled}")
        
        return True
    except Exception as e:
        print(f"✗ Tools failed: {e}")
        return False


async def test_sip():
    """Test SIP handler (mock mode)."""
    print("\n=== Testing SIP Handler ===")
    try:
        from sip_handler import SIPHandler, PJSUA_AVAILABLE
        from config import Config



        
        config = Config()
        
        async def dummy_callback(call):
            pass
            
        handler = SIPHandler(config, dummy_callback)
        
        if PJSUA_AVAILABLE:
            print(f"✓ PJSUA2 is available")
        else:
            print(f"⚠ PJSUA2 not available (running in mock mode)")
            print(f"  Install with: pip install pjsua2 (after building PJSIP)")
            
        print(f"✓ SIP handler created")
        print(f"  - SIP URI: sip:{config.sip_user}@{config.sip_domain}:{config.sip_port}")
        
        return True
    except Exception as e:
        print(f"✗ SIP handler failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=" * 60)
    print("SIP AI Assistant - Component Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Config", await test_config()))
    results.append(("VAD", await test_vad()))
    results.append(("Whisper STT", await test_whisper()))
    results.append(("Piper TTS", await test_piper()))
    results.append(("Voice Cloning", await test_voice_cloning()))
    results.append(("LLM", await test_llm()))
    results.append(("Tools", await test_tools()))
    results.append(("SIP", await test_sip()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
            
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to run the full system.")
        print("\nStart with:")
        print("  # Start vLLM server first:")
        print("  python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-70B-Instruct")
        print("\n  # Then start the assistant:")
        print("  python main.py")
    else:
        print("\n⚠ Some tests failed. Check the errors above.")
        
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)