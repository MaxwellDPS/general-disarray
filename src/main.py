#!/usr/bin/env python3
"""
Local SIP AI Assistant
======================
A fully local voice AI assistant that communicates via SIP.
Designed for GB10 with <=100GB vRAM, single user at a time.

Components:
- SIP handling via PJSUA2
- Speech-to-Text via faster-whisper
- LLM via vLLM (Llama 3.1 70B or similar)
- Text-to-Speech via Piper TTS
- Tool system for callbacks, timers, and extensible functions
"""

import asyncio
import signal
import sys
import time
import logging
from pathlib import Path

from config import Config
from sip_handler import SIPHandler
from audio_pipeline import AudioPipeline
from llm_engine import LLMEngine
from tool_manager import ToolManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sip_assistant.log')
    ]
)
logger = logging.getLogger(__name__)


class SIPAIAssistant:
    """Main application orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.running = False
        
        # Initialize components
        logger.info("Initializing SIP AI Assistant...")
        
        self.tool_manager = ToolManager(self)
        self.llm_engine = LLMEngine(config, self.tool_manager)
        self.audio_pipeline = AudioPipeline(config)
        self.sip_handler = SIPHandler(config, self._on_call_received)
        
        # Conversation state
        self.conversation_history = []
        self.current_call = None
        
    async def start(self):
        """Start all components."""
        logger.info("Starting SIP AI Assistant...")
        self.running = True
        
        # Start components
        await self.llm_engine.start()
        await self.audio_pipeline.start()
        await self.sip_handler.start()
        await self.tool_manager.start()
        
        logger.info("SIP AI Assistant is ready and listening!")
        logger.info(f"SIP URI: sip:{self.config.sip_user}@{self.config.sip_domain}")
        
        # Keep running
        while self.running:
            await asyncio.sleep(1)
            
    async def stop(self):
        """Stop all components gracefully."""
        logger.info("Stopping SIP AI Assistant...")
        self.running = False
        
        await self.tool_manager.stop()
        await self.sip_handler.stop()
        await self.audio_pipeline.stop()
        await self.llm_engine.stop()
        
        logger.info("SIP AI Assistant stopped.")
        
    async def _on_call_received(self, call):
        """Handle incoming call."""
        logger.info(f"Incoming call from: {call.remote_uri}")
        self.current_call = call
        self.conversation_history = []
        
        # Call is already answered by SIP handler (auto-answer in PJSUA2 thread)
        # Wait for media to be ready before playing greeting
        wait_start = time.time()
        while not call.media_ready and time.time() - wait_start < 5.0:
            await asyncio.sleep(0.1)
            
        if not call.media_ready:
            logger.warning("Media not ready after 5 seconds")
            return
            
        logger.info("Media ready, playing greeting...")
        
        # Play greeting
        greeting = await self.llm_engine.generate_greeting()
        logger.info(f"Generated greeting: {greeting}")
        await self._speak(greeting)
        
        # Start listening loop
        await self._conversation_loop(call)
        
    async def _conversation_loop(self, call):
        """
        Main conversation loop with streaming audio and barge-in support.
        """
        # Lazy import to avoid circular dependency if sip_handler imports main
        from sip_handler import PlaylistPlayer, PJSUA_AVAILABLE

        # Initialize the PlaylistPlayer if this is a real PJSUA call
        if PJSUA_AVAILABLE and call.pj_call and call.pj_call.aud_med:
            if not hasattr(call, 'stream_player') or call.stream_player is None:
                try:
                    call.stream_player = PlaylistPlayer(call.pj_call.aud_med)
                    logger.info("PlaylistPlayer initialized for streaming")
                except Exception as e:
                    logger.warning(f"Could not init PlaylistPlayer: {e}")
        
        logger.info("Listening...")

        while call.is_active and self.running:
            try:
                # 1. Listen for audio (Non-blocking check)
                # We use a short timeout so we can check other flags/states often
                audio_data = await self.sip_handler.receive_audio(call, timeout=0.05)
                
                if audio_data is None:
                    continue

                # 2. Process Audio (VAD + Buffering)
                # We use process_audio instead of transcribe directly.
                # It buffers chunks and only returns text when a sentence finishes.
                transcription = await self.audio_pipeline.process_audio(audio_data)
                
                # Check if the user is currently speaking (for Barge-In)
                if self.audio_pipeline.vad.is_speaking:
                    # --- BARGE-IN TRIGGER ---
                    # If the bot is currently talking, SHUT IT UP.
                    if hasattr(call, 'stream_player') and call.stream_player:
                        if call.stream_player.is_playing:
                            logger.info("Barge-in detected! Stopping playback.")
                            call.stream_player.stop_all()

                # If no complete sentence yet, keep listening
                if not transcription or len(transcription.strip()) < 2:
                    continue

                logger.info(f"User said: {transcription}")
                
                # Add to history
                self.conversation_history.append({
                    "role": "user",
                    "content": transcription
                })

                # 3. Generate Response
                # Note: We generate the full text first, then stream the TTS.
                # (You can upgrade this to stream tokens from the LLM later)
                response_text = await self.llm_engine.generate_response(
                    self.conversation_history,
                    call_context={
                        "remote_uri": call.remote_uri,
                        "call_id": call.call_id,
                        "duration": call.duration
                    }
                )
                
                logger.info(f"Assistant: {response_text}")

                # Add to history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response_text
                })

                # Check for hangup
                if self._should_hangup(transcription, response_text):
                    await self._stream_response(call, "Goodbye!")
                    # Wait briefly for audio to queue before hanging up
                    await asyncio.sleep(2.0)
                    await self.sip_handler.hangup_call(call)
                    break

                # 4. Stream Audio Response
                # This yields chunks of audio to play immediately
                await self._stream_response(call, response_text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                import traceback
                traceback.print_exc()
                
        # Cleanup
        if hasattr(call, 'stream_player') and call.stream_player:
            call.stream_player.stop_all()
        self.current_call = None
        self.conversation_history = []

    async def _stream_response(self, call, text: str):
        """
        Synthesize text into audio chunks and feed them to the playlist player.
        """
        # Fallback for mock mode or if player init failed
        if not hasattr(call, 'stream_player') or not call.stream_player:
            await self._speak(text)
            return

        import tempfile
        import os
        import wave

        # Assuming audio_pipeline.tts has the synthesize_stream method 
        # (as discussed in the previous turn)
        try:
            async for audio_chunk in self.audio_pipeline.tts.synthesize_stream(text):
                # If call ended mid-stream, stop generating
                if not call.is_active:
                    break
                    
                # If user barged in (player stopped), stop generating
                if not call.stream_player.is_playing and call.stream_player.queue.empty():
                    # Check if it was a barge-in (force stop) vs just buffer underrun
                    # For simplicity, we continue unless explicitly cancelled, 
                    # but you could add a 'interrupted' flag to the call object.
                    pass

                # Write chunk to a temp WAV file
                # The PlaylistPlayer is responsible for deleting this file!
                fd, path = tempfile.mkstemp(suffix='.wav', prefix='stream_')
                os.close(fd)
                
                with wave.open(path, 'wb') as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2) # 16-bit
                    wav.setframerate(self.config.sample_rate)
                    wav.writeframes(audio_chunk)
                
                # Hand off to the specific call's player
                call.stream_player.enqueue_file(path)
                
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            import traceback
            traceback.print_exc()
        
    async def _speak(self, text: str):
        """Convert text to speech and send to call."""
        if not self.current_call or not self.current_call.is_active:
            return
            
        # Generate audio
        audio_data = await self.audio_pipeline.synthesize(text)
        
        # Send to call
        await self.sip_handler.send_audio(self.current_call, audio_data)
        
    def _should_hangup(self, user_text: str, response: str) -> bool:
        """Check if we should end the call."""
        hangup_phrases = [
            "goodbye", "bye", "hang up", "end call",
            "that's all", "thanks bye", "talk later"
        ]
        user_lower = user_text.lower()
        return any(phrase in user_lower for phrase in hangup_phrases)
        
    # Methods for tools to call back into
    async def schedule_callback(self, delay_seconds: int, message: str):
        """Schedule a callback (used by timer tool)."""
        await self.tool_manager.schedule_callback(
            delay_seconds, 
            self.current_call.remote_uri if self.current_call else None,
            message
        )
        
    async def make_outbound_call(self, uri: str, message: str):
        """Make an outbound call (used by callback tool)."""
        call = await self.sip_handler.make_call(uri)
        if call:
            await self._speak(message)
            await asyncio.sleep(2)
            await self.sip_handler.hangup_call(call)


async def main():
    """Main entry point."""
    config = Config()
    assistant = SIPAIAssistant(config)
    
    # Handle signals for graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        logger.info("Received shutdown signal")
        asyncio.create_task(assistant.stop())
        
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)
        
    try:
        await assistant.start()
    except KeyboardInterrupt:
        pass
    finally:
        await assistant.stop()


if __name__ == "__main__":
    asyncio.run(main())