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
        await self._stream_response(call, greeting)
        
        # Start listening loop
        await self._conversation_loop(call)
        
    async def _conversation_loop(self, call):
        """
        Main conversation loop with Smart Barging (Echo Prevention).
        """
        from sip_handler import PlaylistPlayer, PJSUA_AVAILABLE

        # Initialize player if needed
        if PJSUA_AVAILABLE and call.pj_call and call.pj_call.aud_med:
            if not hasattr(call, 'stream_player') or call.stream_player is None:
                call.stream_player = PlaylistPlayer(call.pj_call.aud_med)
        
        # 1. Reset Pipeline on Entry (Clear any noise from connection)
        self.audio_pipeline.vad.reset()
        self.audio_pipeline.audio_buffer.clear()
        
        logger.info("Listening...")

        import random
        acknowledgements = ["Okay", "Sure", "Got it", "One moment", "Copy that", "On it"]

        while call.is_active and self.running:
            try:
                # 1. Receive Audio
                audio_data = await self.sip_handler.receive_audio(call, timeout=0.05)
                if audio_data is None:
                    continue

                # 2. Check Bot State
                is_bot_speaking = False
                if hasattr(call, 'stream_player') and call.stream_player:
                    is_bot_speaking = call.stream_player.is_playing

                # 3. Process Audio
                if is_bot_speaking:
                    # --- SMART BARGE-IN MODE (With Energy Gate) ---
                    import numpy as np
                    
                    # 1. Calculate Energy (Volume)
                    # Convert bytes to 16-bit integers
                    samples = np.frombuffer(audio_data, dtype=np.int16)
                    # Calculate RMS (Root Mean Square) energy
                    if len(samples) > 0:
                        energy = np.sqrt(np.mean(samples.astype(np.float32)**2))
                    else:
                        energy = 0

                    # 2. Check conditions: Is it Speech? AND Is it Loud enough?
                    # We check energy FIRST to filter out quiet echoes
                    if energy > self.config.barge_in_energy_threshold and self.audio_pipeline.vad.is_speech(audio_data):
                        
                        # Add to counter
                        if not hasattr(self, '_barge_in_counter'): self._barge_in_counter = 0
                        self._barge_in_counter += len(audio_data)
                        
                        # Calculate duration
                        barge_ms = (self._barge_in_counter / (self.config.sample_rate * 2)) * 1000
                        
                        if barge_ms >= self.config.barge_in_min_duration_ms:
                            logger.info(f"Barge-in! Energy={energy:.0f}, Duration={barge_ms:.0f}ms")
                            call.stream_player.stop_all()
                            self._barge_in_counter = 0
                    else:
                        # Reset if user goes silent OR drops below volume threshold
                        # This prevents "accumulating" brief noises over time
                        self._barge_in_counter = 0
                        
                else:
                    # --- NORMAL LISTENING MODE ---
                    # Bot is silent. Record normally.
                    
                    # We use process_audio which handles buffering and end-of-utterance
                    transcription = await self.audio_pipeline.process_audio(audio_data)
                    
                    if transcription and len(transcription.strip()) >= 2:
                        logger.info(f"User said: {transcription}")

                        # --- NEW: Immediate Acknowledgement ---
                        # Only do this for "commands" (longer phrases) to avoid being weird on "Hi"
                        # Simple heuristic: > 2 words or contains command verbs
                        is_command = len(transcription.split()) > 2 or \
                                     any(w in transcription.lower() for w in ['call', 'set', 'timer', 'remind', 'cancel'])
                        
                        if is_command:
                            filler = random.choice(acknowledgements)
                            logger.info(f"Playing filler: {filler}")
                            # This enqueues audio immediately while the LLM thinks
                            await self._stream_response(call, filler)
                        # --------------------------------------
                        else:
                            await self._stream_response(call, "Checking")
                        
                        # Add to history
                        self.conversation_history.append({
                            "role": "user",
                            "content": transcription
                        })

                        # Generate Response
                        response_text = await self.llm_engine.generate_response(
                            self.conversation_history,
                            call_context={
                                "remote_uri": call.remote_uri,
                                "duration": call.duration
                            }
                        )
                        # --- FIX: Check for empty response ---
                        if not response_text:
                            logger.warning("No response generated. Clearing buffer and continuing.")
                            self.audio_pipeline.audio_buffer.clear()
                            self.audio_pipeline.vad.reset()
                            continue
                        
                        logger.info(f"Assistant: {response_text}")
                        self.conversation_history.append({
                            "role": "assistant", 
                            "content": response_text
                        })

                        # Check Hangup
                        if self._should_hangup(transcription, response_text):
                            await self._stream_response(call, "Goodbye!")
                            await asyncio.sleep(2.0)
                            await self.sip_handler.hangup_call(call)
                            break

                        # Speak
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
    async def schedule_callback(self, delay_seconds: int, message: str, destination: str = None):
        """
        Schedule a callback, optionally to a specific number.
        """
        target_uri = destination
        
        # 1. Fallback to current caller if no destination provided
        if not target_uri and self.current_call:
             target_uri = self.current_call.remote_uri
        
        if not target_uri:
            logger.warning("Callback requested but no destination available")
            return

        # 2. Format as SIP URI if it's just a number (e.g. "405")
        # Remove any spaces or hyphens first
        target_uri = target_uri.replace("-", "").replace(" ", "")
        
        if "sip:" not in target_uri:
             # Construct full URI: sip:NUMBER@DOMAIN
             target_uri = f"sip:{target_uri}@{self.config.sip_domain}"
        
        logger.info(f"Scheduling callback to {target_uri} in {delay_seconds}s")
        
        # 3. Pass to ToolManager
        await self.tool_manager.schedule_callback(
            delay_seconds, 
            message,
            target_uri,
        )
        
    async def make_outbound_call(self, uri: str, message: str):
        """
        Make an outbound call and transition to a full interactive session.
        """
        logger.info(f"Initiating callback to {uri}...")
        
        call = await self.sip_handler.make_call(uri)
        if not call:
            logger.error("Failed to create call")
            return

        # Ringing Loop (Max 30s)
        logger.info("Ringing...")
        timeout = 30
        start_time = time.time()
        answered = False
        
        while time.time() - start_time < timeout:
            if call.is_active:
                answered = True
                break
            if call.call_id not in self.sip_handler.active_calls:
                logger.info("Call rejected or failed.")
                return
            await asyncio.sleep(0.5)

        if answered:
            logger.info("Call answered! Starting interactive session...")
            
            self.current_call = call
            self.conversation_history = []
            
            # Init Player
            from sip_handler import PlaylistPlayer, PJSUA_AVAILABLE
            if PJSUA_AVAILABLE and call.pj_call and call.pj_call.aud_med:
                 if not hasattr(call, 'stream_player') or call.stream_player is None:
                    call.stream_player = PlaylistPlayer(call.pj_call.aud_med)

            # --- FIX: Stabilize Audio ---
            # Wait for RTP to stabilize
            await asyncio.sleep(1.0)
            
            # Drain the VAD buffer so previous ringback tone isn't counted as user speech
            self.audio_pipeline.vad.reset()
            self.audio_pipeline.audio_buffer.clear()
            if hasattr(self, '_barge_in_counter'):
                self._barge_in_counter = 0
            # ----------------------------

            # Play the message
            await self._stream_response(call, message)
            
            # Enter loop
            await self._conversation_loop(call)
            
        else:
            logger.info("No answer after 30 seconds. Hanging up.")
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