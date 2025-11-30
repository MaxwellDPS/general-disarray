"""
WebSocket Realtime Client for Speaches API
===========================================
Uses the /v1/realtime WebSocket endpoint for low-latency streaming STT.

This implements the OpenAI Realtime API protocol which Speaches supports:
- WebSocket connection to /v1/realtime?model=<model>&intent=transcription
- Send audio via input_audio_buffer.append events
- Receive transcriptions via conversation.item.input_audio_transcription.completed events

This provides significantly lower latency compared to batch transcription
by streaming audio in real-time and receiving transcription results as they're ready.
"""

import asyncio
import base64
import json
import logging
import time
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
import urllib.parse

try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketClientProtocol = None

from config import Config
from telemetry import create_span, Metrics
from logging_utils import (
    log_event, 
    RECONNECT_BASE_DELAY_SECONDS, 
    RECONNECT_MAX_DELAY_SECONDS
)

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from realtime transcription."""
    text: str
    is_final: bool
    item_id: Optional[str] = None
    confidence: float = 1.0
    latency_ms: float = 0.0


class RealtimeWebSocketClient:
    """
    WebSocket-based realtime STT client for Speaches API.
    
    Uses the /v1/realtime WebSocket endpoint with OpenAI Realtime API protocol
    for streaming audio and receiving transcriptions in real-time.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.base_url = config.speaches_api_url.rstrip('/')
        self.model = config.whisper_model
        self.language = config.whisper_language
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._session_id: Optional[str] = None
        
        self._transcription_callback: Optional[Callable[[TranscriptionResult], Awaitable[None]]] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Audio buffering for batch fallback
        self._audio_buffer = bytearray()
        self._last_audio_time = 0.0
        
        # Track connection state for metrics
        self._connection_attempts = 0
        self._last_connection_error: Optional[str] = None
        
        self.available = WEBSOCKETS_AVAILABLE
        
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("websockets package not installed - WebSocket realtime mode unavailable")
            logger.warning("Install with: pip install websockets")
            
    def _build_ws_url(self) -> str:
        """Build the WebSocket URL with query parameters."""
        # Convert http(s) to ws(s)
        ws_base = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        
        # Build query parameters
        params = {
            'model': self.model,
            'intent': 'transcription',  # Transcription-only mode
        }
        if self.language:
            params['language'] = self.language
            
        query_string = urllib.parse.urlencode(params)
        return f"{ws_base}/v1/realtime?{query_string}"
            
    async def initialize(self):
        """Initialize the WebSocket client and establish connection."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("Cannot initialize realtime client - websockets not available")
            return
            
        # Establish WebSocket connection
        await self._connect()
        
    async def _connect(self):
        """Establish WebSocket connection to Speaches realtime endpoint."""
        if not WEBSOCKETS_AVAILABLE:
            return
            
        self._connection_attempts += 1
        Metrics.record_realtime_connection_attempt()
        
        try:
            ws_url = self._build_ws_url()
            logger.info(f"Connecting to Speaches realtime API: {ws_url}")
            
            # Connect with reasonable timeouts
            self._ws = await websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=10 * 1024 * 1024,  # 10MB max message size
            )
            
            self._connected = True
            self._last_connection_error = None
            Metrics.record_realtime_connection_state("connected")
            logger.info("WebSocket connection established with Speaches realtime API")
            
            # Start receiving messages
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            # Send session configuration
            await self._configure_session()
            
        except Exception as e:
            self._connected = False
            self._last_connection_error = str(e)
            Metrics.record_realtime_connection_state("failed")
            Metrics.record_realtime_connection_error(type(e).__name__)
            logger.error(f"WebSocket connection error: {e}")
            self.available = False
            
    async def _configure_session(self):
        """Send session configuration after connection."""
        if not self._ws or not self._connected:
            return
            
        # Configure the session for transcription
        session_config = {
            "type": "session.update",
            "session": {
                "input_audio_transcription": {
                    "model": self.model
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": self.config.silence_duration_ms
                }
            }
        }
        
        if self.language:
            session_config["session"]["input_audio_transcription"]["language"] = self.language
            
        await self._ws.send(json.dumps(session_config))
        logger.debug("Session configuration sent")
        
    async def _receive_loop(self):
        """Receive and process messages from the WebSocket."""
        if not self._ws:
            return
            
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message[:100]}")
                    
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self._connected = False
            Metrics.record_realtime_connection_state("disconnected")
            await self._handle_connection_failure()
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            self._connected = False
            Metrics.record_realtime_connection_error(type(e).__name__)
            await self._handle_connection_failure()
            
    async def _handle_message(self, data: dict):
        """Handle a message from the WebSocket."""
        msg_type = data.get("type", "")
        
        if msg_type == "session.created":
            self._session_id = data.get("session", {}).get("id")
            logger.info(f"Realtime session created: {self._session_id}")
            
        elif msg_type == "session.updated":
            logger.debug("Session configuration updated")
            
        elif msg_type == "input_audio_buffer.speech_started":
            log_event(logger, logging.DEBUG, "Speech started detected",
                     event="realtime_speech_started")
            
        elif msg_type == "input_audio_buffer.speech_stopped":
            log_event(logger, logging.DEBUG, "Speech stopped detected",
                     event="realtime_speech_stopped")
            
        elif msg_type == "input_audio_buffer.committed":
            logger.debug("Audio buffer committed")
            
        elif msg_type == "conversation.item.input_audio_transcription.completed":
            # This is the final transcription result
            transcript = data.get("transcript", "")
            item_id = data.get("item_id")
            
            if transcript:
                result = TranscriptionResult(
                    text=transcript,
                    is_final=True,
                    item_id=item_id
                )
                
                log_event(logger, logging.DEBUG, f"Transcription received: {transcript[:50]}...",
                         event="realtime_transcription", text_length=len(transcript))
                
                # Record metrics
                Metrics.record_stt_latency(0, self.model)  # Latency tracked elsewhere
                
                if self._transcription_callback:
                    await self._transcription_callback(result)
                    
        elif msg_type == "conversation.item.input_audio_transcription.delta":
            # Partial transcription (streaming)
            delta = data.get("delta", "")
            if delta and self._transcription_callback:
                result = TranscriptionResult(
                    text=delta,
                    is_final=False,
                    item_id=data.get("item_id")
                )
                await self._transcription_callback(result)
                
        elif msg_type == "error":
            error = data.get("error", {})
            error_msg = error.get("message", "Unknown error")
            error_type = error.get("type", "unknown")
            logger.error(f"Realtime API error: {error_type} - {error_msg}")
            Metrics.record_stt_error(self.model, f"realtime_{error_type}")
            
        else:
            logger.debug(f"Unhandled message type: {msg_type}")
            
    async def _handle_connection_failure(self):
        """Handle WebSocket connection failure - attempt reconnect."""
        logger.warning("WebSocket connection failed, attempting reconnect...")
        self._connected = False
        
        if self._reconnect_task is None or self._reconnect_task.done():
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())
            
    async def _reconnect_loop(self):
        """Attempt to reconnect with exponential backoff."""
        backoff = RECONNECT_BASE_DELAY_SECONDS
        
        while not self._connected:
            try:
                await asyncio.sleep(backoff)
                await self._connect()
                if self._connected:
                    logger.info("WebSocket reconnection successful")
                    Metrics.record_realtime_reconnection()
                    return
            except Exception as e:
                logger.warning(f"Reconnection attempt failed: {e}")
                
            backoff = min(backoff * 2, RECONNECT_MAX_DELAY_SECONDS)
            
    def set_transcription_callback(self, callback: Callable[[TranscriptionResult], Awaitable[None]]):
        """Set callback for receiving transcription results."""
        self._transcription_callback = callback
        
    async def push_audio(self, audio_data: bytes):
        """
        Push audio data to be transcribed via WebSocket.
        
        Audio should be 16-bit PCM at the configured sample rate.
        """
        if not self._ws or not self._connected:
            return
            
        try:
            # Encode audio as base64 for the OpenAI Realtime API format
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            message = {
                "type": "input_audio_buffer.append",
                "audio": audio_base64
            }
            
            await self._ws.send(json.dumps(message))
            self._last_audio_time = time.time()
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket closed while sending audio")
            self._connected = False
            await self._handle_connection_failure()
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            
    async def commit_audio_buffer(self):
        """
        Commit the current audio buffer for transcription.
        
        Call this when you want to force transcription of buffered audio
        (e.g., after silence is detected).
        """
        if not self._ws or not self._connected:
            return
            
        try:
            message = {"type": "input_audio_buffer.commit"}
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error committing audio buffer: {e}")
            
    async def clear_audio_buffer(self):
        """Clear the audio buffer without transcribing."""
        if not self._ws or not self._connected:
            return
            
        try:
            message = {"type": "input_audio_buffer.clear"}
            await self._ws.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error clearing audio buffer: {e}")
            
    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe audio using the realtime connection.
        
        In realtime mode, audio should already be streaming via push_audio().
        This method is called when local VAD detects end-of-utterance.
        
        We DON'T commit the buffer here because the server VAD handles speech
        detection. Instead, we just wait for any pending transcription result.
        If no result comes (because server VAD timing differs), fall back to
        returning empty string and let the caller handle it.
        
        For cases where we need forced transcription (like hangup), the caller
        should use commit_audio_buffer() first.
        """
        if not self._connected or not self._ws:
            logger.warning("Realtime connection not available")
            Metrics.record_stt_error(self.model, "connection_unavailable")
            return ""
            
        with create_span("stt.transcribe.realtime", {
            "stt.model": self.model,
            "stt.mode": "realtime",
            "audio.bytes": len(audio_data)
        }) as span:
            start_time = time.time()
            result_text = ""
            result_event = asyncio.Event()
            
            async def capture_result(result: TranscriptionResult):
                nonlocal result_text
                if result.is_final:
                    result_text = result.text
                    result_event.set()
                    
            # Temporarily set callback
            old_callback = self._transcription_callback
            self._transcription_callback = capture_result
            
            try:
                # If audio_data is provided, send it first
                # (This handles the case where audio wasn't already streamed)
                if audio_data and len(audio_data) > 0:
                    await self.push_audio(audio_data)
                
                # Wait for transcription result with a short timeout
                # Server VAD should have already triggered transcription if speech was detected
                try:
                    await asyncio.wait_for(result_event.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    # No transcription came back - server VAD may not have detected speech end yet
                    # This is okay in streaming mode - transcription will come via callback
                    logger.debug("Realtime transcription wait timeout - server VAD may have different timing")
                    span.set_attribute("stt.timeout", True)
                    
            finally:
                self._transcription_callback = old_callback
                
            latency_ms = (time.time() - start_time) * 1000
            span.set_attribute("stt.latency_ms", latency_ms)
            span.set_attribute("stt.text_length", len(result_text))
            
            if result_text:
                Metrics.record_stt_latency(latency_ms, self.model)
            
            return result_text
            
    async def close(self):
        """Close the WebSocket connection."""
        self._connected = False
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
                
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
                
        if self._ws:
            await self._ws.close()
            self._ws = None
            
        Metrics.record_realtime_connection_state("closed")
        logger.info("Realtime WebSocket client closed")


class RealtimeSTTManager:
    """
    Manager for STT that can use either realtime (WebSocket) or batch mode.
    
    Provides a unified interface regardless of the underlying mode.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.mode = config.stt_mode
        
        self._realtime_client: Optional[RealtimeWebSocketClient] = None
        self._batch_client = None  # WhisperAPIClient - imported lazily to avoid circular imports
        
        self._transcription_callback: Optional[Callable[[str], Awaitable[None]]] = None
        
    async def initialize(self):
        """Initialize the appropriate STT client based on configuration."""
        if self.config.use_realtime_stt and WEBSOCKETS_AVAILABLE:
            logger.info("Initializing STT in realtime (WebSocket) mode")
            self._realtime_client = RealtimeWebSocketClient(self.config)
            await self._realtime_client.initialize()
            
            if self._realtime_client.available and self._realtime_client._connected:
                # Set up callback wrapper
                async def callback_wrapper(result: TranscriptionResult):
                    if result.is_final and result.text and self._transcription_callback:
                        await self._transcription_callback(result.text)
                        
                self._realtime_client.set_transcription_callback(callback_wrapper)
                logger.info("Realtime STT initialized successfully")
                Metrics.record_stt_mode("realtime")
                return
            else:
                logger.warning("Realtime STT unavailable, falling back to batch mode")
                
        # Fallback to batch mode
        logger.info("Initializing STT in batch mode")
        Metrics.record_stt_mode("batch")
        from audio_pipeline import WhisperAPIClient
        self._batch_client = WhisperAPIClient(self.config)
        await self._batch_client.initialize()
        
    @property
    def available(self) -> bool:
        """Check if STT is available."""
        if self._realtime_client and self._realtime_client.available and self._realtime_client._connected:
            return True
        if self._batch_client and self._batch_client.available:
            return True
        return False
        
    @property 
    def is_realtime(self) -> bool:
        """Check if using realtime mode."""
        return (self._realtime_client is not None and 
                self._realtime_client.available and 
                self._realtime_client._connected)
        
    def set_transcription_callback(self, callback: Callable[[str], Awaitable[None]]):
        """Set callback for receiving transcription results (realtime mode only)."""
        self._transcription_callback = callback
        
    async def push_audio(self, audio_data: bytes):
        """Push audio for realtime transcription."""
        if self._realtime_client and self._realtime_client._connected:
            await self._realtime_client.push_audio(audio_data)
            
    async def commit_audio(self):
        """Commit audio buffer for transcription (realtime mode)."""
        if self._realtime_client and self._realtime_client._connected:
            await self._realtime_client.commit_audio_buffer()
            
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcribe audio using the active client."""
        if self._realtime_client and self._realtime_client._connected:
            return await self._realtime_client.transcribe(audio_data)
        elif self._batch_client and self._batch_client.available:
            return await self._batch_client.transcribe(audio_data)
        else:
            logger.error("No STT client available")
            return ""
            
    async def close(self):
        """Close all clients."""
        if self._realtime_client:
            await self._realtime_client.close()
        if self._batch_client:
            await self._batch_client.close()
