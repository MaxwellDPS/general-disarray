"""
SIP Handler using PJSUA2
========================
Handles SIP registration, incoming/outgoing calls, and audio streams.
"""

import asyncio
import logging
import struct
import time
import queue
from dataclasses import dataclass
from typing import Callable, Optional, Dict, Any
from collections import deque
import threading
import numpy as np

try:
    import pjsua2 as pj
    PJSUA_AVAILABLE = True
except ImportError:
    PJSUA_AVAILABLE = False
    print("WARNING: pjsua2 not available, using mock SIP handler")

from config import Config

logger = logging.getLogger(__name__)

# Thread-local storage for pjlib thread registration
_thread_registered = threading.local()

def ensure_thread_registered():
    """Ensure current thread is registered with pjlib."""
    if not PJSUA_AVAILABLE:
        return
        
    # We use a try/catch block because asking if registered can throw 
    # if the library is in a weird state, but usually it's safe.
    try:
        if not pj.Endpoint.instance().libIsThreadRegistered():
            pj.Endpoint.instance().libRegisterThread(threading.current_thread().name)
    except Exception as e:
        logger.debug(f"Thread registration check failed: {e}")


@dataclass
class CallInfo:
    """Information about an active call."""
    call_id: str
    remote_uri: str
    is_active: bool
    start_time: float
    pj_call: Any = None  # pjsua2 Call object
    audio_buffer: deque = None
    
    # Audio file paths for this call
    record_file: Optional[str] = None
    playback_queue: deque = None
    
    # Media state
    media_ready: bool = False
    
    def __post_init__(self):
        if self.audio_buffer is None:
            self.audio_buffer = deque(maxlen=1000)  # ~10 seconds at 100 chunks/sec
        if self.playback_queue is None:
            self.playback_queue = deque()
            
    @property
    def duration(self) -> float:
        """Call duration in seconds."""
        return time.time() - self.start_time


class SIPCall(pj.Call if PJSUA_AVAILABLE else object):
    """Custom call class with callbacks."""
    
    def __init__(self, acc, call_id: int, handler: 'SIPHandler'):
        if PJSUA_AVAILABLE:
            super().__init__(acc, call_id)
        self.handler = handler
        self.call_info: Optional[CallInfo] = None
        self.aud_med: Any = None  # pjsua2 AudioMedia
        self.recorder: Any = None  # AudioMediaRecorder
        self.player: Any = None  # AudioMediaPlayer
        
    def onCallState(self, prm):
        """Called when call state changes."""
        ci = self.getInfo()
        logger.info(f"Call state: {ci.stateText}")
        
        if ci.state == pj.PJSIP_INV_STATE_CONFIRMED:
            # Call connected
            if self.call_info:
                self.call_info.is_active = True
                
        elif ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
            # Call ended
            if self.call_info:
                self.call_info.is_active = False
            self._cleanup_media()
            self.handler._on_call_ended(self)
            
    def onCallMediaState(self, prm):
        """Called when media state changes."""
        ci = self.getInfo()
        
        for mi in ci.media:
            if mi.type == pj.PJMEDIA_TYPE_AUDIO and \
               mi.status == pj.PJSUA_CALL_MEDIA_ACTIVE:
                
                # Get the call's audio media
                self.aud_med = self.getAudioMedia(mi.index)
                
                # Set up recording (incoming audio from remote)
                try:
                    import tempfile
                    import os
                    
                    # Create temp file for recording
                    fd, record_path = tempfile.mkstemp(suffix='.wav', prefix='sip_in_')
                    os.close(fd)
                    
                    if self.call_info:
                        self.call_info.record_file = record_path
                    
                    # Create recorder and connect
                    self.recorder = pj.AudioMediaRecorder()
                    self.recorder.createRecorder(record_path)
                    self.aud_med.startTransmit(self.recorder)
                    
                    logger.info(f"Recording to: {record_path}")
                except Exception as e:
                    logger.error(f"Failed to set up recording: {e}")
                
                # Mark media as ready
                if self.call_info:
                    self.call_info.media_ready = True
                    logger.info("Audio media ready for playback")
                
                logger.info("Audio media connected")
                
    def _cleanup_media(self):
        """Clean up media resources safely."""
        try:
            # Stop recorder
            if self.recorder:
                # Only stop if media is actually active
                if self.aud_med: 
                    try:
                        self.aud_med.stopTransmit(self.recorder)
                    except Exception: 
                        pass # Ignore errors on cleanup
                self.recorder = None
                
            # Stop player
            if self.player:
                if self.aud_med:
                    try:
                        self.player.stopTransmit(self.aud_med)
                    except Exception:
                        pass
                self.player = None
                
            # Clean up stream player (Playlist)
            if hasattr(self, 'stream_player') and self.stream_player:
                self.stream_player.stop_all()
                self.stream_player = None

            self.aud_med = None
            
            # Clean up recording file
            if self.call_info and self.call_info.record_file:
                try:
                    import os
                    if os.path.exists(self.call_info.record_file):
                        os.unlink(self.call_info.record_file)
                except Exception:
                    pass
                self.call_info.record_file = None
                
        except Exception as e:
            logger.debug(f"Error during media cleanup: {e}")
            
    def play_audio_file(self, wav_path: str):
        """Play a WAV file to the remote party."""
        logger.info(f"play_audio_file called with: {wav_path}")
        
        if not self.aud_med:
            logger.warning("No audio media available for playback")
            return False
        
        # Check file exists and has content
        try:
            import os
            file_size = os.path.getsize(wav_path)
            logger.info(f"Audio file size: {file_size} bytes")
            if file_size < 100:
                logger.warning(f"Audio file too small: {file_size} bytes")
        except Exception as e:
            logger.error(f"Cannot check audio file: {e}")
            
        try:
            # Create player for this file
            player = pj.AudioMediaPlayer()
            player.createPlayer(wav_path, pj.PJMEDIA_FILE_NO_LOOP)
            
            # Connect player to the call's audio media
            player.startTransmit(self.aud_med)
            
            # Store for cleanup
            self.player = player
            
            logger.info(f"Audio playback started: {wav_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            import traceback
            traceback.print_exc()
            return False


class SIPAccount(pj.Account if PJSUA_AVAILABLE else object):
    """Custom account class with callbacks."""
    
    def __init__(self, handler: 'SIPHandler'):
        if PJSUA_AVAILABLE:
            super().__init__()
        self.handler = handler
        
    def onRegState(self, prm):
        """Called when registration state changes."""
        ai = self.getInfo()
        logger.info(f"Registration state: {ai.regStatusText} (code={ai.regStatus})")
        
        if ai.regStatus == 200:
            logger.info(f"✓ Successfully registered with SIP server")
            logger.info(f"  URI: {ai.uri}")
            if ai.regIsActive:
                logger.info(f"  Registration active, expires in {ai.regExpiresSec}s")
            # Signal successful registration
            self.handler._registered.set()
        elif ai.regStatus >= 400:
            logger.error(f"✗ Registration failed: {ai.regStatusText}")
            if ai.regStatus == 401 or ai.regStatus == 407:
                logger.error("  Check SIP_USER and SIP_PASSWORD credentials")
            elif ai.regStatus == 403:
                logger.error("  Forbidden - check Asterisk ACL/permissions")
            elif ai.regStatus == 404:
                logger.error("  Not found - check SIP_USER exists on Asterisk")
            elif ai.regStatus == 408:
                logger.error("  Request timeout - check network/firewall")
        elif ai.regStatus == 0:
            # Unregistered or in progress
            pass
        
    def onIncomingCall(self, prm):
        """Called when there's an incoming call."""
        call = SIPCall(self, prm.callId, self.handler)
        
        # Get call info
        ci = call.getInfo()
        logger.info(f"Incoming call from: {ci.remoteUri}")
        
        call_info = CallInfo(
            call_id=str(prm.callId),
            remote_uri=ci.remoteUri,
            is_active=False,
            start_time=time.time(),
            pj_call=call
        )
        call.call_info = call_info
        
        # Store the call
        self.handler.active_calls[call_info.call_id] = call
        
        # Answer the call immediately in this thread (PJSUA2 thread)
        # This prevents the async thread from needing to call back into PJSUA2
        try:
            answer_prm = pj.CallOpParam()
            answer_prm.statusCode = 200
            call.answer(answer_prm)
            call_info.is_active = True
            logger.info(f"Auto-answered call {call_info.call_id}")
        except Exception as e:
            logger.error(f"Error answering call: {e}")
        
        # Notify handler asynchronously (call is already answered)
        if self.handler.loop:
            asyncio.run_coroutine_threadsafe(
                self.handler._handle_incoming_call(call),
                self.handler.loop
            )


class SIPHandler:
    """Manages SIP communication."""
    
    def __init__(self, config: Config, on_call_callback: Callable):
        self.config = config
        self.on_call_callback = on_call_callback
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        
        self.endpoint: Optional[pj.Endpoint] = None
        self.account: Optional[SIPAccount] = None
        self.active_calls: Dict[str, SIPCall] = {}
        
        self._running = False
        self._pj_thread: Optional[threading.Thread] = None
        self._initialized = threading.Event()
        self._registered = threading.Event()
        
        # Command queue for thread-safe PJSUA2 operations
        self._cmd_queue: queue.Queue = queue.Queue()
        self._result_queues: Dict[int, queue.Queue] = {}
        self._cmd_id = 0
        self._cmd_lock = threading.Lock()
        
    def _queue_command(self, cmd: str, *args, **kwargs) -> Any:
        """Queue a command to run in the PJSUA2 thread and wait for result."""
        with self._cmd_lock:
            cmd_id = self._cmd_id
            self._cmd_id += 1
            result_queue = queue.Queue()
            self._result_queues[cmd_id] = result_queue
            
        self._cmd_queue.put((cmd_id, cmd, args, kwargs))
        
        # Wait for result (with timeout)
        try:
            result = result_queue.get(timeout=10.0)
            if isinstance(result, Exception):
                raise result
            return result
        finally:
            with self._cmd_lock:
                del self._result_queues[cmd_id]
                
    def _process_commands(self):
        """Process pending commands in the PJSUA2 thread."""
        while not self._cmd_queue.empty():
            try:
                cmd_id, cmd, args, kwargs = self._cmd_queue.get_nowait()
                result = None
                
                try:
                    if cmd == "answer":
                        call = args[0]
                        prm = pj.CallOpParam()
                        prm.statusCode = 200
                        call.answer(prm)
                        result = True
                    elif cmd == "hangup":
                        call = args[0]
                        prm = pj.CallOpParam()
                        call.hangup(prm)
                        result = True
                    elif cmd == "make_call":
                        uri = args[0]
                        result = self._do_make_call(uri)
                    elif cmd == "play_audio":
                        call = args[0]
                        wav_path = args[1]
                        result = call.play_audio_file(wav_path)
                except Exception as e:
                    result = e
                    
                # Send result back
                with self._cmd_lock:
                    if cmd_id in self._result_queues:
                        self._result_queues[cmd_id].put(result)
                        
            except queue.Empty:
                break
        
    async def start(self):
        """Start the SIP handler."""
        self.loop = asyncio.get_event_loop()
        
        if not PJSUA_AVAILABLE:
            logger.warning("PJSUA2 not available - running in mock mode")
            self._running = True
            return
        
        # Set running BEFORE starting thread (fixes race condition)
        self._running = True
            
        # Initialize PJSUA2 in a separate thread
        self._pj_thread = threading.Thread(target=self._init_pjsua, daemon=True)
        self._pj_thread.start()
        
        # Wait for initialization to complete (up to 5 seconds)
        init_timeout = 5.0
        if not self._initialized.wait(timeout=init_timeout):
            logger.warning(f"PJSUA2 initialization taking longer than {init_timeout}s")
        
        # Wait a bit more for registration
        await asyncio.sleep(2)
        
        if self._registered.is_set():
            logger.info("SIP handler started and registered")
        else:
            logger.warning("SIP handler started but registration may still be pending")
            logger.info("Check SIP_USER, SIP_PASSWORD, and SIP_DOMAIN settings")
        
    def _init_pjsua(self):
        """Initialize PJSUA2 library (runs in separate thread)."""
        try:
            # Create endpoint
            self.endpoint = pj.Endpoint()
            self.endpoint.libCreate()
            
            # Configure endpoint
            ep_cfg = pj.EpConfig()
            ep_cfg.logConfig.level = 4  # More verbose for debugging
            ep_cfg.logConfig.consoleLevel = 4
            ep_cfg.uaConfig.maxCalls = 4
            ep_cfg.uaConfig.userAgent = "SIP-AI-Assistant/1.0"
            
            # Initialize
            self.endpoint.libInit(ep_cfg)

            # --- ADD THIS BLOCK ---
            # Fix for PJMEDIA_EAUD_NODEFDEV: 
            # Force usage of the "Null Audio Device" (virtual sound card)
            # This provides the clock needed for the conference bridge without hardware.
            try:
                self.endpoint.audDevManager().setNullDev()
                logger.info("Forced Null Audio Device (Headless mode)")
            except Exception as e:
                logger.warning(f"Could not set null device: {e}")
            
            # Create transport
            transport_cfg = pj.TransportConfig()
            transport_cfg.port = self.config.sip_port
            
            transport_type = {
                "udp": pj.PJSIP_TRANSPORT_UDP,
                "tcp": pj.PJSIP_TRANSPORT_TCP,
                "tls": pj.PJSIP_TRANSPORT_TLS
            }.get(self.config.sip_transport.lower(), pj.PJSIP_TRANSPORT_UDP)
            
            tid = self.endpoint.transportCreate(transport_type, transport_cfg)
            tinfo = self.endpoint.transportGetInfo(tid)
            logger.info(f"Transport created: {tinfo.typeName} on {tinfo.localName}")
            
            # Start library
            self.endpoint.libStart()
            
            # Configure audio
            self._configure_audio()
            
            # Create account
            self._create_account()
            
            logger.info("PJSUA2 initialized successfully")
            
            # Signal that initialization is complete
            self._initialized.set()
            
            # Keep thread alive and handle events
            while self._running:
                # Process SIP events
                self.endpoint.libHandleEvents(50)  # 50ms timeout
                
                # Process command queue
                self._process_commands()
                
        except Exception as e:
            logger.error(f"Error initializing PJSUA2: {e}")
            import traceback
            traceback.print_exc()
            self._initialized.set()  # Signal even on error so we don't hang
            
    def _configure_audio(self):
        """Configure audio codecs and settings."""
        # Set codec priorities
        for i, codec in enumerate(self.config.audio_codecs):
            try:
                self.endpoint.codecSetPriority(codec, 255 - i)
            except:
                pass
                
    def _create_account(self):
        """Create and register SIP account."""
        acc_cfg = pj.AccountConfig()
        
        # Basic account info
        acc_cfg.idUri = f"sip:{self.config.sip_user}@{self.config.sip_domain}"
        
        # ALWAYS register - either to specified registrar or to the domain
        registrar = self.config.sip_registrar or self.config.sip_domain
        acc_cfg.regConfig.registrarUri = f"sip:{registrar}"
        acc_cfg.regConfig.registerOnAdd = True
        acc_cfg.regConfig.timeoutSec = 300  # Re-register every 5 minutes
        acc_cfg.regConfig.retryIntervalSec = 30
        acc_cfg.regConfig.firstRetryIntervalSec = 5

        acc_cfg.mediaConfig.transportConfig.publicAddr = "192.168.0.84"
        
        # Authentication
        if self.config.sip_password:
            cred = pj.AuthCredInfo()
            cred.scheme = "digest"
            cred.realm = "*"
            cred.username = self.config.sip_user
            cred.dataType = 0  # Plain password
            cred.data = self.config.sip_password
            acc_cfg.sipConfig.authCreds.append(cred)
        
        # NAT/Media settings for better compatibility
        acc_cfg.natConfig.iceEnabled = False
        acc_cfg.natConfig.turnEnabled = False
        acc_cfg.natConfig.sipStunUse = pj.PJSUA_STUN_USE_DISABLED
        acc_cfg.natConfig.mediaStunUse = pj.PJSUA_STUN_USE_DISABLED
        
        # Media config - disable direct media for container/NAT environments
        acc_cfg.mediaConfig.transportConfig.port = 10000
        acc_cfg.mediaConfig.transportConfig.portRange = 100
        
        # Create account
        self.account = SIPAccount(self)
        self.account.create(acc_cfg)
        
        logger.info(f"SIP account created: {acc_cfg.idUri}")
        logger.info(f"Registering with: sip:{registrar}")
        
    async def stop(self):
        """Stop the SIP handler."""
        self._running = False
        
        # Hangup all calls
        for call in list(self.active_calls.values()):
            try:
                await self.hangup_call(call.call_info)
            except:
                pass
                
        # Cleanup PJSUA2
        if self.endpoint:
            try:
                self.endpoint.libDestroy()
            except:
                pass
                
        if self._pj_thread:
            self._pj_thread.join(timeout=5)
            
        logger.info("SIP handler stopped")
        
    async def _handle_incoming_call(self, call: SIPCall):
        """Handle incoming call."""
        self.active_calls[call.call_info.call_id] = call
        await self.on_call_callback(call.call_info)
        
    def _on_call_ended(self, call: SIPCall):
        """Handle call ended."""
        if call.call_info and call.call_info.call_id in self.active_calls:
            del self.active_calls[call.call_info.call_id]
            
    async def answer_call(self, call_info: CallInfo):
        """Answer an incoming call."""
        if not PJSUA_AVAILABLE:
            logger.info(f"[MOCK] Answering call {call_info.call_id}")
            call_info.is_active = True
            return
            
        call = call_info.pj_call
        if call:
            # Queue the answer command to run in PJSUA2 thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._queue_command, "answer", call)
            logger.info(f"Answered call {call_info.call_id}")
            
    async def hangup_call(self, call_info: CallInfo):
        """Hangup a call."""
        if not PJSUA_AVAILABLE:
            logger.info(f"[MOCK] Hanging up call {call_info.call_id}")
            call_info.is_active = False
            return
            
        call = call_info.pj_call
        if call:
            # Queue the hangup command to run in PJSUA2 thread
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._queue_command, "hangup", call)
                logger.info(f"Hung up call {call_info.call_id}")
            except Exception as e:
                logger.error(f"Error hanging up call: {e}")
            
    async def make_call(self, uri: str) -> Optional[CallInfo]:
        """Make an outgoing call."""
        if not PJSUA_AVAILABLE:
            logger.info(f"[MOCK] Making call to {uri}")
            call_info = CallInfo(
                call_id=f"mock-{time.time()}",
                remote_uri=uri,
                is_active=True,
                start_time=time.time()
            )
            return call_info
            
        if not self.account:
            logger.error("No SIP account available")
            return None
            
        try:
            # Queue the make_call command to run in PJSUA2 thread
            loop = asyncio.get_event_loop()
            call_info = await loop.run_in_executor(None, self._queue_command, "make_call", uri)
            
            if call_info:
                logger.info(f"Making call to {uri}")
            return call_info
            
        except Exception as e:
            logger.error(f"Error making call: {e}")
            return None
            
    def _do_make_call(self, uri: str) -> Optional[CallInfo]:
        """Actually make a call (runs in PJSUA2 thread)."""
        try:
            call = SIPCall(self.account, pj.PJSUA_INVALID_ID, self)
            
            call_info = CallInfo(
                call_id=str(id(call)),
                remote_uri=uri,
                is_active=False,
                start_time=time.time(),
                pj_call=call
            )
            call.call_info = call_info
            
            prm = pj.CallOpParam()
            prm.opt.audioCount = 1
            prm.opt.videoCount = 0
            
            call.makeCall(uri, prm)
            
            self.active_calls[call_info.call_id] = call
            
            return call_info
            
        except Exception as e:
            logger.error(f"Error in _do_make_call: {e}")
            return None
            
    async def receive_audio(self, call_info: CallInfo, timeout: float = 0.05) -> Optional[bytes]:
        """Receive audio by reading the growing recording file directly."""
        if not PJSUA_AVAILABLE:
            await asyncio.sleep(timeout)
            return None

        # Give the recorder a tiny moment to flush data to disk
        await asyncio.sleep(timeout)

        if not call_info.record_file:
            return None

        try:
            import os
            
            # Initialize file handle if not present
            if not hasattr(call_info, '_record_fh'):
                if not os.path.exists(call_info.record_file):
                    return None
                call_info._record_fh = open(call_info.record_file, "rb")
                # Skip the 44-byte WAV header initially
                call_info._record_fh.seek(0, 2) # Seek to end
                if call_info._record_fh.tell() < 44:
                    return None
                
                # If we just opened it, ensure we are at the end (or start + 44)
                # Ideally, start reading from byte 44
                call_info._record_fh.seek(44)

            # Read new data
            new_data = call_info._record_fh.read()
            
            if len(new_data) > 0:
                return new_data
                
        except Exception as e:
            logger.debug(f"Error reading recording: {e}")
            # If file was deleted/closed, clean up handle
            if hasattr(call_info, '_record_fh') and call_info._record_fh:
                call_info._record_fh.close()
                del call_info._record_fh
                
        return None
        
    async def send_audio(self, call_info: CallInfo, audio_data: bytes):
        """Send audio to call by writing to a WAV file and playing it."""
        if not PJSUA_AVAILABLE:
            # Mock mode - just log
            duration_ms = len(audio_data) / (self.config.sample_rate * 2) * 1000
            logger.info(f"[MOCK] Sending {duration_ms:.0f}ms of audio")
            await asyncio.sleep(duration_ms / 1000)
            return
        
        # Wait for media to be ready (up to 5 seconds)
        wait_start = time.time()
        while not call_info.media_ready and time.time() - wait_start < 5.0:
            await asyncio.sleep(0.1)
            
        if not call_info.media_ready:
            logger.warning("Media not ready after 5s, cannot send audio")
            return
            
        call = call_info.pj_call
        if not call or not call.aud_med:
            logger.warning("No audio media available for playback")
            return
            
        try:
            import tempfile
            import os
            import wave
            
            # Write audio to temp WAV file
            fd, wav_path = tempfile.mkstemp(suffix='.wav', prefix='sip_out_')
            os.close(fd)
            
            # Determine sample rate (PCMU is 8kHz, we might have 16kHz)
            # PJSUA2 will resample if needed
            with wave.open(wav_path, 'wb') as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)  # 16-bit
                wav.setframerate(self.config.sample_rate)
                wav.writeframes(audio_data)
            
            logger.debug(f"Wrote {len(audio_data)} bytes to {wav_path}")
                
            # Queue playback command for PJSUA2 thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._queue_command, 
                "play_audio", 
                call, 
                wav_path
            )
            
            if result:
                logger.debug(f"Playing audio file: {wav_path}")
            else:
                logger.warning(f"Failed to start audio playback")
            
            # Wait for playback duration
            duration = len(audio_data) / (self.config.sample_rate * 2)
            await asyncio.sleep(duration + 0.5)  # Add buffer
            
            # Clean up temp file
            try:
                os.unlink(wav_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            import traceback
            traceback.print_exc()


# Alternative: Asterisk ARI Handler for more robust deployments
class AsteriskARIHandler:
    """
    Alternative SIP handler using Asterisk ARI.
    More robust for production deployments.
    
    Requires Asterisk with ARI enabled and ari-py library.
    """
    
    def __init__(self, config: Config, on_call_callback: Callable):
        self.config = config
        self.on_call_callback = on_call_callback
        self.ari_url = os.getenv("ARI_URL", "http://localhost:8088")
        self.ari_user = os.getenv("ARI_USER", "asterisk")
        self.ari_password = os.getenv("ARI_PASSWORD", "asterisk")
        self.app_name = "sip-ai-assistant"
        
    async def start(self):
        """Connect to Asterisk ARI."""
        try:
            import ari
            self.client = ari.connect(
                self.ari_url,
                self.ari_user,
                self.ari_password
            )
            self.client.on_channel_event('StasisStart', self._on_stasis_start)
            self.client.run(apps=self.app_name)
        except ImportError:
            logger.warning("ari-py not installed, ARI handler disabled")
            
    def _on_stasis_start(self, channel, event):
        """Handle new channel entering Stasis."""
        asyncio.run_coroutine_threadsafe(
            self._handle_channel(channel),
            asyncio.get_event_loop()
        )
        
    async def _handle_channel(self, channel):
        """Handle incoming channel."""
        call_info = CallInfo(
            call_id=channel.id,
            remote_uri=channel.json.get('caller', {}).get('number', 'unknown'),
            is_active=True,
            start_time=time.time()
        )
        await self.on_call_callback(call_info)


import os  # Needed for ARI handler

class PlaylistPlayer:
    """
    Manages a queue of audio files and plays them sequentially.
    Does NOT inherit from pj.AudioMediaPlayer to avoid object lifecycle issues.
    """
    def __init__(self, call_audio_media):
        self.call_med = call_audio_media
        self.queue = queue.Queue()
        self.current_player = None # The active pj.AudioMediaPlayer
        self.current_file = None
        self.is_playing = False
        self._lock = threading.Lock()

    def enqueue_file(self, file_path):
        """Add a file to the playback queue."""
        ensure_thread_registered()
        self.queue.put(file_path)
        with self._lock:
            if not self.is_playing:
                self._play_next()

    def _play_next(self):
        """Play the next file in the queue."""
        # Ensure previous player is destroyed
        if self.current_player:
            self._destroy_current_player()

        if self.queue.empty():
            self.is_playing = False
            return

        self.is_playing = True
        self.current_file = self.queue.get()
        
        try:
            # Create a FRESH player instance for this file
            if PJSUA_AVAILABLE:
                self.current_player = pj.AudioMediaPlayer()
                self.current_player.createPlayer(self.current_file, pj.PJMEDIA_FILE_NO_LOOP)
                self.current_player.startTransmit(self.call_med)
                
                # We need to know when it finishes.
                # Since we can't easily subclass the ephemeral player for callbacks in Python
                # without memory leaks, we use the file duration to schedule the next play.
                # This is actually MORE robust in Python than C++ callbacks.
                
                import wave
                import contextlib
                duration = 0.5 # default safety
                try:
                    with contextlib.closing(wave.open(self.current_file, 'r')) as f:
                        frames = f.getnframes()
                        rate = f.getframerate()
                        duration = frames / float(rate)
                except Exception as e:
                    logger.warning(f"Could not read WAV duration: {e}")

                # Schedule next track slightly before this one ends to gapless-ish
                threading.Timer(duration + 0.1, self._on_file_finished).start()
                
                logger.debug(f"Playing segment: {self.current_file} ({duration:.2f}s)")
            else:
                # Mock mode
                self._on_file_finished()

        except Exception as e:
            logger.error(f"Error playing segment {self.current_file}: {e}")
            self._destroy_current_player()
            # Try next one immediately
            self._play_next()

    def _on_file_finished(self):
        """Called when a file duration has elapsed."""
        # This runs in a timer thread, so register it!
        ensure_thread_registered()
        
        # Trigger next play
        threading.Thread(target=self._transition_next, name="PlaylistWorker").start()

    def _transition_next(self):
        """Worker to clean up and start next."""
        ensure_thread_registered()
        self._destroy_current_player()
        
        # Delete the file
        if self.current_file:
            try:
                import os
                if os.path.exists(self.current_file):
                    os.unlink(self.current_file)
            except:
                pass
            self.current_file = None

        with self._lock:
            self._play_next()

    def _destroy_current_player(self):
        """Cleanly destroy the PJSIP object."""
        if self.current_player:
            try:
                if self.call_med:
                    try:
                        self.current_player.stopTransmit(self.call_med)
                    except:
                        pass
                # There is no explicit destroy() in Python binding, 
                # just removing the reference allows GC to handle it.
            except Exception as e:
                logger.debug(f"Error destroying player: {e}")
            self.current_player = None

    def stop_all(self):
        """Stop everything."""
        ensure_thread_registered()
        with self._lock:
            # Clear queue
            while not self.queue.empty():
                try:
                    f = self.queue.get_nowait()
                    import os
                    os.unlink(f)
                except:
                    pass
            
            self.is_playing = False
            self._destroy_current_player()