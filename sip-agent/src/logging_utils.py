"""
Logging Utilities
=================
Shared logging functions and utilities used across the application.
"""

import logging
from typing import Any, Dict, Optional


def log_event(
    log: logging.Logger,
    level: int,
    msg: str,
    event: Optional[str] = None,
    **data: Any
) -> None:
    """
    Helper to log structured events.
    
    Args:
        log: Logger instance
        level: Log level (e.g., logging.INFO)
        msg: Log message
        event: Optional event type for structured logging
        **data: Additional event data
    """
    extra: Dict[str, Any] = {}
    if event:
        extra['event_type'] = event
    if data:
        extra['event_data'] = data
    log.log(level, msg, extra=extra)


def format_duration(seconds: int) -> str:
    """
    Format duration in seconds for natural speech.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string
    """
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining = seconds % 60
        if remaining == 0:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        return f"{minutes} minute{'s' if minutes != 1 else ''} and {remaining} second{'s' if remaining != 1 else ''}"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        parts = [f"{hours} hour{'s' if hours != 1 else ''}"]
        if minutes:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        return " and ".join(parts)


# Common constants
WAV_HEADER_SIZE = 44  # Standard PCM WAV header size
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SAMPLE_WIDTH = 2  # 16-bit audio

# Timing constants (in seconds)
HANGUP_DELAY_SECONDS = 3.0
AUDIO_POLL_INTERVAL_SECONDS = 0.05
CALL_CHECK_INTERVAL_SECONDS = 0.5
RECONNECT_BASE_DELAY_SECONDS = 1.0
RECONNECT_MAX_DELAY_SECONDS = 30.0
