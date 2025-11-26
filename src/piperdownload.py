
from pathlib import Path
import re
import shutil
from urllib.request import urlopen
import logging

logger = logging.getLogger(__name__)

def _needs_download(path: Path) -> bool:
    """Return True if file needs to be downloaded."""
    if not path.exists():
        return True

    if path.stat().st_size == 0:
        # Empty
        return True

    return False

def download_voice(
    voice: str, download_dir: Path, force_redownload: bool = False
) -> None:
    """Download a voice model and config file to a directory."""
    voice = voice.strip()
    VOICE_PATTERN = re.compile(
        r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$"
    )
    voice_match = VOICE_PATTERN.match(voice)
    if not voice_match:
        raise ValueError(
            f"Voice '{voice}' did not match pattern: <language>-<name>-<quality> like 'en_US-lessac-medium'",
        )
    URL_FORMAT = "https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang_family}/{lang_code}/{voice_name}/{voice_quality}/{lang_code}-{voice_name}-{voice_quality}{extension}?download=true"

    lang_family = voice_match.group("lang_family")
    lang_code = lang_family + "_" + voice_match.group("lang_region")
    voice_name = voice_match.group("voice_name")
    voice_quality = voice_match.group("voice_quality")

    voice_code = f"{lang_code}-{voice_name}-{voice_quality}"
    format_args = {
        "lang_family": lang_family,
        "lang_code": lang_code,
        "voice_name": voice_name,
        "voice_quality": voice_quality,
    }

    model_path = download_dir / f"{voice_code}.onnx"
    if force_redownload or _needs_download(model_path):
        model_url = URL_FORMAT.format(extension=".onnx", **format_args)
        logger.debug("Downloading model from '%s' to '%s'", model_url, model_path)
        with urlopen(model_url) as response:
            with open(model_path, "wb") as model_file:
                shutil.copyfileobj(response, model_file)

        logger.debug("Downloaded: '%s'", model_path)

    config_path = download_dir / f"{voice_code}.onnx.json"
    if force_redownload or _needs_download(config_path):
        config_url = URL_FORMAT.format(extension=".onnx.json", **format_args)
        logger.debug("Downloading config from '%s' to '%s'", config_url, config_path)
        with urlopen(config_url) as response:
            with open(config_path, "wb") as config_file:
                shutil.copyfileobj(response, config_file)

        logger.debug("Downloaded: '%s'", config_path)

    logger.info("Downloaded: %s", voice)


