"""
Text-to-Speech (TTS) synthesis module for Bharat Chatbot.
Generates speech using gTTS across 10 Indian vernacular languages.
"""

import io
import logging
import tempfile
from typing import Optional

from gtts import gTTS

logger = logging.getLogger("chatbot.tts")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

SUPPORTED_TTS_LANGS = {
    "hi": "hi", "en": "en", "ta": "ta", "te": "te",
    "bn": "bn", "mr": "mr", "gu": "gu", "kn": "kn",
    "ml": "ml", "pa": "pa",
}


def synthesize_speech(text: str, language: str = "hi") -> bytes:
    """
    Synthesize audio speech for a given text message using gTTS.
    Returns raw MP3 bytes or empty bytes on error.
    """
    if not text or not text.strip():
        return b""

    # Strip mock prefix if present for clean voice playback
    clean_text = text.replace("[MOCK]", "").strip()
    if not clean_text:
        clean_text = text

    lang_code = SUPPORTED_TTS_LANGS.get(language.lower(), "hi")
    try:
        tts = gTTS(text=clean_text, lang=lang_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as err:
        logger.error(f"TTS synthesis failed for lang='{language}': {err}")
        return b""


def save_audio_to_temp(mp3_bytes: bytes) -> str:
    """
    Persist MP3 bytes to a temporary audio file and return the path.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(mp3_bytes)
        return f.name
