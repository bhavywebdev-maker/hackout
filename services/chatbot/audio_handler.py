"""
Audio processing and vernacular language detection module for Bharat Chatbot.
Handles audio transcription requests and script-based language identification.
"""

import logging
import re
from typing import Any, Dict

from shared import gemini_client

logger = logging.getLogger("chatbot.audio")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Unicode regex patterns for Indian scripts
SCRIPT_PATTERNS = [
    ("hi", re.compile(r"[\u0900-\u097F]")),  # Devanagari (Hindi/Marathi)
    ("ta", re.compile(r"[\u0B80-\u0BFF]")),  # Tamil
    ("te", re.compile(r"[\u0C00-\u0C7F]")),  # Telugu
    ("bn", re.compile(r"[\u0980-\u09FF]")),  # Bengali
    ("gu", re.compile(r"[\u0A80-\u0AFF]")),  # Gujarati
    ("kn", re.compile(r"[\u0C80-\u0CFF]")),  # Kannada
    ("ml", re.compile(r"[\u0D00-\u0D7F]")),  # Malayalam
    ("pa", re.compile(r"[\u0A00-\u0A7F]")),  # Gurmukhi (Punjabi)
]

# Common romanized Hindi / Hinglish tokens
HINGLISH_KEYWORDS = {
    "mujhe", "chahiye", "kya", "hai", "hain", "mera", "meri", "mere", "aap", "kaise",
    "karna", "karein", "batao", "namaste", "paisa", "paise", "khata", "byaj", "kitna",
    "dena", "lekin", "nahi", "nahin", "shukriya", "dhanyawad"
}


def handle_audio_upload(audio_bytes: bytes, language_hint: str = "hi") -> Dict[str, Any]:
    """
    Transcribe audio bytes using the Gemini client with language hints.
    Returns transcript and confidence, or an error payload.
    """
    try:
        transcript = gemini_client.transcribe_audio(audio_bytes, language_hint=language_hint)
        return {
            "transcript": transcript,
            "language_hint": language_hint,
            "confidence": 0.95 if transcript else 0.0,
        }
    except Exception as err:
        logger.error(f"Audio transcription failed: {err}")
        return {
            "transcript": "",
            "error": str(err),
            "language_hint": language_hint,
            "confidence": 0.0,
        }


def detect_language_from_text(text: str) -> str:
    """
    Detect language using Unicode ranges for Indic scripts, Hinglish keywords,
    or default to English.
    """
    if not text or not text.strip():
        return "en"

    # Check Unicode script ranges first
    for lang_code, pattern in SCRIPT_PATTERNS:
        if pattern.search(text):
            return lang_code

    # Check for Romanized Hindi / Hinglish keywords
    tokens = re.findall(r"\b\w+\b", text.lower())
    for token in tokens:
        if token in HINGLISH_KEYWORDS:
            return "hi"

    return "en"
