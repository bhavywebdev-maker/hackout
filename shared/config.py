"""Central config loader for all services."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Standard library fallback to load .env when python-dotenv is not installed
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_raw_mock = os.getenv("USE_MOCK_LLM", "auto").lower()
USE_MOCK_LLM = (_raw_mock == "auto" and not GEMINI_API_KEY) or _raw_mock == "true"


def is_gemini_available() -> bool:
    return bool(GEMINI_API_KEY) and not USE_MOCK_LLM
