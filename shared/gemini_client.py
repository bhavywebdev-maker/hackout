"""
Gemini integration client for Bharat AI Banking.
Provides centralized LLM calls with retry logic, latency logging,
and deterministic mock fallbacks.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional
import warnings

warnings.filterwarnings("ignore")

from shared import config

logger = logging.getLogger("gemini_client")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(getattr(logging, config.LOG_LEVEL.upper(), logging.INFO))

# Graceful optional import of google.generativeai
try:
    import google.generativeai as genai
    _GENAI_INSTALLED = True
except ImportError:
    genai = None
    _GENAI_INSTALLED = False

_INITIALIZED = False


def _init_gemini() -> bool:
    """Initialize Gemini SDK if credentials and package are available."""
    global _INITIALIZED
    if _INITIALIZED:
        return True
    if _GENAI_INSTALLED and config.is_gemini_available():
        try:
            genai.configure(api_key=config.GEMINI_API_KEY)
            _INITIALIZED = True
            return True
        except Exception as e:
            logger.warning(f"Failed to configure Google Generative AI: {e}")
            return False
    return False


def _call_with_retry(fn, func_name: str, language: str, **kwargs) -> Any:
    """Execute LLM call with 3 retries and exponential backoff (1s, 2s, 4s)."""
    delays = [1.0, 2.0, 4.0]
    model_name = config.LLM_MODEL
    start_time = time.time()

    for attempt, delay in enumerate(delays):
        try:
            result = fn(**kwargs)
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"{func_name} lang={language} latency={latency_ms}ms "
                f"model={model_name} mock=False"
            )
            return result
        except Exception as err:
            logger.warning(
                f"{func_name} attempt {attempt + 1} failed: {err}. Retrying in {delay}s..."
            )
            time.sleep(delay)

    # Final attempt failed; log and raise to trigger mock fallback
    latency_ms = int((time.time() - start_time) * 1000)
    logger.error(
        f"{func_name} all retries exhausted after {latency_ms}ms. Triggering mock fallback."
    )
    raise RuntimeError(f"All retries failed for {func_name}")


def chat(prompt: str, system_prompt: Optional[str] = None, language: str = "en") -> str:
    """Single-turn chat completion with mock fallback."""
    start_time = time.time()
    if _init_gemini():
        try:
            def _execute():
                model = genai.GenerativeModel(
                    model_name=config.LLM_MODEL,
                    system_instruction=system_prompt if system_prompt else None
                )
                response = model.generate_content(prompt)
                return response.text
            return _call_with_retry(_execute, "chat", language)
        except Exception:
            pass

    # Mock Fallback
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"chat lang={language} latency={latency_ms}ms model={config.LLM_MODEL} mock=True"
    )
    if language == "hi":
        return "[MOCK] नमस्ते! मैं आपकी बैंकिंग सहायता के लिए यहाँ हूँ।"
    return "[MOCK] Hello! I am here to help with your banking needs."


def chat_with_history(messages: List[Dict[str, str]], system_prompt: Optional[str] = None) -> str:
    """Multi-turn chat completion with message history and mock fallback."""
    start_time = time.time()
    lang = "en"
    if _init_gemini():
        try:
            def _execute():
                model = genai.GenerativeModel(
                    model_name=config.LLM_MODEL,
                    system_instruction=system_prompt if system_prompt else None
                )
                chat_session = model.start_chat(history=[])
                last_msg = messages[-1]["content"] if messages else ""
                response = chat_session.send_message(last_msg)
                return response.text
            return _call_with_retry(_execute, "chat_with_history", lang)
        except Exception:
            pass

    # Mock Fallback
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"chat_with_history lang={lang} latency={latency_ms}ms model={config.LLM_MODEL} mock=True"
    )
    return "[MOCK] Hello! I am here to help with your banking needs."


def transcribe_audio(audio_bytes: bytes, language_hint: str = "hi") -> str:
    """Transcribe audio bytes using Gemini multimodal audio or fallback."""
    start_time = time.time()
    if _init_gemini() and audio_bytes:
        try:
            def _execute():
                model = genai.GenerativeModel(model_name=config.LLM_MODEL)
                prompt = (
                    f"Transcribe this audio clip accurately. Language hint: {language_hint}. "
                    "Return only the transcription text."
                )
                response = model.generate_content([
                    prompt,
                    {"mime_type": "audio/mp3", "data": audio_bytes}
                ])
                return response.text.strip()
            return _call_with_retry(_execute, "transcribe_audio", language_hint)
        except Exception:
            pass

    # Mock Fallback
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"transcribe_audio lang={language_hint} latency={latency_ms}ms "
        f"model={config.LLM_MODEL} mock=True"
    )
    if language_hint == "hi":
        return "[MOCK] मुझे लोन के बारे में जानकारी चाहिए।"
    return "[MOCK] I need information regarding loans."


def extract_entities(text: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Extract structured entities from text conforming to a given schema dict."""
    start_time = time.time()
    lang = "en"
    if _init_gemini():
        try:
            def _execute():
                prompt = (
                    f"Extract entities from this text:\n'{text}'\n"
                    f"Return a strict JSON object matching this schema keys: {list(schema.keys())}."
                )
                model = genai.GenerativeModel(
                    model_name=config.LLM_MODEL,
                    generation_config={"response_mime_type": "application/json"}
                )
                response = model.generate_content(prompt)
                return json.loads(response.text)
            return _call_with_retry(_execute, "extract_entities", lang)
        except Exception:
            pass

    # Mock Fallback: dict with every schema key present, values None
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"extract_entities lang={lang} latency={latency_ms}ms model={config.LLM_MODEL} mock=True"
    )
    return {k: None for k in schema.keys()}


def generate_intervention(stress_reasons: List[str], language: str = "en") -> str:
    """Generate a supportive, non-predatory intervention message for financial stress."""
    start_time = time.time()
    if _init_gemini():
        try:
            def _execute():
                prompt = (
                    f"A customer is showing financial stress due to: {', '.join(stress_reasons)}. "
                    f"Draft a warm, empathetic banking intervention message in {language} offering support, "
                    "EMI rescheduling, or financial counseling without threatening legal action or penalties."
                )
                model = genai.GenerativeModel(model_name=config.LLM_MODEL)
                response = model.generate_content(prompt)
                return response.text.strip()
            return _call_with_retry(_execute, "generate_intervention", language)
        except Exception:
            pass

    # Mock Fallback
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"generate_intervention lang={language} latency={latency_ms}ms "
        f"model={config.LLM_MODEL} mock=True"
    )
    if language == "hi":
        return "[MOCK] हमने देखा कि आपकी EMI समय पर नहीं जमा हुई। हम 3 महीने की मोहलत दे सकते हैं। क्या आप बात करना चाहेंगे?"
    return "[MOCK] We noticed your EMI payment was delayed. We can offer a 3-month moratorium or restructuring. Would you like to connect?"


def explain_recommendation(shap_values: Dict[str, float], customer_context: Dict[str, Any], language: str = "en") -> str:
    """Generate a personalized natural language explanation for a product recommendation."""
    stress_level = (customer_context or {}).get("stress_level", "none")
    if stress_level == "high":
        if language == "hi":
            return "[MOCK] हमने आपके हाल के लेन-देन में वित्तीय तनाव के संकेत देखे हैं। यह विकल्प आपको स्थिरता की ओर लौटने में मदद कर सकता है।"
        return "[MOCK] We noticed signs of financial strain in your recent transactions. This option may help you regain stability."

    start_time = time.time()
    if _init_gemini():
        try:
            def _execute():
                prompt = (
                    f"Explain why this product is recommended to the customer based on: "
                    f"Context: {customer_context}, SHAP factors: {shap_values}. "
                    f"Language: {language}. Write a friendly, transparent 2-sentence explanation."
                )
                model = genai.GenerativeModel(model_name=config.LLM_MODEL)
                response = model.generate_content(prompt)
                return response.text.strip()
            return _call_with_retry(_execute, "explain_recommendation", language)
        except Exception:
            pass

    # Mock Fallback
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"explain_recommendation lang={language} latency={latency_ms}ms "
        f"model={config.LLM_MODEL} mock=True"
    )
    if language == "hi":
        return "[MOCK] हमने आपकी सैलरी में वृद्धि देखी है और आपके पास कोई निवेश नहीं है, इसलिए हम म्यूचुअल फंड सुझाते हैं।"
    return "[MOCK] We noticed an increase in your salary and you have no active investments, so we recommend a Mutual Fund SIP."


def answer_faq(question: str, faq_context: str, language: str = "en") -> str:
    """Answer a banking question using verified FAQ context."""
    start_time = time.time()
    if _init_gemini():
        try:
            def _execute():
                prompt = (
                    f"Answer this banking question based ONLY on the provided FAQ context:\n"
                    f"FAQ Context:\n{faq_context}\n\nQuestion: {question}\nLanguage: {language}"
                )
                model = genai.GenerativeModel(model_name=config.LLM_MODEL)
                response = model.generate_content(prompt)
                return response.text.strip()
            return _call_with_retry(_execute, "answer_faq", language)
        except Exception:
            pass

    # Mock Fallback
    latency_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"answer_faq lang={language} latency={latency_ms}ms model={config.LLM_MODEL} mock=True"
    )
    if language == "hi":
        return "[MOCK] कृपया अपनी बैंक शाखा से संपर्क करें।"
    return "[MOCK] Please contact your nearest bank branch."
