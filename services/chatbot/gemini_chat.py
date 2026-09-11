"""
Gemini Chat orchestration module for Bharat Chatbot.
Handles conversational state, FAQ grounding, identity verification, and multi-turn chat.
"""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional

from shared import config, gemini_client
from shared.constants import SUPPORTED_LANGUAGES
from services.chatbot.audio_handler import detect_language_from_text

logger = logging.getLogger("chatbot.chat")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# In-memory session store
SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_MAX_TURNS = 20
FAQ_CONTENT: str = ""
FAQ_PAIRS: List[Dict[str, str]] = []
CUSTOMER_PROFILES_SET = set()


def load_faq() -> str:
    """Read banking_faq.md at startup and parse Q&A pairs for prompt grounding."""
    global FAQ_CONTENT, FAQ_PAIRS
    if FAQ_CONTENT:
        return FAQ_CONTENT

    faq_path = Path(__file__).parent / "knowledge_base" / "banking_faq.md"
    if not faq_path.exists():
        logger.warning(f"FAQ file not found at {faq_path}")
        FAQ_CONTENT = "# FAQ Content Placeholder"
        return FAQ_CONTENT

    with open(faq_path, encoding="utf-8") as f:
        FAQ_CONTENT = f.read()

    # Parse into structured pairs
    raw_sections = FAQ_CONTENT.split("## Q:")
    FAQ_PAIRS = []
    for sec in raw_sections[1:]:
        parts = sec.split("**A:**", 1)
        if len(parts) == 2:
            q_text = parts[0].strip()
            a_text = parts[1].split("## Q:")[0].strip()
            FAQ_PAIRS.append({"q": q_text, "a": a_text})

    logger.info(f"Loaded FAQ knowledge base ({len(FAQ_CONTENT)} bytes, {len(FAQ_PAIRS)} pairs)")
    return FAQ_CONTENT


def load_customer_profiles() -> None:
    """Load customer profile IDs for identity verification."""
    global CUSTOMER_PROFILES_SET
    data_dir = config.DATA_DIR
    if not data_dir.exists():
        fallback_path = Path(__file__).resolve().parent.parent.parent / "data"
        if fallback_path.exists():
            data_dir = fallback_path

    profiles_path = data_dir / "customer_profiles.json"
    if profiles_path.exists():
        try:
            with open(profiles_path, encoding="utf-8") as f:
                profiles = json.load(f)
                CUSTOMER_PROFILES_SET = {p["customer_id"] for p in profiles}
        except Exception as err:
            logger.error(f"Failed to load customer profiles: {err}")


def find_relevant_faq_context(user_query: str, top_k: int = 3) -> str:
    """Find top matching FAQ entries based on keyword overlap."""
    if not FAQ_PAIRS:
        load_faq()

    query_words = set(re.findall(r"\w+", user_query.lower()))
    scored_pairs = []

    for item in FAQ_PAIRS:
        item_words = set(re.findall(r"\w+", (item["q"] + " " + item["a"]).lower()))
        score = len(query_words.intersection(item_words))
        scored_pairs.append((score, item))

    scored_pairs.sort(key=lambda x: x[0], reverse=True)
    selected = [p[1] for p in scored_pairs[:top_k]]
    if not selected:
        selected = FAQ_PAIRS[:top_k]

    context_lines = ["Relevant Banking FAQ context:"]
    for i, pair in enumerate(selected, 1):
        context_lines.append(f"Q{i}: {pair['q']}\nA{i}: {pair['a']}")
    return "\n".join(context_lines)


def build_system_prompt(language: str, customer_context: Optional[Dict[str, Any]] = None) -> str:
    """Compose structured system prompt for Bharat Sahayak."""
    lang_name = SUPPORTED_LANGUAGES.get(language, "English")
    cust_str = json.dumps(customer_context) if customer_context else "anonymous"

    return (
        f"You are 'Bharat Sahayak', a friendly, patient banking assistant for Indian bank customers.\n"
        f"Always reply in {lang_name}. Support code-mixed language like Hinglish if the user writes that way.\n"
        f"Use simple, short sentences. Avoid banking jargon. If the user is confused, offer to explain step-by-step.\n"
        f"Never share specific account balances, loan amounts, or OTPs unless the user is verified.\n"
        f"When you don't know something, suggest visiting the branch or calling the helpline 1800-XXX-XXXX.\n"
        f"Keep replies under 80 words unless the user asks for detail.\n"
        f"Customer context: {cust_str}"
    )


def get_reply(
    session_id: str,
    user_message: str,
    language: str = "auto",
    customer_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Process user message and return assistant reply within conversation session."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "messages": [],
            "language": "hi",
            "form_data": {},
            "form_active": False,
            "verified": False,
            "customer_id": customer_id,
        }

    session = SESSIONS[session_id]
    if customer_id and not session.get("customer_id"):
        session["customer_id"] = customer_id

    # Detect language if auto
    if language == "auto":
        language = detect_language_from_text(user_message)
    session["language"] = language

    # Append user turn
    session["messages"].append({"role": "user", "content": user_message})

    # Rolling window trim
    if len(session["messages"]) > SESSION_MAX_TURNS * 2:
        session["messages"] = session["messages"][-SESSION_MAX_TURNS * 2:]

    # Build prompt with FAQ grounding
    faq_context = find_relevant_faq_context(user_message)
    system_prompt = build_system_prompt(language)
    full_system_prompt = f"{system_prompt}\n\n{faq_context}"

    reply = gemini_client.chat_with_history(session["messages"], full_system_prompt)

    # Mock response localization check
    if reply.startswith("[MOCK]"):
        if language == "hi":
            reply = "[MOCK] नमस्ते! मैं आपकी बैंकिंग सहायता के लिए यहाँ हूँ। आप लोन, खाता, या अन्य सेवाओं के बारे में पूछ सकते हैं।"
        elif language == "en":
            reply = "[MOCK] Hello! I am here to help with your banking needs. How can I assist you today?"
        else:
            lang_name = SUPPORTED_LANGUAGES.get(language, "your preferred language")
            reply = f"[MOCK] Hello! I am here to help you in {lang_name}."

    # Append assistant turn
    session["messages"].append({"role": "assistant", "content": reply})

    return {
        "reply": reply,
        "language": language,
        "session_id": session_id,
    }


def verify_identity(session_id: str, dob: str, mother_name: str, customer_id: str) -> bool:
    """Mock identity verification against customer dataset."""
    if not CUSTOMER_PROFILES_SET:
        load_customer_profiles()

    if not dob or not mother_name or not customer_id:
        return False

    is_valid_customer = customer_id in CUSTOMER_PROFILES_SET
    if is_valid_customer and session_id in SESSIONS:
        SESSIONS[session_id]["verified"] = True
        SESSIONS[session_id]["customer_id"] = customer_id
        return True

    return False


def reset_session(session_id: str) -> None:
    """Clear session from memory."""
    SESSIONS.pop(session_id, None)
