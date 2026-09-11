"""
FastAPI Vernacular Chatbot Service for Bharat AI Banking.
Provides multilingual multi-turn chat, speech-to-text, text-to-speech,
grounded FAQ answering, identity verification, and conversational loan form filling.
"""

import logging
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

from shared.constants import SUPPORTED_LANGUAGES
from services.chatbot.audio_handler import detect_language_from_text, handle_audio_upload
from services.chatbot.form_fill import (
    LOAN_FORM_SCHEMA,
    extract_form_entities,
    generate_loan_summary,
    get_missing_fields,
    get_next_question,
)
from services.chatbot.gemini_chat import (
    SESSIONS,
    get_reply,
    load_customer_profiles,
    load_faq,
    reset_session,
    verify_identity,
)
from services.chatbot.tts import save_audio_to_temp, synthesize_speech

logger = logging.getLogger("chatbot.main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Bharat AI Banking - Vernacular Chatbot Service",
    description="Conversational multilingual banking assistant supporting 10 Indian languages and voice",
    version="1.0.0",
)

# In-memory map for generated audio files: {filename: absolute_path}
AUDIO_STORAGE: Dict[str, str] = {}


class VerifyRequest(BaseModel):
    session_id: str
    customer_id: str
    dob: str
    mother_name: str


class FormStartRequest(BaseModel):
    session_id: str
    language: Optional[str] = "en"


class FormSummaryRequest(BaseModel):
    session_id: str
    language: Optional[str] = "en"


class SessionResetRequest(BaseModel):
    session_id: str


@app.on_event("startup")
def on_startup() -> None:
    """Initialize FAQ knowledge base and customer profiles."""
    load_faq()
    load_customer_profiles()
    logger.info("Chatbot Service initialized with FAQ and Customer Profiles")


@app.get("/health")
def health() -> Dict[str, Any]:
    """Health check endpoint exposing supported Indian languages."""
    return {
        "status": "ok",
        "service": "chatbot",
        "languages": list(SUPPORTED_LANGUAGES.keys()),
    }


@app.post("/chat")
async def chat_endpoint(request: Request) -> Dict[str, Any]:
    """
    Multilingual multi-turn chat endpoint accepting JSON or multipart form data.
    """
    content_type = request.headers.get("content-type", "")
    session_id = "default"
    user_message = ""
    language = "auto"
    customer_id = None
    transcript = None

    if "application/json" in content_type:
        body = await request.json()
        session_id = body.get("session_id", "default")
        user_message = body.get("message", "")
        language = body.get("language", "auto")
        customer_id = body.get("customer_id")
    elif "multipart/form-data" in content_type:
        form = await request.form()
        session_id = str(form.get("session_id", "default"))
        user_message = str(form.get("message", ""))
        language = str(form.get("language", "auto"))
        customer_id = form.get("customer_id")
        audio_file = form.get("audio")
        if audio_file and hasattr(audio_file, "read"):
            audio_bytes = await audio_file.read()
            if audio_bytes:
                audio_res = handle_audio_upload(audio_bytes, language_hint=language if language != "auto" else "hi")
                transcript = audio_res.get("transcript", "")
                user_message = transcript
    else:
        try:
            body = await request.json()
            session_id = body.get("session_id", "default")
            user_message = body.get("message", "")
            language = body.get("language", "auto")
            customer_id = body.get("customer_id")
        except Exception:
            pass

    # Ensure session exists
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "messages": [],
            "language": language,
            "form_data": {},
            "form_active": False,
            "verified": False,
            "customer_id": customer_id,
        }

    session = SESSIONS[session_id]
    if customer_id and not session.get("customer_id"):
        session["customer_id"] = customer_id

    # Auto detect language if requested
    if language == "auto":
        language = detect_language_from_text(user_message)

    # Fetch chat response
    chat_res = get_reply(
        session_id=session_id,
        user_message=user_message,
        language=language,
        customer_id=session.get("customer_id"),
    )
    reply_text = chat_res["reply"]
    detected_lang = chat_res["language"]

    # Conversational loan form processing
    form_data = session.get("form_data", {})
    missing_fields: List[str] = []
    next_question: Optional[str] = None

    if session.get("form_active") or session.get("verified"):
        form_data = extract_form_entities(user_message, form_data)
        session["form_data"] = form_data
        missing_fields = get_missing_fields(form_data)
        if missing_fields:
            next_question = get_next_question(form_data, language=detected_lang)

    return {
        "session_id": session_id,
        "reply": reply_text,
        "language": detected_lang,
        "transcript": transcript,
        "form_data": form_data,
        "missing_fields": missing_fields,
        "next_question": next_question,
        "verified": session.get("verified", False),
    }


@app.post("/chat/voice")
async def chat_voice_endpoint(
    audio: UploadFile = File(...),
    session_id: str = Form("voice_session"),
    language_hint: str = Form("hi"),
) -> Dict[str, Any]:
    """Voice input and voice response conversational endpoint."""
    audio_bytes = await audio.read()
    audio_res = handle_audio_upload(audio_bytes, language_hint=language_hint)
    transcript = audio_res.get("transcript", "")

    # Execute chat pipeline
    chat_res = get_reply(
        session_id=session_id,
        user_message=transcript,
        language=language_hint,
    )
    reply_text = chat_res["reply"]
    detected_lang = chat_res["language"]

    # Synthesize audio speech
    mp3_bytes = synthesize_speech(reply_text, language=detected_lang)
    audio_reply_url = None
    if mp3_bytes:
        file_path = save_audio_to_temp(mp3_bytes)
        filename = Path(file_path).name
        AUDIO_STORAGE[filename] = file_path
        audio_reply_url = f"/audio/{filename}"

    session = SESSIONS.get(session_id, {})
    form_data = session.get("form_data", {})
    missing_fields: List[str] = []
    next_question: Optional[str] = None

    if session.get("form_active") or session.get("verified"):
        form_data = extract_form_entities(transcript, form_data)
        session["form_data"] = form_data
        missing_fields = get_missing_fields(form_data)
        if missing_fields:
            next_question = get_next_question(form_data, language=detected_lang)

    return {
        "session_id": session_id,
        "reply": reply_text,
        "language": detected_lang,
        "transcript": transcript,
        "audio_reply_url": audio_reply_url,
        "form_data": form_data,
        "missing_fields": missing_fields,
        "next_question": next_question,
        "verified": session.get("verified", False),
    }


@app.get("/audio/{filename}")
def get_audio_file(filename: str) -> FileResponse:
    """Serve synthesized MP3 audio file."""
    clean_name = Path(filename).name
    file_path = AUDIO_STORAGE.get(clean_name)
    if not file_path or not Path(file_path).exists():
        temp_candidate = Path(tempfile.gettempdir()) / clean_name
        if temp_candidate.exists():
            file_path = str(temp_candidate)
        else:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found")

    return FileResponse(file_path, media_type="audio/mpeg")


@app.post("/verify")
def verify_customer(req: VerifyRequest) -> Dict[str, Any]:
    """Verify customer identity before sharing sensitive account details."""
    success = verify_identity(req.session_id, req.dob, req.mother_name, req.customer_id)
    return {
        "verified": success,
        "message": "Identity verified successfully." if success else "Identity verification failed.",
    }


@app.post("/form/start")
def start_form(req: FormStartRequest) -> Dict[str, Any]:
    """Initialize a conversational loan application session."""
    if req.session_id not in SESSIONS:
        SESSIONS[req.session_id] = {
            "messages": [],
            "language": req.language or "en",
            "form_data": {},
            "form_active": True,
            "verified": False,
            "customer_id": None,
        }

    session = SESSIONS[req.session_id]
    session["form_data"] = dict(LOAN_FORM_SCHEMA)
    session["form_active"] = True

    missing = get_missing_fields(session["form_data"])
    next_q = get_next_question(session["form_data"], language=req.language or "en")

    return {
        "session_id": req.session_id,
        "form_data": session["form_data"],
        "missing_fields": missing,
        "next_question": next_q,
    }


@app.post("/form/summary")
def form_summary(req: FormSummaryRequest) -> Dict[str, Any]:
    """Retrieve pre-filled loan application summary and eligibility calculation."""
    session = SESSIONS.get(req.session_id)
    if not session or not session.get("form_data"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active loan form found for this session",
        )

    return generate_loan_summary(session["form_data"], language=req.language or "en")


@app.post("/session/reset")
def reset_session_endpoint(req: SessionResetRequest) -> Dict[str, str]:
    """Clear conversational session from memory."""
    reset_session(req.session_id)
    return {"status": "reset"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)
