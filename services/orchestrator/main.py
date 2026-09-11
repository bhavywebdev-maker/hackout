"""
FastAPI Orchestrator Service for Bharat AI Banking.
Acts as the central gateway enforcing DPDP consent, managing stress-gated product delivery,
and auditing all operations for RBI compliance.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
import httpx
from pydantic import BaseModel, Field
import uvicorn

from shared import config
from services.orchestrator.audit import _get_audit_path, get_recent_events, log_event
from services.orchestrator.consent import (
    CONSENT_DATA,
    check_consent,
    get_all_consents,
    load_consents,
    record_consent_event,
)
from services.orchestrator.pipeline import (
    call_chatbot_chat,
    call_early_warning_check,
    call_early_warning_intervene,
    call_recommender,
    get_service_urls,
)

logger = logging.getLogger("orchestrator.main")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Bharat AI Banking - Orchestrator Service",
    description="Unified API gateway coordinating DPDP consent, life-stage recommender, chatbot, and early warning",
    version="1.0.0",
)


class ActionRequest(BaseModel):
    event_type: str = Field(default="salary_credit")
    language: str = Field(default="en")
    payload: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChatForwardRequest(BaseModel):
    session_id: str = Field(default="orch_session")
    customer_id: Optional[str] = None
    message: str = Field(default="")
    language: str = Field(default="auto")


class ConsentRevokeRequest(BaseModel):
    customer_id: str
    purpose: str


@app.on_event("startup")
def on_startup() -> None:
    """Initialize consent cache and audit log."""
    load_consents()
    # Ensure audit log file exists
    audit_file = _get_audit_path()
    if not audit_file.exists():
        audit_file.parent.mkdir(parents=True, exist_ok=True)
        audit_file.touch()
    logger.info(f"Orchestrator Service initialized with DPDP consent and audit foundation at {audit_file}")


@app.get("/health")
async def health() -> Dict[str, Any]:
    """Health check endpoint probing downstream services in parallel."""
    urls = get_service_urls()

    async def _ping(url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{url}/health")
                return "up" if resp.status_code == 200 else "down"
        except Exception:
            return "down"

    rec_status, chat_status, ew_status = await asyncio.gather(
        _ping(urls["recommender"]),
        _ping(urls["chatbot"]),
        _ping(urls["early_warning"]),
    )

    return {
        "status": "ok",
        "service": "orchestrator",
        "services": {
            "recommender": rec_status,
            "chatbot": chat_status,
            "early_warning": ew_status,
        },
    }


@app.post("/customer/{customer_id}/action")
async def customer_action(customer_id: str, req: ActionRequest) -> Dict[str, Any]:
    """Execute customer event workflow with consent enforcement and stress gating."""
    # 1. Enforce recommendation consent
    rec_consent = check_consent(customer_id, "recommendation")
    record_consent_event(customer_id, "recommendation", "check", rec_consent["allowed"])

    if not rec_consent["allowed"]:
        log_event(
            event_type="consent_denied",
            customer_id=customer_id,
            payload=req.payload or {},
            decision="blocked",
            outcome_summary=f"Recommendation blocked: {rec_consent['reason']}",
        )
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "consent_denied", "reason": rec_consent["reason"]},
        )

    # 2. Check stress detection consent
    stress_consent = check_consent(customer_id, "stress_detection")
    record_consent_event(customer_id, "stress_detection", "check", stress_consent["allowed"])

    if stress_consent["allowed"]:
        stress_check = await call_early_warning_check(customer_id)
        stress_level = stress_check.get("stress_level", "none")
    else:
        stress_level = "none"
        stress_check = {"stress_level": "none", "risk_score": 0, "note": "consent_denied"}
        log_event(
            event_type="stress_check",
            customer_id=customer_id,
            payload=req.payload or {},
            decision="info",
            outcome_summary="Stress check skipped: consent denied",
        )

    # 3. Apply Stress Gate
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if stress_level == "high":
        intervene_res = await call_early_warning_intervene(customer_id, req.language)
        rec_res = await call_recommender(customer_id, "high", req.language)

        log_event(
            event_type="intervene",
            customer_id=customer_id,
            payload=req.payload or {},
            decision="escalated",
            outcome_summary="High stress detected; intervention triggered and safe products recommended",
        )

        return {
            "customer_id": customer_id,
            "event_type": req.event_type,
            "action_taken": "intervention",
            "life_stage": rec_res.get("life_stage", "young_saver"),
            "life_stage_confidence": rec_res.get("life_stage_confidence", 0.85),
            "stress_check": {
                "stress_level": stress_check.get("stress_level", "high"),
                "risk_score": stress_check.get("risk_score", 75),
                "top_reasons": stress_check.get("top_reasons", []),
            },
            "intervention": {
                "message": intervene_res.get("intervention_message", ""),
                "suggested_actions": intervene_res.get("suggested_actions", []),
            },
            "recommendations": rec_res.get("recommendations", []),
            "consent_status": {
                "recommendation": rec_consent["reason"],
                "stress_detection": stress_consent["reason"],
            },
            "generated_at": now_iso,
        }
    else:
        rec_res = await call_recommender(customer_id, stress_level, req.language)

        log_event(
            event_type="recommend",
            customer_id=customer_id,
            payload=req.payload or {},
            decision="allowed",
            outcome_summary=f"Recommended products with stress_level={stress_level}",
        )

        return {
            "customer_id": customer_id,
            "event_type": req.event_type,
            "action_taken": "recommendation",
            "life_stage": rec_res.get("life_stage", "young_saver"),
            "life_stage_confidence": rec_res.get("life_stage_confidence", 0.85),
            "stress_check": {
                "stress_level": stress_check.get("stress_level", "none"),
                "risk_score": stress_check.get("risk_score", 0),
                "top_reasons": stress_check.get("top_reasons", []),
            },
            "recommendations": rec_res.get("recommendations", []),
            "consent_status": {
                "recommendation": rec_consent["reason"],
                "stress_detection": stress_consent["reason"],
            },
            "generated_at": now_iso,
        }


@app.post("/chat")
async def chat_proxy(req: ChatForwardRequest) -> Dict[str, Any]:
    """Forward chat queries after verifying chatbot consent if customer_id is provided."""
    if req.customer_id:
        chat_consent = check_consent(req.customer_id, "chatbot")
        record_consent_event(req.customer_id, "chatbot", "check", chat_consent["allowed"])
        if not chat_consent["allowed"]:
            log_event(
                event_type="consent_denied",
                customer_id=req.customer_id,
                payload={"session_id": req.session_id},
                decision="blocked",
                outcome_summary=f"Chatbot access blocked: {chat_consent['reason']}",
            )
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"error": "consent_denied", "reason": chat_consent["reason"]},
            )

    res = await call_chatbot_chat(
        session_id=req.session_id,
        message=req.message,
        language=req.language,
        customer_id=req.customer_id,
    )

    log_event(
        event_type="chat",
        customer_id=req.customer_id or "anonymous",
        payload={"session_id": req.session_id},
        decision="allowed",
        outcome_summary="Chat message processed via orchestrator",
    )

    return res


@app.get("/customer/{customer_id}/consents")
def customer_consents(customer_id: str) -> Dict[str, Any]:
    """Retrieve full customer consent records."""
    return get_all_consents(customer_id)



@app.post("/consent/revoke")
def revoke_consent(req: ConsentRevokeRequest) -> Dict[str, Any]:
    """Revoke customer consent for a specific processing purpose."""
    if not CONSENT_DATA:
        load_consents()

    record = CONSENT_DATA.get(req.customer_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")

    item = next((c for c in record.get("consents", []) if c.get("purpose") == req.purpose), None)
    if item:
        item["given"] = False

    record_consent_event(req.customer_id, req.purpose, "revoke", False)
    log_event(
        event_type="consent_revoke",
        customer_id=req.customer_id,
        payload={"purpose": req.purpose},
        decision="info",
        outcome_summary=f"Consent revoked for {req.purpose}",
    )

    return {
        "status": "revoked",
        "customer_id": req.customer_id,
        "purpose": req.purpose,
    }


class ConsentGrantRequest(BaseModel):
    customer_id: str
    purpose: str


@app.post("/consent/grant")
def grant_consent(req: ConsentGrantRequest) -> Dict[str, Any]:
    """Grant or re-grant customer consent for a specific processing purpose."""
    if not CONSENT_DATA:
        load_consents()

    record = CONSENT_DATA.get(req.customer_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Customer not found")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    expiry = "2027-12-31T23:59:59Z"
    item = next((c for c in record.get("consents", []) if c.get("purpose") == req.purpose), None)
    if item:
        item["given"] = True
        item["timestamp"] = now
        item["expires_at"] = expiry
    else:
        record.setdefault("consents", []).append({
            "purpose": req.purpose,
            "given": True,
            "timestamp": now,
            "expires_at": expiry,
        })

    record_consent_event(req.customer_id, req.purpose, "grant", True)
    log_event(
        event_type="consent_grant",
        customer_id=req.customer_id,
        payload={"purpose": req.purpose},
        decision="info",
        outcome_summary=f"Consent granted for {req.purpose}",
    )

    return {
        "status": "granted",
        "customer_id": req.customer_id,
        "purpose": req.purpose,
        "expires_at": expiry,
    }


@app.get("/audit/recent")
def recent_audit(
    customer_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50),
) -> List[Dict[str, Any]]:
    """Retrieve recent regulatory audit events."""
    return get_recent_events(customer_id=customer_id, limit=limit)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
