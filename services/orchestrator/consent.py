"""
Consent Management module for Bharat AI Banking Orchestrator.
Enforces Digital Personal Data Protection (DPDP) consent requirements before any processing.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared import config

logger = logging.getLogger("orchestrator.consent")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

CONSENT_DATA: Dict[str, Dict[str, Any]] = {}
VALID_PURPOSES = {"recommendation", "stress_detection", "chatbot", "marketing"}


def _get_data_dir() -> Path:
    """Resolve data directory location, preferring repository root data directory."""
    root_data = Path(__file__).resolve().parent.parent.parent / "data"
    if root_data.exists():
        return root_data
    return config.DATA_DIR


def load_consents() -> None:
    """Load consent records into memory cache."""
    global CONSENT_DATA
    data_dir = _get_data_dir()
    consent_file = data_dir / "consent_records.json"

    if not consent_file.exists():
        logger.warning(f"Consent file not found at {consent_file}")
        CONSENT_DATA.clear()
        return

    try:
        with open(consent_file, encoding="utf-8") as f:
            records = json.load(f)
            new_data = {r["customer_id"]: r for r in records}
        CONSENT_DATA.clear()
        CONSENT_DATA.update(new_data)
        logger.info(f"Loaded consent records for {len(CONSENT_DATA)} customers")
    except Exception as err:
        logger.error(f"Failed to load consent records: {err}")
        CONSENT_DATA.clear()


def check_consent(customer_id: str, purpose: str) -> Dict[str, Any]:
    """
    Check DPDP consent validity for a given customer and processing purpose.
    Returns: {"allowed": bool, "reason": str}
    """
    if not CONSENT_DATA:
        load_consents()

    if purpose not in VALID_PURPOSES:
        return {"allowed": False, "reason": f"invalid_purpose: {purpose}"}

    record = CONSENT_DATA.get(customer_id)
    if not record:
        return {"allowed": False, "reason": "no_consent_record"}

    # Find the specific consent purpose item
    consent_item = next((c for c in record.get("consents", []) if c.get("purpose") == purpose), None)
    if not consent_item:
        return {"allowed": False, "reason": "no_consent_record"}

    if not consent_item.get("given", False):
        return {"allowed": False, "reason": "consent_denied"}

    # Expiry verification against simulated data epoch (2025-06-30 by default)
    expires_at = consent_item.get("expires_at")
    if expires_at:
        ref_str = os.getenv("CURRENT_SYSTEM_DATE", "2025-06-30T00:00:00Z")
        if expires_at < ref_str:
            return {"allowed": False, "reason": "consent_expired"}

    return {"allowed": True, "reason": "consent_valid"}


def record_consent_event(customer_id: str, purpose: str, action: str, granted: bool) -> None:
    """Append a consent event directly to audit_log.jsonl."""
    data_dir = _get_data_dir()
    audit_file = data_dir / "audit_log.jsonl"
    event = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event_type": "consent_check" if action == "check" else f"consent_{action}",
        "customer_id": customer_id,
        "decision": "allowed" if granted else "blocked",
        "outcome_summary": f"Consent {action} for {purpose}: granted={granted}",
        "payload": {
            "purpose": purpose,
            "action": action,
            "granted": granted,
        },
    }
    try:
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as err:
        logger.error(f"Failed to record consent event: {err}")


def get_all_consents(customer_id: str) -> Dict[str, Any]:
    """Retrieve full consent portfolio for a customer for transparency."""
    if not CONSENT_DATA:
        load_consents()

    record = CONSENT_DATA.get(customer_id)
    if not record:
        return {
            "customer_id": customer_id,
            "consents": [],
            "status": "not_found",
        }
    return record
