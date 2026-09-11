"""
Audit and compliance logging module for Bharat AI Banking.
Maintains append-only JSONL log for RBI regulatory traceability and compliance.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared import config

logger = logging.getLogger("orchestrator.audit")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _get_audit_path() -> Path:
    """Resolve absolute path to audit_log.jsonl in the root data directory."""
    root_data = Path(__file__).resolve().parent.parent.parent / "data"
    if root_data.exists():
        data_dir = root_data
    else:
        data_dir = config.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "audit_log.jsonl"


def log_event(
    event_type: str,
    customer_id: str,
    payload: Dict[str, Any],
    decision: str,
    outcome_summary: str = "",
) -> None:
    """
    Append a structured audit event to audit_log.jsonl.
    Valid event_types: "recommend", "stress_check", "intervene", "chat", "consent_check", "consent_denied"
    Valid decisions: "allowed", "blocked", "escalated", "info"
    """
    audit_file = _get_audit_path()
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event_type": event_type,
        "customer_id": customer_id,
        "decision": decision,
        "outcome_summary": outcome_summary,
        "payload": payload,
    }
    try:
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as err:
        logger.error(f"Failed to write to audit log: {err}")


def get_recent_events(customer_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Read the most recent audit records, newest first, optionally filtered by customer_id.
    """
    audit_file = _get_audit_path()
    if not audit_file.exists():
        return []

    events: List[Dict[str, Any]] = []
    try:
        with open(audit_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if customer_id is None or record.get("customer_id") == customer_id:
                        events.append(record)
                except json.JSONDecodeError:
                    continue
    except Exception as err:
        logger.error(f"Failed to read audit log: {err}")
        return []

    # Return newest first up to limit
    events.reverse()
    return events[:limit]
