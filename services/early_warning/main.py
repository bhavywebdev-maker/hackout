"""
FastAPI Early Warning Service for Bharat AI Banking.
Detects early indicators of financial distress and spending anomalies,
and generates supportive, non-punitive intervention pathways.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import uvicorn

from shared import config
from services.early_warning.baseline import compute_baseline
from services.early_warning.anomaly import (
    FEATURE_NAMES,
    compute_risk_score,
    detect,
    train_models,
)
from services.early_warning.explain import (
    generate_intervention_message,
    get_top_reasons,
    humanize_reasons,
)

logger = logging.getLogger("early_warning.service")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(
    title="Bharat AI Banking - Early Warning Service",
    description="Proactive financial distress detection and intervention platform",
    version="1.0.0",
)

# Global in-memory cache
CUSTOMER_DATA: Dict[str, Dict[str, Any]] = {}
CLASSIFIER_MODEL: Any = None
ISOLATION_MODEL: Any = None
FALLBACK_USED: bool = False
TEST_ACCURACY: float = 0.0


class CheckRequest(BaseModel):
    recent_days: Optional[int] = Field(default=30, description="Recent analysis window in days")


class InterveneRequest(BaseModel):
    language: str = Field(default="en", description="Preferred intervention language")


def _load_and_train() -> None:
    """Load transaction records and train anomaly and classification models."""
    global CUSTOMER_DATA, CLASSIFIER_MODEL, ISOLATION_MODEL, FALLBACK_USED, TEST_ACCURACY

    data_dir = config.DATA_DIR
    if not data_dir.exists():
        fallback_path = Path(__file__).resolve().parent.parent.parent / "data"
        if fallback_path.exists():
            data_dir = fallback_path

    profiles_path = data_dir / "customer_profiles.json"
    txns_path = data_dir / "synthetic_transactions.json"

    if not profiles_path.exists() or not txns_path.exists():
        logger.warning(f"Data files missing in {data_dir}. Early Warning service standby.")
        return

    with open(profiles_path, encoding="utf-8") as f:
        profiles = json.load(f)
    with open(txns_path, encoding="utf-8") as f:
        txns = json.load(f)

    # Combine into CUSTOMER_DATA cache
    p_map = {p["customer_id"]: p for p in profiles}
    CUSTOMER_DATA = {}
    for t in txns:
        cid = t["customer_id"]
        CUSTOMER_DATA[cid] = {
            "customer_id": cid,
            "profile": p_map.get(cid, {}),
            "transactions": t.get("transactions", []),
        }

    # Train RandomForest + IsolationForest
    all_items = list(CUSTOMER_DATA.values())
    CLASSIFIER_MODEL, ISOLATION_MODEL, TEST_ACCURACY, FALLBACK_USED = train_models(all_items)
    logger.info(
        f"Early Warning Service initialized with {len(CUSTOMER_DATA)} customers. "
        f"Test Acc: {TEST_ACCURACY:.2f}, Fallback: {FALLBACK_USED}"
    )


@app.on_event("startup")
def on_startup() -> None:
    _load_and_train()


@app.get("/health")
def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "early_warning"}


@app.get("/customer/{customer_id}/baseline")
def get_customer_baseline(customer_id: str) -> Dict[str, Any]:
    """Return raw baseline statistics and z-scores for debugging."""
    if customer_id not in CUSTOMER_DATA:
        _load_and_train()
    if customer_id not in CUSTOMER_DATA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_id} not found",
        )

    txns = CUSTOMER_DATA[customer_id]["transactions"]
    return compute_baseline(customer_id, txns)


@app.post("/check/{customer_id}")
def check_financial_stress(customer_id: str, request: Optional[CheckRequest] = None) -> Dict[str, Any]:
    """
    Check for early financial stress and anomalous behavior in customer transactions.
    """
    if customer_id not in CUSTOMER_DATA:
        _load_and_train()
    if customer_id not in CUSTOMER_DATA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_id} not found",
        )

    txns = CUSTOMER_DATA[customer_id]["transactions"]
    det = detect(
        customer_id=customer_id,
        transactions=txns,
        classifier=CLASSIFIER_MODEL,
        isolation_model=ISOLATION_MODEL,
        fallback=FALLBACK_USED,
    )

    baseline = compute_baseline(customer_id, txns)
    top_reasons = det["top_reasons"]
    humanized = humanize_reasons(top_reasons, baseline.get("raw_metrics", {}), language="en")

    return {
        "customer_id": customer_id,
        "stress_level": det["stress_level"],
        "risk_score": det["risk_score"],
        "classification": det["classification"],
        "anomaly_score": det["anomaly_score"],
        "has_sufficient_history": det["has_sufficient_history"],
        "top_reasons": top_reasons,
        "humanized_reasons": humanized,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


@app.post("/intervene/{customer_id}")
def generate_intervention(customer_id: str, request: InterveneRequest) -> Dict[str, Any]:
    """
    Generate empathetic, non-punitive intervention pathways and counseling offers.
    """
    if customer_id not in CUSTOMER_DATA:
        _load_and_train()
    if customer_id not in CUSTOMER_DATA:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Customer {customer_id} not found",
        )

    txns = CUSTOMER_DATA[customer_id]["transactions"]
    det = detect(
        customer_id=customer_id,
        transactions=txns,
        classifier=CLASSIFIER_MODEL,
        isolation_model=ISOLATION_MODEL,
        fallback=FALLBACK_USED,
    )
    stress_level = det["stress_level"]
    lang = request.language.lower()

    if stress_level == "none":
        return {
            "customer_id": customer_id,
            "stress_level": "none",
            "language": lang,
            "intervention_message": "No intervention needed — all clear.",
            "suggested_actions": [],
            "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    baseline = compute_baseline(customer_id, txns)
    raw_metrics = baseline.get("raw_metrics", {})
    reasons_text = humanize_reasons(det["top_reasons"], raw_metrics, language=lang)

    msg = generate_intervention_message(
        customer_id=customer_id,
        stress_level=stress_level,
        reasons=reasons_text,
        language=lang,
    )

    suggested_actions = [
        "3-month EMI moratorium",
        "Speak with financial counselor",
        "Restructure loan tenure",
    ]

    return {
        "customer_id": customer_id,
        "stress_level": stress_level,
        "language": lang,
        "intervention_message": msg,
        "suggested_actions": suggested_actions,
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
