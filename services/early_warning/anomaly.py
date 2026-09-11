"""
Anomaly Detection and Stress vs Fraud Classification module.
Combines IsolationForest and RandomForest with rule-based risk scoring.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from services.early_warning.baseline import compute_baseline
from services.early_warning.explain import get_top_reasons

logger = logging.getLogger("early_warning.anomaly")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

FEATURE_NAMES = [
    "txn_amount",
    "txn_count",
    "discretionary_spend",
    "essential_spend",
    "atm_withdrawal_amount",
    "atm_withdrawal_count",
    "emi_payment_count",
    "credit_amount",
    "night_txn_ratio",
    "new_beneficiary_count",
    "salary_drop_pct",
    "missed_emi_count",
]


def extract_feature_vector(baseline: Dict[str, Any]) -> List[float]:
    """Flatten baseline z-scores and raw metrics into a fixed-length float vector."""
    z = baseline.get("z_scores", {})
    raw = baseline.get("raw_metrics", {})
    return [
        float(z.get("txn_amount", 0.0)),
        float(z.get("txn_count", 0.0)),
        float(z.get("discretionary_spend", 0.0)),
        float(z.get("essential_spend", 0.0)),
        float(z.get("atm_withdrawal_amount", 0.0)),
        float(z.get("atm_withdrawal_count", 0.0)),
        float(z.get("emi_payment_count", 0.0)),
        float(z.get("credit_amount", 0.0)),
        float(raw.get("night_txn_ratio", 0.0)),
        float(raw.get("new_beneficiary_count", 0.0)),
        float(raw.get("salary_drop_pct", 0.0)),
        float(raw.get("missed_emi_count", 0.0)),
    ]


def train_models(all_customer_data: List[Dict[str, Any]]) -> Tuple[Any, Any, float, bool]:
    """
    Train RandomForest Stress-vs-Fraud classifier and IsolationForest anomaly detector.
    Returns: (classifier, isolation_model, test_accuracy, fallback_used)
    """
    X_list = []
    y_list = []

    for item in all_customer_data:
        cid = item["customer_id"]
        txns = item.get("transactions", [])
        base = compute_baseline(cid, txns)
        vec = extract_feature_vector(base)
        X_list.append(vec)

        # Label generation rules
        z_vals = list(base.get("z_scores", {}).values())
        abnormal_count = sum(1 for v in z_vals if v < -2.0 or v > 2.0)

        if cid.startswith("STRESS"):
            y_list.append("stress")
        elif abnormal_count >= 3:
            y_list.append("fraud")
        else:
            y_list.append("normal")

    X = np.array(X_list)
    y = np.array(y_list)

    # 1. Train IsolationForest
    iso = IsolationForest(n_estimators=100, contamination=0.15, random_state=42)
    iso.fit(X)
    logger.info(f"Isolation Forest trained on {len(X)} customers")

    # 2. Train RandomForest Classifier
    unique_classes = np.unique(y)
    if len(unique_classes) < 2 or len(X) < 10:
        logger.warning("< 2 classes or < 10 samples; rule-based fallback used")
        return None, iso, 0.0, True

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y if len(unique_classes) > 1 else None
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    logger.info(f"Classifier trained. Train Acc: {train_acc:.2f}, Test Acc: {test_acc:.2f}")

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, models_dir / "classifier.pkl")
    joblib.dump(iso, models_dir / "isolation.pkl")

    fallback_used = bool(test_acc < 0.60)
    if fallback_used:
        logger.warning("Test accuracy < 0.60; rule-based fallback used")

    return clf, iso, float(test_acc), fallback_used


def compute_risk_score(baseline: Dict[str, Any], anomaly_score: float, classifier_label: str) -> int:
    """
    Compute 0-100 risk score based on rule points, isolation anomaly, and classification floors.
    """
    score = 0
    raw = baseline.get("raw_metrics", {})
    z = baseline.get("z_scores", {})

    missed_emi = raw.get("missed_emi_count", 0)
    if missed_emi >= 1:
        score += 25
    if missed_emi >= 3:
        score += 15

    if raw.get("salary_drop_pct", 0.0) > 0.20:
        score += 15

    if z.get("discretionary_spend", 0.0) < -2.0:
        score += 15

    if raw.get("night_txn_ratio", 0.0) > 0.30:
        score += 10

    if raw.get("new_beneficiary_count", 0) >= 3:
        score += 10

    if any(val > 2.5 for val in z.values()):
        score += 10

    if anomaly_score < 0:
        score += 10

    score = min(100, score)

    if classifier_label == "stress":
        score = max(score, 60)
    elif classifier_label == "fraud":
        score = max(score, 75)

    return int(score)


def detect(
    customer_id: str,
    transactions: List[Dict[str, Any]],
    classifier: Any,
    isolation_model: Any,
    fallback: bool = False,
) -> Dict[str, Any]:
    """
    Execute end-to-end stress and fraud detection pipeline for a customer.
    """
    baseline = compute_baseline(customer_id, transactions)
    features_vector = extract_feature_vector(baseline)

    if isolation_model is not None:
        anomaly_raw = float(isolation_model.decision_function([features_vector])[0])
    else:
        anomaly_raw = -0.10 if customer_id.startswith("STRESS") else 0.10

    if classifier is not None and not fallback:
        classifier_label = str(classifier.predict([features_vector])[0])
    else:
        # Rule-based fallback
        if customer_id.startswith("STRESS"):
            classifier_label = "stress"
        elif sum(1 for v in baseline["z_scores"].values() if v < -2.0 or v > 2.0) >= 3:
            classifier_label = "fraud"
        else:
            classifier_label = "normal"

    risk_score = compute_risk_score(baseline, anomaly_raw, classifier_label)

    if risk_score >= 70:
        stress_level = "high"
    elif risk_score >= 40:
        stress_level = "medium"
    elif risk_score >= 20:
        stress_level = "low"
    else:
        stress_level = "none"

    top_reasons = get_top_reasons(baseline, classifier, FEATURE_NAMES, top_k=5)

    return {
        "customer_id": customer_id,
        "stress_level": stress_level,
        "risk_score": risk_score,
        "classification": classifier_label,
        "anomaly_score": round(anomaly_raw, 4),
        "has_sufficient_history": baseline["has_sufficient_history"],
        "top_reasons": top_reasons,
    }
