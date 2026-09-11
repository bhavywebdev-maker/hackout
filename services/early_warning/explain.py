"""
Explainability and empathetic intervention messaging module for Early Warning Service.
Provides SHAP feature attribution and multilingual intervention scripts.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from shared import config, gemini_client

logger = logging.getLogger("early_warning.explain")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def get_top_reasons(
    baseline: Dict[str, Any],
    classifier_model: Any,
    feature_names: List[str],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Extract top risk-contributing features using SHAP TreeExplainer on the classifier.
    Falls back to highest absolute z-scores / deviations if SHAP is unavailable.
    """
    z = baseline.get("z_scores", {})
    raw = baseline.get("raw_metrics", {})

    # Try SHAP
    if classifier_model is not None:
        try:
            import shap

            vector = [
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
            X = np.array([vector])
            explainer = shap.TreeExplainer(classifier_model)
            shap_values = explainer.shap_values(X)

            # For multiclass, take the 'stress' class if present, else mean
            classes = list(getattr(classifier_model, "classes_", []))
            target_class = "stress" if "stress" in classes else (classes[0] if classes else None)
            if isinstance(shap_values, list):
                class_idx = classes.index(target_class) if target_class in classes else 0
                sample_shap = shap_values[class_idx][0]
            elif len(shap_values.shape) == 3:
                class_idx = classes.index(target_class) if target_class in classes else 0
                sample_shap = shap_values[0, :, class_idx]
            else:
                sample_shap = shap_values[0]

            results = []
            for i, name in enumerate(feature_names):
                val = float(sample_shap[i])
                direction = "increases_risk" if val >= 0 else "decreases_risk"
                results.append({
                    "feature": name,
                    "shap_value": round(val, 4),
                    "direction": direction,
                })

            results.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
            return results[:top_k]
        except Exception as err:
            logger.debug(f"SHAP explanation fallback triggered: {err}")

    # Fallback to feature deviations
    reasons = []
    if raw.get("missed_emi_count", 0) > 0:
        reasons.append({
            "feature": "missed_emi_count",
            "shap_value": 0.45,
            "direction": "increases_risk",
        })
    if raw.get("salary_drop_pct", 0.0) > 0.15:
        reasons.append({
            "feature": "salary_drop_pct",
            "shap_value": 0.35,
            "direction": "increases_risk",
        })

    for feat, z_val in z.items():
        if abs(z_val) > 1.0:
            reasons.append({
                "feature": feat,
                "shap_value": round(float(z_val) * 0.1, 4),
                "direction": "increases_risk" if (z_val > 1.0 or (feat == "discretionary_spend" and z_val < -1.0)) else "decreases_risk",
            })

    reasons.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return reasons[:top_k] if reasons else [
        {"feature": "txn_amount", "shap_value": 0.05, "direction": "increases_risk"}
    ]


def humanize_reasons(
    top_reasons: List[Dict[str, Any]],
    raw_metrics: Dict[str, Any],
    language: str = "en",
) -> List[str]:
    """Map technical feature identifiers to clear, empathetic natural language statements."""
    messages = []
    is_hindi = (language.lower() == "hi")
    missed_count = raw_metrics.get("missed_emi_count", 1)

    for item in top_reasons:
        feat = item["feature"]
        if feat == "missed_emi_count":
            if is_hindi:
                messages.append(f"आपने हाल ही में {missed_count} EMI भुगतान चूक दिए")
            else:
                messages.append(f"You missed {missed_count} EMI payment(s) recently")
        elif feat == "discretionary_spend":
            if is_hindi:
                messages.append("आपका विवेकाधीन खर्च तेजी से गिरा है")
            else:
                messages.append("Your discretionary spending dropped sharply")
        elif feat == "salary_drop_pct":
            if is_hindi:
                messages.append("आपकी हाल की सैलरी आमतौर पर कम थी")
            else:
                messages.append("Your recent salary credit was lower than usual")
        elif feat == "night_txn_ratio":
            if is_hindi:
                messages.append("असामान्य देर रात लेनदेन पाए गए")
            else:
                messages.append("Unusual late-night transactions detected")
        elif feat == "new_beneficiary_count":
            if is_hindi:
                messages.append("कई नए भुगतान प्राप्तकर्ता पाए गए")
            else:
                messages.append("Multiple new payment recipients detected")
        else:
            clean_feat = feat.replace("_", " ").title()
            if is_hindi:
                messages.append(f"{clean_feat} में असामान्य गतिविधि")
            else:
                messages.append(f"{clean_feat} showed unusual activity")

    return messages


def generate_intervention_message(
    customer_id: str,
    stress_level: str,
    reasons: List[str],
    language: str = "en",
) -> str:
    """Generate empathetic, supportive intervention copy for customers facing financial strain."""
    if stress_level == "none":
        return "No intervention needed — all clear."

    try:
        return gemini_client.generate_intervention(reasons, language)
    except Exception:
        pass

    is_hindi = (language.lower() == "hi")

    if stress_level == "high":
        if is_hindi:
            return (
                "[MOCK] हमने आपके हाल के लेन-देन में कुछ बदलाव देखे हैं। हम मदद के लिए यहाँ हैं — "
                "क्या आप 3 महीने की ईएमआई मोहलत लेना चाहेंगे या हमारी वित्तीय वेलनेस टीम से बात करना चाहेंगे?"
            )
        return (
            "We noticed some changes in your recent transactions. We're here to help — "
            "would you like to explore a 3-month EMI moratorium or speak with our financial wellness team?"
        )
    elif stress_level == "medium":
        if is_hindi:
            return (
                "[MOCK] हमने आपके हाल के लेन-देन में कुछ बदलाव देखे हैं। "
                "क्या आप किसी रिलेशनशिप मैनेजर से बात करना चाहेंगे?"
            )
        return (
            "We noticed some changes in your recent transactions. "
            "Would you like to speak with a relationship manager?"
        )
    else:
        if is_hindi:
            return "[MOCK] सब कुछ स्थिर लग रहा है — आपके वित्त को स्वस्थ रखने के लिए यहाँ कुछ सुझाव दिए गए हैं।"
        return "Everything looks steady — here are some tips to keep your finances healthy."
