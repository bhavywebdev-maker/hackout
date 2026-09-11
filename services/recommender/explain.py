"""
Explainability module for Bharat AI Banking Recommender Service.
Computes SHAP feature contributions and generates natural language explanations.
"""

import logging
from typing import Any, Dict, List

from shared import config, gemini_client
from services.recommender.model import PRODUCT_NAMES

logger = logging.getLogger("recommender.explain")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def compute_shap(model: Any, X_row: Dict[str, Any], feature_names: List[str]) -> Dict[str, float]:
    """
    Compute SHAP attribution values using TreeExplainer.
    Returns a dict mapping feature names to their importance values.
    Returns empty dict on any failure.
    """
    if model is None:
        return {}

    try:
        import numpy as np
        import shap

        row_vals = np.array([[float(X_row.get(col, 0.0)) for col in feature_names]])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(row_vals)

        # Handle multiclass shap values list or 3D array
        if isinstance(shap_values, list):
            # Take the mean absolute contribution across classes for this row
            mean_shap = np.mean([np.abs(cls_shap[0]) for cls_shap in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            mean_shap = np.mean(np.abs(shap_values[0]), axis=1)
        else:
            mean_shap = np.abs(shap_values[0])

        shap_dict = {
            feature_names[i]: round(float(mean_shap[i]), 4)
            for i in range(len(feature_names))
        }
        # Sort descending by importance
        sorted_shap = dict(sorted(shap_dict.items(), key=lambda item: item[1], reverse=True))
        return sorted_shap
    except Exception as e:
        logger.debug(f"SHAP computation skipped or failed ({e}). Returning empty dict.")
        return {}


def explain(
    product_id: str,
    features: Dict[str, Any],
    shap_values: Dict[str, float],
    customer_context: Dict[str, Any],
    language: str = "en",
) -> str:
    """
    Generate an intuitive, human-friendly explanation of why a product was recommended.
    Delegates to Gemini if available, or falls back to template using top-2 SHAP features.
    """
    stress_level = (customer_context or {}).get("stress_level", "none")
    # Empathetic explanation for ALL products if financial stress is detected
    if stress_level == "high" or product_id == "counseling":
        if config.is_gemini_available():
            try:
                return gemini_client.explain_recommendation(
                    shap_values=shap_values,
                    customer_context=customer_context,
                    language=language,
                )
            except Exception as e:
                logger.warning(f"Gemini explanation error ({e}); using template fallback.")
        if language == "hi":
            return "[MOCK] हमने आपके हाल के लेन-देन में वित्तीय तनाव के संकेत देखे हैं। यह विकल्प आपको स्थिरता की ओर लौटने में मदद कर सकता है।"
        return "[MOCK] We noticed signs of financial strain in your recent transactions. This option may help you regain stability."

    if config.is_gemini_available():
        try:
            return gemini_client.explain_recommendation(
                shap_values=shap_values,
                customer_context=customer_context,
                language=language
            )
        except Exception as e:
            logger.warning(f"Gemini explanation error ({e}); using template fallback.")

    # Template fallback using top-2 features
    product_name = PRODUCT_NAMES.get(product_id, product_id.replace("_", " ").title())

    if shap_values:
        top_keys = list(shap_values.keys())[:2]
        f1 = top_keys[0].replace("_", " ")
        f2 = top_keys[1].replace("_", " ") if len(top_keys) > 1 else "spending patterns"
    else:
        # Fallback to key features from features dict
        f1 = "savings rate"
        f2 = "income stability"

    if language == "hi":
        return f"[MOCK] हमने आपकी {f1} और {f2} में अनुकूल रुझान देखा है, इसलिए आपके लिए {product_name} सबसे उपयुक्त है।"

    return (
        f"Based on your profile, your {f1} and {f2} indicate "
        f"that {product_name} is an ideal fit for your current financial goals."
    )
