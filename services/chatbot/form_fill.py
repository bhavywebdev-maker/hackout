"""
Loan application conversational form auto-filling and EMI calculation module.
Extracts structured loan parameters from multi-turn dialogues.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from shared import gemini_client

logger = logging.getLogger("chatbot.form_fill")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

LOAN_FORM_SCHEMA: Dict[str, Any] = {
    "full_name": None,
    "age": None,
    "monthly_income": None,
    "loan_amount": None,
    "loan_purpose": None,        # "home" | "auto" | "personal" | "education"
    "employment_type": None,     # "salaried" | "self_employed" | "gig_worker" | "farmer"
    "city": None,
    "existing_emi": None,
    "preferred_language": None,
}

REQUIRED_FIELDS = [
    "full_name",
    "age",
    "monthly_income",
    "loan_amount",
    "loan_purpose",
    "employment_type",
]

QUESTION_TEMPLATES = {
    "full_name": {
        "en": "What is your full name?",
        "hi": "आपका पूरा नाम क्या है?",
    },
    "age": {
        "en": "How old are you?",
        "hi": "आपकी उम्र कितनी है?",
    },
    "monthly_income": {
        "en": "What is your monthly income?",
        "hi": "आपकी मासिक आय कितनी है?",
    },
    "loan_amount": {
        "en": "How much loan do you need?",
        "hi": "आपको कितने रुपये का लोन चाहिए?",
    },
    "loan_purpose": {
        "en": "What is the loan for — home, auto, personal, or education?",
        "hi": "लोन किसलिए — घर, गाड़ी, व्यक्तिगत, या शिक्षा?",
    },
    "employment_type": {
        "en": "Are you salaried, self-employed, gig worker, or farmer?",
        "hi": "आप सैलरीड, स्व-रोज़गार, गिग वर्कर, या किसान हैं?",
    },
    "city": {
        "en": "Which city do you live in?",
        "hi": "आप किस शहर में रहते हैं?",
    },
    "existing_emi": {
        "en": "How much EMI do you pay monthly?",
        "hi": "आप मासिक कितनी EMI देते हैं?",
    },
    "preferred_language": {
        "en": "Which language do you prefer?",
        "hi": "आप कौन सी भाषा पसंद करते हैं?",
    },
}


def _rule_based_entity_extraction(text: str) -> Dict[str, Any]:
    """Fallback rule-based extraction for common loan applicant inputs."""
    extracted: Dict[str, Any] = {}

    # Full name
    name_match = re.search(
        r"(?:my name is|i am|name is|मेरा नाम|नाम है|naam hai)\s+([A-Za-z\s]+?)(?:[,\.]|$|\band\b)",
        text,
        re.IGNORECASE,
    )
    if name_match:
        cand = name_match.group(1).strip()
        if len(cand.split()) >= 1 and not any(w in cand.lower() for w in ["looking", "applying", "salaried"]):
            extracted["full_name"] = cand
    elif re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$", text.strip()):
        extracted["full_name"] = text.strip()

    # Age
    age_match = re.search(r"(?:age is|age|i am|उम्र|साल)\s*(\d{2})\b|\b(\d{2})\s*(?:years old|साल|yrs)", text, re.IGNORECASE)
    if age_match:
        val = age_match.group(1) or age_match.group(2)
        if 18 <= int(val) <= 100:
            extracted["age"] = int(val)

    # Monthly income
    inc_match = re.search(r"(?:income|salary|earn|कमाई|वेतन)\s*(?:is|of|₹|rs\.?|inr)?\s*([\d,]+)", text, re.IGNORECASE)
    if inc_match:
        clean_num = inc_match.group(1).replace(",", "")
        extracted["monthly_income"] = float(clean_num)

    # Loan amount
    loan_match = re.search(r"(?:need|want|loan amount|loan of|लोन)\s*(?:of|is|₹|rs\.?|inr)?\s*([\d,]+)", text, re.IGNORECASE)
    if loan_match:
        clean_num = loan_match.group(1).replace(",", "")
        extracted["loan_amount"] = float(clean_num)

    # Loan purpose
    lower = text.lower()
    if any(w in lower for w in ["home", "house", "घर", "makan"]):
        extracted["loan_purpose"] = "home"
    elif any(w in lower for w in ["car", "auto", "bike", "vehicle", "गाड़ी"]):
        extracted["loan_purpose"] = "auto"
    elif any(w in lower for w in ["study", "education", "college", "शिक्षा", "पढ़ाई"]):
        extracted["loan_purpose"] = "education"
    elif any(w in lower for w in ["personal", "व्यक्तिगत", "emergency"]):
        extracted["loan_purpose"] = "personal"

    # Employment type
    if any(w in lower for w in ["salaried", "job", "नौकरी", "वेतन"]):
        extracted["employment_type"] = "salaried"
    elif any(w in lower for w in ["self-employed", "self employed", "business", "दुकान", "व्यापार"]):
        extracted["employment_type"] = "self_employed"
    elif any(w in lower for w in ["gig", "freelance", "driver", "delivery", "गिग"]):
        extracted["employment_type"] = "gig_worker"
    elif any(w in lower for w in ["farmer", "agriculture", "किसान", "खेती"]):
        extracted["employment_type"] = "farmer"

    # City
    city_match = re.search(r"\b(?:in|city|lives in|रहता हूँ)\s+([A-Z][a-z]+)", text)
    if city_match:
        extracted["city"] = city_match.group(1)

    return extracted


def extract_form_entities(user_message: str, existing_form: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract loan application entities from message and merge into existing form.
    Utilizes Gemini entity extraction with deterministic regex fallback.
    """
    updated_form = dict(existing_form) if existing_form else dict(LOAN_FORM_SCHEMA)

    # 1. Try Gemini entity extraction
    try:
        gemini_res = gemini_client.extract_entities(user_message, LOAN_FORM_SCHEMA)
        if isinstance(gemini_res, dict):
            for k, v in gemini_res.items():
                if v is not None and k in updated_form and updated_form[k] is None:
                    updated_form[k] = v
    except Exception as err:
        logger.debug(f"Gemini entity extraction skipped: {err}")

    # 2. Rule-based entity extraction fallback
    rule_res = _rule_based_entity_extraction(user_message)
    for k, v in rule_res.items():
        if v is not None and k in updated_form and updated_form[k] is None:
            updated_form[k] = v

    return updated_form


def get_missing_fields(form: Dict[str, Any]) -> List[str]:
    """Return ordered list of form fields that have not yet been collected."""
    order = list(LOAN_FORM_SCHEMA.keys())
    return [k for k in order if form.get(k) is None]


def get_next_question(form: Dict[str, Any], language: str = "en") -> Optional[str]:
    """Retrieve localized question prompt for the next pending form parameter."""
    missing = get_missing_fields(form)
    if not missing:
        return None

    next_field = missing[0]
    lang_key = "hi" if language.lower() == "hi" else "en"
    templates = QUESTION_TEMPLATES.get(next_field, {})
    return templates.get(lang_key, templates.get("en", f"Please provide your {next_field}."))


def is_form_complete(form: Dict[str, Any]) -> bool:
    """Check if all mandatory loan application parameters have been populated."""
    return all(form.get(k) is not None for k in REQUIRED_FIELDS)


def compute_emi(loan_amount: float, annual_rate: float, tenure_months: int) -> float:
    """Calculate Equated Monthly Installment (EMI) using reducing balance formula."""
    if loan_amount <= 0 or annual_rate <= 0 or tenure_months <= 0:
        return 0.0

    r = (annual_rate / 12.0) / 100.0
    n = tenure_months
    factor = (1.0 + r) ** n
    return float(loan_amount * r * factor / (factor - 1.0))


def generate_loan_summary(form: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    """Generate financial assessment and structured loan offer breakdown."""
    loan_amount = float(form.get("loan_amount") or 0.0)
    monthly_income = float(form.get("monthly_income") or 0.0)
    age = int(form.get("age") or 0)

    emi_at_11pct_5yr = compute_emi(loan_amount, 11.0, 60)
    emi_at_13pct_3yr = compute_emi(loan_amount, 13.0, 36)

    eligible = bool(monthly_income >= 3.0 * emi_at_11pct_5yr and 21 <= age <= 60)

    if eligible:
        next_step = "Application pre-filled. Proceed to upload documents."
    else:
        next_step = (
            "Your current income does not meet our criteria. "
            "Consider a smaller loan amount or adding a co-applicant."
        )

    return {
        "form": form,
        "emi_options": [
            {"rate_pct": 11.0, "tenure_months": 60, "emi": round(emi_at_11pct_5yr, 2)},
            {"rate_pct": 13.0, "tenure_months": 36, "emi": round(emi_at_13pct_3yr, 2)},
        ],
        "eligible": eligible,
        "next_step": next_step,
    }
