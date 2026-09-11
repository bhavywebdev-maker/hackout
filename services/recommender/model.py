"""
XGBoost-based Life-Stage Classifier and Hybrid Recommender for Bharat AI Banking.
Trains on customer transaction features with rule-based fallback and stress gate.
"""

from collections import defaultdict
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("recommender.model")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

LIFE_STAGE_LABELS = [
    "young_saver",
    "first_borrower",
    "growing_family",
    "near_retirement",
    "gig_worker",
]

PRODUCT_NAMES = {
    "personal_loan": "Personal Loan",
    "home_loan": "Home Loan",
    "auto_loan": "Auto Loan",
    "credit_card": "Credit Card",
    "term_insurance": "Term Life Insurance",
    "health_insurance": "Comprehensive Health Cover",
    "mutual_fund": "Mutual Fund SIP",
    "fixed_deposit": "Fixed Deposit",
    "recurring_deposit": "Recurring Deposit",
}

NUMERIC_FEATURE_NAMES = [
    "income_30d",
    "income_60d",
    "income_90d",
    "spending_30d",
    "spending_60d",
    "spending_90d",
    "savings_rate",
    "emi_to_income_ratio",
    "discretionary_ratio",
    "essential_ratio",
    "avg_txn_amount_30d",
    "txn_count_30d",
    "salary_regularity_score",
    "salary_growth_pct",
    "age",
    "dependents",
    "city_tier",
]


def assign_rule_life_stage(age: int, emi_to_income_ratio: float, dependents: int) -> int:
    """Deterministic life-stage label builder based on regulatory guidance."""
    if age < 30 and emi_to_income_ratio < 0.10:
        return 0  # young_saver
    elif 0.10 <= emi_to_income_ratio <= 0.40 and age < 40:
        return 1  # first_borrower
    elif dependents >= 2:
        return 2  # growing_family
    elif age >= 55:
        return 3  # near_retirement
    else:
        return 4  # gig_worker


def train_life_stage_model(customer_features_list: List[Dict[str, Any]]) -> Tuple[Any, float, bool]:
    """
    Train an XGBoost multiclass model on extracted customer features.
    Returns: (model_or_None, accuracy, rule_based_fallback_used)
    """
    import joblib

    try:
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        try:
            import xgboost as xgb
            has_xgb = True
        except Exception:
            has_xgb = False
            from sklearn.ensemble import GradientBoostingClassifier
    except Exception as e:
        logger.warning(f"ML runtime not available ({e}); rule-based fallback used")
        return None, 0.0, True

    if len(customer_features_list) < 10:
        logger.info("< 10 samples available; rule-based fallback used")
        return None, 0.0, True

    X = []
    y = []
    for f in customer_features_list:
        row = [float(f.get(col, 0.0)) for col in NUMERIC_FEATURE_NAMES]
        label = assign_rule_life_stage(
            age=int(f.get("age", 30)),
            emi_to_income_ratio=float(f.get("emi_to_income_ratio", 0.0)),
            dependents=int(f.get("dependents", 0))
        )
        X.append(row)
        y.append(label)

    X_arr = np.array(X)
    y_arr = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.20, random_state=42, stratify=y_arr if len(set(y_arr)) > 1 else None
    )

    if has_xgb:
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=5,
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            eval_metric="mlogloss"
        )
    else:
        clf = GradientBoostingClassifier(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        )
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)

    train_acc = float(accuracy_score(y_train, train_preds))
    test_acc = float(accuracy_score(y_test, test_preds))

    model_type = "XGBoost" if has_xgb else "GradientBoosting"
    logger.info(f"{model_type} Life-Stage Classifier trained. Train Acc: {train_acc:.2%}, Test Acc: {test_acc:.2%}")

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "life_stage_xgb.pkl"
    joblib.dump(clf, model_path)
    logger.info(f"Model successfully saved to {model_path}")

    if test_acc < 0.60:
        logger.warning(f"Test accuracy {test_acc:.2%} < 60%; rule-based fallback used")
        return clf, test_acc, True

    return clf, test_acc, False


def predict_life_stage(model: Any, feature_row: Dict[str, Any], fallback: bool = False) -> Tuple[str, float]:
    """Predict life-stage name and confidence score for a customer."""
    age = int(feature_row.get("age", 30))
    emi_ratio = float(feature_row.get("emi_to_income_ratio", 0.0))
    dependents = int(feature_row.get("dependents", 0))

    if model is None or fallback:
        label_idx = assign_rule_life_stage(age, emi_ratio, dependents)
        return LIFE_STAGE_LABELS[label_idx], 0.95

    try:
        import numpy as np
        row_vec = np.array([[float(feature_row.get(col, 0.0)) for col in NUMERIC_FEATURE_NAMES]])
        probs = model.predict_proba(row_vec)[0]
        best_idx = int(np.argmax(probs))
        raw_max = float(max(probs))
        if raw_max < 0.50:
            scaled = 0.5 + (raw_max * 0.9)
        elif raw_max > 0.95:
            scaled = 0.95
        else:
            scaled = raw_max
        confidence = round(min(max(scaled, 0.5), 0.95), 2)
        return LIFE_STAGE_LABELS[best_idx], confidence
    except Exception as err:
        logger.warning(f"Prediction exception ({err}); applying rule fallback")
        label_idx = assign_rule_life_stage(age, emi_ratio, dependents)
        return LIFE_STAGE_LABELS[label_idx], 0.90


def build_cooccurrence_matrix(all_profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Build normalized collaborative co-occurrence matrix by life stage."""
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stage_totals: Dict[str, int] = defaultdict(int)

    for p in all_profiles:
        stage = p.get("life_stage", "young_saver")
        stage_totals[stage] += 1
        for prod in p.get("existing_products", []):
            counts[stage][prod] += 1

    # Normalize frequencies per stage
    collab_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for stage, prods in counts.items():
        max_c = max(prods.values()) if prods else 1
        for prod, c in prods.items():
            collab_scores[stage][prod] = round(c / max_c, 2)

    return collab_scores


def recommend(
    customer_id: str,
    features: Dict[str, Any],
    life_stage: str,
    existing_products: List[str],
    stress_level: str = "none",
    top_k: int = 3,
    cooccurrence_matrix: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate top_k product recommendations combining content rules & collaborative filtering,
    enforcing a strict stress gate when stress_level == 'high'.
    """
    content_scores: Dict[str, float] = defaultdict(float)
    reasons: Dict[str, List[str]] = defaultdict(list)

    # 1. Evaluate content-score rules
    savings_surplus = features.get("savings_surplus", False)
    savings_rate = features.get("savings_rate", 0.0)
    emi_to_income_ratio = features.get("emi_to_income_ratio", 0.0)
    dependents = features.get("dependents", 0)
    age = features.get("age", 30)
    salary_increase_20pct = features.get("salary_increase_20pct", False)
    emi_completed = features.get("emi_completed", False)
    large_one_time_purchase = features.get("large_one_time_purchase", False)

    if savings_surplus and savings_rate > 0.30:
        content_scores["mutual_fund"] += 0.9
        reasons["mutual_fund"].append("Savings rate above 30% with monthly surplus")
        content_scores["fixed_deposit"] += 0.7
        reasons["fixed_deposit"].append("Healthy surplus available for safe fixed yield")

    if emi_to_income_ratio < 0.30 and life_stage == "first_borrower":
        content_scores["auto_loan"] += 0.8
        reasons["auto_loan"].append("Low debt-to-income profile suited for vehicle loan")
        content_scores["personal_loan"] += 0.7
        reasons["personal_loan"].append("Moderate obligations ideal for initial personal credit line")

    if dependents >= 2 and life_stage == "growing_family":
        content_scores["term_insurance"] += 0.9
        reasons["term_insurance"].append("Growing family with 2+ dependents needs income protection")
        content_scores["health_insurance"] += 0.8
        reasons["health_insurance"].append("Comprehensive family health shield required")

    if life_stage == "young_saver":
        content_scores["credit_card"] += 0.7
        reasons["credit_card"].append("Young saver building initial credit history")
        content_scores["recurring_deposit"] += 0.6
        reasons["recurring_deposit"].append("Disciplined recurring monthly savings habit")

    if life_stage == "gig_worker":
        content_scores["recurring_deposit"] += 0.8
        reasons["recurring_deposit"].append("Flexible micro-savings tailored for variable income earners")
        content_scores["health_insurance"] += 0.75
        reasons["health_insurance"].append("Essential health safety net protecting independent gig earners")
        content_scores["fixed_deposit"] += 0.6
        reasons["fixed_deposit"].append("Liquid reserve buffer to navigate seasonal earnings fluctuation")

    if life_stage == "near_retirement":
        content_scores["fixed_deposit"] += 0.95
        reasons["fixed_deposit"].append("Assured capital preservation and regular interest payouts")
        content_scores["health_insurance"] += 0.85
        reasons["health_insurance"].append("Comprehensive senior medical cover for peace of mind")
        content_scores["mutual_fund"] += 0.65
        reasons["mutual_fund"].append("Conservative hybrid growth strategy to protect retirement corpus")

    if age >= 55:
        content_scores["fixed_deposit"] += 0.9
        reasons["fixed_deposit"].append("Senior citizen preferential interest rates available")
        content_scores["health_insurance"] += 0.8
        reasons["health_insurance"].append("Critical health protection for retirement stage")

    if salary_increase_20pct and "credit_card" not in existing_products:
        content_scores["credit_card"] += 0.8
        reasons["credit_card"].append("Salary increased over 20% over past 3 months")

    if emi_completed:
        content_scores["home_loan"] += 0.7
        reasons["home_loan"].append("Demonstrated clean repayment history qualifies for home loan")
        content_scores["auto_loan"] += 0.6
        reasons["auto_loan"].append("Successful loan track record enables auto financing")

    if large_one_time_purchase:
        content_scores["personal_loan"] += 0.5
        reasons["personal_loan"].append("Large one-time expense can be converted into convenient EMI")

    # Curated life-stage baseline reasons for candidate products
    PRODUCT_STAGE_REASONS = {
        "mutual_fund": f"Systematic wealth creation aligned with your {life_stage.replace('_', ' ')} goals",
        "fixed_deposit": f"Guaranteed yield and capital security suited for {life_stage.replace('_', ' ')}",
        "recurring_deposit": f"Disciplined monthly savings accumulation for {life_stage.replace('_', ' ')}",
        "health_insurance": f"Comprehensive health safeguard for your {life_stage.replace('_', ' ')} journey",
        "term_insurance": "Long-term financial protection safeguarding family security",
        "credit_card": "Convenient digital payment credit line and reward benefits",
        "personal_loan": "Flexible credit buffer supporting planned life expenses",
        "home_loan": "Long-term residential asset creation with structured financing",
        "auto_loan": "Structured mobility financing supporting your personal travel needs",
    }

    # Fallback base scores if no rules fired
    all_candidate_keys = list(PRODUCT_NAMES.keys())
    for prod in all_candidate_keys:
        if prod not in content_scores:
            content_scores[prod] = 0.2
            default_reason = PRODUCT_STAGE_REASONS.get(
                prod, f"Financial solution aligned with your {life_stage.replace('_', ' ')} stage"
            )
            reasons[prod].append(default_reason)

    # 2. Collaborative score computation
    collab_scores = cooccurrence_matrix.get(life_stage, {}) if cooccurrence_matrix else {}

    # 3. Final score blending: 0.6 * content + 0.4 * collaborative
    blended_scores = []
    credit_debt_products = {"personal_loan", "home_loan", "auto_loan", "credit_card"}

    for prod_id, prod_name in PRODUCT_NAMES.items():
        # Exclude credit/debt products if stress is high
        if stress_level.lower() == "high" and prod_id in credit_debt_products:
            continue

        c_score = content_scores.get(prod_id, 0.2)
        collab_score = collab_scores.get(prod_id, 0.3)
        final_score = round(0.6 * c_score + 0.4 * collab_score, 2)

        # Prioritize products not already held
        if prod_id in existing_products:
            final_score = round(final_score * 0.5, 2)

        blended_scores.append({
            "product_id": prod_id,
            "product_name": prod_name,
            "score": final_score,
            "reasons": reasons.get(prod_id, [f"Recommended for {life_stage.replace('_', ' ')}"]),
        })

    # Sort descending by score
    blended_scores.sort(key=lambda x: x["score"], reverse=True)
    results = blended_scores[:top_k]

    # Stress gate enforcement
    if stress_level.lower() == "high":
        # Remove any remaining credit products and append counseling card
        results = [r for r in results if r["product_id"] not in credit_debt_products]
        counseling_card = {
            "product_id": "counseling",
            "product_name": "Financial Counseling Session",
            "score": 0.9,
            "reasons": ["Financial stress detected — let's talk first"],
        }
        results.insert(0, counseling_card)
        results = results[:top_k]

    return results
