"""
Feature extraction module for Bharat AI Banking Recommender Service.
Extracts rolling transaction window features and life-stage signals.
"""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, List


def extract_features(customer_id: str, transactions: List[Dict[str, Any]], profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract demographic and behavioral financial features for a customer.
    Reference date is the max transaction date for this customer.
    """
    if not transactions:
        # Default empty behavior if customer has no transactions
        ref_date = datetime(2025, 6, 30)
    else:
        dates = [datetime.strptime(t["date"], "%Y-%m-%d") for t in transactions]
        ref_date = max(dates)

    d30_cutoff = ref_date - timedelta(days=30)
    d60_cutoff = ref_date - timedelta(days=60)
    d90_cutoff = ref_date - timedelta(days=90)

    # Windowed buckets
    txns_30d = []
    txns_60d = []
    txns_90d = []
    txns_30_90d = []
    all_salary_dates: List[datetime] = []

    for t in transactions:
        t_date = datetime.strptime(t["date"], "%Y-%m-%d")
        if t_date >= d30_cutoff:
            txns_30d.append(t)
        if t_date >= d60_cutoff:
            txns_60d.append(t)
        if t_date >= d90_cutoff:
            txns_90d.append(t)
        if d90_cutoff <= t_date < d30_cutoff:
            txns_30_90d.append(t)

        if t["type"] == "credit" and t.get("category") == "salary":
            all_salary_dates.append(t_date)

    all_salary_dates.sort()

    # Incomes: sum of credits category="salary"
    income_30d = float(sum(t["amount"] for t in txns_30d if t["type"] == "credit" and t.get("category") == "salary"))
    income_60d = float(sum(t["amount"] for t in txns_60d if t["type"] == "credit" and t.get("category") == "salary"))
    income_90d = float(sum(t["amount"] for t in txns_90d if t["type"] == "credit" and t.get("category") == "salary"))

    # If no explicit salary credits in window, fall back to profile income
    if income_30d == 0 and profile.get("monthly_income"):
        income_30d = float(profile["monthly_income"])
        income_60d = income_30d * 2
        income_90d = income_30d * 3

    # Spendings: sum of debits
    spending_30d = float(sum(t["amount"] for t in txns_30d if t["type"] == "debit"))
    spending_60d = float(sum(t["amount"] for t in txns_60d if t["type"] == "debit"))
    spending_90d = float(sum(t["amount"] for t in txns_90d if t["type"] == "debit"))

    # Ratios
    denom_income = max(income_30d, 1.0)
    denom_spending = max(spending_30d, 1.0)

    savings_rate = (income_30d - spending_30d) / denom_income
    emi_debits_30d = sum(t["amount"] for t in txns_30d if t["type"] == "debit" and t.get("category") == "emi")
    emi_to_income_ratio = emi_debits_30d / denom_income

    discretionary_categories = {"dining", "shopping", "entertainment"}
    discretionary_debits = sum(t["amount"] for t in txns_30d if t["type"] == "debit" and t.get("category") in discretionary_categories)
    discretionary_ratio = discretionary_debits / denom_spending

    essential_categories = {"groceries", "utilities", "rent", "medical", "education"}
    essential_debits = sum(t["amount"] for t in txns_30d if t["type"] == "debit" and t.get("category") in essential_categories)
    essential_ratio = essential_debits / denom_spending

    # Transaction statistics 30d
    txn_count_30d = len(txns_30d)
    avg_txn_amount_30d = float(spending_30d / max(txn_count_30d, 1))

    # Salary regularity score: 1 - (std(days between salary credits) / 30), clipped [0, 1]
    if len(all_salary_dates) >= 2:
        intervals = [(all_salary_dates[i] - all_salary_dates[i - 1]).days for i in range(1, len(all_salary_dates))]
        mean_inter = sum(intervals) / len(intervals)
        var_inter = sum((x - mean_inter) ** 2 for x in intervals) / len(intervals)
        std_inter = math.sqrt(var_inter)
        salary_regularity_score = max(0.0, min(1.0, 1.0 - (std_inter / 30.0)))
    else:
        salary_regularity_score = 0.5

    # Salary growth percentage
    avg_90d_monthly_income = income_90d / 3.0
    salary_growth_pct = (income_30d - avg_90d_monthly_income) / max(avg_90d_monthly_income, 1.0)

    # Boolean life-stage signals
    salary_increase_20pct = bool(salary_growth_pct > 0.20)

    # new_recurring_education = any education debit in last 30d AND none in 30-90d window
    edu_30d = any(t["type"] == "debit" and t.get("category") == "education" for t in txns_30d)
    edu_30_90d = any(t["type"] == "debit" and t.get("category") == "education" for t in txns_30_90d)
    new_recurring_education = bool(edu_30d and not edu_30_90d)

    # emi_completed = no EMI debits 30d AND >=2 EMI debits in 30-90d
    emi_30d_count = sum(1 for t in txns_30d if t["type"] == "debit" and t.get("category") == "emi")
    emi_30_90d_count = sum(1 for t in txns_30_90d if t["type"] == "debit" and t.get("category") == "emi")
    emi_completed = bool(emi_30d_count == 0 and emi_30_90d_count >= 2)

    # large_one_time_purchase = any debit > 1.5 * avg monthly spend in last 30d
    avg_monthly_spend = spending_90d / 3.0 if spending_90d > 0 else spending_30d
    large_one_time_purchase = bool(any(t["type"] == "debit" and t["amount"] > 1.5 * max(avg_monthly_spend, 1.0) for t in txns_30d))

    # savings_surplus = savings_rate > 0.25 AND no investment debit in last 30d
    has_inv_30d = any(t["type"] == "debit" and t.get("category") == "investment" for t in txns_30d)
    savings_surplus = bool(savings_rate > 0.25 and not has_inv_30d)

    return {
        "customer_id": customer_id,
        "income_30d": round(income_30d, 2),
        "income_60d": round(income_60d, 2),
        "income_90d": round(income_90d, 2),
        "spending_30d": round(spending_30d, 2),
        "spending_60d": round(spending_60d, 2),
        "spending_90d": round(spending_90d, 2),
        "savings_rate": round(savings_rate, 4),
        "emi_to_income_ratio": round(emi_to_income_ratio, 4),
        "discretionary_ratio": round(discretionary_ratio, 4),
        "essential_ratio": round(essential_ratio, 4),
        "avg_txn_amount_30d": round(avg_txn_amount_30d, 2),
        "txn_count_30d": txn_count_30d,
        "salary_regularity_score": round(salary_regularity_score, 4),
        "salary_growth_pct": round(salary_growth_pct, 4),
        "salary_increase_20pct": salary_increase_20pct,
        "new_recurring_education": new_recurring_education,
        "emi_completed": emi_completed,
        "large_one_time_purchase": large_one_time_purchase,
        "savings_surplus": savings_surplus,
        "age": profile.get("age", 30),
        "dependents": profile.get("dependents", 0),
        "occupation": profile.get("occupation", "salaried"),
        "city_tier": profile.get("city_tier", 1),
        "preferred_language": profile.get("preferred_language", "en"),
        "existing_products": profile.get("existing_products", []),
    }
