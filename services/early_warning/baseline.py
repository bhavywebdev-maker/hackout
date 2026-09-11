"""
Baseline and Z-Score computation module for Early Warning Service.
Calculates per-user statistical deviations and behavioral stress metrics.
"""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, List


def _safe_z_score(val: float, mean: float, std: float) -> float:
    """Compute z-score with safe denominator fallback."""
    denom = std if std > 1e-4 else max(0.15 * abs(mean), 1.0)
    return round((val - mean) / denom, 2)


def compute_baseline(customer_id: str, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute baseline vs 30-day recent behavioral deviations for a customer.
    Reference date = max transaction date.
    """
    if not transactions:
        return {
            "customer_id": customer_id,
            "z_scores": {
                "txn_amount": 0.0,
                "txn_count": 0.0,
                "discretionary_spend": 0.0,
                "essential_spend": 0.0,
                "atm_withdrawal_amount": 0.0,
                "atm_withdrawal_count": 0.0,
                "emi_payment_count": 0.0,
                "credit_amount": 0.0,
            },
            "raw_metrics": {
                "night_txn_ratio": 0.0,
                "new_beneficiary_count": 0,
                "days_since_last_salary": 0,
                "salary_drop_pct": 0.0,
                "missed_emi_count": 0,
            },
            "has_sufficient_history": False,
        }

    parsed_txns = []
    for t in transactions:
        t_date = datetime.strptime(t["date"], "%Y-%m-%d")
        parsed_txns.append((t_date, t))
    parsed_txns.sort(key=lambda x: x[0])

    ref_date = parsed_txns[-1][0]
    min_date = parsed_txns[0][0]

    if (ref_date - min_date).days < 30:
        return {
            "customer_id": customer_id,
            "z_scores": {
                "txn_amount": 0.0,
                "txn_count": 0.0,
                "discretionary_spend": 0.0,
                "essential_spend": 0.0,
                "atm_withdrawal_amount": 0.0,
                "atm_withdrawal_count": 0.0,
                "emi_payment_count": 0.0,
                "credit_amount": 0.0,
            },
            "raw_metrics": {
                "night_txn_ratio": 0.0,
                "new_beneficiary_count": 0,
                "days_since_last_salary": 0,
                "salary_drop_pct": 0.0,
                "missed_emi_count": 0,
            },
            "has_sufficient_history": False,
        }

    d30_cutoff = ref_date - timedelta(days=30)
    d60_cutoff = ref_date - timedelta(days=60)
    d90_cutoff = ref_date - timedelta(days=90)

    # Windows:
    # Recent 30d: [d30_cutoff, ref_date]
    # Baseline 60-90d history: [d90_cutoff, d30_cutoff)
    recent_txns: List[Dict[str, Any]] = []
    base_b1: List[Dict[str, Any]] = []
    base_b2: List[Dict[str, Any]] = []
    pre_stress_txns: List[Dict[str, Any]] = []
    all_salary_txns: List[Dict[str, Any]] = []

    for dt, t in parsed_txns:
        if dt >= d30_cutoff:
            recent_txns.append(t)
        elif dt >= d60_cutoff:
            base_b1.append(t)
        elif dt >= d90_cutoff:
            base_b2.append(t)

        if dt < d60_cutoff:
            pre_stress_txns.append(t)

        if t.get("category") == "salary" and t["type"] == "credit":
            all_salary_txns.append(t)

    # If base buckets are empty, use pre_stress_txns partitioned into two halves
    if not base_b1 or not base_b2:
        mid_idx = len(pre_stress_txns) // 2
        base_b1 = pre_stress_txns[mid_idx:]
        base_b2 = pre_stress_txns[:mid_idx]

    def _stat_z(val_fn):
        v_rec = val_fn(recent_txns)
        v1 = val_fn(base_b1)
        v2 = val_fn(base_b2)
        mean = (v1 + v2) / 2.0
        diff = abs(v1 - v2)
        std = diff / 2.0 if diff > 0 else max(0.15 * abs(mean), 1.0)
        return _safe_z_score(v_rec, mean, std)

    # 1. txn_amount (mean transaction amount)
    def _mean_amt(txns):
        debits = [t["amount"] for t in txns if t["type"] == "debit"]
        return sum(debits) / len(debits) if debits else 0.0
    z_txn_amount = _stat_z(_mean_amt)

    # 2. txn_count (daily transaction count)
    def _daily_cnt(txns):
        return len(txns) / 30.0
    z_txn_count = _stat_z(_daily_cnt)

    # 3. discretionary_spend (dining + shopping + entertainment)
    disc_cats = {"dining", "shopping", "entertainment"}
    def _disc_spend(txns):
        return sum(t["amount"] for t in txns if t["type"] == "debit" and t.get("category") in disc_cats)

    # Use baseline monthly discretionary spend across pre-stress baseline
    v_disc_rec = _disc_spend(recent_txns)
    pre_disc_spends = []
    # Compute 30-day chunks from pre-stress window
    if len(pre_stress_txns) > 10:
        chunk_days = 30
        for b_start in range(0, (d60_cutoff - min_date).days, chunk_days):
            t_start = min_date + timedelta(days=b_start)
            t_end = t_start + timedelta(days=chunk_days)
            chunk = [t for dt, t in parsed_txns if t_start <= dt < t_end]
            if chunk:
                pre_disc_spends.append(_disc_spend(chunk))

    if len(pre_disc_spends) >= 2:
        m_disc = sum(pre_disc_spends) / len(pre_disc_spends)
        var_disc = sum((x - m_disc) ** 2 for x in pre_disc_spends) / len(pre_disc_spends)
        std_disc = math.sqrt(var_disc)
        z_discretionary = _safe_z_score(v_disc_rec, m_disc, max(std_disc, 0.15 * m_disc))
    else:
        z_discretionary = _stat_z(_disc_spend)

    # 4. essential_spend (groceries + utilities + rent + medical + education)
    ess_cats = {"groceries", "utilities", "rent", "medical", "education"}
    def _ess_spend(txns):
        return sum(t["amount"] for t in txns if t["type"] == "debit" and t.get("category") in ess_cats)
    z_essential = _stat_z(_ess_spend)

    # 5. atm_withdrawal_amount
    def _atm_amt(txns):
        return sum(t["amount"] for t in txns if t["type"] == "debit" and t.get("category") == "atm_withdrawal")
    z_atm_amt = _stat_z(_atm_amt)

    # 6. atm_withdrawal_count
    def _atm_cnt(txns):
        return sum(1 for t in txns if t["type"] == "debit" and t.get("category") == "atm_withdrawal")
    z_atm_cnt = _stat_z(_atm_cnt)

    # 7. emi_payment_count
    def _is_valid_emi(t):
        if t.get("category") != "emi" or t["type"] != "debit":
            return False
        merch = t.get("merchant", "")
        return "Bounce" not in merch and "Return" not in merch

    def _emi_cnt(txns):
        return sum(1 for t in txns if _is_valid_emi(t))
    z_emi_cnt = _stat_z(_emi_cnt)

    # 8. credit_amount (incoming credits)
    def _credit_amt(txns):
        return sum(t["amount"] for t in txns if t["type"] == "credit")
    z_credit_amt = _stat_z(_credit_amt)

    # RAW METRICS
    # Night txn ratio: fraction of last-30d txns occurring 23:00-05:00
    night_count = 0
    for t in recent_txns:
        merch = t.get("merchant", "")
        if "Unusual Hour" in merch or "01:" in merch or "02:" in merch or "03:" in merch or "04:" in merch:
            night_count += 1
    night_txn_ratio = round(night_count / max(len(recent_txns), 1), 4)

    # New beneficiary count in last 30d not seen in pre-stress baseline
    recent_bens = {t.get("merchant", "") for t in recent_txns if t.get("category") == "transfer"}
    baseline_bens = {t.get("merchant", "") for t in pre_stress_txns if t.get("category") == "transfer"}
    new_beneficiary_count = len(recent_bens - baseline_bens)

    # Days since last salary credit
    if all_salary_txns:
        last_sal_date = datetime.strptime(all_salary_txns[-1]["date"], "%Y-%m-%d")
        days_since_last_salary = max(0, (ref_date - last_sal_date).days)
        last_salary_amt = all_salary_txns[-1]["amount"]
    else:
        days_since_last_salary = 99
        last_salary_amt = 0

    # Salary drop pct: (avg_salary_90d_prior - last_salary) / max(avg_salary_90d_prior, 1)
    prior_salaries = [t["amount"] for t in all_salary_txns if datetime.strptime(t["date"], "%Y-%m-%d") < d30_cutoff]
    if prior_salaries:
        avg_prior_sal = sum(prior_salaries) / len(prior_salaries)
        salary_drop_pct = round(max(0.0, (avg_prior_sal - last_salary_amt) / max(avg_prior_sal, 1.0)), 4)
    else:
        salary_drop_pct = 0.0

    # Missed EMI count
    bounce_charges = sum(1 for t in recent_txns if "Bounce" in t.get("merchant", "") or "Return" in t.get("merchant", ""))
    total_prior_emis = sum(1 for t in pre_stress_txns if _is_valid_emi(t))
    expected_emis = max(1, round(total_prior_emis / 4.0)) if total_prior_emis > 0 else 0
    recent_valid_emis = sum(1 for t in recent_txns if _is_valid_emi(t))
    cadence_missed = max(0, expected_emis - recent_valid_emis) if expected_emis > 0 else 0
    missed_emi_count = max(bounce_charges, cadence_missed)

    return {
        "customer_id": customer_id,
        "z_scores": {
            "txn_amount": z_txn_amount,
            "txn_count": z_txn_count,
            "discretionary_spend": z_discretionary,
            "essential_spend": z_essential,
            "atm_withdrawal_amount": z_atm_amt,
            "atm_withdrawal_count": z_atm_cnt,
            "emi_payment_count": z_emi_cnt,
            "credit_amount": z_credit_amt,
        },
        "raw_metrics": {
            "night_txn_ratio": night_txn_ratio,
            "new_beneficiary_count": new_beneficiary_count,
            "days_since_last_salary": days_since_last_salary,
            "salary_drop_pct": salary_drop_pct,
            "missed_emi_count": missed_emi_count,
        },
        "has_sufficient_history": True,
    }
