#!/usr/bin/env python3
"""
Generate realistic synthetic data for Bharat AI Banking.
Produces:
  - data/customer_profiles.json (50 Indian banking customers)
  - data/synthetic_transactions.json (6 months of transactions, Jan-Jun 2025)
  - data/consent_records.json (DPDP-compliant consent records)
"""

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List

# Attempt imports of optional dependencies
try:
    import numpy as np
    np.random.seed(42)
except ImportError:
    np = None

try:
    from faker import Faker
    fake = Faker("en_IN")
    Faker.seed(42)
except ImportError:
    fake = None

# Deterministic seed for Python random
random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Curated authentic Indian names fallback if Faker is not installed
INDIAN_NAMES = [
    ("Aarav", "Sharma", "male"),
    ("Priya", "Patel", "female"),
    ("Vivaan", "Verma", "male"),
    ("Ananya", "Singh", "female"),
    ("Vihaan", "Kumar", "male"),
    ("Diya", "Gupta", "female"),
    ("Arjun", "Reddy", "male"),
    ("Ishita", "Nair", "female"),
    ("Reyansh", "Chatterjee", "male"),
    ("Kavya", "Banerjee", "female"),
    ("Aayan", "Mukherjee", "male"),
    ("Khushi", "Das", "female"),
    ("Krishna", "Ghosh", "male"),
    ("Meera", "Iyer", "female"),
    ("Ishan", "Iyengar", "male"),
    ("Pari", "Rao", "female"),
    ("Shaurya", "Joshi", "male"),
    ("Pooja", "Kulkarni", "female"),
    ("Atharva", "Deshmukh", "male"),
    ("Riya", "Patil", "female"),
    ("Advik", "Mehta", "male"),
    ("Sneha", "Shah", "female"),
    ("Pranav", "Agarwal", "male"),
    ("Tanvi", "Bansal", "female"),
    ("Rajesh", "Choudhury", "male"),
    ("Sunita", "Yadav", "female"),
    ("Suresh", "Tiwari", "male"),
    ("Geeta", "Mishra", "female"),
    ("Ramesh", "Gill", "male"),
    ("Anita", "Dhillon", "female"),
    ("Sunil", "Sandhu", "male"),
    ("Rekha", "Bhatia", "female"),
    ("Anil", "Saxena", "male"),
    ("Lakshmi", "Bhattacharya", "female"),
    ("Manoj", "Sengupta", "male"),
    ("Manpreet", "Kaur", "female"),
    ("Karthik", "Subramanian", "male"),
    ("Swati", "Nambiar", "female"),
    ("Venkatesh", "Pillai", "male"),
    ("Deepa", "Menon", "female"),
    ("Murugan", "Chettiar", "male"),
    ("Kiran", "Venkatesan", "other"),
    # 8 Stress customers
    ("Amit", "Malhotra", "male"),
    ("Neha", "Kapoor", "female"),
    ("Vikram", "Chauhan", "male"),
    ("Suman", "Pandey", "female"),
    ("Gaurav", "Dubey", "male"),
    ("Pooja", "Tripathi", "female"),
    ("Sachin", "Bhardwaj", "male"),
    ("Simran", "Arora", "female"),
]

INDIAN_MERCHANTS = {
    "groceries": [
        "BigBasket Online", "Blinkit Instant", "DMart Supermarket",
        "Reliance Smart Bazaar", "Nature's Basket", "Local Kirana Store",
        "Spencers Retail", "More Supermarket"
    ],
    "dining": [
        "Swiggy Delivery", "Zomato Online", "Haldiram's", "Saravana Bhavan",
        "McDonald's India", "Chai Point", "Cafe Coffee Day", "Barbeque Nation"
    ],
    "shopping": [
        "Amazon India", "Flipkart Internet", "Myntra Fashion", "Ajio Trends",
        "Tata CLiQ", "Croma Electronics", "Nykaa E-retail"
    ],
    "utilities": [
        "BESCOM Electricity", "Tata Power", "Mahanagar Gas Ltd",
        "Indane Gas Agency", "Airtel Broadband", "Jio Postpaid", "Delhi Jal Board"
    ],
    "medical": [
        "Apollo Pharmacy", "Tata 1mg", "MedPlus Health", "Netmeds Online",
        "Fortis Healthcare", "Dr. Lal PathLabs", "Practo Care"
    ],
    "education": [
        "Kendriya Vidyalaya Fee", "Delhi Public School", "BYJU'S Learning",
        "Coursera Education", "Unacademy Subscription", "EuroKids Preschool"
    ],
    "entertainment": [
        "Netflix India", "BookMyShow Movies", "PVR Cinemas", "Spotify India",
        "Disney+ Hotstar", "YouTube Premium"
    ],
    "fuel": [
        "Indian Oil Petrol Pump", "Bharat Petroleum", "Hindustan Petroleum",
        "Shell Select"
    ],
    "rent": [
        "House Rent Transfer via Cred", "Nobroker Rent Pay",
        "Direct Landlord NEFT"
    ],
    "investment": [
        "Zerodha Broking", "Groww Mutual Funds", "SBI Mutual Fund SIP",
        "HDFC Life RD", "PPF Deposit SBI"
    ],
    "insurance": [
        "LIC Premium Payment", "HDFC ERGO Health", "Star Health Insurance",
        "ICICI Lombard Motor"
    ],
    "emi": [
        "HDFC Bank Loan EMI", "SBI Home Finance EMI", "Bajaj Finserv EMI",
        "ICICI Auto Loan EMI"
    ],
    "transfer": [
        "UPI P2P to Family", "NEFT to Landlord", "IMPS Fund Transfer",
        "Google Pay Transfer"
    ],
    "atm_withdrawal": [
        "SBI ATM Cash Withdrawal", "HDFC Bank ATM Cash", "Axis Bank ATM",
        "ICICI Bank ATM Withdrawal"
    ]
}


def get_indian_name(idx: int, gender: str) -> str:
    """Generate or retrieve realistic Indian name."""
    if fake is not None:
        try:
            if gender == "male":
                return fake.name_male()
            elif gender == "female":
                return fake.name_female()
            return fake.name()
        except Exception:
            pass
    first, last, _ = INDIAN_NAMES[idx % len(INDIAN_NAMES)]
    return f"{first} {last}"


def generate_customer_profiles() -> List[Dict[str, Any]]:
    """
    Generate 50 customer profiles with required distributions:
    - 40% Tier 2/3/4 cities (20 / 50 = 40.0%)
    - 25% gig workers or farmers (13 / 50 = 26.0% ~ 25%)
    - 20% young_saver (10 / 50 = 20.0%)
    - At least 5 languages represented (all 10 Indian languages used)
    - 8 customers identified by prefix STRESS in customer_id
    """
    customers: List[Dict[str, Any]] = []

    # Distribution mapping for 50 customers:
    # 42 standard customers: CUST0001 to CUST0042
    # 8 stress customers: STRESS0001 to STRESS0008

    # Life stages: exactly 10 young_saver (20%), 8 gig_worker, 13 first_borrower, 11 growing_family, 8 near_retirement
    standard_life_stages = (
        ["young_saver"] * 9 +
        ["gig_worker"] * 7 +
        ["first_borrower"] * 10 +
        ["growing_family"] * 9 +
        ["near_retirement"] * 7
    )
    random.shuffle(standard_life_stages)

    stress_life_stages = [
        "first_borrower", "growing_family", "gig_worker", "first_borrower",
        "growing_family", "young_saver", "gig_worker", "near_retirement"
    ]

    # City tiers: exactly 20 Tier 2/3/4 (40%) and 30 Tier 1
    standard_tiers = [1] * 25 + [2] * 8 + [3] * 5 + [4] * 4  # 17 in 2/3/4
    stress_tiers = [1] * 5 + [2] * 2 + [3] * 1               # 3 in 2/3/4
    random.shuffle(standard_tiers)
    random.shuffle(stress_tiers)

    # 10 Indian languages (all 10 used, > 5 represented)
    languages_pool = ["hi", "en", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa"]

    # Target 13 gig workers or farmers (9 gig_worker, 4 farmer)
    farmer_indices = {1, 3, 4, 5}

    for i in range(50):
        is_stress = (i >= 42)
        cust_id = f"STRESS{i - 41:04d}" if is_stress else f"CUST{i + 1:04d}"

        first, last, gdr = INDIAN_NAMES[i]
        gender = gdr if gdr != "other" else "other"
        name = get_indian_name(i, gender)

        if is_stress:
            life_stage = stress_life_stages[i - 42]
            city_tier = stress_tiers[i - 42]
            if life_stage == "gig_worker":
                occupation = "gig_worker"
            elif life_stage == "young_saver":
                occupation = "salaried"
            elif life_stage == "near_retirement":
                occupation = "retired"
            else:
                occupation = "salaried" if (i % 2 == 0) else "self_employed"
        else:
            life_stage = standard_life_stages[i]
            city_tier = standard_tiers[i]
            if i in farmer_indices:
                occupation = "farmer"
            elif life_stage == "gig_worker":
                occupation = "gig_worker"
            elif life_stage == "near_retirement":
                occupation = random.choice(["retired", "salaried"])
            elif life_stage == "young_saver":
                occupation = random.choice(["salaried", "student"])
            elif life_stage == "first_borrower":
                occupation = random.choice(["salaried", "self_employed"])
            else:
                occupation = random.choice(["salaried", "self_employed"])

        # Determine age
        if occupation == "student":
            age = random.randint(22, 24)
        elif occupation == "retired" or life_stage == "near_retirement":
            age = random.randint(58, 69)
        elif life_stage == "young_saver":
            age = random.randint(22, 28)
        elif life_stage == "first_borrower":
            age = random.randint(26, 36)
        elif life_stage == "growing_family":
            age = random.randint(34, 52)
        else:
            age = random.randint(25, 52)

        # Monthly income (15,000 to 250,000)
        if occupation == "student":
            monthly_income = random.randint(15, 22) * 1000
        elif occupation == "farmer":
            monthly_income = random.randint(18, 55) * 1000
        elif occupation == "gig_worker":
            monthly_income = random.randint(20, 48) * 1000
        elif occupation == "retired":
            monthly_income = random.randint(28, 75) * 1000
        elif life_stage == "young_saver":
            monthly_income = random.randint(32, 85) * 1000
        elif life_stage == "growing_family":
            monthly_income = random.randint(65, 240) * 1000
        else:
            monthly_income = random.randint(35, 175) * 1000

        # Dependents (0 to 5)
        if life_stage in ["young_saver", "student"]:
            dependents = 0 if random.random() > 0.3 else 1
        elif life_stage == "growing_family":
            dependents = random.randint(2, 4)
        elif life_stage == "near_retirement":
            dependents = random.randint(1, 3)
        else:
            dependents = random.randint(0, 3)

        pref_lang = languages_pool[i % len(languages_pool)]

        # Existing products based on life stage
        products = ["savings"]
        if life_stage == "young_saver":
            if random.random() > 0.4:
                products.append("credit_card")
            if random.random() > 0.5:
                products.append("recurring_deposit")
        elif life_stage == "first_borrower":
            products.append(random.choice(["personal_loan", "auto_loan"]))
            products.append("credit_card")
        elif life_stage == "growing_family":
            products.extend(["home_loan", "credit_card"])
            if random.random() > 0.4:
                products.append("auto_loan")
            if random.random() > 0.5:
                products.append("fixed_deposit")
        elif life_stage == "near_retirement":
            products.extend(["fixed_deposit", "recurring_deposit"])
            if random.random() > 0.5:
                products.append("credit_card")
        elif life_stage == "gig_worker":
            if occupation == "farmer":
                products.append("kisan_credit_card")
            if random.random() > 0.5:
                products.append("gold_loan")
            if random.random() > 0.6:
                products.append("personal_loan")
        if occupation == "farmer" and "kisan_credit_card" not in products:
            products.append("kisan_credit_card")

        # KYC status (complete, pending, not_started)
        if i < 42:
            kyc_status = random.choices(
                ["complete", "pending", "not_started"],
                weights=[0.85, 0.10, 0.05]
            )[0]
        else:
            kyc_status = random.choice(["complete", "pending"])

        profile = {
            "customer_id": cust_id,
            "name": name,
            "age": age,
            "gender": gender,
            "occupation": occupation,
            "monthly_income": monthly_income,
            "dependents": dependents,
            "city_tier": city_tier,
            "preferred_language": pref_lang,
            "existing_products": sorted(list(set(products))),
            "life_stage": life_stage,
            "kyc_status": kyc_status
        }
        customers.append(profile)

    return customers


def generate_synthetic_transactions(customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate 6 months of transactions (Jan 2025 – Jun 2025) for each customer.
    Average ~180 transactions/customer (~1 txn/day).
    Inject stress signals for 8 STRESS customers in May-June 2025:
    - Missed EMI (2 or more missed payments + bounce charges)
    - Sudden drop in discretionary spending (>60% reduction)
    - Multiple ATM withdrawals in unusual hours (1 AM - 4 AM)
    - Late salary credit or reduced salary amount
    - UPI transfers to new/unusual beneficiaries
    """
    all_customer_txns: List[Dict[str, Any]] = []
    txn_global_counter = 100000

    for cust in customers:
        cust_id = cust["customer_id"]
        is_stress = cust_id.startswith("STRESS")
        income = cust["monthly_income"]
        life_stage = cust["life_stage"]
        occupation = cust["occupation"]

        # Initial balance (1.8x - 2.6x monthly income)
        balance = int(income * random.uniform(1.8, 2.6))
        customer_txns: List[Dict[str, Any]] = []

        # Determine EMI parameters if applicable
        has_emi = False
        emi_amount = 0
        emi_merchant = "HDFC Bank Loan EMI"
        second_emi_amount = 0
        second_emi_merchant = "SBI Home Finance EMI"

        if "personal_loan" in cust["existing_products"]:
            has_emi = True
            emi_amount = int(income * 0.15)
            emi_merchant = "HDFC Bank Personal Loan EMI"
        elif "auto_loan" in cust["existing_products"]:
            has_emi = True
            emi_amount = int(income * 0.18)
            emi_merchant = "ICICI Auto Loan EMI"
        elif "home_loan" in cust["existing_products"]:
            has_emi = True
            emi_amount = int(income * 0.28)
            emi_merchant = "SBI Home Finance EMI"

        if life_stage == "growing_family" and ("home_loan" in cust["existing_products"] or "personal_loan" in cust["existing_products"]):
            has_emi = True
            if emi_amount == 0:
                emi_amount = int(income * 0.25)
            second_emi_amount = int(income * 0.10)
            second_emi_merchant = "Bajaj Finserv Consumer EMI"

        # Force EMI for stress customers to demonstrate missed EMI stress signal
        if is_stress and not has_emi:
            has_emi = True
            emi_amount = int(income * 0.20)
            emi_merchant = "HDFC Personal Loan EMI"

        # Generate month by month (Jan to Jun 2025)
        for month in range(1, 7):
            m_start = datetime(2025, month, 1)
            next_m = datetime(2025, month + 1, 1) if month < 6 else datetime(2025, 7, 1)
            days_in_month = (next_m - m_start).days
            is_last_60_days = (month in [5, 6])

            # 1. Income Credit
            if is_stress and is_last_60_days:
                # Stress: Late salary or reduced salary amount
                if month == 5:
                    salary_date = datetime(2025, 5, 24)
                    salary_amt = int(income * 0.65)
                    salary_desc = "NEFT Inward - Delayed Salary / Partial Remittance"
                else:
                    salary_date = datetime(2025, 6, 26)
                    salary_amt = int(income * 0.45)
                    salary_desc = "NEFT Inward - Reduced Salary / Stipend"

                balance += salary_amt
                txn_global_counter += 1
                customer_txns.append({
                    "txn_id": f"TXN{txn_global_counter:07d}",
                    "date": salary_date.strftime("%Y-%m-%d"),
                    "amount": salary_amt,
                    "type": "credit",
                    "category": "salary",
                    "merchant": salary_desc,
                    "balance_after": balance
                })
            elif occupation in ["gig_worker", "farmer"]:
                # Irregular credits (weekly/biweekly, varying amounts)
                payout_days = [3, 10, 17, 24] if days_in_month >= 28 else [5, 15, 25]
                for p_day in payout_days:
                    p_amt = int((income / len(payout_days)) * random.uniform(0.75, 1.25))
                    balance += p_amt
                    txn_global_counter += 1
                    merchant_src = "Swiggy Delivery Payout" if occupation == "gig_worker" else "APMC Mandi Crop Sale"
                    customer_txns.append({
                        "txn_id": f"TXN{txn_global_counter:07d}",
                        "date": datetime(2025, month, min(p_day, days_in_month)).strftime("%Y-%m-%d"),
                        "amount": p_amt,
                        "type": "credit",
                        "category": "salary",
                        "merchant": merchant_src,
                        "balance_after": balance
                    })
            else:
                sal_day = random.randint(1, 3)
                balance += income
                txn_global_counter += 1
                customer_txns.append({
                    "txn_id": f"TXN{txn_global_counter:07d}",
                    "date": datetime(2025, month, sal_day).strftime("%Y-%m-%d"),
                    "amount": income,
                    "type": "credit",
                    "category": "salary",
                    "merchant": "Employer Corp Payroll Salary Credit",
                    "balance_after": balance
                })

            # 2. EMI Payments
            if has_emi:
                if is_stress and is_last_60_days:
                    # Stress: Missed EMI in May & June (2 missed payments + bounce fee)
                    bounce_fee = 450
                    if balance >= bounce_fee:
                        balance -= bounce_fee
                    txn_global_counter += 1
                    customer_txns.append({
                        "txn_id": f"TXN{txn_global_counter:07d}",
                        "date": datetime(2025, month, 5).strftime("%Y-%m-%d"),
                        "amount": bounce_fee,
                        "type": "debit",
                        "category": "emi",
                        "merchant": f"NACH Return / EMI Bounce Charge - {emi_merchant}",
                        "balance_after": balance
                    })
                else:
                    if balance >= emi_amount:
                        balance -= emi_amount
                    txn_global_counter += 1
                    customer_txns.append({
                        "txn_id": f"TXN{txn_global_counter:07d}",
                        "date": datetime(2025, month, 5).strftime("%Y-%m-%d"),
                        "amount": emi_amount,
                        "type": "debit",
                        "category": "emi",
                        "merchant": emi_merchant,
                        "balance_after": balance
                    })
                    if second_emi_amount > 0:
                        if balance >= second_emi_amount:
                            balance -= second_emi_amount
                        txn_global_counter += 1
                        customer_txns.append({
                            "txn_id": f"TXN{txn_global_counter:07d}",
                            "date": datetime(2025, month, 10).strftime("%Y-%m-%d"),
                            "amount": second_emi_amount,
                            "type": "debit",
                            "category": "emi",
                            "merchant": second_emi_merchant,
                            "balance_after": balance
                        })

            # 3. Discretionary Spending
            # Baseline: 13-16 txns/month. Stress: 1-3 txns/month (>60% drop)
            discretionary_count = random.randint(13, 16)
            if is_stress and is_last_60_days:
                discretionary_count = random.randint(1, 3)

            for _ in range(discretionary_count):
                cat = random.choice(["dining", "shopping", "entertainment"])
                merch = random.choice(INDIAN_MERCHANTS[cat])
                amt = random.randint(150, 1800)
                if is_stress and is_last_60_days:
                    amt = random.randint(60, 300)
                day = random.randint(1, days_in_month)
                if balance > amt + 1000:
                    balance -= amt
                else:
                    amt = max(50, balance - 500)
                    balance -= amt
                txn_global_counter += 1
                customer_txns.append({
                    "txn_id": f"TXN{txn_global_counter:07d}",
                    "date": datetime(2025, month, day).strftime("%Y-%m-%d"),
                    "amount": amt,
                    "type": "debit",
                    "category": cat,
                    "merchant": merch,
                    "balance_after": balance
                })

            # 4. Essentials (groceries, utilities, fuel, medical, rent)
            routine_cats = ["groceries", "utilities", "fuel"]
            if life_stage in ["growing_family", "near_retirement"]:
                routine_cats.extend(["medical", "education"])
            for _ in range(random.randint(11, 15)):
                cat = random.choice(routine_cats)
                merch = random.choice(INDIAN_MERCHANTS[cat])
                amt = random.randint(250, 3200)
                day = random.randint(1, days_in_month)
                if balance > amt + 500:
                    balance -= amt
                else:
                    amt = max(100, balance - 200)
                    balance -= amt
                txn_global_counter += 1
                customer_txns.append({
                    "txn_id": f"TXN{txn_global_counter:07d}",
                    "date": datetime(2025, month, day).strftime("%Y-%m-%d"),
                    "amount": amt,
                    "type": "debit",
                    "category": cat,
                    "merchant": merch,
                    "balance_after": balance
                })

            # 5. Investments / Savings (monthly SIP or deposits)
            if not (is_stress and is_last_60_days) and random.random() > 0.25:
                inv_amt = int(income * random.uniform(0.05, 0.12))
                if balance > inv_amt + 2000:
                    balance -= inv_amt
                    txn_global_counter += 1
                    customer_txns.append({
                        "txn_id": f"TXN{txn_global_counter:07d}",
                        "date": datetime(2025, month, random.randint(7, 14)).strftime("%Y-%m-%d"),
                        "amount": inv_amt,
                        "type": "debit",
                        "category": "investment",
                        "merchant": random.choice(INDIAN_MERCHANTS["investment"]),
                        "balance_after": balance
                    })

            # 6. Stress injection specific transactions in last 60 days
            if is_stress and is_last_60_days:
                # Multiple ATM withdrawals at unusual hours (1 AM - 4 AM)
                unusual_hours = ["01:15 AM", "02:40 AM", "03:10 AM", "03:55 AM"]
                for _ in range(random.randint(3, 5)):
                    atm_day = random.randint(1, days_in_month)
                    atm_amt = random.choice([2000, 4500, 5000, 10000])
                    if balance > atm_amt + 200:
                        balance -= atm_amt
                    else:
                        atm_amt = max(500, balance - 100)
                        balance -= atm_amt
                    unusual_time = random.choice(unusual_hours)
                    txn_global_counter += 1
                    customer_txns.append({
                        "txn_id": f"TXN{txn_global_counter:07d}",
                        "date": datetime(2025, month, atm_day).strftime("%Y-%m-%d"),
                        "amount": atm_amt,
                        "type": "debit",
                        "category": "atm_withdrawal",
                        "merchant": f"SBI ATM Cash Withdrawal - Unusual Hour ({unusual_time})",
                        "balance_after": balance
                    })

                # UPI transfers to new/unusual beneficiaries
                for _ in range(random.randint(2, 4)):
                    transfer_day = random.randint(5, days_in_month)
                    transfer_amt = random.randint(4000, 15000)
                    if balance > transfer_amt + 200:
                        balance -= transfer_amt
                    else:
                        transfer_amt = max(500, balance - 100)
                        balance -= transfer_amt
                    txn_global_counter += 1
                    customer_txns.append({
                        "txn_id": f"TXN{txn_global_counter:07d}",
                        "date": datetime(2025, month, transfer_day).strftime("%Y-%m-%d"),
                        "amount": transfer_amt,
                        "type": "debit",
                        "category": "transfer",
                        "merchant": "UPI P2P to Emergency Private Lender / Hand Loan",
                        "balance_after": balance
                    })

        # Sort transactions chronologically
        customer_txns.sort(key=lambda x: x["date"])

        # Re-compute balance_after sequentially to guarantee absolute mathematical consistency
        rolling_balance = int(income * 2.2)
        for t in customer_txns:
            if t["type"] == "credit":
                rolling_balance += t["amount"]
            else:
                rolling_balance -= t["amount"]
            t["balance_after"] = rolling_balance

        all_customer_txns.append({
            "customer_id": cust_id,
            "transactions": customer_txns
        })

    return all_customer_txns


def generate_consent_records(customers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate consent records for all 50 customers.
    Ensure at least 5 customers have missing consent for one or more purposes.
    """
    records: List[Dict[str, Any]] = []

    # 6 designated customers with missing/revoked consents to demo consent enforcement
    missing_consent_map = {
        "CUST0005": ["recommendation"],
        "CUST0012": ["stress_detection"],
        "CUST0019": ["chatbot"],
        "CUST0025": ["recommendation", "marketing"],
        "CUST0033": ["stress_detection", "recommendation"],
        "STRESS0003": ["stress_detection"]
    }

    for cust in customers:
        cid = cust["customer_id"]
        missing_purposes = missing_consent_map.get(cid, [])

        consents = []
        for purpose in ["recommendation", "stress_detection", "chatbot"]:
            if purpose in missing_purposes:
                consents.append({
                    "purpose": purpose,
                    "given": False,
                    "timestamp": None,
                    "expires_at": None
                })
            else:
                consents.append({
                    "purpose": purpose,
                    "given": True,
                    "timestamp": "2025-01-01T10:00:00Z",
                    "expires_at": "2026-01-01T10:00:00Z"
                })

        # Marketing consent (opt-in for privacy compliance)
        marketing_given = (cid not in missing_purposes) and (random.random() > 0.7)
        consents.append({
            "purpose": "marketing",
            "given": marketing_given,
            "timestamp": "2025-01-01T10:00:00Z" if marketing_given else None,
            "expires_at": "2026-01-01T10:00:00Z" if marketing_given else None
        })

        records.append({
            "customer_id": cid,
            "consents": consents
        })

    return records


def main() -> None:
    print("Generating synthetic data for Bharat AI Banking...")
    profiles = generate_customer_profiles()
    transactions = generate_synthetic_transactions(profiles)
    consents = generate_consent_records(profiles)

    # Save to files
    profiles_path = os.path.join(DATA_DIR, "customer_profiles.json")
    txns_path = os.path.join(DATA_DIR, "synthetic_transactions.json")
    consents_path = os.path.join(DATA_DIR, "consent_records.json")

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False)

    with open(txns_path, "w", encoding="utf-8") as f:
        json.dump(transactions, f, indent=2, ensure_ascii=False)

    with open(consents_path, "w", encoding="utf-8") as f:
        json.dump(consents, f, indent=2, ensure_ascii=False)

    # Summary Statistics Calculation
    total_customers = len(profiles)
    stress_customers = [c["customer_id"] for c in profiles if c["customer_id"].startswith("STRESS")]
    tier_234_count = sum(1 for c in profiles if c["city_tier"] in [2, 3, 4])
    gig_or_farmer = sum(1 for c in profiles if c["occupation"] in ["gig_worker", "farmer"])
    young_savers = sum(1 for c in profiles if c["life_stage"] == "young_saver")
    languages = set(c["preferred_language"] for c in profiles)
    
    total_txns = sum(len(ct["transactions"]) for ct in transactions)
    avg_txns = total_txns / total_customers if total_customers else 0

    missing_consent_count = 0
    for r in consents:
        core_consents = [c["given"] for c in r["consents"] if c["purpose"] in ["recommendation", "stress_detection", "chatbot"]]
        if not all(core_consents):
            missing_consent_count += 1

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA SUMMARY:")
    print("=" * 60)
    print(f"Total Customer Profiles: {total_customers}")
    print(f"Customers with Transaction Histories: {len(transactions)}")
    print(f"Stress-Injected Customers: {len(stress_customers)} ({', '.join(stress_customers)})")
    print(f"Tier 2/3/4 Distribution: {tier_234_count}/{total_customers} ({tier_234_count/total_customers*100:.1f}%)")
    print(f"Gig Workers / Farmers: {gig_or_farmer}/{total_customers} ({gig_or_farmer/total_customers*100:.1f}%)")
    print(f"Life Stage 'young_saver': {young_savers}/{total_customers} ({young_savers/total_customers*100:.1f}%)")
    print(f"Distinct Languages Represented: {len(languages)} ({', '.join(sorted(languages))})")
    print(f"Total Transactions Generated: {total_txns}")
    print(f"Average Transactions / Customer: {avg_txns:.1f}")
    print(f"Customers with Missing Core Consent: {missing_consent_count}")
    print("=" * 60)
    print(f"Files successfully written to: {DATA_DIR}")


if __name__ == "__main__":
    main()
