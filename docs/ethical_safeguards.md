# Bharat AI Banking: Ethical Safeguards & Regulatory Compliance

## 1. Regulatory Context

Digital lending in India requires strict adherence to regulatory guidelines established by the **Reserve Bank of India (RBI)** and statutory privacy mandates under the **Digital Personal Data Protection (DPDP) Act, 2023**.

Bharat AI Banking is engineered from the ground up to be compliant by design, rejecting predatory credit practices and prioritizing customer financial health.

---

## 2. DPDP Act 2023 Compliance Framework

### 2.1 Granular Purpose-Based Consent
The platform eliminates bundled consent checkboxes. Users provide independent, granular consent for four distinct processing purposes:
1. `recommendation`: Permission to evaluate financial attributes to suggest suitable banking products.
2. `stress_detection`: Permission to analyze transaction patterns to identify early signs of financial difficulty.
3. `chatbot`: Permission to process conversational text and voice queries in regional languages.
4. `marketing`: Permission to receive proactive communication regarding special offers.

### 2.2 Strict Purpose Limitation & Fail-Closed Gating
Every request arriving at the **Orchestrator Gateway (`services/orchestrator/main.py`)** undergoes automated consent verification:
```python
# Enforce DPDP recommendation consent before processing
rec_consent = check_consent(customer_id, "recommendation")
if not rec_consent["allowed"]:
    return JSONResponse(status_code=403, content={"error": "consent_denied"})
```
If consent has not been granted or has been revoked, the request fails closed immediately with HTTP `403 Forbidden`, blocking any downstream model inference or data processing.

### 2.3 Real-Time Consent Revocation
Under Section 6(4) of the DPDP Act, data principals possess the unconditional right to withdraw consent. The platform provides an instantaneous revocation endpoint (`POST /consent/revoke`):
- Immediately updates `data/consent_records.json`.
- Flushes in-memory gateway caches.
- Appends an audit event to the regulatory compliance log.

### 2.4 Immutable Regulatory Audit Trail (`data/audit_log.jsonl`)
In compliance with RBI digital lending guidelines, every algorithmic recommendation, consent check, revocation, and stress intervention is written to an immutable append-only JSONL log containing:
- ISO 8601 UTC timestamp
- Customer identifier
- Event type (`recommend`, `stress_check`, `intervene`, `consent_check`, `consent_revoke`)
- Regulatory decision (`allowed`, `blocked`, `escalated`, `info`)
- Detailed outcome summary and input payload metadata

---

## 3. Anti-Predatory Lending & Stress Gating

### 3.1 Algorithmic Credit Suppression
Traditional digital lending algorithms optimize for short-term loan disbursement volume, frequently targeting financially vulnerable individuals with high-interest payday loans and credit cards.

Bharat AI Banking implements an algorithmic **Anti-Predatory Stress Gate**:
- When the Early Warning service computes a `stress_level == "high"` (or risk score > 70), the Orchestrator automatically intercepts the recommendation pipeline.
- All revolving credit and loan products (`personal_loan`, `credit_card`, `home_loan`, `auto_loan`) are **strictly purged** from the candidate recommendations list.
- A **Financial Counseling Session** card is prepended at top priority, offering free, confidential financial advisory and restructuring assistance.

### 3.2 Empathetic, Non-Punitive Tone
When early stress is identified, the system never displays punitive alerts or derogatory notices. Instead, the intervention copy is framed with dignity and empathy:
> *"We noticed unexpected disruptions in your cash flow this month. You have been a valued customer with a strong repayment record. Would you like to explore a 30-day EMI pause or tenure extension to ease this month's burden?"*

---

## 4. Bias Mitigation & Demographic Fairness

### 4.1 Cohort Representation
The synthetic dataset and feature engineering pipeline are audited to ensure fairness across multiple socioeconomic cohorts:
- **Geographic Parity**: Equal treatment of Tier 2/3/4 rural profiles vs. Tier 1 metro profiles.
- **Occupational Inclusion**: Specific models and rules for informal gig workers and seasonal agricultural producers rather than requiring standard salary slips.
- **Language Parity**: Equal quality of explanations across all 10 supported regional Indian languages.

### 4.2 Protected Attribute Separation
Sensory demographic attributes (caste, religion, gender) are excluded from the feature space used by the XGBoost life-stage classifier and Random Forest risk scoring models.

---

## 5. Algorithmic Transparency & Human-in-the-Loop Oversight

### 5.1 The Borrower's Right to Explanation
Borrowers are never presented with arbitrary approval or rejection decisions. Every recommendation is paired with intuitive natural language rationales derived from mathematical **TreeSHAP** feature attributions.

### 5.2 Relationship Manager (RM) Co-Pilot
The platform does not automate punitive measures (e.g. account freezing or negative credit bureau reporting). High-stress alerts are surfaced to human Relationship Managers via the **RM View** in the Streamlit dashboard, enabling human relationship officers to provide empathetic, personalized customer assistance.
