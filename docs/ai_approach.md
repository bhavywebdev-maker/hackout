# Bharat AI Banking: AI & Machine Learning Methodology

## 1. Architectural Philosophy

The AI/ML engine of Bharat AI Banking is designed specifically around the socio-economic realities of Indian retail lending:
1. **Explainability Over Black-Box Decisions**: Financial decisions must be explainable to borrowers in their native tongues and auditable by regulators (RBI).
2. **Anti-Predatory Guardrails**: Profit-maximizing recommender algorithms must not offer loans to financially stressed borrowers.
3. **Resilience & Hybrid Design**: Machine learning models (XGBoost, Isolation Forest, Random Forest) operate reliably on-device/in-container without requiring GPU hardware or proprietary cloud APIs, while Generative AI (Gemini 2.5 Flash) provides fluid vernacular personalization and grounding.

---

## 2. Recommender Service (`services/recommender/`)

### 2.1 Feature Engineering Pipeline (`features.py`)
From 90-day transaction logs and profile metadata, the system computes 10 standardized numeric features:
- `salary_growth_pct`: Percentage increase in salary over the past 3-month window.
- `savings_rate`: Ratio of monthly surplus to net monthly income.
- `discretionary_spend_ratio`: Spending on dining, entertainment, and shopping relative to fixed commitments.
- `emi_to_income_ratio`: Monthly debt service obligations divided by net income.
- `atm_withdrawal_frequency`: Average weekly count of cash withdrawals.
- `avg_transaction_amount`: Mean value of debit transactions.
- `late_night_tx_pct`: Percentage of transactions occurring between 11 PM and 5 AM.
- `failed_recurring_tx_count`: Count of bounced auto-debits or failed SIPs.
- `income_volatility`: Standard deviation of monthly cash inflows (critical for gig workers and farmers).
- `credit_utilization_ratio`: Balance divided by credit limit across active revolving lines.

### 2.2 XGBoost Life-Stage Classifier (`model.py`)
- **Classes**:
  - `young_saver`: Age 18–26, early career or student, building initial credit history.
  - `first_borrower`: Age 25–35, stable salary, seeking mobility or home financing.
  - `growing_family`: Age 30–48, dependents ≥ 2, prioritizing term/health insurance and child education.
  - `near_retirement`: Age 50+, capital preservation, pension, high fixed-deposit allocation.
  - `gig_worker`: Variable income, multi-source inflows, micro-savings and emergency liquidity needs.
- **Model**: Gradient-boosted decision trees (`XGBClassifier`) with softmax multi-class output.
- **Inference**: Returns the predicted life stage along with calibrated probability confidence (`confidence >= 0.6`). If model accuracy falls below acceptable bounds, the system gracefully falls back to deterministic rule-based heuristics.

### 2.3 Hybrid Scoring & Recommendation Engine
The candidate product ranker integrates content-based domain rules with collaborative user-item co-occurrence:
$$\text{Score}(p) = 0.6 \cdot \text{Score}_{\text{content}}(p) + 0.4 \cdot \text{Score}_{\text{collaborative}}(p)$$

- **Content Rules**: Specific lifecycle triggers (e.g. salary increase > 20% boosts credit card priority; growing family with dependents boosts term insurance; gig workers receive flexible recurring deposits and health protection).
- **Collaborative Co-occurrence**: Normalizes historical product adoption patterns across similar life-stage cohorts.
- **Stress-Gate Enforcement**: If `stress_level == "high"`, the engine removes all unsecured credit products (`personal_loan`, `credit_card`, `home_loan`, `auto_loan`) and injects a prioritized **Financial Counseling Session** card.

### 2.4 Mathematical Explainability (`explain.py`)
- **TreeSHAP Integration**: Computes exact Shapley feature attribution values using `shap.TreeExplainer`.
- **Natural Language Translation**: The top 2 SHAP drivers are converted into empathetic, intuitive explanations in the customer's selected language using Gemini 2.5 Flash, or structured deterministic regional templates.

---

## 3. Vernacular Chatbot Service (`services/chatbot/`)

### 3.1 Grounded FAQ RAG (`gemini_chat.py`)
- **Knowledge Base**: Policy guidelines, interest rates, eligibility criteria, and grievance procedures defined in `knowledge_base/banking_faq.md`.
- **Zero-Hallucination Prompting**: Gemini 2.5 Flash is conditioned strictly on the retrieved banking FAQ document. If a query falls outside the knowledge base, the bot politely redirects the customer to their Relationship Manager.

### 3.2 Multimodal Voice Pipeline (`audio_handler.py`)
- **Speech-to-Text (STT)**: User voice queries (WAV/MP3) are submitted directly to Gemini's native multimodal audio encoder, bypassing brittle third-party transcription steps.
- **Text-to-Speech (TTS)**: Bot responses are synthesized into clean regional speech audio using `gTTS` (Google Text-to-Speech), cached, and returned as Base64 MP3 for immediate browser playback.

### 3.3 Guided Form Auto-Fill (`form_fill.py`)
During multi-turn conversation, a structured entity-extraction pipeline identifies loan application fields:
- `monthly_income`, `desired_amount`, `tenure_months`, `loan_purpose`, `employment_type`.
- Automatically populates the customer's digital loan application as they speak naturally.

### 3.4 Pre-Disclosure Identity Guard (`identity.py`)
To prevent unauthorized access over conversational channels, the bot asks for customer verification (e.g., date of birth or last four digits of PAN) before revealing account balances, transaction history, or loan approval statuses.

---

## 4. Early Warning Stress Detection (`services/early_warning/`)

### 4.1 Rolling Spending Baselines (`baseline.py`)
Maintains a 90-day rolling baseline for each customer across four expenditure dimensions:
- Total monthly spend
- Discretionary spend
- Utility payment timeliness
- ATM cash withdrawal volume

Calculates rolling Z-scores:
$$Z = \frac{x_{\text{current}} - \mu_{90}}{\sigma_{90} + \epsilon}$$

### 4.2 Anomaly Detection & Risk Scoring (`anomaly.py`)
1. **Unsupervised Isolation Forest**: Identifies multidimensional behavioral outliers (e.g., unexpected sudden drops in grocery spending coupled with spikes in cash withdrawals).
2. **Supervised Random Forest Risk Scorer**: Predicts probability of 30-day delinquency (outputting a composite 0–100 risk score).
3. **Stress vs. Fraud Disentanglement**:
   - **Financial Stress**: High risk accompanied by missed utility bills, salary delay, or rising debt-to-income.
   - **Fraud**: High risk driven by sudden velocity surges, unusual late-night transactions, or foreign merchant codes without income drop.

### 4.3 Empathetic Intervention Engine (`intervene.py`)
Instead of punitive credit limit slashing or aggressive debt collection calls, the system activates **Empathetic Intervention**:
- Reassuring, stigma-free messages in the customer's native language.
- Proactive relief options: 30-day EMI moratorium, tenure extension to lower monthly installments, or no-fee restructuring.

---

## 5. Dual-Execution: Gemini Flash vs. Deterministic Mock

Bharat AI Banking operates with a unified interface (`shared/gemini_client.py`):
- **Live Mode**: Uses Google Gemini 2.5 Flash (`gemini-2.5-flash`) for rich, contextual natural language reasoning and multilingual explanations.
- **Mock Mode**: When `GEMINI_API_KEY` is absent or `USE_MOCK_LLM=true`, the client automatically utilizes rule-based, deterministic mock generators across all 10 Indian languages. This guarantees 100% testability and reliability in offline, CI/CD, or air-gapped environments.

---

## 6. Limitations & Future Roadmap

1. **Synthetic Training Data**: While calibrated to Indian banking distributions, models should be fine-tuned on anonymized core banking system (CBS) datasets.
2. **Account Aggregator (AA) Real-Time Telemetry**: Future iterations will integrate directly with Sahamati AA SDKs for real-time consent-based bank statement ingestion.
3. **Edge Model Deployment**: Quantizing the XGBoost and language models to run directly on mobile devices for offline vernacular advisory in low-connectivity rural areas.
