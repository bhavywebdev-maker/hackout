# Bharat AI Banking: 3-Minute Video Demo Script

## Total Duration: 03:00
## Target Audience: Hackathon Judges & Industry Reviewers
## Core Theme: Digital Transformation in Lending

---

## 🛠️ Pre-Requisites & Demo Setup

1. All 5 Docker containers running via:
   ```bash
   docker compose up --build -d
   ```
2. Browser tabs prepared:
   - **Tab A**: `http://localhost:8501` (Streamlit Frontend)
   - **Tab B**: `http://localhost:8000/docs` (Orchestrator Swagger Docs)
3. Ensure audio output / speakers are enabled for gTTS voice demonstration.

---

## 🎬 Timed Storyboard & Script

### Scene 1: Introduction & The Bharat Problem (0:00 – 0:30)
- **Visual**: Title slide / Browser opening on Bharat AI Banking header.
- **Presenter (Voiceover)**:
  > *"Over 400 million citizens across Bharat—from gig workers to small shop owners—struggle with traditional banking. They face generic financial products pushed without context, aggressive predatory credit when under strain, linguistic alienation, and a lack of data privacy.*
  >
  > *Welcome to **Bharat AI Banking**, an intelligent, life-stage aware, vernacular financial co-pilot driving digital transformation in lending."*

---

### Scene 2: Hyper-Personalized Recommender & SHAP Explanations (0:30 – 1:00)
- **Visual**:
  - Switch to **Customer View** on `http://localhost:8501`.
  - Select customer `CUST0001 (Ramesh Kumar - Young Saver / Salaried)`.
  - Click **"Get Recommendations"**.
  - Show recommended products: *Mutual Fund SIP, Fixed Deposit, Recurring Deposit*.
  - Point to the **Why this was recommended** section showing SHAP feature importances.
- **Presenter (Voiceover)**:
  > *"Here is Ramesh, a young salaried professional in Jaipur. Our XGBoost engine analyzes his transaction history and identifies him as a 'Young Saver' with 88% confidence.*
  >
  > *Instead of generic credit cards, it recommends wealth accumulation products tailored to his high savings rate. Look at the explainability box: using TreeSHAP, we translate complex mathematical attributions into plain English or Hindi—guaranteeing algorithmic transparency."*

---

### Scene 3: Vernacular Voice & Grounded Chatbot (1:00 – 1:30)
- **Visual**:
  - Scroll down to the **Vernacular Banking Assistant**.
  - Switch language dropdown to **Hindi (हिन्दी)**.
  - Type query: `"मुझे पर्सनल लोन की ब्याज दर बताओ"` (Tell me personal loan interest rates) or click voice sample.
  - Show grounded answer retrieved from `banking_faq.md` and play audio response via gTTS.
- **Presenter (Voiceover)**:
  > *"Bharat speaks over 10 major languages. Our chatbot service supports 10 Indian languages plus code-mixed Hinglish.*
  >
  > *Powered by Gemini 2.5 Flash and grounded strictly in our verified banking FAQ knowledge base, it answers complex queries without hallucination and speaks back in authentic regional audio using our integrated voice pipeline."*

---

### Scene 4: Early Warning & Anti-Predatory Stress Gating (1:30 – 2:00)
- **Visual**:
  - In Customer View, switch customer dropdown to `STRESS0001 (Suresh Patel - Gig Worker)`.
  - Click **"Get Recommendations"**.
  - Notice the **Red/Orange Alert Badge**: *"High Financial Stress Detected (Risk Score: 78/100)"*.
  - Show the **Empathetic Intervention Card**: *"Let's talk first - Complimentary Financial Counseling Session"*.
  - Highlight that loans and credit cards are completely absent.
- **Presenter (Voiceover)**:
  > *"Now witness the core innovation: anti-predatory lending.*
  >
  > *Suresh is a gig delivery driver who recently experienced an income shock and delayed bill payments. Our Early Warning Isolation Forest flags this behavioral anomaly with a risk score of 78.*
  >
  > *Where predatory lenders would push high-interest instant loans, our stress gate strictly blocks all credit lines and presents an empathetic intervention card offering a free counseling session and loan restructuring."*

---

### Scene 5: Relationship Manager (RM) View & Portfolio Health (2:00 – 2:30)
- **Visual**:
  - Click on **Tab 2: Relationship Manager View**.
  - Show the **Portfolio Health Overview** (Safe vs Stressed customer distribution chart).
  - Click on `STRESS0001` in the customer drilldown to view transaction anomalies and suggested relief plans (30-day EMI moratorium).
- **Presenter (Voiceover)**:
  > *"Switching to the Relationship Manager View, bank officers get a proactive co-pilot. Instead of waiting for a loan to default into an NPA, RMs see portfolio stress distributions in real time.*
  >
  > *They can trigger non-punitive interventions, offer tenure extensions, and support customers before financial distress worsens."*

---

### Scene 6: DPDP Consent, Audit Trail & Conclusion (2:30 – 3:00)
- **Visual**:
  - Show the **Live Regulatory Audit Trail** table streaming in RM View.
  - Demonstrate revoking consent for `recommendation` in the sidebar and show the instant `HTTP 403 Forbidden` response.
  - Conclude on the architecture slide.
- **Presenter (Voiceover)**:
  > *"Every single action—from consent verification to stress interventions—is recorded in an immutable, RBI-compliant audit trail.*
  >
  > *Under the DPDP Act 2023, if a user revokes consent, processing stops instantly. Bharat AI Banking transforms lending from transactional debt-pushing into empathetic, life-stage financial empowerment for the next billion users.*
  >
  > *Thank you!"*
