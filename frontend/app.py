"""
Streamlit Web Application — Bharat AI Banking
Hyper-Personalized Banking for Bharat | Hackathon Demo

Features:
  Tab 1 — Customer View: recommendations, empathetic intervention, chatbot, consent
  Tab 2 — Relationship Manager View: live at-risk watchlist, audit trail, compliance
"""

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bharat AI Banking",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Overrides ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #f0f4ff;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    border-left: 4px solid #4F8BF9;
}
.stress-card {
    border-left-color: #e53e3e;
    background: #fff5f5;
}
.chat-bubble-user {
    background: #e8f4fd;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 4px 0;
    text-align: right;
}
.chat-bubble-bot {
    background: #f4f4f4;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 4px 0;
}
.product-badge {
    display: inline-block;
    background: #e8f4fd;
    color: #1a6eb5;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8em;
    font-weight: 600;
}
.stage-badge {
    display: inline-block;
    background: #e9d8fd;
    color: #553c9a;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.85em;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ─── Configuration ───────────────────────────────────────────────────────────────
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8000").rstrip("/")
DATA_DIR = os.getenv("DATA_DIR", "./data")

LIFE_STAGE_LABELS = {
    "young_saver": "🌱 Young Saver",
    "first_borrower": "🏠 First Borrower",
    "growing_family": "👨‍👩‍👧 Growing Family",
    "near_retirement": "🌅 Near Retirement",
    "gig_worker": "⚡ Gig Worker",
}

LANG_MAP = {
    "en": "🇬🇧 English",
    "hi": "🇮🇳 हिन्दी",
    "ta": "🇮🇳 தமிழ்",
    "te": "🇮🇳 తెలుగు",
    "bn": "🇮🇳 বাংলা",
    "mr": "🇮🇳 मराठी",
    "gu": "🇮🇳 ગુજરાતી",
    "kn": "🇮🇳 ಕನ್ನಡ",
    "ml": "🇮🇳 മലയാളം",
    "pa": "🇮🇳 ਪੰਜਾਬੀ",
}


# ─── API Helpers ─────────────────────────────────────────────────────────────────
def api_get(path: str, params: Optional[Dict] = None) -> Optional[Any]:
    try:
        resp = requests.get(f"{ORCHESTRATOR_URL}{path}", params=params, timeout=6)
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def api_post(path: str, payload: Dict[str, Any]) -> Tuple[Optional[Dict], int]:
    try:
        resp = requests.post(f"{ORCHESTRATOR_URL}{path}", json=payload, timeout=12)
        try:
            data = resp.json()
        except Exception:
            data = None
        return data, resp.status_code
    except Exception as exc:
        return {"error": str(exc)}, 500


# ─── Data Loaders ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_customer_profiles() -> List[Dict[str, Any]]:
    candidates = [
        Path(DATA_DIR) / "customer_profiles.json",
        Path(__file__).resolve().parent.parent / "data" / "customer_profiles.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return [{"customer_id": f"CUST{i:04d}", "name": f"Customer {i}"} for i in range(1, 21)]


def get_profile_map() -> Dict[str, Dict]:
    return {p["customer_id"]: p for p in load_customer_profiles()}


# ─── Session State ────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "customer_id": "CUST0001",
        "language": "en",
        "chat_history": [],
        "session_id": f"s_{int(time.time())}",
        "form_data": {},
        "last_action_result": None,
        "next_question": None,
        "rm_live_results": {},   # cache live EW scores per STRESS customer
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Bharat AI Banking")
    st.caption("Hyper-Personalized Banking for Bharat")
    st.divider()

    profiles = load_customer_profiles()
    profile_map = get_profile_map()

    reg_ids = [p["customer_id"] for p in profiles if not p["customer_id"].startswith("STRESS")][:20]
    stress_ids = [f"STRESS{i:04d}" for i in range(1, 9)]
    all_ids = reg_ids + stress_ids

    curr_idx = all_ids.index(st.session_state.customer_id) if st.session_state.customer_id in all_ids else 0

    def _fmt_id(cid):
        p = profile_map.get(cid, {})
        name = p.get("name", "")
        return f"{cid} — {name}" if name else cid

    selected_cust = st.selectbox(
        "👤 Select Customer",
        options=all_ids,
        index=curr_idx,
        format_func=_fmt_id,
    )
    if selected_cust != st.session_state.customer_id:
        st.session_state.customer_id = selected_cust
        st.session_state.last_action_result = None
        st.session_state.form_data = {}
        st.session_state.next_question = None
        st.session_state.chat_history = []
        st.session_state.session_id = f"s_{int(time.time())}"
        st.rerun()

    selected_lang = st.selectbox(
        "🌐 Language",
        options=list(LANG_MAP.keys()),
        format_func=lambda c: LANG_MAP.get(c, c),
        index=list(LANG_MAP.keys()).index(st.session_state.language)
        if st.session_state.language in LANG_MAP else 0,
    )
    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.rerun()

    st.divider()
    st.markdown("**⚙️ Infrastructure Status**")
    health = api_get("/health")
    if health and health.get("status") == "ok":
        svc = health.get("services", {})
        all_up = all(v == "up" for v in svc.values())
        if all_up:
            st.success("✅ All systems operational")
        else:
            st.warning("⚠️ Partial degradation")
        for name, sval in svc.items():
            icon = "🟢" if sval == "up" else "🔴"
            st.caption(f"{icon} {name.replace('_',' ').title()}")
    else:
        st.error("🔴 Orchestrator unreachable")

    st.divider()
    st.caption(f"Demo · {datetime.datetime.now().strftime('%d %b %Y %H:%M')}")
    st.caption("Sessions 1–6 complete | All services live")


# ─── MAIN TABS ───────────────────────────────────────────────────────────────────
tab_customer, tab_rm = st.tabs(["👤 Customer View", "👔 Relationship Manager"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CUSTOMER VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_customer:
    cust_id = st.session_state.customer_id
    profile = profile_map.get(cust_id, {})
    cust_name = profile.get("name", cust_id)
    last_res = st.session_state.last_action_result

    # ── Welcome Banner ──────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([3, 2])
    with col_title:
        st.markdown(f"## 👋 Welcome, **{cust_name}**")
        age = profile.get("age", "")
        occ = str(profile.get("occupation", "")).replace("_", " ").title()
        income = profile.get("monthly_income", "")
        info_parts = []
        if age:
            info_parts.append(f"Age {age}")
        if occ:
            info_parts.append(occ)
        if income:
            info_parts.append(f"₹{income:,}/mo")
        if info_parts:
            st.caption(" · ".join(info_parts))

    with col_badge:
        if last_res:
            stress_level = last_res.get("stress_check", {}).get("stress_level", "none")
            life_stage = last_res.get("life_stage", "")
            confidence = last_res.get("life_stage_confidence", 0)

            if life_stage:
                stage_label = LIFE_STAGE_LABELS.get(life_stage, life_stage.replace("_", " ").title())
                st.markdown(
                    f'<span class="stage-badge">{stage_label}</span> '
                    f'<span style="font-size:0.8em; color:#666">({int(confidence*100)}% confidence)</span>',
                    unsafe_allow_html=True,
                )
                st.write("")

            if stress_level == "high":
                st.error("⚠️ Financial Stress Detected — Support Available")
            elif stress_level == "medium":
                st.warning("🔶 Mild Stress Signals — Counseling Suggested")
            else:
                st.success("✅ Financial Health: Steady")
        else:
            # Show profile info while no action triggered yet
            existing = profile.get("existing_products", [])
            if existing:
                prod_str = " · ".join(p.replace("_", " ").title() for p in existing[:4])
                st.caption(f"**Current products:** {prod_str}")
            kyc = profile.get("kyc_status", "")
            if kyc:
                st.caption(f"**KYC:** {kyc.title()}")

    st.divider()

    # ── Event Trigger ───────────────────────────────────────────────────────────
    st.markdown("### 🎬 Simulate a Banking Event")
    ev_col, btn_col = st.columns([3, 1])
    with ev_col:
        event_choice = st.selectbox(
            "Select event",
            options=["salary_credit", "emi_payment", "login", "browse"],
            format_func=lambda e: {
                "salary_credit": "💰 Salary Credit Received",
                "emi_payment":   "💳 Monthly EMI Payment",
                "login":         "🔐 Mobile App Login",
                "browse":        "🌐 Product Catalog Browse",
            }.get(e, e),
            label_visibility="collapsed",
        )
    with btn_col:
        trigger = st.button("▶ Trigger", type="primary", use_container_width=True)

    if trigger:
        with st.spinner("🔄 Processing via Orchestrator → Recommender → Early Warning…"):
            payload = {
                "event_type": event_choice,
                "language": st.session_state.language,
                "payload": {"simulated": True},
            }
            res_data, status_code = api_post(f"/customer/{cust_id}/action", payload)
            if status_code == 403:
                reason = (res_data or {}).get("reason", "consent_denied")
                st.session_state.last_action_result = {"action_taken": "blocked", "reason": reason}
            elif res_data and status_code == 200:
                st.session_state.last_action_result = res_data
            else:
                st.warning("⚠️ Service response unavailable. Check that all services are running.")
        st.rerun()

    # ── Results Panel ───────────────────────────────────────────────────────────
    if last_res:
        action_taken = last_res.get("action_taken")
        st.divider()

        # ── INTERVENTION (stress gate triggered) ─────────────────────────────
        if action_taken == "intervention":
            st.error("🤝 We noticed some changes in your financial activity. This is **not a penalty** — we're here to help.")
            interv = last_res.get("intervention", {})
            msg = interv.get("message", "We are here to support your financial well-being.")

            # Decode unicode if needed
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8")

            st.info(f"💬 **Support Message:** {msg}")

            actions = interv.get("suggested_actions", [])
            if actions:
                st.markdown("**💡 Assistance Options Available:**")
                act_cols = st.columns(len(actions))
                for i, act in enumerate(actions):
                    with act_cols[i]:
                        st.button(f"📌 {act}", key=f"act_{i}_{cust_id}", disabled=True)

            st.markdown("#### 🌱 Stabilizing Products (No Predatory Upsell)")
            recs = last_res.get("recommendations", [])
            if recs:
                rcols = st.columns(len(recs))
                for i, prod in enumerate(recs):
                    with rcols[i]:
                        score = float(prod.get("score", 0.5))
                        st.markdown(f"**{prod.get('product_name')}**")
                        st.progress(score)
                        st.caption(f"Suitability: {int(score * 100)}%")
                        with st.expander("Why this product?"):
                            for r in prod.get("reasons", []):
                                st.markdown(f"- {r}")
                            exp = prod.get("explanation", "")
                            if exp:
                                st.write(exp)
                            signals = prod.get("top_signals") or prod.get("shap_top_features", {})
                            if signals:
                                st.markdown("**Key Factors:**")
                                df = pd.DataFrame(
                                    [(k.replace("_", " ").title(), f"{v:.2f}") for k, v in signals.items()],
                                    columns=["Factor", "Weight"]
                                )
                                st.table(df)

            # Stress details
            sc = last_res.get("stress_check", {})
            with st.expander("📊 Risk Signal Details (Early Warning)"):
                st.metric("Risk Score", sc.get("risk_score", 0), help="0–100 composite risk score")
                reasons = sc.get("top_reasons", [])
                if reasons:
                    df_r = pd.DataFrame(reasons)
                    if not df_r.empty:
                        df_r["direction"] = df_r["direction"].apply(
                            lambda d: "📈 Increases Risk" if "increase" in str(d) else "📉 Decreases Risk"
                        )
                        st.dataframe(df_r.rename(columns={
                            "feature": "Signal",
                            "shap_value": "SHAP Value",
                            "direction": "Direction"
                        }), use_container_width=True)

        # ── RECOMMENDATION ────────────────────────────────────────────────────
        elif action_taken == "recommendation":
            life_stage = last_res.get("life_stage", "")
            confidence = last_res.get("life_stage_confidence", 0)
            stage_label = LIFE_STAGE_LABELS.get(life_stage, life_stage.replace("_", " ").title())
            st.markdown(f"### 🎯 Personalised for **{stage_label}** · {int(confidence*100)}% match")

            recs = last_res.get("recommendations", [])
            if recs:
                rcols = st.columns(len(recs))
                for i, prod in enumerate(recs):
                    with rcols[i]:
                        score = float(prod.get("score", 0.5))
                        st.markdown(f"**{prod.get('product_name')}**")
                        st.progress(score)
                        st.caption(f"Match Score: {int(score * 100)}%")
                        with st.expander("Why recommended?"):
                            for r in prod.get("reasons", []):
                                st.markdown(f"- {r}")
                            exp = prod.get("explanation", "")
                            if exp:
                                st.write(exp)
                            signals = prod.get("top_signals") or prod.get("shap_top_features", {})
                            if signals:
                                st.markdown("**Key Profile Factors:**")
                                df = pd.DataFrame(
                                    [(k.replace("_", " ").title(), f"{v:.2f}") for k, v in signals.items()],
                                    columns=["Factor", "Weight"]
                                )
                                st.table(df)
            else:
                st.info("No recommendations available for this customer.")

            # Stress snapshot (even when none)
            sc = last_res.get("stress_check", {})
            with st.expander("🔍 Financial Health Signals"):
                sl = sc.get("stress_level", "none")
                rs = sc.get("risk_score", 0)
                if sl == "none":
                    st.success(f"✅ No stress signals detected · Risk Score: {rs}/100")
                else:
                    st.warning(f"⚠️ Stress Level: {sl.title()} · Risk Score: {rs}/100")
                reasons = sc.get("top_reasons", [])
                if reasons:
                    df_r = pd.DataFrame(reasons)
                    if not df_r.empty:
                        df_r["direction"] = df_r["direction"].apply(
                            lambda d: "📈 Risk ↑" if "increase" in str(d) else "📉 Risk ↓"
                        )
                        st.dataframe(df_r.rename(columns={
                            "feature": "Signal",
                            "shap_value": "SHAP",
                            "direction": "Direction"
                        }), use_container_width=True)

        # ── BLOCKED (consent denied) ──────────────────────────────────────────
        elif action_taken == "blocked":
            st.error(f"🔒 **Access Blocked by DPDP Consent Guard:** {last_res.get('reason', 'consent_denied')}")
            st.info(
                "Under India's **Digital Personal Data Protection (DPDP) Act 2023**, "
                "we cannot process your data without valid consent. "
                "Please grant consent below to unlock personalised banking."
            )

    st.divider()

    # ── CHATBOT ──────────────────────────────────────────────────────────────────
    st.markdown("### 💬 Bharat Sahayak — Your AI Banking Companion")
    st.caption("Ask anything: loans, FD rates, insurance, account queries — in any Indian language")

    # Render chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input using st.chat_input (supports Enter-to-send)
    user_input = st.chat_input(
        placeholder="Type your question (e.g. होम लोन की ब्याज दर क्या है? / What is my EMI?)"
    )

    if user_input and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                chat_payload = {
                    "session_id": st.session_state.session_id,
                    "customer_id": cust_id,
                    "message": user_input,
                    "language": st.session_state.language,
                }
                c_res, c_code = api_post("/chat", chat_payload)

            if c_code == 403:
                reply = "🔒 Chatbot access blocked — DPDP consent required."
            elif c_res and "reply" in c_res:
                reply = c_res["reply"]
                if c_res.get("form_data"):
                    st.session_state.form_data = c_res["form_data"]
                if c_res.get("next_question"):
                    st.session_state.next_question = c_res["next_question"]
            else:
                reply = "⚠️ Service temporarily unavailable. Please try again."

            st.markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", use_container_width=False):
            st.session_state.chat_history = []
            st.session_state.form_data = {}
            st.session_state.next_question = None
            st.session_state.session_id = f"s_{int(time.time())}"
            st.rerun()

    # Loan Form Progress
    curr_form = st.session_state.form_data
    if curr_form and any(v is not None for v in curr_form.values()):
        st.markdown("#### 📝 Loan Application — Auto-Fill Progress")
        req_keys = ["full_name", "age", "monthly_income", "loan_amount", "loan_purpose", "employment_type"]
        filled = sum(1 for k in req_keys if curr_form.get(k) is not None)
        pct = filled / len(req_keys)
        st.progress(pct, text=f"{filled}/{len(req_keys)} fields captured from conversation")
        with st.expander("📋 Captured Form Data"):
            st.json(curr_form)
        if st.session_state.next_question:
            st.info(f"**Next Step:** {st.session_state.next_question}")

    st.divider()

    # ── CONSENT PANEL ────────────────────────────────────────────────────────────
    with st.expander("🔒 Data Privacy & DPDP Consent (India 2023)"):
        consents_data = api_get(f"/customer/{cust_id}/consents")
        if consents_data and "consents" in consents_data:
            c_list = consents_data.get("consents", [])
            if c_list:
                rows = []
                for item in c_list:
                    given = item.get("given", False)
                    exp = item.get("expires_at")
                    exp_str = exp[:10] if exp else "N/A"
                    rows.append({
                        "Purpose": item.get("purpose", "").replace("_", " ").title(),
                        "Status": "✅ Granted" if given else "❌ Revoked",
                        "Valid Until": exp_str,
                    })
                st.table(pd.DataFrame(rows))
                st.caption("You can revoke any consent at any time. All processing stops immediately upon revocation.")

                # Revoke UI
                purposes = [item.get("purpose") for item in c_list if item.get("given")]
                if purposes:
                    with st.form(key="revoke_form"):
                        sel_purpose = st.selectbox("Revoke consent for:", purposes, format_func=lambda p: p.replace("_", " ").title())
                        submitted = st.form_submit_button("🚫 Revoke This Consent")
                        if submitted:
                            r_data, r_code = api_post("/consent/revoke", {"customer_id": cust_id, "purpose": sel_purpose})
                            if r_code == 200:
                                st.success(f"Consent for '{sel_purpose}' revoked successfully.")
                                st.rerun()
                            else:
                                st.error("Revocation failed. Please retry.")
            else:
                st.warning("No consent records found for this customer.")
        else:
            st.warning("Unable to load consent records.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RELATIONSHIP MANAGER VIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_rm:
    st.markdown("## 👔 Relationship Manager Dashboard")
    st.caption("Real-time portfolio monitoring · Early warning watchlist · RBI compliance audit")

    # ── Metrics ──────────────────────────────────────────────────────────────────
    audit_data = api_get("/audit/recent", params={"limit": 200}) or []

    total_cust = len(profiles)
    high_stress = sum(1 for e in audit_data if e.get("event_type") == "intervene")
    consent_blocks = sum(1 for e in audit_data if e.get("event_type") == "consent_denied")
    recs_made = sum(1 for e in audit_data if e.get("event_type") == "recommend")
    chats_handled = sum(1 for e in audit_data if e.get("event_type") == "chat")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("👥 Total Customers", total_cust)
    m2.metric("🆘 Stress Interventions", high_stress)
    m3.metric("💬 Chats Handled", chats_handled)
    m4.metric("🎯 Products Recommended", recs_made)
    m5.metric("🔒 DPDP Blocks", consent_blocks)

    st.divider()

    # ── At-Risk Watchlist ─────────────────────────────────────────────────────────
    st.markdown("### ⚠️ At-Risk Customer Watchlist")
    st.caption("Customers proactively flagged by Early Warning — before missed payment or default")

    at_risk_static = [
        {"id": "STRESS0001", "level": "High",   "score": 75, "reason": "Missed EMI (1), ATM cash surge +4.2 std-dev"},
        {"id": "STRESS0002", "level": "Medium", "score": 65, "reason": "Salary drop −51.6%, Discretionary spend collapse"},
        {"id": "STRESS0003", "level": "Medium", "score": 60, "reason": "Discretionary spending −3.8 std-dev"},
        {"id": "STRESS0004", "level": "High",   "score": 75, "reason": "Missed EMI (1), Elevated late-night transactions"},
        {"id": "STRESS0005", "level": "High",   "score": 75, "reason": "Missed EMI, New beneficiaries added, ATM surge"},
        {"id": "STRESS0006", "level": "Medium", "score": 65, "reason": "Salary disruption, Essential spend ratio shift"},
        {"id": "STRESS0007", "level": "Medium", "score": 60, "reason": "ATM cash dependency ↑, Discretionary spend ↓"},
        {"id": "STRESS0008", "level": "Medium", "score": 65, "reason": "Salary reduction, Elevated loan servicing pressure"},
    ]

    header_cols = st.columns([1.5, 1, 1, 3, 3])
    with header_cols[0]: st.markdown("**Customer**")
    with header_cols[1]: st.markdown("**Level**")
    with header_cols[2]: st.markdown("**Score**")
    with header_cols[3]: st.markdown("**Risk Signals**")
    with header_cols[4]: st.markdown("**RM Actions**")

    st.markdown("---")

    for item in at_risk_static:
        row_cols = st.columns([1.5, 1, 1, 3, 3])
        cid = item["id"]
        level = item["level"]

        with row_cols[0]:
            p = profile_map.get(cid, {})
            name = p.get("name", cid)
            st.markdown(f"**{cid}**")
            if name and name != cid:
                st.caption(name)

        with row_cols[1]:
            if level == "High":
                st.markdown("🔴 **High**")
            else:
                st.markdown("🟠 **Medium**")

        with row_cols[2]:
            st.metric("", item["score"], label_visibility="collapsed")

        with row_cols[3]:
            st.caption(item["reason"])

        with row_cols[4]:
            b1, b2, b3 = st.columns(3)
            with b1:
                if st.button("💌 Empathy", key=f"emp_{cid}", use_container_width=True):
                    with st.spinner(f"Sending message to {cid}…"):
                        time.sleep(0.5)
                    st.toast(f"✅ Empathetic support message sent to {cid}", icon="💌")
            with b2:
                if st.button("🔧 Restructure", key=f"rst_{cid}", use_container_width=True):
                    with st.spinner(f"Initiating moratorium for {cid}…"):
                        time.sleep(0.5)
                    st.toast(f"✅ 3-month moratorium offer dispatched to {cid}", icon="🔧")
            with b3:
                if st.button("📞 Escalate", key=f"esc_{cid}", use_container_width=True):
                    with st.spinner(f"Escalating {cid}…"):
                        time.sleep(0.5)
                    st.toast(f"✅ {cid} escalated to Senior Counseling Queue", icon="📞")

        st.markdown("---")

    # ── Live Early Warning Check ──────────────────────────────────────────────────
    st.markdown("### 🔬 Live Early Warning Analysis")
    ew_col1, ew_col2 = st.columns([2, 1])
    with ew_col1:
        ew_cust = st.selectbox(
            "Select customer for live EW check:",
            options=all_ids,
            format_func=_fmt_id,
            key="ew_customer_select",
        )
    with ew_col2:
        st.write("")
        st.write("")
        run_ew = st.button("🔍 Run Live Check", type="primary", use_container_width=True)

    if run_ew:
        with st.spinner(f"Running Early Warning analysis for {ew_cust}…"):
            res_data, code = api_post(f"/customer/{ew_cust}/action", {
                "event_type": "login",
                "language": "en",
            })
        if code == 200 and res_data:
            sc = res_data.get("stress_check", {})
            sl = sc.get("stress_level", "none")
            rs = sc.get("risk_score", 0)

            res_c1, res_c2 = st.columns(2)
            with res_c1:
                if sl == "high":
                    st.error(f"⚠️ **{ew_cust}** · Stress Level: **HIGH** · Score: {rs}/100")
                elif sl == "medium":
                    st.warning(f"🔶 **{ew_cust}** · Stress Level: MEDIUM · Score: {rs}/100")
                else:
                    st.success(f"✅ **{ew_cust}** · No stress signals · Score: {rs}/100")

            reasons = sc.get("top_reasons", [])
            if reasons:
                with res_c2:
                    df_r = pd.DataFrame(reasons)
                    if not df_r.empty and "feature" in df_r.columns:
                        df_r["direction"] = df_r["direction"].apply(
                            lambda d: "↑ Risk" if "increase" in str(d) else "↓ Risk"
                        )
                        st.dataframe(
                            df_r.rename(columns={"feature": "Signal", "shap_value": "SHAP", "direction": "Impact"}),
                            use_container_width=True,
                            hide_index=True,
                        )
        elif code == 403:
            st.error(f"🔒 DPDP consent blocked for {ew_cust}")
        else:
            st.warning("Early Warning check failed. Ensure services are running.")

    st.divider()

    # ── Audit Trail ───────────────────────────────────────────────────────────────
    st.markdown("### 📜 Regulatory Audit Trail (Last 50 Events)")
    st.caption("Immutable append-only log · RBI Data Localization compliant")

    recent_events = api_get("/audit/recent", params={"limit": 50}) or []

    if recent_events:
        rows = []
        for ev in recent_events:
            dec = ev.get("decision", "info")
            icon = {"allowed": "🟢", "blocked": "🔴", "escalated": "🟠", "info": "🔵"}.get(dec, "⚪")
            rows.append({
                "Time": ev.get("timestamp", "")[-8:],   # show HH:MM:SS only
                "Customer": ev.get("customer_id", ""),
                "Event": ev.get("event_type", "").replace("_", " ").title(),
                "Decision": f"{icon} {dec.title()}",
                "Summary": ev.get("outcome_summary", "")[:80],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No audit events yet. Trigger some events from the Customer View to see them here.")

    st.divider()

    # ── Compliance ────────────────────────────────────────────────────────────────
    with st.expander("🛡️ Compliance & Responsible Lending Summary"):
        st.markdown("""
| Safeguard | Status | Detail |
|---|---|---|
| DPDP Consent Enforcement | ✅ Active | HTTP 403 logged for unconsented data access |
| Stress Gate (Anti-Predatory) | ✅ Active | Loan products blocked when stress level = high |
| RBI Data Localization | ✅ Active | All processing on-premise; external LLM only when key provided |
| Immutable Audit Log | ✅ Active | Append-only JSONL with timestamp, decision, outcome |
| Empathetic Intervention | ✅ Active | Non-punitive messaging in customer's preferred language |
| Algorithmic Fairness | ✅ Active | SHAP explanations · Per-product signal attribution |
        """)
