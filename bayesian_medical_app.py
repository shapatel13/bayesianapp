# app_priori_auto_router.py
# PRIORI — Auto-parameters + Auto-Policy Routing + Digestible Output + Auto Prompts

import re
import json
from typing import Dict

import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration & Setup (unchanged for your testing) ---
try:
    API_KEY = 'AIzaSyDmcPbEDAEojTomYs7vLKu107fOa7c6500'
except (FileNotFoundError, KeyError):
    st.error("API Key not found. Please create a .streamlit/secrets.toml file with your API_KEY.")
    st.stop()

# ---------- JSON tail extractor ----------
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

def extract_json_tail(text: str) -> Dict:
    matches = JSON_BLOCK_PATTERN.findall(text or "")
    for raw in reversed(matches):
        try:
            return json.loads(raw)
        except Exception:
            continue
    return {}

# ---------- Lightweight policy router ----------
def detect_policy(user_text: str) -> str:
    t = (user_text or "").lower()
    if any(k in t for k in ["pulmonary embol", " pe ", " pe.", "wells", "perc", "d-dimer", "d dimer"]):
        return "PE"
    if any(k in t for k in ["pneumonia", "cap", "community-acquired", "infiltrate", "procalcitonin"]):
        return "CAP"
    if any(k in t for k in ["septic shock", "sepsis", "norepinephrine", "vasopressin", "vexus", "plr", "capillary refill"]):
        return "SEPSIS"
    if any(k in t for k in ["aki", "oligur", "furosemide stress", "fst", "dialysis", "crrt"]):
        return "AKI"
    if any(k in t for k in ["gi bleed", "hematemesis", "melena", "varice", "egd", "octreotide"]):
        return "UGIB"
    if any(k in t for k in ["chest pain", "heart score", "troponin", "ecg", "acs"]):
        return "CHESTPAIN"
    return "GEN"

# Scenario-specific priors/thresholds cue (concise, for the LLM to reason with)
POLICY_CONTEXT = {
    "PE": """Scenario: Pulmonary Embolism rule-in/out.
- Use Wells/PERC intuition; D-dimer useful in low/mod risk; CTA only when posterior remains above threshold.
- Prefer avoiding CTA if EVI low. Costs (US approx): CTA $500–$800; D-dimer $15–$40.
- Safety: anticoagulate only if posterior high / imaging positive or high clinical suspicion with instability.""",
    "CAP": """Scenario: Community-acquired pneumonia (floor).
- Favor CXR + procalcitonin, avoid CT unless atypical course.
- Narrow/shorten antibiotics at 48–72h if improving (5 days typical).
- Costs (US approx): CT chest $500–$700; daily broad labs $150–$300; broad abx/day $70–$150; narrow/day ~$20–$40.""",
    "SEPSIS": """Scenario: Septic shock (ICU).
- Early norepinephrine if MAP<65 after ≤1–2L; PLR/VExUS to guide fluids.
- Balanced crystalloids preferred; add vasopressin when NE ~0.25–0.5 µg/kg/min; steroids if pressor-dependent.
- Costs approx: 500 mL crystalloids ~$8; day-1 NE ~$25; CT chest/abd ~$500–$700. Avoid low-EVI imaging early.""",
    "AKI": """Scenario: Oliguric AKI after sepsis.
- Rule out obstruction (renal/bladder US). If VExUS≥2, avoid fluids; FST (1–1.5 mg/kg) stratifies RRT risk.
- Start CRRT only for AEIOU or progression.
- Costs approx: FST ~$2; US bedside ~$35; CRRT/day ~$1600; daily labs ~$150–$300.""",
    "UGIB": """Scenario: Upper GI bleed (ICU).
- PPI + resuscitation first. Add octreotide + ceftriaxone only if variceal probability high.
- Urgent EGD ≤12h once stable; avoid CT unless perforation/unclear source post-EGD.
- Costs approx: PPI infusion ~$35; octreotide day ~$85; urgent EGD ~$1100; CT abd ~$500–$700.""",
    "CHESTPAIN": """Scenario: Chest pain (ED/inpatient ACS evaluation).
- Use HEART or similar risk. If low risk and ECG/troponins negative (serial), discharge with follow-up.
- Intermediate risk → serial hs-troponins, observation; avoid stress/CTCA if EVI low.
- Costs (US approx): single hs-trop $20–$40; CTCA $400–$900; stress imaging $600–$1200.""",
    "GEN": """Scenario: General inpatient reasoning.
- Reduce uncertainty (Bayes), avoid low-value tests (EVI), choose highest-utility action (Benefit−Harm−Cost).
- Use reasonable US costs and mark approximations; keep one clear recommendation."""
}

# ---------- Agent Instructions (digestible output, math hidden) ----------
HYBRID_INSTRUCTIONS = """
You are PRIORI — a Bayesian clinical rounding assistant for inpatient/ICU care.
Tone: professional, collegial, concise. Use internal medical knowledge to supply LRs and US cost ranges; mark approximations.
Do all math silently; present an easy-to-digest summary. Keep one clear recommendation.

Required sections (Markdown, concise):

**Executive Summary:** One bold sentence with the key clinical conclusion (posterior + next action).

### Clinical TL;DR (for bedside)
- Posterior: X%
- Do now: single best next step (plain language)
- Why: 2–3 bullets (Bayes → EVI → Utility)

*(Math below is optional for readers)*

<details>
<summary>Details (math & rationale)</summary>

### 1) Pre-Test Probability
- Initial estimate (%), with brief rationale (rule or gestalt). Assumptions (one line).

### 2) Evidence Update (LR Table)
| Finding/Test | Result | LR (or LR+/LR−) | Source/Note |
|---|---:|---:|---|
| ... | ... | ... | ... |

### 3) Post-Test Probability (quiet math)
- Pre-odds × LR product → Post-odds → Posterior (%).

### 4) Decision & Rationale (Bayes → EVI → Utility)
- **Recommendation:** single best action.
- Why now: link Bayes (posterior) → EVI (will testing change management?) → Utility (benefit−harm−cost).
- Include simple US cost estimates (mean $ and uncertainty) for key actions.
- Safety overrides (if any).
- Devil’s advocate (one-line alternative).

</details>

### 5) JSON (machine-readable)
Return a fenced JSON block with:
{
  "posterior": 0.xx,
  "evi_table": [
    {"test":"...","p_change":0.xx,"value_if_change":0.XX,"evi":0.xx}
  ],
  "costs": [
    {"item":"...","mean_usd":123,"note":"approx US"}
  ],
  "utility_rank": [
    {"action":"...","benefit":0.xx,"harm":0.xx,"cost_usd":123,"utility":0.xx}
  ],
  "best_action": "plain-language single action"
}
Ensure valid JSON (no comments) in the final fenced block.
"""

clinical_reasoner = Agent(
    name="PRIORI",
    model=Gemini(id="gemini-2.5-flash", api_key=API_KEY),  # <-- unchanged
    markdown=True,
    description="PRIORI: Bayesian + EVI + Utility bedside reasoning partner (auto-parameters, policy-routed).",
    instructions=[HYBRID_INSTRUCTIONS],
)

# ---------- Prompt runner (shared by chat input and auto buttons) ----------
def run_case(user_text: str):
    # Route + preface
    policy = detect_policy(user_text)
    preface = f"Policy: {policy}\n\n{POLICY_CONTEXT.get(policy,'')}\n\nUser question:\n"
    routed_prompt = preface + user_text

    # Call agent
    run = clinical_reasoner.run(routed_prompt)
    content = run.content

    # Render narrative (digestible; details hidden)
    st.markdown(content, unsafe_allow_html=True)

    # Parse JSON tail → TL;DR + Tables
    data = extract_json_tail(content)
    if data:
        best = data.get("best_action")
        posterior = data.get("posterior")
        if posterior is not None or best:
            st.divider()
            st.subheader("Clinical TL;DR (auto)")
            cols = st.columns(3)
            if posterior is not None:
                cols[0].metric("Posterior", f"{float(posterior)*100:.1f}%")
            if best:
                cols[1].markdown(f"**Do now:** {best}")
            if data.get("utility_rank"):
                top = sorted(data["utility_rank"], key=lambda x: x.get("utility", 0.0), reverse=True)[0]
                cols[2].markdown(f"**Why:** {top.get('action','Best action')} has the highest utility.")

        st.divider()
        st.subheader("Structured Summary")
        if data.get("evi_table"):
            st.markdown("**EVI (Expected Value of Information)**")
            st.table({
                "Test": [x.get("test","") for x in data["evi_table"]],
                "ΔP": [round(float(x.get("p_change",0.0)), 3) for x in data["evi_table"]],
                "Value(if change)": [round(float(x.get("value_if_change",0.0)), 3) for x in data["evi_table"]],
                "EVI": [round(float(x.get("evi",0.0)), 3) for x in data["evi_table"]],
            })
        if data.get("costs"):
            st.markdown("**Cost Snapshot (US, approx)**")
            st.table({
                "Item": [x.get("item","") for x in data["costs"]],
                "Mean $": [x.get("mean_usd","") for x in data["costs"]],
                "Note": [x.get("note","") for x in data["costs"]],
            })
        if data.get("utility_rank"):
            st.markdown("**Utility Ranking (Higher = better value)**")
            st.table({
                "Action": [x.get("action","") for x in data["utility_rank"]],
                "Benefit": [round(float(x.get("benefit",0.0)),3) for x in data["utility_rank"]],
                "Harm": [round(float(x.get("harm",0.0)),3) for x in data["utility_rank"]],
                "Cost ($)": [x.get("cost_usd","") for x in data["utility_rank"]],
                "Utility": [round(float(x.get("utility",0.0)),3) for x in data["utility_rank"]],
            })

    # Save to history
    st.session_state["messages"].append({"role": "assistant", "content": content})

# ---------- UI ----------
st.set_page_config(page_title="PRIORI — Bayesian/EVI/Utility (Auto + Routed)", page_icon="🩺", layout="wide")
st.title("🩺 PRIORI — Bayesian • EVI • Utility")
st.caption("Math in the background. Clinician-ready output up front.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Left: chat history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Sidebar: Auto Prompts (one-click)
with st.sidebar:
    st.header("Auto Prompts (One-click cases)")
    if st.button("PE — Low/Intermediate Risk (D-dimer vs CTA)"):
        user_q = (
            "55M with pleuritic chest pain, HR 102, O2 95% RA. "
            "Wells low-intermediate; no unilateral leg swelling. Should I get a D-dimer first or go straight to CTA?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        with st.chat_message("user"): st.markdown(user_q)
        run_case(user_q)

    if st.button("Septic Shock — Fluids vs Pressors (PLR/VExUS)"):
        user_q = (
            "ICU patient with septic shock after 1.5L crystalloid; MAP 62, on 0.06 NE. "
            "PLR shows no stroke volume increase; VExUS = 2. Fluids or pressor now?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        with st.chat_message("user"): st.markdown(user_q)
        run_case(user_q)

    if st.button("Chest Pain — HEART-style risk, rule-in/out ACS"):
        user_q = (
            "58F with chest pressure, risk factors HTN/HLD, non-ischemic ECG, initial hs-troponin negative. "
            "HEART ~4. What’s the most value-based next step—serial troponins/obs vs stress vs CTCA?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        with st.chat_message("user"): st.markdown(user_q)
        run_case(user_q)

    if st.button("UGIB — Stable, timing of EGD"):
        user_q = (
            "62M with melena, Hgb 8.9, MAP 74 but stable after 1U PRBC. No cirrhosis history. "
            "Should I push for urgent EGD now vs early morning? Imaging needed?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        with st.chat_message("user"): st.markdown(user_q)
        run_case(user_q)

    if st.button("AKI Oliguria — FST vs Early RRT"):
        user_q = (
            "ICU patient oliguric post-sepsis; Cr 2.7 (baseline 1.0), K 5.3, HCO3 18. "
            "POCUS: VExUS 2, no hydronephrosis; bladder 80 mL. Should I do FST, and when to start CRRT?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        with st.chat_message("user"): st.markdown(user_q)
        run_case(user_q)

    if st.button("CAP — Low Risk (avoid low-EVI testing)"):
        user_q = (
            "68M with CAP on CXR, SpO2 93% RA, mild COPD, no chest pain. "
            "Should I order CT chest or stick with CXR + procalcitonin and narrow antibiotics?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        with st.chat_message("user"): st.markdown(user_q)
        run_case(user_q)

# Free-text chat input
if user_free := st.chat_input("Enter your clinical query..."):
    st.session_state["messages"].append({"role": "user", "content": user_free})
    with st.chat_message("user"):
        st.markdown(user_free)
    run_case(user_free)
