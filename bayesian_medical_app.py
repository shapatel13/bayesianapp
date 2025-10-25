# app_priori_auto.py
# Minimal, auto-driven PRIORI (Hybrid Clinical Mode)
# - Keeps your model & API key EXACTLY as-is
# - No user toggles; the LLM supplies LRs, costs, EVI & Utility
# - Parses a JSON tail to show structured results automatically

import re
import json
from typing import List, Dict

import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration & Setup (kept as-is for your testing) ---
try:
    API_KEY = 'AIzaSyDmcPbEDAEojTomYs7vLKu107fOa7c6500'
except (FileNotFoundError, KeyError):
    st.error("API Key not found. Please create a .streamlit/secrets.toml file with your API_KEY.")
    st.stop()

# ---------- Helper: extract last fenced JSON block from model output ----------
JSON_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE
)

def extract_json_tail(text: str) -> Dict:
    """
    Extract the last fenced code block that parses as JSON.
    Return {} if not found or parsing fails.
    """
    matches = JSON_BLOCK_PATTERN.findall(text or "")
    for raw in reversed(matches):
        try:
            return json.loads(raw)
        except Exception:
            continue
    return {}

# ---------- Agent Instructions (LLM supplies LRs, costs, EVI, Utility) ----------
HYBRID_INSTRUCTIONS = """
You are PRIORI — a Bayesian clinical rounding assistant for inpatient/ICU care.
Tone: professional, collegial, concise. Show math lightly unless asked. Use internal medical knowledge
to supply missing parameters (likelihood ratios, costs, typical US hospital ranges) and state assumptions.

Goals:
- Reduce uncertainty (Bayes), avoid low-value tests (EVI), choose best next step (Utility).
- If precise LR/cost is unknown, provide a reasonable range and mark as "approx" with a short basis.
- Keep one clear recommendation; provide alternatives concisely.
- Never request the user to toggle parameters; you must infer reasonable values.

Required sections (Markdown):

**Executive Summary:** One bold sentence with the key clinical conclusion (posterior + next action).

### 1) Pre-Test Probability
- Initial estimate (%), with a brief rationale (validated rule or gestalt).
- Key priors/assumptions (one line).

### 2) Evidence Update (LR Table)
| Finding/Test | Result | LR (or LR+/LR−) | Source/Note |
|---|---:|---:|---|
| ... | ... | ... | ... |

(If LR is approximate, show a reasonable range and mark "approx".)

### 3) Post-Test Probability (quiet math)
- Pre-odds × LR product → Post-odds → Posterior (%). Keep concise.

### 4) Decision & Rationale (Bayes → EVI → Utility)
- **Recommendation:** the single best next action.
- Why now: link Bayes (posterior) → EVI (will testing change management?) → Utility (benefit−harm−cost).
- Include simple US cost estimates (mean $ and uncertainty) for the few key actions you considered.
- Safety overrides (if any).
- Devil’s advocate (one-line alternative).

### 5) JSON (machine-readable)
Return a fenced JSON block with:
{
  "posterior": 0.xx,
  "evi_table": [
    {"test":"CT Chest","p_change":0.xx,"value_if_change":0.xx,"evi":0.xx},
    {"test":"Procalcitonin","p_change":0.xx,"value_if_change":0.xx,"evi":0.xx}
  ],
  "costs": [
    {"item":"CT Chest","mean_usd":600,"note":"approx US inpatient"},
    {"item":"Daily Labs","mean_usd":300,"note":"approx bundle"}
  ],
  "utility_rank": [
    {"action":"Defer CT; re-eval in 12–24h","benefit":0.xx,"harm":0.xx,"cost_usd":0,"utility":0.xx},
    {"action":"Continue broad Abx 24h then narrow","benefit":0.xx,"harm":0.xx,"cost_usd":100,"utility":0.xx},
    {"action":"Order CT Chest","benefit":0.xx,"harm":0.xx,"cost_usd":600,"utility":0.xx}
  ]
}
- Ensure keys and numeric types are valid JSON. Do not include comments in the JSON.
"""

clinical_reasoner = Agent(
    name="PRIORI",
    model=Gemini(id="gemini-2.5-flash", api_key=API_KEY),  # <-- unchanged
    markdown=True,
    description="PRIORI: Bayesian + EVI + Utility bedside reasoning partner (hybrid clinical mode, auto-parameters).",
    instructions=[HYBRID_INSTRUCTIONS],
)

# ---------- UI ----------
st.set_page_config(page_title="PRIORI — Bayesian/EVI/Utility (Auto)", page_icon="🩺", layout="wide")
st.title("🩺 PRIORI — Bayesian • EVI • Utility (Auto-Parameters)")
st.caption("Zero knobs. The model supplies LRs, costs, EVI & Utility with explicit assumptions.")

with st.sidebar:
    st.header("Session")
    if st.button("🔄 New Case"):
        st.session_state.clear()
        st.rerun()
    st.markdown("**Examples**")
    st.code("68M CAP on room air (SpO₂ 93%), low Wells, mild dyspnea. Should I order CT PE?")
    st.code("ED anemia + melena, Hgb 8.1 but stable. What next? Imaging vs endoscopy timing?")
    st.code("ICU oliguric AKI day 1 post-sepsis. FST vs early RRT?")

# Display history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input
if prompt := st.chat_input("Enter your clinical query..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call agent
    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            try:
                run = clinical_reasoner.run(prompt)
                content = run.content

                # Show narrative
                st.markdown(content)

                # Try to parse the JSON tail and render structured tables
                data = extract_json_tail(content)

                if data:
                    st.divider()
                    st.subheader("Structured Summary")
                    col1, col2 = st.columns(2)

                    with col1:
                        if "posterior" in data:
                            st.metric("Posterior (from agent)", f"{float(data['posterior'])*100:.1f}%")

                        if "evi_table" in data and isinstance(data["evi_table"], list) and data["evi_table"]:
                            st.markdown("**EVI (Expected Value of Information)**")
                            st.table({
                                "Test": [x.get("test","") for x in data["evi_table"]],
                                "ΔP": [round(float(x.get("p_change",0.0)), 3) for x in data["evi_table"]],
                                "Value(if change)": [round(float(x.get("value_if_change",0.0)), 3) for x in data["evi_table"]],
                                "EVI": [round(float(x.get("evi",0.0)), 3) for x in data["evi_table"]],
                            })

                    with col2:
                        if "costs" in data and isinstance(data["costs"], list) and data["costs"]:
                            st.markdown("**Cost Snapshot (US, approx)**")
                            st.table({
                                "Item": [x.get("item","") for x in data["costs"]],
                                "Mean $": [x.get("mean_usd","") for x in data["costs"]],
                                "Note": [x.get("note","") for x in data["costs"]],
                            })

                    if "utility_rank" in data and isinstance(data["utility_rank"], list) and data["utility_rank"]:
                        st.markdown("**Utility Ranking (Higher = better value)**")
                        st.table({
                            "Action": [x.get("action","") for x in data["utility_rank"]],
                            "Benefit": [round(float(x.get("benefit",0.0)),3) for x in data["utility_rank"]],
                            "Harm": [round(float(x.get("harm",0.0)),3) for x in data["utility_rank"]],
                            "Cost ($)": [x.get("cost_usd","") for x in data["utility_rank"]],
                            "Utility": [round(float(x.get("utility",0.0)),3) for x in data["utility_rank"]],
                        })

                # Save assistant message
                st.session_state["messages"].append({"role": "assistant", "content": content})

            except Exception as e:
                err = f"An error occurred: {str(e)}"
                st.error(err)
                st.session_state["messages"].append({"role": "assistant", "content": err})
