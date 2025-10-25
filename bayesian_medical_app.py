# app_priori_hybrid.py
import math
from dataclasses import dataclass
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

# -----------------------------
# Bayes / EVI / Utility helpers
# -----------------------------
def _clip01(x: float) -> float:
    return max(1e-6, min(1 - 1e-6, x))

def prob_to_odds(p: float) -> float:
    p = _clip01(p)
    return p / (1 - p)

def odds_to_prob(o: float) -> float:
    return o / (1 + o)

def bayes_update(pretest_p: float, lrs: List[float]) -> float:
    """Multiply LRs (assumes conditional independence for MVP) and update probability."""
    pre_odds = prob_to_odds(pretest_p)
    lr_prod = 1.0
    for lr in lrs:
        try:
            lr_val = float(lr)
        except:
            lr_val = 1.0
        lr_prod *= max(lr_val, 1e-6)
    post_odds = pre_odds * lr_prod
    return odds_to_prob(post_odds)

def evi_for_test(posterior: float, pretest: float, value_if_change: float = 0.5) -> float:
    """
    MVP proxy for Expected Value of Information:
    EVI = |ΔP| * value_if_change
    where |ΔP| is the absolute probability shift after the test and
    value_if_change is a unitless 0–1 knob for how valuable a decision change would be.
    """
    dp = abs(_clip01(posterior) - _clip01(pretest))
    return max(0.0, dp) * max(0.0, min(1.0, value_if_change))

def utility(benefit: float, harm: float, cost_usd: float, harm_weight: float = 0.2) -> float:
    """
    Utility = Benefit − (harm_weight × Harm) − (Cost / 100)
    Units are relative 'utility points'. Keep simple and comparable across options.
    """
    return float(benefit) - (float(harm_weight) * float(harm)) - (float(cost_usd) / 100.0)

# -----------------------------
# Simple cost priors (editable in sidebar)
# -----------------------------
@dataclass
class CostItem:
    name: str
    mean_usd: float
    risk_weight: float  # small proxy for harm
    ops_friction: float # workflow penalty (0–0.3)

DEFAULT_COSTS: Dict[str, CostItem] = {
    "ct_chest": CostItem("CT Chest", 600, 0.05, 0.25),
    "daily_labs": CostItem("Broad Daily Labs", 300, 0.02, 0.05),
    "broad_abx_day": CostItem("Broad Antibiotics (per day)", 100, 0.03, 0.10),
    "narrow_abx_day": CostItem("Narrow Antibiotics (per day)", 30, 0.01, 0.05),
    "balanced_crystalloid_500": CostItem("Balanced Crystalloid 500 mL", 8, 0.01, 0.02),
}

# -----------------------------
# Agent Definition (Hybrid Clinical Mode)
# -----------------------------
HYBRID_INSTRUCTIONS = """
You are PRIORI — a Bayesian clinical rounding assistant for inpatient/ICU care.
Keep a professional, collegial tone; be concise and structured. Show math lightly (quiet background),
but expose it when asked. Combine Bayesian updating, EVI for tests, and Utility ranking for actions.

ALWAYS respond using this Markdown scaffold (short, clinical, readable):

**Executive Summary:** One bold sentence with the key clinical conclusion (probability + next step).

### 1) Pre-Test Probability
- Initial estimate (%), with a brief rationale (validated rule or gestalt).
- Key priors/assumptions (one line).

### 2) Evidence Update (LR Table)
| Finding/Test | Result | LR (or LR+/LR−) | Note |
|---|---:|---:|---|
| ... | ... | ... | ... |

### 3) Post-Test Probability (quiet math)
- Pre-odds × LR product → Post-odds → Posterior (%).

### 4) Decision & Rationale (Bayes → EVI → Utility)
- **Recommendation:** the single best next action.
- Why now: link Bayes (posterior) → EVI (will testing change management?) → Utility (benefit−harm−cost).
- Safety overrides (if any).
- Devil’s advocate (one-line alternative).

### 5) JSON (machine-readable)
Return a fenced JSON block with:
{
  "posterior": 0.xx,
  "evi_table": [{"test":"...", "evi": 0.xx}],
  "utility_rank": [{"action":"...", "utility": 0.xx}]
}
"""

clinical_reasoner = Agent(
    name="PRIORI",
    model=Gemini(id="gemini-2.5-flash", api_key=API_KEY),  # <-- kept exactly as you had it
    markdown=True,
    description="PRIORI: Bayesian + EVI + Utility bedside reasoning partner (hybrid clinical mode).",
    instructions=[HYBRID_INSTRUCTIONS],
)

# -----------------------------
# App Logic / UI
# -----------------------------
def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

def format_message_for_agent(role: str, content: str) -> dict:
    return {"role": "user" if role == "user" else "assistant", "content": content}

st.set_page_config(page_title="PRIORI — Bayesian/EVI/Utility (Hybrid)", page_icon="🩺", layout="wide")
initialize_chat_history()
st.title("🩺 PRIORI — Bayesian • EVI • Utility (Hybrid Clinical Mode)")
st.markdown("> **Bayes narrows uncertainty. EVI prevents waste. Utility chooses the best next step.**")

# Sidebar: controls for simple costs/weights and a mini Bayes/EVI sandbox
with st.sidebar:
    st.header("Session")
    if st.button("🔄 New Case"):
        st.session_state.messages, st.session_state.conversation_history = [], []
        st.rerun()

    st.header("Cost / Risk Tuners")
    ct_cost = st.slider("CT Chest ($)", 200, 1200, int(DEFAULT_COSTS["ct_chest"].mean_usd), 50)
    labs_cost = st.slider("Broad Daily Labs ($)", 50, 600, int(DEFAULT_COSTS["daily_labs"].mean_usd), 25)
    abx_cost = st.slider("Broad Abx / day ($)", 40, 200, int(DEFAULT_COSTS["broad_abx_day"].mean_usd), 10)
    harm_weight = st.slider("Global Harm Weight (0–1)", 0.0, 1.0, 0.2, 0.05)

    # Update runtime costs
    DEFAULT_COSTS["ct_chest"].mean_usd = float(ct_cost)
    DEFAULT_COSTS["daily_labs"].mean_usd = float(labs_cost)
    DEFAULT_COSTS["broad_abx_day"].mean_usd = float(abx_cost)

    st.header("Quick Bayes/EVI Sandbox")
    pretest_pct = st.slider("Pre-test probability (%)", 0.0, 100.0, 10.0, 1.0)
    lrs_csv = st.text_input("LRs (comma-separated)", "0.3")
    try:
        lr_list = [float(x.strip()) for x in lrs_csv.split(",") if x.strip()]
    except:
        lr_list = [1.0]
    posterior_demo = bayes_update(pretest_pct/100.0, lr_list)
    st.write(f"Posterior (demo): **{posterior_demo*100:.1f}%**")
    value_if_change = st.slider("Value(if decision changes) 0–1", 0.0, 1.0, 0.5, 0.05)
    st.write(f"EVI proxy: **{evi_for_test(posterior_demo, pretest_pct/100.0, value_if_change):.2f}**")

    st.header("Examples")
    st.markdown("""
    - *ED CAP, low Wells, mild dyspnea — Should I order CT PE?*  
    - *ICU septic shock after 1L; PLR−, VExUS 2 — Fluids or pressor?*
    """)

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your clinical query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append(format_message_for_agent("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Reasoning..."):
            try:
                # Run the agent
                response = clinical_reasoner.run(prompt, messages=st.session_state.conversation_history)
                response_content = response.content

                # ---- Optional: add a small local utility snapshot (illustrative only) ----
                ct = DEFAULT_COSTS["ct_chest"]
                labs = DEFAULT_COSTS["daily_labs"]
                abx = DEFAULT_COSTS["broad_abx_day"]

                # MVP illustrative benefits/harms (engineers should tune per scenario)
                actions = [
                    ("Defer Imaging; Re-evaluate in 12–24h", utility(benefit=0.35, harm=0.02, cost_usd=0, harm_weight=harm_weight)),
                    ("Continue Broad Abx 24h then Reassess", utility(benefit=0.40, harm=abx.risk_weight, cost_usd=abx.mean_usd, harm_weight=harm_weight)),
                    ("Order Broad Daily Labs", utility(benefit=0.10, harm=labs.risk_weight, cost_usd=labs.mean_usd, harm_weight=harm_weight)),
                    ("Order CT Chest", utility(benefit=0.10, harm=ct.risk_weight, cost_usd=ct.mean_usd, harm_weight=harm_weight)),
                ]
                actions_sorted = sorted(actions, key=lambda x: x[1], reverse=True)

                # Display the agent result
                st.markdown(response_content)

                # Display compact local utility table for transparency
                st.markdown("#### Local Utility Snapshot (MVP — adjustable weights)")
                st.table(
                    {
                        "Action": [a for a, _ in actions_sorted],
                        "Utility (relative)": [round(u, 2) for _, u in actions_sorted],
                    }
                )

                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.session_state.conversation_history.append(
                    format_message_for_agent("assistant", response_content)
                )
            except Exception as e:
                response_content = f"An error occurred: {str(e)}"
                st.error(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.session_state.conversation_history.append(
                    format_message_for_agent("assistant", response_content)
                )

if __name__ == "__main__":
    # Streamlit runs this file directly
    pass
