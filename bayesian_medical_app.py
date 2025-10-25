# app_priori_improved.py
# PRIORI — Bayesian + EVI + Utility with Relative Cost Tiers & Improved UX

import re
import json
from typing import Dict

import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini

# --- Configuration ---
API_KEY = 'AIzaSyDmcPbEDAEojTomYs7vLKu107fOa7c6500'

# ---------- Cost Tiers (Simplified for MVP) ----------
COST_TIERS = {
    # PE workup
    "D-dimer": {"tier": "$", "relative_cost": 1, "note": "Low-cost screening"},
    "CTA chest": {"tier": "$$$", "relative_cost": 40, "note": "40× more than D-dimer"},
    "VQ scan": {"tier": "$$", "relative_cost": 20, "note": "Mid-cost alternative"},
    
    # Chest pain workup
    "hs-troponin (single)": {"tier": "$", "relative_cost": 1, "note": "Low-cost biomarker"},
    "Serial troponins": {"tier": "$", "relative_cost": 2, "note": "2 draws + monitoring"},
    "CTCA": {"tier": "$$$", "relative_cost": 30, "note": "30× more than troponin"},
    "Stress echo": {"tier": "$$$", "relative_cost": 35, "note": "High-cost functional test"},
    "Observation stay": {"tier": "$$", "relative_cost": 15, "note": "Mid-cost disposition"},
    
    # Sepsis management
    "500mL crystalloid": {"tier": "$", "relative_cost": 0.5, "note": "Minimal cost"},
    "Norepinephrine (day 1)": {"tier": "$", "relative_cost": 1.5, "note": "Low-cost pressor"},
    "Vasopressin (day 1)": {"tier": "$", "relative_cost": 2, "note": "Low-cost add-on"},
    "CT chest/abdomen": {"tier": "$$$", "relative_cost": 35, "note": "High-cost imaging"},
    "Procalcitonin": {"tier": "$", "relative_cost": 2, "note": "Stewardship biomarker"},
    
    # AKI workup
    "Renal ultrasound (bedside)": {"tier": "$", "relative_cost": 2, "note": "POCUS, low-cost"},
    "Furosemide stress test": {"tier": "$", "relative_cost": 0.2, "note": "Drug + monitoring"},
    "CRRT (per day)": {"tier": "$$$$$", "relative_cost": 100, "note": "Very high ongoing cost"},
    "Daily CMP": {"tier": "$", "relative_cost": 1, "note": "Standard monitoring"},
    
    # UGIB management
    "PPI infusion (day)": {"tier": "$", "relative_cost": 2, "note": "Standard therapy"},
    "Octreotide (day)": {"tier": "$$", "relative_cost": 5, "note": "Variceal-specific"},
    "Urgent EGD": {"tier": "$$$", "relative_cost": 70, "note": "Procedure + anesthesia"},
    "CT abdomen": {"tier": "$$$", "relative_cost": 35, "note": "High-cost imaging"},
    "PRBC (1 unit)": {"tier": "$$", "relative_cost": 10, "note": "Blood product"},
    
    # CAP management
    "Chest X-ray": {"tier": "$", "relative_cost": 1, "note": "Standard screening"},
    "CT chest": {"tier": "$$$", "relative_cost": 35, "note": "35× more than CXR"},
    "Broad-spectrum antibiotics (day)": {"tier": "$$", "relative_cost": 8, "note": "Piperacillin-tazobactam"},
    "Narrow-spectrum antibiotics (day)": {"tier": "$", "relative_cost": 2, "note": "Ceftriaxone, azithro"},
}

def get_cost_info(item_name: str) -> dict:
    """Return cost tier info for display."""
    return COST_TIERS.get(item_name, {
        "tier": "?",
        "relative_cost": "unknown",
        "note": "Not in reference"
    })

# ---------- JSON tail extractor ----------
JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

def extract_json_tail(text: str) -> Dict:
    """Extract last valid JSON block from markdown response."""
    matches = JSON_BLOCK_PATTERN.findall(text or "")
    for raw in reversed(matches):
        try:
            return json.loads(raw)
        except Exception:
            continue
    return {}

# ---------- Policy router ----------
def detect_policy(user_text: str) -> str:
    """Route query to appropriate clinical scenario context."""
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

# ---------- Policy context (updated with relative costs) ----------
POLICY_CONTEXT = {
    "PE": """Scenario: Pulmonary Embolism rule-in/out.
- Use Wells/PERC; D-dimer ($) first if low/moderate risk; CTA ($$$) only when posterior remains high.
- Cost context: D-dimer is baseline cost; CTA is 40× more expensive. Avoid CTA if EVI is low.
- Safety: anticoagulate only if posterior high / imaging positive or clinical instability.""",
    
    "CAP": """Scenario: Community-acquired pneumonia.
- Favor CXR ($) + procalcitonin ($); avoid CT chest ($$$, 35× more) unless atypical course.
- Narrow antibiotics ($$) preferred over broad ($$$) when appropriate; aim for 5-day course.
- Cost context: CT adds minimal EVI in uncomplicated CAP.""",
    
    "SEPSIS": """Scenario: Septic shock (ICU).
- Early norepinephrine ($) if MAP<65 after ≤1–2L crystalloids ($); use PLR/VExUS to guide fluids.
- Add vasopressin ($) when NE ~0.25–0.5 µg/kg/min; steroids if pressor-dependent.
- Cost context: Avoid low-EVI imaging ($$$) early; procalcitonin ($) helps stewardship.""",
    
    "AKI": """Scenario: Oliguric AKI post-sepsis.
- Rule out obstruction with bedside ultrasound ($). If VExUS≥2, avoid fluids.
- FST ($) stratifies RRT risk. Start CRRT ($$$$$, 500× more expensive) only for AEIOU or progression.
- Cost context: FST is 500× cheaper than a day of CRRT.""",
    
    "UGIB": """Scenario: Upper GI bleed.
- PPI ($) + resuscitation first. Octreotide ($$) + ceftriaxone only if variceal probability high.
- Urgent EGD ($$$) ≤12h once stable. Avoid CT ($$$) unless perforation suspected.
- Cost context: EGD is therapeutic; CT adds little EVI in typical UGIB.""",
    
    "CHESTPAIN": """Scenario: Chest pain ACS evaluation.
- Use HEART score. Serial hs-troponins ($) + observation ($$) for low-intermediate risk.
- Avoid stress imaging ($$$, 35× more) or CTCA ($$$, 30× more) if EVI low.
- Cost context: Serial troponins have excellent NPV at 2× baseline cost.""",
    
    "GEN": """Scenario: General inpatient reasoning.
- Apply Bayes (reduce uncertainty), EVI (avoid low-value tests), Utility (benefit−harm−cost).
- Use relative cost tiers: $ = baseline, $$ = 5–15×, $$$ = 30–70×, $$$$$ = 100+×.
- Keep one clear recommendation."""
}

# ---------- Agent Instructions (updated for relative costs) ----------
HYBRID_INSTRUCTIONS = """
You are PRIORI — a Bayesian clinical rounding assistant for inpatient/ICU care.
Tone: professional, collegial, concise. Use internal medical knowledge for LRs and reasoning.
Do all math silently; present an easy-to-digest summary with ONE clear recommendation.

Required sections (Markdown, concise):

**Executive Summary:** One bold sentence with the key clinical conclusion (posterior + next action).

### Clinical TL;DR (for bedside)
- Posterior: X%
- Do now: single best next step (plain language)
- Why: 2–3 bullets linking Bayes → EVI → Utility

*(Math below is optional for readers)*

<details>
<summary>Details (math & rationale)</summary>

### 1) Pre-Test Probability
- Initial estimate (%), with brief rationale (rule or gestalt).

### 2) Evidence Update (LR Table)
| Finding/Test | Result | LR | Source/Note |
|---|---:|---:|---|
| ... | ... | ... | ... |

### 3) Post-Test Probability
- Pre-odds × LR product → Post-odds → Posterior (%).

### 4) Decision & Rationale
- **Recommendation:** single best action.
- Why: Link Bayes (posterior) → EVI (will testing change management?) → Utility (benefit−harm−cost).
- Use RELATIVE cost tiers: $ (baseline), $$ (5–15× baseline), $$$ (30–70×), $$$$$ (100+×).
- Safety overrides (if any).
- Devil's advocate (one-line alternative).

</details>

### 5) JSON (machine-readable)
Return a fenced JSON block with:
{
  "posterior": 0.xx,
  "evi_table": [
    {"test":"...","p_change":0.xx,"value_if_change":0.XX,"evi":0.xx}
  ],
  "costs": [
    {"item":"...","tier":"$","relative_cost":1,"note":"vs comparison"}
  ],
  "utility_rank": [
    {"action":"...","benefit":0.xx,"harm":0.xx,"cost_tier":"$","utility":0.xx}
  ],
  "best_action": "plain-language single action"
}
Ensure valid JSON (no comments).
"""

# ---------- Create Agent ----------
clinical_reasoner = Agent(
    name="PRIORI",
    model=Gemini(id="gemini-2.5-flash", api_key=API_KEY),
    markdown=True,
    description="PRIORI: Bayesian + EVI + Utility clinical reasoning assistant with relative cost awareness.",
    instructions=[HYBRID_INSTRUCTIONS],
)

# ---------- Case Runner ----------
def run_case(user_text: str):
    """Process clinical query and display results in main chat."""
    # Route to appropriate policy
    policy = detect_policy(user_text)
    preface = f"Policy: {policy}\n\n{POLICY_CONTEXT.get(policy,'')}\n\nUser question:\n"
    routed_prompt = preface + user_text

    # Show thinking indicator
    with st.chat_message("assistant"):
        with st.spinner("🧠 Reasoning through the case..."):
            try:
                # Call agent
                run = clinical_reasoner.run(routed_prompt)
                content = run.content
                
                # Render narrative
                st.markdown(content, unsafe_allow_html=True)
                
                # Parse JSON for structured summary
                data = extract_json_tail(content)
                if data:
                    # Clinical TL;DR metrics
                    best = data.get("best_action")
                    posterior = data.get("posterior")
                    
                    if posterior is not None or best:
                        st.divider()
                        st.subheader("📊 Clinical TL;DR (Auto-Extracted)")
                        cols = st.columns(3)
                        
                        if posterior is not None:
                            cols[0].metric("Posterior Probability", f"{float(posterior)*100:.1f}%")
                        
                        if best:
                            cols[1].markdown(f"**Recommended Action:**\n{best}")
                        
                        if data.get("utility_rank"):
                            top = sorted(data["utility_rank"], key=lambda x: x.get("utility", 0.0), reverse=True)[0]
                            cols[2].markdown(f"**Highest Utility:**\n{top.get('action','Best action')}")
                    
                    # Structured tables
                    st.divider()
                    st.subheader("📋 Structured Summary")
                    
                    # EVI Table
                    if data.get("evi_table"):
                        with st.expander("🔬 **Expected Value of Information (EVI)**", expanded=True):
                            st.markdown("*Which tests will actually change management?*")
                            st.table({
                                "Test/Intervention": [x.get("test","") for x in data["evi_table"]],
                                "ΔP (Change in Probability)": [f"{float(x.get('p_change',0.0)):.3f}" for x in data["evi_table"]],
                                "Value if Positive": [f"{float(x.get('value_if_change',0.0)):.3f}" for x in data["evi_table"]],
                                "EVI Score": [f"{float(x.get('evi',0.0)):.3f}" for x in data["evi_table"]],
                            })
                    
                    # Cost Comparison
                    if data.get("costs"):
                        with st.expander("💰 **Relative Cost Comparison**", expanded=True):
                            st.markdown("*Resource stewardship — what's the opportunity cost?*")
                            st.table({
                                "Test/Intervention": [x.get("item","") for x in data["costs"]],
                                "Cost Tier": [x.get("tier","?") for x in data["costs"]],
                                "× Baseline": [x.get("relative_cost","?") for x in data["costs"]],
                                "Context": [x.get("note","") for x in data["costs"]],
                            })
                    
                    # Utility Ranking
                    if data.get("utility_rank"):
                        with st.expander("⚖️ **Utility Ranking (Benefit − Harm − Cost)**", expanded=True):
                            st.markdown("*Higher utility = better value for the patient*")
                            st.table({
                                "Action": [x.get("action","") for x in data["utility_rank"]],
                                "Benefit": [f"{float(x.get('benefit',0.0)):.3f}" for x in data["utility_rank"]],
                                "Harm": [f"{float(x.get('harm',0.0)):.3f}" for x in data["utility_rank"]],
                                "Cost Tier": [x.get("cost_tier","?") for x in data["utility_rank"]],
                                "Net Utility": [f"{float(x.get('utility',0.0)):.3f}" for x in data["utility_rank"]],
                            })
                
                # Save to history
                st.session_state["messages"].append({"role": "assistant", "content": content})
                
            except Exception as e:
                st.error(f"❌ Error processing request: {str(e)}")
                st.info("💡 Tip: Try rephrasing your question or use one of the auto-prompts.")

# ---------- UI Setup ----------
st.set_page_config(
    page_title="PRIORI — Clinical Decision Support", 
    page_icon="🩺", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("🩺 PRIORI — Bayesian Clinical Reasoning")
st.caption("*Probabilistic thinking + Evidence-based decisions + Resource stewardship*")
st.markdown("---")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar with auto-prompts
with st.sidebar:
    st.header("🚀 Quick Start Cases")
    st.caption("Click any button to load a clinical scenario")
    
    st.markdown("### 🫁 Pulmonary Medicine")
    if st.button("📍 PE — D-dimer vs CTA decision", use_container_width=True):
        user_q = (
            "55M with pleuritic chest pain, HR 102, O2 95% RA. "
            "Wells score low-intermediate; no unilateral leg swelling. "
            "Should I get a D-dimer first or go straight to CTA?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        st.rerun()
    
    if st.button("📍 CAP — Avoid low-EVI testing", use_container_width=True):
        user_q = (
            "68M with CAP on CXR, SpO2 93% RA, mild COPD, no chest pain. "
            "Should I order CT chest or stick with CXR + procalcitonin?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        st.rerun()
    
    st.markdown("### 🫀 Critical Care")
    if st.button("📍 Septic Shock — Fluids vs Pressors", use_container_width=True):
        user_q = (
            "ICU patient with septic shock after 1.5L crystalloid; MAP 62, on 0.06 NE. "
            "PLR shows no stroke volume increase; VExUS = 2. More fluids or increase pressor?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        st.rerun()
    
    if st.button("📍 AKI Oliguria — FST vs Early RRT", use_container_width=True):
        user_q = (
            "ICU patient oliguric post-sepsis; Cr 2.7 (baseline 1.0), K 5.3, HCO3 18. "
            "POCUS: VExUS 2, no hydronephrosis; bladder 80 mL. Should I do FST, and when to start CRRT?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        st.rerun()
    
    st.markdown("### ❤️ Cardiology")
    if st.button("📍 Chest Pain — HEART score approach", use_container_width=True):
        user_q = (
            "58F with chest pressure, risk factors HTN/HLD, non-ischemic ECG, initial hs-troponin negative. "
            "HEART score ~4. Serial troponins vs stress test vs CTCA?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        st.rerun()
    
    st.markdown("### 🩸 Gastroenterology")
    if st.button("📍 UGIB — EGD timing decision", use_container_width=True):
        user_q = (
            "62M with melena, Hgb 8.9, MAP 74 stable after 1U PRBC. No cirrhosis history. "
            "Push for urgent EGD now vs early morning? Imaging needed?"
        )
        st.session_state["messages"].append({"role":"user","content":user_q})
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔧 Session Controls")
    if st.button("🆕 Clear Chat History", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()
    
    st.markdown("---")
    st.caption("💡 **How to use:**\n1. Click an auto-prompt\n2. Or type your own case below\n3. Get Bayesian reasoning + EVI + Cost analysis")

# Main chat area
st.subheader("💬 Clinical Conversation")

# Display chat history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            # For assistant messages, just show the content
            # (structured tables are regenerated in run_case)
            st.markdown(m["content"])

# Process any pending user message from auto-prompt
if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    last_user_msg = st.session_state["messages"][-1]["content"]
    # Check if we've already responded to this
    if len(st.session_state["messages"]) == 1 or st.session_state["messages"][-2]["role"] != "assistant":
        run_case(last_user_msg)

# Chat input
if user_free := st.chat_input("💬 Enter your clinical question or case details..."):
    # Add to history
    st.session_state["messages"].append({"role": "user", "content": user_free})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_free)
    
    # Process and respond
    run_case(user_free)

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** PRIORI is an MVP demonstration tool for educational/research purposes. Not for clinical use. All recommendations require physician oversight.")
