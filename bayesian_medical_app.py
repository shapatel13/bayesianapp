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

def hide_json_blocks(content: str) -> str:
    """Remove JSON code blocks from content before displaying to user."""
    return JSON_BLOCK_PATTERN.sub("", content)

# ---------- Guardrails & Safety Validation ----------

class SafetyGuardrails:
    """Multi-layer safety validation for clinical recommendations."""
    
    # Critical value thresholds (when to flag for immediate review)
    CRITICAL_VALUES = {
        "K": (6.5, "Potassium >6.5 — life-threatening hyperkalemia"),
        "K+": (6.5, "Potassium >6.5 — life-threatening hyperkalemia"),
        "potassium": (6.5, "Potassium >6.5 — life-threatening hyperkalemia"),
        "glucose": (600, "Glucose >600 — severe hyperglycemia/DKA risk"),
        "Na": (120, "Sodium <120 — seizure risk"),
        "sodium": (120, "Sodium <120 — seizure risk"),
        "pH": (7.1, "pH <7.1 — severe acidemia"),
        "lactate": (4.0, "Lactate >4 — severe tissue hypoperfusion"),
        "INR": (8.0, "INR >8 — critical bleeding risk"),
        "platelets": (20, "Platelets <20k — critical bleeding risk"),
        "Hgb": (6.0, "Hemoglobin <6 — critical anemia"),
        "hemoglobin": (6.0, "Hemoglobin <6 — critical anemia"),
    }
    
    # Treatment recommendations that require extra scrutiny
    HIGH_RISK_INTERVENTIONS = [
        "thrombolytics", "tPA", "alteplase", "tenecteplase",
        "dialysis", "CRRT", "hemodialysis",
        "intubation", "mechanical ventilation",
        "cardioversion", "defibrillation",
        "surgery", "operative", "surgical intervention",
        "paracentesis", "thoracentesis",
        "lumbar puncture", "LP"
    ]
    
    # Drugs that require renal dosing with AKI mentioned
    NEPHROTOXIC_DRUGS = [
        "acyclovir", "vancomycin", "gentamicin", "tobramycin", "amikacin",
        "amphotericin", "cidofovir", "tenofovir", "NSAIDs", "ibuprofen",
        "contrast", "cisplatin"
    ]
    
    @staticmethod
    def check_critical_values(user_text: str) -> dict:
        """Scan for critical lab values that require immediate attention."""
        warnings = []
        text_lower = user_text.lower()
        
        for key, (threshold, message) in SafetyGuardrails.CRITICAL_VALUES.items():
            # Look for patterns like "K 7.2" or "potassium 7.2" or "K+ 7.2"
            import re
            patterns = [
                rf"{key}\s*[=>:]\s*(\d+\.?\d*)",
                rf"{key}\s+(\d+\.?\d*)",
            ]
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    try:
                        value = float(match)
                        if key in ["Na", "sodium", "pH", "platelets", "Hgb", "hemoglobin"]:
                            # These are dangerous when LOW
                            if value < threshold:
                                warnings.append(f"⚠️ **CRITICAL VALUE DETECTED**: {message}")
                        else:
                            # These are dangerous when HIGH
                            if value > threshold:
                                warnings.append(f"⚠️ **CRITICAL VALUE DETECTED**: {message}")
                    except ValueError:
                        continue
        
        return {"has_critical": len(warnings) > 0, "warnings": warnings}
    
    @staticmethod
    def check_high_risk_intervention(recommendation: str) -> dict:
        """Flag high-risk interventions for human oversight."""
        rec_lower = recommendation.lower()
        flagged = []
        
        for intervention in SafetyGuardrails.HIGH_RISK_INTERVENTIONS:
            if intervention.lower() in rec_lower:
                flagged.append(intervention)
        
        if flagged:
            return {
                "requires_oversight": True,
                "message": f"🔴 **HIGH-RISK INTERVENTION**: Recommendation includes {', '.join(flagged)}. Requires attending physician approval."
            }
        return {"requires_oversight": False}
    
    @staticmethod
    def check_nephrotoxic_with_aki(user_text: str, recommendation: str) -> dict:
        """Flag nephrotoxic drugs when AKI is mentioned."""
        if any(term in user_text.lower() for term in ["aki", "acute kidney", "renal failure", "creatinine", "oligur"]):
            rec_lower = recommendation.lower()
            flagged = []
            
            for drug in SafetyGuardrails.NEPHROTOXIC_DRUGS:
                if drug.lower() in rec_lower and "stop" not in rec_lower and "discontinue" not in rec_lower:
                    flagged.append(drug)
            
            if flagged:
                return {
                    "nephrotoxicity_risk": True,
                    "message": f"⚠️ **NEPHROTOXICITY ALERT**: Recommending {', '.join(flagged)} in setting of AKI. Verify renal dosing and risk-benefit."
                }
        
        return {"nephrotoxicity_risk": False}
    
    @staticmethod
    def validate_posterior_threshold(data: dict, user_text: str) -> dict:
        """Check if treatment aligns with posterior probability."""
        posterior = data.get("posterior")
        recommendation = data.get("recommendation", "") or data.get("best_action", "")
        
        if not posterior or not recommendation:
            return {"threshold_warning": False}
        
        warnings = []
        
        # Very low posterior (<5%) but recommending treatment
        if posterior < 0.05:
            treat_keywords = ["start", "begin", "initiate", "give", "administer", "continue"]
            if any(kw in recommendation.lower() for kw in treat_keywords):
                # Check if high harm treatment
                high_harm = any(drug in user_text.lower() for drug in SafetyGuardrails.NEPHROTOXIC_DRUGS)
                if high_harm:
                    warnings.append(
                        f"⚠️ **LOW PROBABILITY WARNING**: Posterior probability is {posterior*100:.1f}% but recommending treatment with known toxicity. "
                        f"Verify that expected benefit exceeds expected harm."
                    )
        
        # Very high posterior (>80%) but recommending against treatment
        if posterior > 0.80:
            stop_keywords = ["stop", "discontinue", "avoid", "do not", "defer"]
            if any(kw in recommendation.lower() for kw in stop_keywords):
                warnings.append(
                    f"⚠️ **HIGH PROBABILITY WARNING**: Posterior probability is {posterior*100:.1f}% but recommending against treatment. "
                    f"Verify this is appropriate given high disease probability."
                )
        
        return {"threshold_warning": len(warnings) > 0, "warnings": warnings}
    
    @staticmethod
    def check_harm_benefit_mismatch(data: dict) -> dict:
        """Verify harm-benefit calculation is internally consistent."""
        utility_rank = data.get("utility_rank", [])
        best_action = data.get("recommendation", "") or data.get("best_action", "")
        
        if not utility_rank or not best_action:
            return {"mismatch": False}
        
        # Find highest utility action
        sorted_actions = sorted(utility_rank, key=lambda x: x.get("utility", 0), reverse=True)
        if not sorted_actions:
            return {"mismatch": False}
        
        top_action = sorted_actions[0].get("action", "").lower()
        best_lower = best_action.lower()
        
        # Check if recommendation matches highest utility action
        # Simple keyword overlap check
        top_keywords = set(top_action.split())
        best_keywords = set(best_lower.split())
        overlap = len(top_keywords & best_keywords) / max(len(top_keywords), 1)
        
        if overlap < 0.3:  # Less than 30% keyword overlap
            return {
                "mismatch": True,
                "message": f"⚠️ **INTERNAL INCONSISTENCY**: Highest utility action is '{sorted_actions[0].get('action')}' but recommendation is '{best_action}'. Review reasoning."
            }
        
        return {"mismatch": False}
    
    @staticmethod
    def run_all_checks(user_text: str, data: dict) -> list:
        """Run all guardrail checks and return list of warnings."""
        warnings = []
        
        # Critical values
        crit_check = SafetyGuardrails.check_critical_values(user_text)
        if crit_check["has_critical"]:
            warnings.extend(crit_check["warnings"])
        
        # Get recommendation from data
        recommendation = data.get("recommendation", "") or data.get("best_action", "")
        
        # High-risk interventions
        risk_check = SafetyGuardrails.check_high_risk_intervention(recommendation)
        if risk_check["requires_oversight"]:
            warnings.append(risk_check["message"])
        
        # Nephrotoxicity in AKI
        nephro_check = SafetyGuardrails.check_nephrotoxic_with_aki(user_text, recommendation)
        if nephro_check["nephrotoxicity_risk"]:
            warnings.append(nephro_check["message"])
        
        # Posterior threshold alignment
        threshold_check = SafetyGuardrails.validate_posterior_threshold(data, user_text)
        if threshold_check["threshold_warning"]:
            warnings.extend(threshold_check.get("warnings", []))
        
        # Harm-benefit internal consistency
        mismatch_check = SafetyGuardrails.check_harm_benefit_mismatch(data)
        if mismatch_check["mismatch"]:
            warnings.append(mismatch_check["message"])
        
        return warnings

# ---------- Scenario hint detection (lightweight, non-prescriptive) ----------
def detect_scenario_hint(user_text: str) -> str:
    """Detect scenario type to provide minimal context hint, NOT full instructions."""
    t = (user_text or "").lower()
    
    # Just give the LLM a hint about what domain this is in
    # Don't tell it what to do
    if any(k in t for k in ["pulmonary embol", " pe ", " pe.", "wells", "perc", "d-dimer", "d dimer"]):
        return "pulmonary embolism workup"
    if any(k in t for k in ["encephalitis", "hsv", "acyclovir", "temporal lobe", "meningitis", "csf", "lumbar puncture"]):
        return "CNS infection (encephalitis/meningitis)"
    if any(k in t for k in ["pneumonia", "cap", "community-acquired", "infiltrate", "procalcitonin"]):
        return "community-acquired pneumonia"
    if any(k in t for k in ["septic shock", "sepsis", "norepinephrine", "vasopressin", "vexus", "plr"]):
        return "septic shock/sepsis management"
    if any(k in t for k in ["aki", "oligur", "furosemide stress", "fst", "dialysis", "crrt", "acute kidney"]):
        return "acute kidney injury"
    if any(k in t for k in ["gi bleed", "hematemesis", "melena", "varice", "egd", "octreotide"]):
        return "upper GI bleeding"
    if any(k in t for k in ["chest pain", "heart score", "troponin", "ecg", "acs", "acute coronary"]):
        return "chest pain/ACS evaluation"
    
    return "general inpatient medicine"

def detect_disagreement_frame(user_text: str) -> bool:
    """Check if user is questioning or comparing to an existing clinical plan."""
    disagreement_cues = [
        "but the", "however the", "the doctor wants", "they want to", 
        "we're planning to", "my attending says", "recommends", 
        "should we continue", "should i continue", "keep going",
        "my thought is", "i think", "despite", "even though"
    ]
    return any(cue in user_text.lower() for cue in disagreement_cues)

# ---------- Core Reasoning Framework (principle-based, not prescriptive) ----------
REASONING_FRAMEWORK = """
You are reasoning about: {scenario_hint}

Apply these core principles using YOUR medical knowledge:

**Bayesian Foundation:**
1. Estimate pre-test probability from base rates, epidemiology, clinical gestalt
2. Identify available evidence (exam findings, labs, imaging) and their likelihood ratios from medical literature
3. Update probability using: posterior odds = prior odds × LR₁ × LR₂ × ... × LRₙ
4. State your assumptions clearly (if you don't know exact LR, give reasonable range)

**Treatment Decision Thresholds:**
- Test threshold: probability above which testing changes management
- Treatment threshold: probability above which treatment benefit exceeds harm
- These thresholds SHIFT based on:
  * Severity of disease if untreated (higher = lower threshold to treat)
  * Toxicity/risk of treatment (higher = higher threshold to treat)
  * Patient-specific factors (renal/hepatic dysfunction, drug interactions, frailty)

**Harm-Benefit Analysis (CRITICAL):**
When treatment has significant toxicity OR patient has contraindications:
- Calculate: E(Benefit) = P(disease) × P(harm prevented by treatment) × value
- Calculate: E(Harm) = P(adverse event from treatment) × severity
- If E(Harm) > E(Benefit) → DO NOT TREAT, regardless of specialty opinion
- State the threshold: "Treatment justified only if P(disease) > [X]%"

**Expected Value of Information (EVI):**
- Will this test result change management? (If no → don't order)
- What is the opportunity cost? (Resource stewardship)
- Use relative cost tiers: $ (baseline), $$ (5-15×), $$$ (30-70×), $$$$$ (100+×)

**Independence & Anti-Anchoring:**
- If user mentions "the [specialist] wants to do X", acknowledge but reason INDEPENDENTLY
- Do the math FIRST, then compare to stated plan
- If your analysis differs, say so explicitly with quantitative reasoning
- Avoid sycophantic phrases like "you're right to consider..." 
- Instead: "Based on posterior of X%, expected harm exceeds benefit by Y-fold"

**Guardrails:**
- Very low probability (<5%) + high treatment harm → strongly favor stopping
- Very high probability (>80%) + low treatment harm → favor treating
- Middle range (5-80%) → detailed harm-benefit analysis required
"""

# ---------- Agent Instructions (principle-based reasoning) ----------
AGENT_INSTRUCTIONS = """
You are PRIORI — a Bayesian clinical reasoning assistant.

Your role: INDEPENDENT probabilistic analysis using first-principles medical reasoning.
NOT: Validation of existing decisions or specialty opinions.

**Core Method:**
1. Estimate pre-test probability (use base rates, epidemiology, clinical prediction rules)
2. Identify relevant evidence and apply likelihood ratios from your medical knowledge
3. Calculate posterior probability using Bayesian updating
4. Perform harm-benefit analysis: E(Benefit of action) vs E(Harm of action)
5. Make ONE clear recommendation based on expected utility

**When treatment has toxicity OR disease probability is low:**
- Calculate: P(disease) × P(benefit if treated) vs P(harm from treatment) × severity
- State threshold: "Treatment justified only if P(disease) > X%"
- If current posterior < threshold → recommend AGAINST treatment

**Output Format (Markdown, concise):**

**Executive Summary:** One sentence with posterior probability and recommendation.

### Clinical Reasoning
- **Posterior Probability:** X% (show brief calculation path)
- **Recommendation:** Single clear next step
- **Rationale:** 2-3 bullets connecting Bayes → harm-benefit → decision

<details>
<summary>Detailed Analysis (optional for reader)</summary>

### Pre-Test Probability
[Brief justification with base rates/clinical context]

### Evidence & Likelihood Ratios
| Finding/Test | Result | LR | Note |
|---|---:|---:|---|
[Use YOUR medical knowledge for LRs]

### Post-Test Calculation
[Show odds multiplication]

### Harm-Benefit Analysis
**If proposing treatment:**
- E(Benefit) = P(disease) × P(harm prevented) = [calculate]
- E(Harm from Rx) = P(adverse event) × severity = [calculate]
- Net utility = [Benefit - Harm]
- Threshold for treatment = [X%]

**If proposing testing:**
- Will result change management? (EVI analysis)
- Cost tier comparison (use relative costs: $, $$, $$$)

</details>

### Structured Output (JSON)
```json
{
  "posterior": 0.xx,
  "recommendation": "single action",
  "threshold_analysis": "Treatment justified if P>X% given harm profile",
  "evi_table": [{"test":"...","evi":0.xx}],
  "costs": [{"item":"...","tier":"$","relative_cost":X}],
  "utility_rank": [{"action":"...","benefit":0.xx,"harm":0.xx,"cost_tier":"$","utility":0.xx}]
}
```

**Critical Rules:**
- Use YOUR medical knowledge for base rates, LRs, and clinical context
- If uncertain about exact values, give reasonable ranges and state assumptions
- Do NOT defer to specialty opinion — reason independently
- Avoid sycophantic language ("you're right", "good thinking")
- Use phrases like "Based on posterior of X%, the math shows..."
"""

# ---------- Create Agent ----------
clinical_reasoner = Agent(
    name="PRIORI",
    model=Gemini(id="gemini-2.5-flash", api_key=API_KEY),
    markdown=True,
    description="PRIORI: Independent Bayesian clinical reasoning with harm-benefit analysis and anti-anchoring safeguards.",
    instructions=[AGENT_INSTRUCTIONS],
)

# ---------- Case Runner ----------
def run_case(user_text: str):
    """Process clinical query and display results in main chat."""
    # Detect scenario and disagreement framing
    scenario_hint = detect_scenario_hint(user_text)
    has_disagreement = detect_disagreement_frame(user_text)
    
    # Build reasoning framework prompt
    framework_prompt = REASONING_FRAMEWORK.format(scenario_hint=scenario_hint)
    
    # Add anti-anchoring instruction if user is questioning existing plan
    if has_disagreement:
        framework_prompt += """

**CRITICAL — INDEPENDENT REASONING REQUIRED:**
The user appears to be questioning or comparing to an existing clinical plan.
- Do NOT anchor on the stated plan or specialty opinion
- Calculate probabilities and utilities INDEPENDENTLY first
- Then compare your math-based recommendation to the existing plan
- If they differ, state explicitly: "The stated plan recommends X, but based on posterior probability of Y% and harm-benefit ratio of Z, the evidence supports [your recommendation]"
- Quantify the disagreement (e.g., "harm exceeds benefit by 10-fold")
"""
    
    routed_prompt = framework_prompt + "\n\nUser's clinical question:\n" + user_text

    # Show thinking indicator
    with st.chat_message("assistant"):
        with st.spinner("🧠 Reasoning through the case..."):
            try:
                # Call agent
                run = clinical_reasoner.run(routed_prompt)
                content = run.content
                
                # Parse JSON FIRST (before we hide it)
                data = extract_json_tail(content)
                
                # Run safety guardrails
                safety_warnings = SafetyGuardrails.run_all_checks(user_text, data)
                
                # Hide JSON blocks from user view
                clean_content = hide_json_blocks(content)
                
                # Display safety warnings FIRST if any
                if safety_warnings:
                    st.error("### 🛡️ Safety Guardrails Triggered")
                    for warning in safety_warnings:
                        st.warning(warning)
                    st.info("💡 **Note**: These are automated safety checks. Review the reasoning below and use clinical judgment.")
                    st.divider()
                
                # Render narrative (without JSON blocks)
                st.markdown(clean_content, unsafe_allow_html=True)
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
                
                # Save to history (without JSON blocks)
                st.session_state["messages"].append({"role": "assistant", "content": clean_content})
                
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
    st.markdown("### 🛡️ Active Guardrails")
    with st.expander("Safety Checks Enabled"):
        st.markdown("""
        **Automated safety validation:**
        - ⚠️ Critical lab values (K>6.5, Na<120, etc.)
        - 🔴 High-risk interventions (thrombolytics, dialysis, intubation)
        - 💊 Nephrotoxic drugs in AKI setting
        - 📊 Posterior probability vs treatment alignment
        - ⚖️ Harm-benefit internal consistency
        
        *These checks flag potential issues but do NOT override clinical judgment.*
        """)
    
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
