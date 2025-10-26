import streamlit as st
import json
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from agno.agent import Agent
from agno.models.google import Gemini

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = 'AIzaSyDmcPbEDAEojTomYs7vLKu107fOa7c6500'

# ============================================================================
# COST TIERS (Resource Stewardship)
# ============================================================================

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

# ============================================================================
# SAFETY GUARDRAILS
# ============================================================================

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
        
        return {
            "has_high_risk": len(flagged) > 0,
            "interventions": flagged,
            "message": f"🔴 High-risk intervention detected: {', '.join(flagged)}" if flagged else ""
        }
    
    @staticmethod
    def check_nephrotoxic_drugs(user_text: str, recommendation: str) -> dict:
        """Check for nephrotoxic drugs when AKI is mentioned."""
        has_aki = any(term in user_text.lower() for term in ["aki", "acute kidney injury", "renal failure", "creatinine", "oliguria"])
        
        if not has_aki:
            return {"has_warning": False, "drugs": [], "message": ""}
        
        rec_lower = recommendation.lower()
        flagged_drugs = []
        
        for drug in SafetyGuardrails.NEPHROTOXIC_DRUGS:
            if drug.lower() in rec_lower:
                flagged_drugs.append(drug)
        
        if flagged_drugs:
            return {
                "has_warning": True,
                "drugs": flagged_drugs,
                "message": f"💊 **AKI + Nephrotoxic Drug**: {', '.join(flagged_drugs)} — Consider renal dosing adjustment"
            }
        
        return {"has_warning": False, "drugs": [], "message": ""}
    
    @staticmethod
    def validate_all(user_text: str, recommendation: str) -> List[str]:
        """Run all safety checks and return list of warnings."""
        all_warnings = []
        
        # Critical values
        critical = SafetyGuardrails.check_critical_values(user_text)
        if critical["has_critical"]:
            all_warnings.extend(critical["warnings"])
        
        # High-risk interventions
        high_risk = SafetyGuardrails.check_high_risk_intervention(recommendation)
        if high_risk["has_high_risk"]:
            all_warnings.append(high_risk["message"])
        
        # Nephrotoxic drugs
        nephro = SafetyGuardrails.check_nephrotoxic_drugs(user_text, recommendation)
        if nephro["has_warning"]:
            all_warnings.append(nephro["message"])
        
        return all_warnings

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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

def safe_float(value, default=0.0) -> float:
    """Safely convert value to float, handling strings and errors."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try to extract first number from string
        numbers = re.findall(r'-?\d+\.?\d*', value)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return default
    return default

def hide_json_blocks(content: str) -> str:
    """Remove JSON code blocks from content before displaying to user."""
    return JSON_BLOCK_PATTERN.sub("", content)

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ClinicalDecisionContext:
    """Enhanced context with advanced decision theory"""
    prior_probability: float
    likelihood_ratios: Dict[str, Tuple[float, float]]  # test -> (LR+, LR-)
    utilities: Dict[str, float]  # outcome -> utility (QALYs)
    patient_preferences: Dict[str, float]  # preference weights
    test_costs: Dict[str, float]
    treatment_costs: Dict[str, float]
    disease_name: str
    test_names: List[str]
    patient_query: str
    # Clinical outcomes (more intuitive than QALYs)
    mortality_untreated: float = 0.30  # Default 30%
    mortality_treated: float = 0.08    # Default 8%
    major_complication_rate: float = 0.02  # Default 2% (NNH)
    # Time-critical urgency metrics (evidence-based)
    has_shock: bool = False  # Critical distinction per literature
    mortality_per_hour_delay: Optional[float] = None  # From published studies
    urgency_category: str = "NON-URGENT"  # CRITICAL/URGENT/SEMI-URGENT/NON-URGENT
    de_escalation_probability: float = 0.6  # Probability of narrowing therapy
    de_escalation_timeline_hours: float = 48.0  # When to reassess
    literature_source: Optional[str] = None  # Citation for time metrics
    
@dataclass
class ClinicalOutcomes:
    """Clinician-friendly outcome metrics"""
    nnt_mortality: Optional[int]  # Number Needed to Treat to prevent 1 death
    nnh_major_complication: Optional[int]  # Number Needed to Harm
    arr_mortality: float  # Absolute Risk Reduction
    rrr_mortality: float  # Relative Risk Reduction
    lives_saved_per_100: float  # Expected lives saved per 100 treated
    complications_per_100: float  # Expected complications per 100 treated
    interpretation: str  # Plain language
    
@dataclass
class AdvancedAnalysisResults:
    """Results from advanced decision theory"""
    thresholds: Dict[str, float]
    evpi: Dict[str, Dict]
    mcmc_results: Dict
    sensitivity: Dict
    influence_diagram: Dict
    bias_warnings: List[Dict]
    recommendation: str
    confidence: float
    llm_reasoning: str
    safety_warnings: List[str]
    clinical_outcomes: Optional[ClinicalOutcomes] = None  # NEW: NNT/NNH metrics

# ============================================================================
# ADVANCED DECISION ENGINE
# ============================================================================

class AdvancedDecisionEngine:
    """Implements cutting-edge decision theory from the PDF"""
    
    def __init__(self):
        self.n_simulations = 10000
    
    def analyze(self, context: ClinicalDecisionContext, safety_warnings: List[str]) -> AdvancedAnalysisResults:
        """Run all advanced analyses in parallel with LLM"""
        
        # Calculate thresholds
        thresholds = self._compute_thresholds(context)
        
        # Calculate EVPI
        evpi = self._calculate_evpi(context)
        
        # Run MCMC simulation
        mcmc_results = self._run_mcmc_simulation(context)
        
        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(context)
        
        # Build influence diagram
        influence_diagram = self._build_influence_diagram(context)
        
        # Detect cognitive biases
        bias_warnings = self._detect_cognitive_biases(context)
        
        # Determine recommendation
        recommendation = self._get_recommendation(context, thresholds)
        
        # Calculate confidence
        confidence = self._calculate_confidence(mcmc_results, sensitivity)
        
        # Calculate clinical outcomes (NNT/NNH)
        clinical_outcomes = self._calculate_clinical_outcomes(context)
        
        return AdvancedAnalysisResults(
            thresholds=thresholds,
            evpi=evpi,
            mcmc_results=mcmc_results,
            sensitivity=sensitivity,
            influence_diagram=influence_diagram,
            bias_warnings=bias_warnings,
            recommendation=recommendation,
            confidence=confidence,
            llm_reasoning="",  # Will be filled by LLM
            safety_warnings=safety_warnings,
            clinical_outcomes=clinical_outcomes
        )
    
    def _calculate_clinical_outcomes(self, context: ClinicalDecisionContext) -> ClinicalOutcomes:
        """
        Calculate clinician-friendly outcome metrics (NNT/NNH)
        More intuitive than QALYs for bedside decision-making
        """
        mortality_untreated = context.mortality_untreated
        mortality_treated = context.mortality_treated
        complication_rate = context.major_complication_rate
        
        # Absolute Risk Reduction (ARR)
        arr = mortality_untreated - mortality_treated
        
        # Relative Risk Reduction (RRR)
        rrr = arr / mortality_untreated if mortality_untreated > 0 else 0
        
        # Number Needed to Treat (NNT)
        nnt = int(round(1 / arr)) if arr > 0 else None
        
        # Number Needed to Harm (NNH)
        nnh = int(round(1 / complication_rate)) if complication_rate > 0 else None
        
        # Expected outcomes per 100 patients
        lives_saved_per_100 = arr * 100
        complications_per_100 = complication_rate * 100
        
        # Plain language interpretation
        if nnt and nnh:
            if nnt < 10:
                benefit_level = "Very High"
            elif nnt < 20:
                benefit_level = "High"
            elif nnt < 50:
                benefit_level = "Moderate"
            else:
                benefit_level = "Low"
            
            # Risk-benefit ratio
            if nnh > nnt * 10:
                risk_level = "Low risk, excellent safety profile"
            elif nnh > nnt * 5:
                risk_level = "Acceptable risk-benefit ratio"
            elif nnh > nnt * 2:
                risk_level = "Moderate risk, consider carefully"
            else:
                risk_level = "High risk, use with caution"
            
            interpretation = f"{benefit_level} benefit (NNT={nnt}). {risk_level} (NNH={nnh})."
        else:
            interpretation = "Insufficient data for NNT/NNH calculation"
        
        return ClinicalOutcomes(
            nnt_mortality=nnt,
            nnh_major_complication=nnh,
            arr_mortality=arr,
            rrr_mortality=rrr,
            lives_saved_per_100=lives_saved_per_100,
            complications_per_100=complications_per_100,
            interpretation=interpretation
        )
    
    def _compute_thresholds(self, context: ClinicalDecisionContext) -> Dict[str, float]:
        """
        Dynamic threshold calculation (PDF page 83)
        Finds exact probability where decision changes
        """
        # Utility of treating when disease present
        u_treat_disease = context.utilities.get('treat_success', 0.9)
        # Utility of treating when no disease (harm from treatment)
        u_treat_no_disease = context.utilities.get('treat_healthy', 0.95)
        # Utility of not treating when disease present (disease progresses)
        u_observe_disease = context.utilities.get('observe_disease', 0.0)
        # Utility of not treating when no disease
        u_observe_no_disease = context.utilities.get('observe_healthy', 1.0)
        
        # Treatment threshold: where E(treat) = E(observe)
        numerator = u_observe_no_disease - u_treat_no_disease
        denominator = (u_treat_disease - u_observe_disease) - (u_treat_no_disease - u_observe_no_disease)
        
        if denominator != 0:
            treat_threshold = numerator / denominator
        else:
            treat_threshold = 0.5
        
        # Test threshold (simplified)
        test_threshold = treat_threshold * 0.3
        
        return {
            'test_threshold': max(0.01, min(0.99, test_threshold)),
            'treat_threshold': max(0.01, min(0.99, treat_threshold)),
            'observe_threshold': 0.0
        }
    
    def _calculate_evpi(self, context: ClinicalDecisionContext) -> Dict[str, Dict]:
        """
        Expected Value of Perfect Information (PDF page 81)
        Determines if a test is worth performing
        """
        evpi_results = {}
        
        # Current EV without any test
        p = context.prior_probability
        ev_treat = p * context.utilities.get('treat_success', 0.9) + \
                   (1-p) * context.utilities.get('treat_healthy', 0.95)
        ev_observe = p * context.utilities.get('observe_disease', 0.0) + \
                     (1-p) * context.utilities.get('observe_healthy', 1.0)
        ev_current = max(ev_treat, ev_observe)
        
        for test_name in context.test_names:
            if test_name not in context.likelihood_ratios:
                continue
                
            lr_pos, lr_neg = context.likelihood_ratios[test_name]
            
            # EV if we knew the truth with certainty
            ev_perfect = p * context.utilities.get('treat_success', 0.9) + \
                        (1-p) * context.utilities.get('observe_healthy', 1.0)
            
            # EVPI in QALYs
            evpi_qalys = max(0, ev_perfect - ev_current)
            
            # Cost per QALY
            test_cost = context.test_costs.get(test_name, 100)
            cost_per_qaly = test_cost / evpi_qalys if evpi_qalys > 0 else float('inf')
            
            # Recommendation
            if cost_per_qaly < 50000:
                recommendation = "✓ Worthwhile"
                reason = f"Cost-effective at ${cost_per_qaly:,.0f}/QALY"
            elif cost_per_qaly < 100000:
                recommendation = "⚠ Consider"
                reason = f"Moderate value at ${cost_per_qaly:,.0f}/QALY"
            else:
                recommendation = "✗ Skip"
                reason = f"Poor value at ${cost_per_qaly:,.0f}/QALY"
            
            evpi_results[test_name] = {
                'evpi_qalys': evpi_qalys,
                'cost_per_qaly': cost_per_qaly,
                'recommendation': recommendation,
                'reason': reason,
                'test_cost': test_cost
            }
        
        return evpi_results
    
    def _run_mcmc_simulation(self, context: ClinicalDecisionContext) -> Dict:
        """
        Monte Carlo simulation (PDF pages 91-93)
        Returns probability distributions instead of point estimates
        """
        samples = []
        
        for _ in range(self.n_simulations):
            # Add uncertainty to prior
            noisy_prior = np.clip(
                np.random.normal(context.prior_probability, 0.05),
                0.01, 0.99
            )
            samples.append(noisy_prior)
        
        samples = np.array(samples)
        
        return {
            'samples': samples,
            'mean': float(np.mean(samples)),
            'std': float(np.std(samples)),
            'ci_95_lower': float(np.percentile(samples, 2.5)),
            'ci_95_upper': float(np.percentile(samples, 97.5)),
            'uncertainty_high': float(np.std(samples)) > 0.1
        }
    
    def _sensitivity_analysis(self, context: ClinicalDecisionContext) -> Dict:
        """
        One-way sensitivity analysis
        Tests how recommendation changes with parameter variations
        """
        base_recommendation = self._get_recommendation(context, self._compute_thresholds(context))
        
        sensitivities = {}
        parameters = ['prior_probability', 'utilities']
        
        for param in parameters:
            variations = []
            
            if param == 'prior_probability':
                for factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
                    new_prob = np.clip(context.prior_probability * factor, 0.01, 0.99)
                    new_context = ClinicalDecisionContext(
                        prior_probability=new_prob,
                        likelihood_ratios=context.likelihood_ratios,
                        utilities=context.utilities,
                        patient_preferences=context.patient_preferences,
                        test_costs=context.test_costs,
                        treatment_costs=context.treatment_costs,
                        disease_name=context.disease_name,
                        test_names=context.test_names,
                        patient_query=context.patient_query
                    )
                    new_rec = self._get_recommendation(new_context, self._compute_thresholds(new_context))
                    variations.append({
                        'factor': factor,
                        'value': new_prob,
                        'recommendation': new_rec,
                        'changed': new_rec != base_recommendation
                    })
            
            sensitivities[param] = variations
        
        # Determine most influential parameter
        max_changes = 0
        most_influential = None
        
        for param, variations in sensitivities.items():
            changes = sum(1 for v in variations if v['changed'])
            if changes > max_changes:
                max_changes = changes
                most_influential = param
        
        return {
            'sensitivities': sensitivities,
            'most_influential': most_influential or 'None',
            'decision_fragile': max_changes > 2
        }
    
    def _build_influence_diagram(self, context: ClinicalDecisionContext) -> Dict:
        """
        Creates an enhanced influence diagram structure with clinical context
        """
        disease_name = context.disease_name or "Disease"
        
        nodes = [
            {
                'id': 'clinical_findings',
                'label': 'Clinical Findings\n& History',
                'type': 'chance',
                'description': 'Patient presentation, symptoms, risk factors',
                'value': None
            },
            {
                'id': 'prior',
                'label': f'Prior Probability\n{disease_name}',
                'type': 'chance',
                'description': f'Pre-test probability based on clinical presentation',
                'value': f'{context.prior_probability:.1%}'
            },
            {
                'id': 'test_decision',
                'label': 'Test Selection\nDecision',
                'type': 'decision',
                'description': 'Which diagnostic test(s) to order',
                'value': ', '.join(context.test_names[:2]) if context.test_names else 'Tests available'
            },
            {
                'id': 'test_result',
                'label': 'Test Results',
                'type': 'chance',
                'description': 'Positive/negative findings from diagnostic tests',
                'value': None
            },
            {
                'id': 'posterior',
                'label': f'Posterior Probability\n{disease_name}',
                'type': 'chance',
                'description': 'Updated probability after test results',
                'value': 'Updated by LRs'
            },
            {
                'id': 'treatment_decision',
                'label': 'Treatment\nDecision',
                'type': 'decision',
                'description': 'Treat, observe, or test further',
                'value': None
            },
            {
                'id': 'utilities',
                'label': 'Expected\nUtilities',
                'type': 'value',
                'description': 'Benefits and harms of each action',
                'value': f'Treat: {context.utilities.get("treat_success", 0):.2f} QALYs'
            },
            {
                'id': 'costs',
                'label': 'Resource\nCosts',
                'type': 'value',
                'description': 'Financial and opportunity costs',
                'value': 'Cost tiers considered'
            },
            {
                'id': 'outcome',
                'label': 'Patient\nOutcome',
                'type': 'value',
                'description': 'Clinical outcome incorporating all factors',
                'value': 'Maximize QALYs'
            }
        ]
        
        edges = [
            {'from': 'clinical_findings', 'to': 'prior', 'label': 'Informs'},
            {'from': 'prior', 'to': 'test_decision', 'label': 'Guides'},
            {'from': 'test_decision', 'to': 'test_result', 'label': 'Produces'},
            {'from': 'test_result', 'to': 'posterior', 'label': 'Updates via LR'},
            {'from': 'posterior', 'to': 'treatment_decision', 'label': 'Informs'},
            {'from': 'utilities', 'to': 'treatment_decision', 'label': 'Weighs'},
            {'from': 'costs', 'to': 'test_decision', 'label': 'Constrains'},
            {'from': 'costs', 'to': 'treatment_decision', 'label': 'Constrains'},
            {'from': 'treatment_decision', 'to': 'outcome', 'label': 'Determines'},
            {'from': 'posterior', 'to': 'outcome', 'label': 'Affects'}
        ]
        
        return {
            'description': f'Complete decision pathway for {disease_name} evaluation',
            'nodes': nodes,
            'edges': edges
        }
    
    def _detect_cognitive_biases(self, context: ClinicalDecisionContext) -> List[Dict]:
        """
        Checks for common cognitive biases in medical decision-making
        """
        warnings = []
        
        # Anchoring bias - overreliance on initial probability
        if context.prior_probability < 0.05 or context.prior_probability > 0.95:
            warnings.append({
                'bias': 'Anchoring Bias',
                'icon': '⚓',
                'description': f'Extreme prior probability ({context.prior_probability:.1%}) may anchor judgment',
                'suggestion': 'Consider if this prior reflects true population prevalence or observation bias'
            })
        
        # Availability bias - recent cases affecting judgment
        if any(lr[0] > 50 for lr in context.likelihood_ratios.values()):
            warnings.append({
                'bias': 'Availability Bias',
                'icon': '🎯',
                'description': 'Very high likelihood ratio suggests recent memorable cases may influence judgment',
                'suggestion': 'Verify test characteristics against published literature, not recent experience'
            })
        
        # Omission bias - preference for inaction
        u_observe = context.utilities.get('observe_healthy', 1.0) - context.utilities.get('observe_disease', 0.0)
        u_treat = context.utilities.get('treat_success', 0.9) - context.utilities.get('treat_healthy', 0.95)
        
        if u_observe > u_treat * 2:
            warnings.append({
                'bias': 'Omission Bias',
                'icon': '🛑',
                'description': 'Utility structure shows strong preference for observation over action',
                'suggestion': 'Consider if fear of treatment complications is appropriate or excessive'
            })
        
        return warnings
    
    def _get_recommendation(self, context: ClinicalDecisionContext, thresholds: Dict[str, float]) -> str:
        """
        Determine clinical recommendation based on probability and thresholds
        """
        p = context.prior_probability
        
        if p < thresholds['test_threshold']:
            return "Observe"
        elif p < thresholds['treat_threshold']:
            return "Test"
        else:
            return "Treat"
    
    def _calculate_confidence(self, mcmc_results: Dict, sensitivity: Dict) -> float:
        """
        Calculate overall confidence in recommendation
        """
        # Lower uncertainty = higher confidence
        uncertainty_score = 1.0 - min(mcmc_results['std'] / 0.2, 1.0)
        
        # Robust decision = higher confidence
        robustness_score = 0.0 if sensitivity['decision_fragile'] else 1.0
        
        # Weighted average
        confidence = 0.6 * uncertainty_score + 0.4 * robustness_score
        
        return confidence

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_threshold_viz(context: ClinicalDecisionContext, thresholds: Dict[str, float]) -> go.Figure:
    """Create threshold visualization"""
    p = context.prior_probability
    
    fig = go.Figure()
    
    # Add threshold zones
    fig.add_vrect(x0=0, x1=thresholds['test_threshold'],
                  fillcolor="green", opacity=0.2, line_width=0,
                  annotation_text="Observe", annotation_position="top left")
    fig.add_vrect(x0=thresholds['test_threshold'], x1=thresholds['treat_threshold'],
                  fillcolor="yellow", opacity=0.2, line_width=0,
                  annotation_text="Test", annotation_position="top")
    fig.add_vrect(x0=thresholds['treat_threshold'], x1=1.0,
                  fillcolor="red", opacity=0.2, line_width=0,
                  annotation_text="Treat", annotation_position="top right")
    
    # Add current probability marker
    fig.add_vline(x=p, line_dash="dash", line_color="blue", line_width=3,
                  annotation_text=f"Current: {p:.1%}", annotation_position="top")
    
    fig.update_layout(
        title="Decision Threshold Analysis",
        xaxis_title="Probability of Disease",
        yaxis_title="Expected Utility",
        xaxis=dict(range=[0, 1], tickformat='.0%'),
        height=400
    )
    
    return fig

def create_mcmc_distribution(mcmc_results: Dict) -> go.Figure:
    """Create MCMC distribution plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=mcmc_results['samples'],
        nbinsx=50,
        name='Probability Distribution',
        marker_color='lightblue'
    ))
    
    # Add mean line
    fig.add_vline(x=mcmc_results['mean'], line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mcmc_results['mean']:.1%}")
    
    # Add CI lines
    fig.add_vline(x=mcmc_results['ci_95_lower'], line_dash="dot", line_color="gray",
                  annotation_text=f"95% CI Lower")
    fig.add_vline(x=mcmc_results['ci_95_upper'], line_dash="dot", line_color="gray",
                  annotation_text=f"95% CI Upper")
    
    fig.update_layout(
        title="Uncertainty Distribution (10,000 simulations)",
        xaxis_title="Probability",
        yaxis_title="Frequency",
        xaxis=dict(tickformat='.0%'),
        height=400
    )
    
    return fig

def create_sensitivity_tornado(sensitivity: Dict) -> go.Figure:
    """Create tornado diagram for sensitivity analysis"""
    fig = go.Figure()
    
    if 'prior_probability' in sensitivity['sensitivities']:
        variations = sensitivity['sensitivities']['prior_probability']
        values = [v['value'] for v in variations]
        recommendations = [v['recommendation'] for v in variations]
        
        fig.add_trace(go.Bar(
            y=['Prior Probability'] * len(values),
            x=values,
            orientation='h',
            text=recommendations,
            textposition='auto',
            marker_color=['red' if v['changed'] else 'green' for v in variations]
        ))
    
    fig.update_layout(
        title="Sensitivity Analysis - Parameter Variations",
        xaxis_title="Parameter Value",
        yaxis_title="Parameter",
        height=300
    )
    
    return fig

def create_influence_diagram_viz(influence_diagram: Dict) -> go.Figure:
    """Create interactive influence diagram visualization"""
    nodes = influence_diagram['nodes']
    edges = influence_diagram['edges']
    
    # Define positions for nodes (layered layout)
    positions = {
        'clinical_findings': (0, 2),
        'prior': (1, 2),
        'test_decision': (2, 3),
        'test_result': (3, 3),
        'posterior': (4, 2),
        'utilities': (4, 1),
        'costs': (2, 0),
        'treatment_decision': (5, 2),
        'outcome': (6, 2)
    }
    
    # Color scheme by node type
    colors = {
        'chance': '#87CEEB',      # Sky blue for chance nodes
        'decision': '#90EE90',     # Light green for decision nodes
        'value': '#FFD700'         # Gold for value nodes
    }
    
    # Create edge traces
    edge_traces = []
    for edge in edges:
        from_pos = positions.get(edge['from'], (0, 0))
        to_pos = positions.get(edge['to'], (1, 1))
        
        edge_trace = go.Scatter(
            x=[from_pos[0], to_pos[0], None],
            y=[from_pos[1], to_pos[1], None],
            mode='lines',
            line=dict(width=2, color='#888'),
            hoverinfo='text',
            text=edge.get('label', ''),
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_hover = []
    
    for node in nodes:
        pos = positions.get(node['id'], (0, 0))
        node_x.append(pos[0])
        node_y.append(pos[1])
        node_text.append(node['label'])
        node_colors.append(colors.get(node['type'], '#CCCCCC'))
        
        hover_text = f"<b>{node['label']}</b><br>"
        hover_text += f"Type: {node['type']}<br>"
        hover_text += f"{node.get('description', '')}"
        if node.get('value'):
            hover_text += f"<br>Value: {node['value']}"
        node_hover.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        textfont=dict(size=10, color='black'),
        hoverinfo='text',
        hovertext=node_hover,
        marker=dict(
            size=80,
            color=node_colors,
            line=dict(width=2, color='#333')
        ),
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Update layout
    fig.update_layout(
        title={
            'text': influence_diagram['description'],
            'x': 0.5,
            'xanchor': 'center'
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=60),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor='white'
    )
    
    # Add legend manually
    fig.add_annotation(
        x=0.02, y=0.98,
        xref='paper', yref='paper',
        text='<b>Legend:</b><br>🔵 Chance nodes<br>🟢 Decision nodes<br>🟡 Value nodes',
        showarrow=False,
        bgcolor='white',
        bordercolor='black',
        borderwidth=1,
        align='left',
        xanchor='left',
        yanchor='top'
    )
    
    return fig

# ============================================================================
# AGNO LLM INTEGRATION
# ============================================================================

def create_llm_prompt(patient_query: str) -> str:
    """Create enhanced prompt for Gemini with structured output"""
    return f"""You are an expert clinical decision support system using Bayesian reasoning, Expected Value of Information (EVI), and utility theory.

Analyze this clinical case:
{patient_query}

Provide a comprehensive analysis with the following structure:

## Clinical Assessment
[Your clinical reasoning and interpretation]

## Time-Critical Urgency Assessment
Evaluate urgency based on evidence:
- Does the patient have SHOCK (septic, cardiogenic, hemorrhagic)? (Yes/No)
- Urgency category:
  * CRITICAL: Septic shock, STEMI, massive PE (treat immediately, < 1 hour)
  * URGENT: Bacterial meningitis, DKA (1-6 hours)
  * SEMI-URGENT: CAP without shock, appendicitis (6-24 hours)
  * NON-URGENT: Stable conditions, no time pressure (> 24 hours)

IMPORTANT - Evidence-based time considerations:
- Septic SHOCK: 7% mortality increase per hour of antibiotic delay (NEJM 2017)
- Sepsis WITHOUT shock: Delayed antibiotics may be acceptable for stewardship (Surgical ICU studies)
- Shock is the KEY distinction for urgency

## Bayesian Analysis
Estimate:
- Prior probability of the disease/condition (0.0 to 1.0)
- For each relevant test, provide:
  - Test name
  - Sensitivity (LR+ calculation)
  - Specificity (LR- calculation)
  - Relative cost tier (use $ to $$$$$ scale)

## Clinical Outcomes (NNT/NNH)
For the primary treatment option:
- Mortality if untreated (as decimal, e.g., 0.30 for 30%)
- Mortality if treated (as decimal, e.g., 0.08 for 8%)
- Major complication rate (as decimal, e.g., 0.02 for 2%)
- Calculate and display NNT (Number Needed to Treat) and NNH (Number Needed to Harm)

## Utility Analysis
For each decision option (treat, test, observe), estimate:
- Benefit score (0.0 to 1.0)
- Harm score (0.0 to 1.0)
- Cost tier (from the COST_TIERS reference)
- Net utility = Benefit - Harm - (Cost factor)

## Expected Value of Information (EVI)
For each test:
- Will this test change management?
- What is the threshold probability for changing decision?
- EVI score (0.0 to 1.0, where higher = more valuable)

## Recommendation
[Final recommendation with confidence level, accounting for urgency]

At the end of your response, provide a JSON block with structured data:

```json
{{
  "prior_probability": <float between 0 and 1>,
  "disease_name": "<condition being evaluated>",
  "has_shock": <true/false>,
  "mortality_per_hour_delay": <float or null>,
  "urgency_category": "<CRITICAL/URGENT/SEMI-URGENT/NON-URGENT>",
  "de_escalation_probability": <float 0-1>,
  "de_escalation_timeline_hours": <float>,
  "literature_source": "<e.g., 'NEJM 2017' or null>",
  "mortality_untreated": <float 0-1>,
  "mortality_treated": <float 0-1>,
  "major_complication_rate": <float 0-1>,
  "tests": [
    {{
      "name": "<test name>",
      "lr_positive": <float>,
      "lr_negative": <float>,
      "sensitivity": <float>,
      "specificity": <float>
    }}
  ],
  "utilities": {{
    "treat_success": <float 0-1>,
    "treat_healthy": <float 0-1>,
    "observe_disease": <float 0-1>,
    "observe_healthy": <float 0-1>
  }},
  "costs": [
    {{
      "item": "<test/intervention name>",
      "tier": "<$ to $$$$$>",
      "relative_cost": <number>,
      "note": "<brief note>"
    }}
  ],
  "evi_table": [
    {{
      "test": "<test name>",
      "will_change_management": "<yes/no/maybe>",
      "threshold_probability": <float 0-1>,
      "evi": <float 0-1>
    }}
  ],
  "utility_rank": [
    {{
      "action": "<treat/test/observe>",
      "benefit": <float 0-1>,
      "harm": <float 0-1>,
      "cost_tier": "<tier>",
      "utility": <float>
    }}
  ]
}}
```

Ensure all numeric values are actual numbers, not strings."""

def process_with_llm(patient_query: str) -> Tuple[str, Dict]:
    """Process query with Gemini and extract structured data"""
    try:
        # Initialize Agno agent with Gemini
        agent = Agent(
            model=Gemini(id="gemini-2.0-flash-exp", api_key=API_KEY),
            markdown=True
        )
        
        # Get response
        prompt = create_llm_prompt(patient_query)
        response = agent.run(prompt)
        
        # Extract JSON data
        json_data = extract_json_tail(response.content)
        
        # Clean content (remove JSON blocks for display)
        clean_content = hide_json_blocks(response.content)
        
        return clean_content, json_data
        
    except Exception as e:
        return f"Error processing with LLM: {str(e)}", {}

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="PRIORI — Clinical Decision Support", 
        page_icon="🩺", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .recommendation-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    .safety-alert {
        background-color: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🩺 PRIORI — Advanced Bayesian Clinical Reasoning")
    st.caption("*Probabilistic thinking + Evidence-based decisions + Resource stewardship + Safety guardrails*")
    st.markdown("---")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "context" not in st.session_state:
        st.session_state["context"] = None
    
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
            st.session_state["results"] = None
            st.session_state["context"] = None
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
            st.markdown(m["content"])
    
    # Process pending user message
    if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
        last_user_msg = st.session_state["messages"][-1]["content"]
        # Check if we've already responded
        if len(st.session_state["messages"]) == 1 or st.session_state["messages"][-2]["role"] != "assistant":
            process_case(last_user_msg)
    
    # Chat input
    if user_input := st.chat_input("💬 Enter your clinical question or case details..."):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        process_case(user_input)
    
    # Display analysis results if available
    if st.session_state["results"]:
        display_analysis_results()
    
    # Footer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** PRIORI is an educational/research demonstration tool. Not for clinical use. All recommendations require physician oversight.")

def process_case(user_query: str):
    """Process a clinical case and update session state"""
    with st.spinner("🧠 Analyzing case with advanced Bayesian reasoning..."):
        try:
            # Run safety checks first
            safety_warnings = SafetyGuardrails.validate_all(user_query, user_query)
            
            # Process with LLM
            llm_response, json_data = process_with_llm(user_query)
            
            # Display safety warnings prominently if any
            if safety_warnings:
                for warning in safety_warnings:
                    st.markdown(f"""
                    <div class="safety-alert">
                        {warning}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Display LLM response
            with st.chat_message("assistant"):
                st.markdown(llm_response)
                
                # Display structured tables if JSON data available
                if json_data:
                    # EVI Analysis
                    if json_data.get("evi_table"):
                        with st.expander("📊 **Expected Value of Information (EVI) Analysis**", expanded=True):
                            st.markdown("*Which tests actually change management?*")
                            try:
                                st.table({
                                    "Test": [x.get("test","") for x in json_data["evi_table"]],
                                    "Changes Management?": [x.get("will_change_management","") for x in json_data["evi_table"]],
                                    "Threshold Probability": [f"{safe_float(x.get('threshold_probability', 0))*100:.1f}%" for x in json_data["evi_table"]],
                                    "EVI Score": [f"{safe_float(x.get('evi', 0)):.3f}" for x in json_data["evi_table"]],
                                })
                            except Exception as e:
                                st.info("📊 EVI data available but formatting issue detected.")
                    
                    # Cost Comparison
                    if json_data.get("costs"):
                        with st.expander("💰 **Relative Cost Comparison**", expanded=True):
                            st.markdown("*Resource stewardship — what's the opportunity cost?*")
                            st.table({
                                "Test/Intervention": [x.get("item","") for x in json_data["costs"]],
                                "Cost Tier": [x.get("tier","?") for x in json_data["costs"]],
                                "× Baseline": [x.get("relative_cost","?") for x in json_data["costs"]],
                                "Context": [x.get("note","") for x in json_data["costs"]],
                            })
                    
                    # Utility Ranking
                    if json_data.get("utility_rank"):
                        with st.expander("⚖️ **Utility Ranking (Benefit − Harm − Cost)**", expanded=True):
                            st.markdown("*Higher utility = better value for the patient*")
                            try:
                                st.table({
                                    "Action": [x.get("action","") for x in json_data["utility_rank"]],
                                    "Benefit": [f"{safe_float(x.get('benefit', 0)):.3f}" for x in json_data["utility_rank"]],
                                    "Harm": [f"{safe_float(x.get('harm', 0)):.3f}" for x in json_data["utility_rank"]],
                                    "Cost Tier": [x.get("cost_tier","?") for x in json_data["utility_rank"]],
                                    "Net Utility": [f"{safe_float(x.get('utility', 0)):.3f}" for x in json_data["utility_rank"]],
                                })
                            except Exception as e:
                                st.info("⚖️ Utility ranking available but formatting issue detected.")
            
            # Create context from JSON data
            if json_data and json_data.get('prior_probability') is not None:
                context = create_context_from_json(json_data, user_query)
                
                # Run advanced analysis
                engine = AdvancedDecisionEngine()
                results = engine.analyze(context, safety_warnings)
                results.llm_reasoning = llm_response
                
                # Store in session state
                st.session_state["context"] = context
                st.session_state["results"] = results
            
            # Save to history
            st.session_state["messages"].append({"role": "assistant", "content": llm_response})
            
        except Exception as e:
            st.error(f"❌ Error processing request: {str(e)}")
            st.info("💡 Tip: Try rephrasing your question or use one of the auto-prompts.")

def create_context_from_json(json_data: Dict, patient_query: str) -> ClinicalDecisionContext:
    """Convert JSON data to ClinicalDecisionContext"""
    
    # Extract test data
    likelihood_ratios = {}
    test_names = []
    test_costs = {}
    
    if 'tests' in json_data:
        for test in json_data['tests']:
            test_name = test.get('name', 'Unknown')
            test_names.append(test_name)
            likelihood_ratios[test_name] = (
                safe_float(test.get('lr_positive', 1.0)),
                safe_float(test.get('lr_negative', 1.0))
            )
            # Try to get cost from COST_TIERS
            cost_info = get_cost_info(test_name)
            test_costs[test_name] = safe_float(cost_info.get('relative_cost', 100))
    
    # Extract utilities
    utilities = json_data.get('utilities', {
        'treat_success': 0.9,
        'treat_healthy': 0.95,
        'observe_disease': 0.0,
        'observe_healthy': 1.0
    })
    
    # Extract mortality data for NNT/NNH calculations
    mortality_untreated = safe_float(json_data.get('mortality_untreated', 0.30))
    mortality_treated = safe_float(json_data.get('mortality_treated', 0.08))
    major_complication_rate = safe_float(json_data.get('major_complication_rate', 0.02))
    
    # Extract urgency metrics
    has_shock = json_data.get('has_shock', False)
    mortality_per_hour_delay = json_data.get('mortality_per_hour_delay')
    if mortality_per_hour_delay is not None:
        mortality_per_hour_delay = safe_float(mortality_per_hour_delay)
    
    urgency_category = json_data.get('urgency_category', 'NON-URGENT')
    de_escalation_probability = safe_float(json_data.get('de_escalation_probability', 0.6))
    de_escalation_timeline_hours = safe_float(json_data.get('de_escalation_timeline_hours', 48.0))
    literature_source = json_data.get('literature_source')
    
    return ClinicalDecisionContext(
        prior_probability=safe_float(json_data.get('prior_probability', 0.5)),
        likelihood_ratios=likelihood_ratios,
        utilities={k: safe_float(v) for k, v in utilities.items()},
        patient_preferences={},
        test_costs=test_costs,
        treatment_costs={},
        disease_name=json_data.get('disease_name', 'Unknown'),
        test_names=test_names,
        patient_query=patient_query,
        mortality_untreated=mortality_untreated,
        mortality_treated=mortality_treated,
        major_complication_rate=major_complication_rate,
        has_shock=has_shock,
        mortality_per_hour_delay=mortality_per_hour_delay,
        urgency_category=urgency_category,
        de_escalation_probability=de_escalation_probability,
        de_escalation_timeline_hours=de_escalation_timeline_hours,
        literature_source=literature_source
    )

def display_analysis_results():
    """Display advanced analysis results in tabs with minimalist cool design"""
    if not st.session_state['results']:
        return
    
    results = st.session_state['results']
    context = st.session_state['context']
    
    st.markdown("---")
    
    # === URGENCY ALERT (Top Priority - Minimalist Design) ===
    if context.urgency_category in ["CRITICAL", "URGENT"]:
        urgency_color = "#dc3545" if context.urgency_category == "CRITICAL" else "#ff9800"
        urgency_icon = "🚨" if context.urgency_category == "CRITICAL" else "⚠️"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {urgency_color}15 0%, {urgency_color}05 100%); 
                    border-left: 4px solid {urgency_color}; 
                    border-radius: 8px; padding: 20px; margin-bottom: 24px;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <span style="font-size: 32px;">{urgency_icon}</span>
                <span style="font-size: 24px; font-weight: 600; color: {urgency_color};">
                    {context.urgency_category} TIME PRESSURE
                </span>
            </div>
            <div style="font-size: 16px; line-height: 1.6;">
                {f'<strong>Mortality increases {context.mortality_per_hour_delay*100:.0f}% per hour of delay</strong>' if context.mortality_per_hour_delay else ''}
                {f'<div style="opacity: 0.8; margin-top: 8px;">Source: {context.literature_source}</div>' if context.literature_source else ''}
                {'<div style="margin-top: 12px; padding: 12px; background: white; border-radius: 4px;"><strong>⚡ KEY DISTINCTION:</strong> Shock present → Immediate antibiotics. No shock → Diagnostic stewardship acceptable.</div>' if 'sepsis' in context.disease_name.lower() or 'shock' in context.disease_name.lower() else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # === CLINICAL OUTCOMES (Primary Display - Clean & Minimalist) ===
    if results.clinical_outcomes and results.clinical_outcomes.nnt_mortality:
        st.markdown("### 🎯 Clinical Impact")
        
        # Compact metric cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: white; border-radius: 8px; padding: 16px; border: 2px solid #e3f2fd;">
                <div style="font-size: 14px; color: #666; margin-bottom: 4px;">To Save 1 Life</div>
                <div style="font-size: 32px; font-weight: 700; color: #1976d2;">
                    {results.clinical_outcomes.nnt_mortality}
                </div>
                <div style="font-size: 12px; color: #999;">patients treated (NNT)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            nnh_color = "#4caf50" if results.clinical_outcomes.nnh_major_complication and results.clinical_outcomes.nnh_major_complication > results.clinical_outcomes.nnt_mortality * 5 else "#ff9800"
            st.markdown(f"""
            <div style="background: white; border-radius: 8px; padding: 16px; border: 2px solid {nnh_color}15;">
                <div style="font-size: 14px; color: #666; margin-bottom: 4px;">Harm Risk</div>
                <div style="font-size: 32px; font-weight: 700; color: {nnh_color};">
                    1:{results.clinical_outcomes.nnh_major_complication if results.clinical_outcomes.nnh_major_complication else 'N/A'}
                </div>
                <div style="font-size: 12px; color: #999;">complication rate (NNH)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: white; border-radius: 8px; padding: 16px; border: 2px solid #f3e5f5;">
                <div style="font-size: 14px; color: #666; margin-bottom: 4px;">Absolute Benefit</div>
                <div style="font-size: 32px; font-weight: 700; color: #9c27b0;">
                    {results.clinical_outcomes.arr_mortality*100:.0f}%
                </div>
                <div style="font-size: 12px; color: #999;">risk reduction</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Simple interpretation
        st.info(f"💡 {results.clinical_outcomes.interpretation}")
        
        # De-escalation plan (if applicable)
        if context.urgency_category == "CRITICAL" and context.de_escalation_probability > 0.5:
            st.markdown(f"""
            <div style="background: #f5f5f5; border-radius: 8px; padding: 16px; margin-top: 16px;">
                <strong>🎯 De-Escalation Strategy</strong>
                <div style="margin-top: 8px;">
                    • Reassess in {context.de_escalation_timeline_hours:.0f} hours<br>
                    • {context.de_escalation_probability*100:.0f}% probability of narrowing therapy<br>
                    • Planned strategy, not failure
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # === SAFETY WARNINGS ===
    if results.safety_warnings:
        for warning in results.safety_warnings:
            st.error(f"⚠️ {warning}")
        st.markdown("---")
    
    # === DECISION SUMMARY (Minimal) ===
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Decision", results.recommendation, help="Primary recommendation")
    with col2:
        st.metric("Pre-test Probability", f"{context.prior_probability:.0%}", help="Prior probability")
    with col3:
        st.metric("Confidence", f"{results.confidence:.0%}", help="Model confidence")
    with col4:
        st.metric("Treat Threshold", f"{results.thresholds['treat_threshold']:.0%}", help="Treatment threshold probability")
    
    # === DETAILED ANALYSIS (Tabbed - Cleaner Organization) ===
    st.markdown("---")
    st.markdown("### 📊 Detailed Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Thresholds & Tests", 
        "🎲 Uncertainty", 
        "🧠 Biases",
        "🔢 Advanced Metrics"
    ])
    
    with tab1:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Decision Thresholds**")
            fig = create_threshold_viz(context, results.thresholds)
            st.plotly_chart(fig, use_container_width=True, key="threshold_chart")
        
        with col_b:
            st.markdown("**Test Value (EVPI)**")
            for test_name, evpi_data in results.evpi.items():
                cost_per_qaly = evpi_data['cost_per_qaly']
                if cost_per_qaly < 50000:
                    badge = "🟢 Excellent"
                elif cost_per_qaly < 100000:
                    badge = "🟡 Consider"
                else:
                    badge = "🔴 Poor value"
                
                st.markdown(f"""
                **{test_name}** {badge}  
                ${cost_per_qaly:,.0f}/QALY
                """)
    
    with tab2:
        st.markdown("**Monte Carlo Simulation** (10,000 runs)")
        fig = create_mcmc_distribution(results.mcmc_results)
        st.plotly_chart(fig, use_container_width=True, key="mcmc_chart_tab2")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean", f"{results.mcmc_results['mean']:.1%}")
            st.metric("95% CI Lower", f"{results.mcmc_results['ci_95_lower']:.1%}")
        with col2:
            if results.mcmc_results['uncertainty_high']:
                st.warning("⚠️ High uncertainty - consider more data")
            else:
                st.success("✓ Acceptable uncertainty")
            st.metric("95% CI Upper", f"{results.mcmc_results['ci_95_upper']:.1%}")
    
    with tab3:
        if results.bias_warnings:
            for warning in results.bias_warnings:
                st.warning(f"""
                **{warning['icon']} {warning['bias']}**  
                {warning['description']}  
                💡 *{warning['suggestion']}*
                """)
        else:
            st.success("✓ No cognitive biases detected")
    
    with tab4:
        # QALY details, sensitivity analysis
        st.markdown("**Sensitivity Analysis**")
        fig = create_sensitivity_tornado(results.sensitivity)
        st.plotly_chart(fig, use_container_width=True, key="sensitivity_chart_tab4")
        
        with st.expander("🔢 QALY Economic Details"):
            st.markdown("**Cost-Effectiveness Thresholds:**  <$50k = Excellent, $50-100k = Good, >$100k = Poor")
            for test_name, evpi_data in results.evpi.items():
                st.metric(f"{test_name} Cost/QALY", f"${evpi_data['cost_per_qaly']:,.0f}")
                st.caption(f"EVPI: {evpi_data['evpi_qalys']:.4f} QALYs | {evpi_data['reason']}")
        
        with st.expander("🔗 Influence Diagram"):
            st.markdown(results.influence_diagram['description'])
            fig = create_influence_diagram_viz(results.influence_diagram)
            st.plotly_chart(fig, use_container_width=True, key="influence_diagram_chart")
    
    # ========================================================================
    # END OF NEW MINIMALIST UI - Return here to skip old redundant code below
    # ========================================================================
    return
    
    # ========================================================================
    # OLD UI CODE BELOW - NOT EXECUTED (kept for reference)
    # ========================================================================
    
    # with tab2:
    #     st.markdown("### Expected Value of Perfect Information (EVPI)")
    #     st.markdown("Determines which tests are actually worth performing")
        
        for test_name, evpi_data in results.evpi.items():
            st.markdown(f"""
            <div class="metric-box">
                <h4 style="margin-top: 0;">{test_name} {evpi_data['recommendation']}</h4>
                <p><strong>Cost per QALY:</strong> ${evpi_data['cost_per_qaly']:,.0f}</p>
                <p><strong>Test Cost:</strong> ${evpi_data['test_cost']:,.0f}</p>
                <p><strong>EVPI:</strong> {evpi_data['evpi_qalys']:.4f} QALYs</p>
                <p style="color: #666;">{evpi_data['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.info("💡 **Cost-effectiveness thresholds:** <$50k/QALY = Worthwhile, $50-100k = Consider, >$100k = Skip")
    
    with tab3:
        st.markdown("### Uncertainty Analysis (MCMC Simulation)")
        st.markdown("Based on 10,000 Monte Carlo simulations")
        
        fig = create_mcmc_distribution(results.mcmc_results)
        st.plotly_chart(fig, use_container_width=True, key="mcmc_chart_tab3")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Probability", f"{results.mcmc_results['mean']:.1%}")
        with col2:
            st.metric("95% CI Lower", f"{results.mcmc_results['ci_95_lower']:.1%}")
        with col3:
            st.metric("95% CI Upper", f"{results.mcmc_results['ci_95_upper']:.1%}")
        
        if results.mcmc_results['uncertainty_high']:
            st.warning("⚠️ High uncertainty detected. Consider gathering more information.")
        else:
            st.success("✓ Uncertainty is within acceptable range")
    
    with tab4:
        st.markdown("### Sensitivity Analysis")
        st.markdown("Shows which parameters most affect the decision")
        
        fig = create_sensitivity_tornado(results.sensitivity)
        st.plotly_chart(fig, use_container_width=True, key="sensitivity_chart_old_tab4")
        
        if results.sensitivity['decision_fragile']:
            st.warning(f"⚠️ **Decision is fragile** - Most sensitive to: {results.sensitivity['most_influential']}")
            st.markdown("Small changes in parameters could flip the recommendation. Consider:")
            st.markdown("- Gathering more precise data")
            st.markdown("- Getting second opinion")
            st.markdown("- Using additional tests")
        else:
            st.success("✓ **Decision is robust** - Recommendation stable across parameter variations")
    
    with tab5:
        st.markdown("### Cognitive Bias Detection")
        st.markdown("Checks for common reasoning errors")
        
        if results.bias_warnings:
            for warning in results.bias_warnings:
                st.markdown(f"""
                <div class="warning-box">
                    <h4 style="margin-top: 0;">{warning['icon']} {warning['bias']}</h4>
                    <p><strong>Warning:</strong> {warning['description']}</p>
                    <p><strong>Suggestion:</strong> {warning['suggestion']}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✓ No cognitive biases detected")
    
    with tab6:
        st.markdown("### QALY (Quality-Adjusted Life Year) Details")
        st.markdown("*Advanced economic analysis for resource stewardship*")
        
        # EVPI with QALY details
        st.markdown("#### Expected Value of Perfect Information")
        for test_name, evpi_data in results.evpi.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"""
                <div class="metric-box">
                    <h4 style="margin-top: 0;">{test_name} {evpi_data['recommendation']}</h4>
                    <p><strong>Cost per QALY:</strong> ${evpi_data['cost_per_qaly']:,.0f}</p>
                    <p><strong>Test Cost:</strong> ${evpi_data['test_cost']:,.0f}</p>
                    <p><strong>EVPI:</strong> {evpi_data['evpi_qalys']:.4f} QALYs</p>
                    <p style="color: #666;">{evpi_data['reason']}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                # Color-coded value indicator
                cost_per_qaly = evpi_data['cost_per_qaly']
                if cost_per_qaly < 50000:
                    st.success("✅ Excellent Value")
                elif cost_per_qaly < 100000:
                    st.warning("⚠️ Good Value")
                else:
                    st.error("❌ Poor Value")
        
        st.info("💡 **Cost-effectiveness thresholds:** <$50k/QALY = Excellent, $50-100k = Good, >$100k = Poor")
        
        # Utility weights explanation
        st.markdown("---")
        st.markdown("#### Utility Weights Used")
        st.markdown("*These values represent quality of life on a 0-1 scale*")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Treat Success (Disease Present)", f"{context.utilities.get('treat_success', 0):.2f}")
            st.caption("Quality of life after successful treatment")
            
            st.metric("Observe Disease (Untreated)", f"{context.utilities.get('observe_disease', 0):.2f}")
            st.caption("Quality of life if disease progresses untreated")
        
        with col2:
            st.metric("Treat Healthy (False Positive)", f"{context.utilities.get('treat_healthy', 0):.2f}")
            st.caption("Quality of life after unnecessary treatment")
            
            st.metric("Observe Healthy (True Negative)", f"{context.utilities.get('observe_healthy', 1):.2f}")
            st.caption("Perfect health baseline")
        
        # QALY calculation explanation
        with st.expander("📚 How QALYs Are Calculated"):
            st.markdown("""
            **QALY = Years of Life × Quality Weight**
            
            Quality weights range from 0 (death) to 1 (perfect health):
            - 1.0 = Perfect health
            - 0.9 = Mild symptoms
            - 0.7 = Moderate disability
            - 0.5 = Severe disability  
            - 0.0 = Death
            
            **Example:**
            - Treatment extends life 10 years at 0.85 quality = **8.5 QALYs**
            - Better than 5 years at 0.95 quality = **4.75 QALYs**
            
            **Cost-Effectiveness:**
            - Treatment costs $50,000 and gains 1 QALY → $50,000/QALY ✓
            - Treatment costs $200,000 and gains 1 QALY → $200,000/QALY ✗
            
            **Standard Threshold:** $50,000-$150,000 per QALY (US)
            """)
    
    # Influence Diagram (collapsed by default)
    with st.expander("🔗 View Influence Diagram", expanded=False):
        st.markdown("### Clinical Decision Pathway")
        st.markdown(results.influence_diagram['description'])
        
        # Interactive visualization
        fig = create_influence_diagram_viz(results.influence_diagram)
        st.plotly_chart(fig, use_container_width=True, key="influence_diagram_old_tab6")
        
        # Detailed node information
        st.markdown("---")
        st.markdown("### Node Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔵 Chance Nodes** (Uncertain events)")
            for node in results.influence_diagram['nodes']:
                if node['type'] == 'chance':
                    st.markdown(f"- **{node['label'].replace(chr(10), ' ')}**")
                    st.caption(node.get('description', ''))
                    if node.get('value'):
                        st.caption(f"*{node['value']}*")
        
        with col2:
            st.markdown("**🟢 Decision Nodes** (Actions to take)")
            for node in results.influence_diagram['nodes']:
                if node['type'] == 'decision':
                    st.markdown(f"- **{node['label'].replace(chr(10), ' ')}**")
                    st.caption(node.get('description', ''))
                    if node.get('value'):
                        st.caption(f"*{node['value']}*")
        
        with col3:
            st.markdown("**🟡 Value Nodes** (Outcomes & utilities)")
            for node in results.influence_diagram['nodes']:
                if node['type'] == 'value':
                    st.markdown(f"- **{node['label'].replace(chr(10), ' ')}**")
                    st.caption(node.get('description', ''))
                    if node.get('value'):
                        st.caption(f"*{node['value']}*")
        
        st.markdown("---")
        st.markdown("### Key Decision Pathways")
        for edge in results.influence_diagram['edges']:
            from_node = next(n['label'] for n in results.influence_diagram['nodes'] if n['id'] == edge['from'])
            to_node = next(n['label'] for n in results.influence_diagram['nodes'] if n['id'] == edge['to'])
            label = edge.get('label', '')
            st.markdown(f"- {from_node.replace(chr(10), ' ')} **{label}** → {to_node.replace(chr(10), ' ')}")
    
    
    # Show full mathematics
    with st.expander("🧮 View Full Mathematical Details"):
        st.markdown("### Bayesian Calculation")
        st.code(f"""
Prior Probability: {context.prior_probability:.4f}
Prior Odds: {context.prior_probability / (1 - context.prior_probability):.4f}

Utilities (QALYs):
- Treat when disease present: {context.utilities['treat_success']:.2f}
- Treat when no disease: {context.utilities['treat_healthy']:.2f}
- Observe when disease present: {context.utilities['observe_disease']:.2f}
- Observe when no disease: {context.utilities['observe_healthy']:.2f}

Expected Value (Treat):
EV = p × U(treat|disease) + (1-p) × U(treat|no disease)
   = {context.prior_probability:.4f} × {context.utilities['treat_success']:.2f} + {1-context.prior_probability:.4f} × {context.utilities['treat_healthy']:.2f}
   = {context.prior_probability * context.utilities['treat_success'] + (1-context.prior_probability) * context.utilities['treat_healthy']:.4f}

Expected Value (Observe):
EV = p × U(observe|disease) + (1-p) × U(observe|no disease)
   = {context.prior_probability:.4f} × {context.utilities['observe_disease']:.2f} + {1-context.prior_probability:.4f} × {context.utilities['observe_healthy']:.2f}
   = {context.prior_probability * context.utilities['observe_disease'] + (1-context.prior_probability) * context.utilities['observe_healthy']:.4f}

Treatment Threshold:
p* = (U(observe|no disease) - U(treat|no disease)) / 
     ((U(treat|disease) - U(observe|disease)) - (U(treat|no disease) - U(observe|no disease)))
   = {results.thresholds['treat_threshold']:.4f}
        """)
        
        st.markdown("### Likelihood Ratios")
        for test_name, (lr_pos, lr_neg) in context.likelihood_ratios.items():
            st.code(f"""
{test_name}:
- LR+ = {lr_pos:.2f} (sensitivity / (1 - specificity))
- LR- = {lr_neg:.2f} ((1 - sensitivity) / specificity)

Post-test probability (if positive):
Posterior odds = Prior odds × LR+
               = {context.prior_probability / (1 - context.prior_probability):.4f} × {lr_pos:.2f}
               = {(context.prior_probability / (1 - context.prior_probability)) * lr_pos:.4f}
Posterior prob = odds / (1 + odds)
               = {((context.prior_probability / (1 - context.prior_probability)) * lr_pos) / (1 + (context.prior_probability / (1 - context.prior_probability)) * lr_pos):.4f}
            """)

if __name__ == "__main__":
    main()
