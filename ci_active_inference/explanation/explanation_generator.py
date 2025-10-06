"""Enhanced explanation generation for Active Inference agent suggestions."""

import math
from typing import Dict, List, Any, Optional, Tuple
from ..utils.matrix_utils import SimpleActiveInferenceMatrices
from .efe_decomposer import EFEDecomposer
from .causality_analyzer import CausalityAnalyzer
from .explanation_templates import ExplanationTemplates


class ExplanationGenerator:
    """
    Advanced explanation generator for Active Inference agent suggestions.
    
    Provides detailed, human-readable explanations of why specific actions
    are recommended, including EFE decomposition and causal reasoning.
    """
    
    def __init__(self, matrices: SimpleActiveInferenceMatrices):
        """
        Initialize explanation generator.
        
        Args:
            matrices: Active Inference matrices
        """
        self.matrices = matrices
        self.efe_decomposer = EFEDecomposer(matrices)
        self.causality_analyzer = CausalityAnalyzer(matrices)
        self.templates = ExplanationTemplates()
        
    def generate_detailed_explanation(self, 
                                    current_state: Dict[str, Any],
                                    suggested_action: str,
                                    predicted_default: str,
                                    confidence: float,
                                    efe_breakdown: Dict[str, float],
                                    beliefs: List[List[float]]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for agent suggestion.
        
        Args:
            current_state: Current observed state
            suggested_action: Recommended action
            predicted_default: Predicted default behavior
            confidence: Confidence in suggestion
            efe_breakdown: EFE component breakdown
            beliefs: Current state beliefs
            
        Returns:
            Detailed explanation dictionary
        """
        explanation = {
            "summary": self._generate_summary(suggested_action, predicted_default, confidence),
            "reasoning": self._generate_reasoning(efe_breakdown, beliefs),
            "causal_factors": self._analyze_causal_factors(current_state, beliefs),
            "consequences": self._predict_consequences(suggested_action, beliefs),
            "confidence_explanation": self._explain_confidence(confidence, efe_breakdown),
            "alternative_options": self._analyze_alternatives(beliefs),
            "contextual_factors": self._identify_context_factors(current_state, beliefs)
        }
        
        # Generate natural language explanation
        explanation["natural_language"] = self._generate_natural_language(explanation)
        
        return explanation
    
    def _generate_summary(self, suggested_action: str, 
                         predicted_default: str, 
                         confidence: float) -> str:
        """Generate high-level summary of the suggestion."""
        if confidence > 0.8:
            confidence_desc = "highly confident"
        elif confidence > 0.6:
            confidence_desc = "moderately confident"
        else:
            confidence_desc = "somewhat uncertain"
            
        return self.templates.get_summary_template().format(
            suggested_action=suggested_action,
            predicted_default=predicted_default,
            confidence_desc=confidence_desc
        )
    
    def _generate_reasoning(self, efe_breakdown: Dict[str, float], 
                          beliefs: List[List[float]]) -> Dict[str, Any]:
        """Generate detailed reasoning based on EFE decomposition."""
        reasoning = {
            "efe_analysis": self.efe_decomposer.decompose_efe(efe_breakdown),
            "belief_state": self._describe_belief_state(beliefs),
            "decision_factors": self._identify_decision_factors(efe_breakdown)
        }
        
        return reasoning
    
    def _analyze_causal_factors(self, current_state: Dict[str, Any], 
                              beliefs: List[List[float]]) -> List[Dict[str, Any]]:
        """Analyze causal factors influencing the recommendation."""
        return self.causality_analyzer.identify_causal_factors(current_state, beliefs)
    
    def _predict_consequences(self, action: str, 
                            beliefs: List[List[float]]) -> Dict[str, Any]:
        """Predict likely consequences of taking the suggested action."""
        # Encode action and predict next states
        action_idx = self._encode_action(action)
        if action_idx is None:
            return {"error": "Unknown action"}
        
        next_beliefs = self._predict_next_beliefs(beliefs, action_idx)
        
        consequences = {
            "immediate_effects": self._describe_immediate_effects(next_beliefs),
            "probability_changes": self._calculate_probability_changes(beliefs, next_beliefs),
            "outcome_scenarios": self._generate_outcome_scenarios(next_beliefs)
        }
        
        return consequences
    
    def _explain_confidence(self, confidence: float, 
                          efe_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """Explain why the agent has this level of confidence."""
        risk = efe_breakdown.get('risk', 0.5)
        ambiguity = efe_breakdown.get('ambiguity', 0.5)
        
        factors = []
        
        if confidence > 0.8:
            factors.append("Clear preference satisfaction expected")
        elif confidence < 0.4:
            factors.append("High uncertainty in outcomes")
            
        if risk > 0.7:
            factors.append("High risk of preference mismatch")
        if ambiguity > 0.7:
            factors.append("Uncertain about current state")
            
        explanation = {
            "confidence_level": confidence,
            "contributing_factors": factors,
            "risk_assessment": self._assess_risk(risk),
            "uncertainty_sources": self._identify_uncertainty_sources(ambiguity)
        }
        
        return explanation
    
    def _analyze_alternatives(self, beliefs: List[List[float]]) -> List[Dict[str, Any]]:
        """Analyze alternative action options."""
        alternatives = []
        
        for action in range(len(self.matrices.B)):
            utility = self._calculate_action_utility(beliefs, action)
            alternatives.append({
                "action": self._decode_action(action),
                "expected_utility": utility,
                "pros_cons": self._analyze_action_pros_cons(beliefs, action)
            })
        
        # Sort by utility
        alternatives.sort(key=lambda x: x["expected_utility"], reverse=True)
        
        return alternatives[:3]  # Top 3 alternatives
    
    def _identify_context_factors(self, current_state: Dict[str, Any], 
                                beliefs: List[List[float]]) -> Dict[str, Any]:
        """Identify important contextual factors affecting the decision."""
        context = {
            "time_factors": self._analyze_temporal_context(current_state),
            "environmental_factors": self._analyze_environment(current_state),
            "behavioral_patterns": self._analyze_behavioral_patterns(beliefs),
            "preference_alignment": self._analyze_preference_alignment(beliefs)
        }
        
        return context
    
    def _generate_natural_language(self, explanation: Dict[str, Any]) -> str:
        """Generate natural language explanation from structured components."""
        sections = []
        
        # Summary
        sections.append(explanation["summary"])
        
        # Main reasoning
        reasoning = explanation["reasoning"]
        if "efe_analysis" in reasoning:
            efe_text = self._efe_to_text(reasoning["efe_analysis"])
            sections.append(f"This recommendation is based on: {efe_text}")
        
        # Causal factors
        if explanation["causal_factors"]:
            causal_text = self._causal_factors_to_text(explanation["causal_factors"])
            sections.append(f"Key influencing factors: {causal_text}")
        
        # Confidence explanation
        conf_exp = explanation["confidence_explanation"]
        if conf_exp["contributing_factors"]:
            factors_text = ", ".join(conf_exp["contributing_factors"])
            sections.append(f"Confidence factors: {factors_text}")
        
        # Consequences
        consequences = explanation["consequences"]
        if "immediate_effects" in consequences:
            effects_text = consequences["immediate_effects"]
            sections.append(f"Expected outcomes: {effects_text}")
        
        return " ".join(sections)
    
    # Helper methods
    def _describe_belief_state(self, beliefs: List[List[float]]) -> Dict[str, Any]:
        """Describe the current belief state in human terms."""
        descriptions = {}
        
        factor_names = ["Energy", "Mood", "Context"]  # Default names
        
        for i, factor_beliefs in enumerate(beliefs):
            if i < len(factor_names):
                max_idx = factor_beliefs.index(max(factor_beliefs))
                confidence = max(factor_beliefs)
                
                state_names = ["Low", "Medium", "High"]  # Default state names
                if max_idx < len(state_names):
                    descriptions[factor_names[i]] = {
                        "state": state_names[max_idx],
                        "confidence": confidence
                    }
        
        return descriptions
    
    def _identify_decision_factors(self, efe_breakdown: Dict[str, float]) -> List[str]:
        """Identify key factors driving the decision."""
        factors = []
        
        risk = efe_breakdown.get('risk', 0.5)
        ambiguity = efe_breakdown.get('ambiguity', 0.5)
        
        if risk > 0.6:
            factors.append("Preference mismatch risk")
        if ambiguity > 0.6:
            factors.append("State uncertainty")
        if risk < 0.3 and ambiguity < 0.3:
            factors.append("Clear preference alignment")
            
        return factors
    
    def _encode_action(self, action: str) -> Optional[int]:
        """Encode action string to index."""
        action_mapping = {
            "no_action": 0,
            "exercise": 1,
            "work": 2,
            "social": 3,
            "rest": 4,
            "entertainment": 5,
            "food": 6,
            "outdoor": 7,
            "focus": 8
        }
        return action_mapping.get(action)
    
    def _decode_action(self, action_idx: int) -> str:
        """Decode action index to string."""
        action_names = [
            "no_action", "exercise", "work", "social", 
            "rest", "entertainment", "food", "outdoor", "focus"
        ]
        if action_idx < len(action_names):
            return action_names[action_idx]
        return f"action_{action_idx}"
    
    def _predict_next_beliefs(self, current_beliefs: List[List[float]], 
                            action: int) -> List[List[float]]:
        """Predict next beliefs given action."""
        # This is a simplified version - use planning module for full implementation
        next_beliefs = []
        
        for factor in range(len(current_beliefs)):
            next_belief = current_beliefs[factor].copy()  # Simplified
            next_beliefs.append(next_belief)
        
        return next_beliefs
    
    def _calculate_action_utility(self, beliefs: List[List[float]], 
                                action: int) -> float:
        """Calculate utility of an action."""
        # Simplified utility calculation
        utility = 0.0
        
        if hasattr(self.matrices, 'C') and self.matrices.C:
            for modality in range(len(self.matrices.C)):
                if modality < len(beliefs):
                    for obs in range(len(self.matrices.C[modality])):
                        obs_prob = 0.1  # Simplified
                        if obs < len(self.matrices.C[modality]):
                            utility += obs_prob * self.matrices.C[modality][obs]
        
        return utility
    
    def _efe_to_text(self, efe_analysis: Dict[str, Any]) -> str:
        """Convert EFE analysis to readable text."""
        components = []
        
        if efe_analysis.get("risk_component", 0) > 0.5:
            components.append("minimizing preference conflicts")
        if efe_analysis.get("ambiguity_component", 0) > 0.5:
            components.append("reducing uncertainty")
        if not components:
            components.append("optimizing expected outcomes")
            
        return " and ".join(components)
    
    def _causal_factors_to_text(self, factors: List[Dict[str, Any]]) -> str:
        """Convert causal factors to readable text."""
        if not factors:
            return "general behavioral patterns"
        
        factor_names = [f["name"] for f in factors[:3]]  # Top 3
        return ", ".join(factor_names)
    
    # Placeholder methods for advanced features
    def _describe_immediate_effects(self, beliefs): return "Improved state likelihood"
    def _calculate_probability_changes(self, before, after): return {}
    def _generate_outcome_scenarios(self, beliefs): return []
    def _assess_risk(self, risk): return "moderate" if risk > 0.5 else "low"
    def _identify_uncertainty_sources(self, ambiguity): return ["state ambiguity"] if ambiguity > 0.5 else []
    def _analyze_action_pros_cons(self, beliefs, action): return {"pros": [], "cons": []}
    def _analyze_temporal_context(self, state): return "current time context"
    def _analyze_environment(self, state): return "environmental conditions"
    def _analyze_behavioral_patterns(self, beliefs): return "typical patterns"
    def _analyze_preference_alignment(self, beliefs): return "alignment status"