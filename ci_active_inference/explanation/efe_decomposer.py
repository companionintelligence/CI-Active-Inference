"""Expected Free Energy decomposer for detailed analysis of agent decisions."""

import math
from typing import Dict, List, Any, Optional


class EFEDecomposer:
    """
    Decomposes Expected Free Energy into interpretable components.
    
    Provides detailed analysis of risk (preference mismatch) and 
    ambiguity (uncertainty) components of EFE.
    """
    
    def __init__(self, matrices):
        """
        Initialize EFE decomposer.
        
        Args:
            matrices: Active Inference matrices
        """
        self.matrices = matrices
    
    def decompose_efe(self, efe_breakdown: Dict[str, float]) -> Dict[str, Any]:
        """
        Decompose EFE into detailed components.
        
        Args:
            efe_breakdown: Basic EFE breakdown from agent
            
        Returns:
            Detailed EFE analysis
        """
        risk = efe_breakdown.get('risk', 0.5)
        ambiguity = efe_breakdown.get('ambiguity', 0.5)
        
        analysis = {
            "risk_component": {
                "value": risk,
                "interpretation": self._interpret_risk(risk),
                "sources": self._identify_risk_sources(risk),
                "impact": self._assess_risk_impact(risk)
            },
            "ambiguity_component": {
                "value": ambiguity,
                "interpretation": self._interpret_ambiguity(ambiguity),
                "sources": self._identify_ambiguity_sources(ambiguity),
                "impact": self._assess_ambiguity_impact(ambiguity)
            },
            "overall_efe": risk + ambiguity,
            "dominant_component": "risk" if risk > ambiguity else "ambiguity",
            "balance_assessment": self._assess_balance(risk, ambiguity)
        }
        
        return analysis
    
    def calculate_detailed_efe(self, beliefs: List[List[float]], 
                             action: int) -> Dict[str, Any]:
        """
        Calculate detailed EFE breakdown for a specific action.
        
        Args:
            beliefs: Current state beliefs
            action: Action to analyze
            
        Returns:
            Detailed EFE calculation
        """
        # Risk component: preference mismatch
        risk_components = self._calculate_risk_components(beliefs, action)
        
        # Ambiguity component: uncertainty
        ambiguity_components = self._calculate_ambiguity_components(beliefs, action)
        
        return {
            "risk": {
                "total": sum(risk_components.values()),
                "components": risk_components
            },
            "ambiguity": {
                "total": sum(ambiguity_components.values()),
                "components": ambiguity_components
            },
            "total_efe": sum(risk_components.values()) + sum(ambiguity_components.values())
        }
    
    def _interpret_risk(self, risk: float) -> str:
        """Interpret risk component value."""
        if risk > 0.8:
            return "Very high risk of preference mismatch"
        elif risk > 0.6:
            return "Moderate risk of undesired outcomes"
        elif risk > 0.4:
            return "Low risk with some uncertainty"
        else:
            return "Minimal risk, good preference alignment"
    
    def _interpret_ambiguity(self, ambiguity: float) -> str:
        """Interpret ambiguity component value."""
        if ambiguity > 0.8:
            return "Very uncertain about current state"
        elif ambiguity > 0.6:
            return "Moderate uncertainty in state estimation"
        elif ambiguity > 0.4:
            return "Some uncertainty, but manageable"
        else:
            return "High confidence in state estimation"
    
    def _identify_risk_sources(self, risk: float) -> List[str]:
        """Identify sources of risk."""
        sources = []
        
        if risk > 0.6:
            sources.extend([
                "Potential preference conflicts",
                "Uncertain outcome satisfaction",
                "Suboptimal action selection"
            ])
        elif risk > 0.3:
            sources.extend([
                "Minor preference misalignment",
                "Alternative options available"
            ])
        
        return sources
    
    def _identify_ambiguity_sources(self, ambiguity: float) -> List[str]:
        """Identify sources of ambiguity."""
        sources = []
        
        if ambiguity > 0.6:
            sources.extend([
                "Uncertain state observations",
                "Incomplete information",
                "Noisy sensor data"
            ])
        elif ambiguity > 0.3:
            sources.extend([
                "Some observational noise",
                "Partial state information"
            ])
        
        return sources
    
    def _assess_risk_impact(self, risk: float) -> str:
        """Assess impact of risk component."""
        if risk > 0.7:
            return "High impact - may lead to undesired outcomes"
        elif risk > 0.4:
            return "Moderate impact - some caution warranted"
        else:
            return "Low impact - minimal concern"
    
    def _assess_ambiguity_impact(self, ambiguity: float) -> str:
        """Assess impact of ambiguity component."""
        if ambiguity > 0.7:
            return "High impact - decision confidence reduced"
        elif ambiguity > 0.4:
            return "Moderate impact - some uncertainty in outcomes"
        else:
            return "Low impact - confident in state estimation"
    
    def _assess_balance(self, risk: float, ambiguity: float) -> str:
        """Assess balance between risk and ambiguity."""
        diff = abs(risk - ambiguity)
        
        if diff < 0.2:
            return "Balanced - both risk and uncertainty contribute equally"
        elif risk > ambiguity:
            return "Risk-dominated - preference mismatch is main concern"
        else:
            return "Ambiguity-dominated - uncertainty is main issue"
    
    def _calculate_risk_components(self, beliefs: List[List[float]], 
                                 action: int) -> Dict[str, float]:
        """Calculate detailed risk components."""
        components = {}
        
        # Preference mismatch risk
        if hasattr(self.matrices, 'C') and self.matrices.C:
            for modality in range(len(self.matrices.C)):
                modality_risk = 0.0
                for obs in range(len(self.matrices.C[modality])):
                    if self.matrices.C[modality][obs] < 0:  # Negative preference
                        # Calculate probability of this observation
                        obs_prob = self._calculate_observation_probability(
                            beliefs, modality, obs, action
                        )
                        modality_risk += obs_prob * abs(self.matrices.C[modality][obs])
                
                components[f"modality_{modality}_risk"] = modality_risk
        
        return components
    
    def _calculate_ambiguity_components(self, beliefs: List[List[float]], 
                                      action: int) -> Dict[str, float]:
        """Calculate detailed ambiguity components."""
        components = {}
        
        # State uncertainty (entropy)
        for factor in range(len(beliefs)):
            factor_entropy = 0.0
            for prob in beliefs[factor]:
                if prob > 1e-10:
                    factor_entropy -= prob * math.log(prob)
            components[f"factor_{factor}_entropy"] = factor_entropy
        
        # Observation uncertainty
        if hasattr(self.matrices, 'A') and self.matrices.A:
            for modality in range(len(self.matrices.A)):
                obs_entropy = self._calculate_observation_entropy(beliefs, modality)
                components[f"modality_{modality}_entropy"] = obs_entropy
        
        return components
    
    def _calculate_observation_probability(self, beliefs: List[List[float]], 
                                         modality: int, obs: int, 
                                         action: int) -> float:
        """Calculate probability of observation given beliefs and action."""
        prob = 0.0
        
        # Predict next beliefs after action
        next_beliefs = self._predict_next_beliefs(beliefs, action)
        
        # Calculate observation likelihood
        for factor in range(len(next_beliefs)):
            for state in range(len(next_beliefs[factor])):
                if (modality < len(self.matrices.A) and
                    obs < len(self.matrices.A[modality]) and
                    state < len(self.matrices.A[modality][obs])):
                    
                    likelihood = self.matrices.A[modality][obs][state]
                    prob += next_beliefs[factor][state] * likelihood
        
        return prob
    
    def _calculate_observation_entropy(self, beliefs: List[List[float]], 
                                     modality: int) -> float:
        """Calculate entropy of observations for a modality."""
        entropy = 0.0
        
        if modality < len(self.matrices.A):
            for obs in range(len(self.matrices.A[modality])):
                obs_prob = self._calculate_observation_probability(beliefs, modality, obs, 0)
                if obs_prob > 1e-10:
                    entropy -= obs_prob * math.log(obs_prob)
        
        return entropy
    
    def _predict_next_beliefs(self, current_beliefs: List[List[float]], 
                            action: int) -> List[List[float]]:
        """Predict next beliefs given action."""
        next_beliefs = []
        
        for factor in range(len(current_beliefs)):
            next_belief = [0.0] * len(current_beliefs[factor])
            
            for curr_state in range(len(current_beliefs[factor])):
                for next_state in range(len(next_belief)):
                    if (factor < len(self.matrices.B) and 
                        next_state < len(self.matrices.B[factor]) and
                        curr_state < len(self.matrices.B[factor][next_state]) and
                        action < len(self.matrices.B[factor][next_state][curr_state])):
                        
                        transition_prob = self.matrices.B[factor][next_state][curr_state][action]
                        next_belief[next_state] += current_beliefs[factor][curr_state] * transition_prob
            
            next_beliefs.append(next_belief)
        
        return next_beliefs