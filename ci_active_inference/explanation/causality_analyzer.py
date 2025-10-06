"""Causality analyzer for identifying factors influencing agent decisions."""

from typing import Dict, List, Any, Optional
import math


class CausalityAnalyzer:
    """
    Analyzes causal factors influencing Active Inference agent decisions.
    
    Identifies key environmental, temporal, and behavioral factors that
    lead to specific recommendations.
    """
    
    def __init__(self, matrices):
        """
        Initialize causality analyzer.
        
        Args:
            matrices: Active Inference matrices
        """
        self.matrices = matrices
        self.causal_factors = {}
        
    def identify_causal_factors(self, current_state: Dict[str, Any], 
                              beliefs: List[List[float]]) -> List[Dict[str, Any]]:
        """
        Identify causal factors influencing the current recommendation.
        
        Args:
            current_state: Current observed state
            beliefs: Current state beliefs
            
        Returns:
            List of causal factors with importance scores
        """
        factors = []
        
        # Temporal factors
        temporal_factors = self._analyze_temporal_factors(current_state)
        factors.extend(temporal_factors)
        
        # Environmental factors  
        env_factors = self._analyze_environmental_factors(current_state)
        factors.extend(env_factors)
        
        # Behavioral state factors
        behavioral_factors = self._analyze_behavioral_factors(beliefs)
        factors.extend(behavioral_factors)
        
        # Context factors
        context_factors = self._analyze_context_factors(current_state, beliefs)
        factors.extend(context_factors)
        
        # Sort by importance
        factors.sort(key=lambda x: x.get('importance', 0), reverse=True)
        
        return factors[:5]  # Top 5 factors
    
    def analyze_counterfactuals(self, current_state: Dict[str, Any], 
                              beliefs: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze counterfactual scenarios to understand causal influence.
        
        Args:
            current_state: Current state
            beliefs: Current beliefs
            
        Returns:
            Counterfactual analysis
        """
        counterfactuals = {}
        
        # What if different time?
        time_counterfactuals = self._generate_time_counterfactuals(current_state)
        counterfactuals['temporal'] = time_counterfactuals
        
        # What if different context?
        context_counterfactuals = self._generate_context_counterfactuals(current_state)
        counterfactuals['contextual'] = context_counterfactuals
        
        # What if different belief state?
        belief_counterfactuals = self._generate_belief_counterfactuals(beliefs)
        counterfactuals['belief'] = belief_counterfactuals
        
        return counterfactuals
    
    def calculate_causal_strength(self, factor: str, 
                                current_state: Dict[str, Any],
                                beliefs: List[List[float]]) -> float:
        """
        Calculate causal strength of a specific factor.
        
        Args:
            factor: Factor name to analyze
            current_state: Current state
            beliefs: Current beliefs
            
        Returns:
            Causal strength score (0-1)
        """
        # Calculate baseline utility
        baseline_action = self._get_optimal_action(beliefs)
        baseline_utility = self._calculate_action_utility(beliefs, baseline_action)
        
        # Calculate utility with factor removed/changed
        modified_state = self._modify_state_factor(current_state, factor)
        modified_beliefs = self._estimate_beliefs_from_state(modified_state)
        modified_action = self._get_optimal_action(modified_beliefs)
        modified_utility = self._calculate_action_utility(modified_beliefs, modified_action)
        
        # Causal strength is the difference
        causal_strength = abs(baseline_utility - modified_utility)
        
        # Normalize to 0-1 range
        return min(1.0, causal_strength)
    
    def _analyze_temporal_factors(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze temporal factors influencing decisions."""
        factors = []
        
        # Time of day
        if 'timestamp' in current_state:
            time_factor = {
                'name': 'Time of Day',
                'type': 'temporal',
                'value': current_state.get('hour_of_day', 12),
                'importance': self._calculate_time_importance(current_state),
                'description': self._describe_time_influence(current_state)
            }
            factors.append(time_factor)
        
        # Day of week
        if 'day_of_week' in current_state:
            day_factor = {
                'name': 'Day of Week',
                'type': 'temporal',
                'value': current_state.get('day_of_week', 'unknown'),
                'importance': 0.3,  # Moderate importance
                'description': f"It's {current_state.get('day_of_week', 'a weekday')}, which affects activity patterns"
            }
            factors.append(day_factor)
        
        return factors
    
    def _analyze_environmental_factors(self, current_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze environmental factors."""
        factors = []
        
        # Location context
        if 'location_context' in current_state:
            location_factor = {
                'name': 'Location Context',
                'type': 'environmental',
                'value': current_state['location_context'],
                'importance': self._calculate_location_importance(current_state['location_context']),
                'description': f"Currently in {current_state['location_context']} environment"
            }
            factors.append(location_factor)
        
        # Social context
        if 'social_context' in current_state:
            social_factor = {
                'name': 'Social Context',
                'type': 'environmental',
                'value': current_state['social_context'],
                'importance': self._calculate_social_importance(current_state['social_context']),
                'description': f"Social situation: {current_state['social_context']}"
            }
            factors.append(social_factor)
        
        return factors
    
    def _analyze_behavioral_factors(self, beliefs: List[List[float]]) -> List[Dict[str, Any]]:
        """Analyze behavioral state factors."""
        factors = []
        
        if len(beliefs) >= 1:  # Energy factor
            energy_belief = beliefs[0]
            max_energy_state = energy_belief.index(max(energy_belief))
            energy_confidence = max(energy_belief)
            
            energy_factor = {
                'name': 'Energy Level',
                'type': 'behavioral',
                'value': ['Low', 'Medium', 'High'][max_energy_state] if max_energy_state < 3 else 'Unknown',
                'importance': energy_confidence,
                'description': f"Current energy level affects activity choices"
            }
            factors.append(energy_factor)
        
        if len(beliefs) >= 2:  # Mood factor
            mood_belief = beliefs[1]
            max_mood_state = mood_belief.index(max(mood_belief))
            mood_confidence = max(mood_belief)
            
            mood_factor = {
                'name': 'Mood State',
                'type': 'behavioral',
                'value': ['Low', 'Neutral', 'High'][max_mood_state] if max_mood_state < 3 else 'Unknown',
                'importance': mood_confidence,
                'description': f"Current mood influences activity preferences"
            }
            factors.append(mood_factor)
        
        return factors
    
    def _analyze_context_factors(self, current_state: Dict[str, Any], 
                               beliefs: List[List[float]]) -> List[Dict[str, Any]]:
        """Analyze contextual factors."""
        factors = []
        
        # Digital behavior context
        if 'digital_behavior' in current_state:
            digital_factor = {
                'name': 'Digital Activity',
                'type': 'contextual',
                'value': current_state['digital_behavior'],
                'importance': 0.4,
                'description': f"Recent digital activity: {current_state['digital_behavior']}"
            }
            factors.append(digital_factor)
        
        return factors
    
    def _calculate_time_importance(self, current_state: Dict[str, Any]) -> float:
        """Calculate importance of time factor."""
        hour = current_state.get('hour_of_day', 12)
        
        # Peak importance during transition times
        if hour in [6, 7, 8, 12, 13, 17, 18, 22, 23]:
            return 0.8
        elif hour in [9, 10, 11, 14, 15, 16, 19, 20, 21]:
            return 0.6
        else:
            return 0.3
    
    def _describe_time_influence(self, current_state: Dict[str, Any]) -> str:
        """Describe how time influences decisions."""
        hour = current_state.get('hour_of_day', 12)
        
        if 6 <= hour <= 9:
            return "Morning routine time affects activity choices"
        elif 10 <= hour <= 12:
            return "Mid-morning productive period"
        elif 12 <= hour <= 14:
            return "Lunch and midday break time"
        elif 15 <= hour <= 17:
            return "Afternoon productive period"
        elif 18 <= hour <= 20:
            return "Evening transition and personal time"
        elif 21 <= hour <= 23:
            return "Wind-down and relaxation time"
        else:
            return "Late night/early morning rest period"
    
    def _calculate_location_importance(self, location: str) -> float:
        """Calculate importance of location factor."""
        high_impact_locations = ['home', 'office', 'gym', 'social-venue']
        medium_impact_locations = ['commute', 'outdoor', 'public']
        
        if location in high_impact_locations:
            return 0.8
        elif location in medium_impact_locations:
            return 0.5
        else:
            return 0.3
    
    def _calculate_social_importance(self, social_context: str) -> float:
        """Calculate importance of social factor."""
        high_impact_social = ['with-others', 'group-activity', 'meeting']
        medium_impact_social = ['family-time', 'casual-social']
        
        if social_context in high_impact_social:
            return 0.7
        elif social_context in medium_impact_social:
            return 0.5
        else:
            return 0.2
    
    # Simplified helper methods
    def _generate_time_counterfactuals(self, state): return []
    def _generate_context_counterfactuals(self, state): return []
    def _generate_belief_counterfactuals(self, beliefs): return []
    def _get_optimal_action(self, beliefs): return 0
    def _calculate_action_utility(self, beliefs, action): return 0.5
    def _modify_state_factor(self, state, factor): return state.copy()
    def _estimate_beliefs_from_state(self, state): return [[0.33, 0.33, 0.34]] * 3