"""Preference variants for A/B testing different agent configurations."""

import copy
from typing import Dict, List, Any, Optional
import numpy as np


class PreferenceVariants:
    """
    Generates different preference configurations for A/B testing.
    
    Creates variations of preference matrices (C matrices) to test
    different agent behaviors and optimization targets.
    """
    
    def __init__(self, base_preferences: Dict[str, Any] = None):
        """
        Initialize preference variants generator.
        
        Args:
            base_preferences: Base preference configuration
        """
        self.base_preferences = base_preferences or self._get_default_preferences()
        
    def generate_variants(self, variant_types: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generate preference variants for A/B testing.
        
        Args:
            variant_types: List of variant types to generate
            
        Returns:
            Dictionary of variant_name -> preference_configuration
        """
        variants = {"control": copy.deepcopy(self.base_preferences)}
        
        for variant_type in variant_types:
            if variant_type == "health_focused":
                variants["health_focused"] = self._create_health_focused_variant()
            elif variant_type == "productivity_focused":
                variants["productivity_focused"] = self._create_productivity_focused_variant()
            elif variant_type == "social_focused":
                variants["social_focused"] = self._create_social_focused_variant()
            elif variant_type == "exploration_focused":
                variants["exploration_focused"] = self._create_exploration_focused_variant()
            elif variant_type == "conservative":
                variants["conservative"] = self._create_conservative_variant()
            elif variant_type == "aggressive":
                variants["aggressive"] = self._create_aggressive_variant()
        
        return variants
    
    def create_custom_variant(self, name: str, 
                            preference_modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a custom preference variant.
        
        Args:
            name: Variant name
            preference_modifications: Modifications to apply to base preferences
            
        Returns:
            Custom preference configuration
        """
        variant = copy.deepcopy(self.base_preferences)
        
        for key, value in preference_modifications.items():
            if key in variant:
                if isinstance(variant[key], dict) and isinstance(value, dict):
                    variant[key].update(value)
                else:
                    variant[key] = value
            else:
                variant[key] = value
        
        return variant
    
    def _create_health_focused_variant(self) -> Dict[str, Any]:
        """Create health-focused preference variant."""
        variant = copy.deepcopy(self.base_preferences)
        
        # Increase preference for health-related activities
        if "activity_preferences" in variant:
            variant["activity_preferences"]["exercise"] = 0.9
            variant["activity_preferences"]["outdoor"] = 0.8
            variant["activity_preferences"]["rest"] = 0.7
            variant["activity_preferences"]["food"] = 0.6
            variant["activity_preferences"]["entertainment"] = 0.3
        
        # Adjust outcome preferences
        if "outcome_preferences" in variant:
            variant["outcome_preferences"]["energy_high"] = 0.9
            variant["outcome_preferences"]["mood_positive"] = 0.8
            variant["outcome_preferences"]["health_improved"] = 0.95
        
        return variant
    
    def _create_productivity_focused_variant(self) -> Dict[str, Any]:
        """Create productivity-focused preference variant."""
        variant = copy.deepcopy(self.base_preferences)
        
        # Increase preference for productive activities
        if "activity_preferences" in variant:
            variant["activity_preferences"]["work"] = 0.9
            variant["activity_preferences"]["focus"] = 0.85
            variant["activity_preferences"]["exercise"] = 0.6  # For energy
            variant["activity_preferences"]["entertainment"] = 0.2
            variant["activity_preferences"]["social"] = 0.4
        
        # Adjust outcome preferences
        if "outcome_preferences" in variant:
            variant["outcome_preferences"]["productivity_high"] = 0.95
            variant["outcome_preferences"]["focus_improved"] = 0.9
            variant["outcome_preferences"]["energy_high"] = 0.8
        
        return variant
    
    def _create_social_focused_variant(self) -> Dict[str, Any]:
        """Create social-focused preference variant."""
        variant = copy.deepcopy(self.base_preferences)
        
        # Increase preference for social activities
        if "activity_preferences" in variant:
            variant["activity_preferences"]["social"] = 0.9
            variant["activity_preferences"]["entertainment"] = 0.7
            variant["activity_preferences"]["outdoor"] = 0.6
            variant["activity_preferences"]["work"] = 0.4
            variant["activity_preferences"]["focus"] = 0.3
        
        # Adjust outcome preferences
        if "outcome_preferences" in variant:
            variant["outcome_preferences"]["social_satisfaction"] = 0.95
            variant["outcome_preferences"]["mood_positive"] = 0.9
            variant["outcome_preferences"]["connection_improved"] = 0.85
        
        return variant
    
    def _create_exploration_focused_variant(self) -> Dict[str, Any]:
        """Create exploration-focused preference variant."""
        variant = copy.deepcopy(self.base_preferences)
        
        # Add exploration bonus to all activities
        if "activity_preferences" in variant:
            for activity in variant["activity_preferences"]:
                variant["activity_preferences"][activity] += 0.1
        
        # Add uncertainty tolerance
        variant["uncertainty_tolerance"] = 0.8
        variant["exploration_bonus"] = 0.3
        
        return variant
    
    def _create_conservative_variant(self) -> Dict[str, Any]:
        """Create conservative preference variant."""
        variant = copy.deepcopy(self.base_preferences)
        
        # Reduce preference for risky activities
        if "activity_preferences" in variant:
            variant["activity_preferences"]["outdoor"] *= 0.8
            variant["activity_preferences"]["social"] *= 0.9
            variant["activity_preferences"]["entertainment"] *= 0.7
            variant["activity_preferences"]["rest"] *= 1.2  # Prefer safe options
        
        # Increase risk aversion
        variant["risk_aversion"] = 0.8
        variant["uncertainty_tolerance"] = 0.3
        
        return variant
    
    def _create_aggressive_variant(self) -> Dict[str, Any]:
        """Create aggressive preference variant."""
        variant = copy.deepcopy(self.base_preferences)
        
        # Increase preference for challenging activities
        if "activity_preferences" in variant:
            variant["activity_preferences"]["exercise"] *= 1.3
            variant["activity_preferences"]["work"] *= 1.2
            variant["activity_preferences"]["focus"] *= 1.2
            variant["activity_preferences"]["rest"] *= 0.7
        
        # Reduce risk aversion
        variant["risk_aversion"] = 0.2
        variant["uncertainty_tolerance"] = 0.9
        
        return variant
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Get default preference configuration."""
        return {
            "activity_preferences": {
                "no_action": 0.1,
                "exercise": 0.6,
                "work": 0.5,
                "social": 0.6,
                "rest": 0.5,
                "entertainment": 0.4,
                "food": 0.5,
                "outdoor": 0.6,
                "focus": 0.5
            },
            "outcome_preferences": {
                "energy_high": 0.8,
                "mood_positive": 0.8,
                "productivity_high": 0.7,
                "social_satisfaction": 0.6,
                "health_improved": 0.7
            },
            "risk_aversion": 0.5,
            "uncertainty_tolerance": 0.5,
            "exploration_bonus": 0.1
        }
    
    def compare_variants(self, variant1: Dict[str, Any], 
                        variant2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two preference variants.
        
        Args:
            variant1: First variant to compare
            variant2: Second variant to compare
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "differences": {},
            "similarity_score": 0.0,
            "key_differences": []
        }
        
        # Compare activity preferences
        if ("activity_preferences" in variant1 and 
            "activity_preferences" in variant2):
            
            act_diff = {}
            for activity in variant1["activity_preferences"]:
                if activity in variant2["activity_preferences"]:
                    diff = (variant1["activity_preferences"][activity] - 
                           variant2["activity_preferences"][activity])
                    act_diff[activity] = diff
                    
                    if abs(diff) > 0.2:
                        comparison["key_differences"].append(
                            f"{activity}: {diff:+.2f} preference difference"
                        )
            
            comparison["differences"]["activity_preferences"] = act_diff
        
        # Calculate similarity score
        total_diff = sum(abs(diff) for diff in 
                        comparison["differences"].get("activity_preferences", {}).values())
        max_possible_diff = len(variant1.get("activity_preferences", {})) * 2
        comparison["similarity_score"] = 1.0 - (total_diff / max_possible_diff) if max_possible_diff > 0 else 1.0
        
        return comparison