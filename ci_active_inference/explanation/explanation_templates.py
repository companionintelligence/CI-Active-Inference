"""Explanation templates for generating natural language explanations."""

from typing import Dict, List, Any


class ExplanationTemplates:
    """
    Provides templates and formatting for natural language explanations.
    
    Contains various templates for different types of explanations
    and contexts to generate human-readable text.
    """
    
    def __init__(self):
        """Initialize explanation templates."""
        self.summary_templates = self._load_summary_templates()
        self.reasoning_templates = self._load_reasoning_templates()
        self.consequence_templates = self._load_consequence_templates()
        self.confidence_templates = self._load_confidence_templates()
        
    def get_summary_template(self) -> str:
        """Get summary template for basic explanation."""
        return ("Based on your current situation, you would likely default to '{predicted_default}'. "
                "However, I suggest '{suggested_action}' for better outcomes. "
                "I am {confidence_desc} in this recommendation.")
    
    def get_reasoning_template(self, reasoning_type: str) -> str:
        """Get reasoning template for specific type."""
        templates = {
            "efe_based": "This recommendation minimizes expected free energy by {efe_factors}.",
            "preference_based": "This aligns with your preferences for {preference_factors}.",
            "exploration_based": "This helps gather information about {exploration_factors}.",
            "habit_based": "This follows your typical patterns while {improvement_factors}."
        }
        return templates.get(reasoning_type, templates["efe_based"])
    
    def get_consequence_template(self, consequence_type: str) -> str:
        """Get consequence template for outcomes."""
        templates = {
            "positive": "This action is likely to result in {positive_outcomes}.",
            "mixed": "This action may lead to {positive_outcomes} but also {potential_risks}.",
            "uncertain": "The outcomes are uncertain, but this action {rationale}.",
            "preventive": "This action helps avoid {negative_outcomes}."
        }
        return templates.get(consequence_type, templates["positive"])
    
    def get_confidence_template(self, confidence_level: str) -> str:
        """Get confidence explanation template."""
        templates = {
            "high": "I'm highly confident because {high_confidence_reasons}.",
            "medium": "I'm moderately confident, though {uncertainty_factors}.",
            "low": "I have low confidence due to {low_confidence_reasons}.",
            "uncertain": "The situation is uncertain because {uncertainty_sources}."
        }
        return templates.get(confidence_level, templates["medium"])
    
    def format_causal_explanation(self, factors: List[Dict[str, Any]]) -> str:
        """Format causal factors into readable explanation."""
        if not factors:
            return "No specific causal factors identified."
        
        factor_descriptions = []
        for factor in factors[:3]:  # Top 3 factors
            name = factor.get('name', 'Unknown factor')
            importance = factor.get('importance', 0.5)
            description = factor.get('description', f'{name} influences the decision')
            
            if importance > 0.7:
                factor_descriptions.append(f"{description} (high influence)")
            elif importance > 0.4:
                factor_descriptions.append(f"{description} (moderate influence)")
            else:
                factor_descriptions.append(f"{description} (low influence)")
        
        if len(factor_descriptions) == 1:
            return f"Key factor: {factor_descriptions[0]}"
        elif len(factor_descriptions) == 2:
            return f"Key factors: {factor_descriptions[0]} and {factor_descriptions[1]}"
        else:
            return f"Key factors: {', '.join(factor_descriptions[:-1])}, and {factor_descriptions[-1]}"
    
    def format_efe_explanation(self, efe_analysis: Dict[str, Any]) -> str:
        """Format EFE analysis into readable explanation."""
        risk_component = efe_analysis.get('risk_component', {})
        ambiguity_component = efe_analysis.get('ambiguity_component', {})
        dominant = efe_analysis.get('dominant_component', 'unknown')
        
        risk_value = risk_component.get('value', 0.5)
        ambiguity_value = ambiguity_component.get('value', 0.5)
        
        explanations = []
        
        if dominant == 'risk':
            explanations.append(f"preference alignment concerns (risk: {risk_value:.2f})")
            if ambiguity_value > 0.4:
                explanations.append(f"some uncertainty in outcomes (ambiguity: {ambiguity_value:.2f})")
        else:
            explanations.append(f"uncertainty in current situation (ambiguity: {ambiguity_value:.2f})")
            if risk_value > 0.4:
                explanations.append(f"some preference considerations (risk: {risk_value:.2f})")
        
        return " and ".join(explanations)
    
    def format_alternative_explanation(self, alternatives: List[Dict[str, Any]]) -> str:
        """Format alternative actions explanation."""
        if not alternatives or len(alternatives) < 2:
            return "No significant alternatives identified."
        
        top_alternative = alternatives[1]  # Second best (first is recommended)
        alt_action = top_alternative.get('action', 'unknown')
        alt_utility = top_alternative.get('expected_utility', 0)
        
        recommended_utility = alternatives[0].get('expected_utility', 0)
        utility_diff = recommended_utility - alt_utility
        
        if utility_diff > 0.1:
            return f"The next best option would be '{alt_action}', but it's significantly less optimal."
        elif utility_diff > 0.05:
            return f"'{alt_action}' is a reasonable alternative, though slightly less optimal."
        else:
            return f"'{alt_action}' is nearly as good an option with similar expected outcomes."
    
    def format_temporal_context(self, time_info: Dict[str, Any]) -> str:
        """Format temporal context explanation."""
        if not time_info:
            return ""
        
        hour = time_info.get('hour_of_day')
        day = time_info.get('day_of_week')
        
        context_parts = []
        
        if hour is not None:
            if 6 <= hour <= 9:
                context_parts.append("it's morning routine time")
            elif 10 <= hour <= 12:
                context_parts.append("it's mid-morning productive period")
            elif 12 <= hour <= 14:
                context_parts.append("it's lunch break time")
            elif 15 <= hour <= 17:
                context_parts.append("it's afternoon productive period")
            elif 18 <= hour <= 21:
                context_parts.append("it's evening personal time")
            elif 22 <= hour <= 6:
                context_parts.append("it's rest and recovery time")
        
        if day:
            if day in ['Saturday', 'Sunday']:
                context_parts.append("it's the weekend")
            else:
                context_parts.append("it's a weekday")
        
        if context_parts:
            return f"Given that {' and '.join(context_parts)}, this recommendation fits your typical patterns."
        else:
            return ""
    
    def format_consequence_scenarios(self, consequences: Dict[str, Any]) -> str:
        """Format consequence scenarios explanation."""
        immediate = consequences.get('immediate_effects', '')
        scenarios = consequences.get('outcome_scenarios', [])
        
        parts = []
        
        if immediate:
            parts.append(f"Immediate effects: {immediate}")
        
        if scenarios:
            scenario_text = self._format_scenarios(scenarios)
            parts.append(f"Likely outcomes: {scenario_text}")
        
        return ". ".join(parts)
    
    def _format_scenarios(self, scenarios: List[Dict[str, Any]]) -> str:
        """Format outcome scenarios."""
        if not scenarios:
            return "positive changes expected"
        
        scenario_descriptions = []
        for scenario in scenarios[:2]:  # Top 2 scenarios
            prob = scenario.get('probability', 0.5)
            desc = scenario.get('description', 'beneficial outcome')
            
            if prob > 0.7:
                scenario_descriptions.append(f"{desc} (very likely)")
            elif prob > 0.5:
                scenario_descriptions.append(f"{desc} (likely)")
            else:
                scenario_descriptions.append(f"{desc} (possible)")
        
        return " and ".join(scenario_descriptions)
    
    def _load_summary_templates(self) -> List[str]:
        """Load summary templates."""
        return [
            "Based on current conditions, I suggest {action} for optimal outcomes.",
            "Given your situation, {action} would be most beneficial right now.",
            "Considering your preferences and context, {action} is recommended."
        ]
    
    def _load_reasoning_templates(self) -> List[str]:
        """Load reasoning templates."""
        return [
            "This recommendation is based on {reasoning_factors}.",
            "The decision considers {reasoning_factors}.",
            "This choice optimizes for {reasoning_factors}."
        ]
    
    def _load_consequence_templates(self) -> List[str]:
        """Load consequence templates."""
        return [
            "This action should lead to {positive_outcomes}.",
            "Expected results include {positive_outcomes}.",
            "You can expect {positive_outcomes} from this choice."
        ]
    
    def _load_confidence_templates(self) -> List[str]:
        """Load confidence templates."""
        return [
            "I'm {confidence_level} confident in this recommendation.",
            "The confidence level for this suggestion is {confidence_level}.",
            "This recommendation has {confidence_level} certainty."
        ]