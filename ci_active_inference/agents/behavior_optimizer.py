"""
Behavior optimizer using Active Inference principles.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .simple_agent import SimpleActiveInferenceAgent


class CIBehaviorOptimizer:
    """
    Behavior optimization agent using Active Inference principles.
    
    This agent focuses on optimizing long-term behavior patterns
    for improved health and happiness outcomes.
    """
    
    def __init__(self, base_agent: Optional[SimpleActiveInferenceAgent] = None):
        """
        Initialize behavior optimizer.
        
        Args:
            base_agent: Base Active Inference agent (creates new if None)
        """
        self.base_agent = base_agent or SimpleActiveInferenceAgent()
        self.optimization_history = []
        self.behavior_patterns = {}
        self.goal_preferences = {
            "health_score": 0.8,
            "happiness_score": 0.8,
            "productivity_score": 0.7,
            "stress_level": 0.2  # Lower is better
        }
    
    def analyze_behavior_patterns(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze historical behavior patterns.
        
        Args:
            historical_data: List of historical state observations
            
        Returns:
            Analysis of behavior patterns
        """
        if not historical_data:
            return {"status": "insufficient_data"}
        
        # Extract patterns
        patterns = {
            "activity_cycles": self._detect_activity_cycles(historical_data),
            "resource_utilization": self._analyze_resource_patterns(historical_data),
            "state_transitions": self._analyze_state_transitions(historical_data)
        }
        
        self.behavior_patterns = patterns
        return patterns
    
    def _detect_activity_cycles(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect cyclical patterns in activity levels."""
        # Simple pattern detection
        activities = [d.get('active_events', 0) for d in data]
        
        if not activities:
            return {"status": "no_data"}
        
        avg_activity = sum(activities) / len(activities)
        peak_activity = max(activities)
        
        return {
            "average_activity": avg_activity,
            "peak_activity": peak_activity,
            "activity_variance": sum((a - avg_activity)**2 for a in activities) / len(activities)
        }
    
    def _analyze_resource_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze resource utilization patterns."""
        resources = [d.get('available_resources', 0) for d in data]
        
        if not resources:
            return {"status": "no_data"}
        
        return {
            "average_resources": sum(resources) / len(resources),
            "resource_efficiency": len([r for r in resources if r > 5]) / len(resources)
        }
    
    def _analyze_state_transitions(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze state transition patterns."""
        # Simple transition analysis
        transitions = []
        for i in range(1, len(data)):
            prev_events = data[i-1].get('active_events', 0)
            curr_events = data[i].get('active_events', 0)
            transitions.append(curr_events - prev_events)
        
        if not transitions:
            return {"status": "no_data"}
        
        return {
            "avg_transition": sum(transitions) / len(transitions),
            "stability": len([t for t in transitions if abs(t) <= 1]) / len(transitions)
        }
    
    def optimize_behavior(self, current_state: Dict[str, Any], 
                         optimization_horizon: int = 10) -> Dict[str, Any]:
        """
        Generate behavior optimization recommendations.
        
        Args:
            current_state: Current system state
            optimization_horizon: Planning horizon for optimization
            
        Returns:
            Optimization recommendations
        """
        # Get base predictions
        future_states = self.base_agent.predict_future_state(
            current_state, optimization_horizon
        )
        
        # Evaluate each future state against goals
        evaluations = []
        for state in future_states:
            score = self._evaluate_state_quality(state)
            evaluations.append(score)
        
        # Generate recommendations
        recommendations = {
            "current_trajectory_score": sum(evaluations) / len(evaluations) if evaluations else 0,
            "improvement_opportunities": self._identify_improvements(future_states),
            "recommended_adjustments": self._generate_adjustments(current_state, evaluations),
            "optimization_confidence": max(evaluations) if evaluations else 0
        }
        
        self.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "current_state": current_state,
            "recommendations": recommendations
        })
        
        return recommendations
    
    def _evaluate_state_quality(self, state: Dict[str, Any]) -> float:
        """Evaluate how well a state aligns with optimization goals."""
        # Simple scoring based on predefined preferences
        score = 0.0
        
        # Activity level scoring
        activity = state.get('predicted_active_events', state.get('active_events', 0))
        if 2 <= activity <= 5:  # Optimal activity range
            score += 0.3
        
        # Resource availability scoring
        resources = state.get('predicted_available_resources', state.get('available_resources', 0))
        if resources >= 10:  # Good resource availability
            score += 0.3
        
        # Confidence scoring
        confidence = state.get('confidence', 0.5)
        score += confidence * 0.4
        
        return min(1.0, score)
    
    def _identify_improvements(self, future_states: List[Dict[str, Any]]) -> List[str]:
        """Identify potential areas for improvement."""
        improvements = []
        
        # Check for resource depletion
        resource_trend = []
        for state in future_states:
            resources = state.get('predicted_available_resources', 
                                state.get('available_resources', 0))
            resource_trend.append(resources)
        
        if len(resource_trend) >= 2 and resource_trend[-1] < resource_trend[0]:
            improvements.append("Consider resource conservation strategies")
        
        # Check for activity imbalance
        activities = [s.get('predicted_active_events', s.get('active_events', 0)) 
                     for s in future_states]
        avg_activity = sum(activities) / len(activities)
        
        if avg_activity < 2:
            improvements.append("Consider increasing activity levels")
        elif avg_activity > 6:
            improvements.append("Consider reducing activity to prevent burnout")
        
        return improvements
    
    def _generate_adjustments(self, current_state: Dict[str, Any], 
                            evaluations: List[float]) -> List[str]:
        """Generate specific behavioral adjustments."""
        adjustments = []
        
        # Base adjustments on evaluation scores
        avg_score = sum(evaluations) / len(evaluations) if evaluations else 0
        
        if avg_score < 0.4:
            adjustments.append("Focus on sustainable activity patterns")
            adjustments.append("Improve resource management")
        elif avg_score < 0.7:
            adjustments.append("Fine-tune current approach")
            adjustments.append("Monitor resource utilization")
        else:
            adjustments.append("Maintain current patterns")
        
        return adjustments