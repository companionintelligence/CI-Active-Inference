"""Active Inference agent for CI system behavior prediction and optimization."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pymdp import Agent
from pymdp.utils import random_A_matrix, random_B_matrix
import logging
from datetime import datetime, timedelta

from ci_active_inference.database.dao import DataAnalyzer

logger = logging.getLogger(__name__)


class CIActiveInferenceAgent:
    """
    Active Inference agent for CI system using pymdp library.
    
    This agent learns patterns in user behavior, resource utilization,
    and system dynamics to predict and optimize future states.
    """
    
    def __init__(self, 
                 num_obs: List[int] = [10, 5, 8],  # observations: time, activity, resources
                 num_states: List[int] = [5, 3, 4],  # hidden states: user_intent, system_load, availability
                 num_controls: List[int] = [3, 2],   # controls: scheduling, resource_allocation
                 learning_rate: float = 0.1,
                 planning_horizon: int = 5):
        """
        Initialize the Active Inference agent.
        
        Args:
            num_obs: Number of observations for each modality
            num_states: Number of hidden states for each factor
            num_controls: Number of control actions for each factor
            learning_rate: Learning rate for belief updates
            planning_horizon: Planning horizon for future predictions
        """
        
        self.num_obs = num_obs
        self.num_states = num_states
        self.num_controls = num_controls
        self.learning_rate = learning_rate
        self.planning_horizon = planning_horizon
        
        # Initialize observation and transition models
        self._initialize_models()
        
        # Create pymdp agent
        self.agent = Agent(
            A=self.A_matrices,
            B=self.B_matrices,
            C=self.C_vectors,
            planning_horizon=planning_horizon,
            inference_algo="VMP",  # Variational Message Passing
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            action_precision=16.0,
            learning_rate=learning_rate
        )
        
        # State tracking
        self.current_state = None
        self.observation_history = []
        self.action_history = []
        self.belief_history = []
        
        logger.info("CI Active Inference Agent initialized successfully")
    
    def _initialize_models(self):
        """Initialize the generative model matrices."""
        
        # A matrices (observation model) - how hidden states generate observations
        self.A_matrices = []
        
        # Time-based observations (hour of day patterns)
        A_time = random_A_matrix([self.num_obs[0], self.num_states[0]])
        # Make time observations more predictable based on user intent
        A_time = A_time + np.eye(min(self.num_obs[0], self.num_states[0])) * 2
        A_time = A_time / A_time.sum(axis=0, keepdims=True)
        self.A_matrices.append(A_time)
        
        # Activity observations (event types, content creation)
        A_activity = random_A_matrix([self.num_obs[1], self.num_states[1]])
        A_activity = A_activity / A_activity.sum(axis=0, keepdims=True)
        self.A_matrices.append(A_activity)
        
        # Resource observations (availability, usage)
        A_resources = random_A_matrix([self.num_obs[2], self.num_states[2]])
        A_resources = A_resources / A_resources.sum(axis=0, keepdims=True)
        self.A_matrices.append(A_resources)
        
        # B matrices (transition model) - how states evolve with actions
        self.B_matrices = []
        
        # User intent transitions
        B_intent = random_B_matrix([self.num_states[0], self.num_states[0], self.num_controls[0]])
        # Add some structure - intents tend to persist
        for i in range(self.num_controls[0]):
            B_intent[:, :, i] += np.eye(self.num_states[0]) * 0.5
        B_intent = B_intent / B_intent.sum(axis=0, keepdims=True)
        self.B_matrices.append(B_intent)
        
        # System load transitions
        B_load = random_B_matrix([self.num_states[1], self.num_states[1], self.num_controls[1]])
        B_load = B_load / B_load.sum(axis=0, keepdims=True)
        self.B_matrices.append(B_load)
        
        # Availability transitions (no control - depends on usage patterns)
        B_availability = random_B_matrix([self.num_states[2], self.num_states[2], 1])
        B_availability = B_availability / B_availability.sum(axis=0, keepdims=True)
        self.B_matrices.append(B_availability)
        
        # C vectors (preferences) - what outcomes the agent prefers
        self.C_vectors = []
        
        # Prefer balanced time usage
        C_time = np.ones(self.num_obs[0]) * 0.1
        C_time[self.num_obs[0]//2] = 1.0  # Prefer moderate activity
        self.C_vectors.append(C_time)
        
        # Prefer productive activities
        C_activity = np.ones(self.num_obs[1]) * 0.1
        C_activity[0] = 1.0  # Prefer high productivity state
        self.C_vectors.append(C_activity)
        
        # Prefer resource availability
        C_resources = np.ones(self.num_obs[2]) * 0.1
        C_resources[-1] = 1.0  # Prefer high availability
        self.C_vectors.append(C_resources)
    
    def encode_observations(self, system_state: Dict[str, Any]) -> List[int]:
        """
        Encode system state into observation indices for the agent.
        
        Args:
            system_state: Current system state from DataAnalyzer
            
        Returns:
            List of observation indices
        """
        observations = []
        
        # Time observation (hour of day mapped to discrete bins)
        hour = system_state.get("hour_of_day", 12)
        time_obs = min(hour * self.num_obs[0] // 24, self.num_obs[0] - 1)
        observations.append(time_obs)
        
        # Activity observation (based on active events and recent content)
        active_events = system_state.get("active_events", 0)
        recent_content = system_state.get("recent_content_items", 0)
        activity_level = min((active_events + recent_content) // 2, self.num_obs[1] - 1)
        observations.append(activity_level)
        
        # Resource observation (based on available resources)
        available_resources = system_state.get("available_resources", 0)
        total_things = max(system_state.get("total_things", 1), 1)
        resource_availability = min(
            int((available_resources / total_things) * self.num_obs[2]),
            self.num_obs[2] - 1
        )
        observations.append(resource_availability)
        
        return observations
    
    def predict_future_state(self, current_state: Dict[str, Any], 
                           time_steps: int = 5) -> List[Dict[str, Any]]:
        """
        Predict future system states using Active Inference.
        
        Args:
            current_state: Current system state
            time_steps: Number of time steps to predict
            
        Returns:
            List of predicted future states
        """
        observations = self.encode_observations(current_state)
        
        # Update agent with current observations
        qs = self.agent.infer_states(observations)
        
        # Get expected future states under different policies
        predictions = []
        
        for t in range(time_steps):
            # Sample from posterior beliefs about future states
            future_qs = self.agent.infer_policies()
            
            # Decode predictions back to system state format
            predicted_state = self._decode_state_prediction(future_qs, current_state)
            predictions.append(predicted_state)
        
        return predictions
    
    def _decode_state_prediction(self, beliefs: List[np.ndarray], 
                                reference_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decode agent beliefs back into system state format.
        
        Args:
            beliefs: Agent's beliefs about hidden states
            reference_state: Reference state for scaling
            
        Returns:
            Predicted system state
        """
        # Extract most likely states
        intent_state = np.argmax(beliefs[0])
        load_state = np.argmax(beliefs[1])
        availability_state = np.argmax(beliefs[2])
        
        # Map back to system metrics
        base_events = reference_state.get("active_events", 0)
        base_resources = reference_state.get("available_resources", 0)
        
        # Predict changes based on inferred states
        predicted_events = base_events + (intent_state - 2)  # Center around current
        predicted_resources = base_resources + (availability_state - 2)
        
        predicted_state = reference_state.copy()
        predicted_state.update({
            "predicted_active_events": max(0, predicted_events),
            "predicted_available_resources": max(0, predicted_resources),
            "predicted_system_load": ["low", "medium", "high"][load_state] if load_state < 3 else "high",
            "confidence": float(np.max(beliefs[0]) * np.max(beliefs[1]) * np.max(beliefs[2]))
        })
        
        return predicted_state
    
    def recommend_actions(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend optimal actions based on current state and predictions.
        
        Args:
            current_state: Current system state
            
        Returns:
            Recommended actions and their expected outcomes
        """
        observations = self.encode_observations(current_state)
        
        # Infer current state
        qs = self.agent.infer_states(observations)
        
        # Get action recommendations
        policy = self.agent.infer_policies()
        action = self.agent.sample_action()
        
        # Interpret actions
        recommendations = {
            "scheduling_action": self._interpret_scheduling_action(action[0]),
            "resource_action": self._interpret_resource_action(action[1]),
            "confidence": float(np.max(qs[0])),
            "expected_outcome": self._predict_action_outcome(action, current_state),
            "timestamp": datetime.now().isoformat()
        }
        
        return recommendations
    
    def _interpret_scheduling_action(self, action_idx: int) -> str:
        """Interpret scheduling action index."""
        actions = ["reduce_scheduling", "maintain_current", "increase_scheduling"]
        return actions[min(action_idx, len(actions) - 1)]
    
    def _interpret_resource_action(self, action_idx: int) -> str:
        """Interpret resource allocation action index."""
        actions = ["optimize_allocation", "maintain_current"]
        return actions[min(action_idx, len(actions) - 1)]
    
    def _predict_action_outcome(self, action: List[int], 
                               current_state: Dict[str, Any]) -> Dict[str, str]:
        """Predict the outcome of taking specific actions."""
        return {
            "productivity": "increased" if action[0] == 2 else "maintained",
            "resource_efficiency": "optimized" if action[1] == 0 else "maintained",
            "system_load": "balanced" if action[0] == 1 else "variable"
        }
    
    def update_beliefs(self, observations: List[int], action: Optional[List[int]] = None):
        """
        Update agent beliefs based on new observations.
        
        Args:
            observations: New observations
            action: Action that was taken (if any)
        """
        # Update agent's beliefs
        qs = self.agent.infer_states(observations)
        
        # Store history
        self.observation_history.append(observations)
        self.belief_history.append(qs)
        
        if action is not None:
            self.action_history.append(action)
            # Update transition models based on experience
            self.agent.update_B(qs)
        
        # Update observation models
        self.agent.update_A(observations)
        
        logger.debug(f"Updated beliefs - Confidence: {np.max(qs[0]):.3f}")
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get metrics about the agent's learning progress."""
        if not self.belief_history:
            return {"status": "No learning data available"}
        
        # Calculate confidence trends
        confidences = [np.max(beliefs[0]) for beliefs in self.belief_history[-10:]]
        
        return {
            "total_observations": len(self.observation_history),
            "recent_confidence": np.mean(confidences) if confidences else 0.0,
            "confidence_trend": "improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "stable",
            "learning_rate": self.learning_rate,
            "planning_horizon": self.planning_horizon
        }


class CIBehaviorOptimizer:
    """
    High-level behavior optimization system using Active Inference.
    """
    
    def __init__(self, agent: CIActiveInferenceAgent, data_analyzer: DataAnalyzer):
        self.agent = agent
        self.data_analyzer = data_analyzer
        self.optimization_history = []
    
    def optimize_user_schedule(self, person_id: int) -> Dict[str, Any]:
        """
        Optimize schedule for a specific user using Active Inference.
        
        Args:
            person_id: ID of the person to optimize for
            
        Returns:
            Optimization recommendations
        """
        # Get user activity patterns
        patterns = self.data_analyzer.get_user_activity_patterns(person_id)
        current_state = self.data_analyzer.get_system_state_vector()
        current_state.update(patterns)
        
        # Get predictions and recommendations
        future_states = self.agent.predict_future_state(current_state, time_steps=7)
        recommendations = self.agent.recommend_actions(current_state)
        
        optimization_result = {
            "person_id": person_id,
            "current_patterns": patterns,
            "predicted_outcomes": future_states,
            "recommendations": recommendations,
            "optimization_score": self._calculate_optimization_score(patterns, future_states),
            "timestamp": datetime.now().isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """
        Optimize system-wide resource allocation.
        
        Returns:
            Resource optimization recommendations
        """
        current_state = self.data_analyzer.get_system_state_vector()
        
        # Predict resource needs
        predictions = self.agent.predict_future_state(current_state, time_steps=24)  # 24 hours
        
        # Generate allocation recommendations
        allocation_recommendations = []
        for i, pred_state in enumerate(predictions):
            predicted_load = pred_state.get("predicted_active_events", 0)
            available_resources = pred_state.get("predicted_available_resources", 0)
            
            if predicted_load > available_resources:
                allocation_recommendations.append({
                    "hour": i,
                    "action": "increase_availability",
                    "urgency": "high" if predicted_load > available_resources * 1.5 else "medium"
                })
            elif available_resources > predicted_load * 2:
                allocation_recommendations.append({
                    "hour": i,
                    "action": "reduce_allocation", 
                    "urgency": "low"
                })
        
        return {
            "predictions": predictions,
            "recommendations": allocation_recommendations,
            "optimization_score": self._calculate_resource_score(predictions),
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_optimization_score(self, patterns: Dict[str, Any], 
                                    predictions: List[Dict[str, Any]]) -> float:
        """Calculate optimization score based on patterns and predictions."""
        base_score = 0.5
        
        # Reward consistent patterns
        if patterns.get("total_events", 0) > 0:
            base_score += 0.2
        
        # Reward confident predictions
        avg_confidence = np.mean([p.get("confidence", 0) for p in predictions])
        base_score += avg_confidence * 0.3
        
        return min(base_score, 1.0)
    
    def _calculate_resource_score(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate resource optimization score."""
        if not predictions:
            return 0.0
        
        # Calculate efficiency based on resource utilization predictions
        utilization_scores = []
        for pred in predictions:
            events = pred.get("predicted_active_events", 0)
            resources = pred.get("predicted_available_resources", 1)
            utilization = events / max(resources, 1)
            # Optimal utilization is around 0.7-0.8
            if 0.6 <= utilization <= 0.9:
                utilization_scores.append(1.0)
            else:
                utilization_scores.append(max(0.0, 1.0 - abs(utilization - 0.75) * 2))
        
        return np.mean(utilization_scores)