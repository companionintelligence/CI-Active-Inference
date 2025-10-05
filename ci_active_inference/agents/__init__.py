"""Active Inference agent for CI system behavior prediction and optimization."""

import math
import random
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SimpleActiveInferenceAgent:
    """
    Simplified Active Inference agent for CI system using core Free Energy Principle concepts.
    
    This agent learns patterns in user behavior, resource utilization,
    and system dynamics to predict and optimize future states using
    a simplified implementation of Active Inference principles.
    """
    
    def __init__(self, 
                 num_obs_dims: int = 10,      # observation dimensions
                 num_state_dims: int = 5,     # hidden state dimensions  
                 num_action_dims: int = 3,    # action dimensions
                 learning_rate: float = 0.1,
                 planning_horizon: int = 5):
        """
        Initialize the Active Inference agent.
        
        Args:
            num_obs_dims: Number of observation dimensions
            num_state_dims: Number of hidden state dimensions
            num_action_dims: Number of action dimensions
            learning_rate: Learning rate for belief updates
            planning_horizon: Planning horizon for future predictions
        """
        
        self.num_obs_dims = num_obs_dims
        self.num_state_dims = num_state_dims
        self.num_action_dims = num_action_dims
        self.learning_rate = learning_rate
        self.planning_horizon = planning_horizon
        
        # Initialize model parameters
        self._initialize_models()
        
        # State tracking
        self.current_beliefs = [1.0/num_state_dims] * num_state_dims  # Uniform prior
        self.observation_history = []
        self.action_history = []
        self.belief_history = []
        self.free_energy_history = []
        
        logger.info("Simple Active Inference Agent initialized successfully")
    
    def _initialize_models(self):
        """Initialize the generative model parameters."""
        
        # Observation model: P(o|s) - how hidden states generate observations
        # Initialize with random values and normalize
        self.observation_model = []
        for i in range(self.num_obs_dims):
            row = [random.random() for _ in range(self.num_state_dims)]
            total = sum(row)
            if total > 0:
                row = [x/total for x in row]
            self.observation_model.append(row)
        
        # Transition model: P(s'|s,a) - how states evolve with actions
        self.transition_model = []
        for a in range(self.num_action_dims):
            action_transitions = []
            for s_from in range(self.num_state_dims):
                row = [random.random() for _ in range(self.num_state_dims)]
                total = sum(row)
                if total > 0:
                    row = [x/total for x in row]
                action_transitions.append(row)
            self.transition_model.append(action_transitions)
        
        # Prior preferences: C(o) - preferred observations
        self.preferences = [0.1] * self.num_obs_dims
        for i in range(self.num_obs_dims//2, self.num_obs_dims):
            self.preferences[i] = 1.0  # Prefer later observation states
        
        # Action precision (inverse temperature)
        self.action_precision = 2.0
    
    def encode_observations(self, system_state: Dict[str, Any]) -> List[float]:
        """
        Encode system state into observation vector.
        
        Args:
            system_state: Current system state from DataAnalyzer
            
        Returns:
            Observation vector
        """
        observations = [0.0] * self.num_obs_dims
        
        # Extract key features and normalize
        hour = system_state.get("hour_of_day", 12) / 24.0
        active_events = min(system_state.get("active_events", 0) / 10.0, 1.0)
        available_resources = min(system_state.get("available_resources", 0) / 20.0, 1.0)
        total_people = min(system_state.get("total_people", 0) / 100.0, 1.0)
        day_of_week = system_state.get("day_of_week", 0) / 7.0
        
        # Map to observation vector
        observations[0] = hour
        observations[1] = active_events
        observations[2] = available_resources
        observations[3] = total_people
        observations[4] = day_of_week
        
        # Add some derived features
        if self.num_obs_dims > 5:
            observations[5] = 1.0 if 9 <= system_state.get("hour_of_day", 12) <= 17 else 0.0  # Business hours
            observations[6] = active_events * available_resources  # Resource utilization
            observations[7] = math.sin(2 * math.pi * hour)  # Cyclic time feature
            observations[8] = math.cos(2 * math.pi * hour)  # Cyclic time feature
            
        if self.num_obs_dims > 9:
            observations[9] = (active_events + available_resources) / 2.0  # Overall activity
        
        return observations
    
    def infer_states(self, observations: List[float]) -> List[float]:
        """
        Infer hidden states from observations using Bayesian inference.
        
        Args:
            observations: Observation vector
            
        Returns:
            Posterior beliefs over hidden states
        """
        # Compute likelihood for each state
        likelihood = []
        
        for s in range(self.num_state_dims):
            # Simplified likelihood: Gaussian around expected observation
            expected_obs = [self.observation_model[i][s] for i in range(self.num_obs_dims)]
            diff_squared = sum((observations[i] - expected_obs[i])**2 for i in range(len(observations)))
            likelihood.append(math.exp(-0.5 * diff_squared))
        
        # Combine with prior beliefs (current beliefs act as prior)
        posterior = [likelihood[s] * self.current_beliefs[s] for s in range(self.num_state_dims)]
        
        # Normalize to get probability distribution
        total = sum(posterior)
        if total > 0:
            posterior = [x/total for x in posterior]
        else:
            posterior = [1.0/self.num_state_dims] * self.num_state_dims
        
        return posterior
    
    def calculate_free_energy(self, observations: List[float], beliefs: List[float]) -> float:
        """
        Calculate variational free energy.
        
        Args:
            observations: Current observations
            beliefs: Current beliefs over states
            
        Returns:
            Free energy value
        """
        # Expected likelihood
        expected_likelihood = 0.0
        for s in range(self.num_state_dims):
            expected_obs = [self.observation_model[i][s] for i in range(self.num_obs_dims)]
            diff_squared = sum((observations[i] - expected_obs[i])**2 for i in range(len(observations)))
            expected_likelihood += beliefs[s] * math.exp(-0.5 * diff_squared)
        
        # Entropy of beliefs (complexity cost)
        belief_entropy = -sum(b * math.log(b + 1e-10) for b in beliefs)  # Add small value to avoid log(0)
        
        # Free energy = negative log likelihood + complexity
        free_energy = -math.log(expected_likelihood + 1e-10) - belief_entropy
        
        return free_energy
    
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
        beliefs = self.infer_states(observations)
        
        predictions = []
        current_beliefs = beliefs[:]
        
        for t in range(time_steps):
            # Sample action (simplified policy)
            action = self._sample_action(current_beliefs)
            
            # Predict next state using transition model
            next_beliefs = [0.0] * self.num_state_dims
            for s_next in range(self.num_state_dims):
                for s_curr in range(self.num_state_dims):
                    next_beliefs[s_next] += (
                        current_beliefs[s_curr] * 
                        self.transition_model[action][s_curr][s_next]
                    )
            
            # Predict observations from beliefs
            predicted_obs = [0.0] * self.num_obs_dims
            for i in range(self.num_obs_dims):
                for s in range(self.num_state_dims):
                    predicted_obs[i] += self.observation_model[i][s] * next_beliefs[s]
            
            # Decode back to system state format
            predicted_state = self._decode_state_prediction(predicted_obs, current_state, t+1)
            predictions.append(predicted_state)
            
            # Update beliefs for next iteration
            current_beliefs = next_beliefs
        
        return predictions
    
    def _sample_action(self, beliefs: List[float]) -> int:
        """
        Sample action based on current beliefs and expected free energy.
        
        Args:
            beliefs: Current beliefs over states
            
        Returns:
            Selected action index
        """
        # Calculate expected free energy for each action
        expected_free_energies = []
        
        for action in range(self.num_action_dims):
            # Predict next state distribution
            next_beliefs = [0.0] * self.num_state_dims
            for s_next in range(self.num_state_dims):
                for s_curr in range(self.num_state_dims):
                    next_beliefs[s_next] += (
                        beliefs[s_curr] * 
                        self.transition_model[action][s_curr][s_next]
                    )
            
            # Expected observations
            expected_obs = [0.0] * self.num_obs_dims
            for i in range(self.num_obs_dims):
                for s in range(self.num_state_dims):
                    expected_obs[i] += self.observation_model[i][s] * next_beliefs[s]
            
            # Calculate expected free energy (simplified)
            # Pragmatic value (preference satisfaction)
            pragmatic_value = sum(expected_obs[i] * self.preferences[i] for i in range(len(expected_obs)))
            
            # Epistemic value (information gain) - simplified as state entropy
            epistemic_value = -sum(b * math.log(b + 1e-10) for b in next_beliefs)
            
            expected_free_energies.append(-(pragmatic_value + epistemic_value))
        
        # Sample action using softmax
        # Convert to probabilities
        max_fe = max(expected_free_energies)
        exp_values = [math.exp(-self.action_precision * (fe - max_fe)) for fe in expected_free_energies]
        total = sum(exp_values)
        action_probs = [x/total for x in exp_values]
        
        # Sample from distribution
        rand_val = random.random()
        cumsum = 0.0
        for action, prob in enumerate(action_probs):
            cumsum += prob
            if rand_val <= cumsum:
                return action
        
        return self.num_action_dims - 1  # Fallback
    
    def _decode_state_prediction(self, predicted_obs: List[float], 
                                reference_state: Dict[str, Any],
                                time_step: int) -> Dict[str, Any]:
        """
        Decode predicted observations back into system state format.
        
        Args:
            predicted_obs: Predicted observation vector
            reference_state: Reference state for scaling
            time_step: Future time step
            
        Returns:
            Predicted system state
        """
        # Extract predictions from observation vector
        predicted_hour = max(0, min(23, int(predicted_obs[0] * 24)))
        predicted_events = max(0, int(predicted_obs[1] * 10))
        predicted_resources = max(0, int(predicted_obs[2] * 20))
        
        # Calculate confidence as inverse of prediction uncertainty
        variance = sum((x - 0.5)**2 for x in predicted_obs) / len(predicted_obs)
        confidence = 1.0 / (1.0 + variance)
        
        predicted_state = reference_state.copy()
        predicted_state.update({
            "predicted_hour": predicted_hour,
            "predicted_active_events": predicted_events,
            "predicted_available_resources": predicted_resources,
            "confidence": float(confidence),
            "time_step": time_step,
            "timestamp": (datetime.now() + timedelta(hours=time_step)).isoformat()
        })
        
        return predicted_state
    
    def recommend_actions(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend optimal actions based on current state.
        
        Args:
            current_state: Current system state
            
        Returns:
            Recommended actions and their expected outcomes
        """
        observations = self.encode_observations(current_state)
        beliefs = self.infer_states(observations)
        
        # Sample optimal action
        action = self._sample_action(beliefs)
        
        # Calculate expected outcomes
        next_beliefs = [0.0] * self.num_state_dims
        for s_next in range(self.num_state_dims):
            for s_curr in range(self.num_state_dims):
                next_beliefs[s_next] += (
                    beliefs[s_curr] * 
                    self.transition_model[action][s_curr][s_next]
                )
        
        recommendations = {
            "action_type": self._interpret_action(action),
            "confidence": float(max(beliefs)),
            "expected_state_change": float(max(next_beliefs) - max(beliefs)),
            "predicted_outcome": self._predict_action_outcome(action, current_state),
            "free_energy": self.calculate_free_energy(observations, beliefs),
            "timestamp": datetime.now().isoformat()
        }
        
        return recommendations
    
    def _interpret_action(self, action_idx: int) -> str:
        """Interpret action index into meaningful action."""
        actions = ["optimize_resources", "maintain_current", "increase_activity"]
        return actions[min(action_idx, len(actions) - 1)]
    
    def _predict_action_outcome(self, action: int, current_state: Dict[str, Any]) -> str:
        """Predict the qualitative outcome of an action."""
        if action == 0:
            return "improved_efficiency"
        elif action == 1:
            return "stable_state"
        else:
            return "increased_productivity"
    
    def update_beliefs(self, observations: List[float], action: Optional[int] = None):
        """
        Update agent beliefs and learn from experience.
        
        Args:
            observations: New observations
            action: Action that was taken (if any)
        """
        # Infer current state
        new_beliefs = self.infer_states(observations)
        
        # Calculate free energy
        free_energy = self.calculate_free_energy(observations, new_beliefs)
        
        # Update models with learning
        if len(self.belief_history) > 0:
            prev_beliefs = self.belief_history[-1]
            
            # Update observation model (simplified learning)
            for s in range(self.num_state_dims):
                for i in range(self.num_obs_dims):
                    update = self.learning_rate * new_beliefs[s] * observations[i]
                    self.observation_model[i][s] += update
                
                # Re-normalize observation model column
                col_sum = sum(self.observation_model[i][s] for i in range(self.num_obs_dims))
                if col_sum > 0:
                    for i in range(self.num_obs_dims):
                        self.observation_model[i][s] /= col_sum
            
            # Update transition model if action was taken
            if action is not None and len(self.action_history) > 0:
                prev_action = self.action_history[-1]
                for s_prev in range(self.num_state_dims):
                    for s_curr in range(self.num_state_dims):
                        update = (self.learning_rate * prev_beliefs[s_prev] * new_beliefs[s_curr])
                        self.transition_model[prev_action][s_prev][s_curr] += update
                
                # Re-normalize transition model
                for a in range(self.num_action_dims):
                    for s in range(self.num_state_dims):
                        row_sum = sum(self.transition_model[a][s])
                        if row_sum > 0:
                            for s_next in range(self.num_state_dims):
                                self.transition_model[a][s][s_next] /= row_sum
        
        # Store history
        self.current_beliefs = new_beliefs
        self.observation_history.append(observations)
        self.belief_history.append(new_beliefs)
        self.free_energy_history.append(free_energy)
        
        if action is not None:
            self.action_history.append(action)
        
        logger.debug(f"Updated beliefs - Confidence: {max(new_beliefs):.3f}, "
                    f"Free Energy: {free_energy:.3f}")
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get metrics about the agent's learning progress."""
        if not self.belief_history:
            return {"status": "No learning data available"}
        
        # Calculate confidence trends
        recent_beliefs = self.belief_history[-10:] if len(self.belief_history) >= 10 else self.belief_history
        confidences = [max(beliefs) for beliefs in recent_beliefs]
        
        recent_fe = self.free_energy_history[-10:] if len(self.free_energy_history) >= 10 else self.free_energy_history
        
        return {
            "total_observations": len(self.observation_history),
            "recent_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "confidence_trend": "improving" if len(confidences) > 1 and confidences[-1] > confidences[0] else "stable",
            "recent_free_energy": sum(recent_fe) / len(recent_fe) if recent_fe else 0.0,
            "free_energy_trend": "decreasing" if len(recent_fe) > 1 and recent_fe[-1] < recent_fe[0] else "stable",
            "learning_rate": self.learning_rate,
            "planning_horizon": self.planning_horizon
        }


# Alias for compatibility
CIActiveInferenceAgent = SimpleActiveInferenceAgent


class CIBehaviorOptimizer:
    """
    High-level behavior optimization system using Active Inference.
    """
    
    def __init__(self, agent: CIActiveInferenceAgent, data_analyzer):
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
            predicted_events = pred_state.get("predicted_active_events", 0)
            predicted_resources = pred_state.get("predicted_available_resources", 0)
            
            if predicted_events > predicted_resources:
                allocation_recommendations.append({
                    "hour": i,
                    "action": "increase_availability",
                    "urgency": "high" if predicted_events > predicted_resources * 1.5 else "medium"
                })
            elif predicted_resources > predicted_events * 2:
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
        confidences = [p.get("confidence", 0) for p in predictions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
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
        
        return sum(utilization_scores) / len(utilization_scores)