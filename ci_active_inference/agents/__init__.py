"""Active Inference agent for CI system behavior prediction and optimization using pymdp."""

import numpy as np
from pymdp.agent import Agent
from pymdp import utils
from pymdp.maths import softmax
import math
import json
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PyMDPActiveInferenceAgent:
    """
    Active Inference agent for CI system using pymdp library.
    
    This agent uses the pymdp framework to implement proper Active Inference
    for behavior prediction and optimization in the CI system.
    """
    
    def __init__(self, 
                 user_id: str,
                 num_obs: List[int] = [4, 5, 4, 3],  # [Digital, Location, Social, Content]
                 num_states: List[int] = [3, 3, 5],  # [Energy, Mood, Context]
                 num_actions: List[int] = [9],        # Available actions
                 learning_rate: float = 0.1,
                 planning_horizon: int = 5):
        """
        Initialize the Active Inference agent using pymdp.
        
        Args:
            user_id: User identifier
            num_obs: Number of observations for each modality
            num_states: Number of hidden states for each factor
            num_actions: Number of actions for each factor
            learning_rate: Learning rate for Dirichlet updates
            planning_horizon: Planning horizon for policy evaluation
        """
        
        self.user_id = user_id
        self.num_obs = num_obs
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.planning_horizon = planning_horizon
        
        # Initialize model matrices
        self.A = self._initialize_observation_model()  # Observation likelihood
        self.B = self._initialize_transition_model()   # State transitions
        self.C = self._initialize_preferences()        # Preferences
        self.D = self._initialize_priors()            # Prior beliefs
        
        # Initialize pymdp agent
        self.agent = Agent(
            A=self.A,
            B=self.B, 
            C=self.C,
            D=self.D,
            use_utility=True,
            use_states_info_gain=True,
            use_param_info_gain=True,
            policy_len=planning_horizon,
            inference_algo="VANILLA"
        )
        
        # Learning and tracking
        self.observation_history = []
        self.action_history = []
        self.qs_history = []  # Belief history
        self.learning_iterations = 0
        
        logger.info(f"PyMDP Active Inference Agent initialized for user {user_id}")
    
    def _initialize_observation_model(self) -> np.ndarray:
        """Initialize observation model A matrices."""
        # Initialize A matrices for each observation modality using proper pymdp format
        A = utils.obj_array([len(self.num_obs)])
        
        for g in range(len(self.num_obs)):
            # Create A matrix with proper shape: [num_obs_g, num_states[0], num_states[1], ...]
            shape = [self.num_obs[g]] + self.num_states
            A[g] = np.random.dirichlet(np.ones(self.num_obs[g]), size=np.prod(self.num_states)).T.reshape(shape)
            # Normalize to ensure proper probability distributions
            A[g] = utils.norm_dist(A[g])
        
        return A
    
    def _initialize_transition_model(self) -> np.ndarray:
        """Initialize transition model B matrices."""
        # Initialize B matrices for each hidden state factor using proper pymdp format
        B = utils.obj_array([len(self.num_states)])
        
        for f in range(len(self.num_states)):
            B[f] = np.zeros((self.num_states[f], self.num_states[f], self.num_actions[0]))
            for action in range(self.num_actions[0]):
                # Initialize with identity + small random transitions
                identity_part = np.eye(self.num_states[f]) * 0.8
                random_part = np.random.dirichlet(np.ones(self.num_states[f]), size=self.num_states[f]).T * 0.2
                B[f][:, :, action] = identity_part + random_part
                # Normalize columns
                B[f][:, :, action] = utils.norm_dist(B[f][:, :, action])
        
        return B
    
    def _initialize_preferences(self) -> np.ndarray:
        """Initialize preference vectors C."""
        # Initialize C vectors for each observation modality using proper pymdp format
        C = utils.obj_array([len(self.num_obs)])
        
        # Set preferences for each modality
        # Digital behavior preferences (0: passive, 1: active, 2: social, 3: productive)
        C[0] = np.array([0.2, 0.4, 0.8, 1.0])  # Prefer productive digital behavior
        
        # Location context preferences (0: home, 1: work, 2: social, 3: transit, 4: other)
        C[1] = np.array([0.8, 0.6, 0.9, 0.3, 0.4])  # Prefer home and social locations
        
        # Social context preferences (0: alone, 1: family, 2: friends, 3: colleagues)
        C[2] = np.array([0.5, 0.9, 0.8, 0.7])  # Prefer social interactions
        
        # Content preferences (0: entertainment, 1: educational, 2: work)
        C[3] = np.array([0.6, 0.8, 0.5])  # Prefer educational content
        
        return C
    
    def _initialize_priors(self) -> np.ndarray:
        """Initialize prior beliefs D."""
        # Initialize D vectors for each hidden state factor using proper pymdp format
        D = utils.obj_array([len(self.num_states)])
        
        # Energy levels (0: low, 1: medium, 2: high)
        D[0] = np.array([0.2, 0.6, 0.2])  # Prior belief: mostly medium energy
        
        # Mood (0: negative, 1: neutral, 2: positive)
        D[1] = np.array([0.1, 0.7, 0.2])  # Prior belief: mostly neutral mood
        
        # Context (0: work, 1: personal, 2: social, 3: health, 4: learning)
        D[2] = np.array([0.3, 0.3, 0.2, 0.1, 0.1])  # Prior belief: work/personal contexts most common
        
        return D
    
    def encode_observations(self, system_state: Dict[str, Any]) -> List[int]:
        """
        Encode system state into observation indices for pymdp.
        
        Args:
            system_state: Current system state from DataAnalyzer
            
        Returns:
            List of observation indices for each modality
        """
        observations = []
        
        # Digital behavior modality
        digital_activity = system_state.get("recent_content_items", 0)
        if digital_activity == 0:
            digital_obs = 0  # passive
        elif digital_activity < 3:
            digital_obs = 1  # active
        elif digital_activity < 6:
            digital_obs = 2  # social
        else:
            digital_obs = 3  # productive
        observations.append(digital_obs)
        
        # Location context modality  
        hour = system_state.get("hour_of_day", 12)
        if 9 <= hour <= 17:
            location_obs = 1  # work
        elif 18 <= hour <= 22:
            location_obs = 2  # social
        elif 6 <= hour <= 8 or 23 <= hour <= 24:
            location_obs = 3  # transit
        else:
            location_obs = 0  # home
        observations.append(location_obs)
        
        # Social context modality
        active_events = system_state.get("active_events", 0)
        total_people = system_state.get("total_people", 0)
        if active_events > 0 and total_people > 2:
            social_obs = 2  # friends
        elif active_events > 0 and total_people > 0:
            social_obs = 1  # family
        elif 9 <= hour <= 17:
            social_obs = 3  # colleagues
        else:
            social_obs = 0  # alone
        observations.append(social_obs)
        
        # Content modality
        if 9 <= hour <= 17:
            content_obs = 2  # work
        elif digital_activity > 5:
            content_obs = 1  # educational
        else:
            content_obs = 0  # entertainment
        observations.append(content_obs)
        
        return observations
    
    def predict_and_suggest(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction and suggestion pipeline using pymdp.
        
        Args:
            system_state: Current system state
            
        Returns:
            Predictions and suggestions with confidence metrics
        """
        # Encode observations
        obs = self.encode_observations(system_state)
        
        # Infer current states
        qs = self.agent.infer_states(obs)
        
        # Infer policies and select action
        q_pi, G = self.agent.infer_policies()
        action = self.agent.sample_action()
        
        # Calculate confidence metrics
        state_confidence = self._calculate_state_confidence(qs)
        policy_confidence = np.max(q_pi)
        
        # Generate human-readable suggestions
        suggestion = self._interpret_action(action[0], qs, system_state)
        
        # Decompose expected free energy
        efe_breakdown = self._decompose_efe(G, q_pi)
        
        result = {
            'current_beliefs': {
                'energy_level': self._interpret_state_factor(qs[0], ['low', 'medium', 'high']),
                'mood': self._interpret_state_factor(qs[1], ['negative', 'neutral', 'positive']),
                'context': self._interpret_state_factor(qs[2], ['work', 'personal', 'social', 'health', 'learning'])
            },
            'suggested_action': suggestion,
            'confidence': float(state_confidence * policy_confidence),
            'state_confidence': float(state_confidence),
            'policy_confidence': float(policy_confidence),
            'efe_breakdown': efe_breakdown,
            'timestamp': datetime.now().isoformat(),
            'user_id': self.user_id
        }
        
        # Store for learning
        self.observation_history.append(obs)
        self.action_history.append(action)
        self.qs_history.append(qs)
        
        return result
    
    def _calculate_state_confidence(self, qs: List[np.ndarray]) -> float:
        """Calculate confidence based on state belief sharpness."""
        confidences = []
        for q in qs:
            # Confidence is inverse of entropy (sharpness of distribution)
            entropy = -np.sum(q * np.log(q + 1e-16))
            max_entropy = np.log(len(q))
            confidence = 1.0 - (entropy / max_entropy)
            confidences.append(confidence)
        return float(np.mean(confidences))
    
    def _interpret_state_factor(self, beliefs: np.ndarray, labels: List[str]) -> Dict[str, float]:
        """Interpret state beliefs with labels."""
        return {label: float(prob) for label, prob in zip(labels, beliefs)}
    
    def _interpret_action(self, action_idx: int, qs: List[np.ndarray], system_state: Dict[str, Any]) -> str:
        """Interpret action index into meaningful suggestion."""
        actions = [
            "Take a break and rest to restore energy",
            "Engage in light physical activity", 
            "Focus on important work tasks",
            "Connect with friends or family",
            "Learn something new",
            "Organize your environment",
            "Reflect and plan ahead",
            "Engage in creative activities",
            "Maintain current activity pattern"
        ]
        
        base_action = actions[min(action_idx, len(actions) - 1)]
        
        # Add context based on current beliefs
        energy_level = np.argmax(qs[0])
        mood_level = np.argmax(qs[1])
        
        if energy_level == 0:  # low energy
            base_action += " (prioritize rest and recovery)"
        elif energy_level == 2:  # high energy
            base_action += " (good time for challenging activities)"
            
        if mood_level == 0:  # negative mood
            base_action += " - consider mood-boosting activities"
        elif mood_level == 2:  # positive mood
            base_action += " - leverage your positive state"
        
        return base_action
    
    def _decompose_efe(self, G: np.ndarray, q_pi: np.ndarray) -> Dict[str, float]:
        """Decompose expected free energy into components."""
        # Get the expected free energy for the selected policy
        selected_policy_idx = np.argmax(q_pi)
        selected_efe = G[selected_policy_idx]
        
        return {
            'total_efe': float(selected_efe),
            'policy_probability': float(q_pi[selected_policy_idx]),
            'relative_efe': float(selected_efe - np.min(G)),
            'policy_diversity': float(-np.sum(q_pi * np.log(q_pi + 1e-16)))
        }
    
    def update_model(self, observations: List[int], actions: List[int] = None) -> Dict[str, Any]:
        """
        Update model parameters using Dirichlet learning.
        
        Args:
            observations: Observed outcomes
            actions: Actions that were taken
            
        Returns:
            Learning metrics
        """
        if len(self.qs_history) == 0:
            return {"status": "No prior beliefs to update from"}
        
        # Get the most recent beliefs
        qs_prev = self.qs_history[-1]
        
        # Update A matrices (observation model) using Dirichlet learning
        if len(observations) == len(self.num_obs):
            for g, obs in enumerate(observations):
                # Dirichlet update for observation model
                # Update the column corresponding to the inferred state
                state_idx = tuple(np.argmax(q) for q in qs_prev)
                
                # Add count to the observed outcome
                update_vector = np.zeros(self.num_obs[g])
                update_vector[obs] = self.learning_rate
                
                # Update A matrix
                flat_state_idx = np.ravel_multi_index(state_idx, self.num_states)
                A_column = self.A[g].reshape(self.num_obs[g], -1)[:, flat_state_idx]
                A_column += update_vector
                
                # Renormalize
                A_column = A_column / np.sum(A_column)
                self.A[g].reshape(self.num_obs[g], -1)[:, flat_state_idx] = A_column
        
        # Update B matrices if actions are provided
        if actions is not None and len(self.qs_history) > 1:
            qs_curr = self.qs_history[-1]
            qs_prev = self.qs_history[-2]
            
            for f in range(len(self.num_states)):
                action = actions[0] if len(actions) > 0 else 0
                
                # Dirichlet update for transition model
                prev_state = np.argmax(qs_prev[f])
                curr_state = np.argmax(qs_curr[f])
                
                # Add count to the observed transition
                self.B[f][curr_state, prev_state, action] += self.learning_rate
                
                # Renormalize column
                col_sum = np.sum(self.B[f][:, prev_state, action])
                if col_sum > 0:
                    self.B[f][:, prev_state, action] /= col_sum
        
        # Update agent with new matrices
        self.agent.A = self.A
        self.agent.B = self.B
        
        self.learning_iterations += 1
        
        return {
            "learning_iterations": self.learning_iterations,
            "model_entropy": self._calculate_model_entropy(),
            "update_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_model_entropy(self) -> float:
        """Calculate entropy of the model parameters (measure of uncertainty)."""
        total_entropy = 0.0
        
        # A matrix entropy
        for g in range(len(self.A)):
            A_flat = self.A[g].reshape(self.num_obs[g], -1)
            for col in range(A_flat.shape[1]):
                col_entropy = -np.sum(A_flat[:, col] * np.log(A_flat[:, col] + 1e-16))
                total_entropy += col_entropy
        
        return float(total_entropy)
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """Get metrics about the agent's learning progress."""
        if not self.qs_history:
            return {"status": "No learning data available"}
        
        # Calculate confidence trends
        recent_confidences = [self._calculate_state_confidence(qs) for qs in self.qs_history[-10:]]
        
        return {
            "total_observations": len(self.observation_history),
            "learning_iterations": self.learning_iterations,
            "recent_confidence": float(np.mean(recent_confidences)) if recent_confidences else 0.0,
            "confidence_trend": "improving" if len(recent_confidences) > 1 and recent_confidences[-1] > recent_confidences[0] else "stable",
            "model_entropy": self._calculate_model_entropy(),
            "user_id": self.user_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_model(self) -> Dict[str, Any]:
        """Save model parameters for versioning."""
        model_data = {
            "user_id": self.user_id,
            "A_matrices": [A.tolist() for A in self.A],
            "B_matrices": [B.tolist() for B in self.B], 
            "C_vectors": [C.tolist() for C in self.C],
            "D_vectors": [D.tolist() for D in self.D],
            "learning_iterations": self.learning_iterations,
            "num_obs": self.num_obs,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "saved_at": datetime.now().isoformat()
        }
        return model_data
    
    def load_model(self, model_data: Dict[str, Any]):
        """Load model parameters from saved data."""
        self.A = utils.obj_array([np.array(A) for A in model_data["A_matrices"]])
        self.B = utils.obj_array([np.array(B) for B in model_data["B_matrices"]])
        self.C = utils.obj_array([np.array(C) for C in model_data["C_vectors"]])
        self.D = utils.obj_array([np.array(D) for D in model_data["D_vectors"]])
        self.learning_iterations = model_data.get("learning_iterations", 0)
        
        # Update agent with loaded parameters
        self.agent.A = self.A
        self.agent.B = self.B
        self.agent.C = self.C
        self.agent.D = self.D
        
        logger.info(f"Model loaded for user {self.user_id} with {self.learning_iterations} learning iterations")


# Alias for compatibility
CIActiveInferenceAgent = PyMDPActiveInferenceAgent


class CIBehaviorOptimizer:
    """
    High-level behavior optimization system using Active Inference with pymdp.
    """
    
    def __init__(self, agent: PyMDPActiveInferenceAgent, data_analyzer):
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
        
        # Get predictions and recommendations from pymdp agent
        recommendations = self.agent.predict_and_suggest(current_state)
        
        optimization_result = {
            "person_id": person_id,
            "current_patterns": patterns,
            "recommendations": recommendations,
            "optimization_score": self._calculate_optimization_score(patterns, recommendations),
            "timestamp": datetime.now().isoformat()
        }
        
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def optimize_resource_allocation(self) -> Dict[str, Any]:
        """
        Optimize system-wide resource allocation using pymdp predictions.
        
        Returns:
            Resource optimization recommendations
        """
        current_state = self.data_analyzer.get_system_state_vector()
        
        # Get predictions from the agent
        predictions = self.agent.predict_and_suggest(current_state)
        
        # Generate allocation recommendations based on confidence and beliefs
        allocation_recommendations = []
        
        current_beliefs = predictions.get('current_beliefs', {})
        confidence = predictions.get('confidence', 0.0)
        
        # Make recommendations based on beliefs about energy and context
        energy_beliefs = current_beliefs.get('energy_level', {})
        context_beliefs = current_beliefs.get('context', {})
        
        if energy_beliefs.get('low', 0) > 0.6:
            allocation_recommendations.append({
                "action": "reduce_workload",
                "urgency": "high",
                "confidence": confidence,
                "reason": "Low energy state predicted"
            })
        elif energy_beliefs.get('high', 0) > 0.6:
            allocation_recommendations.append({
                "action": "increase_challenging_tasks", 
                "urgency": "medium",
                "confidence": confidence,
                "reason": "High energy state predicted"
            })
        
        if context_beliefs.get('work', 0) > 0.7:
            allocation_recommendations.append({
                "action": "optimize_work_resources",
                "urgency": "medium", 
                "confidence": confidence,
                "reason": "Work context highly probable"
            })
        
        return {
            "predictions": predictions,
            "recommendations": allocation_recommendations,
            "optimization_score": confidence,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_optimization_score(self, patterns: Dict[str, Any], 
                                    recommendations: Dict[str, Any]) -> float:
        """Calculate optimization score based on patterns and recommendations."""
        base_score = 0.5
        
        # Reward consistent patterns
        if patterns.get("total_events", 0) > 0:
            base_score += 0.2
        
        # Reward confident recommendations
        confidence = recommendations.get("confidence", 0)
        base_score += confidence * 0.3
        
        return min(base_score, 1.0)