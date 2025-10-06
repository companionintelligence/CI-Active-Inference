"""
Simple Active Inference agent for CI system without external dependencies.

This module provides a basic Active Inference implementation that works
with the CI database and matrix utilities.
"""

import sqlite3
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from ..utils.matrix_utils import SimpleActiveInferenceMatrices
from ..utils.discretization import DataDiscretizer
from ..utils.matrix_builder import MatrixBuilder


class SimpleActiveInferenceAgent:
    """
    Simple Active Inference agent using matrix utilities and database integration.
    
    This agent builds A/B/C/D matrices from CI database data and performs
    basic Active Inference operations for prediction and action selection.
    """
    
    def __init__(self, user_id: str = None):
        """
        Initialize the agent.
        
        Args:
            user_id: User ID for database queries (creates default if None)
        """
        self.user_id = user_id or "default_user"
        self.discretizer = DataDiscretizer()
        self.matrix_builder = MatrixBuilder(self.user_id, self.discretizer)
        self.matrices = None
        self.last_update = None
        
        # Learning tracking
        self.observation_count = 0
        self.recent_predictions = []
    
    def initialize_from_data(self, conn: sqlite3.Connection, 
                           start_date: str = None, end_date: str = None):
        """
        Initialize matrices from database data.
        
        Args:
            conn: Database connection
            start_date: Start date for data extraction (default: 30 days ago)
            end_date: End date for data extraction (default: now)
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).isoformat()
        if end_date is None:
            end_date = datetime.now().isoformat()
        
        # Build matrices from data
        self.matrices = self.matrix_builder.build_matrices_from_data(
            conn, start_date, end_date
        )
        self.last_update = datetime.now()
    
    def encode_observations(self, system_state: Dict[str, Any]) -> List[int]:
        """
        Encode system state into discrete observation indices.
        
        Args:
            system_state: Current system state
        
        Returns:
            List of observation indices for each modality
        """
        if not system_state:
            return [0, 0, 0, 0]  # Default observations
        
        # Extract relevant features
        digital_behavior = system_state.get('digital_behavior', 'other')
        location_context = system_state.get('location_context', 'other')
        social_context = system_state.get('social_context', 'alone')
        content_type = system_state.get('content_type', 'other')
        
        # Map to discrete indices
        observations = [
            self.discretizer.map_value_to_state(digital_behavior, self.discretizer.digital_mapping),
            self.discretizer.map_value_to_state(location_context, self.discretizer.location_mapping),
            self.discretizer.map_value_to_state(social_context, self.discretizer.social_mapping),
            self.discretizer.map_value_to_state(content_type, self.discretizer.content_mapping)
        ]
        
        return observations
    
    def predict_future_state(self, current_state: Dict[str, Any], 
                           time_steps: int = 5) -> List[Dict[str, Any]]:
        """
        Predict future states using Active Inference with enhanced multi-step planning.
        
        Args:
            current_state: Current system state
            time_steps: Number of steps to predict
        
        Returns:
            List of predicted future states
        """
        if self.matrices is None:
            return [{"error": "Agent not initialized with data"}] * time_steps
        
        # Encode current observations
        observations = self.encode_observations(current_state)
        
        # Infer current states
        beliefs = self.matrices.infer_states(observations)
        
        predictions = []
        current_beliefs = beliefs
        
        for t in range(time_steps):
            # Sample action based on current beliefs
            action = self.matrices.sample_action(current_beliefs)
            
            # Predict next state distribution for each factor
            next_beliefs = []
            for factor in range(len(self.matrices.num_states)):
                next_belief = [0.0] * self.matrices.num_states[factor]
                for curr_state in range(self.matrices.num_states[factor]):
                    for next_state in range(self.matrices.num_states[factor]):
                        transition_prob = self.matrices.B[factor][next_state][curr_state][action]
                        next_belief[next_state] += current_beliefs[factor][curr_state] * transition_prob
                next_beliefs.append(next_belief)
            
            # Generate predicted observations
            predicted_obs = self._predict_observations(next_beliefs)
            
            # Convert to state format
            predicted_state = self._decode_predicted_state(
                predicted_obs, next_beliefs, current_state, t + 1
            )
            predictions.append(predicted_state)
            
            current_beliefs = next_beliefs
        
        # Store for learning
        self.recent_predictions.extend(predictions)
        if len(self.recent_predictions) > 50:
            self.recent_predictions = self.recent_predictions[-50:]
        
        return predictions
    
    def get_multi_step_plan(self, current_state: Dict[str, Any], 
                           horizon: int = 10) -> Dict[str, Any]:
        """
        Generate sophisticated multi-step plan using advanced planning algorithms.
        
        Args:
            current_state: Current system state
            horizon: Planning horizon
            
        Returns:
            Multi-step plan with policy tree and utilities
        """
        if self.matrices is None:
            return {"error": "Agent not initialized with data"}
        
        try:
            from ..planning.multi_step_planner import MultiStepPlanner
            
            # Create planner with current matrices
            planner = MultiStepPlanner(self.matrices, max_horizon=horizon)
            
            # Encode current observations and infer beliefs
            observations = self.encode_observations(current_state)
            beliefs = self.matrices.infer_states(observations)
            
            # Generate multi-step plan
            action_sequence, expected_utility = planner.plan_multi_step(beliefs, horizon)
            
            # Get planning statistics
            planning_stats = planner.get_planning_statistics()
            
            return {
                "action_sequence": action_sequence,
                "expected_utility": expected_utility,
                "horizon": horizon,
                "planning_stats": planning_stats,
                "current_beliefs": beliefs,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Multi-step planning failed: {str(e)}"}
    
    def get_detailed_explanation(self, current_state: Dict[str, Any], 
                               suggestion_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate detailed explanation for agent suggestion.
        
        Args:
            current_state: Current system state
            suggestion_result: Result from recommend_actions
            
        Returns:
            Detailed explanation with causality analysis
        """
        if self.matrices is None:
            return {"error": "Agent not initialized with data"}
        
        try:
            from ..explanation.explanation_generator import ExplanationGenerator
            
            # Create explanation generator
            generator = ExplanationGenerator(self.matrices)
            
            # Encode current observations and infer beliefs
            observations = self.encode_observations(current_state)
            beliefs = self.matrices.infer_states(observations)
            
            # Generate comprehensive explanation
            explanation = generator.generate_detailed_explanation(
                current_state=current_state,
                suggested_action=suggestion_result.get("suggested_action", "no_action"),
                predicted_default=suggestion_result.get("predicted_default", "unknown"),
                confidence=suggestion_result.get("confidence", 0.5),
                efe_breakdown=suggestion_result.get("efe_breakdown", {}),
                beliefs=beliefs
            )
            
            return explanation
            
        except Exception as e:
            return {"error": f"Explanation generation failed: {str(e)}"}
    
    def recommend_actions(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend optimal actions based on current state.
        
        Args:
            current_state: Current system state
        
        Returns:
            Action recommendations
        """
        if self.matrices is None:
            return {"error": "Agent not initialized with data"}
        
        # Encode observations and infer states
        observations = self.encode_observations(current_state)
        beliefs = self.matrices.infer_states(observations)
        
        # Sample optimal action
        action = self.matrices.sample_action(beliefs)
        
        # Get action description
        action_mapping = self.matrix_builder.action_mapping
        action_names = {v: k for k, v in action_mapping.items()}
        action_type = action_names.get(action, "unknown_action")
        
        # Calculate confidence
        confidence = max(max(factor_beliefs) for factor_beliefs in beliefs)
        
        return {
            "action_type": action_type,
            "action_index": action,
            "confidence": float(confidence),
            "current_beliefs": beliefs,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_beliefs(self, observations: Dict[str, Any]):
        """
        Update agent beliefs based on new observations.
        
        Args:
            observations: New observation data
        """
        if self.matrices is None:
            return
        
        # Encode observations
        obs_indices = self.encode_observations(observations)
        
        # Perform inference (this updates internal beliefs)
        beliefs = self.matrices.infer_states(obs_indices)
        
        self.observation_count += 1
    
    def get_learning_progress(self) -> Dict[str, Any]:
        """
        Get learning progress metrics.
        
        Returns:
            Progress information
        """
        recent_confidence = 0.5
        if self.recent_predictions:
            confidences = [p.get('confidence', 0.5) for p in self.recent_predictions 
                          if 'confidence' in p]
            if confidences:
                recent_confidence = sum(confidences) / len(confidences)
        
        return {
            "total_observations": self.observation_count,
            "recent_confidence": float(recent_confidence),
            "confidence_trend": "stable",
            "learning_rate": 0.1,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "matrix_initialized": self.matrices is not None
        }
    
    def _predict_observations(self, beliefs: List[List[float]]) -> List[int]:
        """
        Predict most likely observations given state beliefs.
        
        Args:
            beliefs: Beliefs over each state factor
        
        Returns:
            Most likely observation for each modality
        """
        predicted_obs = []
        
        for modality in range(len(self.matrices.num_obs)):
            # Calculate expected observation
            obs_probs = [0.0] * self.matrices.num_obs[modality]
            
            # For simplicity, assume single factor dominates (could be improved)
            dominant_factor = 0  # Use first factor (energy)
            if len(beliefs) > dominant_factor:
                for obs in range(self.matrices.num_obs[modality]):
                    for state in range(self.matrices.num_states[dominant_factor]):
                        if len(self.matrices.A[modality]) > obs and len(self.matrices.A[modality][obs]) > state:
                            likelihood = self.matrices.A[modality][obs][state] if isinstance(self.matrices.A[modality][obs], list) else 0.0
                            obs_probs[obs] += beliefs[dominant_factor][state] * likelihood
            
            # Select most likely observation
            predicted_obs.append(obs_probs.index(max(obs_probs)) if obs_probs else 0)
        
        return predicted_obs
    
    def _decode_predicted_state(self, observations: List[int], beliefs: List[List[float]], 
                              reference_state: Dict[str, Any], time_step: int) -> Dict[str, Any]:
        """
        Convert predicted observations back to state dictionary.
        
        Args:
            observations: Predicted observation indices
            beliefs: State beliefs
            reference_state: Reference state for context
            time_step: Time step into future
        
        Returns:
            Predicted state dictionary
        """
        # Map observations back to categories
        obs_mappings = [
            self.discretizer.digital_mapping,
            self.discretizer.location_mapping,
            self.discretizer.social_mapping,
            self.discretizer.content_mapping
        ]
        
        predicted_state = reference_state.copy()
        
        # Add predicted categorical values
        for i, (obs_idx, mapping) in enumerate(zip(observations, obs_mappings)):
            # Reverse lookup
            category_names = {v: k for k, v in mapping.mapping.items()}
            category = category_names.get(obs_idx, "other")
            
            if i == 0:
                predicted_state["predicted_digital_behavior"] = category
            elif i == 1:
                predicted_state["predicted_location_context"] = category
            elif i == 2:
                predicted_state["predicted_social_context"] = category
            elif i == 3:
                predicted_state["predicted_content_type"] = category
        
        # Add confidence and metadata
        confidence = max(max(factor_beliefs) for factor_beliefs in beliefs)
        predicted_state.update({
            "confidence": float(confidence),
            "time_step": time_step,
            "timestamp": (datetime.now() + timedelta(hours=time_step)).isoformat(),
            "prediction_type": "active_inference"
        })
        
        return predicted_state