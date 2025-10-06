"""
Matrix builder for Active Inference from CI database data.

This module constructs A, B, C, D matrices from discretized temporal data
extracted from the CI database.
"""

import sqlite3
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta

from .matrix_utils import SimpleActiveInferenceMatrices, normalize, add_laplace_smoothing
from .discretization import DataDiscretizer


class MatrixBuilder:
    """
    Builds Active Inference matrices from CI database data.
    
    Constructs observation likelihood (A), state transition (B), preference (C),
    and prior belief (D) matrices from temporal behavioral data.
    """
    
    def __init__(self, user_id: str, discretizer: Optional[DataDiscretizer] = None):
        """
        Initialize matrix builder for a specific user.
        
        Args:
            user_id: User ID to build matrices for
            discretizer: Data discretizer instance (creates new if None)
        """
        self.user_id = user_id
        self.discretizer = discretizer or DataDiscretizer()
        
        # Get dimensions from discretizer
        self.num_obs = self.discretizer.get_observation_dimensions()
        self.num_states = self.discretizer.get_state_factor_dimensions()
        self.num_actions = 9  # Standard set of actions (see _get_action_mapping)
        
        # Initialize matrix container
        self.matrices = SimpleActiveInferenceMatrices(
            num_obs=self.num_obs,
            num_states=self.num_states,
            num_actions=self.num_actions
        )
        
        self.action_mapping = self._get_action_mapping()
    
    def _get_action_mapping(self) -> Dict[str, int]:
        """
        Define mapping from action types to indices.
        
        Returns:
            Dictionary mapping action names to indices
        """
        return {
            "work_focus": 0,        # Focus on work tasks
            "social_engage": 1,     # Engage in social activities
            "physical_exercise": 2, # Physical activity/exercise
            "rest_recover": 3,      # Rest and recovery
            "learn_skill": 4,       # Learning and skill development
            "creative_work": 5,     # Creative activities
            "maintain_health": 6,   # Health maintenance
            "organize_plan": 7,     # Organization and planning
            "no_action": 8          # Take no specific action
        }
    
    def build_A_matrices_from_data(self, conn: sqlite3.Connection, 
                                  start_date: str, end_date: str,
                                  smoothing_alpha: float = 0.01) -> List:
        """
        Build observation likelihood matrices from temporal data.
        
        Args:
            conn: Database connection
            start_date: Start date for data extraction
            end_date: End date for data extraction
            smoothing_alpha: Laplace smoothing parameter
        
        Returns:
            List of A matrices for each observation modality
        """
        # Extract and discretize temporal features
        features = self.discretizer.extract_temporal_features(
            conn, self.user_id, start_date, end_date
        )
        discretized = self.discretizer.discretize_features(features)
        
        # Initialize A matrices with smoothing
        A_matrices = []
        for modality in range(len(self.num_obs)):
            # Shape: (num_observations, *num_states)
            if len(self.num_states) == 1:
                shape = (self.num_obs[modality], self.num_states[0])
            elif len(self.num_states) == 2:
                shape = (self.num_obs[modality], self.num_states[0], self.num_states[1])
            else:  # 3 state factors [Energy, Mood, Context]
                shape = (self.num_obs[modality], self.num_states[0], self.num_states[1], self.num_states[2])
            
            # Initialize with smoothing
            if len(shape) == 2:
                matrix = [[smoothing_alpha for _ in range(shape[1])] for _ in range(shape[0])]
            elif len(shape) == 3:
                matrix = [[[smoothing_alpha for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
            else:  # 4D
                matrix = [[[[smoothing_alpha for _ in range(shape[3])] for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
            
            A_matrices.append(matrix)
        
        # Fill matrices with data
        for record in discretized:
            # Infer energy and mood states from context and time
            energy_state = self._infer_energy_state(record)
            mood_state = self._infer_mood_state(record)
            context_state = record.get('context_idx', 0)
            
            # Update A matrices
            # Digital behavior observations
            digital_obs = record.get('digital_idx', 0)
            if len(self.num_states) == 3:
                A_matrices[0][digital_obs][energy_state][mood_state][context_state] += 1.0
            
            # Location observations  
            location_obs = record.get('location_idx', 0)
            if len(A_matrices) > 1:
                A_matrices[1][location_obs][energy_state][mood_state][context_state] += 1.0
            
            # Social observations
            social_obs = record.get('social_idx', 0)
            if len(A_matrices) > 2:
                A_matrices[2][social_obs][energy_state][mood_state][context_state] += 1.0
            
            # Content observations
            content_obs = record.get('content_idx', 0)
            if len(A_matrices) > 3:
                A_matrices[3][content_obs][energy_state][mood_state][context_state] += 1.0
        
        # Normalize matrices (each column should sum to 1)
        for i, matrix in enumerate(A_matrices):
            A_matrices[i] = self._normalize_A_matrix(matrix)
        
        return A_matrices
    
    def build_B_matrices_from_data(self, conn: sqlite3.Connection,
                                  start_date: str, end_date: str,
                                  smoothing_alpha: float = 0.01) -> List:
        """
        Build state transition matrices from temporal sequences.
        
        Args:
            conn: Database connection
            start_date: Start date for data extraction
            end_date: End date for data extraction
            smoothing_alpha: Laplace smoothing parameter
        
        Returns:
            List of B matrices for each state factor
        """
        # Extract temporal sequences
        features = self.discretizer.extract_temporal_features(
            conn, self.user_id, start_date, end_date
        )
        discretized = self.discretizer.discretize_features(features)
        
        # Sort by time to create sequences
        discretized.sort(key=lambda x: x.get('time_bucket', ''))
        
        # Initialize B matrices
        B_matrices = []
        for factor in range(len(self.num_states)):
            # Shape: (num_states, num_states, num_actions)
            n_states = self.num_states[factor]
            matrix = [[[smoothing_alpha for _ in range(self.num_actions)] 
                      for _ in range(n_states)] 
                      for _ in range(n_states)]
            B_matrices.append(matrix)
        
        # Build transition sequences
        for i in range(len(discretized) - 1):
            current = discretized[i]
            next_record = discretized[i + 1]
            
            # Infer states for current and next time steps
            current_energy = self._infer_energy_state(current)
            current_mood = self._infer_mood_state(current)
            current_context = current.get('context_idx', 0)
            
            next_energy = self._infer_energy_state(next_record)
            next_mood = self._infer_mood_state(next_record)
            next_context = next_record.get('context_idx', 0)
            
            # Infer action from transition
            action = self._infer_action_from_transition(current, next_record)
            
            # Update B matrices
            # Energy transitions
            B_matrices[0][next_energy][current_energy][action] += 1.0
            
            # Mood transitions
            if len(B_matrices) > 1:
                B_matrices[1][next_mood][current_mood][action] += 1.0
            
            # Context transitions
            if len(B_matrices) > 2:
                B_matrices[2][next_context][current_context][action] += 1.0
        
        # Normalize matrices (each [current_state, action] column should sum to 1)
        for i, matrix in enumerate(B_matrices):
            B_matrices[i] = self._normalize_B_matrix(matrix)
        
        return B_matrices
    
    def build_C_vectors(self) -> List[List[float]]:
        """
        Build preference vectors based on expected user preferences.
        
        Returns:
            List of preference vectors for each observation modality
        """
        C_vectors = []
        
        # Digital behavior preferences (prefer work-focus and communication over entertainment)
        digital_prefs = [2.0, 2.0, 0.0, 1.0]  # [work-focus, communication, entertainment, other]
        C_vectors.append(digital_prefs)
        
        # Location preferences (neutral - no strong preference)
        location_prefs = [0.0] * self.num_obs[1]
        C_vectors.append(location_prefs)
        
        # Social preferences (slight preference for small groups)
        social_prefs = [0.0, 1.0, 0.5]  # [alone, small-group, large-group]
        C_vectors.append(social_prefs)
        
        # Content preferences (prefer work and communication content)
        content_prefs = [2.0, 0.0, 1.0, 0.0]  # [work, entertainment, communication, other]
        C_vectors.append(content_prefs)
        
        return C_vectors
    
    def build_D_vectors(self, conn: sqlite3.Connection, 
                       start_date: str, end_date: str) -> List[List[float]]:
        """
        Build prior belief vectors from historical start-of-day data.
        
        Args:
            conn: Database connection
            start_date: Start date for data extraction
            end_date: End date for data extraction
        
        Returns:
            List of prior belief vectors for each state factor
        """
        # Extract morning data (6-9 AM) to establish priors
        morning_query = """
        SELECT 
            strftime('%H', e.start_time) as hour,
            COUNT(*) as event_count,
            AVG(CASE 
              WHEN e.end_time IS NOT NULL 
              THEN (strftime('%s', e.end_time) - strftime('%s', e.start_time)) / 60.0 
              ELSE 60 
            END) as avg_duration
        FROM events e
        WHERE e.user_id = ? 
          AND e.start_time BETWEEN ? AND ?
          AND CAST(strftime('%H', e.start_time) AS INTEGER) BETWEEN 6 AND 9
        GROUP BY hour
        ORDER BY hour
        """
        
        cursor = conn.cursor()
        cursor.execute(morning_query, (self.user_id, start_date, end_date))
        morning_data = cursor.fetchall()
        
        # Initialize with uniform priors
        D_vectors = []
        for factor in range(len(self.num_states)):
            n_states = self.num_states[factor]
            prior = [1.0 / n_states] * n_states
            D_vectors.append(prior)
        
        # Adjust priors based on morning patterns
        if morning_data:
            avg_morning_events = sum(row[1] for row in morning_data) / len(morning_data)
            
            # Energy prior (assume higher energy in morning if active)
            if avg_morning_events > 1:
                D_vectors[0] = [0.1, 0.3, 0.6]  # [low, medium, high] - more energy
            else:
                D_vectors[0] = [0.2, 0.6, 0.2]  # [low, medium, high] - moderate energy
            
            # Mood prior (assume neutral to positive in morning)
            if len(D_vectors) > 1:
                D_vectors[1] = [0.1, 0.4, 0.5]  # [negative, neutral, positive]
            
            # Context prior (based on typical morning activities)
            if len(D_vectors) > 2:
                D_vectors[2] = [0.4, 0.3, 0.1, 0.1, 0.1]  # [home, work, social, health, other]
        
        return D_vectors
    
    def _infer_energy_state(self, record: Dict[str, Any]) -> int:
        """
        Infer energy state from activity patterns.
        
        Args:
            record: Discretized temporal record
        
        Returns:
            Energy state index (0: low, 1: medium, 2: high)
        """
        activity_level = record.get('activity_level', 1)
        time_period = record.get('time_period', 1)
        
        # High activity suggests high energy
        if activity_level == 2:  # High activity
            return 2
        elif activity_level == 0:  # Low activity
            # Low activity at night is normal, but during day suggests low energy
            return 0 if time_period in [1, 2] else 1  # Morning/afternoon vs night/evening
        else:
            return 1  # Medium energy
    
    def _infer_mood_state(self, record: Dict[str, Any]) -> int:
        """
        Infer mood state from context and social patterns.
        
        Args:
            record: Discretized temporal record
        
        Returns:
            Mood state index (0: negative, 1: neutral, 2: positive)
        """
        context_idx = record.get('context_idx', 4)  # Default to 'other'
        social_idx = record.get('social_idx', 0)    # Default to 'alone'
        
        # Social activities and health activities suggest positive mood
        if context_idx == 2 or context_idx == 3:  # Social or health context
            return 2
        elif social_idx > 0:  # With other people
            return 2
        elif context_idx == 1:  # Work context - neutral to negative
            return 1
        else:
            return 1  # Default to neutral
    
    def _infer_action_from_transition(self, current: Dict[str, Any], 
                                    next_record: Dict[str, Any]) -> int:
        """
        Infer action taken based on transition between states.
        
        Args:
            current: Current state record
            next_record: Next state record
        
        Returns:
            Action index
        """
        current_context = current.get('context_idx', 4)
        next_context = next_record.get('context_idx', 4)
        
        # Infer action from context transitions
        if next_context == 1:  # Moving to work
            return self.action_mapping["work_focus"]
        elif next_context == 2:  # Moving to social
            return self.action_mapping["social_engage"]
        elif next_context == 3:  # Moving to health
            return self.action_mapping["physical_exercise"]
        elif next_context == 0:  # Moving to home
            return self.action_mapping["rest_recover"]
        else:
            return self.action_mapping["no_action"]
    
    def _normalize_A_matrix(self, matrix: List) -> List:
        """Normalize A matrix so each column sums to 1."""
        if isinstance(matrix[0], (int, float)):  # 1D
            return matrix
        elif isinstance(matrix[0][0], (int, float)):  # 2D
            return normalize(matrix, axis=0)
        else:  # Higher dimensional - normalize over first dimension
            # For now, just return as-is (would need recursive normalization)
            return matrix
    
    def _normalize_B_matrix(self, matrix: List) -> List:
        """Normalize B matrix so each [current_state, action] column sums to 1."""
        n_states = len(matrix)
        n_actions = len(matrix[0][0])
        
        for curr_state in range(n_states):
            for action in range(n_actions):
                # Normalize transition probabilities from current_state under action
                column_sum = sum(matrix[next_state][curr_state][action] 
                               for next_state in range(n_states))
                
                if column_sum > 0:
                    for next_state in range(n_states):
                        matrix[next_state][curr_state][action] /= column_sum
        
        return matrix
    
    def build_matrices_from_data(self, conn: sqlite3.Connection,
                               start_date: str, end_date: str) -> SimpleActiveInferenceMatrices:
        """
        Build complete set of Active Inference matrices from database data.
        
        Args:
            conn: Database connection
            start_date: Start date for data extraction
            end_date: End date for data extraction
        
        Returns:
            Complete matrix set for Active Inference
        """
        # Build all matrices
        A_matrices = self.build_A_matrices_from_data(conn, start_date, end_date)
        B_matrices = self.build_B_matrices_from_data(conn, start_date, end_date)
        C_vectors = self.build_C_vectors()
        D_vectors = self.build_D_vectors(conn, start_date, end_date)
        
        # Update the matrix container
        self.matrices.A = A_matrices
        self.matrices.B = B_matrices
        self.matrices.C = C_vectors
        self.matrices.D = D_vectors
        
        # Normalize all matrices
        self.matrices.normalize_matrices()
        
        return self.matrices