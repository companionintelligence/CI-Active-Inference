"""
Matrix utilities for Active Inference - simplified pymdp-compatible implementation.

This module provides utilities for constructing and manipulating the A, B, C, D matrices
needed for Active Inference without requiring the full pymdp library.
"""

import math
from typing import List, Dict, Any, Tuple, Optional, Union


def normalize(matrix: List[List[float]], axis: int = 0) -> List[List[float]]:
    """
    Normalize a matrix along the specified axis.
    
    Args:
        matrix: Input matrix as nested lists
        axis: Axis to normalize along (0 for columns, 1 for rows)
    
    Returns:
        Normalized matrix
    """
    if not matrix or not matrix[0]:
        return matrix
    
    rows, cols = len(matrix), len(matrix[0])
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    
    if axis == 0:  # Normalize columns
        for j in range(cols):
            col_sum = sum(matrix[i][j] for i in range(rows))
            if col_sum > 0:
                for i in range(rows):
                    result[i][j] = matrix[i][j] / col_sum
            else:
                for i in range(rows):
                    result[i][j] = 1.0 / rows  # Uniform distribution
    else:  # Normalize rows
        for i in range(rows):
            row_sum = sum(matrix[i][j] for j in range(cols))
            if row_sum > 0:
                for j in range(cols):
                    result[i][j] = matrix[i][j] / row_sum
            else:
                for j in range(cols):
                    result[i][j] = 1.0 / cols  # Uniform distribution
    
    return result


def create_uniform_matrix(shape: Tuple[int, ...]) -> List:
    """
    Create a uniform probability matrix with given shape.
    
    Args:
        shape: Dimensions of the matrix
    
    Returns:
        Matrix initialized with uniform probabilities
    """
    if len(shape) == 1:
        return [1.0 / shape[0]] * shape[0]
    elif len(shape) == 2:
        rows, cols = shape
        value = 1.0 / rows
        return [[value for _ in range(cols)] for _ in range(rows)]
    elif len(shape) == 3:
        # For 3D matrices (e.g., B matrices with actions)
        depth, rows, cols = shape
        value = 1.0 / rows
        return [[[value for _ in range(cols)] for _ in range(rows)] for _ in range(depth)]
    elif len(shape) == 4:
        # For 4D matrices (e.g., complex A matrices)
        d1, d2, d3, d4 = shape
        value = 1.0 / d1
        return [[[[value for _ in range(d4)] for _ in range(d3)] for _ in range(d2)] for _ in range(d1)]
    else:
        raise ValueError(f"Unsupported matrix shape: {shape}")


def add_laplace_smoothing(matrix: List, alpha: float = 0.01) -> List:
    """
    Add Laplace smoothing to a matrix to avoid zero probabilities.
    
    Args:
        matrix: Input matrix
        alpha: Smoothing parameter
    
    Returns:
        Smoothed matrix
    """
    if isinstance(matrix[0], (int, float)):  # 1D array
        return [x + alpha for x in matrix]
    elif isinstance(matrix[0][0], (int, float)):  # 2D array
        return [[x + alpha for x in row] for row in matrix]
    elif isinstance(matrix[0][0][0], (int, float)):  # 3D array
        return [[[x + alpha for x in row] for row in plane] for plane in matrix]
    else:  # 4D array
        return [[[[x + alpha for x in depth] for depth in row] for row in plane] for plane in matrix]


def softmax(x: List[float], temperature: float = 1.0) -> List[float]:
    """
    Apply softmax function to convert logits to probabilities.
    
    Args:
        x: Input logits
        temperature: Temperature parameter for softmax
    
    Returns:
        Probability distribution
    """
    if not x:
        return []
    
    # Subtract max for numerical stability
    max_x = max(x)
    exp_x = [math.exp((xi - max_x) / temperature) for xi in x]
    sum_exp = sum(exp_x)
    
    if sum_exp == 0:
        return [1.0 / len(x)] * len(x)
    
    return [xi / sum_exp for xi in exp_x]


def matrix_multiply_2d(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
    """
    Multiply two 2D matrices.
    
    Args:
        a: First matrix
        b: Second matrix
    
    Returns:
        Product matrix
    """
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("Invalid matrix dimensions for multiplication")
    
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    
    result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
    
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    
    return result


def kl_divergence(p: List[float], q: List[float]) -> float:
    """
    Calculate KL divergence between two probability distributions.
    
    Args:
        p: First distribution
        q: Second distribution
    
    Returns:
        KL divergence D(p||q)
    """
    if len(p) != len(q):
        raise ValueError("Distributions must have same length")
    
    kl = 0.0
    epsilon = 1e-10  # Small value to avoid log(0)
    
    for i in range(len(p)):
        if p[i] > epsilon:
            kl += p[i] * math.log(p[i] / max(q[i], epsilon))
    
    return kl


class SimpleActiveInferenceMatrices:
    """
    Simple Active Inference matrix container and operations.
    
    This class provides a minimal implementation of Active Inference matrices
    without requiring the full pymdp library.
    """
    
    def __init__(self, 
                 num_obs: List[int],
                 num_states: List[int], 
                 num_actions: int):
        """
        Initialize matrices with given dimensions.
        
        Args:
            num_obs: Number of observations for each modality
            num_states: Number of states for each factor
            num_actions: Number of actions
        """
        self.num_obs = num_obs
        self.num_states = num_states
        self.num_actions = num_actions
        
        # Initialize A matrices (observation likelihood)
        self.A = []
        for i, n_obs in enumerate(num_obs):
            # Shape: (n_obs, *num_states)
            if len(num_states) == 1:
                shape = (n_obs, num_states[0])
            else:
                shape = (n_obs, *num_states)
            self.A.append(create_uniform_matrix(shape))
        
        # Initialize B matrices (state transitions)
        self.B = []
        for i, n_states in enumerate(num_states):
            # Shape: (n_states, n_states, num_actions)
            shape = (n_states, n_states, num_actions)
            self.B.append(create_uniform_matrix(shape))
        
        # Initialize C vectors (preferences)
        self.C = []
        for n_obs in num_obs:
            # Preference over observations
            self.C.append([0.0] * n_obs)  # Start with neutral preferences
        
        # Initialize D vectors (prior beliefs)
        self.D = []
        for n_states in num_states:
            # Uniform prior over states
            self.D.append([1.0 / n_states] * n_states)
    
    def normalize_matrices(self):
        """Normalize all matrices to be proper probability distributions."""
        # Normalize A matrices (columns should sum to 1)
        for i in range(len(self.A)):
            if isinstance(self.A[i][0], list):  # 2D matrix
                self.A[i] = normalize(self.A[i], axis=0)
            # For higher dimensions, skip normalization for now
        
        # Normalize B matrices - each column [current_state, action] should sum to 1
        for i in range(len(self.B)):
            if len(self.B[i]) > 0 and isinstance(self.B[i][0], list):
                n_states = len(self.B[i])
                n_actions = len(self.B[i][0][0]) if len(self.B[i][0]) > 0 else 0
                
                for curr_state in range(n_states):
                    for action in range(n_actions):
                        # Normalize column [next_state] for (curr_state, action)
                        column_sum = sum(self.B[i][next_state][curr_state][action] 
                                       for next_state in range(n_states))
                        
                        if column_sum > 0:
                            for next_state in range(n_states):
                                self.B[i][next_state][curr_state][action] /= column_sum
        
        # Normalize D vectors
        for i in range(len(self.D)):
            total = sum(self.D[i])
            if total > 0:
                self.D[i] = [x / total for x in self.D[i]]
    
    def update_A_matrix(self, modality: int, obs_idx: int, state_indices: List[int], count: float):
        """
        Update A matrix with observed data.
        
        Args:
            modality: Which observation modality
            obs_idx: Index of observed value
            state_indices: Indices of hidden states
            count: Count to add
        """
        if len(state_indices) == 1:
            self.A[modality][obs_idx][state_indices[0]] += count
        else:
            # For multi-factor states, would need more complex indexing
            pass  # Simplified for now
    
    def update_B_matrix(self, factor: int, curr_state: int, next_state: int, action: int, count: float):
        """
        Update B matrix with transition data.
        
        Args:
            factor: Which state factor
            curr_state: Current state index
            next_state: Next state index
            action: Action taken
            count: Count to add
        """
        self.B[factor][next_state][curr_state][action] += count
    
    def infer_states(self, observations: List[int]) -> List[List[float]]:
        """
        Infer posterior beliefs over states given observations.
        
        Args:
            observations: Observed values for each modality
        
        Returns:
            Posterior beliefs for each state factor
        """
        beliefs = []
        
        for factor in range(len(self.num_states)):
            # Start with prior
            posterior = self.D[factor][:]
            
            # Update with likelihood for each observation modality
            for modality, obs in enumerate(observations):
                if modality < len(self.A) and obs < len(self.A[modality]):
                    # Multiply by likelihood
                    for state in range(self.num_states[factor]):
                        if len(self.num_states) == 1:
                            likelihood = self.A[modality][obs][state]
                        else:
                            # Simplified: assume independence for multi-factor case
                            likelihood = self.A[modality][obs][state] if state < len(self.A[modality][obs]) else 0.0
                        
                        posterior[state] *= likelihood
            
            # Normalize
            total = sum(posterior)
            if total > 0:
                posterior = [x / total for x in posterior]
            else:
                posterior = [1.0 / self.num_states[factor]] * self.num_states[factor]
            
            beliefs.append(posterior)
        
        return beliefs
    
    def sample_action(self, beliefs: List[List[float]], preferences: Optional[List[List[float]]] = None) -> int:
        """
        Sample an action based on expected free energy minimization.
        
        Args:
            beliefs: Current posterior beliefs
            preferences: Preferences over outcomes (uses self.C if None)
        
        Returns:
            Selected action index
        """
        if preferences is None:
            preferences = self.C
        
        # Calculate expected free energy for each action
        efe = []
        for action in range(self.num_actions):
            total_efe = 0.0
            
            # For each state factor
            for factor in range(len(self.num_states)):
                # Predict next state distribution
                next_beliefs = [0.0] * self.num_states[factor]
                for curr_state in range(self.num_states[factor]):
                    for next_state in range(self.num_states[factor]):
                        transition_prob = self.B[factor][next_state][curr_state][action]
                        next_beliefs[next_state] += beliefs[factor][curr_state] * transition_prob
                
                # Calculate expected observations
                for modality in range(len(self.num_obs)):
                    expected_obs = [0.0] * self.num_obs[modality]
                    for state in range(self.num_states[factor]):
                        for obs in range(self.num_obs[modality]):
                            if len(self.num_states) == 1:
                                likelihood = self.A[modality][obs][state]
                            else:
                                likelihood = self.A[modality][obs][state] if state < len(self.A[modality][obs]) else 0.0
                            expected_obs[obs] += next_beliefs[state] * likelihood
                    
                    # Add preference-based cost
                    if modality < len(preferences):
                        for obs in range(self.num_obs[modality]):
                            total_efe -= expected_obs[obs] * preferences[modality][obs]
            
            efe.append(total_efe)
        
        # Convert to action probabilities and sample
        action_probs = softmax([-x for x in efe])  # Negative because we want to minimize EFE
        
        # Sample from distribution (simplified: take argmax)
        return action_probs.index(max(action_probs))