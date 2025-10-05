"""
Unit tests for Active Inference matrix utilities.
"""

import sys
import os

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from ci_active_inference.utils.matrix_utils import (
    normalize, create_uniform_matrix, add_laplace_smoothing, 
    softmax, kl_divergence, SimpleActiveInferenceMatrices
)


class TestMatrixUtils:
    """Test matrix utility functions."""
    
    def test_normalize_2d_matrix(self):
        """Test 2D matrix normalization."""
        matrix = [[1.0, 2.0], [3.0, 4.0]]
        normalized = normalize(matrix, axis=0)
        
        # Check that columns sum to 1
        assert abs(normalized[0][0] + normalized[1][0] - 1.0) < 1e-10
        assert abs(normalized[0][1] + normalized[1][1] - 1.0) < 1e-10
    
    def test_create_uniform_matrix_2d(self):
        """Test creation of uniform 2D matrix."""
        matrix = create_uniform_matrix((3, 4))
        
        assert len(matrix) == 3
        assert len(matrix[0]) == 4
        assert abs(matrix[0][0] - 1.0/3.0) < 1e-10
    
    def test_create_uniform_matrix_3d(self):
        """Test creation of uniform 3D matrix."""
        matrix = create_uniform_matrix((2, 3, 4))
        
        assert len(matrix) == 2
        assert len(matrix[0]) == 3
        assert len(matrix[0][0]) == 4
        assert abs(matrix[0][0][0] - 1.0/3.0) < 1e-10
    
    def test_add_laplace_smoothing(self):
        """Test Laplace smoothing."""
        matrix = [[0.0, 1.0], [2.0, 0.0]]
        smoothed = add_laplace_smoothing(matrix, alpha=0.1)
        
        assert smoothed[0][0] == 0.1
        assert smoothed[0][1] == 1.1
        assert smoothed[1][0] == 2.1
        assert smoothed[1][1] == 0.1
    
    def test_softmax(self):
        """Test softmax function."""
        logits = [1.0, 2.0, 3.0]
        probs = softmax(logits)
        
        # Check probabilities sum to 1
        assert abs(sum(probs) - 1.0) < 1e-10
        
        # Check ordering is preserved
        assert probs[2] > probs[1] > probs[0]
    
    def test_kl_divergence(self):
        """Test KL divergence calculation."""
        p = [0.5, 0.3, 0.2]
        q = [0.4, 0.4, 0.2]
        
        kl = kl_divergence(p, q)
        assert kl >= 0.0  # KL divergence is always non-negative
    
    def test_kl_divergence_identical(self):
        """Test KL divergence of identical distributions."""
        p = [0.5, 0.3, 0.2]
        kl = kl_divergence(p, p)
        assert abs(kl) < 1e-10  # Should be 0 for identical distributions


class TestSimpleActiveInferenceMatrices:
    """Test Simple Active Inference Matrices class."""
    
    def test_initialization(self):
        """Test matrix initialization."""
        num_obs = [4, 5, 3]
        num_states = [3, 3, 5]
        num_actions = 9
        
        matrices = SimpleActiveInferenceMatrices(num_obs, num_states, num_actions)
        
        assert len(matrices.A) == len(num_obs)
        assert len(matrices.B) == len(num_states)
        assert len(matrices.C) == len(num_obs)
        assert len(matrices.D) == len(num_states)
    
    def test_infer_states_simple(self):
        """Test state inference with simple case."""
        num_obs = [2]
        num_states = [2]
        num_actions = 2
        
        matrices = SimpleActiveInferenceMatrices(num_obs, num_states, num_actions)
        
        # Simple observation
        observations = [0]
        beliefs = matrices.infer_states(observations)
        
        assert len(beliefs) == 1
        assert len(beliefs[0]) == 2
        assert abs(sum(beliefs[0]) - 1.0) < 1e-10  # Should sum to 1
    
    def test_sample_action(self):
        """Test action sampling."""
        num_obs = [2, 3]
        num_states = [2, 3]
        num_actions = 3
        
        matrices = SimpleActiveInferenceMatrices(num_obs, num_states, num_actions)
        
        # Simple beliefs
        beliefs = [[0.7, 0.3], [0.4, 0.3, 0.3]]
        action = matrices.sample_action(beliefs)
        
        assert 0 <= action < num_actions
    
    def test_update_matrices(self):
        """Test matrix updates."""
        num_obs = [2]
        num_states = [2]
        num_actions = 2
        
        matrices = SimpleActiveInferenceMatrices(num_obs, num_states, num_actions)
        
        # Update A matrix
        initial_value = matrices.A[0][0][0]
        matrices.update_A_matrix(0, 0, [0], 1.0)
        assert matrices.A[0][0][0] == initial_value + 1.0
        
        # Update B matrix
        initial_value = matrices.B[0][0][0][0]
        matrices.update_B_matrix(0, 0, 0, 0, 1.0)
        assert matrices.B[0][0][0][0] == initial_value + 1.0
    
    def test_normalize_matrices(self):
        """Test matrix normalization."""
        num_obs = [2]
        num_states = [2]
        num_actions = 2
        
        matrices = SimpleActiveInferenceMatrices(num_obs, num_states, num_actions)
        
        # Add some counts
        matrices.update_A_matrix(0, 0, [0], 5.0)
        matrices.update_A_matrix(0, 1, [0], 3.0)
        matrices.update_B_matrix(0, 0, 0, 0, 4.0)
        matrices.update_B_matrix(0, 1, 0, 0, 2.0)
        
        matrices.normalize_matrices()
        
        # Check A matrix normalization (columns sum to 1)
        col_sum = matrices.A[0][0][0] + matrices.A[0][1][0]
        assert abs(col_sum - 1.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])