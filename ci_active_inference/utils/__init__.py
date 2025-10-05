"""Utility functions for CI Active Inference system."""

# Core utilities that don't require external dependencies
try:
    from .matrix_utils import SimpleActiveInferenceMatrices, normalize
    from .discretization import DataDiscretizer
except ImportError as e:
    print(f"Warning: Could not import some utility modules: {e}")
    
    # Provide basic fallbacks
    class SimpleActiveInferenceMatrices:
        pass
    
    def normalize(matrix, axis=0):
        return matrix
    
    class DataDiscretizer:
        pass

__all__ = [
    'SimpleActiveInferenceMatrices',
    'normalize', 
    'DataDiscretizer'
]