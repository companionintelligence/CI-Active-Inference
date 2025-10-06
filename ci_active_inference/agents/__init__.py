"""Active Inference agents for CI prediction system."""

from .simple_agent import SimpleActiveInferenceAgent
from .pymdp_agent import PyMDPActiveInferenceAgent
from .behavior_optimizer import CIBehaviorOptimizer
from .personal_life_agent import PersonalLifeAgent

# Use simplified agent by default (pymdp may not be available)
try:
    CIActiveInferenceAgent = PyMDPActiveInferenceAgent
except (ImportError, NameError):
    CIActiveInferenceAgent = SimpleActiveInferenceAgent

__all__ = [
    'SimpleActiveInferenceAgent',
    'PyMDPActiveInferenceAgent',
    'CIActiveInferenceAgent', 
    'CIBehaviorOptimizer',
    'PersonalLifeAgent'
]