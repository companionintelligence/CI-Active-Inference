"""Planning utilities for multi-step planning and policy evaluation."""

import math
from typing import List, Dict, Any, Tuple


def calculate_expected_utility(beliefs: List[List[float]], 
                             action_sequence: List[int],
                             matrices,
                             discount_factor: float = 0.95) -> float:
    """
    Calculate expected utility of an action sequence.
    
    Args:
        beliefs: Initial belief state
        action_sequence: Sequence of actions to evaluate
        matrices: Active Inference matrices
        discount_factor: Temporal discount factor
        
    Returns:
        Expected cumulative utility
    """
    current_beliefs = beliefs
    total_utility = 0.0
    
    for t, action in enumerate(action_sequence):
        # Calculate immediate utility
        step_utility = _calculate_step_utility(current_beliefs, action, matrices)
        discounted_utility = step_utility * (discount_factor ** t)
        total_utility += discounted_utility
        
        # Update beliefs for next step
        current_beliefs = _predict_next_beliefs(current_beliefs, action, matrices)
    
    return total_utility


def generate_policy_sequences(num_actions: int, 
                            horizon: int, 
                            max_sequences: int = 100) -> List[List[int]]:
    """
    Generate diverse policy sequences for evaluation.
    
    Args:
        num_actions: Number of available actions
        horizon: Planning horizon
        max_sequences: Maximum number of sequences to generate
        
    Returns:
        List of action sequences
    """
    sequences = []
    
    # Generate systematic sequences
    if horizon <= 3:
        # Exhaustive for short horizons
        def generate_recursive(current_seq, remaining_depth):
            if remaining_depth == 0:
                sequences.append(current_seq.copy())
                return
            for action in range(num_actions):
                current_seq.append(action)
                generate_recursive(current_seq, remaining_depth - 1)
                current_seq.pop()
        
        generate_recursive([], horizon)
    else:
        # Sampling for longer horizons
        import random
        for _ in range(max_sequences):
            sequence = [random.randint(0, num_actions - 1) for _ in range(horizon)]
            sequences.append(sequence)
    
    # Limit to max_sequences
    return sequences[:max_sequences]


def _calculate_step_utility(beliefs: List[List[float]], 
                          action: int, 
                          matrices) -> float:
    """Calculate utility for a single step."""
    utility = 0.0
    
    # Preference-based utility
    if hasattr(matrices, 'C') and matrices.C:
        for modality in range(len(matrices.C)):
            if modality < len(beliefs):
                for obs in range(len(matrices.C[modality])):
                    # Calculate expected observation probability
                    obs_prob = 0.0
                    for factor in range(len(beliefs)):
                        for state in range(len(beliefs[factor])):
                            if (modality < len(matrices.A) and
                                obs < len(matrices.A[modality]) and
                                state < len(matrices.A[modality][obs])):
                                
                                likelihood = matrices.A[modality][obs][state]
                                obs_prob += beliefs[factor][state] * likelihood
                    
                    # Add preference-weighted utility
                    if obs < len(matrices.C[modality]):
                        utility += obs_prob * matrices.C[modality][obs]
    
    return utility


def _predict_next_beliefs(current_beliefs: List[List[float]], 
                        action: int, 
                        matrices) -> List[List[float]]:
    """Predict next beliefs given action."""
    next_beliefs = []
    
    for factor in range(len(current_beliefs)):
        next_belief = [0.0] * len(current_beliefs[factor])
        
        for curr_state in range(len(current_beliefs[factor])):
            for next_state in range(len(next_belief)):
                if (factor < len(matrices.B) and 
                    next_state < len(matrices.B[factor]) and
                    curr_state < len(matrices.B[factor][next_state]) and
                    action < len(matrices.B[factor][next_state][curr_state])):
                    
                    transition_prob = matrices.B[factor][next_state][curr_state][action]
                    next_belief[next_state] += current_beliefs[factor][curr_state] * transition_prob
        
        next_beliefs.append(next_belief)
    
    return next_beliefs


def evaluate_policy_tree(root_node, matrices, horizon: int) -> Dict[str, Any]:
    """
    Evaluate a complete policy tree.
    
    Args:
        root_node: Root of policy tree
        matrices: Active Inference matrices
        horizon: Planning horizon
        
    Returns:
        Evaluation statistics
    """
    def traverse_tree(node, depth=0):
        stats = {
            'nodes_visited': 1,
            'max_depth': depth,
            'total_utility': node.expected_utility,
            'leaf_nodes': 0
        }
        
        if not node.children:
            stats['leaf_nodes'] = 1
        else:
            for child in node.children:
                child_stats = traverse_tree(child, depth + 1)
                stats['nodes_visited'] += child_stats['nodes_visited']
                stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
                stats['leaf_nodes'] += child_stats['leaf_nodes']
        
        return stats
    
    return traverse_tree(root_node)