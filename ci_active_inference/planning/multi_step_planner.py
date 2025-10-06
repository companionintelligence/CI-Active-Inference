"""Multi-step planning with longer horizons for Active Inference agents."""

import math
from typing import Dict, List, Any, Optional, Tuple
from ..utils.matrix_utils import SimpleActiveInferenceMatrices


class PolicyTree:
    """Tree structure for representing multi-step policies."""
    
    def __init__(self, action: int = None, depth: int = 0):
        self.action = action
        self.depth = depth
        self.children: List['PolicyTree'] = []
        self.expected_utility = 0.0
        self.state_beliefs = None
        
    def add_child(self, action: int) -> 'PolicyTree':
        """Add a child node with the given action."""
        child = PolicyTree(action, self.depth + 1)
        self.children.append(child)
        return child
    
    def get_action_sequence(self) -> List[int]:
        """Get the sequence of actions from root to this node."""
        if self.action is None:
            return []
        path = []
        node = self
        while node.action is not None:
            path.insert(0, node.action)
            if hasattr(node, 'parent'):
                node = node.parent
            else:
                break
        return path


class MultiStepPlanner:
    """
    Advanced multi-step planner for Active Inference agents.
    
    Implements sophisticated planning algorithms with longer horizons,
    policy trees, and temporal utility calculations.
    """
    
    def __init__(self, matrices: SimpleActiveInferenceMatrices, 
                 max_horizon: int = 10, 
                 branching_factor: int = None):
        """
        Initialize the planner.
        
        Args:
            matrices: Active Inference matrices
            max_horizon: Maximum planning horizon
            branching_factor: Max actions to consider per step (None = all actions)
        """
        self.matrices = matrices
        self.max_horizon = max_horizon
        self.branching_factor = branching_factor or matrices.num_actions
        self.policy_cache = {}
        
    def plan_multi_step(self, initial_beliefs: List[List[float]], 
                       horizon: int = None) -> Tuple[List[int], float]:
        """
        Plan multiple steps ahead using policy tree search.
        
        Args:
            initial_beliefs: Current belief state
            horizon: Planning horizon (uses max_horizon if None)
            
        Returns:
            Tuple of (optimal_action_sequence, expected_utility)
        """
        horizon = horizon or self.max_horizon
        
        # Build policy tree
        root = PolicyTree()
        root.state_beliefs = initial_beliefs
        
        best_policy = self._build_policy_tree(root, horizon)
        
        if best_policy:
            return best_policy.get_action_sequence(), best_policy.expected_utility
        else:
            # Fallback to single-step planning
            action = self.matrices.sample_action(initial_beliefs)
            utility = self._calculate_single_step_utility(initial_beliefs, action)
            return [action], utility
    
    def _build_policy_tree(self, node: PolicyTree, remaining_depth: int) -> PolicyTree:
        """
        Recursively build policy tree and find optimal policy.
        
        Args:
            node: Current tree node
            remaining_depth: Remaining planning depth
            
        Returns:
            Best child policy tree
        """
        if remaining_depth <= 0:
            node.expected_utility = self._evaluate_terminal_utility(node.state_beliefs)
            return node
        
        best_child = None
        best_utility = float('-inf')
        
        # Consider top actions based on branching factor
        action_utilities = []
        for action in range(self.matrices.num_actions):
            utility = self._calculate_single_step_utility(node.state_beliefs, action)
            action_utilities.append((action, utility))
        
        # Sort and take top k actions
        action_utilities.sort(key=lambda x: x[1], reverse=True)
        top_actions = action_utilities[:self.branching_factor]
        
        for action, _ in top_actions:
            child = node.add_child(action)
            child.parent = node
            
            # Predict next state
            next_beliefs = self._predict_next_beliefs(node.state_beliefs, action)
            child.state_beliefs = next_beliefs
            
            # Recursively plan from child
            best_subtree = self._build_policy_tree(child, remaining_depth - 1)
            
            # Calculate total utility (immediate + future)
            immediate_utility = self._calculate_single_step_utility(node.state_beliefs, action)
            future_utility = best_subtree.expected_utility * 0.95  # Discount factor
            total_utility = immediate_utility + future_utility
            
            child.expected_utility = total_utility
            
            if total_utility > best_utility:
                best_utility = total_utility
                best_child = child
        
        if best_child:
            node.expected_utility = best_utility
            return best_child
        else:
            node.expected_utility = self._evaluate_terminal_utility(node.state_beliefs)
            return node
    
    def _predict_next_beliefs(self, current_beliefs: List[List[float]], 
                            action: int) -> List[List[float]]:
        """
        Predict next state beliefs given current beliefs and action.
        
        Args:
            current_beliefs: Current belief state
            action: Action to take
            
        Returns:
            Predicted next beliefs
        """
        next_beliefs = []
        
        for factor in range(len(self.matrices.num_states)):
            next_belief = [0.0] * self.matrices.num_states[factor]
            
            for curr_state in range(self.matrices.num_states[factor]):
                for next_state in range(self.matrices.num_states[factor]):
                    if (factor < len(self.matrices.B) and 
                        next_state < len(self.matrices.B[factor]) and
                        curr_state < len(self.matrices.B[factor][next_state]) and
                        action < len(self.matrices.B[factor][next_state][curr_state])):
                        
                        transition_prob = self.matrices.B[factor][next_state][curr_state][action]
                        next_belief[next_state] += current_beliefs[factor][curr_state] * transition_prob
            
            next_beliefs.append(next_belief)
        
        return next_beliefs
    
    def _calculate_single_step_utility(self, beliefs: List[List[float]], 
                                     action: int) -> float:
        """
        Calculate expected utility for a single step.
        
        Args:
            beliefs: Current beliefs
            action: Action to evaluate
            
        Returns:
            Expected utility
        """
        utility = 0.0
        
        # Calculate preference-based utility
        if hasattr(self.matrices, 'C') and self.matrices.C:
            for modality in range(len(self.matrices.C)):
                if modality < len(beliefs):
                    for obs in range(len(self.matrices.C[modality])):
                        # Calculate expected observation probability
                        obs_prob = 0.0
                        for factor in range(len(beliefs)):
                            if factor < len(self.matrices.num_states):
                                for state in range(self.matrices.num_states[factor]):
                                    if (modality < len(self.matrices.A) and
                                        obs < len(self.matrices.A[modality]) and
                                        state < len(self.matrices.A[modality][obs])):
                                        
                                        likelihood = self.matrices.A[modality][obs][state]
                                        obs_prob += beliefs[factor][state] * likelihood
                        
                        # Add preference-weighted utility
                        if obs < len(self.matrices.C[modality]):
                            utility += obs_prob * self.matrices.C[modality][obs]
        
        # Add exploration bonus (information gain)
        entropy = self._calculate_state_entropy(beliefs)
        utility += 0.1 * entropy  # Small exploration bonus
        
        return utility
    
    def _evaluate_terminal_utility(self, beliefs: List[List[float]]) -> float:
        """
        Evaluate utility at terminal nodes.
        
        Args:
            beliefs: Terminal state beliefs
            
        Returns:
            Terminal utility
        """
        # For terminal states, just evaluate current preference satisfaction
        return self._calculate_single_step_utility(beliefs, 0)  # Action doesn't matter for terminal
    
    def _calculate_state_entropy(self, beliefs: List[List[float]]) -> float:
        """
        Calculate entropy of belief state.
        
        Args:
            beliefs: Belief state
            
        Returns:
            Entropy value
        """
        total_entropy = 0.0
        
        for factor_beliefs in beliefs:
            factor_entropy = 0.0
            for prob in factor_beliefs:
                if prob > 1e-10:  # Avoid log(0)
                    factor_entropy -= prob * math.log(prob)
            total_entropy += factor_entropy
        
        return total_entropy
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the planning process.
        
        Returns:
            Dictionary of planning statistics
        """
        return {
            "max_horizon": self.max_horizon,
            "branching_factor": self.branching_factor,
            "cache_size": len(self.policy_cache),
            "num_states": self.matrices.num_states,
            "num_actions": self.matrices.num_actions
        }