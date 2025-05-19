import random
from typing import List, Optional, Tuple


from typing import Optional, Tuple, List

import numpy as np

from comec_simulator.core.components import BaseStation, EdgeServer, Task

random.seed(187)
np.random.seed(187)

class Node:
    def __init__(self, action: Optional[Tuple[int, int, float]] = None, depth: int = 0, num_subactions: int = 5, parent = None):
        self.children: List[Node] = []
        self.parent: Optional[Node] = parent
        self.N = 0  # Visit count
        self.Q = 0.0  # Total value
        self.action = action  # (task_idx, subaction_idx, bin_value)
        self.expanded = False
        self.depth = depth
        self.num_subactions = num_subactions

    def is_terminal(self, total_tasks: int) -> bool:
        return self.depth == total_tasks * self.num_subactions

    def get_node_index(self) -> Tuple[int, int]:
        idx = self.depth
        return idx // self.num_subactions, idx % self.num_subactions

    def is_fully_expanded(self) -> bool:
        return self.expanded

    def __str__(self):
        return f"[ClassicNode] depth={self.depth}, action={self.action}, Q={self.Q:.3f}, N={self.N}"

class AlphaZeroNode(Node):
    def __init__(self, action: Optional[Tuple[int, int, float]] = None, prior: float = 0.0, depth: int = 0, num_subactions: int = 5):
        super().__init__(action, depth, num_subactions)
        self.prior = prior  # Prior from policy network
        self.value_sum = 0.0  # Sum of NN values for average estimation

    def get_mean_value(self) -> float:
        return self.value_sum / self.N if self.N > 0 else 0.0

    def __str__(self):
        return f"[AlphaZeroNode] depth={self.depth}, action={self.action}, prior={self.prior:.3f}, Q={self.Q:.3f}, N={self.N}"

class MCTS:
    def __init__(self, exploration_constant=0.8, num_subactions: int = 5, bins_per_subaction_list: List[int] = [20, 10, 10, 10, 10]):
        # self.model = model
        self.c = exploration_constant
        self.num_subactions = num_subactions
        self.bins_per_subaction_list = bins_per_subaction_list
        self.root = Node(depth=0)
        self.current_node = self.root
        self.bins = [np.linspace(0, 1, bins_per_subaction + 2)[1:-1].tolist() for bins_per_subaction in self.bins_per_subaction_list]
        
    
    def backprop(self, reward: float):
        """Update node statistics upward through the tree"""
        action_sequence = []
        while self.current_node.parent is not None:
            action_sequence.append(self.current_node.action[2])
            self.current_node.N += 1
            self.current_node.Q += reward
            self.current_node = self.current_node.parent
        self.current_node.N += 1
        self.current_node.Q += reward
        action_sequence.reverse()
        action_sequence = np.array(action_sequence).reshape(-1, 5) 

    def best_child(self, node: Node) -> Node:
        """Select the best child based on UCB score"""
        def uct(child: Node):
            # For negative rewards, higher (less negative) Q/N is better
            exploitation = child.Q / (child.N + 1e-6)
            
            # Add exploration bonus
            exploration = self.c * np.sqrt(np.log(node.N+1) / (1+child.N))
            
            return exploitation + exploration
        
        def puct(child: AlphaZeroNode):
            exploitation = child.Q
            exploration = self.c * child.prior * np.sqrt(np.log(node.N+1)) / (1+ child.N)
            return exploitation + exploration

        # Select best child based on UCB
        scores = [uct(child) for child in node.children]
        max_score = max(scores)
        best_indices = [i for i, score in enumerate(scores) if score == max_score]
        chosen_index = random.choice(best_indices)
        return node.children[chosen_index]
    
    
    def get_ratios(self, env_resources) -> Tuple[float, float, float, float, float]:
        """
        Get the ratios for the task: So for each task other environment conditions, we c
        Args:
            env_resources: List[float]
        Returns:
            Tuple[float, float, float, float, float]
        """
        ratios = np.ones(5)
        for i in range(5):
            # Expand if not expanded
            if not self.current_node.expanded:
                num_bins = self.bins_per_subaction_list[i]
                for j in range(num_bins):
                    depth = self.current_node.depth + 1
                    task_idx, subaction_idx = self.current_node.get_node_index()
                    child = Node(action=(task_idx, subaction_idx, self.bins[i][j]), depth=depth, parent=self.current_node)
                    self.current_node.children.append(child)
                self.current_node.expanded = True
            # Select
            max_child = self.best_child(self.current_node)
            ratios[i] = max_child.action[2]
            self.current_node = max_child
        assert np.any(ratios >= 1.0) == False and np.any(ratios < 0.0) == False, f"Ratios are out of range, {ratios}"
        return tuple(ratios)
    
    def get_best_action(self):
        best_action = []
        node = self.current_node
        while node.children:
            best_child = max(node.children, key=lambda x: x.Q if x.N > 0 else -float('inf')) # Greedy selection
            best_action.append(best_child.action[2])
            node = best_child
        best_action = np.array(best_action).reshape(-1, 5).round(3)
        return best_action
    
    def extract_action_probabilities(self) -> np.ndarray:
        """Extract action probabilities from tree statistics"""
        π = np.zeros((20, self.num_subactions, max(self.bins_per_subaction_list)))
        node = self.current_node
        list_children = node.children
        while list_children:
            child = list_children.pop()
            if child.action is not None:
                t_idx, s_idx, val = child.action
                if s_idx == 0:
                    bin_id = int(round(val / (0.95 / (self.bins_per_subaction_list[0]-1))))
                else:
                    bin_id = int(round(val / (0.9 / (self.bins_per_subaction_list[s_idx]-1))))
                bin_id = min(bin_id, self.bins_per_subaction_list[s_idx] - 1)
                π[t_idx][s_idx][bin_id] += child.N
            list_children.extend(child.children)
                    
        # Normalize
        for t in range(20):
            for s in range(self.num_subactions):
                total = np.sum(π[t][s])
                if total > 0:
                    π[t][s] /= total
    
        return π