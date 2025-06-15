from typing import List, Optional, Tuple
from torch.distributions import Beta
from sympy import beta
import torch

from typing import Optional, Tuple, List

import numpy as np
import torch.nn as nn
from comec_simulator.core.components import BaseStation, EdgeServer, Task
from iraf_engine.dnn import IRafMultiTaskDNN
import random
from iraf_engine.mcts_pw import MIN_PW_FLOOR
from iraf_engine.node import Node, AlphaZeroNode, Node_PW


class A0C_Node:
    def __init__(self, action: Tuple[float, float, float, float, float] = None, depth: int = 0, num_subactions: int = 5, parent = None):
        self.children: List[A0C_Node] = []
        self.parent: Optional[A0C_Node] = parent
        self.N = 0  # Visit count
        self.Q = 0.0  # Total value
        self.W = 0.0  # Total weight
        self.reward = 0.0  # Total reward   
        self.action = action  # [bin_value1, bin_value2, bin_value3, bin_value4, bin_value5]
        self.expanded = False
        self.depth = depth
        self.num_subactions = num_subactions

    def is_terminal(self, total_tasks: int) -> bool:
        return self.depth == total_tasks * self.num_subactions

    # def get_node_index(self) -> Tuple[int, int]:
    #     idx = self.depth
    #     return idx // self.num_subactions, idx % self.num_subactions

    def is_fully_expanded(self) -> bool:
        return self.expanded

    def __str__(self):
        return f"[A0C_Node] depth={self.depth}, action={self.action}, Q={self.Q:.3f}, N={self.N}"
    
class A0C:
    def __init__(self, input_dim, exploration_constant=0.8, num_subactions: int = 5, use_dnn: bool = False, k_pw: float = 2.0, alpha_pw: float = 0.7):
        self.c = exploration_constant
        self.num_subactions = num_subactions
        self.use_dnn = use_dnn
        self.total_nodes = 1  # Start with 1 for root node
        self.root = A0C_Node(depth=0)
        self.current_node = self.root
        self.k_pw = k_pw
        self.alpha_pw = alpha_pw
        
    def backprop(self, reward: float):
        """Update node statistics upward through the tree"""
        action_sequence = []
        # assert self.current_node.depth == 10, f"This should be 10, the fuck is {self.current_node.depth}"
        while self.current_node.parent is not None:
            action_sequence.append(self.current_node.action[2])
            self.current_node.N += 1
            self.current_node.Q += reward
            self.current_node = self.current_node.parent
        # print(action_sequence)
        # exit()
        self.current_node.N += 1
        self.current_node.Q += reward
        # for child in self.current_node.children:
        #     print(child)

        action_sequence.reverse()
        action_sequence = np.array(action_sequence).reshape(-1, 5) 


    def best_child(self, node: A0C_Node) -> A0C_Node:
        """Select the best child based on UCB score"""
        def uct(child: A0C_Node):
            # For negative rewards, higher (less negative) Q/N is better
            exploitation = child.Q / (child.N + 1e-6)
            # Add exploration bonus
            exploration = self.c * np.sqrt(np.log(node.N+1) / (1+child.N))
            
            return exploitation + exploration
                
        # print(f"Node at depth {node.depth} has {len(node.children)} children")
        assert len(node.children) > 0, f"Node {node} has no children"
        # Select best child based on UCB
        scores = [uct(child) for child in node.children]
        max_score = max(scores)
        best_indices = [i for i, score in enumerate(scores) if score == max_score]
        chosen_index = random.choice(best_indices)
        return node.children[chosen_index]
    
    def _get_ratios_a0c(self) -> Tuple[float, float, float, float, float]:
        progressive_widening_floor = self._get_progressive_widening_floor(self.current_node)
        
        if progressive_widening_floor > self.current_node.N:
            sampled_action = self.sample_action_5d(device='cpu')
            # task_idx, subaction_idx = self.current_node.get_node_index()
            node = A0C_Node(action=sampled_action, depth=self.current_node.depth+1, parent=self.current_node)
            self.current_node.children.append(node)
            self.total_nodes += 1
            self.current_node.expanded = True
            self.current_node = node
            return tuple(sampled_action)
        else:
            max_child = self.best_child(self.current_node)
            # print(max_child.action)
            # exit()
            ratios = max_child.action
            self.current_node = max_child
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
        NUM_TASKS = 20 # TODO: Make this dynamic
        π = np.zeros((NUM_TASKS, self.num_subactions, max(self.bins_per_subaction_list)))
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
                if t_idx < NUM_TASKS:
                    π[t_idx][s_idx][bin_id] += child.N
                else:
                    print(f"t_idx: {t_idx}, s_idx: {s_idx}, bin_id: {bin_id}")
            list_children.extend(child.children)
                    
        # Normalize
        for t in range(20):
            for s in range(self.num_subactions):
                total = np.sum(π[t][s])
                if total > 0:
                    π[t][s] /= total
    
        return π
    
    def get_node_count(self):
        return self.total_nodes

    def count_unique_dnn_outputs(self) -> int:
        """
        Walks the entire tree, collects all node.task_dnn_output references,
        and returns how many unique objects there are.
        """
        seen = set()       # to hold id(probs) of each distinct tensor list
        stack = [self.root]

        while stack:
            node = stack.pop()
            stack.extend(node.children)

            # only DNN nodes have this attribute
            probs = getattr(node, "task_dnn_output", None)
            if probs is not None:
                seen.add(id(probs))

        return len(seen)
    
        
    def _get_progressive_widening_floor(self, node: A0C_Node) -> int:
        progressive_widening_floor = int(self.k_pw * (node.N ** self.alpha_pw))
        return max(progressive_widening_floor, MIN_PW_FLOOR)


    def sample_action_5d(self, alpha_val=2.0, beta_val=0.5, device='cpu'):
        """
        Sample a 5-dimensional action vector from independent Beta distributions.

        Args:
            alpha_val (float): The alpha parameter for each Beta distribution.
            beta_val (float): The beta parameter for each Beta distribution.
            device (str): The device to perform computation on (e.g., 'cpu' or 'cuda').

        Returns:
            torch.Tensor: A tensor of shape (5,) containing the sampled action.
        """
        alpha = torch.full((5,), alpha_val, device=device)
        beta = torch.full((5,), beta_val, device=device)
        dist = Beta(alpha, beta)
        action = dist.sample()
        return action
