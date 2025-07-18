from typing import List, Optional, Tuple

import torch
import numpy as np
import random
from scipy.stats import beta

from iraf_engine.dnn import IRafMultiTaskDNN
from iraf_engine.node import AlphaZeroNode, Node, Node_PW
from comec_simulator.core.components import BaseStation, EdgeServer, Task

# Assuming MCTS and necessary imports are available
from iraf_engine.mcts import MCTS

MIN_PW_FLOOR = 3

class MCTS_PW(MCTS):
    def __init__(
        self, 
        bins_per_subaction_list: List[int] = [20, 10, 10, 10, 10],
        use_dnn: bool = False, 
        k_pw: float = 1.0, 
        alpha_pw: float = 0.5
        ):
        super().__init__(bins_per_subaction_list=bins_per_subaction_list, use_dnn=use_dnn)
        # Progressive Widening parameters
        self.k_pw = k_pw
        self.alpha_pw = alpha_pw
        self.root = Node_PW(depth=0)
        self.current_node = self.root
        if not use_dnn: # This is for mcts with pw only, try this as mcts+dnn+pw is now too subjective
            self.model = None
            
    def get_ratios(self, env_resources) -> Tuple[float, float, float, float, float]:
        """
        Get the ratios for the task: So for each task other environment conditions, we can get the best ratio for the task
        Args:
            env_resources: List[float]
        Returns:
            Tuple[float, float, float, float, float]
        """
        if self.use_dnn:
            return self._get_ratios_dnn(env_resources)
        else:
            return self._get_ratios_pw(env_resources)

    def _get_ratios_dnn(self, env_resources) -> Tuple[float, float, float, float, float]:
        ratios = np.ones(5)
        if self._is_unexpanded(self.current_node): # This is the case of the first rollout of undiscovered node
            env_resources = torch.tensor(env_resources, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():   
                probs = self.model(env_resources)
            probs = [p.detach() for p in probs]
            self.dnn_call_count += 1
            self.current_node.task_dnn_output = probs # No need for children_dnn_output, so let's think like dnn output is needed for the later subactions, so for the last node of the previous task, it does not need to store the dnn for above subactions, so it can store the dnn output for the current task
            for i in range(5):
                progressive_widening_floor = self._get_progressive_widening_floor(self.current_node)
                bins_of_subaction: List[float] = self.bins[i]
                # Get indices of top progressive_widening_floor bins based on priors
                top_k_indices: List[int] = torch.topk(probs[i], min(progressive_widening_floor, len(probs[i]))).indices
                self.current_node.update_selected_bins(top_k_indices)
                # Expand the node with the top k indices
                for j in top_k_indices:
                    prior = probs[i][j].item()
                    depth = self.current_node.depth + 1
                    task_idx, subaction_idx = self.current_node.get_node_index()
                    child = Node_PW(action=(task_idx, subaction_idx, bins_of_subaction[j]), prior=prior, depth=depth, parent=self.current_node, task_dnn_output=probs)
                    self.current_node.children.append(child)
                    self.total_nodes += 1
                
                # Set the node as expanded and select the best child
                self.current_node.expanded = True
                max_child = self.best_child(self.current_node)
                
                # Record the best ratio and move to the best child
                ratios[i] = max_child.action[2]
                self.current_node = max_child
            assert self.current_node.depth % 5 == 0, f"Current node depth is not a multiple of 5, {self.current_node.depth}"
        else:
            for i in range(5):
                progressive_widening_floor = self._get_progressive_widening_floor(self.current_node)
                if progressive_widening_floor > len(self.current_node.children):
                    # If the number of children is less than the progressive widening floor, we need to expand the node
                    top_k_indices: List[int] = torch.topk(self.current_node.task_dnn_output[i], min(progressive_widening_floor, len(self.current_node.task_dnn_output[i]))).indices
                    # Expand the node with the top k indices
                    new_bins: List[int] = self.current_node.get_unselected_bins(top_k_indices)
                    self.current_node.update_selected_bins(top_k_indices)

                    for j in new_bins:
                        task_idx, subaction_idx = self.current_node.get_node_index()
                        prior = self.current_node.task_dnn_output[i][j].item()
                        depth = self.current_node.depth + 1
                        child = Node_PW(action=(task_idx, subaction_idx, self.bins[i][j]), prior=prior, depth=depth, parent=self.current_node, task_dnn_output=self.current_node.task_dnn_output)
                        self.current_node.children.append(child)
                        self.total_nodes += 1

                # Select
                max_child = self.best_child(self.current_node)
                ratios[i] = max_child.action[2]
                self.current_node = max_child
        assert np.any(ratios >= 1.0) == False and np.any(ratios < 0.0) == False, f"Ratios are out of range, {ratios}"
        return tuple(ratios)

    def _get_ratios_pw(self, env_resources) -> Tuple[float, float, float, float, float]:
        ratios = np.ones(5)
        if self._is_unexpanded(self.current_node): # This is the case of the first rollout of undiscovered node            
            for i in range(5):
                progressive_widening_floor = self._get_progressive_widening_floor(self.current_node)
                bins_of_subaction: List[float] = self.bins[i]
                beta_distribution = self._get_beta_distribution(self.current_node)
                # Get indices of top progressive_widening_floor bins based on beta distribution
                beta_tensor = torch.tensor(beta_distribution)
                top_k_indices: List[int] = torch.topk(beta_tensor, min(progressive_widening_floor, len(beta_tensor))).indices.tolist()
                self.current_node.update_selected_bins(top_k_indices)
                
                # Expand the node with the top k indices
                for j in top_k_indices:
                    depth = self.current_node.depth + 1
                    task_idx, subaction_idx = self.current_node.get_node_index()
                    child = Node_PW(action=(task_idx, subaction_idx, bins_of_subaction[j]), depth=depth, parent=self.current_node)
                    self.current_node.children.append(child)
                    self.total_nodes += 1
                
                # Set the node as expanded and select the best child
                self.current_node.expanded = True
                max_child = self.best_child(self.current_node)
                
                # Record the best ratio and move to the best child
                ratios[i] = max_child.action[2]
                self.current_node = max_child
            assert self.current_node.depth % 5 == 0, f"Current node depth is not a multiple of 5, {self.current_node.depth}"
        else:
            for i in range(5):
                progressive_widening_floor = self._get_progressive_widening_floor(self.current_node)
                if progressive_widening_floor > len(self.current_node.children):
                    # If the number of children is less than the progressive widening floor, we need to expand the node
                    beta_distribution = self._get_beta_distribution(self.current_node)
                    beta_tensor = torch.tensor(beta_distribution)   
                    top_k_indices: List[int] = torch.topk(beta_tensor, min(progressive_widening_floor, len(beta_tensor))).indices.tolist()
                    # Expand the node with the top k indices
                    new_bins: List[int] = self.current_node.get_unselected_bins(top_k_indices)
                    self.current_node.update_selected_bins(top_k_indices)

                    for j in new_bins:
                        task_idx, subaction_idx = self.current_node.get_node_index()
                        depth = self.current_node.depth + 1
                        child = Node_PW(action=(task_idx, subaction_idx, self.bins[i][j]), depth=depth, parent=self.current_node)
                        self.current_node.children.append(child)
                        self.total_nodes += 1

                # Select
                max_child = self.best_child(self.current_node)
                ratios[i] = max_child.action[2]
                self.current_node = max_child
        assert np.any(ratios >= 1.0) == False and np.any(ratios < 0.0) == False, f"Ratios are out of range, {ratios}"
        return tuple(ratios)

    
    def _is_unexpanded(self, node: Node) -> bool:
        return node.children == []
    
    def _get_progressive_widening_floor(self, node: Node) -> int:
        progressive_widening_floor = int(self.k_pw * (node.N ** self.alpha_pw))
        return max(progressive_widening_floor, MIN_PW_FLOOR)
    
    def _get_beta_distribution(self, node: Node, a=2., b=0.5) -> List[float]:
        """
        Get a beta distribution PDF over bins ∈ [0, 1], biased toward 1 (right skew)
        """
        task_idx, subaction_idx = node.get_node_index()
        bins = self.bins[subaction_idx]
        bins = np.array(bins)
        
        # Ensure bins are strictly inside (0,1) to avoid PDF=0 at boundary
        eps = 1e-5
        bins = np.clip(bins, eps, 1 - eps)
        
        # Evaluate beta PDF at each bin
        probs = beta.pdf(bins, a, b)

        # Normalize to get a proper probability distribution
        probs /= probs.sum()
        
        return probs.tolist()
