from typing import List, Optional, Tuple

import torch

from typing import Optional, Tuple, List

import numpy as np
import torch.nn as nn
from comec_simulator.core.components import BaseStation, EdgeServer, Task
from iraf_engine.dnn import IRafMultiTaskDNN
import random
from iraf_engine.node import Node, AlphaZeroNode

class MCTS:
    def __init__(self, input_dim, exploration_constant=0.8, num_subactions: int = 5, bins_per_subaction_list: List[int] = [20, 10, 10, 10, 10], use_dnn: bool = False):
        self.c = exploration_constant
        self.num_subactions = num_subactions
        self.bins_per_subaction_list = bins_per_subaction_list
        self.use_dnn = use_dnn
        self.total_nodes = 1  # Start with 1 for root node
        print(f"Using DNN: {use_dnn}")
        if use_dnn:
            print("Loading DNN model")
            self.model = IRafMultiTaskDNN(input_dim=input_dim, head_dims=[20, 10, 10, 10, 10])
            self.model.load_state_dict(torch.load("D:\\Research\\IoT\\iRAF-CoMEC-RL\\trained_iraf_policy_small.pt", weights_only=True))
            self.root = AlphaZeroNode(depth=0)
        else:
            self.model = None
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
            exploitation = child.Q / (child.N + 1e-6)
            exploration = self.c * child.prior * np.sqrt(np.log(node.N+1)) / (1+ child.N)
            return exploitation + exploration
        
        # print(f"Node at depth {node.depth} has {len(node.children)} children")
        assert len(node.children) > 0, f"Node {node} has no children"
        # Select best child based on UCB
        if self.use_dnn:
            scores = [puct(child) for child in node.children]
        else:
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
        if self.use_dnn:
            if not self.current_node.expanded:
                env_resources = torch.tensor(env_resources, dtype=torch.float32).unsqueeze(0)
                probs = self.model(env_resources) # Shape: (5, ?, 20|10|10|10|10), the ? is like to access the values in Torch stupid stuff
                for i in range(5):
                    for j in range(self.bins_per_subaction_list[i]):
                        prior = probs[i][j].item()
                        depth = self.current_node.depth + 1
                        task_idx, subaction_idx = self.current_node.get_node_index()
                        child = AlphaZeroNode(action=(task_idx, subaction_idx, self.bins[i][j]), prior=prior, depth=depth, parent=self.current_node, task_dnn_output=probs)
                        self.current_node.children.append(child)
                        self.total_nodes += 1
                    self.current_node.expanded = True
                    max_child = self.best_child(self.current_node)
                    ratios[i] = max_child.action[2]
                    self.current_node = max_child
                assert self.current_node.depth % 5 == 0, f"Current node depth is not a multiple of 5, {self.current_node.depth}"
            else:
                for i in range(5):
                    if not self.current_node.expanded:
                        task_idx, subaction_idx = self.current_node.get_node_index()
                        probs = self.current_node.task_dnn_output[subaction_idx] # Quite stupid but as this is he case of the last node of the previous is expanded (this branch is discovered at least 1), so always the node at task k-1, subaction 4 is expanded with all 20 children of task k, subaction 0
                        bins = self.bins[subaction_idx]
                        assert len(bins) == len(probs), f"Bins and probs have different lengths, {len(bins)} != {len(probs)}"
                        for j in range(len(bins)):
                            prior = probs[j].item()
                            depth = self.current_node.depth + 1
                            child = AlphaZeroNode(action=(task_idx, subaction_idx, bins[j]), prior=prior, depth=depth, parent=self.current_node, task_dnn_output=self.current_node.task_dnn_output)
                            self.current_node.children.append(child)
                            self.total_nodes += 1
                        self.current_node.expanded = True
                    # Select
                    max_child = self.best_child(self.current_node)
                    ratios[i] = max_child.action[2]
                    self.current_node = max_child
        else:
            ratios = self._expand_node_no_dnn(ratios)
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
    
    def _expand_node_no_dnn(self, ratios: List[float]):
        for i in range(5):
            # Expand if not expanded
            if not self.current_node.expanded:
                num_bins = self.bins_per_subaction_list[i]
                for j in range(num_bins):
                    depth = self.current_node.depth + 1
                    task_idx, subaction_idx = self.current_node.get_node_index()
                    child = Node(action=(task_idx, subaction_idx, self.bins[i][j]), depth=depth, parent=self.current_node)
                    self.current_node.children.append(child)
                    self.total_nodes += 1
                self.current_node.expanded = True
            # Select
            max_child = self.best_child(self.current_node)
            ratios[i] = max_child.action[2]
            self.current_node = max_child
        return ratios

# from typing import List, Optional, Tuple, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import random

# from comec_simulator.core.components import BaseStation, EdgeServer, Task
# from iraf_engine.dnn import IRafMultiTaskDNN
# from iraf_engine.node import Node, AlphaZeroNode


# class MCTS:
#     def __init__(
#         self,
#         input_dim: int,
#         exploration_constant: float = 0.8,
#         num_subactions: int = 5,
#         bins_per_subaction_list: Optional[List[int]] = None,
#         use_dnn: bool = False,
#         num_tasks: int = 20,
#         dnn_checkpoint_path: Optional[str] = None,
#     ):
#         """
#         Monte Carlo Tree Search (MCTS) with optional DNN priors (AlphaZero-style).

#         Args:
#             input_dim: Dimension of the input features for the DNN.
#             exploration_constant: UCT/PUCT exploration coefficient.
#             num_subactions: Number of subactions per task (default: 5).
#             bins_per_subaction_list: List indicating how many discrete bins each subaction has.
#                 Defaults to [20, 10, 10, 10, 10] if None.
#             use_dnn: Whether to load and use a trained IRafMultiTaskDNN.
#             num_tasks: Total number of tasks (used in extract_action_probabilities).
#             dnn_checkpoint_path: Path to a saved DNN checkpoint (required if use_dnn=True).
#         """
#         self.c = exploration_constant
#         self.num_subactions = num_subactions
#         self.bins_per_subaction_list = bins_per_subaction_list or [20, 10, 10, 10, 10]
#         self.use_dnn = use_dnn
#         self.num_tasks = num_tasks

#         # Precompute bin values for each subaction
#         self.bins: List[List[float]] = [
#             np.linspace(0, 1, n_bins + 2)[1:-1].tolist()
#             for n_bins in self.bins_per_subaction_list
#         ]

#         # Initialize total nodes count (including root)
#         self._total_nodes = 1

#         if use_dnn:
#             if not dnn_checkpoint_path:
#                 dnn_checkpoint_path = "D:\\Research\\IoT\\iRAF-CoMEC-RL\\trained_iraf_policy_small.pt"
#             self.model = IRafMultiTaskDNN(input_dim=input_dim, head_dims=self.bins_per_subaction_list)
#             self.model.load_state_dict(torch.load(dnn_checkpoint_path, weights_only=True))
#             self.root: Union[Node, AlphaZeroNode] = AlphaZeroNode(depth=0)
#         else:
#             self.model = None
#             self.root = Node(depth=0)

#         self.current_node = self.root

#     def backprop(self, reward: float) -> None:
#         """
#         Propagate the reward up the tree, updating N and Q for each visited node.
#         """
#         node = self.current_node
#         while node is not None:
#             node.N += 1
#             node.Q += reward
#             node = node.parent

#     def best_child(self, node: Union[Node, AlphaZeroNode]) -> Union[Node, AlphaZeroNode]:
#         """
#         Select the child with the highest UCT (if no DNN) or PUCT (if using DNN) score.

#         Raises:
#             AssertionError: if `node.children` is empty.
#         """
#         assert node.children, f"Node {node} has no children to select from."

#         if self.use_dnn:
#             # PUCT formula
#             def score(child: AlphaZeroNode) -> float:
#                 exploitation = child.Q / (child.N + 1e-6)
#                 exploration = (
#                     self.c
#                     * child.prior
#                     * np.sqrt(np.log(node.N + 1))
#                     / (1 + child.N)
#                 )
#                 return exploitation + exploration
#         else:
#             # Standard UCT formula
#             def score(child: Node) -> float:
#                 exploitation = child.Q / (child.N + 1e-6)
#                 exploration = self.c * np.sqrt(np.log(node.N + 1) / (1 + child.N))
#                 return exploitation + exploration

#         scores = [score(child) for child in node.children]
#         max_score = max(scores)
#         best_indices = [i for i, sc in enumerate(scores) if sc == max_score]
#         chosen_index = random.choice(best_indices)
#         return node.children[chosen_index]

#     def get_ratios(self, env_resources: List[float]) -> Tuple[float, ...]:
#         """
#         From the current_node, expand (if needed) and select one child per subaction,
#         returning a tuple of 5 chosen ratio values in [0,1).

#         Args:
#             env_resources: A list of floats representing the current environment state.

#         Returns:
#             A tuple of length `self.num_subactions` with chosen ratio values.
#         """
#         if self.use_dnn:
#             return self._select_with_dnn(env_resources)
#         else:
#             return self._select_without_dnn(env_resources)

#     def _select_without_dnn(self, env_resources: List[float]) -> Tuple[float, ...]:
#         """
#         Expand & select subaction ratios when not using DNN priors.
#         """
#         ratios = [0.0] * self.num_subactions

#         for i in range(self.num_subactions):
#             if not self.current_node.expanded:
#                 # Create children for all bins of subaction i
#                 for val in self.bins[i]:
#                     depth = self.current_node.depth + 1
#                     task_idx, subaction_idx = self.current_node.get_node_index()
#                     child = Node(
#                         action=(task_idx, subaction_idx, val),
#                         depth=depth,
#                         parent=self.current_node,
#                     )
#                     self.current_node.children.append(child)
#                     self._total_nodes += 1
#                 self.current_node.expanded = True

#             # Select best child for subaction i
#             chosen_child = self.best_child(self.current_node)
#             ratios[i] = chosen_child.action[2]
#             self.current_node = chosen_child

#         return tuple(ratios)

#     def _select_with_dnn(self, env_resources: List[float]) -> Tuple[float, ...]:
#         """
#         Expand & select subaction ratios using DNN priors (PUCT).
#         """
#         ratios = [0.0] * self.num_subactions
#         node = self.current_node

#         # If the DNN hasn't been used to expand this node, run a forward pass
#         if not node.expanded:
#             # Run the DNN once on env_resources
#             input_tensor = torch.tensor([env_resources], dtype=torch.float32)
#             # probs has shape: List of length num_subactions; each is a tensor of size [n_bins]
#             probs_list: List[torch.Tensor] = self.model(input_tensor)

#             # Create an AlphaZeroNode child for each bin of each subaction
#             for sub_idx in range(self.num_subactions):
#                 bin_values = self.bins[sub_idx]
#                 priors = probs_list[sub_idx].squeeze(0)  # shape: (n_bins,)
#                 for bin_idx, val in enumerate(bin_values):
#                     depth = node.depth + 1
#                     task_idx, curr_sub_idx = node.get_node_index()
#                     child = AlphaZeroNode(
#                         action=(task_idx, curr_sub_idx, val),
#                         prior=priors[bin_idx].item(),
#                         depth=depth,
#                         parent=node,
#                         task_dnn_output=probs_list,
#                     )
#                     node.children.append(child)
#                     self._total_nodes += 1

#             node.expanded = True

#         # For each subaction, select a child (possibly expanding deeper if needed)
#         for i in range(self.num_subactions):
#             # If the node for subaction i wasn’t expanded at the previous step,
#             # we may need to expand its children with the correct priors.
#             if not node.children:
#                 raise RuntimeError("No children found after expansion.")

#             # Use PUCT to select the next child
#             chosen_child = self.best_child(node)
#             ratios[i] = chosen_child.action[2]
#             node = chosen_child

#             # If this new node isn't expanded yet, expand it for the next subaction
#             if not node.expanded:
#                 task_idx, subaction_idx = node.get_node_index()
#                 priors = node.task_dnn_output[subaction_idx].squeeze(0)
#                 bin_values = self.bins[subaction_idx]

#                 for bin_idx, val in enumerate(bin_values):
#                     depth = node.depth + 1
#                     child = AlphaZeroNode(
#                         action=(task_idx, subaction_idx, val),
#                         prior=priors[bin_idx].item(),
#                         depth=depth,
#                         parent=node,
#                         task_dnn_output=node.task_dnn_output,
#                     )
#                     node.children.append(child)
#                     self._total_nodes += 1

#                 node.expanded = True

#         # Update current_node to the final selected leaf
#         self.current_node = node
#         return tuple(ratios)

#     def get_best_action(self) -> np.ndarray:
#         """
#         Traverse down from `current_node`, always picking the child with highest Q (if N>0),
#         and collect the action ratio values. Returns an array shaped (?,5).
#         """
#         actions = []
#         node = self.current_node

#         while node.children:
#             # Greedy: pick child with max Q/N (but ensure N > 0)
#             best_child = max(
#                 (c for c in node.children if c.N > 0),
#                 key=lambda x: x.Q / (x.N + 1e-6),
#                 default=None,
#             )
#             if best_child is None:
#                 break

#             actions.append(best_child.action[2])
#             node = best_child

#         if not actions:
#             return np.zeros((0, self.num_subactions))

#         arr = np.array(actions).reshape(-1, self.num_subactions)
#         return np.round(arr, 3)

#     def extract_action_probabilities(self) -> np.ndarray:
#         """
#         Extract the visit‐count–based probabilities (π) from `current_node` subtree.

#         Returns:
#             A NumPy array π of shape (num_tasks, num_subactions, max_n_bins),
#             where each entry is normalized over all bins for that (task, subaction).
#         """
#         max_bins = max(self.bins_per_subaction_list)
#         π = np.zeros((self.num_tasks, self.num_subactions, max_bins))

#         stack = list(self.current_node.children)
#         while stack:
#             child = stack.pop()
#             if child.action is not None:
#                 t_idx, s_idx, val = child.action
#                 # Compute approximate bin index:
#                 n_bins = self.bins_per_subaction_list[s_idx]
#                 # For subaction‐0, we used range [0, 0.95]; for others, [0, 0.9] originally.
#                 # Here we map val∈[0,1) → bin_idx ∈ [0,n_bins-1]
#                 scale = 0.95 if s_idx == 0 else 0.9
#                 bin_id = int(round(val / (scale / (n_bins - 1))))
#                 bin_id = min(bin_id, n_bins - 1)

#                 if 0 <= t_idx < self.num_tasks:
#                     π[t_idx, s_idx, bin_id] += child.N
#                 else:
#                     print(f"Warning: t_idx {t_idx} >= num_tasks {self.num_tasks}")
#             stack.extend(child.children)

#         # Normalize over bins for each (t, s)
#         for t in range(self.num_tasks):
#             for s in range(self.num_subactions):
#                 total = π[t, s].sum()
#                 if total > 0:
#                     π[t, s] /= total

#         return π

#     def get_node_count(self) -> int:
#         """Return total number of nodes created so far."""
#         return self._total_nodes
