from typing import List, Tuple, Union, override
from torch.distributions import Beta
import torch

from typing import Tuple, List

import numpy as np
from iraf_engine.dnn import A0CBetaPolicyNet
from iraf_engine.mcts_pw import MIN_PW_FLOOR
from iraf_engine.node import A0C_Node, A0C_Node_DNN

ADAPTIVE_INITIAL_K = 2
ADAPTIVE_MIN_K = 0.7
ADAPTIVE_INITIAL_ALPHA = 0.8
ADAPTIVE_MIN_ALPHA = 0.3

K_PW = 2 # Great idea, but not so great results, maybe it's due to the so low count in the end of tree
ALPHA_PW = 0.7

MAX_PW_FLOOR = 20 # Limit exploration, empirically best results, need further testing and proof 

class A0C:
    def __init__(
        self,
        input_dim,
        exploration_constant: float = 0.8,
        num_subactions: int = 5,
        use_dnn: bool = False,
        num_iterations: int = 10000
    ):
        self.c = exploration_constant
        self.num_subactions = num_subactions
        self.use_dnn = use_dnn
        self.total_nodes = 1
        self.root = A0C_Node(depth=0)
        self.current_node = self.root
        self.global_step = 0 # used for adaptive
        self.num_iterations = num_iterations
        
    def backprop(self, reward: float):
        """Update node statistics upward through the tree"""
        while self.current_node.parent is not None:
            self.current_node.N += 1
            self.current_node.Q += reward
            self.current_node = self.current_node.parent
        self.current_node.N += 1
        self.current_node.Q += reward

    def get_ratios_a0c(self, env_resources) -> List[float]:
        # Update tree properties
        self.global_step += 1
        self.current_node.state = env_resources

        floor = self._get_progressive_widening_floor(self.current_node, is_adaptive=False, has_max_threshold=True)
        
        # Expand if under floor
        if floor > len(self.current_node.children): 
        # if floor > self.current_node.N: # STUPID DUMPSHITS, yield good results, but interesting good
            action = self._sample_action_5d(device='cpu')
            node = A0C_Node(action=action, 
                            depth=self.current_node.depth+1, 
                            parent=self.current_node)
            self.current_node.children.append(node)
            self.total_nodes += 1
            self.current_node.expanded = True
            self.current_node = node
            return list(action)
        # Otherwise select best
        else:
            chosen = self._best_child(self.current_node)
            self.current_node = chosen
            return chosen.action if chosen.action is not None else [0.0] * self.num_subactions    
    
    def get_best_action(self):
        best_action = []
        node = self.current_node
        while node.children:
            best_child = max(node.children, key=lambda x: x.N) # IMPORTANT: In A0C paper, the optimal policy is proportional to the visits
            best_action.append(best_child.action)
            node = best_child
        best_action = np.array(best_action)
        return best_action
    
    def get_training_dataset(self):
        """Extract (state, action, visit_count) tuples from the entire MCTS tree"""
        dataset = []
        queue = [self.root]  # start from root node

        while queue:
            node = queue.pop(0)
            for child in node.children:
                if node.state is not None: 
                    entry = {
                        "state": torch.tensor(node.state, dtype=torch.float32),     # parent state
                        "action": torch.tensor(child.action, dtype=torch.float32),  # action to child
                        "visit_count": child.N  # visitation count of child
                    }
                    dataset.append(entry)
                    queue.append(child)
        return dataset
                        
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
    
    
    def _sample_action_5d(self, alpha_val=2.0, beta_val=0.5, device='cpu') -> List[float]:
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
        action = dist.sample().tolist()
        return action
    
    
    def _get_progressive_widening_floor(self, node: A0C_Node, is_adaptive=False, has_max_threshold=True) -> int:
        if is_adaptive:
            k, alpha = self._get_adaptive_pw_params()
            floor = int(k * (node.N ** alpha))
        else:
            floor = int(K_PW * (node.N ** ALPHA_PW))
        
        if has_max_threshold:
            return min(max(floor, MIN_PW_FLOOR), MAX_PW_FLOOR)
        else:
            return max(floor, MIN_PW_FLOOR)


    def _get_adaptive_pw_params(self) -> Tuple[float, float]:
        """
        Decay k and alpha linearly over pw_decay_steps iterations,
        then hold at their minimum values.
        """

        t = min(self.global_step / self.num_iterations, 1.0)
        k = ADAPTIVE_INITIAL_K * (1 - t) + ADAPTIVE_MIN_K * t
        alpha = ADAPTIVE_INITIAL_ALPHA * (1 - t) + ADAPTIVE_MIN_ALPHA * t
        return k, alpha
    
    def _best_child(self, node: A0C_Node) -> A0C_Node:
        def uct(child: A0C_Node):
            exploitation = child.Q / child.N
            exploration = self.c * np.sqrt(np.log(node.N) / child.N )
            return exploitation + exploration
        scores = [uct(child) for child in node.children]
        idx = int(np.argmax(scores))
        return node.children[idx]

class A0C_DNN(A0C):
    def __init__(
        self,
        input_dim,
        exploration_constant: float = 0.8,
        num_subactions: int = 5,
        use_dnn: bool = True,
        num_iterations: int = 10000
    ):
        super().__init__(input_dim, exploration_constant, num_subactions, use_dnn, num_iterations)
        self.root = A0C_Node_DNN(depth=0)
        self.current_node = self.root
        self.dnn = A0CBetaPolicyNet(input_dim, hidden_dim=128)
        state_dict = torch.load("D:/Research/IoT/iRAF-CoMEC-RL/best_action_A0C_Policy_Net.pth", map_location="cpu", weights_only=True)
        self.dnn.load_state_dict(state_dict)
        self.dnn.eval()
    
    
    def get_ratios_a0c_dnn(self, env_resources) -> List[float]:
        # Update tree properties
        self.global_step += 1
        self.current_node.state = env_resources

        floor = self._get_progressive_widening_floor(self.current_node, is_adaptive=False, has_max_threshold=False)
        
        # Expand if under floor
        if floor > len(self.current_node.children): 
        # if floor > self.current_node.N: # STUPID DUMPSHITS, yield good results, but interesting good
            # Get DNN output
            if not self.current_node.has_dnn_output():
                dnn_out = self.dnn(torch.tensor(env_resources, dtype=torch.float32)
                                                .unsqueeze(0))
                self.current_node.alphas, self.current_node.betas = dnn_out
                alphas = self.current_node.alphas.squeeze(0).tolist()
                betas = self.current_node.betas.squeeze(0).tolist()
            else:
                alphas, betas = self.current_node.alphas, self.current_node.betas
            action = self._sample_action_5d_dnn(alphas, betas, device='cpu')
            node = A0C_Node_DNN(action=action, 
                            depth=self.current_node.depth+1, 
                            parent=self.current_node)
            self.current_node.children.append(node)
            self.total_nodes += 1
            self.current_node.expanded = True
            self.current_node = node
            assert len(action) == self.num_subactions, f"Expected action length {self.num_subactions}, got {len(action)}: action={action}, alphas={alphas}, betas={betas}"
            return list(action)
        # Otherwise select best
        else:
            chosen = self._best_child(self.current_node)
            self.current_node = chosen
            return chosen.action if chosen.action is not None else [0.0] * self.num_subactions    

    def _sample_action_5d_dnn(self, alphas: Union[List[float], torch.Tensor], betas: Union[List[float], torch.Tensor], device='cpu') -> List[float]:
        """
        Sample a 5-dimensional action vector from independent Beta distributions.

        Args:
            alphas (list or Tensor): Alpha parameters for each of the 5 Beta distributions.
            betas (list or Tensor): Beta parameters for each of the 5 Beta distributions.
            device (str): Computation device ('cpu' or 'cuda').

        Returns:
            List[float]: A list of 5 sampled action values in (0, 1).
        """
        # Safely convert to tensor
        alphas = torch.tensor(alphas, device=device) if not isinstance(alphas, torch.Tensor) else alphas.to(device)
        betas  = torch.tensor(betas,  device=device) if not isinstance(betas,  torch.Tensor) else betas.to(device)

        # Remove batch dim if exists
        if alphas.dim() == 2 and alphas.size(0) == 1:
            alphas = alphas.squeeze(0)
            betas  = betas.squeeze(0)

        dist = Beta(alphas, betas)
        action = dist.sample().tolist()  # flat 5-element list
        return action
