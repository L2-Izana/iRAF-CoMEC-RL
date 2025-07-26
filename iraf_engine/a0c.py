import pickle
import time
from typing import List, Tuple, Union, override
from torch.distributions import Beta
import torch

from typing import Tuple, List

import numpy as np
from comec_simulator.core.constants import *
from iraf_engine.dnn import A0CBetaPolicyNet
from iraf_engine.mcts_pw import MIN_PW_FLOOR
from iraf_engine.node import A0C_Node, A0C_Node_DNN


with open(r"D:\Research\IoT\iRAF-CoMEC-RL\beta_mixture_params_subaction_0.pkl", "rb") as f:
    mixture_params_0 = pickle.load(f)
with open(r"D:\Research\IoT\iRAF-CoMEC-RL\beta_mixture_params_subaction_1.pkl", "rb") as f:
    mixture_params_1 = pickle.load(f)
with open(r"D:\Research\IoT\iRAF-CoMEC-RL\beta_mixture_params_subaction_3.pkl", "rb") as f:
    mixture_params_3 = pickle.load(f)
with open(r"D:\Research\IoT\iRAF-CoMEC-RL\beta_mixture_params_subaction_4.pkl", "rb") as f:
    mixture_params_4 = pickle.load(f)

# Simple Beta params for subaction 2
with open(r"D:\Research\IoT\iRAF-CoMEC-RL\beta_params_subaction_2.pkl", "rb") as f:
    beta_params_2 = pickle.load(f)  # (alpha, beta)

# Organize into dict for easy lookup
mixture_params = {
    0: mixture_params_0,
    1: mixture_params_1,
    3: mixture_params_3,
    4: mixture_params_4,
}

class A0C:
    def __init__(self, has_max_threshold, max_pw_floor, discount_factor):
        self.total_nodes = 1
        self.root = A0C_Node(depth=0)
        self.current_node = self.root
        self.global_step = 0 # used for adaptive
        self.is_adaptive = False  # Set to True if you want to use adaptive progressive widening (still in experimentation)
        self.has_max_threshold = has_max_threshold 
        self.max_pw_floor = max_pw_floor
        self.discount_factor = discount_factor
        
    def backprop_accumulative(self, reward: float):
        """Update node statistics upward through the tree"""
        while self.current_node is not None and self.current_node.parent is not None:
            self.current_node.N += 1
            self.current_node.Q += reward
            self.current_node = self.current_node.parent
        if self.current_node is not None:
            self.current_node.N += 1
            self.current_node.Q += reward

    def backprop_discounted_average(self, reward: float, avg_lat_eng_arr: List[float]):
        # Fuck the reward, use the average immediate reward arr
        num_tasks = len(avg_lat_eng_arr)
        for i in range(-2, -num_tasks-1, -1):
            avg_lat_eng_arr[i] = avg_lat_eng_arr[i] + self.discount_factor*avg_lat_eng_arr[i+1]
        for i in range(-1, -num_tasks-1, -1):
            if self.current_node is None:
                raise AssertionError("Bullshit")
            self.current_node.N += 1
            self.current_node.W += avg_lat_eng_arr[i]
            self.current_node.Q = self.current_node.W / self.current_node.N
            self.current_node = self.current_node.parent
        self.current_node.N += 1
        
        
    def get_ratios(self, env_resources) -> Tuple[float, ...]:
        # Update tree properties
        self.global_step += 1
        self.current_node.state = env_resources

        floor = self._get_progressive_widening_floor(self.current_node)
        
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
            return tuple(action)
        # Otherwise select best
        else:
            chosen = self._best_child(self.current_node)
            self.current_node = chosen
            assert chosen.action is not None, f"Chosen node {chosen} has no action, wrong tree logic"
            return tuple(chosen.action)
    
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
    
    def get_eval_ratios(self, device='cpu') -> Tuple[float, ...]:
        """
        Sample a 5-dimensional action vector from pre-fitted Beta / Beta-mixture distributions.
        Returns a tuple of floats.
        """
        action = torch.zeros(5, device=device)

        for i in range(5):
            if i == 2:
                # Subaction 2: single Beta
                a, b = beta_params_2
                dist = Beta(torch.tensor(a, device=device),
                            torch.tensor(b, device=device))
                action[i] = dist.sample()
            else:
                # Mixture of Betas for subactions 0,1,3,4
                params = mixture_params[i]              # list of (w, α, β)
                weights = torch.tensor([w for w,_,_ in params],
                                    device=device)
                weights = weights / weights.sum()       # normalize
                idx = torch.multinomial(weights, 1).item()
                _, a, b = params[idx]
                dist = Beta(torch.tensor(a, device=device),
                            torch.tensor(b, device=device))
                action[i] = dist.sample()

        assert all(0 < action) and all(action <= 1), \
            f"Sampled action {action} has values outside [0, 1] range"
        
        return tuple(float(x) for x in action)    
    
    def _sample_action_5d(self, device='cpu') -> List[float]:
        """
        Sample a 5-dimensional action vector from independent Beta distributions.

        Args:
            alpha_val (float): The alpha parameter for each Beta distribution.
            beta_val (float): The beta parameter for each Beta distribution.
            device (str): The device to perform computation on (e.g., 'cpu' or 'cuda').

        Returns:
            torch.Tensor: A tensor of shape (5,) containing the sampled action.
        """
        alpha = torch.full((5,), ALPHA_VAL, device=device)
        beta = torch.full((5,), BETA_VAL, device=device)
        dist = Beta(alpha, beta)
        action = dist.sample().tolist()
        return action
        
    def _get_progressive_widening_floor(self, node: A0C_Node) -> int:
        if self.is_adaptive:
            k, alpha = self._get_adaptive_pw_params()
            floor = int(k * (node.N ** alpha))
        else:
            floor = int(K_PW * (node.N ** ALPHA_PW))
        
        if self.has_max_threshold:
            return min(max(floor, MIN_PW_FLOOR), self.max_pw_floor)
        else:
            return max(floor, MIN_PW_FLOOR)


    def _get_adaptive_pw_params(self) -> Tuple[float, float]:
        """
        Decay k and alpha linearly over pw_decay_steps iterations,
        then hold at their minimum values.
        """

        t = min(self.global_step / 10_000, 1.0) # TODO: still in experimentation, 10_000 is a good value for now
        k = ADAPTIVE_INITIAL_K * (1 - t) + ADAPTIVE_MIN_K * t
        alpha = ADAPTIVE_INITIAL_ALPHA * (1 - t) + ADAPTIVE_MIN_ALPHA * t
        return k, alpha
    
    def _best_child(self, node: A0C_Node) -> A0C_Node:
        # This stupid shit, fuck up, if discounted avg, do not divide Q, if accumulate, divide by N
        def uct(child: A0C_Node):
            exploitation = child.Q / child.N # For accumulative 
            # exploitation = child.Q # For discounted avg
            exploration = EXPLORATION_BONUS * np.sqrt(np.log(node.N) / child.N )
            return exploitation + exploration
        scores = [uct(child) for child in node.children]
        idx = int(np.argmax(scores))
        return node.children[idx]

class A0C_DNN(A0C):
    def __init__(self):
        super().__init__()
        self.root = A0C_Node_DNN(depth=0)
        self.current_node: A0C_Node_DNN = self.root
        self.dnn = A0CBetaPolicyNet()
        state_dict = torch.load("D:/Research/IoT/iRAF-CoMEC-RL/best_action_A0C_Policy_Net.pth", map_location="cpu", weights_only=True)
        self.dnn.load_state_dict(state_dict)
        self.dnn.eval()
        self.num_subactions = 5  # Fix: define number of subactions
    
    
    def get_ratios_a0c_dnn(self, env_resources) -> List[float]:
        # Update tree properties
        self.global_step += 1
        self.current_node.state = env_resources

        floor = self._get_progressive_widening_floor(self.current_node, is_adaptive=False, has_max_threshold=False)
        
        # Expand if under floor
        if floor > len(self.current_node.children): 
            # Get DNN output
            if not self.current_node.has_dnn_output():
                dnn_out = self.dnn(torch.tensor(env_resources, dtype=torch.float32)
                                                .unsqueeze(0))
                self.current_node.alphas, self.current_node.betas = dnn_out
                alphas = self.current_node.alphas.squeeze(0).tolist()
                betas = self.current_node.betas.squeeze(0).tolist()
            else:
                alphas, betas = self.current_node.alphas, self.current_node.betas
            if alphas is None or betas is None:
                raise ValueError("alphas and betas must not be None before sampling action.")
            action = self._sample_action_5d_dnn(alphas, betas, device='cpu')
            node = A0C_Node_DNN(action=action, 
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
            if not isinstance(chosen, A0C_Node_DNN):
                chosen = A0C_Node_DNN(action=chosen.action, depth=chosen.depth, parent=chosen.parent)
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
