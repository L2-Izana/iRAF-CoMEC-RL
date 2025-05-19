from typing import List, Optional, Tuple


from typing import Optional, Tuple, List

import numpy as np

from comec_simulator.core.components import BaseStation, EdgeServer, Task

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
        print(f"Backpropping with reward: {reward}")
        """Update node statistics upward through the tree"""
        # node = self.current_node
        while self.current_node.parent is not None:
            self.current_node.N += 1
            self.current_node.Q += reward
            self.current_node = self.current_node.parent
        self.current_node.N += 1
        self.current_node.Q += reward
        # self.current_node = self.root
        # print(f"Done the backprop, check the update of the tree")
        # for child in self.root.children:
        #     print(f"Visits: {child.N}, Q: {child.Q}", end=", ")
        # print()
        
    def best_child(self, node: Node) -> Node:
        """Select the best child based on UCB score"""
        def uct(child: Node):
            if child.N == 0:
                return float('inf')
            # For negative rewards, higher (less negative) Q/N is better
            exploitation = child.Q
            
            # Add exploration bonus
            exploration = self.c * np.sqrt(np.log(node.N+1) / child.N)
            
            return exploitation + exploration
        
        def puct(child: AlphaZeroNode):
            exploitation = child.Q
            exploration = self.c * child.prior * np.sqrt(np.log(node.N+1)) / (1+ child.N)
            return exploitation + exploration

        # Select best child based on UCB
        best_child = max(node.children, key=uct)
        return best_child
    
    
    
    def get_ratios(self, env_resources) -> Tuple[float, float, float, float, float]:
        """
        Get the ratios for the task: So for each task other environment conditions, we c
        Args:
            env_resources: List[float]
        Returns:
            Tuple[float, float, float, float, float]
        """
        # input_tensor = self.create_normalize_input_tensor(task, bs, edge_servers)
        # print(input_tensor)
        # with torch.no_grad():
        #     priors = self.model(input_tensor)
        # # priors_arr = priors.numpy().flatten()
        # ratios = np.ones(5)
        # # At this point, use this priors, need to change to selection (best child) algorithm
        # for i, prior in enumerate(priors):
        #     max_bin = self.bins[i][torch.argmax(prior)]
        #     # print(f"max_bin: {max_bin}")
        #     ratios[i] = max_bin
        # return tuple(ratios)
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
    
    # def expand(self, node: Node):
    #     """Expand a node by adding all possible children"""
    #     task_idx, subaction_idx = node.get_node_index()
    #     for i in range(self.num_subactions):
            
    #     # Prepare input tensor to DNN
    #     input_tensor = normalize_input(F_e, B, T_k)
    #     with torch.no_grad():
    #         priors = self.model(input_tensor)
        
    #     # Define bins based on subaction type
    #     if subaction_idx == 0:
    #         bins = np.linspace(0.0, 0.95, self.bins_per_subaction[0]).tolist()
    #     else:
    #         bins = np.linspace(0.0, 0.9, self.bins_per_subaction[subaction_idx]).tolist()
            
    #     prior_probs = priors[subaction_idx].numpy().flatten()
    #     for j, bin_value in enumerate(bins):
    #         action = (task_idx, subaction_idx, bin_value)
    #         if node.depth % 5 == 4:
    #             new_F_e, new_B =  self.apply_full_task_action(task_idx, node, T_k, F_e, B, subaction_4=bin_value)
    #             new_state = State(new_F_e, new_B, node.state.T)
    #             # self.dataset.append((normalize_input(new_F_e, new_B, T_k)))
    #             child = Node(state=new_state, depth=node.depth + 1)
    #         else:
    #             child = Node(state=node.state, depth=node.depth + 1)
    #         child.action = action
    #         child.parent = node
    #         child.prior = float(prior_probs[j])
    #         node.children.append(child)
            
    #     node.expanded = True

    def tree_policy(self, root: Node, K: int):
        """Traverse the tree to find a leaf node to evaluate"""
        node = root
        
        while not node.is_terminal():
            if not node.is_fully_expanded():
                self.expand(node)
                child = self.best_child(node)
                return self.tree_policy(child, K)
            else:
                child = self.best_child(node)
                node = child
        
        # Terminal node reached, evaluate the action
        # action_matrix = self.get_full_action_matrix(node, K)
        # F_e, B, tasks = original_state.F_e, original_state.B, original_state.T
        # env = MECEnvironment(f_e=F_e, B=B)
        # rewards = [env.step(action_matrix[i], tasks[i]) for i in range(len(tasks))]
        # total_reward = sum(rewards)
        # return node, total_reward

    # def get_full_action_matrix(self, node: Node, K: int) -> np.ndarray:
    #     """Reconstruct the full action matrix from a node and its ancestors"""
    #     A = np.ones((K, self.num_subactions), dtype=np.float32)
    #     current = node
    #     while current is not None:
    #         if current.action is not None:
    #             task_idx, subaction_idx, value = current.action
    #             A[task_idx][subaction_idx] = value
    #         current = current.parent
    #     assert np.any(A >= 1.0) == False and np.any(A < 0.0) == False
    #     return A
        
    # def extract_action_probabilities(self, root: Node, K: int) -> np.ndarray:
    #     """Extract action probabilities from tree statistics"""
    #     π = np.zeros((K, self.num_subactions, max(self.bins_per_subaction)))
        
    #     list_children = root.children
    #     while list_children:
    #         child = list_children.pop()
    #         if child.action is not None:
    #             t_idx, s_idx, val = child.action
    #             if s_idx == 0:
    #                 bin_id = int(round(val / (0.95 / (self.bins_per_subaction[0]-1))))
    #             else:
    #                 bin_id = int(round(val / (0.9 / (self.bins_per_subaction[s_idx]-1))))
    #             bin_id = min(bin_id, self.bins_per_subaction[s_idx] - 1)
    #             π[t_idx][s_idx][bin_id] += child.N
    #         list_children.extend(child.children)
                    
    #     # Normalize
    #     for t in range(K):
    #         for s in range(self.num_subactions):
    #             total = np.sum(π[t][s])
    #             if total > 0:
    #                 π[t][s] /= total
    
    #     return π

    # def extract_best_actions(self, root: Node, K: int) -> np.ndarray:
    #     """Extract the best action sequence from the tree"""
    #     A = np.ones((K, self.num_subactions), dtype=np.float32)
    #     node = root
    #     i = 0
    #     while node.depth < K * self.num_subactions and node.children:
    #         node = max(node.children, key=lambda child: child.Q / child.N if child.N > 0 else -float('inf'))
    #         if node.action is not None:
    #             task_idx, subaction_idx, value = node.action
    #             A[task_idx][subaction_idx] = value
    #     assert np.any(A >= 1.0) == False and np.any(A < 0.0) == False
    #     return A
    
    # def search(self, state: State, iterations: int):
    #     """Main MCTS search procedure"""
    #     F_e, B, tasks = state.F_e, state.B, state.T
    #     K = len(tasks)
        
    #     root = Node(state=state)
    #     rewards = []
        
    #     for i in range(iterations):
    #         leaf, reward = self.tree_policy(root, state, K)
    #         if i % 100 == 0:
    #             print(f"Iteration {i+1}/{iterations} with reward: {reward}")
            
    #         # if (i+1) % (iterations / 20) == 0:
    #         #     best_actions = self.extract_best_actions(root, K)
    #         #     action_probs = self.extract_action_probabilities(root, K)
    #         #     for j, (best_action, action_prob) in enumerate(zip(best_actions, action_probs)):
    #         #         # best_action is a 5x1 array subaction for a task
    #         #         # action_prob is a 5x20 array of probabilities for the subactions for a task
    #         #         self.dataset.append((normalize_input(F_e, B, tasks[j]), action_prob))
    #         #         F_e, B = self.apply_full_task_action_from_best_action(j, tasks[j], F_e, B, best_action)

    #         rewards.append(reward)
    #         self.backup(leaf, reward)
        
    #     with open("rewards.txt", "w") as f:
    #         for reward in rewards:
    #             f.write(f"{reward},")
        
    #     # Extract best actions and probabilities
    #     best_actions = self.extract_best_actions(root, K)
    #     action_probs = self.extract_action_probabilities(root, K)
        
    #     return best_actions, action_probs, rewards

    # def create_normalize_input_tensor(self, task: Task, bs: List[BaseStation], edge_servers: List[EdgeServer]) -> torch.Tensor:
    #     print(f"DEBUG: Task properties: {task}")
    #     task_properties_arr = np.array(task.get_properties_for_dnn())
    #     bs_bw_arr = np.array([bs_e.available_bandwidth / BANDWIDTH_PER_BS for bs_e in bs])
    #     es_cpu_arr = np.array([es_e.available_cpu / EDGE_SERVER_CPU_CAPACITY for es_e in edge_servers])
    #     input_tensor = np.concatenate((task_properties_arr, bs_bw_arr, es_cpu_arr))
    #     return input_tensor
