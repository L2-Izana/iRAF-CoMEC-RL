from typing import Optional, Tuple, List

class Node:
    def __init__(self, action: Optional[Tuple[int, int, float]] = None, depth: int = 0, num_subactions: int = 5, parent = None):
        self.children: List[Node] = []
        self.parent: Optional[Node] = parent
        self.N = 0  # Visit count
        self.Q = 0.0  # Total value
        self.W = 0.0  # Total weight
        self.reward = 0.0  # Total reward   
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
    def __init__(self, action: Optional[Tuple[int, int, float]] = None, prior: float = 0.0, depth: int = 0, num_subactions: int = 5, parent = None, task_dnn_output=None):
        super().__init__(action, depth, num_subactions, parent)
        self.prior = prior  # Prior from policy network
        self.value_sum = 0.0  # Sum of NN values for average estimation
        self.task_dnn_output = task_dnn_output # A helper storage of DNN output for each task, used to solve the bug where for retravesal, it explores new subactions but we only get DNN output for the original subaction

    def get_mean_value(self) -> float:
        return self.value_sum / self.N if self.N > 0 else 0.0

    def __str__(self):
        return f"[AlphaZeroNode] depth={self.depth}, action={self.action}, prior={self.prior:.3f}, Q={self.Q:.3f}, N={self.N}"

class Node_PW(AlphaZeroNode):
    def __init__(self, action: Optional[Tuple[int, int, float]] = None, prior: float = 0.0, depth: int = 0, num_subactions: int = 5, parent = None, task_dnn_output=None):
        super().__init__(action, prior, depth, num_subactions, parent, task_dnn_output)
        self.selected_bins = []
    
    def get_unselected_bins(self, top_k_indices: List[int]) -> List[int]:
        return [i for i in top_k_indices if i not in self.selected_bins]
    
    def update_selected_bins(self, top_k_indices: List[int]):
        self.selected_bins = top_k_indices
        
class A0C_Node:
    def __init__(self, state=None, action: Tuple[float, ...] = None, depth: int = 0, num_subactions: int = 5, parent=None):
        self.children: List[A0C_Node] = []
        self.parent: Optional[A0C_Node] = parent
        self.N = 0  # Visit count
        self.Q = 0.0
        self.action = action
        self.expanded = False
        self.depth = depth
        self.num_subactions = num_subactions
        self.state = state

    def is_terminal(self, total_tasks: int) -> bool:
        return self.depth == total_tasks * self.num_subactions

    def __str__(self):
        return f"[A0C_Node] depth={self.depth}, action={self.action}, Q={self.Q:.3f}, N={self.N}"
