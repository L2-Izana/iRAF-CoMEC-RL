# dnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SubActionHead(nn.Module):
    """
    A small MLP head for one sub-action. Takes the shared embedding as input
    and outputs a probability distribution over that sub-action's bins.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)

class IRafMultiTaskDNN(nn.Module):
    """
    Multi-head network for IRAF:
    - A shared MLP trunk that maps from `input_dim` → 128.
    - Five separate SubActionHead modules, each producing a softmax over its bins.
    """
    def __init__(self, input_dim: int = 9, head_dims: list[int] = [20, 10, 10, 10, 10]):
        super().__init__()
        # shared trunk
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # one head per sub-action
        self.heads = nn.ModuleList([
            SubActionHead(128, dim) for dim in head_dims
        ])

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        x: shape (batch_size, input_dim)
        returns: list of length 5, each a 1D tensor of shape (head_dim_i,)
        """
        shared_feat = self.shared(x)  # → (batch_size, 128)
        raw_outputs = [head(shared_feat) for head in self.heads]  # list of (batch_size, dim_i)
        # Convert to 1D tensors if batch_size=1
        probs = [out[0] for out in raw_outputs]
        return probs

class A0CBetaPolicyNet(nn.Module):
    def __init__(self, state_dim=9, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.alpha_head = nn.Linear(hidden_dim, 5)
        self.beta_head = nn.Linear(hidden_dim, 5)

    def forward(self, state) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(state)
        alpha = F.softplus(self.alpha_head(x)) + 1e-4
        beta = F.softplus(self.beta_head(x))  + 1e-4
        return alpha, beta
    
if __name__ == "__main__":
    # Sanity check
    model = IRafMultiTaskDNN(input_dim=9, head_dims=[20, 10, 10, 10, 10])
    x = torch.randn(1, 9)
    outputs = model(x)
    for i, p in enumerate(outputs):
        print(f"Head {i} output shape: {p.shape}")  # should be (head_dim,)
