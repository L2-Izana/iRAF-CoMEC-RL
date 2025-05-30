import torch.nn as nn
import torch.nn.functional as F
import torch

class SubActionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.out(x), dim=-1)

class IRafMultiTaskDNN(nn.Module):
    def __init__(self, input_dim=9, head_dims=[20, 10, 10, 10, 10]):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.heads = nn.ModuleList([SubActionHead(128, dim) for dim in head_dims])

    def forward(self, x):
        shared = self.shared(x)
        heads = [head(shared) for head in self.heads]
        probs = [head[0] for head in heads]
        return probs
    
if __name__ == "__main__":
    model = IRafMultiTaskDNN()
    x = torch.randn(1, 9)
    print(model(x))