import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Beta
from iraf_engine.dnn import A0CBetaPolicyNet

# === Dataset loader ===
class PiDataset(Dataset):
    def __init__(self, root_dir, head_dims=[20, 10, 10, 10, 10]):
        self.env_data = []
        self.policy_data = []
        self.head_dims = head_dims

        for subdir in sorted(os.listdir(root_dir), key=lambda x: int(x)):
            env_path = os.path.join(root_dir, subdir, "env_resources_record.npy")
            policy_path = os.path.join(root_dir, subdir, "action_probabilities.npy")

            if os.path.exists(env_path) and os.path.exists(policy_path):
                env = np.load(env_path)
                policy = np.load(policy_path)
                # Re-normalize π if necessary
                for i in range(5):
                    valid_bins = head_dims[i]
                    policy[:, i, :valid_bins] /= np.clip(policy[:, i, :valid_bins].sum(axis=1, keepdims=True), 1e-8, None)

                self.env_data.append(env)
                self.policy_data.append(policy)

        self.env_data = np.concatenate(self.env_data, axis=0)  # (N, 9)
        self.policy_data = np.concatenate(self.policy_data, axis=0)  # (N, 5, 20)
        print("Loaded dataset:", self.env_data.shape, self.policy_data.shape)

    def __len__(self):
        return len(self.env_data)

    def __getitem__(self, idx):
        env_sample = torch.tensor(self.env_data[idx], dtype=torch.float32)
        policy_sample = torch.tensor(self.policy_data[idx], dtype=torch.float32)
        return env_sample, policy_sample

# === Model ===
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
        return [head(shared) for head in self.heads]

# === Training Function ===
def train_policy_model(data_path, epochs=400, batch_size=32, lr=1e-3, l2_coeff=1e-4, temperature=1.0):
    head_dims = [20, 10, 10, 10, 10]
    dataset = PiDataset(data_path, head_dims=head_dims)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = IRafMultiTaskDNN(input_dim=9, head_dims=head_dims)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []

    for epoch in range(epochs):
        model.train()
        total_kl = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch)

            loss = 0
            for i in range(5):
                pred = outputs[i]
                target = y_batch[:, i, :head_dims[i]]

                # Optional: temperature smoothing
                if temperature != 1.0:
                    log_pi = torch.log(target + 1e-8)
                    target = F.softmax(log_pi / temperature, dim=-1)

                pred_log = torch.log(pred + 1e-8)
                loss += F.kl_div(pred_log, target, reduction='batchmean')

            l2 = sum(p.pow(2).sum() for p in model.parameters())
            total_loss = loss + l2_coeff * l2

            total_loss.backward()
            optimizer.step()

            total_kl += loss.item()

        avg_kl = total_kl / len(dataloader)
        epoch_losses.append(avg_kl)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d}: KL Loss = {avg_kl:.4f}")

    # --- Save model ---
    torch.save(model.state_dict(), "trained_iraf_policy_small.pt")
    print("Model saved as 'trained_iraf_policy_small.pt'")

    # --- Save raw KL losses ---
    np.save("kl_loss_values_small_400.npy", np.array(epoch_losses))
    print("KL loss values saved to 'kl_loss_values_small_400.npy'")

    # --- Plot with SMA ---
    window = 10
    sma = np.convolve(epoch_losses, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), epoch_losses, label="KL Loss", linewidth=2)
    plt.plot(range(window - 1, epochs), sma, label=f"{window}-Epoch SMA", linestyle='--', linewidth=2)
    plt.title("KL Divergence Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("KL Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("kl_loss_curve_small_400.png")
    print("Loss curve saved as 'kl_loss_curve_small_400.png'")
    plt.show()

    return model


# ---- Training Loop (Simplified KL Loss) ----
def train_policy(policy_net, dataset_path, tau=1.0, lr=1e-3, epochs=100, batch_size=64):
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    loss_log = []
    dataset = torch.load(dataset_path, weights_only=True)

    # Balanced sampling
    high_visit = [d for d in dataset if d['visit_count'] > 1]
    low_visit  = [d for d in dataset if d['visit_count'] == 1]
    n = len(high_visit)
    low_visit_sample = random.sample(low_visit, min(len(low_visit), n))
    balanced_dataset = high_visit + low_visit_sample
    random.shuffle(balanced_dataset)

    for epoch in range(epochs):
        torch.random.manual_seed(epoch)  # reproducibility
        perm = torch.randperm(len(balanced_dataset))
        for i in range(0, len(balanced_dataset), batch_size):
            batch = [balanced_dataset[j] for j in perm[i:i+batch_size]]
            states = torch.stack([b['state'] for b in batch])
            actions = torch.stack([b['action'] for b in batch])
            log_counts = torch.log(torch.tensor([b['visit_count'] for b in batch], dtype=torch.float32)) * tau

            alpha, beta = policy_net(states)
            dist = Beta(alpha, beta)
            log_probs = dist.log_prob(actions).sum(dim=1)

            # Simple KL-inspired loss
            loss = ((log_probs - log_counts) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return loss_log

# === Run ===
if __name__ == "__main__":
    # train_policy_model("D:/Research/IoT/iRAF-CoMEC-RL/pi_dataset_small")
    policy_net = A0CBetaPolicyNet(9)
    train_policy(policy_net, "D:/Research/IoT/iRAF-CoMEC-RL/dataset.pt")
    torch.save(policy_net.state_dict(), "A0C_Policy_Net.pth")
