# import torch
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from torch import nn
# from torch.distributions import Beta

# from iraf_engine.dnn import A0CBetaPolicyNet

# def train_and_plot_a0c_policy(policy_net, dataset_path, tau=1.0, lr=1e-3, epochs=200, batch_size=64,
#                               model_save_path="A0C_Policy_Net.pth",
#                               loss_plot_path="training_loss.png",
#                               beta_plot_path="dnn_beta_distributions.png",
#                               num_beta_samples=3):
#     optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
#     loss_log = []
#     dataset = torch.load(dataset_path, weights_only=True)

#     # Balanced sampling
#     high_visit = [d for d in dataset if d['visit_count'] > 1]
#     low_visit  = [d for d in dataset if d['visit_count'] == 1]
#     n = len(high_visit)
#     low_visit_sample = random.sample(low_visit, min(len(low_visit), n))
#     balanced_dataset = high_visit + low_visit_sample
#     random.shuffle(balanced_dataset)

#     # === Training Loop ===
#     for epoch in range(epochs):
#         torch.random.manual_seed(epoch)
#         perm = torch.randperm(len(balanced_dataset))
#         for i in range(0, len(balanced_dataset), batch_size):
#             batch = [balanced_dataset[j] for j in perm[i:i+batch_size]]
#             states = torch.stack([b['state'] for b in batch])
#             actions = torch.stack([b['action'] for b in batch])
#             log_counts = torch.log(torch.tensor([b['visit_count'] for b in batch], dtype=torch.float32)) * tau

#             alpha, beta = policy_net(states)
#             dist = Beta(alpha, beta)
#             log_probs = dist.log_prob(actions).sum(dim=1)

#             loss = ((log_probs - log_counts) ** 2).mean()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             loss_log.append(loss.item())

#         print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

#     # === Save model ===
#     torch.save(policy_net.state_dict(), model_save_path)
#     print(f"💾 Saved model to: {model_save_path}")

#     # === Plot and save training loss ===
#     plt.figure(figsize=(8, 4))
#     plt.plot(loss_log)
#     plt.xlabel("Training Step")
#     plt.ylabel("Loss")
#     plt.title("A0C Policy Network Training Loss")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(loss_plot_path)
#     print(f"📉 Saved loss plot to: {loss_plot_path}")

#     # === Plot and save Beta distributions for sample states ===
#     fig, axes = plt.subplots(num_beta_samples, 5, figsize=(15, 3 * num_beta_samples))
#     samples = random.sample(balanced_dataset, num_beta_samples)

#     for i, sample in enumerate(samples):
#         state = sample['state'].unsqueeze(0)  # shape (1, D)
#         with torch.no_grad():
#             alpha, beta = policy_net(state)
#         alpha, beta = alpha.squeeze(0), beta.squeeze(0)

#         x = np.linspace(0.001, 0.999, 200)
#         for j in range(5):
#             a, b = alpha[j].item(), beta[j].item()
#             dist = Beta(torch.tensor([a]), torch.tensor([b]))
#             y = dist.log_prob(torch.tensor(x)).exp().numpy()
#             axes[i][j].plot(x, y)
#             axes[i][j].set_title(f"Dim {j+1} | α={a:.2f}, β={b:.2f}")
#             axes[i][j].set_ylim(0, np.max(y)*1.1)

#     plt.tight_layout()
#     plt.savefig(beta_plot_path)
#     print(f"📊 Saved Beta distribution plots to: {beta_plot_path}")

# if __name__ == "__main__":
#     policy_net = A0CBetaPolicyNet(state_dim=9)
#     train_and_plot_a0c_policy(policy_net, "D:/Research/IoT/iRAF-CoMEC-RL/dataset.pt")
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.distributions import Beta

from iraf_engine.dnn import A0CBetaPolicyNet

def train_and_plot_a0c_policy(policy_net,
                              dataset_path,
                              tau: float = 1.0,
                              lr: float = 1e-3,
                              epochs: int = 100,
                              batch_size: int = 64,
                              lambda_entropy: float = 0.01,
                              model_save_path: str = "A0C_Policy_Net_new.pth",
                              loss_plot_path: str = "training_loss_new.png",
                              beta_plot_path: str = "dnn_beta_distributions_new.png",
                              num_beta_samples: int = 3):
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

    # Training loop
    for epoch in range(epochs):
        torch.random.manual_seed(epoch)
        perm = torch.randperm(len(balanced_dataset))

        for i in range(0, len(balanced_dataset), batch_size):
            batch = [balanced_dataset[j] for j in perm[i:i+batch_size]]
            states = torch.stack([b['state']   for b in batch])
            actions= torch.stack([b['action']  for b in batch])
            counts = torch.tensor([b['visit_count'] for b in batch], dtype=torch.float32)
            log_counts = torch.log(counts) * tau  # log π̂ ∝ τ·log n

            # forward
            alpha, beta = policy_net(states)
            dist = Beta(alpha, beta)
            log_probs = dist.log_prob(actions).sum(dim=1)  # log πϕ(a|s)
            entropy  = dist.entropy().sum(dim=1)           # H[πϕ(·|s)]

            # KL loss: E[log πϕ − log π̂]
            # approximate with samples from dataset,
            # using REINFORCE surrogate: (log_probs - log_counts) * log_probs
            policy_loss = ((log_probs - log_counts.detach()) * log_probs).mean()

            # entropy bonus (we *maximize* entropy, so subtract it)
            loss = policy_loss - lambda_entropy * entropy.mean()

            # step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_log.append(loss.item())

        print(f"Epoch {epoch:3d}  loss={loss.item():.5f}  pol={policy_loss.item():.5f}  ent={entropy.mean().item():.3f}")

    # === Save the trained weights ===
    torch.save(policy_net.state_dict(), model_save_path)
    print(f"💾 Saved model to: {model_save_path}")

    # === Plot & save loss curve ===
    plt.figure(figsize=(8,4))
    plt.plot(loss_log, linewidth=1)
    plt.xlabel("Training Step")
    plt.ylabel("Total Loss")
    plt.title("A0C Policy Network Loss (KL + Entropy)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"📉 Saved loss plot to: {loss_plot_path}")

    # === Plot a few Beta(α,β) shapes ===
    fig, axes = plt.subplots(num_beta_samples, 5, figsize=(15, 3*num_beta_samples))
    samples = random.sample(balanced_dataset, num_beta_samples)
    x = np.linspace(0.001, 0.999, 200)

    for i, sample in enumerate(samples):
        state = sample['state'].unsqueeze(0)
        with torch.no_grad():
            α, β = policy_net(state)
        α, β = α.squeeze(0), β.squeeze(0)

        for j in range(5):
            a_, b_ = α[j].item(), β[j].item()
            d = Beta(torch.tensor([a_]), torch.tensor([b_]))
            y = d.log_prob(torch.tensor(x)).exp().numpy()
            axes[i][j].plot(x, y)
            axes[i][j].set_title(f"dim{j+1}: α={a_:.2f}, β={b_:.2f}")
            axes[i][j].set_ylim(0, np.max(y)*1.1)

    plt.tight_layout()
    plt.savefig(beta_plot_path)
    print(f"📊 Saved Beta plots to: {beta_plot_path}")

if __name__ == "__main__":
    net = A0CBetaPolicyNet(state_dim=9)
    train_and_plot_a0c_policy(net, "D:/Research/IoT/iRAF-CoMEC-RL/dataset.pt")
