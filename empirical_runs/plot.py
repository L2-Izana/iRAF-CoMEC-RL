import matplotlib.pyplot as plt
import numpy as np

# Algorithm labels
algorithms = ['MCTS', 'MCTS+DNN', 'MCTS+PW', 'MCTS+PW+DNN']
i = 0
tasks = 20
i = i*4
# Folder names (adjust according to your `dir` listing)
folders = [
    f'number_{i+1}_{tasks}_mcts',
    f'number_{i+2}_{tasks}_mcts-dnn',
    f'number_{i+3}_{tasks}_mcts-pw',
    f'number_{i+4}_{tasks}_mcts-pw-dnn'
]

# Colors for better contrast
colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:red']

# Smoothing window
window = 100

# Load data
rewards = []
node_counts = []
for folder in folders:
    rewards.append(np.load(f"{folder}/rewards.npy")[:12000])
    node_counts.append(np.load(f"{folder}/node_counts.npy")[:12000])

# Create plot
plt.figure(figsize=(12, 10))

# Subplot 1: Smoothed rewards
plt.subplot(2, 1, 1)
for i, reward in enumerate(rewards):
    if len(reward) >= window:
        sma = np.convolve(reward, np.ones(window)/window, mode='valid')
        std = np.array([np.std(reward[j-window:j]) for j in range(window, len(reward)+1)])
        x = np.arange(len(sma))
        plt.plot(x, sma, label=algorithms[i], color=colors[i])
        plt.fill_between(x, sma - std, sma + std, alpha=0.3, color=colors[i])
    else:
        plt.plot(reward, label=f"{algorithms[i]} (raw)", color=colors[i])

plt.title("Smoothed Rewards Over Iterations")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)

# Subplot 2: Node counts
plt.subplot(2, 1, 2)
for i, count in enumerate(node_counts):
    plt.plot(count, label=algorithms[i], color=colors[i], linewidth=2.5)
plt.title("Node Count Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Node Count")
plt.legend()
plt.grid(True)

# Super title
plt.suptitle(f"Number of Tasks = {tasks}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"analysis_plot_task_{tasks}.png", dpi=300)
plt.show()
