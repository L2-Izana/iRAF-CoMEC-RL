import os
import yaml
import random
import numpy as np
import torch
import gymnasium as gym

# Ensure CoMEC-v0 is registered
import comec_gym.env

# Set seeds
torch.manual_seed(187)
np.random.seed(187)
random.seed(187)

# Load config
with open(os.path.join("conf", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Prepare environment
env_kwargs = {
    "num_edge_servers": config["num_es"],
    "num_clusters": config["num_clusters"],
    "cpu_capacity": float(config["cpu_capacity"]),
    "num_devices_per_cluster": config["num_devices_per_cluster"]
}
env = gym.make("CoMEC-v0", **env_kwargs)
obs, _ = env.reset()

# Step through the environment with random actions
print("Initial observation:", obs)
terminated = False
step = 0
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1}:")
    step += 1
    print("  Action:", action)
    print("  Obs:", obs)
    print("  Reward:", reward)
    print("  Terminated:", terminated)
    print("  Truncated:", truncated)
    print("  Info:", info)
    if terminated or truncated:
        obs, _ = env.reset()
        print("  --- Environment reset ---")
# print(env.handle_complete_count)
env.close()
