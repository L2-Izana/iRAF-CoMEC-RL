import logging
import gymnasium as gym
import random
import numpy as np
import torch
import yaml
import os
from stable_baselines3 import A2C, PPO

# ensure environment registration
import comec_gym.env

# Set seeds for reproducibility
random.seed(187)
np.random.seed(187)
torch.manual_seed(187)
torch.cuda.manual_seed_all(187)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    # Load YAML config
    config_path = os.path.join("conf", "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Prepare kwargs for CoMECGymEnv
    env_kwargs = {
        "num_edge_servers": config.get("num_es"),
        "num_clusters": config.get("num_clusters"),
        "cpu_capacity": float(config.get("cpu_capacity")),
        "num_devices_per_cluster": config.get("num_devices_per_cluster")
    }

    # Instantiate environment
    env = gym.make("CoMEC-v0", **env_kwargs)

    # Select baseline algorithm
    algo_name = config.get("algorithm", "a2c").lower()
    if algo_name in ["a2c", "a0c"]:
        model_cls = A2C
    elif algo_name == "ppo":
        model_cls = PPO
    else:
        raise ValueError(f"Unsupported algorithm '{algo_name}' in config")

    # Initialize model
    model = model_cls(
        "MlpPolicy",
        env,
        verbose=1,
        gamma=config.get("discount_factor", 0.99)
    )

    # Train model
    total_timesteps = config.get("num_iter", 100000)
    logging.info(f"Training {algo_name.upper()} for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)

    # Save model
    save_dir = os.path.join("models", algo_name)
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{algo_name}_baseline")
    model.save(model_path)
    logging.info(f"Model saved to {model_path}.zip")

    # (Optional) Evaluate trained model
    obs, _ = env.reset(seed=187)
    done = False
    episode_reward = 0.0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += float(reward)
        done = terminated or truncated
    logging.info(f"Evaluation Episode Reward: {episode_reward:.3f}")


if __name__ == "__main__":
    main()
