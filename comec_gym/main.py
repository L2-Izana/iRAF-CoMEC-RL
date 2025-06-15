# env/main.py

import gymnasium as gym
import env     # triggers registration
import random
import numpy as np
random.seed(187)
np.random.seed(187)

def main():
    env_inst = gym.make("CoMEC-v0")
    obs, _  = env_inst.reset(seed=187)

    rewards = []
    done = False

    while not done:
        action = env_inst.action_space.sample()
        obs, reward, done, truncated, info = env_inst.step(action)
        print(f"step→ reward={reward:.3f}, info={info}")
        rewards.append(reward)

    if rewards:
        avg_reward = sum(rewards) / len(rewards)
        print(f"\nEpisode over: {len(rewards)} steps, average reward = {avg_reward:.3f}")
    else:
        print("No steps taken.")


if __name__ == "__main__":
    main()
