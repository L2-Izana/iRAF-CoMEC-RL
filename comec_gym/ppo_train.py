# ppo_train.py

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# ensure our env is registered
import env  # noqa

def make_env():
    # wrap with Monitor so SB3 logs episode stats
    return Monitor(gym.make("CoMEC-v0"))

if __name__ == "__main__":
    log_dir = "logs/ppo_comec"
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # stop when mean reward > -50
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=-50, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,
        eval_freq=5_000,
        n_eval_episodes=5,
        best_model_save_path=log_dir,
        verbose=1,
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        batch_size=64,
        n_steps=1024,
        learning_rate=3e-4,
    )

    model.learn(total_timesteps=200_000, callback=eval_cb, tb_log_name="ppo_comec")
    model.save(os.path.join(log_dir, "final_model"))

    print("Training complete; logs & models saved in", log_dir)
