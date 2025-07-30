import os
import logging
import random
import yaml

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

# ensure environment registration
import comec_gym.env

# set seeds for reproducibility
torch.manual_seed(187)
torch.cuda.manual_seed_all(187)
np.random.seed(187)
random.seed(187)

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def make_env_fn(env_kwargs):
    def _init():
        env = gym.make("CoMEC-v0", **env_kwargs)
        return Monitor(env)
    return _init


def main():
    # load config
    config_path = os.path.join("conf", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # prepare env kwargs
    env_kwargs = {
        "num_edge_servers": config["num_es"],
        "num_clusters": config["num_clusters"],
        "cpu_capacity": float(config["cpu_capacity"]),
        "num_devices_per_cluster": config["num_devices_per_cluster"]
    }

    # training environment
    train_env = DummyVecEnv([make_env_fn(env_kwargs)])
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(
        train_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=1.0
    )

    # evaluation environment (for early stopping)
    eval_env = DummyVecEnv([make_env_fn(env_kwargs)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=1.0
    )

    # select algorithm
    algo = config.get("algorithm", "a2c").lower()
    if algo in ["a2c", "a0c"]:
        model_cls = A2C
    elif algo == "ppo":
        model_cls = PPO
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # tensorboard log dir
    tb_dir = os.path.join("logs", "tensorboard", algo)
    os.makedirs(tb_dir, exist_ok=True)

    # model initialization
    model = model_cls(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        gamma=config.get("discount_factor", 0.95),
        learning_rate=config.get("learning_rate", 5e-5),
        tensorboard_log=tb_dir,
        max_grad_norm=0.5,
        ent_coef=0.01,
        vf_coef=0.5,
    )

    # early stopping callback: stop if no improvement after X evals
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,  # stop if no improvement for 5 evaluations
        min_evals=3,                  # require at least 3 evaluations before stopping
        verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=50_000,             # evaluate every 50k timesteps
        best_model_save_path=os.path.join("models", algo, "best_model"),
        n_eval_episodes=5,
        verbose=1
    )

    total_timesteps = 5e5
    logging.info(f"Training {algo.upper()} for {total_timesteps} timesteps with early stopping...")
    model.learn(
        total_timesteps=int(total_timesteps),
        tb_log_name=algo,
        callback=eval_callback
    )

    # save final model
    save_path = os.path.join("models", algo)
    os.makedirs(save_path, exist_ok=True)
    model_file = os.path.join(save_path, f"{algo}_baseline")
    model.save(model_file)
    logging.info(f"Model saved to {model_file}.zip")

    # save normalization stats
    train_env.save(os.path.join(save_path, "vec_normalize.pkl"))
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
