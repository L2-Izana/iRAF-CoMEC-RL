import csv
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import argparse
import psutil
import random
import time

import torch
from comec_simulator.core.simulator import CoMECSimulator
from iraf_engine.iraf_engine import IraFEngine
from comec_simulator.core.constants import *

# Reproducibility
random.seed(187)
np.random.seed(187)
torch.manual_seed(187)
torch.cuda.manual_seed_all(187)

NUM_ITERATIONS = 15000

# Memory tracking function
import os
def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"[MEMORY] {note} - RSS: {mem:.2f} MB")

# Argument parsing
# RANKING ALGORITHMS: a0c-static-max > a0c-static > a0c-wrong-implementation > a0c-adaptive > mcts-pw-dnn > mcts-pw > mcts-dnn > mcts > a0c-dnn
MODELS = ['mcts', 'mcts-dnn', 'mcts-pw', 'mcts-pw-dnn', 'random', 'greedy', 'a0c', 'a0c-dnn']
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default='a0c')
parser.add_argument("--num_devices", type=int, default=30)
parser.add_argument("--num_tasks", type=int, default=50)
parser.add_argument("--num_es", type=int, default=4)
parser.add_argument("--num_bs", type=int, default=1)
parser.add_argument("--save_empirical_plot", type=bool, default=False)
parser.add_argument("--save_empirical_data", type=bool, default=False)
parser.add_argument("--message", type=str, default="")
args = parser.parse_args()

# Simulation loop
def bulk_run_data_collection(num_runs: int = 20):
    for i in range(num_runs):
        print(f"\n--- Simulation Run {i+1} ---")
        print_memory_usage("Before simulation")

        sim = CoMECSimulator(
            num_devices=args.num_devices,
            num_tasks=args.num_tasks,
            iterations=args.iterations,
            num_es=args.num_es,
            num_bs=args.num_bs
        )
        # sim.install_iraf_engine(IraFEngine(algorithm=args.algorithm))

        metrics = sim.run(residual=True, optimize_for='latency_energy')
        best_action = sim.iraf_engine.get_best_action()
        env_resources_record = sim.run_with_best_action(best_action)
        action_probabilities = sim.iraf_engine.extract_action_probabilities()

        # Save
        os.makedirs("pi_dataset_small", exist_ok=True)
        os.makedirs(f"pi_dataset_small/{i}", exist_ok=True)
        np.save(f"pi_dataset_small/{i}/action_probabilities.npy", action_probabilities)
        np.save(f"pi_dataset_small/{i}/env_resources_record.npy", env_resources_record)

        print_memory_usage("After simulation")

# Main single-run mode
if __name__ == "__main__":
    print_memory_usage("Before creating simulator")
    sim = CoMECSimulator(
        num_devices=args.num_devices,
        num_tasks=args.num_tasks,
        iterations=NUM_ITERATIONS,
        num_es=args.num_es,
        num_bs=args.num_bs,
        algorithm=args.algorithm
    )
    print_memory_usage("After creating simulator")

    metrics = sim.run(residual=True, optimize_for='latency_energy')
    print_memory_usage("After running simulation")
    
    
    # first_depth_children = sim.iraf_engine.a0c.root.children
    # for child in first_depth_children:
    #     print(child)
    sim.metrics.plot_results(saved=True)
    sim.metrics.save_metrics(saved=True, message=args.message)
    # dataset = sim.iraf_engine.get_training_dataset()
    # print(f"Dataset length: {len(dataset)}")

    
    # states = torch.stack([d['state'] for d in dataset])   # shape (N, D)
    # mask   = torch.any(states[:, :5] != 1.0, dim=1)       # shape (N,), True if any of the first 5 ≠ 1
    # count_diff = int(mask.sum().item())

    # print(f"Entries with ≥1 of first 5 dims ≠ 1.0: {count_diff}")    # for data in dataset:
    
    # states = torch.stack([d['state'] for d in dataset])   # shape (N, D)
    # env_states = states[:, :5]                             # first 5 dims
    # means = env_states.mean(dim=1).numpy()                 # per‐sample mean
    # env_resource_use = 1 - means
    # # plot histogram
    # plt.figure()
    # plt.hist(env_resource_use, bins=30)
    # plt.xlabel('Usage Ratio')
    # plt.ylabel('Count')
    # plt.title('Histogram of environment‐state usage fraction')
    # plt.show()    
    print_memory_usage("After saving results")
