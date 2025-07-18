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

# Memory tracking function
import os
def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"[MEMORY] {note} - RSS: {mem:.2f} MB")

# Argument parsing
# RANKING ALGORITHMS: a0c-mod > a0c-static-max (proportionally increasing) > a0c-static > a0c-wrong-implementation > a0c-adaptive > mcts-pw-dnn > mcts-pw > mcts-dnn > mcts > a0c-dnn
MODELS = ['mcts', 'mcts-dnn', 'mcts-pw', 'mcts-pw-dnn', 'random', 'greedy', 'a0c', 'a0c-dnn']
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default='a0c')
parser.add_argument("--message", type=str, default="")
parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
args = parser.parse_args()

# Simulation loop
def bulk_run_data_collection(num_runs: int = 20):
    for i in range(num_runs):
        print(f"\n--- Simulation Run {i+1} ---")
        print_memory_usage("Before simulation")

        sim = CoMECSimulator(iterations=args.iterations, algorithm=args.algorithm)

        metrics = sim.run(optimize_for='latency_energy')
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
        iterations=NUM_ITERATIONS,
        algorithm=args.algorithm
    )
    print_memory_usage("After creating simulator")

    if args.mode == "train":
        metrics = sim.run(optimize_for='latency_energy')
        print_memory_usage("After running simulation")
        
        sim.metrics.plot_results(saved=False)
        sim.metrics.save_metrics(saved=True, message=args.message)
        print_memory_usage("After saving results")
    else:
        metrics = sim.eval(optimize_for='latency_energy')
        # sim.metrics.plot_results(saved=True)
        # sim.metrics.save_metrics(saved=True, message=args.message)
        print(metrics)
        # print(metrics)