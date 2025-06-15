import csv
import json
import os
import numpy as np
import argparse
import psutil
import random
import time
from comec_simulator.core.simulator import CoMECSimulator
from iraf_engine.iraf_engine import IraFEngine
from comec_simulator.core.constants import *

# Reproducibility
random.seed(187)
np.random.seed(187)

# Memory tracking function
import os
def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"[MEMORY] {note} - RSS: {mem:.2f} MB")

# Argument parsing
MODELS = ['mcts', 'mcts-dnn', 'mcts-pw', 'mcts-pw-dnn', 'random', 'greedy', 'a0c']
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default='mcts-pw-dnn')
parser.add_argument("--num_devices", type=int, default=25)
parser.add_argument("--num_tasks", type=int, default=50)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--num_es", type=int, default=4)
parser.add_argument("--num_bs", type=int, default=1)
parser.add_argument("--save_empirical_plot", type=bool, default=False)
parser.add_argument("--save_empirical_data", type=bool, default=False)
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
        sim.install_iraf_engine(IraFEngine(algorithm=args.algorithm))

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
        iterations=args.iterations,
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
    sim.metrics.save_metrics(saved=True)
    print_memory_usage("After saving results")
