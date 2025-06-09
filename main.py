import csv
import json
import os
import numpy as np
import argparse
from comec_simulator.core.simulator import CoMECSimulator
from iraf_engine.iraf_engine import IraFEngine
from comec_simulator.core.constants import *
import random

random.seed(187)
np.random.seed(187)
# report_cases = [
#     {
#         'name': 'small',
#         'num_devices': 10,
#         'num_tasks': 20,
#         'iterations': 10000
#     },
#     {
#         'name': 'medium',
#         'num_devices': 20,
#         'num_tasks': 50,
#         'iterations': 15000
#     },
#     {
#         'name': 'large',
#         'num_devices': 30,
#         'num_tasks': 100,
#         'iterations': 20000
#     }
# ]

# for case in report_cases:
#     # Create and run simulation
#     sim = CoMECSimulator(num_devices=case['num_devices'], num_tasks=case['num_tasks'], iterations=case['iterations'])
#     sim.install_iraf_engine(IraFEngine(algorithm='mcts'))
#     metrics = sim.run(residual=True, optimize_for='latency_energy')
#     sim.metrics.plot_results(saved=True)

MODELS = ['mcts', 'mcts-dnn', 'mcts-pw', 'mcts-pw-dnn', 'random', 'greedy']
parser = argparse.ArgumentParser()
parser.add_argument("--algorithm", type=str, default='mcts-pw')
parser.add_argument("--num_devices", type=int, default=5)
parser.add_argument("--num_tasks", type=int, default=20)
parser.add_argument("--iterations", type=int, default=10000)
parser.add_argument("--num_es", type=int, default=4)
parser.add_argument("--num_bs", type=int, default=1)
parser.add_argument("--save_empirical_plot", type=bool, default=False)
parser.add_argument("--save_empirical_data", type=bool, default=False)
args = parser.parse_args()


def bulk_run_data_collection(num_runs: int = 20):
    for i in range(0,20):
        sim = CoMECSimulator(num_devices=args.num_devices, num_tasks=args.num_tasks, iterations=args.iterations, num_es=args.num_es, num_bs=args.num_bs)
        sim.install_iraf_engine(IraFEngine(algorithm=args.algorithm))
        metrics = sim.run(residual=True, optimize_for='latency_energy')
        best_action = sim.iraf_engine.get_best_action()
        env_resources_record = sim.run_with_best_action(best_action)
        action_probabilities = sim.iraf_engine.extract_action_probabilities()

        # Save
        os.makedirs(f"pi_dataset_small", exist_ok=True)
        os.makedirs(f"pi_dataset_small/{i}", exist_ok=True)
        np.save(f"pi_dataset_small/{i}/action_probabilities.npy", action_probabilities)
        np.save(f"pi_dataset_small/{i}/env_resources_record.npy", env_resources_record)

if __name__ == "__main__":
    sim = CoMECSimulator(num_devices=args.num_devices, num_tasks=args.num_tasks, iterations=args.iterations, num_es=args.num_es, num_bs=args.num_bs, algorithm=args.algorithm)
    metrics = sim.run(residual=True, optimize_for='latency_energy')
    sim.metrics.plot_results(saved=True)
    sim.metrics.save_metrics(saved=True)
