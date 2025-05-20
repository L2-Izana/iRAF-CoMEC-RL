import csv
import json
import os
import numpy as np
from comec_simulator.core.simulator import CoMECSimulator
from iraf_engine.iraf_engine import IraFEngine
from comec_simulator.core.constants import *
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

for i in range(0,20):
    sim = CoMECSimulator(num_devices=NUM_DEVICES, num_tasks=NUM_TASKS, iterations=ITERATIONS)
    sim.install_iraf_engine(IraFEngine(algorithm='mcts'))
    metrics = sim.run(residual=True, optimize_for='latency_energy')
    best_action = sim.iraf_engine.get_best_action()
    env_resources_record = sim.run_with_best_action(best_action)
    action_probabilities = sim.iraf_engine.extract_action_probabilities()

    # Save
    os.makedirs(f"pi_dataset_small", exist_ok=True)
    os.makedirs(f"pi_dataset_small/{i}", exist_ok=True)
    np.save(f"pi_dataset_small/{i}/action_probabilities.npy", action_probabilities)
    np.save(f"pi_dataset_small/{i}/env_resources_record.npy", env_resources_record)
