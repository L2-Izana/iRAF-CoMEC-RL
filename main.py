import numpy as np
from comec_simulator.core.simulator import CoMECSimulator
from iraf_engine.iraf_engine import IraFEngine

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

sim = CoMECSimulator(num_devices=10, num_tasks=20, iterations=100)
sim.install_iraf_engine(IraFEngine(algorithm='mcts'))
metrics = sim.run(residual=True, optimize_for='latency_energy')
best_action = sim.iraf_engine.get_best_action()
print(best_action)
sim.metrics.plot_results(saved=True)
