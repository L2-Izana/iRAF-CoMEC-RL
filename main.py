import numpy as np
from comec_simulator.core.simulator import CoMECSimulator
from iraf_engine.iraf_engine import IraFEngine
# Create and run simulation
sim = CoMECSimulator(num_devices=10, num_tasks=20, iterations=10000)
sim.install_iraf_engine(IraFEngine(algorithm='mcts'))
metrics = sim.run(residual=True, optimize_for='energy')
sim.metrics.plot_results(saved=False)
