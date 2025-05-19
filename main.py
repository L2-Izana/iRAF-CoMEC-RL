import numpy as np
from comec_simulator.core.simulator import CoMECSimulator
from iraf_engine.iraf_engine import IraFEngine
# Create and run simulation
sim = CoMECSimulator(num_devices=20, num_tasks=100, iterations=2000)
sim.install_iraf_engine(IraFEngine(algorithm='mcts'))
metrics = sim.run(residual=True)
# sim.metrics.plot_results(saved=False)
