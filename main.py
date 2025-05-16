import numpy as np
from comec_simulator import CoMECSimulator

class ToyIraFEngine:
    def get_ratios(self, env_resources):
        return np.random.rand(5)

# Create and run simulation
sim = CoMECSimulator(num_devices=50, num_tasks=2000)
sim.install_iraf_engine(ToyIraFEngine())
metrics = sim.run()
# sim.metrics.plot_results()