import logging
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import random
import psutil
import os

from comec_simulator.core.simulator import CoMECSimulator


# Reproducibility
random.seed(187)
np.random.seed(187)
torch.manual_seed(187)
torch.cuda.manual_seed_all(187)


def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"[MEMORY] {note} - RSS: {mem:.2f} MB")


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    log = logging.getLogger("run_sim")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log.info("Starting simulator in %s mode with algorithm: %s", cfg.mode, cfg.algorithm)

    print_memory_usage("Before creating simulator")
    sim = CoMECSimulator(
        iterations=cfg.num_iter,
        algorithm=cfg.algorithm,
        num_edge_servers=cfg.num_es,
        num_clusters=cfg.num_clusters,
        cpu_capacity=cfg.cpu_capacity,
        num_devices_per_cluster=cfg.num_devices_per_cluster,
        cfg=cfg
    )
    print_memory_usage("After creating simulator")

    if cfg.mode == "train":
        metrics = sim.run(optimize_for="latency_energy")
        print_memory_usage("After running simulation")
        sim.metrics.plot_results(saved=True)
        sim.metrics.save_metrics(saved=True, message=cfg.message, config=cfg)
        print_memory_usage("After saving results")
    else:
        metrics = sim.eval(optimize_for="latency_energy")
        print(metrics)


if __name__ == "__main__":
    main()
