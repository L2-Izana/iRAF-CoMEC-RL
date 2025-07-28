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

    sim = CoMECSimulator(
        iterations=cfg.num_iter,
        algorithm=cfg.algorithm,
        optimize_for="latency",
        num_edge_servers=cfg.num_es,
        num_clusters=cfg.num_clusters,
        cpu_capacity=cfg.cpu_capacity,
        num_devices_per_cluster=cfg.num_devices_per_cluster,
        cfg=cfg
    )
    print_memory_usage("After creating simulator")

    if cfg.mode == "train":
        metrics = sim.run()
        print_memory_usage("After running simulation")
        sim.metrics.plot_results(saved=True)
        sim.metrics.save_metrics(saved=True, message=cfg.message, config=cfg)
        best_action = sim.iraf_engine.get_best_action()
        if best_action is not None:
            np.save("best_action.npy", best_action)
    else:
        metrics = sim.eval()
        print(metrics)


if __name__ == "__main__":
    main()
