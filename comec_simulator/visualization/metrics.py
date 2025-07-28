import os
import time
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

EMPERICAL_RUN_FOLDER = "empirical_runs"

class MetricsTracker:
    def __init__(self, total_tasks, algorithm):
        self.metrics = {
            'total_latency': 0,
            'total_energy': 0,
            'time_points': [],
            'edge_server_cpu_utilization': [],
            'base_station_bandwidth_utilization': [],
            'energy_per_task': [],
            'latency_per_task': [],
        }
        self.total_tasks = total_tasks
        self.node_counts = []
        self.rewards = []
        
    def reset(self):
        self.metrics = {
            'total_latency': 0,
            'total_energy': 0,
            'time_points': [],
            'edge_server_cpu_utilization': [],
            'base_station_bandwidth_utilization': [],
            'energy_per_task': [],
            'latency_per_task': [],
        }
    
    def record_metrics(self, time, edge_servers, base_stations):
        cpu_u = 1 - sum(s.available_cpu for s in edge_servers) / sum(s.cpu_capacity for s in edge_servers)
        bw_u = 1 - sum(bs.available_bandwidth for bs in base_stations) / sum(bs.total_bandwidth for bs in base_stations)
        
        self.metrics['time_points'].append(time)
        self.metrics['edge_server_cpu_utilization'].append(cpu_u)
        self.metrics['base_station_bandwidth_utilization'].append(bw_u)

    def record_task_completion(self, latency, energy):
        self.metrics['total_latency'] += latency
        self.metrics['total_energy'] += energy
        self.metrics['latency_per_task'].append(latency)
        self.metrics['energy_per_task'].append(energy)

    def get_average_metrics(self):
        return {
            'avg_latency': self.metrics['total_latency'] / self.total_tasks,
            'avg_energy': self.metrics['total_energy'] / self.total_tasks
        }

    def record_tree_iteration_step_attributes(self, node_count, reward):
        assert node_count >= 0, f"Node count should be non-negative, got {node_count}"
        assert reward >= 0, f"Reward should be non-negative, got {reward}"
        self.node_counts.append(node_count)
        self.rewards.append(reward)

    def plot_tree_iteration_step_attributes(self, saved=True):
        plt.figure(figsize=(12, 10))

        # Plot node count (top subplot)
        plt.subplot(2, 1, 1)
        plt.plot(self.node_counts)
        plt.title('Node Count')
        plt.xlabel('Iteration')
        plt.ylabel('Node Count')

        # Plot smoothed reward with shaded std (bottom subplot)
        plt.subplot(2, 1, 2)
        rewards = np.array(self.rewards)
        window = 100
        
        assert np.all(rewards > 0), "Rewards should be positive, at indexes: " + \
            ', '.join(str(i) for i, r in enumerate(rewards) if r <= 0)
        
        if len(rewards) >= window:
            sma = np.convolve(rewards, np.ones(window)/window, mode='valid')
            std = np.array([np.std(rewards[i-window:i]) for i in range(window, len(rewards) + 1)])
            x = np.arange(len(sma))
            plt.plot(x, sma, label='Moving Average Reward')
            plt.fill_between(x, sma - std, sma + std, alpha=0.3, label='±1 Std. Dev')
        else:
            # Fallback if too few points
            plt.plot(rewards, label='Reward (Raw)')

        plt.title('Reward')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        if saved:
            plt.savefig(f"tree_iteration_step_attributes.png")
        else:
            plt.show()
        
    def plot_metrics(self, saved=False):
        plt.figure(figsize=(12, 10))
        
        # Task completion plot
        plt.subplot(4, 1, 1)
        
        # Latency plot
        avg_latency = self.metrics['total_latency'] / self.total_tasks
        plt.subplot(4, 1, 1)
        plt.plot(self.metrics['latency_per_task'])
        plt.title(f"Latency Total:{self.metrics['total_latency']:.1f}ms\nAvg:{avg_latency:.1f}ms")
        
        # Energy plot
        avg_energy = self.metrics['total_energy'] / self.total_tasks
        plt.subplot(4, 1, 2)
        plt.plot(self.metrics['energy_per_task'])
        plt.title(f"Energy Total:{self.metrics['total_energy']:.3f}J\nAvg:{avg_energy:.3f}J")
        
        # CPU utilization plot
        plt.subplot(4, 1, 3)
        plt.plot(self.metrics['edge_server_cpu_utilization'], label='CPU')
        plt.title('CPU Utilization')
        
        # Bandwidth utilization plot
        plt.subplot(4, 1, 4)
        plt.plot(self.metrics['base_station_bandwidth_utilization'], label='BW')
        plt.title('Bandwidth Utilization')

        plt.tight_layout()
        
        if saved:
            plt.savefig('metrics.png')
        else:
            plt.show()
        
    def plot_results(self, saved=False):        
        self.plot_metrics(saved)
        self.plot_tree_iteration_step_attributes(saved)

    def save_metrics(self, saved=False, message=None, config=None):
        if saved:
            for key, value in self.metrics.items():
                np.save(f"{key}.npy", value)
            np.save(f"node_counts.npy", self.node_counts)
            np.save(f"rewards.npy", self.rewards)

            if message:
                with open(f"message.txt", 'w') as f:
                    f.write(message)