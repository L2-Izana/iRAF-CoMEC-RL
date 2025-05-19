import os
import time
import matplotlib.pyplot as plt

class IterationMetrics:
    def __init__(self, total_tasks):
        self.metrics = {
            'completed_tasks': 0,
            'total_latency': 0,
            'total_energy': 0,
            'time_points': [],
            'edge_server_cpu_utilization': [],
            'base_station_bandwidth_utilization': [],
            'energy_per_task': [],
            'latency_per_task': [],
        }
        self.total_tasks = total_tasks
    
    def reset(self):
        self.metrics = {
            'completed_tasks': 0,
            'total_latency': 0,
            'total_energy': 0,
        }       

class MetricsTracker:
    def __init__(self, total_tasks):
        self.metrics = {
            'completed_tasks': 0,
            'total_latency': 0,
            'total_energy': 0,
            'time_points': [],
            'edge_server_cpu_utilization': [],
            'base_station_bandwidth_utilization': [],
            'energy_per_task': [],
            'latency_per_task': [],
        }
        self.total_tasks = total_tasks

    def reset(self):
        self.metrics = {
            'completed_tasks': 0,
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
        self.metrics['completed_tasks'] += 1
        self.metrics['total_latency'] += latency
        self.metrics['total_energy'] += energy
        self.metrics['latency_per_task'].append(latency)
        self.metrics['energy_per_task'].append(energy)

    def get_average_metrics(self):
        c = self.metrics['completed_tasks']
        if c:
            return {
                'avg_latency': self.metrics['total_latency'] / c,
                'avg_energy': self.metrics['total_energy'] / c
            }
        return {'avg_latency': 0, 'avg_energy': 0}

    def plot_results(self, saved=False):
        plt.figure(figsize=(12, 10))
        
        # Task completion plot
        plt.subplot(5, 1, 1)
        plt.bar(['Done', 'Failed'],
                [self.metrics['completed_tasks'], self.total_tasks - self.metrics['completed_tasks']])
        plt.title(f"Task Success Rate: {self.metrics['completed_tasks']}/{self.total_tasks}")
        
        # Latency plot
        avg_latency = self.metrics['total_latency'] / self.metrics['completed_tasks']
        plt.subplot(5, 1, 2)
        plt.plot(self.metrics['latency_per_task'])
        plt.title(f"Latency Total:{self.metrics['total_latency']:.1f}ms\nAvg:{avg_latency:.1f}ms")
        
        # Energy plot
        avg_energy = self.metrics['total_energy'] / self.metrics['completed_tasks']
        plt.subplot(5, 1, 3)
        plt.plot(self.metrics['energy_per_task'])
        plt.title(f"Energy Total:{self.metrics['total_energy']:.3f}J\nAvg:{avg_energy:.3f}J")
        
        # CPU utilization plot
        plt.subplot(5, 1, 4)
        plt.plot(self.metrics['edge_server_cpu_utilization'], label='CPU')
        plt.title('CPU Utilization')
        
        # Bandwidth utilization plot
        plt.subplot(5, 1, 5)
        plt.plot(self.metrics['base_station_bandwidth_utilization'], label='BW')
        plt.title('Bandwidth Utilization')

        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        if not os.path.exists('result_plots'):
            os.makedirs('result_plots')
            
        # Save plot
        if saved:
            plt.savefig(f'result_plots/{time.strftime("%Y%m%d_%H%M%S")}.png')
        else:
            plt.show()
