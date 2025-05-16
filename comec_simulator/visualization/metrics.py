import os
import time
import matplotlib.pyplot as plt

class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_latency': 0,
            'total_energy': 0,
            'time_points': [],
            'edge_server_cpu_utilization': [],
            'base_station_bandwidth_utilization': [],
            'energy_per_task': [],
            'latency_per_task': [],
        }

    def reset(self):
        self.metrics = {
            'completed_tasks': 0,
            'failed_tasks': 0,
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

    def record_task_failure(self):
        self.metrics['failed_tasks'] += 1

    def get_average_metrics(self):
        c = self.metrics['completed_tasks']
        if c:
            return {
                'avg_latency': self.metrics['total_latency'] / c,
                'avg_energy': self.metrics['total_energy'] / c
            }
        return {'avg_latency': 0, 'avg_energy': 0}

    def plot_results(self, algo, num_devices, num_tasks, need_duration):
        plt.figure(figsize=(12, 10))
        
        # Task completion plot
        plt.subplot(5, 1, 1)
        plt.bar(['Done', 'Fail'],
                [self.metrics['completed_tasks'], self.metrics['failed_tasks']])
        plt.title(f"Task Success Rate: {self.metrics['completed_tasks']}/{self.metrics['failed_tasks']}")
        
        # Latency plot
        plt.subplot(5, 1, 2)
        plt.plot(self.metrics['latency_per_task'])
        plt.title(f"Latency Total:{self.metrics['total_latency']:.1f}ms")
        
        # Energy plot
        plt.subplot(5, 1, 3)
        plt.plot(self.metrics['energy_per_task'])
        plt.title(f"Energy Total:{self.metrics['total_energy']:.3f}J")
        
        # CPU utilization plot
        plt.subplot(5, 1, 4)
        plt.plot(self.metrics['edge_server_cpu_utilization'], label='CPU')
        
        # Bandwidth utilization plot
        plt.subplot(5, 1, 5)
        plt.plot(self.metrics['base_station_bandwidth_utilization'], label='BW')
        plt.legend()
        plt.title('Utilization')

        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        if not os.path.exists('result_plots'):
            os.makedirs('result_plots')
            
        # Save plot
        plt.savefig(f'result_plots/{algo}_results_{num_devices}_{num_tasks}_{need_duration}_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.close() 