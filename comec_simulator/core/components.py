from typing import List
import numpy as np
import random

from comec_simulator.core.constants import *

class Task:
    _counter = 0
    def __init__(self, data_size, cpu_cycles, device_cpu_freq, channel_gain, max_latency, arrival_time):
        self.task_id = Task._counter
        Task._counter += 1
        self.data_size = data_size
        self.cpu_cycles = cpu_cycles
        self.device_cpu_freq = device_cpu_freq
        self.channel_gain = channel_gain
        self.max_latency = max_latency
        self.arrival_time = arrival_time

    def get_properties_dnn(self):
        return np.array([self.data_size / MAX_DATA_SIZE, self.cpu_cycles / MAX_CPU_CYCLES, self.device_cpu_freq / DEVICE_CPU_FREQ, self.channel_gain / MAX_CHANNEL_GAIN])

class MobileDevice:
    _counter = 0

    def __init__(self, base_station):
        self.device_id = MobileDevice._counter
        self.device_cpu_freq = DEVICE_CPU_FREQ  # Hz
        self.base_station = base_station
        self.transmit_power = random.uniform(MIN_TRANSMIT_POWER, MAX_TRANSMIT_POWER)  # mW
        MobileDevice._counter += 1

    def generate_task(self, arrival_time):
        data_size = random.uniform(MIN_DATA_SIZE, MAX_DATA_SIZE)  # MB 
        cpu_cycles = random.uniform(MIN_CPU_CYCLES, MAX_CPU_CYCLES)  # cycles 
        channel_gain = random.uniform(MIN_CHANNEL_GAIN, MAX_CHANNEL_GAIN)
        max_latency = DEFAULT_MAX_LATENCY  # ms
        return Task(data_size,
                   cpu_cycles, self.device_cpu_freq,
                   channel_gain, max_latency, arrival_time)

class BaseStation:
    _counter = 0
    
    def __init__(self):
        self.bs_id = BaseStation._counter
        BaseStation._counter += 1
        self.total_bandwidth = BANDWIDTH_PER_BS
        self.available_bandwidth = BANDWIDTH_PER_BS

    def allocate_bandwidth(self, amount):
        if 0 < amount <= self.available_bandwidth:
            self.available_bandwidth -= amount
            return True
        return False
    
    def release_bandwidth(self, amount):
        self.available_bandwidth = min(
            self.total_bandwidth,
            self.available_bandwidth + amount)
    
    def reset(self):
        self.available_bandwidth = self.total_bandwidth

    def __str__(self):
        return f"BaseStation(bs_id={self.bs_id}, total_bandwidth={self.total_bandwidth}MHz, available_bandwidth={self.available_bandwidth}MHz)" 
    
class EdgeServer:
    _counter = 0
    
    def __init__(self, cpu_capacity):
        self.server_id = EdgeServer._counter
        EdgeServer._counter += 1
        self.cpu_capacity = cpu_capacity
        self.available_cpu = cpu_capacity

    def allocate_cpu(self, amount):
        if 0 < amount <= self.available_cpu:
            self.available_cpu -= amount
            return True
        return False

    def release_cpu(self, amount):
        self.available_cpu = min(self.cpu_capacity,
                               self.available_cpu + amount)

    def reset(self):
        self.available_cpu = self.cpu_capacity

    def __str__(self):
        return f"EdgeServer(server_id={self.server_id}, cpu_capacity={self.cpu_capacity}, available_cpu={self.available_cpu})"

class TaskRequest:
    def __init__(self, task, mobile_device, base_station, arrival_time):
        self.task = task
        self.mobile_device = mobile_device
        self.base_station = base_station
        self.arrival_time = arrival_time
        
class Cluster:
    _counter = 0
    
    def __init__(self, num_devices):
        self.cluster_id = Cluster._counter
        Cluster._counter += 1
        self.base_station = BaseStation()
        self.mobile_devices = self._init_mobile_devices(self.base_station, num_devices)
        self.num_devices = len(self.mobile_devices)

    def reset_bs_bandwidth(self):
        self.base_station.reset()
    
    def _init_mobile_devices(self, base_station, num_devices):
        return [MobileDevice(base_station) for _ in range(num_devices)]
    
    def __str__(self):
        return f"Cluster(cluster_id={self.cluster_id}, base_station={self.base_station}, num_devices={len(self.mobile_devices)})"
    
class ClusterManager:
    def __init__(self, num_clusters, num_devices_per_cluster):
        self.clusters: List[Cluster] = [Cluster(num_devices_per_cluster) for _ in range(num_clusters)]
        self.task_requests: List[TaskRequest] = self._generate_tasks()
        
    def reset_bs_bandwidth(self):
        for cluster in self.clusters:
            cluster.reset_bs_bandwidth()
    
    def get_base_stations(self) -> List[BaseStation]:
        return [cluster.base_station for cluster in self.clusters]
    
    def _generate_tasks(self):
        task_requests: List[TaskRequest] = []
        for cluster in self.clusters:
            for device in cluster.mobile_devices:
                arrival_time: float = random.uniform(0, ARRIVAL_WINDOW)
                task: Task = device.generate_task(arrival_time)
                task_request: TaskRequest = TaskRequest(task=task, mobile_device=device, base_station=cluster.base_station, arrival_time=arrival_time)
                task_requests.append(task_request)
        return task_requests
    
class EdgeServerCluster:
    
    def __init__(self, num_edge_servers, cpu_capacity):
        self.servers: List[EdgeServer] = [EdgeServer(cpu_capacity) for _ in range(num_edge_servers)]
        
    def reset(self):
        for server in self.servers:
            server.reset()
            
    def get_primary_collab_servers(self) -> tuple[EdgeServer, EdgeServer]:
        """Get the primary (the strongest) and collaborative (the 2nd strongest) edge servers based on available CPU."""
        primary: EdgeServer = max(self.servers, key=lambda s: s.available_cpu)
        others: List[EdgeServer] = [s for s in self.servers if s is not primary]
        collab: EdgeServer = max(others, key=lambda s: s.available_cpu) 
        assert primary is not None and collab is not None, "Primary and Collaborative server cannot be None"
        return primary, collab