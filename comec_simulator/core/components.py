import numpy as np
import random

# Constants
EDGE_SERVER_CPU_CAPACITY = 19.14*1e9  # cycles
BANDWIDTH_PER_BS = 40  # MHz

class Task:
    def __init__(self, device, bs, data_size, cpu_cycles, device_cpu_freq, channel_gain, max_latency, arrival_time):
        self.device = device
        self.bs = bs
        self.data_size = data_size
        self.cpu_cycles = cpu_cycles
        self.device_cpu_freq = device_cpu_freq
        self.channel_gain = channel_gain
        self.max_latency = max_latency
        self.arrival_time = arrival_time

class MobileDevice:
    _counter = 0

    def __init__(self, base_station):
        self.device_id = MobileDevice._counter
        self.device_cpu_freq = 3e8  # Hz
        self.base_station = base_station
        self.transmit_power = random.uniform(32, 197)  # mW
        MobileDevice._counter += 1

    def generate_task(self, arrival_time):
        data_size = random.uniform(0.2, 3)  # MB 
        cpu_cycles = random.uniform(6e9, 9e10)  # cycles 
        channel_gain = random.uniform(0.5, 1.0)
        max_latency = 2000  # ms
        return Task(self, self.base_station, data_size,
                   cpu_cycles, self.device_cpu_freq,
                   channel_gain, max_latency, arrival_time)

class EdgeServer:
    _counter = 0
    
    def __init__(self, cpu_capacity=EDGE_SERVER_CPU_CAPACITY):
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

class BaseStation:
    _counter = 0
    
    def __init__(self, edge_servers=None):
        self.bs_id = BaseStation._counter
        BaseStation._counter += 1
        self.total_bandwidth = BANDWIDTH_PER_BS
        self.available_bandwidth = BANDWIDTH_PER_BS
        self.edge_servers = edge_servers or []

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