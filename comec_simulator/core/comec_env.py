import heapq
import itertools
import random
import numpy as np

from ..core.components import BaseStation, EdgeServer, MobileDevice, Task
from ..core.constants import CHANNEL_NOISE_VARIANCE, CHIP_COEFFICIENT

class CoMECEnvironment:
    """
    A standalone environment for CoMEC simulation:
    - Manages event queue and time progression
    - Handles task generation and state transitions
    - Computes resource allocation, latency, and energy
    """
    def __init__(self, num_devices=20, num_tasks=50, arrival_window=10000,
                 num_edge_servers=12, num_bs=1, retry_interval=10):
        self.num_devices = num_devices
        self.num_tasks = num_tasks
        self.arrival_window = arrival_window
        self.retry_interval = retry_interval

        # simulation state
        self.time = 0
        self.event_queue = []
        self.counter = itertools.count()

        # build infrastructure
        self.edge_servers = [EdgeServer() for _ in range(num_edge_servers)]
        self.base_stations = self._build_base_stations(num_bs)
        self.mobile_devices = [MobileDevice(base_station=self.base_stations[0])
                               for _ in range(num_devices)]

    def _build_base_stations(self, num_bs):
        per_bs = max(1, len(self.edge_servers) // num_bs)
        bs_list = []
        for i in range(num_bs):
            start = i * per_bs
            end = start + per_bs if i < num_bs-1 else len(self.edge_servers)
            bs_list.append(BaseStation(self.edge_servers[start:end]))
        return bs_list

    def reset(self):
        """Reset environment to initial state for a new episode"""
        self.time = 0
        self.event_queue.clear()
        self.generated_tasks = []
        # reset resources
        for bs in self.base_stations:
            bs.reset()
        for es in self.edge_servers:
            es.reset()
        # schedule task arrivals
        self._schedule_arrivals()

    def _schedule_arrivals(self):
        for _ in range(self.num_tasks):
            dev = random.choice(self.mobile_devices)
            arrival = random.uniform(0, self.arrival_window)
            task = dev.generate_task(arrival)
            self._enqueue(arrival, '_handle_request', task)

    def _enqueue(self, t, func_name, *args):
        heapq.heappush(self.event_queue, (t, next(self.counter), func_name, args))

    def step(self, step_args=None):
        """Advance simulation by processing the next event"""
        func_name, args = step_args
        func = getattr(self, func_name)
        func(*args)

    def pop_event(self):
        if not self.event_queue:
            return None
        t, _, func_name, args = heapq.heappop(self.event_queue)
        self.time = t
        return {
            'time': t,
            'func_name': func_name,
            'args': args,
        }

    def get_resources(self, task):
        return {
            'bs': task.bs,
            'edge_servers': self.edge_servers,
            'task': task,
        }

    def _handle_request(self, task, alphas):
        alloc = self.allocate_resources(task, alphas)
        if not alloc:
            self._enqueue(self.time + self.retry_interval, '_handle_request', task)
            return
        # schedule completion
        self._enqueue(self.time + alloc['total_latency'], '_handle_completion', alloc)

    def _handle_completion(self, alloc):
        alloc['bs'].release_bandwidth(alloc['bw_req'])
        alloc['primary'].release_cpu(alloc['p_cpu'])
        if alloc['collab']:
            alloc['collab'].release_cpu(alloc['c_cpu'])
        # user can collect latency and energy here

    def allocate_resources(self, task, alphas):
        """
        Compute resource reservation, latency, and energy for a given task.
        alphas: tuple of (alpha_B, alpha_u2e, alpha_e2ehat, alpha_e, alpha_ehat)
        If None, defaults to random ratios.
        """
        assert alphas is not None, "alphas must not be None"
        alpha_B, alpha_u2e, alpha_e2ehat, alpha_e, alpha_ehat = alphas

        # 1) bandwidth reservation
        bs = task.bs
        bw_req = bs.total_bandwidth * alpha_B
        if not bs.allocate_bandwidth(bw_req):
            return None

        # 2) CPU reservation
        primary = max(bs.edge_servers, key=lambda s: s.available_cpu)
        others = [s for s in self.edge_servers if s is not primary]
        collab = max(others, key=lambda s: s.available_cpu) if others else None

        p_cpu = primary.cpu_capacity * alpha_e
        c_cpu = collab.cpu_capacity * alpha_ehat if collab else 0
        if p_cpu <= 0 or not primary.allocate_cpu(p_cpu):
            bs.release_bandwidth(bw_req)
            return None
        if c_cpu and collab and not collab.allocate_cpu(c_cpu):
            primary.release_cpu(p_cpu)
            bs.release_bandwidth(bw_req)
            return None

        # 3) compute latencies
        t_local = task.cpu_cycles * (1 - alpha_e2ehat) / task.device_cpu_freq
        E_local = CHIP_COEFFICIENT * (task.device_cpu_freq ** 2) * task.cpu_cycles * (1 - alpha_e2ehat)

        rate = alpha_B * np.log2(1 + task.device.transmit_power * task.channel_gain / CHANNEL_NOISE_VARIANCE)
        t_tx = task.data_size * alpha_u2e / rate
        E_tx = task.device.transmit_power * t_tx

        t_edge = task.cpu_cycles * alpha_u2e * (1 - alpha_e2ehat) / p_cpu
        t_collab = (task.cpu_cycles * alpha_u2e * alpha_e2ehat / c_cpu) if collab else 0

        total_latency = max(t_local, t_tx, t_edge, t_tx + t_collab)
        total_energy = E_local + E_tx

        return {
            'task': task,
            'bs': bs,
            'primary': primary,
            'collab': collab,
            'bw_req': bw_req,
            'p_cpu': p_cpu,
            'c_cpu': c_cpu,
            'total_latency': total_latency,
            'total_energy': total_energy,
        }

    def run(self, max_time=None):
        """Run until no events left or until max_time is reached"""
        while self.event_queue:
            if max_time and self.event_queue[0][0] > max_time:
                break
            self.step()
        # return final statistics collected externally
