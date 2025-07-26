import heapq
import itertools
import random
import time
import numpy as np

from ..core.components import BaseStation, Cluster, ClusterManager, EdgeServer, EdgeServerCluster, MobileDevice, Task, TaskRequest
from ..core.constants import *

class CoMECEnvironment:
    """
    A standalone environment for CoMEC simulation:
    - Manages event queue and time progression
    - Handles task generation and state transitions
    - Computes resource allocation, latency, and energy
    """
    def __init__(self, num_edge_servers, num_clusters, cpu_capacity, num_devices_per_cluster):
        # simulation state
        self.time = 0
        self.event_queue = []
        self.counter = itertools.count()

        # build infrastructure
        self.edge_cluster = EdgeServerCluster(num_edge_servers, cpu_capacity)

        self.cluster_manager = ClusterManager(num_clusters, num_devices_per_cluster)
        
    def reset(self, reset_tasks=True):
        """Reset environment to initial state for a new episode"""
        self.time = 0
        self.event_queue.clear()

        # reset resources
        self.cluster_manager.reset_bs_bandwidth()
        self.edge_cluster.reset()

        # schedule task arrivals (keep tasks persistent, no need for residual as the generated tasks in the cluster manager are already persistent)
        assert self.event_queue == [], "Event queue must be empty before resetting"
        self._enqueue_task_requests()


    def step(self, step_args):
        """Advance simulation by processing the next event"""
        if step_args is None:
            return
        func, args = step_args
        assert func.__name__ in ['handle_request', 'handle_completion'], f"Invalid function name: {func}"
        func(*args)

    def pop_event(self):
        if not self.event_queue:
            return None
        t, _, func, args = heapq.heappop(self.event_queue)
        self.time = t
        return {
            'time': t,
            'func': func,
            'args': args,
        }

    def get_resources(self, task_request: TaskRequest) -> dict[str, object]:
        bs = task_request.base_station
        task = task_request.task
        edge_servers = self.edge_cluster.servers
        
        # Get base station and edge server resources
        bs_resource = np.array([bs.available_bandwidth / bs.total_bandwidth])
        es_resource = np.array([es.available_cpu / es.cpu_capacity for es in self.edge_cluster.servers])

        # Get task properties
        task_props = task.get_properties_dnn()  # Expected shape: (4,)
        resources_dnn = np.concatenate([bs_resource, es_resource, task_props])

        resource_dnn_len = 1 + len(self.edge_cluster.servers) + len(task_props)
        assert len(resources_dnn) == resource_dnn_len, \
            f"A vector of state for DNN has to match the length of {resource_dnn_len}, not {len(resources_dnn)}"
        assert np.all(resources_dnn >= 0), \
            f"All elements in resources_dnn must be non-negative, found {resources_dnn[resources_dnn < 0]}"

        return {
            'task': task,
            'bs': bs,
            'edge_servers': edge_servers,
            'resources_dnn': resources_dnn,
        }

        
    def allocate_resources(self, task_request: TaskRequest, alphas):
        """
        Compute resource reservation, latency, and energy for a given task.
        alphas: tuple of (alpha_B, alpha_u2e, alpha_e2ehat, alpha_e, alpha_ehat)
        If None, defaults to random ratios.
        """
        assert alphas is not None, "alphas must not be None"
        assert len(alphas) == 5, "alphas must be a tuple of 5 elements"
        assert all(0 <= alpha <= 1 for alpha in alphas), "alphas must be in the range [0, 1]"
        # unpack alphas, allocation ratios
        if not isinstance(alphas, tuple):
            raise ValueError(f"alphas must be a tuple of 5 elements, not {alphas}")
        alpha_B, alpha_u2e, alpha_e2ehat, alpha_e, alpha_ehat = alphas

        # Validate and get components from task request
        if not isinstance(task_request, TaskRequest):
            raise ValueError("task_request must be an instance of TaskRequest")
        task: Task = task_request.task
        bs: BaseStation = task_request.base_station
        md: MobileDevice = task_request.mobile_device
        
        # 1) bandwidth reservation
        bw_req = alpha_B * bs.available_bandwidth
        if not bs.allocate_bandwidth(bw_req):
            print(f"Failed to allocate bandwidth for task {task.task_id}")
            return None

        # 2) CPU reservation
        primary, collab = self.edge_cluster.get_primary_collab_servers()
        
        p_cpu = alpha_e * primary.available_cpu
        c_cpu = alpha_ehat * collab.available_cpu
        
        if p_cpu <= 0 or not primary.allocate_cpu(p_cpu):
            print(f"Failed to allocate CPU of primary edge server for task {task.task_id}")
            return None
        
        if c_cpu and collab and not collab.allocate_cpu(c_cpu):
            print(f"Failed to allocate CPU of collab edge server for task {task.task_id}")
            return None

        # 3) compute latency + energy
        t_local = task.cpu_cycles * (1 - alpha_e2ehat) / task.device_cpu_freq
        assert t_local >= 0, f"Local latency must be non-negative, not {t_local}"
        E_local = CHIP_COEFFICIENT * (task.device_cpu_freq ** 2) * task.cpu_cycles * (1 - alpha_e2ehat)
        assert E_local >= 0, f"Local energy must be non-negative, not {E_local}"

        snr = max(md.transmit_power * task.channel_gain / CHANNEL_NOISE_VARIANCE, 1e-8)
        rate = alpha_B * np.log2(1 + snr)
        assert rate > 0, f"Rate must be positive, not {rate}"
        t_tx = task.data_size * alpha_u2e / rate
        assert t_tx >= 0, f"Transmission time must be non-negative, not {t_tx}"
        E_tx = md.transmit_power * t_tx
        assert E_tx >= 0, f"Transmission energy must be non-negative, not {E_tx}"
        
        if c_cpu == 0:
            print(f"Zero CPU allocated to collaborative server for task {task.task_id}")
            return None
        t_edge = task.cpu_cycles * alpha_u2e * (1 - alpha_e2ehat) / p_cpu
        assert t_edge >= 0, f"Edge processing time must be non-negative, not {t_edge}"
        t_collab = (task.cpu_cycles * alpha_u2e * alpha_e2ehat / c_cpu) if collab else 0
        assert t_collab >= 0, f"Collaboration processing time must be non-negative, not {t_collab}"
        
        total_latency = max(t_local, t_tx, t_edge, t_tx + t_collab)
        total_energy = E_local + E_tx
        assert total_latency > 0, f"Total latency must be positive, not {total_latency}"
        assert total_energy > 0, f"Total energy must be positive, not {total_energy}"
        
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
    
    def get_edge_servers(self):
        """Get all edge servers in the environment."""
        return self.edge_cluster.servers
    def get_base_stations(self):
        """Get all base stations in the environment."""
        return self.cluster_manager.get_base_stations()
    def get_num_tasks(self):
        """Get the number of tasks currently in the environment."""
        return len(self.cluster_manager.task_requests)
    def _enqueue_task_requests(self):
        """Enqueue task requests onto the event queue based on their arrival times. """

        for task_request in self.cluster_manager.task_requests:
            self._enqueue(task_request.arrival_time, self.handle_request, task_request)            
    
    def _enqueue(self, t, func, *args):
        heapq.heappush(self.event_queue, (t, next(self.counter), func, args))
    
    def handle_request(self, task_request: TaskRequest, alphas):
        alloc = self.allocate_resources(task_request, alphas)
        if not alloc:
            raise RuntimeError(f"Failed to allocate resources for task {task_request.task.task_id}")

        # schedule completion
        self._enqueue(self.time + alloc['total_latency'], self.handle_completion, alloc)

    def handle_completion(self, alloc):
        alloc['bs'].release_bandwidth(alloc['bw_req'])
        alloc['primary'].release_cpu(alloc['p_cpu'])
        if alloc['collab']:
            alloc['collab'].release_cpu(alloc['c_cpu'])
        task = alloc['task']
        