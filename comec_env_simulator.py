# ----------------------------------------------------------------------------
# Simulator Core
# ----------------------------------------------------------------------------
import heapq
import itertools
import random
import time
import os
import matplotlib.pyplot as plt
import numpy as np

from comec_components import EdgeServer, BaseStation, MobileDevice

# CPU_EFFICIENCY = 1.0
# ENERGY_PER_CYCLE_J = 1.0
# TX_POWER_W = 1.0
CHIP_COEFFICIENT = 1e-26
CHANNEL_NOICE_VARIANCE = 1e-5

class CoMECEnvSimulator:
    def __init__(self, algo='random', need_monitoring=True, need_duration=True, num_devices=20, num_tasks=50, arrival_window=10000, retry_interval=10, num_edge_servers=12, num_bs=1):
        self.time = 0
        self.event_queue = []
        self.counter = itertools.count()
        self.total_tasks = 0
        self.algo = algo
        self.iraf_engine = None
        self.need_monitoring = need_monitoring
        self.need_duration = need_duration
        self.num_devices = num_devices
        self.num_tasks = num_tasks
        self.arrival_window = arrival_window
        self.edge_servers = [EdgeServer() for _ in range(num_edge_servers)]
        self.last_monitor_time = 0
        # self.monitor_interval = MONITOR_INTERVAL  # ms between prints (adjustable)
        self.retry_interval = retry_interval  # ms between retries (adjustable)

        per_bs = max(1, num_edge_servers // num_bs)
        self.base_stations = []
        for i in range(num_bs):
            start = i * per_bs
            end = start + per_bs if i < num_bs-1 else num_edge_servers
            self.base_stations.append(BaseStation(self.edge_servers[start:end]))

        self.metrics = {
            'completed_tasks':0,'failed_tasks':0,
            'total_latency':0,'total_energy':0,
            'time_points':[],
            'edge_server_cpu_utilization':[],
            'base_station_bandwidth_utilization':[],
            'energy_per_task':[],
            'latency_per_task':[],
        }
    
    def schedule(self, t, func, *args):
        heapq.heappush(self.event_queue,
                       (t, next(self.counter), func, args))

    def add_mobile_devices(self):
        self.mobile_devices = [MobileDevice(base_station=self.base_stations[0]) for _ in range(self.num_devices)] # Just for now

    def install_iraf_engine(self, iraf_engine):
        self.iraf_engine = iraf_engine
    
    def generate_and_schedule_requests(self):
        self.total_tasks = self.num_tasks
        for _ in range(self.num_tasks):
            dev = random.choice(self.mobile_devices)
            arrival = random.uniform(0, self.arrival_window)
            task = dev.generate_task(arrival)
            
            self.schedule(arrival, self._handle_request, task)

    def _handle_request(self, task):
        alloc = self.allocate_resources(task)
        if not alloc: # Failed to allocate resources, retry later
            self.schedule(self.time + self.retry_interval,
                          self._handle_request, task)
            return False
        
        # Allocate resources successfully, increment depth with allocated action
        if self.algo == 'mcts':
            self.iraf_engine.increment_depth(alloc['action'])

        # schedule upload finish
        self.schedule(self.time + alloc['total_latency'],
                      self._handle_completion, alloc)
        return True

    def _handle_completion(self, alloc):
        # Release resources and update metrics
        alloc['bs'].release_bandwidth(alloc['bw_req'])
        alloc['primary'].release_cpu(alloc['p_cpu'])
        if alloc['collab']:
            alloc['collab'].release_cpu(alloc['c_cpu'])
        latency = self.time - alloc['task'].arrival_time
        self.metrics['completed_tasks'] += 1
        # self.metrics['total_latency'] += alloc['t_tx'] + alloc['t_proc']
        self.metrics['total_latency'] += latency
        self.metrics['total_energy'] += alloc['total_energy']
        # self.metrics['energy_per_task'].append(alloc['energy'])

    def allocate_resources_ratio(self, task):
        """
        Return offloading and resource ratios (α) based on chosen algorithm.
        - 'greedy': fixed 0.8 for all ratios
        - 'random': random in [0,1]
        """
        if self.algo == 'greedy':
            greedy_ratio = 1
            return tuple([greedy_ratio] * 5)
        elif self.algo == 'random':
            return tuple(random.uniform(0, 1.0) for _ in range(5))
        elif self.algo == 'mcts':
            ratios = self.iraf_engine.get_ratios(task, bs=self.base_stations, edge_servers=self.edge_servers)
            print(ratios)
            return ratios

    def _reserve_resources(self, task, alpha_B, alpha_e, alpha_ehat):
        # 1) Reserve bandwidth
        bs = task.bs
        bw_req = bs.total_bandwidth * alpha_B
        if not bs.allocate_bandwidth(bw_req):
            return None
        # print(f"DEBUG: Allocated bandwidth: {bw_req}MHz")
        # 2) Select servers (edge + best collab)
        primary = max(bs.edge_servers, key=lambda s: s.available_cpu) # Most free server
        others = [s for s in self.edge_servers if s is not primary]
        collab = max(others, key=lambda s: s.available_cpu) if others else None # Most free collab server
        
        # 3) Reserve CPU shares
        p_cpu = primary.cpu_capacity * alpha_e
        c_cpu = collab.cpu_capacity * alpha_ehat if collab else 0
        if p_cpu <= 0 or not primary.allocate_cpu(p_cpu):
            bs.release_bandwidth(bw_req)
            return None
        if c_cpu and collab and not collab.allocate_cpu(c_cpu):
            primary.release_cpu(p_cpu)
            bs.release_bandwidth(bw_req)
            return None
        
        # Everything is reserved successfully
        return {
            'primary': primary,
            'collab': collab,
            'p_cpu': p_cpu,
            'c_cpu': c_cpu,
            'bw_req': bw_req,
        }

    def allocate_resources(self, task):
        """
        Greedy or random baseline reservation:
          1) Obtain allocation ratios via allocate_resources_ratio()
          2) Reserve alpha(B)*bandwidth
          3) Select primary & collaborative servers
          4) Reserve alpha(e)*CPU on primary and alpha(ê)*CPU on collab
          5) Compute t_tx, t_proc with alpha(ηu→e) and alpha(ηe→ê)
          6) Deadline & energy check
        Returns allocation dict or None.
        """
        # print(f"DEBUG: Task properties: {task}")
        # 1) α parameters
        alpha_B, alpha_u2e, alpha_e2ehat, alpha_e, alpha_ehat = self.allocate_resources_ratio(task)
        
        resources = self._reserve_resources(task, alpha_B, alpha_e, alpha_ehat)
        if not resources:
            return None
        
        bw_req = resources['bw_req']
        primary = resources['primary']
        collab = resources['collab']    
        p_cpu = resources['p_cpu']
        c_cpu = resources['c_cpu']
        
        # 1) Local Computing model
        t_local = task.cpu_cycles * (1 - alpha_e2ehat)/task.device_cpu_freq
        E_local = CHIP_COEFFICIENT * (task.device_cpu_freq ** 2) * task.cpu_cycles * (1 - alpha_e2ehat)

        # 2) Computing Offloading model
        transmission_rate = alpha_B * np.log2(1 + task.device.transmit_power * task.channel_gain / CHANNEL_NOICE_VARIANCE)
        t_tx = task.data_size * alpha_u2e / transmission_rate
        E_tx = task.device.transmit_power * t_tx
        
        t_edge = task.cpu_cycles * alpha_u2e * (1 - alpha_e2ehat) / p_cpu
        t_collab = task.cpu_cycles * alpha_u2e * alpha_e2ehat / c_cpu if collab else 0

        total_latency = max(t_local, t_tx, t_edge, t_tx + t_collab)
        total_energy = E_local + E_tx

        # To be SIMPLE FOR NOW, we don't check deadline and rollback
        # 6) Check deadline and rollback 
        # if total_latency > task.max_latency:
        #     # print(f"DEBUG: Failed to allocate resources as it exceeds deadline, total latency: {total_latency}ms, max latency: {task.max_latency}ms")
        #     # print(f"DEBUG: Task: {task}")
        #     primary.release_cpu(p_cpu)
        #     if collab: collab.release_cpu(c_cpu)
        #     bs.release_bandwidth(bw_req)
        #     return None
        self.metrics['latency_per_task'].append(total_latency)
        self.metrics['energy_per_task'].append(total_energy)
        
        # 7) Compute energy
        return {
            'task': task, 
            'bs': task.bs,
            'primary': primary, 'collab': collab,
            'bw_req': bw_req, 'p_cpu': p_cpu, 'c_cpu': c_cpu, # For release resource later
            'total_latency': total_latency,
            'total_energy': total_energy,
            'action': (alpha_B, alpha_u2e, alpha_e2ehat, alpha_e, alpha_ehat),
        }

    def record_metrics(self):
        cpu_u = 1 - sum(s.available_cpu for s in self.edge_servers) / sum(s.cpu_capacity for s in self.edge_servers)
        bw_u  = 1 - sum(bs.available_bandwidth for bs in self.base_stations) / sum(bs.total_bandwidth for bs in self.base_stations)
        # if bw_u > 0.2:
        #     print([bs.available_bandwidth for bs in self.base_stations])
        #     print(f"CPU: {cpu_u:.2%}, BW: {bw_u:.2%}")
        #     print(f"BW: {bw_u}")
        #     time.sleep(10)
        self.metrics['time_points'].append(self.time)
        self.metrics['edge_server_cpu_utilization'].append(cpu_u)
        self.metrics['base_station_bandwidth_utilization'].append(bw_u)

        # if self.need_monitoring and (self.time - self.last_monitor_time >= 1000):
        #     completed = self.metrics['completed_tasks']
        #     failed = self.total_tasks - completed
        #     latency = self.metrics['total_latency'] / completed if completed else 0
        #     energy = self.metrics['total_energy'] / completed if completed else 0
        #     queue_len = len(self.event_queue)

        #     print(f"[t={self.time:6.1f}ms]  CPU: {cpu_u:.2%} | BW: {bw_u:.2%} | "
        #         f"Done: {completed}/{self.total_tasks} | Lat: {latency:.1f}ms | "
        #         f"Energy: {energy:.3f}J | Queue: {queue_len}")
            
        #     self.last_monitor_time = self.time

    def run(self, duration=10000):
        handles_func_count = {'_handle_request':0, '_handle_completion':0}
        while self.event_queue:
            t,_,func,args = heapq.heappop(self.event_queue)
            handles_func_count[func.__name__] += 1
            if self.need_duration and t > duration: break
            self.time = t
            self.record_metrics()
            func(*args)
        print(f"DEBUG: Handles function count: {handles_func_count}")
        self.metrics['failed_tasks'] = (
            self.total_tasks - self.metrics['completed_tasks'])
        c = self.metrics['completed_tasks']
        if c:
            self.metrics['avg_latency'] = self.metrics['total_latency'] / c
            self.metrics['avg_energy']  = self.metrics['total_energy'] / c
        else:
            self.metrics['avg_latency'] = self.metrics['avg_energy'] = 0
        return self.metrics

    def run_mcts(self, iterations=1000, duration=10000, optimize_for='latency'):
        metrics = []
        iteration = 0
        while iteration < iterations:
            # Process the event queue for one iteration
            while self.event_queue:
                t, _, func, args = heapq.heappop(self.event_queue)
                print(f"DEBUG: Popped event at time {t} with function {func.__name__} and args {args}")
                if t > duration:
                    break
                self.time = t
                self.record_metrics()
                func(*args)
            
            # After finishing event queue, check if all tasks are done
            failed_tasks = self.total_tasks - self.metrics['completed_tasks']
            if failed_tasks > 0:
                # Unfinished tasks => Penalty => Backup, but don't increment iteration
                penalty = -1000 - self.metrics['total_latency']
                self.iraf_engine.backup(penalty)
                self.reset_metrics_for_next_iteration()
                continue  # Retry this iteration
            else:
                # All tasks done => Good => Backup and record reward
                reward = -self.metrics['total_latency'] if optimize_for == 'latency' else -self.metrics['total_energy']
                self.iraf_engine.backup(reward)
                metrics.append(self.metrics)
                iteration += 1  # Only move to next iteration if success
                self.reset_metrics_for_next_iteration()
                print(f"DEBUG: Iteration {iteration} completed")
                time.sleep(10)
        return metrics

    def reset_metrics_for_next_iteration(self):
        self.metrics = {
            'completed_tasks':0,'failed_tasks':0,
            'total_latency':0,'total_energy':0,
            'time_points':[],
            'resource_utilization':[],
            'bandwidth_utilization':[],
            'energy_per_task':[]
        }
        
    def plot_results(self):
        m = self.metrics
        plt.figure(figsize=(12,10))
        # plt.subplot(5,1,0)
        # plt.plot(m['time_points'], m['edge_server_cpu_utilization'], label='CPU')
        # plt.plot(m['time_points'], m['base_station_bandwidth_utilization'], label='BW')
        # plt.legend(); plt.title('Utilization')
        plt.subplot(5,1,1)
        plt.bar(['Done','Fail'],
                [m['completed_tasks'], m['failed_tasks']])
        plt.title(f"Task Success Rate: {m['completed_tasks']}/{m['failed_tasks']}")
        plt.subplot(5,1,2)
        plt.plot(m['latency_per_task'])
        plt.title(f"Latency Total:{m['total_latency']:.1f}ms")
        plt.subplot(5,1,3)
        plt.plot(m['energy_per_task'])
        plt.title(f"Energy Total:{m['total_energy']:.3f}J")
        plt.subplot(5,1,4)
        plt.plot(m['edge_server_cpu_utilization'], label='CPU')
        plt.subplot(5,1,5)
        plt.plot(m['base_station_bandwidth_utilization'], label='BW')
        plt.legend(); plt.title('Utilization')


        plt.tight_layout() 
        if not os.path.exists('result_plots'):
            os.makedirs('result_plots')
        plt.savefig(f'result_plots/{self.algo}_results_{self.num_devices}_{self.num_tasks}_{self.need_duration}_{time.strftime("%Y%m%d_%H%M%S")}.png')

if __name__ == '__main__':
    sim = CoMECEnvSimulator()
    sim.add_mobile_devices()
    sim.generate_and_schedule_requests()
    print(f"Sim: BS={len(sim.base_stations)} svr={len(sim.edge_servers)} dev={len(sim.mobile_devices)} tasks={sim.total_tasks}")
    metrics = sim.run()
    print(f"Done={metrics['completed_tasks']}, Fail={metrics['failed_tasks']}, Avg Lat={metrics['avg_latency']:.1f}ms, Avg E={metrics['avg_energy']:.3f}J")
    sim.plot_results()
