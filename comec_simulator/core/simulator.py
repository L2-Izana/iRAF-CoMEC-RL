import time
import numpy as np

from iraf_engine.iraf_engine import IraFEngine
from ..core.comec_env import CoMECEnvironment
from ..visualization.metrics import MetricsTracker

class CoMECSimulator:
    """
    High-level simulator that runs episodes on CoMECEnvironment,
    applies a selection algorithm (e.g., random, greedy, or MCTS),
    and collects metrics.
    """
    def __init__(
        self,
        num_devices,
        num_tasks,
        iterations,
        num_es,
        num_bs,
        need_duration=False,
        max_time=10000,
        retry_interval=10,
        use_dnn=False,
    ):
        self.need_duration = need_duration
        self.iterations = iterations
        self.max_time = max_time
        self.retry_interval = retry_interval

        # Create environment and metrics
        self.env = CoMECEnvironment(retry_interval=retry_interval, num_devices=num_devices, num_tasks=num_tasks, num_edge_servers=num_es, num_bs=num_bs)
        self.metrics = MetricsTracker(self.env.num_tasks)

        # MCTS engine placeholder
        self.iraf_engine = IraFEngine(input_dim=4+num_es+num_bs, use_dnn=use_dnn)
        self.use_dnn = use_dnn


    def run(self, residual=True, optimize_for='latency'):
        """Run simulation for `iterations` episodes and return metrics."""
        all_metrics = []
        self.env.reset(reset_tasks=True)
        for _ in range(self.iterations):
            if (_ + 1) % 500 == 0:
                print(f"Running iteration {_ + 1}")
                print(f"Current metrics:")
                average_metrics = self.metrics.get_average_metrics()
                print(f"Average latency: {average_metrics['avg_latency']:.2f}")
                print(f"Average energy: {average_metrics['avg_energy']:.2f}")
                print(f"Node count: {self.iraf_engine.get_node_count()}")
            # Restart environment and metrics
            self.env.reset(reset_tasks=False)
            self.metrics.reset()

            # Main simulation loop
            while True:
                # If we've exceeded time, stop
                if self.need_duration and self.env.event_queue and self.env.event_queue[0][0] > self.max_time:
                    break
                # progressed = self.env.step()
                # if not progressed:
                #     break
                event = self.env.pop_event()
                step_args = None
                if event is None:
                    break
                if event['func_name'] == '_handle_request':
                    task = event['args'][0]
                    if self.use_dnn:
                        env_resources = self.env.get_resources_dnn(task)
                    else:
                        env_resources = self.env.get_resources(task)
                    alphas = self.iraf_engine.get_ratios(env_resources)
                    step_args = ("_handle_request", (task, alphas, residual))
                elif event['func_name'] == '_handle_completion':
                    total_latency = event['args'][0]['total_latency']
                    total_energy = event['args'][0]['total_energy']
                    self.metrics.record_task_completion(total_latency, total_energy)
                    step_args = ("_handle_completion", event['args'])

                if step_args:
                    self.env.step(step_args)

                # Collect intermediate system metrics
                current_time = self.env.time
                self.metrics.record_metrics(
                    current_time,
                    self.env.edge_servers,
                    self.env.base_stations
                )

            # After run, backprop the tree and collect final data
            average_metrics = self.metrics.get_average_metrics()
            self.iraf_engine.backprop(average_metrics, optimize_for=optimize_for)
            
            # Record tree iteration step attributes
            if optimize_for == 'latency':
                reward = average_metrics['avg_latency']
            elif optimize_for == 'energy':
                reward = average_metrics['avg_energy']
            else:
                reward = average_metrics['avg_latency'] + average_metrics['avg_energy']
            self.metrics.record_tree_iteration_step_attributes(self.iraf_engine.get_node_count(), reward)
            
            # Note: task completions record latency/energy via callbacks
            all_metrics.append(average_metrics)
            
        if optimize_for == 'latency':
            metrics = [metric['avg_latency'] for metric in all_metrics]
            
        elif optimize_for == 'energy':
            metrics = [metric['avg_energy'] for metric in all_metrics]
        elif optimize_for == 'latency_energy':
            metrics = [metric['avg_latency'] + metric['avg_energy'] for metric in all_metrics]
        else:
            raise ValueError(f"Invalid optimize_for: {optimize_for}")
        with open(f'{optimize_for}_metrics{time.time()}.txt', 'w') as f:
            np.savetxt(f, metrics)
        return all_metrics
    
    def run_with_best_action(self, best_action, residual=True, optimize_for='latency'):
        """Final simulation run for task and env condition recording"""
        # Restart environment and metrics
        self.env.reset(reset_tasks=False)
        idx = 0
        env_resources_record = []
        # Main simulation loop
        while True:
            # If we've exceeded time, stop
            if self.need_duration and self.env.event_queue and self.env.event_queue[0][0] > self.max_time:
                break

            event = self.env.pop_event()
            step_args = None
            if event is None:
                break
            if event['func_name'] == '_handle_request':
                task = event['args'][0]
                env_resources = self.env.get_resources_dnn(task)
                env_resources_record.append(env_resources)
                if idx < len(best_action):
                    alphas = best_action[idx]
                    step_args = ("_handle_request", (task, alphas, residual))
                    idx += 1
                else:
                    print(task.arrival_time)
            elif event['func_name'] == '_handle_completion':
                step_args = ("_handle_completion", event['args'])

            if step_args:
                self.env.step(step_args)

        return np.array(env_resources_record)
