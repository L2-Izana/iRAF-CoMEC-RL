import random
import time
from matplotlib import pyplot as plt
import numpy as np
from ..core.comec_env import CoMECEnvironment
from ..visualization.metrics import MetricsTracker

# random.seed(187)
# np.random.seed(187)

class CoMECSimulator:
    """
    High-level simulator that runs episodes on CoMECEnvironment,
    applies a selection algorithm (e.g., random, greedy, or MCTS),
    and collects metrics.
    """
    def __init__(
        self,
        need_duration=False,
        iterations=1,
        max_time=10000,
        retry_interval=10,
        **env_kwargs
    ):
        self.need_duration = need_duration
        self.iterations = iterations
        self.max_time = max_time
        self.retry_interval = retry_interval

        # Create environment and metrics
        self.env = CoMECEnvironment(retry_interval=retry_interval, **env_kwargs)
        self.metrics = MetricsTracker(self.env.num_tasks)

        # MCTS engine placeholder
        self.iraf_engine = None

    def install_iraf_engine(self, engine):
        self.iraf_engine = engine

    def run(self, residual=True, optimize_for='latency'):
        """Run simulation for `iterations` episodes and return metrics."""
        all_metrics = []
        self.env.reset(reset_tasks=True)
        for _ in range(self.iterations):
            if _ % 1000 == 0:
                print(f"Running iteration {_}")
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
            
            # Note: task completions record latency/energy via callbacks
            all_metrics.append(average_metrics)
            # time.sleep(1)
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
