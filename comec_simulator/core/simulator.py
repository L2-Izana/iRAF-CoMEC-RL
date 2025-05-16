import itertools
import heapq
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
        need_duration=True,
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

    def run(self):
        """Run simulation for `iterations` episodes and return metrics."""
        all_metrics = []
        for _ in range(self.iterations):
            # Restart environment and metrics
            self.env.reset()
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
                if event['func_name'] == '_handle_request':
                    task = event['args'][0]
                    env_resources = self.env.get_resources(task)
                    alphas = self.iraf_engine.get_ratios(env_resources)
                    step_args = ("_handle_request", (task, alphas))
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

            # After run, collect final data
            # Note: task completions record latency/energy via callbacks
            all_metrics.append(self.metrics.get_average_metrics())

        return all_metrics

    def run_mcts(self, iterations=1000, optimize_for='latency'):
        """Run with MCTS optimization, backing up rewards each episode."""
        if not self.iraf_engine:
            raise ValueError("No MCTS engine installed.")

        results = []
        for _ in range(iterations):
            self.env.reset()
            self.metrics.reset()

            # Simulate one episode
            while True:
                if self.need_duration and self.env.event_queue and self.env.event_queue[0][0] > self.max_time:
                    break
                if not self.env.step():
                    break
                # MCTS engine provides ratios from task context
                # engine should be called inside allocate_resources via environment

            # Determine reward
            completed = self.metrics.completed_tasks
            failed = self.env.num_tasks - completed
            if failed > 0:
                penalty = -1000 - self.metrics.total_latency
                self.iraf_engine.backup(penalty)
            else:
                reward = -self.metrics.total_latency if optimize_for == 'latency' else -self.metrics.total_energy
                self.iraf_engine.backup(reward)
                results.append(self.metrics.get_average_metrics())

        return results

