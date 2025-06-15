import time
import numpy as np

from iraf_engine.iraf_engine import IraFEngine
from ..core.comec_env import CoMECEnvironment
from ..visualization.metrics import MetricsTracker

TREE_STORAGE_BUDGET = 1e7 # 10 million nodes, if more than this, RAM explodes :(
TREE_CONVERGENCE_THRESHOLD = 0.05 # lower the threshold a little bit to increase the exploration, as the mcts-pw is too powerful :), only 1001 to converge
TREE_CONVERGENCE_WINDOW = 50 # the same as above
TREE_CONVERGENCE_ITERATION_LIMIT = 1000

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
        algorithm='mcts-pw-dnn' 
    ):
        self.need_duration = need_duration
        self.iterations = iterations
        self.max_time = max_time
        self.retry_interval = retry_interval

        # Create environment and metrics
        self.env = CoMECEnvironment(retry_interval=retry_interval, num_devices=num_devices, num_tasks=num_tasks, num_edge_servers=num_es, num_bs=num_bs)
        self.metrics = MetricsTracker(self.env.num_tasks, algorithm)

        # MCTS engine placeholder
        self.iraf_engine = IraFEngine(input_dim=4+num_es+num_bs, algorithm=algorithm)
        self.algorithm = algorithm
            
    def run(self, residual=True, optimize_for='latency', save_empirical_run=False):
        """Run simulation for `iterations` episodes and return metrics."""
        all_metrics = []
        self.env.reset(reset_tasks=True)
        for _ in range(self.iterations):            
            if self.check_tree_stop_condition(self.iraf_engine.get_node_count(), self.metrics.rewards, self.metrics.get_average_metrics(), self.get_objective_value(self.metrics.get_average_metrics(), optimize_for), _):
                break

            # Restart environment and metrics
            self.env.reset(reset_tasks=False)
            self.metrics.reset()

            # Main simulation loop
            i = 0
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
                    if 'dnn' in self.algorithm:
                        env_resources = self.env.get_resources_dnn(task)
                    else:
                        env_resources = self.env.get_resources(task)
                    if self.algorithm == 'a0c':
                        alphas = self.iraf_engine.get_ratios(env_resources)
                        i += 1
                    else:
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

            # assert i == 10, f"The fuck is calling alphas {i} times"
            # After run, backprop the tree and collect final data
            average_metrics = self.metrics.get_average_metrics()
            reward = self.get_objective_value(average_metrics, optimize_for)
            
            self.iraf_engine.backprop(-reward) # Make it negative to minimize the objective value
            node_count = self.iraf_engine.get_node_count()
            
            
            # Record tree iteration step attributes
            self.metrics.record_tree_iteration_step_attributes(node_count, reward)
            
            # Note: task completions record latency/energy via callbacks
            all_metrics.append(average_metrics)
            
            # Log the performance every 500 iterations            
            if (_ + 1) % 500 == 0:
                self.print_results(average_metrics, node_count, reward, _)

            
        if optimize_for == 'latency':
            metrics = [metric['avg_latency'] for metric in all_metrics]
            
        elif optimize_for == 'energy':
            metrics = [metric['avg_energy'] for metric in all_metrics]
        elif optimize_for == 'latency_energy':
            metrics = [metric['avg_latency'] + metric['avg_energy'] for metric in all_metrics]
        else:
            raise ValueError(f"Invalid optimize_for: {optimize_for}")
        if save_empirical_run:
            with open(f'{optimize_for}_metrics_{self.algorithm}_{time.time()}.txt', 'w') as f:
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

    def get_objective_value(self, average_metrics, optimize_for):
        if optimize_for == 'latency':
            return average_metrics['avg_latency']
        elif optimize_for == 'energy':
            return average_metrics['avg_energy']
        else:
            return average_metrics['avg_latency'] + average_metrics['avg_energy']

    def has_converged(self, rewards):
        if len(rewards) >= TREE_CONVERGENCE_ITERATION_LIMIT: # The tree needs lots of iterations to converge
            reward_std = np.std(rewards[-TREE_CONVERGENCE_WINDOW:])
            return reward_std < TREE_CONVERGENCE_THRESHOLD
        return False
    
    def print_results(self, average_metrics, node_count, reward, iteration):
        # Box drawing characters
        top_bottom = "═" * 40
        side = "║"
        
        # Format the metrics with fixed width
        iteration_str = f"Iteration: {iteration + 1}"
        latency_str = f"Average Latency: {average_metrics['avg_latency']:.2f}"
        energy_str = f"Average Energy: {average_metrics['avg_energy']:.2f}"
        nodes_str = f"Node Count: {node_count}"
        reward_str = f"Reward: {reward:.2f}"
        dnn_call_count = self.iraf_engine.get_dnn_call_count()
        dnn_call_count_str = f"DNN Call Count: {dnn_call_count}"
        # Print the box
        print(f"\n╔{top_bottom}╗")
        print(f"{side}{iteration_str:^40}{side}")
        print(f"{side}{'─' * 40}{side}")
        print(f"{side}{latency_str:^40}{side}")
        print(f"{side}{energy_str:^40}{side}")
        print(f"{side}{nodes_str:^40}{side}")
        print(f"{side}{reward_str:^40}{side}")
        if dnn_call_count > 0:
            print(f"{side}{dnn_call_count_str:^40}{side}")
        print(f"╚{top_bottom}╝\n")

    def check_tree_stop_condition(self, node_count, rewards, average_metrics, reward, _):
        # Check if the tree reaches the storage budget 
        if node_count >= TREE_STORAGE_BUDGET:
            print(f"\n╔{'═' * 40}╗")
            print(f"║{'Tree Reaches Storage Budget':^40}║")
            print(f"╚{'═' * 40}╝\n")
            self.print_results(average_metrics, node_count, reward, _)
            return True
        
        # Check if the reward has converged
        if self.has_converged(self.metrics.rewards):
            print(f"\n╔{'═' * 40}╗")
            print(f"║{'Reward Has Converged':^40}║")
            print(f"╚{'═' * 40}╝\n")
            self.print_results(average_metrics, node_count, reward, _)
            return True
        return False
