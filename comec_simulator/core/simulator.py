import time
import numpy as np

from comec_simulator.core.constants import DNN_INPUT_DIM
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
        iterations,
        algorithm='a0c',
    ):
        self.iterations = iterations

        # Create environment and metrics
        self.env = CoMECEnvironment()
        self.metrics = MetricsTracker(self.env.get_num_tasks(), algorithm)

        # MCTS engine placeholder
        self.iraf_engine = IraFEngine(algorithm=algorithm)
        self.algorithm = algorithm
            
    def run(self, optimize_for, save_empirical_run=False):
        """Run simulation for `iterations` episodes and return metrics."""
        all_metrics = []
        self.env.reset(reset_tasks=True)
        for _ in range(self.iterations):            
            if self._check_tree_stop_condition(self.iraf_engine.get_node_count(), self.metrics.get_average_metrics(), self._get_objective_value(self.metrics.get_average_metrics(), optimize_for), _):
                break

            # Restart environment and metrics
            self.env.reset(reset_tasks=False)
            self.metrics.reset()

            # Main simulation loop
            i = 0
            while True:
                # If we've exceeded time, stop
                # if self.env.event_queue and self.env.event_queue[0][0] > self.max_time:
                # Finish the simulation if no events are left
                if not self.env.event_queue:
                    break
                event = self.env.pop_event()
                step_args = None
                assert event is not None, "Event should not be None at this point"
                if event['func'] == self.env.handle_request:
                    task = event['args'][0]
                    if 'dnn' in self.algorithm or self.algorithm == 'a0c':
                        env_resources = self.env.get_resources_dnn(task)
                    else:
                        env_resources = self.env.get_resources(task)
                    alphas = self.iraf_engine.get_ratios(env_resources)
                    step_args = (self.env.handle_request, (task, alphas))
                elif event['func'] == self.env.handle_completion:
                    total_latency = event['args'][0]['total_latency']
                    total_energy = event['args'][0]['total_energy']
                    self.metrics.record_task_completion(total_latency, total_energy)
                    step_args = (self.env.handle_completion, event['args'])

                if step_args:
                    self.env.step(step_args)

                # Collect intermediate system metrics
                current_time = self.env.time
                self.metrics.record_metrics(
                    current_time,
                    self.env.get_edge_servers(),
                    self.env.get_base_stations()
                )

            # After run, backprop the tree and collect final data
            average_metrics = self.metrics.get_average_metrics()
            reward = self._get_objective_value(average_metrics, optimize_for)
            
            assert reward > 0, f"Reward should be positive, got {reward}"
            self.iraf_engine.backprop(-reward) # Make it negative to minimize the objective value
            node_count = self.iraf_engine.get_node_count()
            
            # Record tree iteration step attributes
            self.metrics.record_tree_iteration_step_attributes(node_count, reward)
            
            # Note: task completions record latency/energy via callbacks
            all_metrics.append(average_metrics)
            
            # Log the performance every 500 iterations            
            if (_ + 1) % 500 == 0:
                self._print_results(average_metrics, node_count, reward, _)

            
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
    
    def eval(self, optimize_for):
        """Run simulation for `iterations` episodes and return metrics."""
        all_metrics = []
        self.env.reset(reset_tasks=True)

        while True:
            if not self.env.event_queue:
                break
            event = self.env.pop_event()
            step_args = None
            assert event is not None, "Event should not be None at this point"
            if event['func'] == self.env.handle_request:
                task = event['args'][0]
                alphas = self.iraf_engine.a0c.get_eval_ratios()
                step_args = (self.env.handle_request, (task, alphas))
            elif event['func'] == self.env.handle_completion:
                total_latency = event['args'][0]['total_latency']
                total_energy = event['args'][0]['total_energy']
                # print(f"E: {total_energy} | L: {total_latency}")
                self.metrics.record_task_completion(total_latency, total_energy)
                step_args = (self.env.handle_completion, event['args'])

            if step_args:
                self.env.step(step_args)

            # Collect intermediate system metrics
            current_time = self.env.time
            self.metrics.record_metrics(
                current_time,
                self.env.get_edge_servers(),
                self.env.get_base_stations()
            )

        # After run, backprop the tree and collect final data
        average_metrics = self.metrics.get_average_metrics()
        
        return average_metrics
    
    def run_with_best_action(self, best_action):
        """Final simulation run for task and env condition recording"""
        # Restart environment and metrics
        self.env.reset(reset_tasks=False)
        idx = 0
        env_resources_record = []   
        # Main simulation loop
        while True:
            # If we've exceeded time, stop
            if not self.env.event_queue:
                break
            event = self.env.pop_event()
            step_args = None
            if event is None:
                break
            if event['func'] == self.env.handle_request:
                task = event['args'][0]
                env_resources = self.env.get_resources(task)
                env_resources_record.append(env_resources)
                if idx < len(best_action):
                    alphas = best_action[idx]
                    step_args = (self.env.handle_request, (task, alphas))
                    idx += 1
                else:
                    print(task.arrival_time)
            elif event['func'] == self.env.handle_completion:
                step_args = (self.env.handle_completion, event['args'])

            if step_args:
                self.env.step(step_args)

        return np.array(env_resources_record)

    def _get_objective_value(self, average_metrics, optimize_for):
        if optimize_for == 'latency':
            return average_metrics['avg_latency']
        elif optimize_for == 'energy':
            return average_metrics['avg_energy']
        else:
            return average_metrics['avg_latency'] + average_metrics['avg_energy']
    
    def _print_results(self, average_metrics, node_count, reward, iteration):
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

    def _check_tree_stop_condition(self, node_count, average_metrics, reward, _):
        def _has_converged(rewards):
            if len(rewards) >= TREE_CONVERGENCE_ITERATION_LIMIT: # The tree needs lots of iterations to converge
                reward_std = np.std(rewards[-TREE_CONVERGENCE_WINDOW:])
                return reward_std < TREE_CONVERGENCE_THRESHOLD
            return False

        # Check if the tree reaches the storage budget 
        if node_count >= TREE_STORAGE_BUDGET:
            print(f"\n╔{'═' * 40}╗")
            print(f"║{'Tree Reaches Storage Budget':^40}║")
            print(f"╚{'═' * 40}╝\n")
            self._print_results(average_metrics, node_count, reward, _)
            return True
        
        # Check if the reward has converged
        if _has_converged(self.metrics.rewards):
            print(f"\n╔{'═' * 40}╗")
            print(f"║{'Reward Has Converged':^40}║")
            print(f"╚{'═' * 40}╝\n")
            self._print_results(average_metrics, node_count, reward, _)
            return True
        return False