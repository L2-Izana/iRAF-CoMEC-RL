import numpy as np

from iraf_engine.mcts import MCTS


class IraFEngine:
    def __init__(self, algorithm: str = 'mcts', use_dnn: bool = False, input_dim=9):
        self.algorithm = algorithm
        self.use_dnn = use_dnn
        # print(f"Using DNN cc: {use_dnn}")
        if self.algorithm == 'mcts':
            self.mcts = MCTS(input_dim, use_dnn=use_dnn)

    def get_ratios(self, env_resources):
        if self.algorithm == 'mcts':
            return self.mcts.get_ratios(env_resources)
        elif self.algorithm == 'random':
            return np.random.rand(5)
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")

    def backprop(self, average_metrics, optimize_for='latency'):
        if self.algorithm == 'mcts':
            if optimize_for == 'latency':           
                reward = -average_metrics['avg_latency']
            elif optimize_for == 'energy':
                reward = -average_metrics['avg_energy']
            elif optimize_for == 'latency_energy':
                reward = -average_metrics['avg_latency'] - average_metrics['avg_energy']
            else:
                raise ValueError(f"Optimize for {optimize_for} not supported")
            self.mcts.backprop(reward)

    def get_best_action(self):
        if self.algorithm == 'mcts':
            return self.mcts.get_best_action()
    
    def extract_action_probabilities(self):
        if self.algorithm == 'mcts':
            return self.mcts.extract_action_probabilities()
    
    def get_node_count(self):
        if self.algorithm == 'mcts':
            return self.mcts.total_nodes
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")
