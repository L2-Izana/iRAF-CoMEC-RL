import numpy as np

from iraf_engine.mcts import MCTS
from iraf_engine.mcts_pw import MCTS_PW

class IraFEngine:
    def __init__(self, algorithm: str = 'mcts', input_dim=9):
        self.algorithm = algorithm
        if self.algorithm == 'mcts':
            self.mcts = MCTS(input_dim)
        elif self.algorithm == 'mcts-dnn':
            self.mcts = MCTS(input_dim, use_dnn=True)
        elif self.algorithm == 'mcts-pw':
            self.mcts_pw = MCTS_PW(input_dim, use_dnn=False)
        elif self.algorithm == 'mcts-pw-dnn':
            self.mcts_pw = MCTS_PW(input_dim, use_dnn=True)
        elif self.algorithm == 'random' or self.algorithm == 'greedy':
            pass
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")


    def get_ratios(self, env_resources):
        if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
            return self.mcts.get_ratios(env_resources)
        elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
            return self.mcts_pw.get_ratios(env_resources)
        elif self.algorithm == 'random':
            return np.random.rand(5)
        elif self.algorithm == 'greedy':
            return np.ones(5)
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")

    def backprop(self, reward):
        if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
            self.mcts.backprop(reward)
        elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
            self.mcts_pw.backprop(reward)
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")

    def get_best_action(self):
        if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
            return self.mcts.get_best_action()
        elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
            return self.mcts_pw.get_best_action()
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")

    def extract_action_probabilities(self):
        if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
            return self.mcts.extract_action_probabilities()
        elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
            return self.mcts_pw.extract_action_probabilities()
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")
    
    def get_node_count(self):
        if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
            return self.mcts.get_node_count()
        elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
            return self.mcts_pw.get_node_count()
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")

    def get_dnn_call_count(self):
        if 'dnn' in self.algorithm:
            if self.algorithm == 'mcts-pw-dnn':
                return self.mcts_pw.dnn_call_count
            elif self.algorithm == 'mcts-dnn':
                return self.mcts.dnn_call_count
            else:
                return 0    
        else:
            return 0
    
        