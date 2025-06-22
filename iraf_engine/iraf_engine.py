import numpy as np

from iraf_engine.mcts import MCTS
from iraf_engine.mcts_pw import MCTS_PW
from iraf_engine.a0c import A0C, A0C_DNN

class IraFEngine:
    """
    Unified interface for selection algorithms:
    - MCTS / MCTS-DNN
    - MCTS-PW / MCTS-PW-DNN
    - A0C / A0C-DNN
    - Random
    - Greedy
    """
    def __init__(self, algorithm):
        self.algorithm = algorithm.lower()
        self.use_dnn = self.algorithm.endswith('-dnn')
        self.base = self.algorithm[:-4] if self.use_dnn else self.algorithm
        if self.base == 'mcts':
            self.model = MCTS(use_dnn=self.use_dnn)
        elif self.base == 'mcts-pw':
            self.model = MCTS_PW(use_dnn=self.use_dnn)
        elif self.algorithm == 'random' or self.algorithm == 'greedy':
            pass
        elif self.algorithm == 'a0c':
            self.a0c = A0C()
        elif self.algorithm == 'a0c-dnn':
            self.a0c = A0C_DNN()
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
        elif self.algorithm == 'a0c':
            return self.a0c.get_ratios_a0c(env_resources)
        elif self.algorithm == 'a0c-dnn':
            return self.a0c.get_ratios_a0c_dnn(env_resources)
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")

    def backprop(self, reward):
        if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
            self.mcts.backprop(reward)
        elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
            self.mcts_pw.backprop(reward)
        elif self.algorithm == 'a0c':
            self.a0c.backprop(reward)
        elif self.algorithm == 'a0c-dnn':
            self.a0c.backprop(reward)   
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
        elif self.algorithm == 'a0c':
            return self.a0c.extract_action_probabilities()
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")
    
    def get_node_count(self):
        if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
            return self.mcts.get_node_count()
        elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
            return self.mcts_pw.get_node_count()
        elif self.algorithm == 'a0c':
            return self.a0c.get_node_count()
        elif self.algorithm == 'a0c-dnn':
            return self.a0c.get_node_count()
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
    
    def get_training_dataset(self):
        if self.algorithm == 'a0c':
            return self.a0c.get_training_dataset()
        else:
            raise ValueError(f"Only A0C supports it right now")