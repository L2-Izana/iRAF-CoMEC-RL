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
    def __init__(self, algorithm, cfg):
        self.algorithm = algorithm.lower()
        if self.algorithm == 'mcts':
            self.model = MCTS(use_dnn=cfg.use_dnn, cfg=cfg)
        elif self.algorithm == 'mcts-pw':
            self.model = MCTS_PW(use_dnn=cfg.use_dnn, cfg=cfg)
        elif self.algorithm == 'a0c':
            self.model = A0C(cfg.has_max_threshold, cfg.max_pw_floor, cfg.discount_factor, use_dnn=cfg.use_dnn)
        elif self.algorithm == 'random' or self.algorithm == 'greedy':
            pass
        else:
            raise ValueError(f"Algorithm {self.algorithm} not supported")

    def get_ratios(self, env_resources):
        if self.algorithm == 'random':
            return np.random.rand(5)
        elif self.algorithm == 'greedy':
            return np.ones(5)*0.99
        else:
            return self.model.get_ratios(env_resources)

    def backprop(self, rewards):
        self.model.backprop(rewards)
        
    # def get_best_action(self):
    #     if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
    #         return self.mcts.get_best_action()
    #     elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
    #         return self.mcts_pw.get_best_action()
    #     else:
    #         raise ValueError(f"Algorithm {self.algorithm} not supported")

    # def extract_action_probabilities(self):
    #     if self.algorithm == 'mcts' or self.algorithm == 'mcts-dnn':
    #         return self.mcts.extract_action_probabilities()
    #     elif self.algorithm == 'mcts-pw' or self.algorithm == 'mcts-pw-dnn':
    #         return self.mcts_pw.extract_action_probabilities()
    #     elif self.algorithm == 'a0c':
    #         return self.a0c.extract_action_probabilities()
    #     else:
    #         raise ValueError(f"Algorithm {self.algorithm} not supported")
    
    def get_node_count(self):
        assert hasattr(self.model,  'get_node_count'), f"Algorithm {self.algorithm} does not support node count retrieval"
        return self.model.get_node_count()
        
    # def get_dnn_call_count(self):
    #     if 'dnn' in self.algorithm:
    #         if self.algorithm == 'mcts-pw-dnn':
    #             return self.mcts_pw.dnn_call_count
    #         elif self.algorithm == 'mcts-dnn':
    #             return self.mcts.dnn_call_count
    #         else:
    #             return 0    
    #     else:
    #         return 0
    
    # def get_training_dataset(self):
    #     if self.algorithm == 'a0c':
    #         return self.model.get_training_dataset()
    #     else:
    #         raise ValueError(f"Only A0C supports it right now")