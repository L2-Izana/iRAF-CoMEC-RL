# # env/gym_env.py

# import numpy as np
# import gym
# from gym import spaces

# from comec_simulator.core.comec_env import CoMECEnvironment


# class CoMECGymEnv(gym.Env):
#     """Gym wrapper around your CoMECEnvironment simulator."""
#     metadata = {'render.modes': []}

#     def __init__(   
#         self,
#         num_devices=20,
#         num_tasks=20,
#         arrival_window=10000,
#         num_edge_servers=4,
#         num_bs=1,
#         retry_interval=10,
#     ):
#         super().__init__()

#         # 1) instantiate your simulator
#         self.sim = CoMECEnvironment(
#             num_devices=num_devices,
#             num_tasks=num_tasks,
#             arrival_window=arrival_window,
#             num_edge_servers=num_edge_servers,
#             num_bs=num_bs,
#             retry_interval=retry_interval,
#         )

#         # 2) define action space: 5 alpha ratios in [0,1]
#         self.action_space = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)

#         # 3) run one reset & pop first arrival just to infer obs‐dim
#         self.sim.reset(reset_tasks=True)
#         first_ev = self._pop_request_event()
#         first_task = first_ev['args'][0]
#         sample_obs = self.sim.get_resources_dnn(first_task)

#         # 4) define observation space from DNN features
#         self.observation_space = spaces.Box(
#             low=0.0,
#             high=1.0,
#             shape=sample_obs.shape,
#             dtype=np.float32,
#         )

#         # keep track of the “current” arrival event
#         self.current_event = first_ev

#     def _pop_request_event(self):
#         """Remove & return the next arrival (_handle_request) event."""
#         ev = self.sim.pop_event()
#         while ev and ev['func_name'] != '_handle_request':
#             ev = self.sim.pop_event()
#         return ev

#     def _peek_request_event(self):
#         """Peek (without popping) at the soonest _handle_request event, or None."""
#         # look through the heap for all request events
#         reqs = [
#             (t, func, args)
#             for (t, _, func, args) in self.sim.event_queue
#             if func == '_handle_request'
#         ]
#         if not reqs:
#             return None
#         # pick the one with smallest timestamp
#         t, func, args = min(reqs, key=lambda x: x[0])
#         return {'time': t, 'func_name': func, 'args': args}

#     def reset(self):
#         """Reset the simulator and return the first observation."""
#         self.sim.reset(reset_tasks=True)
#         self.current_event = self._pop_request_event()
#         task = self.current_event['args'][0]
#         obs = self.sim.get_resources_dnn(task)
#         return obs.astype(np.float32)

#     def step(self, action):
#         """
#         1) Pop exactly one arrival event
#         2) Allocate resources & compute reward
#         3) Schedule completion (for logging) but don’t block on it
#         4) Peek next arrival for the next obs, or end episode.
#         """
#         # 1) get the next arrival
#         ev = self._pop_request_event()
#         task = ev['args'][0]

#         # 2) allocate + reward = -average(latency, energy)
#         alloc = self.sim.allocate_resources(task, action, residual=True)
#         lat, eng = alloc['total_latency'], alloc['total_energy']
#         reward = - (lat + eng) / 2.0
#         info = {'latency': lat, 'energy': eng}

#         # 3) enqueue completion event (for later logging/analysis)
#         self.sim._enqueue(self.sim.time + lat, '_handle_completion', alloc)

#         # 4) build next obs or end
#         next_ev = self._peek_request_event()
#         if next_ev is None:
#             done = True
#             obs = np.zeros(self.observation_space.shape, dtype=np.float32)
#         else:
#             done = False
#             obs = self.sim.get_resources_dnn(next_ev['args'][0])

#         return obs.astype(np.float32), float(reward), done, info

#     def render(self, mode='human'):
#         print(f"[Time={self.sim.time:.2f}] Queue={len(self.sim.event_queue)}")

#     def close(self):
#         pass
# env/gym_env.py

# env/env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from comec_simulator.core.comec_env import CoMECEnvironment


class CoMECGymEnv(gym.Env):
    """Gymnasium‐compatible wrapper around CoMECEnvironment."""
    metadata = {'render_modes': []}

    def __init__(
        self,
        num_devices=20,
        num_tasks=20,
        arrival_window=10000,
        num_edge_servers=4,
        num_bs=1,   
        retry_interval=10,
    ):
        super().__init__()

        # 1) instantiate your simulator
        self.sim = CoMECEnvironment(
            num_devices=num_devices,
            num_tasks=num_tasks,
            arrival_window=arrival_window,
            num_edge_servers=num_edge_servers,
            num_bs=num_bs,
            retry_interval=retry_interval,
        )

        # 2) define action space: 5 alpha ratios in [0,1]
        self.action_space = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)

        # 3) infer obs‐dim by popping first arrival
        self.sim.reset(reset_tasks=True)
        first_ev = self._pop_request_event()
        sample_obs = self.sim.get_resources_dnn(first_ev['args'][0])

        # 4) define observation space
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=sample_obs.shape,
            dtype=np.float32,
        )

        # store the current arrival event
        self.current_event = first_ev

    def _pop_request_event(self):
        """Pop & return next '_handle_request' event."""
        ev = self.sim.pop_event()
        while ev and ev['func_name'] != '_handle_request':
            ev = self.sim.pop_event()
        return ev

    def _peek_request_event(self):
        """Peek (without removing) at the earliest '_handle_request', or None."""
        reqs = [
            (t, func, args)
            for (t, _, func, args) in self.sim.event_queue
            if func == '_handle_request'
        ]
        if not reqs:
            return None
        t, func, args = min(reqs, key=lambda x: x[0])
        return {'time': t, 'func_name': func, 'args': args}

    def reset(self, *, seed=None, options=None):
        """Reset simulator & return first observation (new_step_api)."""
        super().reset(seed=seed)
        self.sim.reset(reset_tasks=True)
        self.current_event = self._pop_request_event()
        obs = self.sim.get_resources_dnn(self.current_event['args'][0])
        return obs.astype(np.float32), {}

    def step(self, action):
        """
        1) Pop one arrival
        2) Attempt allocation (penalize on failure)
        3) Enqueue completion for logging
        4) Peek next arrival for next obs or end episode
        """
        # 1) arrival
        ev = self._pop_request_event()
        task = ev['args'][0]

        # 2) allocate + reward
        alloc = self.sim.allocate_resources(task, action, residual=True)
        if alloc is None:
            # allocation failed → big negative penalty
            reward = - float(task.max_latency)
            info = {'latency': None, 'energy': None, 'failure': True}
        else:
            lat, eng = alloc['total_latency'], alloc['total_energy']
            reward = - (lat + eng) / 2.0
            info = {'latency': lat, 'energy': eng}

        # 3) enqueue completion if alloc succeeded
        if alloc is not None:
            self.sim._enqueue(self.sim.time + alloc['total_latency'],
                              '_handle_completion',
                              alloc)

        # 4) next observation or done
        next_ev = self._peek_request_event()
        if next_ev is None:
            done = True
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            done = False
            obs = self.sim.get_resources_dnn(next_ev['args'][0])

         # Gymnasium requires: obs, reward, terminated, truncated, info
        terminated = done
        truncated  = False
        return (
            obs.astype(np.float32),
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        print(f"[Time={self.sim.time:.2f}] Queue={len(self.sim.event_queue)}")

    def close(self):
        pass
