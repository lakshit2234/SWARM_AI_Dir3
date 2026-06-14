# algos/mappo/env_wrapper.py
import supersuit as ss
from env.swarm_env import SwarmEnv
import numpy as np

class SwarmParallelEnv:
    """
    Thin wrapper that makes SwarmEnv look like a PettingZoo ParallelEnv.
    Each agent is addressed by string id: '0', '1', ..., 'n-1'.
    """
    def __init__(self, config):
        self.env    = SwarmEnv(config)
        self.n      = config.get('n_agents', 10)
        self.agents = [str(i) for i in range(self.n)]
        self.observation_spaces = {a: self.env.observation_space for a in self.agents}
        self.action_spaces      = {a: self.env.action_space      for a in self.agents}

    def reset(self, seed=None):
        obs_arr, info = self.env.reset(seed=seed)
        return {a: obs_arr[int(a)] for a in self.agents}, info

    def step(self, actions_dict):
        arr = np.stack([actions_dict[a] for a in self.agents])
        obs_arr, rew_arr, done, _, info = self.env.step(arr)
        obs   = {a: obs_arr[int(a)]        for a in self.agents}
        rews  = {a: float(rew_arr[int(a)]) for a in self.agents}
        dones = {a: done                   for a in self.agents}
        return obs, rews, dones, {a: False for a in self.agents}, info