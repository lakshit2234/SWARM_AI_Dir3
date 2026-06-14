# algos/qmix/replay_buffer.py

import numpy as np


class ReplayBuffer:
    """
    Stores past (obs, action, reward, next_obs, done) transitions.

    QMIX is off-policy: it trains on random samples from this
    buffer rather than the most recent rollout. This breaks
    the correlation between consecutive training samples and
    makes learning much more stable.

    Stores TEAM transitions — all agents' data per timestep.
    """

    def __init__(
        self,
        capacity:  int = 5000,
        n_agents:  int = 6,
        obs_dim:   int = 29,
        n_actions: int = 5,
    ):
        self.capacity  = capacity
        self.n_agents  = n_agents
        self.ptr       = 0      # write pointer
        self.size      = 0      # current fill level

        # Pre-allocate arrays — faster than a list of dicts
        self.obs      = np.zeros((capacity, n_agents, obs_dim),  dtype=np.float32)
        self.next_obs = np.zeros((capacity, n_agents, obs_dim),  dtype=np.float32)
        self.actions  = np.zeros((capacity, n_agents),           dtype=np.int64)
        self.rewards  = np.zeros((capacity, n_agents),           dtype=np.float32)
        self.dones    = np.zeros(capacity,                        dtype=np.float32)

    def add(
        self,
        obs:      np.ndarray,   # [n_agents, obs_dim]
        next_obs: np.ndarray,   # [n_agents, obs_dim]
        actions:  np.ndarray,   # [n_agents]
        rewards:  np.ndarray,   # [n_agents]
        done:     bool,
    ):
        self.obs[self.ptr]      = obs
        self.next_obs[self.ptr] = next_obs
        self.actions[self.ptr]  = actions
        self.rewards[self.ptr]  = rewards
        self.dones[self.ptr]    = float(done)

        # Circular buffer — overwrite oldest when full
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 64) -> dict:
        idx = np.random.randint(0, self.size, batch_size)
        return {
            "obs":      self.obs[idx],        # [B, n_agents, obs_dim]
            "next_obs": self.next_obs[idx],   # [B, n_agents, obs_dim]
            "actions":  self.actions[idx],    # [B, n_agents]
            "rewards":  self.rewards[idx],    # [B, n_agents]
            "dones":    self.dones[idx],      # [B]
        }

    def __len__(self) -> int:
        return self.size