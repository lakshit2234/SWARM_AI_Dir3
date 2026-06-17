# algos/vdn/networks.py

import torch
import torch.nn as nn


class AgentQNetwork(nn.Module):
    """
    Per-agent Q-network. Identical to QMIX's AgentQNetwork.
    VDN difference: no mixer network needed.
    Joint Q = sum(Q_1, Q_2, ..., Q_n) — done in train loop directly.
    """
    def __init__(self, obs_dim: int = 29, n_actions: int = 9, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)