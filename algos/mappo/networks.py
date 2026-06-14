# algos/mappo/networks.py
import torch
import torch.nn as nn

class Actor(nn.Module):
    """Local policy — each drone only sees its own 29-dim observation"""
    def __init__(self, obs_dim=29, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        mean     = self.net(obs)
        std      = self.log_std.exp()
        dist     = torch.distributions.Normal(mean, std)
        action   = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob

    def evaluate(self, obs, action):
        mean     = self.net(obs)
        std      = self.log_std.exp()
        dist     = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy  = dist.entropy().sum(-1)
        return log_prob, entropy


class CentralCritic(nn.Module):
    """Centralised value function — sees ALL agents' observations concatenated"""
    def __init__(self, obs_dim=29, n_agents=10, hidden=256):
        super().__init__()
        global_dim = obs_dim * n_agents
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),      nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, global_obs):
        return self.net(global_obs)