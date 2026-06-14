# # algos/qmix/networks.py

# import torch
# import torch.nn as nn


# class AgentQNetwork(nn.Module):
#     """
#     Individual Q-network for one drone.

#     Input:  29-dim local observation
#     Output: Q-value for each of 5 discrete actions
#             (0=stay, 1=up, 2=down, 3=left, 4=right)

#     Why discrete actions for QMIX?
#     QMIX's mixer needs a single scalar Q-value per agent
#     (the Q of the chosen action). Discrete actions make
#     this clean — just index into the output vector.
#     """

#     def __init__(self, obs_dim: int = 29, n_actions: int = 5, hidden: int = 64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(obs_dim, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, hidden),
#             nn.ReLU(),
#             nn.Linear(hidden, n_actions),
#         )

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         """
#         obs: shape [batch, obs_dim]
#         returns: shape [batch, n_actions]  — Q-value per action
#         """
#         return self.net(obs)


# class QMIXMixer(nn.Module):
#     """
#     Combines N individual agent Q-values into one joint Q-value.

#     CRITICAL RULE: mixing weights must always be non-negative.
#     This is enforced by passing hypernetwork outputs through
#     torch.abs(). Without this, QMIX loses its theoretical
#     guarantee and the training becomes unstable.

#     How it works:
#     1. Hypernetworks take the GLOBAL state and produce weights
#     2. Those weights combine the per-agent Q-values
#     3. Result: a single team Q-value
#     """

#     def __init__(
#         self,
#         n_agents: int = 6,
#         state_dim: int = 174,   # 29 * 6 agents concatenated
#         embed_dim: int = 32,
#     ):
#         super().__init__()
#         self.n_agents  = n_agents
#         self.embed_dim = embed_dim

#         # Hypernetwork 1: produces first-layer mixing weights
#         # Uses abs() in forward() — weights MUST be non-negative
#         self.hyper_w1 = nn.Sequential(
#             nn.Linear(state_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, n_agents * embed_dim),
#         )
#         self.hyper_b1 = nn.Linear(state_dim, embed_dim)

#         # Hypernetwork 2: produces second-layer mixing weights
#         self.hyper_w2 = nn.Sequential(
#             nn.Linear(state_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, embed_dim),
#         )
#         self.hyper_b2 = nn.Sequential(
#             nn.Linear(state_dim, embed_dim),
#             nn.ReLU(),
#             nn.Linear(embed_dim, 1),
#         )

#     def forward(
#         self,
#         agent_qs: torch.Tensor,   # [batch, n_agents]
#         states:   torch.Tensor,   # [batch, state_dim]
#     ) -> torch.Tensor:
#         """
#         Returns joint Q-value: shape [batch, 1]
#         """
#         B = agent_qs.size(0)

#         # ── Layer 1 ──────────────────────────────────────
#         # abs() enforces non-negative weights — DO NOT REMOVE
#         w1 = torch.abs(self.hyper_w1(states))
#         w1 = w1.view(B, self.n_agents, self.embed_dim)
#         b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)

#         # [batch, 1, n_agents] x [batch, n_agents, embed] = [batch, 1, embed]
#         h = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

#         # ── Layer 2 ──────────────────────────────────────
#         w2 = torch.abs(self.hyper_w2(states)).view(B, self.embed_dim, 1)
#         b2 = self.hyper_b2(states).view(B, 1, 1)

#         # [batch, 1, embed] x [batch, embed, 1] = [batch, 1, 1]
#         q_total = torch.bmm(h, w2) + b2

#         return q_total.view(B, 1)   # [batch, 1]





# algos/qmix/networks.py
# Replace the entire file

import torch
import torch.nn as nn


class AgentQNetwork(nn.Module):
    """
    Individual Q-network. Bigger hidden layer (128) for 9-action space.
    Uses LayerNorm for training stability — helps when observations
    have mixed scales (positions 0-1, velocities -1 to 1, etc.)
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


class QMIXMixer(nn.Module):
    """
    Monotonic mixer with bigger embed_dim (64 default).
    abs() on hypernetwork outputs enforces non-negative weights — never remove this.
    """
    def __init__(self, n_agents: int = 6, state_dim: int = 174, embed_dim: int = 64):
        super().__init__()
        self.n_agents  = n_agents
        self.embed_dim = embed_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        B  = agent_qs.size(0)
        w1 = torch.abs(self.hyper_w1(states)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(B, 1, self.embed_dim)
        h  = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(B, 1, 1)
        return torch.bmm(h, w2).add(b2).view(B, 1)