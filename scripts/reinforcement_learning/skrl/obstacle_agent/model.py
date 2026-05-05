"""Obstacle agent 網路定義 — Policy + Value。

架構: FC (parameter shared across all N obstacles)
WD 對照: obstacle 用 FullyConnected (256×256)，我們用 128×128 (較小場景)
"""

import torch
import torch.nn as nn

from .config import OBS_DIM, ACT_DIM


class ObstaclePolicyFC(nn.Module):
    """Obstacle agent FC policy — parameter shared across all N obstacles.

    Input:  9D per-obstacle observation
    Output: 2D continuous velocity (vx, vy)，由 tanh 限制到 [-1, 1]
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        Args:
            obs: [B, 9] per-obstacle observation (B = num_envs * N_active)
        Returns:
            mean: [B, 2] action mean ∈ [-1, 1]
            std:  [B, 2] action std
        """
        h = self.net(obs)
        mean = torch.tanh(self.mean_head(h))
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def sample(self, obs: torch.Tensor):
        """Sample action + compute log_prob for PPO."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        """Recompute log_prob + entropy for stored actions (PPO update)."""
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class ObstacleValueFC(nn.Module):
    """Obstacle agent value head."""

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # [B, 1]
