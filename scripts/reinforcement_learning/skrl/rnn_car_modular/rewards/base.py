"""Base protocol for reward modules."""

from __future__ import annotations

from typing import Protocol

import torch


class RewardModule(Protocol):
    """Protocol that all reward modules must satisfy.

    Returns (total_reward, breakdown_dict) where breakdown_dict contains
    per-component tensors for WandB logging and diagnostics.
    """

    name: str

    def compute(
        self,
        env_unwrapped: object,
        actions: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        context: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward.

        Args:
            env_unwrapped: The unwrapped IsaacLab environment.
            actions: [N, action_dim] tensor of agent actions.
            terminated: [N] or [N,1] termination flags.
            truncated: [N] or [N,1] truncation flags.
            context: Additional per-step context (phase config, etc.).

        Returns:
            reward: [N] total reward tensor.
            breakdown: dict of named [N] tensors for logging.
        """
        ...

    def update_params(self, curriculum_info: dict) -> None:
        """Update reward parameters from curriculum phase sync.

        Called once per iteration when curriculum info changes (e.g. penalty_hit,
        reward_get_goal, cost_operate). Implementations should silently ignore
        unknown keys.
        """
        ...
