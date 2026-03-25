# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Learning rate schedules for RL training."""

from typing import Callable


# ============================================================================
# Linear Learning Rate Decay Schedule (for breaking through 30% plateau)
# ============================================================================

def linear_schedule_with_warmup(
    initial_value: float,
    final_value: float = 1e-5,
    warmup_ratio: float = 0.1
) -> Callable[[float], float]:
    """Linear learning rate decay scheduler with warmup.

    Args:
        initial_value: Initial learning rate (e.g., 3e-4)
        final_value: Final learning rate (e.g., 1e-5)
        warmup_ratio: Warmup ratio (0.1 = 10% of steps for warmup)

    Returns:
        schedule_fn: Function that takes training progress (0.0 to 1.0) and returns learning rate

    Example:
        >>> schedule = linear_schedule_with_warmup(3e-4, 1e-5, 0.1)
        >>> lr = schedule(0.5)  # Returns LR at 50% progress
    """
    def schedule(progress: float) -> float:
        # Warmup phase: linear increase
        if progress < warmup_ratio:
            return initial_value * (progress / warmup_ratio)
        # Decay phase: linear decay to final value
        remaining_progress = (progress - warmup_ratio) / (1.0 - warmup_ratio)
        return initial_value + (final_value - initial_value) * remaining_progress

    return schedule


# ============================================================================
# Learning Rate Schedule Configuration
# ============================================================================

DEFAULT_LR_INITIAL_VALUE = 3e-4  # Initial learning rate
DEFAULT_LR_FINAL_VALUE = 1e-5    # Final learning rate
DEFAULT_WARMUP_RATIO = 0.1       # Warmup ratio (10%)
ENABLE_LR_DECAY = True           # Enable learning rate decay
