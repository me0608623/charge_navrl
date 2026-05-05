"""obstacle_agent — Learned obstacle policy (Warp Drive 移植)

障礙物不是 scripted，而是一個獨立的 learned agent：
  - 與 charge agent 交替訓練 (PPO)
  - Parameter shared across all N obstacles
  - 用途: 生成自然、對抗性的動態障礙物行為

用法:
    from obstacle_agent import (
        ObstaclePolicyFC, ObstacleValueFC,
        build_obstacle_obs, apply_obstacle_actions, compute_obstacle_reward,
        OBS_DIM, ACT_DIM, DEFAULT_CONFIG,
    )
"""

from .config import OBS_DIM, ACT_DIM, DEFAULT_CONFIG
from .model import ObstaclePolicyFC, ObstacleValueFC
from .observation import build_obstacle_obs
from .action import apply_obstacle_actions
from .reward import compute_obstacle_reward
