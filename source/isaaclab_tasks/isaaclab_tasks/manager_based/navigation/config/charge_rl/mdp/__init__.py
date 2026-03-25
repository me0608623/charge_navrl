# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
導航任務的 MDP（馬可夫決策過程）組件模組 (MDP Components for Navigation Task)

此子模組包含導航任務特定的 MDP 組件，包括：
- 預訓練策略動作：實現分層控制架構，使用預訓練的低層級策略執行動作
- 獎勵函數：定義位置追蹤和方向追蹤的獎勵計算

這些組件與 isaaclab.envs.mdp 中的通用 MDP 組件（如命令生成、觀測計算等）配合使用。
"""

# 導入 Isaac Lab 環境的通用 MDP 組件
# 包括：命令生成器、觀測計算函數、終止條件檢查等
from isaaclab.envs.mdp import *  # noqa: F401, F403

# 導入導航任務特定的 MDP 組件
from .pre_trained_policy_action import *  # noqa: F401, F403  # 預訓練策略動作項
from .rewards import *  # noqa: F401, F403  # 獎勵函數（位置追蹤、方向追蹤）
