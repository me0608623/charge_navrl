"""Obstacle agent reward 計算。

設計目標: 讓障礙物學會「攔截 charge 的前進方向」，而不是單純追尾。

三種模式:
  - "zero":      WD 原始 — 不給 reward（obstacle 隨機走）
  - "approach":  弱對抗 — 離 robot 越近越好（追尾行為）
  - "intercept": 攔截行為 — 移動到 charge 即將前進的位置（推薦）

intercept 設計原理:
  charge 的 velocity 向量指向它「下一步會去的地方」。
  obstacle 如果能提前到那個位置等著，就能有效阻擋。

  predicted_pos = robot_rel_xy + robot_vel * lookahead_steps
  → 這是 obstacle 應該要去的「攔截點」
  → reward 鼓勵 obstacle 靠近這個攔截點
  → 額外 bonus: 如果 obstacle 正在往攔截點的方向移動
"""

import torch


def compute_obstacle_reward(
    env_unwrapped,
    obs_obs: torch.Tensor,
    max_obstacles: int,
    mode: str = "zero",
    intercept_lookahead: float = 3.0,
) -> torch.Tensor:
    """計算 per-obstacle reward。

    Args:
        env_unwrapped: Isaac Lab env (unused in zero mode)
        obs_obs: [num_envs, N, 9] obstacle observations
            [0:2] own_xy, [2:4] own_vel, [4:6] robot_rel, [6:8] robot_vel, [8] d_wall
        max_obstacles: N
        mode:
            "zero"      — 不給 reward，obstacle 完全隨機（WD 原始）
            "approach"  — 追尾：離 robot 越近獎勵越高
            "intercept" — 攔截：移動到 charge 即將前進的位置
        intercept_lookahead: 預測幾步後的 charge 位置 (步數，dt=0.2s)

    Returns:
        [num_envs * N] flat reward tensor
    """
    num_envs = obs_obs.shape[0]
    N = max_obstacles
    device = obs_obs.device

    if mode == "zero":
        return torch.zeros(num_envs * N, device=device)

    # 共用觀測解析
    robot_rel = obs_obs[:, :, 4:6]      # [E, N, 2] charge 相對於 obstacle 的位置
    robot_vel = obs_obs[:, :, 6:8]      # [E, N, 2] charge 的速度向量
    obs_vel = obs_obs[:, :, 2:4]        # [E, N, 2] obstacle 自己的速度
    d_wall = obs_obs[:, :, 8]           # [E, N] 到邊界距離

    if mode == "approach":
        # 追尾模式: 離 charge 越近越好
        dist_to_robot = robot_rel.norm(dim=-1)  # [E, N]
        reward = 0.1 * (2.0 - dist_to_robot).clamp(0.0, 2.0) / 2.0
        speed = obs_vel.norm(dim=-1)
        reward = reward - 0.01 * speed
        return reward.reshape(-1)

    # ═══════════════════════════════════════════════════════════════════
    # "intercept" 模式: 攔截 charge 的前進路徑
    # ═══════════════════════════════════════════════════════════════════

    # 1. 預測 charge 未來位置 (相對於 obstacle)
    #    predicted_pos = 現在的相對位置 + charge速度 × lookahead
    #    意義: 「charge 再走幾步後，會在我的哪個方向」
    predicted_pos = robot_rel + robot_vel * intercept_lookahead  # [E, N, 2]
    intercept_dist = predicted_pos.norm(dim=-1)  # [E, N]

    # 2. 攔截距離獎勵: 離預測位置越近越好
    #    最大獎勵 0.15 (距離=0)，距離 3m 以上無獎勵
    reward_intercept = 0.15 * (3.0 - intercept_dist).clamp(0.0, 3.0) / 3.0  # [E, N]

    # 3. 方向對齊 bonus: obstacle 正在往攔截點方向移動嗎？
    #    計算 obstacle 速度方向 與 「通往攔截點方向」的 cosine similarity
    obs_speed = obs_vel.norm(dim=-1, keepdim=True).clamp(min=1e-6)  # [E, N, 1]
    obs_heading = obs_vel / obs_speed  # [E, N, 2] 歸一化速度方向

    # 通往攔截點的方向 (從 obstacle 看，預測位置就是目標)
    to_intercept = predicted_pos  # 已經是相對座標
    to_intercept_norm = to_intercept.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    to_intercept_dir = to_intercept / to_intercept_norm  # [E, N, 2]

    # cos(θ) = dot(heading, target_dir)，∈ [-1, 1]
    alignment = (obs_heading * to_intercept_dir).sum(dim=-1)  # [E, N]
    # 只獎勵正方向 (obstacle 正在靠近攔截點)
    reward_alignment = 0.05 * alignment.clamp(min=0.0)  # [E, N]

    # 4. 靠近邊界懲罰: 避免 obstacle 被擠到角落
    wall_penalty = -0.02 * (1.0 - d_wall.clamp(0.0, 2.0) / 2.0)  # [E, N]

    # 合計
    reward = reward_intercept + reward_alignment + wall_penalty

    return reward.reshape(-1)  # [E*N]
