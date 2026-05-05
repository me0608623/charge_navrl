"""Obstacle agent 超參數集中定義。"""

# ═══════════════════════════════════════════════════════════════════════════
# 觀測空間 (per-obstacle, env local frame)
# ═══════════════════════════════════════════════════════════════════════════
# [0:2]  own_local_xy   — 障礙物在 env 中的局部座標
# [2:4]  own_vel        — 障礙物當前速度 (vx, vy)
# [4:6]  robot_rel_xy   — robot 相對於障礙物的位置
# [6:8]  robot_vel      — robot 速度 (vx, vy)
# [8]    d_wall         — 到最近邊界的距離
OBS_DIM = 9

# ═══════════════════════════════════════════════════════════════════════════
# 動作空間
# ═══════════════════════════════════════════════════════════════════════════
# [0]  vx — X 方向速度命令 ∈ [-1, 1]
# [1]  vy — Y 方向速度命令 ∈ [-1, 1]
# 實際速度 = action × speed_limit，L2 norm clamp 到 speed_limit
ACT_DIM = 2

# ═══════════════════════════════════════════════════════════════════════════
# 預設超參數
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    # 網路
    "hidden_dim": 128,              # FC 隱藏層維度
    "num_layers": 2,                # 隱藏層數
    "value_hidden": [128, 64],      # Value head 層級

    # 物理
    "speed_limit": 0.8,             # 最大速度 (m/s)，由 curriculum obstacle_speed 覆蓋
    "dt": 0.2,                      # 決策間隔 (s)
    "bound_limit": 7.0,             # 活動範圍 fallback (由 _scene_bounds 覆蓋)

    # 訓練
    "max_active": 10,               # 同時控制的障礙物上限
    "lr": 3e-4,                     # Obstacle PPO learning rate
    "ppo_epochs": 4,
    "clip_eps": 0.2,
    "gamma": 0.99,
    "gae_lambda": 0.95,

    # Reward mode
    "reward_mode": "zero",          # "zero" | "approach" | "intercept"
    "intercept_lookahead": 3.0,     # intercept 模式: 預測幾步後的 charge 位置
}
