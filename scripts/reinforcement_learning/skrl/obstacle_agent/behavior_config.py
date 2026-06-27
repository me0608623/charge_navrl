"""Rule-Based Obstacle Behavior 參數集中定義。

所有 8 種 behavior 的物理參數 + safety constraints 在此管理。
Curriculum stage 可透過 overrides 覆寫特定參數。

物理基準:
  robot_radius = 0.35m
  obs_radius = 0.20~0.35m
  dt = 0.2s
  v_max_robot = 1.0 m/s
  scene = 20×20m (effective interior 18×18m)
  LiDAR = 72 bins, 5°/bin, 360°, r_max=20m
"""

from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════════
# Behavior Type Enum (用 int 方便 tensor 操作)
# ═══════════════════════════════════════════════════════════════════════════

BEHAVIOR_INACTIVE = 0
BEHAVIOR_STATIC = 1
BEHAVIOR_PATROL = 2
BEHAVIOR_RANDOM_WALK = 3
BEHAVIOR_HORIZONTAL_CROSSING = 4
BEHAVIOR_PATH_CROSSING = 5
BEHAVIOR_NEAR_MISS = 6
BEHAVIOR_CORRIDOR_CROSSING = 7
BEHAVIOR_OCCLUSION = 8
BEHAVIOR_HEAD_ON = 9  # 直線迎面：spawn 後對準 robot 當下位置直直衝過來 (訓練提早避讓)

BEHAVIOR_NAMES = {
    0: "inactive",
    1: "static",
    2: "patrol",
    3: "random_walk",
    4: "horizontal_crossing",
    5: "path_crossing",
    6: "near_miss",
    7: "corridor_crossing",
    8: "occlusion",
    9: "head_on",
}

NUM_BEHAVIOR_TYPES = 10  # 0~9


# ═══════════════════════════════════════════════════════════════════════════
# Safety Constraints — 所有 behavior 必須遵守
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SafetyConstraints:
    """全域安全約束，spawn 和 step 時都要檢查。"""

    # Spawn 距離限制
    min_spawn_from_robot: float = 2.0       # obstacle 生成時離 robot 至少 2m
    min_spawn_from_goal: float = 1.5        # obstacle 生成時離 goal 至少 1.5m
    min_wall_clearance: float = 0.6         # obstacle 離牆壁至少 0.6m (obs_r + buffer)
    min_obs_obs_distance: float = 0.8       # obstacle 之間至少 0.8m

    # Runtime 限制
    max_blocking_goal_line_steps: int = 25  # 在 robot→goal 連線上最多停留 25 步 (5s)
    reset_timeout_steps: int = 750          # 750 步 (150s) 未到目標強制 respawn (bug catcher)

    # Occlusion 專用
    min_visible_frames_before_occlusion: int = 8  # 被遮擋前至少可見 8 frames (1.6s)
    max_position_drift_on_reveal: float = 0.5     # 重現時位置偏差上限 0.5m


# ═══════════════════════════════════════════════════════════════════════════
# Per-Behavior Configs
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StaticConfig:
    """靜態障礙物 — 提供 stationary LiDAR baseline。

    用途: 讓 RNN 區分 persistent geometry (不變回波) 與 dynamic temporal changes (變化回波)。
    即 RNN 學會「這個回波不隨時間變 = 不需要時序追蹤」。
    """
    radius_range: tuple[float, float] = (0.20, 0.35)
    # Safety override: static 可以離 robot 近一點 (因為不會動)
    min_spawn_from_robot: float = 1.5


@dataclass
class PatrolConfig:
    """兩點/多點巡邏 — 可預測週期性運動。

    用途: 讓 RNN 學習週期性 LiDAR 信號 (距離先減後增再減)。
    Policy 學會「等待 obstacle 遠離後通過」。
    """
    num_waypoints_range: tuple[int, int] = (2, 4)     # 2~4 個巡邏點
    waypoint_spread_max: float = 8.0                  # 巡邏點之間最大距離 (m)
    speed_range: tuple[float, float] = (0.3, 0.6)    # 巡邏速度 (m/s)
    pause_steps_range: tuple[int, int] = (0, 5)       # 到達 waypoint 後暫停 0~5 步 (0~1s)
    ping_pong_ratio: float = 0.5                      # 50% ping-pong / 50% loop


@dataclass
class RandomWalkConfig:
    """隨機漫步 — 不可預測運動。

    用途: 迫使 RNN 學短期趨勢外推 (不能靠記住 pattern)。
    互補 patrol 的週期性。
    """
    speed_range: tuple[float, float] = (0.2, 0.7)
    direction_change_interval_range: tuple[int, int] = (5, 25)  # 每 1~5s 改變方向
    max_turn_angle: float = 1.5708                               # 最大轉角 pi/2 (90°)
    smooth_turn_steps: int = 3                                   # 轉向平滑過渡 3 步 (0.6s)


@dataclass
class HorizontalCrossingConfig:
    """水平穿越 robot 前方 — 模擬行人橫穿。

    用途: 訓練 RNN 偵測「突然出現在視野邊緣、快速橫掃」的 LiDAR 變化。
    真實世界最常見的動態避障情境。
    """
    speed_range: tuple[float, float] = (0.5, 1.0)
    offset_ahead_range: tuple[float, float] = (2.0, 6.0)    # 在 robot 前方多遠穿越
    crossing_width: float = 12.0                              # 穿越總寬度 (m)
    spawn_side_offset_range: tuple[float, float] = (3.0, 6.0)  # 從 robot 側方多遠 spawn
    repeat_interval_range: tuple[int, int] = (30, 60)         # 穿越完成後等 6~12s 再次
    min_spawn_from_robot: float = 3.0


@dataclass
class PathCrossingConfig:
    """穿越 robot→goal 連線 — 直接阻擋前進方向。

    用途: 訓練 policy 判斷「等它過」vs「繞行」。
    RNN 學習前方逼近回波的時間預測。
    """
    speed_range: tuple[float, float] = (0.4, 0.9)
    crossing_angle_range: tuple[float, float] = (0.5236, 2.6180)  # 30°~150°
    ttc_to_path_range: tuple[float, float] = (1.5, 4.0)           # obstacle 到路徑的 TTC
    spawn_perp_distance_range: tuple[float, float] = (2.0, 5.0)   # 離路徑多遠 spawn
    min_distance_ahead: float = 2.5                                 # 必須在 robot 前方
    activation_delay_range: tuple[int, int] = (10, 25)             # robot 開始動後延遲 2~5s 激活
    min_spawn_from_robot: float = 3.5


@dataclass
class NearMissConfig:
    """極近距離擦過 — 訓練精細距離判斷。

    設計保證:
    - 軌跡在 spawn 時預計算，不依賴 robot 動作
    - 即使 robot 完全不動/不閃，obstacle 也不會碰撞
    - 防止 robot 學到「obstacle 會自己讓路」

    用途: 訓練 RNN 區分「需緊急迴避」vs「僅經過不需反應」。
    """
    # Clearance bucket 分佈 (surface distance, 不是 center distance)
    clearance_tight: tuple[float, float] = (0.10, 0.20)   # 表面間距 10~20cm
    clearance_medium: tuple[float, float] = (0.20, 0.40)  # 表面間距 20~40cm
    clearance_loose: tuple[float, float] = (0.40, 0.60)   # 表面間距 40~60cm
    clearance_weights: tuple[float, float, float] = (0.30, 0.50, 0.20)  # tight/med/loose 比例

    # TTC: 從 obstacle 可見到最近接觸點的時間
    ttc_range: tuple[float, float] = (2.0, 5.0)  # seconds
    speed_range: tuple[float, float] = (0.4, 1.0)

    # 每 episode 最多幾次 near-miss
    max_events_per_episode: int = 3
    min_spawn_from_robot: float = 2.5


@dataclass
class CorridorCrossingConfig:
    """通道內移動 — 在牆壁 gap 中製造交通。

    用途: 訓練 policy 在受限空間判斷 timing (何時進入通道)。
    RNN 同時處理牆壁 + obstacle 的複合信號。
    """
    speed_range: tuple[float, float] = (0.3, 0.7)
    min_gap_width: float = 2.5       # corridor 最小寬度 (robot+obs+buffer)
    max_gap_width: float = 5.0
    spawn_offset_range: tuple[float, float] = (1.0, 3.0)  # 在 corridor 端 spawn 偏移
    min_spawn_from_robot: float = 2.5


@dataclass
class OcclusionGroupConfig:
    """多 obstacle 前後遮擋 — 訓練 RNN 時間記憶。

    設計保證:
    - 被遮擋 obstacle 遮擋前可見 >= min_visible_frames
    - 重現位置符合 constant velocity extrapolation (偏差 < 0.5m)
    - 不允許從完全不可觀測位置突然出現

    用途: 訓練 RNN aux target 的 2nd-nearest prediction。
    RNN 必須記住「有東西在後面，即使現在看不到」。
    """
    group_size_range: tuple[int, int] = (2, 3)             # 1 front blocker + 1~2 back
    front_back_spacing_range: tuple[float, float] = (0.8, 1.5)  # front/back 間距
    front_speed_range: tuple[float, float] = (0.2, 0.5)    # blocker 慢速
    back_speed_range: tuple[float, float] = (0.3, 0.8)     # target 正常速
    visible_frames_range: tuple[int, int] = (8, 25)         # 可見階段 1.6~5s
    occluded_frames_range: tuple[int, int] = (5, 20)        # 遮擋階段 1~4s
    position_noise_on_reveal: float = 0.1                   # 重現時小量偏差 (m)
    min_obs_obs_distance: float = 0.5                       # group 內可以較近
    min_spawn_from_robot: float = 2.5


@dataclass
class HeadOnConfig:
    """直線迎面 — spawn 後等待 activation_delay，激活當下對準 robot 位置，
    之後等速直線衝過去 (不再重新瞄準 → 軌跡固定、不依賴 robot 動作)。

    用途: 訓練 policy「有東西直直朝我來」時提早偵測 + 側讓。
    對症 SA4 診斷的「晚反應全速撞動態」(現有 crossing/near_miss 都不正面來)。
    設計刻意只在激活瞬間瞄準一次 (非持續 homing),避免與致動延遲疊成舞龍舞獅極限環。
    """
    speed_range: tuple[float, float] = (0.30, 0.65)        # 迎面速度 (m/s),由 stage speed_overrides 覆寫
    spawn_distance_range: tuple[float, float] = (5.5, 8.0)  # spawn 離場景中心距離 (LiDAR 邊界附近)
    activation_delay_range: tuple[int, int] = (5, 15)       # 激活前等待 1~3s (10steps=2s),讓 robot 先動
    aim_jitter_deg: float = 8.0                             # 瞄準角度抖動 ±8°,避免完全精準(防 overfit)
    travel_cap_mult: float = 2.2                            # 行進超過 spawn_distance×此倍數即完成(穿過離場)


# ═══════════════════════════════════════════════════════════════════════════
# 完整 Config (組合所有 behavior)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BehaviorConfig:
    """所有 behavior 的完整配置。

    用法:
        cfg = BehaviorConfig()              # 全部預設值
        cfg.patrol.speed_range = (0.4, 0.8) # 覆寫特定參數
    """
    safety: SafetyConstraints = field(default_factory=SafetyConstraints)
    static: StaticConfig = field(default_factory=StaticConfig)
    patrol: PatrolConfig = field(default_factory=PatrolConfig)
    random_walk: RandomWalkConfig = field(default_factory=RandomWalkConfig)
    horizontal_crossing: HorizontalCrossingConfig = field(default_factory=HorizontalCrossingConfig)
    path_crossing: PathCrossingConfig = field(default_factory=PathCrossingConfig)
    near_miss: NearMissConfig = field(default_factory=NearMissConfig)
    corridor_crossing: CorridorCrossingConfig = field(default_factory=CorridorCrossingConfig)
    occlusion: OcclusionGroupConfig = field(default_factory=OcclusionGroupConfig)
    head_on: HeadOnConfig = field(default_factory=HeadOnConfig)

    def apply_stage_overrides(self, overrides: dict) -> None:
        """從 curriculum stage config 覆寫參數。

        overrides 格式:
            {"patrol": {"speed_range": (0.4, 0.8)},
             "safety": {"min_spawn_from_robot": 2.5}}
        """
        for section_name, params in overrides.items():
            section = getattr(self, section_name, None)
            if section is None:
                continue
            for key, value in params.items():
                if hasattr(section, key):
                    setattr(section, key, value)
