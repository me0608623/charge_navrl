"""VLP-16 Curriculum 環境配置 — 場景骨架定義

═══════════════════════════════════════════════════════════════════════════
本檔案定義「場景骨架」— 所有 curriculum version 共用的物理場景結構。
不直接決定每個 phase 的內容（那由 curriculum/phases/*.py 控制）。

本檔案負責：
  1. 物理場景 (MySceneCfgVLP16_20x20):
     - 外牆: 20×20m (固定，對齊 WD grid_length_bias=20)
     - 內牆 slots: 8 個不同長度 (2.5~5.0m)，初始隱藏在 Z=-10
       → 由 randomize_walls event 依 curriculum phase 決定啟用哪些
       → target_wall_length 參數控制優先啟用接近目標長度的 slot
     - 障礙物 slots: 100 個 (cuboid/cylinder 混合)，初始隱藏在 Z=-10
       → 注意: train_rnn_car_wdclip.py 不使用 scripted 障礙物移動
       → 障礙物由 learned obstacle policy (ObstaclePolicyFC) 控制
       → max_active_obstacles=10，實際數量由 curriculum phase 決定
     - LiDAR: VLP16, z=1.43m, 72 bins, raycast Wall_.* + Obstacle_.*

  2. 命令初始值 (CommandsCfgVLP16Curriculum):
     - Phase 1 預設: 8 goals, 距離 2-13m
     → curriculum 動態覆蓋 num_goals / goal_distance

  3. 事件初始值 (EventCfgVLP16Curriculum):
     - Phase 1 預設: 100% empty, 0 obstacles, 0-2 walls
     → curriculum 動態覆蓋 obstacle/wall 參數

  4. NavRL 獎勵 (RewardsCfgVLP16NavRL):
     - 6 項 dense rewards (velocity_to_goal, safe_progress, etc.)
     → 注意: train_rnn_car_wdclip.py 完全繞過這些 rewards！
       WD 訓練用自己的 compute_wd_charge_reward() (sparse: +40 goal, -5 collision)
     → 這些 rewards 只對 train_charge_ac.py (NavRL 路線) 有效
     → WD 路線請用 charge_env_cfg_wd_sparse.py (rewards 全歸零)

  5. 課程函式指標 (CurriculumCfgVLP16):
     → 指向 goal_obstacle_curriculum()
     → CLI --curriculum_version 決定使用哪個 phase 定義
     → phase 定義在 curriculum/phases/*.py (一個檔案一個 version)

Obstacle Agent (不在此 cfg 定義):
  - 網路: models/modular_rnn_models.py → ObstaclePolicyFC (FC 128×128, 9D→2D)
  - 訓練: train_rnn_car_wdclip.py (PPO, 與 charge 交替)
  - 觀測: 9D per obstacle (robot相對位置/速度/距離/朝向/active)
  - 動作: continuous (vx, vy) ∈ [-speed_limit, +speed_limit]
═══════════════════════════════════════════════════════════════════════════
"""

import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
import isaaclab.sim as sim_utils
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from .charge_env_cfg_vlp16 import (
    ChargeNavigationEnvCfgVLP16,
    MySceneCfgVLP16,
    RewardsCfgVLP16,
    EventCfgVLP16,
    ROBOT_BODY_RADIUS,
    GOAL_REACH_THRESHOLD,
    COLLISION_THRESHOLD,
)
from ..multi_goal_command import MultiGoalCommandCfg
from ..goal_command import GoalCommandCfg

# 獎勵函數
from ..mdp.rewards import (
    reaching_goal,
)
from ..mdp.rewards.potential_based_rewards import (
    potential_progress_reward,
    exponential_obstacle_penalty,
    collision_terminal_penalty,
    per_step_time_penalty,
    velocity_too_low_penalty,
)
from ..mdp.rewards.smoothness_rewards import (
    deadzone_acceleration_penalty,
    context_aware_angular_velocity_penalty,
)
from ..mdp.rewards.navrl_rewards import (
    velocity_to_goal_reward,
    safety_log_distance_reward,
    safe_progress_reward,
)

# 事件
from ..mdp.events import (
    randomize_obstacles_by_difficulty,
    move_obstacles_vectorized,
    move_goal_positions,
)
from ..mdp.events.reset import reset_root_state_random_safe
from ..mdp.events.walls import init_perenv_walls, randomize_walls
from ..domain_randomization import apply_domain_randomization
from ..mdp.events.state import set_obstacle_metadata
from ..mdp.events import corridor_crossing as _corridor_crossing_mod
from ..mdp.events import long_corridor_replay as _long_corridor_replay_mod
from ..mdp.events import narrow_passage_bridge as _narrow_passage_bridge_mod
from ..mdp.events import previous_stage_replay as _previous_stage_replay_mod

# 課程
from ..curriculum.goal_obstacle_curriculum import goal_obstacle_curriculum

# 牆壁
from ..mdp.wall_layout import WALL_SLOT_SPECS, MAX_WALL_SLOTS, BOUNDARY_WALLS_T_CORRIDOR


# ============================================================================
# 20×20m 場景配置
# ============================================================================
@configclass
class MySceneCfgVLP16_20x20(MySceneCfgVLP16):
    """20×20m 場景 — 4 面外牆 + 8 wall slots（per-env 隨機化）"""

    def __post_init__(self):
        # 呼叫祖父類 __post_init__（跳過 MySceneCfgVLP16 的 16×16 牆壁設定）
        from isaaclab.scene import InteractiveSceneCfg
        InteractiveSceneCfg.__post_init__(self)

        room_size = 10.0  # ±10m = 20×20m
        wall_thickness = 1.0  # 1.0m 厚度（原 0.2m）
        wall_height = 3.0   # ★ 真實牆 3m，高於 VLP16 (z=1.43m)，確保 LiDAR 可見
        wall_length = room_size * 2 + wall_thickness
        wall_color = (0.5, 0.5, 0.5)

        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color, metallic=0.1)

        # 四面外牆（20×20m, 1.0m 厚度）
        self.wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_North",
            spawn=sim_utils.CuboidCfg(size=(wall_length, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, room_size, wall_height / 2)),
        )
        self.wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_South",
            spawn=sim_utils.CuboidCfg(size=(wall_length, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -room_size, wall_height / 2)),
        )
        self.wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_East",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(room_size, 0.0, wall_height / 2)),
        )
        self.wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_West",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-room_size, 0.0, wall_height / 2)),
        )

        # 8 個內部牆壁 slots (RigidObjectCfg — 支援 per-env 移動)
        # 初始隱藏在 Z=-10，由 randomize_walls event 控制位置/可見性
        HIDDEN_Z = -10.0
        internal_wall_color = (0.6, 0.55, 0.5)
        internal_wall_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=internal_wall_color, metallic=0.1,
        )

        # 刪除繼承的 16×16 內部牆 (wall_internal_0..5)
        for i in range(6):
            attr_name = f"wall_internal_{i}"
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        # 建立 8 個 wall slots
        for i, (length, width, height) in enumerate(WALL_SLOT_SPECS):
            setattr(self, f"wall_internal_{i}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Wall_Internal_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=(length, width, height),
                    rigid_props=wall_rigid_props,
                    collision_props=wall_collision_props,
                    visual_material=internal_wall_visual,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z)),
            ))

        # LiDAR: 覆蓋繼承的 mesh_prim_paths
        # Wall_.* 必須 track_mesh_transforms=True（牆壁會移動）
        self.lidar.mesh_prim_paths = [
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/ground", track_mesh_transforms=False,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Wall_.*", track_mesh_transforms=True,
            ),
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Obstacle_.*", track_mesh_transforms=True,
            ),
        ]

        # 100 個障礙物（循環外觀模板，初始隱藏在 Z = -10.0）
        # play --no_curriculum 可自由設定 --num_static/--num_dynamic
        # ★ 行人高度 (1.6~1.8m)：確保 VLP16 LiDAR (z=1.43m, lowest beam -15°) 可觀測
        MAX_OBS = 100
        HIDDEN_Z = -10.0
        _static_tpl = [
            {"type": "cuboid",   "size": (0.5, 0.5, 1.7),   "color": (0.8, 0.2, 0.2)},
            {"type": "cylinder", "radius": 0.3, "height": 1.6, "color": (0.8, 0.8, 0.2)},
            {"type": "cuboid",   "size": (0.7, 0.7, 1.8),   "color": (0.2, 0.4, 0.8)},
            {"type": "cylinder", "radius": 0.25, "height": 1.7, "color": (0.2, 0.8, 0.2)},
            {"type": "cuboid",   "size": (0.6, 0.6, 1.6),   "color": (0.8, 0.2, 0.8)},
            {"type": "cylinder", "radius": 0.35, "height": 1.8, "color": (0.8, 0.5, 0.2)},
            {"type": "cuboid",   "size": (0.4, 0.4, 1.7),   "color": (0.2, 0.8, 0.8)},
            {"type": "cylinder", "radius": 0.2, "height": 1.6, "color": (0.5, 0.5, 0.5)},
            {"type": "cuboid",   "size": (0.55, 0.55, 1.8),  "color": (0.9, 0.9, 0.9)},
            {"type": "cylinder", "radius": 0.28, "height": 1.7, "color": (0.3, 0.3, 0.3)},
        ]
        obstacle_sizes: list[float] = []
        for i in range(MAX_OBS):
            cfg = _static_tpl[i % len(_static_tpl)]
            if cfg["type"] == "cuboid":
                spawn_cfg = sim_utils.CuboidCfg(
                    size=cfg["size"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                size_scalar = max(cfg["size"][0], cfg["size"][1])
            else:
                spawn_cfg = sim_utils.CylinderCfg(
                    radius=cfg["radius"], height=cfg["height"],
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg["color"], metallic=0.2),
                )
                size_scalar = cfg["radius"] * 2.0

            setattr(self, f"obstacle_{i}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{i}",
                spawn=spawn_cfg,
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, HIDDEN_Z)),
            ))
            obstacle_sizes.append(size_scalar)

        set_obstacle_metadata(MAX_OBS, obstacle_sizes)


# ============================================================================
# 課程版命令配置（Stage 1 初始值）
# ============================================================================
@configclass
class CommandsCfgVLP16Curriculum:
    """課程命令配置。

    NOTE: 以下欄位在執行時由 goal_obstacle_curriculum.py 動態覆蓋：
      - num_goals, max_goals, distance, wall_boundary, obstacle_safe_distance 等
    這裡的值僅作為 Stage 1 初始 / fallback 使用。
    """
    goal_command = MultiGoalCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        ranges=GoalCommandCfg.Ranges(
            distance=(2.0, 13.0),
            angle=(-math.pi, math.pi),
        ),
        wall_boundary=9.5,
        wall_safe_margin=0.5,
        obstacle_safe_distance=1.0,
        num_obstacles=0,
        num_goals=8,
        max_goals=10,
    )


# ============================================================================
# 課程版事件配置（Stage 1 初始值）
# ============================================================================
@configclass
class EventCfgVLP16Curriculum:
    """課程事件：20×20m 場景 + Stage 1 初始障礙物分布"""

    # --- Startup events (宣告順序 = 執行順序) ---

    # 初始化 per-env 牆壁數據結構（必須最先）
    init_walls = EventTerm(
        func=init_perenv_walls,
        mode="startup",
        params={},
    )

    # Phase 1 初始分布：100% empty（純導航學習，無障礙物）
    randomize_obstacles_startup = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="startup",
        params={
            "empty_ratio": 1.00, "static_ratio": 0.00, "dynamic_ratio": 0.00,
            "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
            "max_obstacles": 50, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.5, "max_spawn_attempts": 50,
            "boundary": 9.5, "active_obstacle_ratio": 1.0, "debug": False,
        },
    )

    # --- Reset events (宣告順序 = 執行順序) ---
    # 牆壁 → 障礙物 → 機器人（牆壁先就位，障礙物/機器人才能避開）

    randomize_wall_positions = EventTerm(
        func=randomize_walls,
        mode="reset",
        params={
            "min_walls": 0, "max_walls": 2,  # Stage 1 初始值（課程動態調整）
            "boundary": 8.5, "min_wall_spacing": 2.0,
            "max_spawn_attempts": 30, "robot_safe_dist": 1.5,
        },
    )

    randomize_obstacles = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="reset",
        params={
            "empty_ratio": 1.00, "static_ratio": 0.00, "dynamic_ratio": 0.00,
            "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
            "max_obstacles": 50, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.5, "max_spawn_attempts": 50,
            "boundary": 9.5, "active_obstacle_ratio": 1.0, "debug": False,
        },
    )

    reset_base = EventTerm(
        func=reset_root_state_random_safe,
        mode="reset",
        params={
            "pose_range": {
                "x": (-7.0, 7.0),
                "y": (-7.0, 7.0),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.15, 0.15),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_obstacles = None  # 由 randomize_obstacles 處理

    # Fixed SA5 general-scene replay for SA6+. It runs after the ordinary
    # current-stage reset, then corridor/narrow injectors reserve disjoint envs.
    previous_stage_replay = EventTerm(
        func=_previous_stage_replay_mod.setup_previous_stage_replay,
        mode="reset",
        params={
            "fraction": 0.0,
            "static_obstacles": 10,
            "dynamic_obstacles": 3,
            "min_walls": 2,
            "max_walls": 3,
            "wall_length": 4.0,
            "obstacle_boundary": 5.5,
        },
    )

    # Corridor-crossing injector: runs AFTER reset_base so it can override the
    # random robot/goal poses for the selected fraction of envs.
    # fraction=0.0 = baseline (no injection) — byte-for-byte identical to baseline.
    # Curriculum phases can override fraction per stage (e.g. 0.12 for SA4+).
    corridor_crossing = EventTerm(
        func=_corridor_crossing_mod.setup_corridor_crossing,
        mode="reset",
        params={
            "fraction": 0.0,          # 0.0 = baseline (no injection); curriculum overrides per stage
            "half_width": 2.0,
            "corridor_half_len": 3.0,
            "crossing_ahead": 2.4,
            "ped_speed": 0.65,
            "ped_slot": 0,
            "wall_z": 1.5,
        },
    )

    # Deployment-corridor replay. Dedicated side-wall assets are added only by
    # an experiment config; the configured 10 m interaction zone is sealed to
    # the room boundary. fraction=0 keeps the baseline scene unchanged.
    long_corridor_replay = EventTerm(
        func=_long_corridor_replay_mod.setup_long_corridor_replay,
        mode="reset",
        params={
            "fraction": 0.0,
            "free_width": 4.0,
            "length": 10.0,
            "static_obstacles": 4,
            "dynamic_obstacles": 2,
            "dynamic_speed_min": 0.30,
            "dynamic_speed_max": 0.60,
            "dynamic_motion_mode": "lateral",
            "wall_z": 1.5,
        },
    )

    # SA5 warm-start bridge injector. The default fraction=0.0 is a strict no-op;
    # an experiment config must add the two bridge wall assets and enable it.
    narrow_passage_bridge = EventTerm(
        func=_narrow_passage_bridge_mod.setup_narrow_passage_bridge,
        mode="reset",
        params={
            "fraction": 0.0,
            "schedule_steps": 19200,
            "room_half_extent": 7.5,
            "boundary_wall_width": 1.0,
            "segment_length": 9.0,
            "gap_center_limit": 1.0,
            "barrier_x_limit": 0.5,
            "start_goal_distance": 3.0,
            # Play-side manual overrides. None/defaults here == the training
            # distribution; only --narrow_replay_* on play_rnn_car.py sets them.
            "gap_center_range": None,
            "barrier_x_range": None,
            "direction_mode": "random",
            "goal_distance": None,
            "goal_lateral_offset": 0.0,
            # N1+ may override these ranges. None preserves historical replay.
            "goal_distance_range": None,
            "goal_lateral_offset_range": None,
            "final_stress_ratio": 0.25,
            "fixed_width_range": None,
            "fixed_yaw_limit_deg": None,
            "exact_width": None,
            "exact_width_ratio": 0.0,
            "wall_z": 1.5,
        },
    )

    domain_randomization = EventTerm(
        func=apply_domain_randomization,
        mode="reset",
        params={"enable_physics": True, "enable_sensor_noise": True, "enable_external_force": True},
    )

    move_dynamic_obstacles = EventTerm(
        func=move_obstacles_vectorized,
        mode="interval",
        interval_range_s=(0.2, 0.2),
        params={
            "move_dt": 0.2, "speed_min": 0.3, "speed_max": 1.2,
            "goal_reach_threshold": 0.5, "speed_resample_steps": 10,
            "area_limit": 8.0, "max_obstacles": 50, "bound_limit": 9.0,
        },
    )

    # Goal 隨機移動（預設 speed=0 關閉，由 curriculum phase 動態啟用）
    move_goal = EventTerm(
        func=move_goal_positions,
        mode="interval",
        interval_range_s=(0.2, 0.2),
        params={
            "goal_move_speed": 0.0,
            "goal_move_max_radius": 3.0,
            "goal_move_behavior": "random_walk",
            "goal_move_angular_speed": 0.5,
            "goal_move_dt": 0.2,
            "goal_move_wall_margin": 0.5,
            "goal_move_obs_margin": 0.8,
            "goal_move_dir_steps_min": 15,
            "goal_move_dir_steps_max": 40,
        },
    )

    # Runs after move_goal and pins only active bridge envs to the exact target
    # across the barrier. Baseline and ordinary SA5 envs are untouched.
    maintain_narrow_passage_goal = EventTerm(
        func=_narrow_passage_bridge_mod.maintain_narrow_passage_goal,
        mode="interval",
        interval_range_s=(0.2, 0.2),
        params={},
    )

    maintain_long_corridor_goal = EventTerm(
        func=_long_corridor_replay_mod.maintain_long_corridor_goal,
        mode="interval",
        interval_range_s=(0.2, 0.2),
        params={},
    )


# ============================================================================
# 課程版獎勵配置（改進平滑懲罰）
# ============================================================================
@configclass
class RewardsCfgVLP16Curriculum(RewardsCfgVLP16):
    """v9 期望值驅動獎勵 — 只保留核心獎勵 + 死亡機制

    設計原則（基於期望值理論）：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 不要為每個次要能力設計獨立的加減分來干擾 Agent。                  │
    │ 確立唯一核心 Reward（到達目標），利用死亡機制讓避障成為            │
    │ 獲取核心獎勵的「必經手段」。                                     │
    └─────────────────────────────────────────────────────────────────┘

    保留（核心驅動力）：
    - reaching_goal (+250): 唯一核心獎勵
    - potential_progress (+60): PBRS 密集引導，數學上不改變最優策略

    歸零（由 RL 機制自然處理）：
    - collision_terminal → 0: 死亡機制已夠（terminal = 失去未來期望值）
    - near_obstacle_penalty → 0: 死亡機制已夠
    - time_penalty → 0: γ=0.995 自然提供時間壓力
    - velocity_too_low → 0: γ + progress reward 自然驅動前進

    保留微量（sim-to-real 平滑）：
    - acceleration_penalty (-0.05): 保護馬達
    - angular_velocity_penalty (-0.05): 平滑控制
    """

    # --- 歸零的懲罰項（由 RL 機制自然處理）---

    # 碰撞懲罰 → 0: 死亡機制 = 失去所有未來獎勵（V ≈ 50-100）
    # 這比任何固定懲罰都更有效，且自動根據 episode 進度調整
    collision_terminal = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=0.0,
    )

    # 障礙物接近懲罰 → 0: 死亡機制自動提供避障動機
    near_obstacle_penalty = RewTerm(
        func=exponential_obstacle_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "safe_distance": 1.0, "steepness": 6.0},
        weight=0.0,
    )

    # 時間懲罰 → 0: γ=0.995 下，20步路徑保留90%價值，60步只剩74%
    time_penalty = RewTerm(
        func=per_step_time_penalty,
        params={},
        weight=0.0,
    )

    # 靜止懲罰 → 0: agent 必須前進才能獲得 potential_progress 和 reaching_goal
    velocity_too_low = RewTerm(
        func=velocity_too_low_penalty,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=0.0,
    )

    # --- 保留微量平滑（sim-to-real）---

    # 加速度懲罰：降低到 -0.05（原 -0.15），只做最低限度的馬達保護
    acceleration_penalty = RewTerm(
        func=deadzone_acceleration_penalty,
        params={"dead_zone": 0.1},
        weight=-0.05,
    )

    # 角速度懲罰：降低到 -0.05，靠近障礙物時允許急轉
    angular_velocity_penalty = RewTerm(
        func=context_aware_angular_velocity_penalty,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "proximity_distance": 1.5,
            "max_reduction": 0.7,
        },
        weight=-0.05,
    )


# ============================================================================
# 課程管理配置
# ============================================================================
@configclass
class CurriculumCfgVLP16:
    """資料驅動課程學習 — 支援 baseline_v1 / goal_first_v1 版本切換"""
    goal_obstacle_curriculum = CurriculumTermCfg(
        func=goal_obstacle_curriculum,
        params={
            "window_size": 2000,
            "min_stage_episodes": 5000,
            "initial_stage": 1,
            "curriculum_version": "baseline_v1",  # CLI --curriculum_version 可覆蓋
        },
    )


# ============================================================================
# 完整課程環境配置
# ============================================================================
@configclass
class ChargeNavigationEnvCfgVLP16Curriculum(ChargeNavigationEnvCfgVLP16):
    """VLP-16 Curriculum 環境 — 20×20m + 4 階段課程學習

    繼承 VLP16 Phase 1 的觀測/動作/終止設定，
    覆蓋 scene/commands/events/rewards/curriculum。
    """

    scene: MySceneCfgVLP16_20x20 = MySceneCfgVLP16_20x20(num_envs=1024, env_spacing=22.0)
    commands: CommandsCfgVLP16Curriculum = CommandsCfgVLP16Curriculum()
    events: EventCfgVLP16Curriculum = EventCfgVLP16Curriculum()
    rewards: RewardsCfgVLP16Curriculum = RewardsCfgVLP16Curriculum()
    curriculum: CurriculumCfgVLP16 = CurriculumCfgVLP16()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 45.0  # Phase 1 初始值（課程動態調整 45→90s）
        self.viewer.eye = (9.0, 9.0, 9.0)
        # Per-env 牆壁由 init_perenv_walls (startup) + randomize_walls (reset) 管理


# ============================================================================
# NavRL-Style Dense Rewards 配置
# ============================================================================
@configclass
class RewardsCfgVLP16NavRL(RewardsCfgVLP16Curriculum):
    """NavRL-style dense reward 配置 — 取代 v9 純死亡機制

    設計原則：
    ┌─────────────────────────────────────────────────────────────────┐
    │ 引入 3 個 NavRL-style 密集獎勵，提供「方向性的安全導航信號」，  │
    │ 讓 agent 同時學會前進和避障，而非只靠稀疏的死亡懲罰。           │
    │ 危險區 progress 翻負 + 碰撞懲罰 -100，雙重保險避免衝撞行為。   │
    └─────────────────────────────────────────────────────────────────┘

    獎勵結構（6 項有效）：
    1. reaching_goal (+500)        — 覆蓋父類 250→500，提高 goal-seeking 動機
    2. velocity_to_goal (+15)      — 新增：雙向速度獎勵 [-1,1]（背離懲罰）
    3. safety_log_distance (+3)    — 新增：速度耦合 log 安全距離（靜止×0.1）
    4. safe_progress (+20)         — 分段線性 gate：danger zone 內 gate=-0.5（前進扣分）
    5. collision_terminal (-100)   — 恢復碰撞懲罰（reaching:collision = 5:1）
    6a. acceleration_penalty (-0.05) — 繼承：deadzone 版
    6b. angular_velocity_penalty (-0.05) — 繼承：context-aware 版

    行為排序：安全前進(+6.8) >> 原地不動(≈0) >> 危險前進(-1.2) >> 危險撤退(-0.8) >> 碰撞(-100)
    """

    # --- 新增 NavRL-style dense rewards ---

    velocity_to_goal = RewTerm(
        func=velocity_to_goal_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "min_goal_dist": 0.5,
            "max_reward_speed": 1.0,
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 10,
            "d_attenuate": 1.0,
            "d_full": 2.5,
        },
        weight=15.0,
    )

    safety_log_distance = RewTerm(
        func=safety_log_distance_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 10,
            "max_distance": 8.0,
            "speed_threshold": 0.1,
            "min_speed_factor": 0.1,
        },
        weight=3.0,
    )

    safe_progress = RewTerm(
        func=safe_progress_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 10,
            "d_danger": 0.8,
            "d_comfort": 2.0,
            "negative_scale": 0.5,
        },
        weight=20.0,
    )

    # --- 恢復碰撞懲罰: 0 → -100（reaching:collision = 5:1）---
    collision_terminal = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-100.0,
    )

    # --- 覆蓋父類 reaching_goal: 250 → 500 ---
    # 讓 terminal reward 從 dense 累計的 ~10% 提升到 ~20%
    reaching_goal = RewTerm(
        func=reaching_goal,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
        weight=500.0,
    )

    # --- 被 safe_progress 取代的舊版 PBRS ---
    potential_progress = RewTerm(
        func=potential_progress_reward,
        params={"robot_cfg": SceneEntityCfg("robot")},
        weight=0.0,
    )


# ============================================================================
# NavRL Curriculum 完整環境配置
# ============================================================================
@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRL(ChargeNavigationEnvCfgVLP16Curriculum):
    """VLP-16 Curriculum + NavRL Dense Rewards

    繼承 VLP16Curriculum 的 scene/commands/events/curriculum，
    只覆蓋 rewards 為 NavRL-style dense rewards。
    """

    rewards: RewardsCfgVLP16NavRL = RewardsCfgVLP16NavRL()


@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRL_PLAY(ChargeNavigationEnvCfgVLP16CurriculumNavRL):
    """NavRL 播放配置。

    目標:
    - GUI 檢視單一環境
    - 不依賴 camera sensors / optical flow
    - 開啟 LiDAR raycaster debug visualization
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 20.0
        self.scene.lidar.debug_vis = True
        # Top-down view (正上方俯視)
        self.viewer.eye = (0.0, 0.0, 25.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


# ============================================================================
# T 字型走廊場景 (Play 專用)
# ============================================================================
@configclass
class MySceneCfgVLP16_TCorridor(MySceneCfgVLP16_20x20):
    """T 字型走廊場景 — 58×20m 外牆 + 2 個填充塊形成 T 形走道

    可行走區域:
      Bar:  x ∈ [-28.5, +28.5], y ∈ [+4.0, +9.5]  → 57m × 5.5m
      Stem: x ∈ [-3.5, +3.5],  y ∈ [-9.5, +4.0]   → 7m × 13.5m
    """

    def __post_init__(self):
        super().__post_init__()

        # --- 加寬外牆: 20×20m → 58×20m ---
        room_x = 29.0   # ±29m (x 方向)
        room_y = 10.0    # ±10m (y 方向，不變)
        wall_thickness = 1.0
        wall_height = 3.0
        wall_length_x = room_x * 2 + wall_thickness  # 59m
        wall_length_y = room_y * 2 + wall_thickness   # 21m

        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_color = (0.5, 0.5, 0.5)
        wall_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=wall_color, metallic=0.1)

        # 覆蓋繼承的 4 面外牆（從 20×20 擴大到 58×20）
        self.wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_North",
            spawn=sim_utils.CuboidCfg(size=(wall_length_x, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, room_y, wall_height / 2)),
        )
        self.wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_South",
            spawn=sim_utils.CuboidCfg(size=(wall_length_x, wall_thickness, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -room_y, wall_height / 2)),
        )
        self.wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_East",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length_y, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(room_x, 0.0, wall_height / 2)),
        )
        self.wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_West",
            spawn=sim_utils.CuboidCfg(size=(wall_thickness, wall_length_y, wall_height),
                                       rigid_props=wall_rigid_props, collision_props=wall_collision_props, visual_material=wall_visual),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-room_x, 0.0, wall_height / 2)),
        )

        # --- 填充塊: 填滿 T 形走道以外的死區 ---
        fill_color = (0.45, 0.45, 0.45)
        fill_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=fill_color, metallic=0.1)

        # 左側填充塊: x ∈ [-28.5, -3.5], y ∈ [-9.5, +6.0]
        self.wall_fill_left = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_Fill_Left",
            spawn=sim_utils.CuboidCfg(
                size=(25.0, 15.5, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=fill_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-16.0, -1.75, wall_height / 2)),
        )

        # 右側填充塊: x ∈ [+3.5, +28.5], y ∈ [-9.5, +6.0]
        self.wall_fill_right = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_Fill_Right",
            spawn=sim_utils.CuboidCfg(
                size=(25.0, 15.5, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=fill_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(16.0, -1.75, wall_height / 2)),
        )


@configclass
class EventCfgVLP16TCorridor(EventCfgVLP16Curriculum):
    """T 走廊事件配置: 牆壁幾何 + 無隨機內牆 + 障礙物邊界調整為 T 走廊尺寸。

    T 走廊: bar (|X|<28.5, Y∈[6,9.5]) + stem (|X|<3.5, Y∈[-9.5,6])
    boundary=(28.0, 9.0) 涵蓋整個 T 形；fill blocks 由 check_wall_proximity_perenv 排除。
    """

    init_walls = EventTerm(
        func=init_perenv_walls,
        mode="startup",
        params={"boundary_walls_spec": BOUNDARY_WALLS_T_CORRIDOR},
    )

    randomize_wall_positions = EventTerm(
        func=randomize_walls,
        mode="reset",
        params={
            "min_walls": 0, "max_walls": 0,
            "boundary": 8.5, "min_wall_spacing": 2.0,
            "max_spawn_attempts": 30, "robot_safe_dist": 1.5,
        },
    )

    # T 走廊可通行區域 (留 0.8m 牆壁邊距):
    #   Bar:  x ∈ [-27.5, +27.5], y ∈ [+6.8, +8.7]  (55m × 1.9m ≈ 104.5 m²)
    #   Stem: x ∈ [-2.7, +2.7],  y ∈ [-8.7, +5.2]   (5.4m × 13.9m ≈ 75.1 m²)
    # 面積加權: Bar ≈ 58%, Stem ≈ 42%

    randomize_obstacles_startup = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="startup",
        params={
            "empty_ratio": 1.00, "static_ratio": 0.00, "dynamic_ratio": 0.00,
            "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
            "max_obstacles": 50, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.5, "max_spawn_attempts": 50,
            "boundary": (28.0, 9.0), "active_obstacle_ratio": 1.0,
            "spawn_zones": [(-27.5, 27.5, 6.8, 8.7), (-2.7, 2.7, -8.7, 5.2)],
            "debug": False,
        },
    )

    randomize_obstacles = EventTerm(
        func=randomize_obstacles_by_difficulty,
        mode="reset",
        params={
            "empty_ratio": 1.00, "static_ratio": 0.00, "dynamic_ratio": 0.00,
            "num_obstacles_static": 0, "num_obstacles_dynamic": 0,
            "max_obstacles": 50, "speed_range": 1.2, "min_speed": 0.3,
            "min_robot_distance": 1.5, "min_goal_distance": 1.0,
            "min_obstacle_spacing": 1.5, "max_spawn_attempts": 50,
            "boundary": (28.0, 9.0), "active_obstacle_ratio": 1.0,
            "spawn_zones": [(-27.5, 27.5, 6.8, 8.7), (-2.7, 2.7, -8.7, 5.2)],
            "debug": False,
        },
    )

    # 動態障礙物移動: 目標點只在 T 走廊可通行區域內生成
    # wall bounce 已內建（check_wall_proximity_perenv），fill blocks 會正確反彈
    move_dynamic_obstacles = EventTerm(
        func=move_obstacles_vectorized,
        mode="interval",
        interval_range_s=(0.2, 0.2),
        params={
            "move_dt": 0.2, "speed_min": 0.3, "speed_max": 1.2,
            "goal_reach_threshold": 0.5, "speed_resample_steps": 10,
            "area_limit": (27.0, 8.0), "max_obstacles": 50, "bound_limit": (28.0, 9.0),
            "goal_zones": [(-27.5, 27.5, 6.8, 8.7), (-2.7, 2.7, -8.7, 5.2)],
        },
    )


@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRL_PLAY_TCorridor(
    ChargeNavigationEnvCfgVLP16CurriculumNavRL
):
    """NavRL T 字型走廊播放配置。

    CLI: --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Play-TCorridor
    """

    scene: MySceneCfgVLP16_TCorridor = MySceneCfgVLP16_TCorridor()
    events: EventCfgVLP16TCorridor = EventCfgVLP16TCorridor()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 65.0  # 58m 場景 + 安全間距
        self.scene.lidar.debug_vis = True
        # 俯瞰 58×20m T 走廊（高度 50m 可覽全景）
        self.viewer.eye = (0.0, 0.0, 50.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # T 走廊: bar (|X|<28.5, Y>6) + stem (|X|<3.5, Y<6), fill blocks 由 AABB check 排除
        self.commands.goal_command.wall_boundary = (28.0, 9.0)


@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRL_TCorridor(
    ChargeNavigationEnvCfgVLP16CurriculumNavRL
):
    """NavRL T 字型走廊訓練配置。

    與 NavRL 完全一致，只改牆壁幾何（T 走廊 + 無隨機內牆）。
    Curriculum、reward、obstacle 全部繼承。
    """

    scene: MySceneCfgVLP16_TCorridor = MySceneCfgVLP16_TCorridor()
    events: EventCfgVLP16TCorridor = EventCfgVLP16TCorridor()

    def __post_init__(self):
        super().__post_init__()
        self.scene.env_spacing = 65.0  # 58m 場景 + 安全間距
        self.commands.goal_command.wall_boundary = (28.0, 9.0)



# ============================================================================
# 消融實驗 Configs (NavRL01-04)
# ============================================================================

# --- Gap reward imports ---
from ..mdp.rewards.gap_rewards import (
    heading_to_gap_reward,
    forward_clearance_improvement_reward,
)

# --- Shield action import ---
from ..mdp.actions.safety_shield import ShieldedDiscreteDifferentialDriveActionCfg

# --- NavRL01: v_gate floor=0.2 (保留 20% goal attraction near obstacles) ---
@configclass
class RewardsCfgVLP16NavRL01(RewardsCfgVLP16NavRL):
    """NavRL01: v_gate 永不完全關閉 — floor=0.2"""
    velocity_to_goal = RewTerm(
        func=velocity_to_goal_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "min_goal_dist": 0.5,
            "max_reward_speed": 1.0,
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 10,
            "d_attenuate": 1.0,
            "d_full": 2.5,
            "v_gate_floor": 0.2,
        },
        weight=15.0,
    )

@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRL_NavRL01(ChargeNavigationEnvCfgVLP16CurriculumNavRL):
    """NavRL01: v_gate floor 消融"""
    rewards: RewardsCfgVLP16NavRL01 = RewardsCfgVLP16NavRL01()


# --- NavRL02: safe_progress d_danger 0.8→0.55 (縮小 danger zone) ---
@configclass
class RewardsCfgVLP16NavRL02(RewardsCfgVLP16NavRL):
    """NavRL02: 延遲 danger zone — d_danger=0.55"""
    safe_progress = RewTerm(
        func=safe_progress_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 10,
            "d_danger": 0.55,
            "d_comfort": 2.0,
            "negative_scale": 0.5,
        },
        weight=20.0,
    )

@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRL_NavRL02(ChargeNavigationEnvCfgVLP16CurriculumNavRL):
    """NavRL02: safe_progress delayed danger zone"""
    rewards: RewardsCfgVLP16NavRL02 = RewardsCfgVLP16NavRL02()


# --- NavRL03: baseline + gap-seeking rewards ---
@configclass
class RewardsCfgVLP16NavRL03(RewardsCfgVLP16NavRL):
    """NavRL03: 新增 gap reward 引導繞行"""
    heading_to_gap = RewTerm(
        func=heading_to_gap_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "min_gap_width": 0.9,
            "activation_d_safe": 2.0,
            "speed_threshold": 0.05,
        },
        weight=5.0,
    )
    forward_clearance = RewTerm(
        func=forward_clearance_improvement_reward,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "front_arc_bins": 12,
            "activation_d_safe": 2.0,
        },
        weight=8.0,
    )

@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRL_NavRL03(ChargeNavigationEnvCfgVLP16CurriculumNavRL):
    """NavRL03: gap reward 消融"""
    rewards: RewardsCfgVLP16NavRL03 = RewardsCfgVLP16NavRL03()


# --- NavRL04: soft safety shield (rewards 不變) ---
@configclass
class ActionsCfgVLP16Shielded:
    """VLP16 動作 + soft safety shield"""
    diff_drive = ShieldedDiscreteDifferentialDriveActionCfg(
        asset_name="robot",
        debug_vis=True,
        num_bins=19,
        max_linear_velocity=1.0,
        max_linear_accel=0.5,
        max_angular_vel=1.2,    # 2026-06-02: 2.0 → 1.2 rad/s 對齊馬達 profile_omega_max
        max_angular_accel=3.0,  # rad/s² 對齊 cmd filter slew，防止舞龍舞獅
        reverse_velocity_scale=0.2,  # 反向上限 -0.2 m/s（前進偏好，防倒車鎖死）
        shield_mode="soft",
        shield_d_danger=0.55,
        shield_d_safe=1.2,
        sensor_name="lidar",
        body_radius=ROBOT_BODY_RADIUS,
    )

@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRL_NavRL04(ChargeNavigationEnvCfgVLP16CurriculumNavRL):
    """NavRL04: soft safety shield 消融"""
    actions: ActionsCfgVLP16Shielded = ActionsCfgVLP16Shielded()


# ============================================================================
# NavRL-Ground v1: 地面差速機器人版 NavRL 獎勵
# ============================================================================

from ..mdp.rewards.navrl_ground_rewards import (
    goal_velocity_reward as _ground_goal_vel,
    goal_progress_reward as _ground_progress,
    static_safety_reward as _ground_static,
    dynamic_safety_reward as _ground_dynamic,
    control_smoothness_penalty as _ground_smooth,
)

@configclass
class RewardsCfgVLP16NavRLGround(RewardsCfgVLP16Curriculum):
    """NavRL-Ground v1: 8 項 reward，無硬關閉 gate / 無翻負 progress。

    設計原則：
    1. goal_velocity: soft gate (beta=0.2 floor)，永不完全歸零
    2. goal_progress: soft scale (gamma=0.3 floor)，永不翻負
    3. static_safety: 72-bin log clearance + 前向堵塞懲罰
    4. dynamic_safety: per-obstacle log clearance + closing risk
    5. smoothness: dv² + dw²
    6. time: 每步小懲罰
    7. goal: 到達目標終端獎勵（沿用）
    8. collision: 碰撞終端懲罰（沿用）
    """

    # --- 新版 reward 項 ---
    goal_velocity = RewTerm(
        func=_ground_goal_vel,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 10,
            "v_max": 1.0,
            "min_goal_dist": 0.5,
            "use_soft_gate": True,
            "gate_beta": 0.15,  # v15: 0.05→0.15 搭配 kl_coeff=0.1 應能控制 KL
            "gate_dmin": 0.6,
            "gate_dmax": 2.0,
        },
        weight=10.0,
    )

    goal_progress = RewTerm(
        func=_ground_progress,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 10,
            "progress_clip": 1.0,
            "use_soft_scale": True,
            "scale_gamma": 0.15,  # v15: 同上
            "scale_dmin": 0.5,
            "scale_dmax": 2.0,
        },
        weight=12.0,
    )

    static_safety = RewTerm(
        func=_ground_static,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "a_global": 1.0,
            "a_front_block": 1.0,
            "front_half_angle_deg": 20.0,
            "front_nearest_k": 5,
            "front_warn_dist": 1.2,
        },
        weight=3.0,
    )

    dynamic_safety = RewTerm(
        func=_ground_dynamic,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "body_radius": ROBOT_BODY_RADIUS,
            "max_obstacles": 50,
            "mode": "log_distance",
            "risk_sigma": 1.0,
            "b_log": 1.0,
            "b_risk": 1.0,
        },
        weight=4.0,
    )

    smoothness = RewTerm(
        func=_ground_smooth,
        params={"smooth_v_coeff": 1.0, "smooth_w_coeff": 1.0},
        weight=-0.05,
    )

    # v15: -0.1→-0.2 搭配 bootstrap=True，讓 timeout 有 V(s) 但每步扣更多
    time_penalty = RewTerm(
        func=per_step_time_penalty,
        params={},
        weight=-0.2,
    )

    # --- 沿用的終端 reward ---
    reaching_goal = RewTerm(
        func=reaching_goal,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
        weight=500.0,
    )

    collision_ground = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-100.0,
    )

    # --- 歸零父類繼承的舊版 reward（避免重複計算）---
    potential_progress = RewTerm(func=potential_progress_reward, params={"robot_cfg": SceneEntityCfg("robot")}, weight=0.0)
    collision_terminal = RewTerm(func=collision_terminal_penalty, params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD}, weight=0.0)
    near_obstacle_penalty = RewTerm(func=exponential_obstacle_penalty, params={"sensor_cfg": SceneEntityCfg("lidar"), "safe_distance": 1.0, "steepness": 6.0}, weight=0.0)
    velocity_too_low = RewTerm(func=velocity_too_low_penalty, params={"robot_cfg": SceneEntityCfg("robot")}, weight=0.0)
    acceleration_penalty = RewTerm(func=deadzone_acceleration_penalty, params={"dead_zone": 0.1}, weight=0.0)
    angular_velocity_penalty = RewTerm(func=context_aware_angular_velocity_penalty, params={"robot_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("lidar"), "proximity_distance": 1.5, "max_reduction": 0.7}, weight=0.0)


@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRLGround(ChargeNavigationEnvCfgVLP16Curriculum):
    """VLP-16 Curriculum + NavRL-Ground v1 Rewards"""
    rewards: RewardsCfgVLP16NavRLGround = RewardsCfgVLP16NavRLGround()


# ============================================================================
# NavRL-Ground v2: 對齊 NavRL 原論文設計 — 無 gate + 存活獎勵
# ============================================================================

from ..mdp.rewards.navrl_ground_rewards import alive_reward as _ground_alive
from ..mdp.rewards.goal_rewards import alignment_reward as _ground_alignment

@configclass
class RewardsCfgVLP16NavRLGroundV2(RewardsCfgVLP16NavRLGround):
    """NavRL-Ground v2: 對齊 NavRL 原論文 — r_vel 無 gate + per-step 存活獎勵

    vs v1 的核心差異:
    1. goal_velocity: use_soft_gate=False (r_vel 100% 保持，對齊 NavRL)
    2. goal_progress: use_soft_scale=False (純 PBRS，無縮放)
    3. +1.0 per-step alive reward (NavRL 核心設計)
    4. 權重重新平衡：接近 NavRL 的 1:1:1 比例
    5. 移除 time_penalty（alive reward + γ 折扣已提供時間壓力）
    """

    # --- 存活獎勵 (NavRL 核心，降低佔比避免「活著比到達更好」) ---
    alive = RewTerm(
        func=_ground_alive,
        params={},
        weight=0.2,
    )

    # --- r_vel: 移除 soft_gate ---
    goal_velocity = RewTerm(
        func=_ground_goal_vel,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 36,          # 10→36: 更穩定的 d_safe 估計 (36/5760=0.6%)
            "v_max": 1.0,
            "min_goal_dist": 0.1,    # 0.5→0.1: 消除 0.35~0.5m 死區
            "use_soft_gate": False,   # 無 gate，對齊 NavRL
        },
        weight=2.0,
    )

    # --- r_progress: 移除 soft_scale ---
    goal_progress = RewTerm(
        func=_ground_progress,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "bottom_k": 36,           # 10→36: 與 goal_velocity 一致
            "progress_clip": 0.25,    # 1.0→0.25: max displacement=0.2m/step, 留 25% margin
            "use_soft_scale": False,   # 無縮放，對齊 NavRL
        },
        weight=3.0,
    )

    # --- 安全項：權重降低接近 1:1:1 ---
    static_safety = RewTerm(
        func=_ground_static,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "a_global": 1.0,
            "a_front_block": 1.0,
            "front_half_angle_deg": 30.0,  # 20→30: ±30°=60° 前方扇區，覆蓋側碰風險
            "front_nearest_k": 5,
            "front_warn_dist": 1.2,
        },
        weight=2.0,
    )

    dynamic_safety = RewTerm(
        func=_ground_dynamic,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "body_radius": ROBOT_BODY_RADIUS,
            "max_obstacles": 50,
            "mode": "log_distance",
            "risk_sigma": 2.0,        # 1.0→2.0: 更平緩的衰減，0.5m clearance 從 61%→78%
            "b_log": 1.0,
            "b_risk": 1.0,
        },
        weight=2.0,
    )

    # --- smoothness: 接近 NavRL 的 0.1 ---
    smoothness = RewTerm(
        func=_ground_smooth,
        params={"smooth_v_coeff": 1.0, "smooth_w_coeff": 1.0},
        weight=-0.1,
    )

    # --- 移除 time penalty (alive + γ 已足夠) ---
    time_penalty = RewTerm(
        func=per_step_time_penalty,
        params={},
        weight=0.0,
    )

    # --- 終端獎勵：降低佔比 ---
    reaching_goal = RewTerm(
        func=reaching_goal,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
        weight=100.0,
    )

    collision_ground = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-50.0,
    )


# ============================================================================
# NavRL-Ground v3: v2 + 20 obstacles + density-based reward weights
# ============================================================================

@configclass
class RewardsCfgVLP16NavRLGroundV3(RewardsCfgVLP16NavRLGroundV2):
    """NavRL-Ground v3: 與 v2 相同但支援 20 個障礙物。

    唯一差異：dynamic_safety 的 max_obstacles 從 10 改為 20。
    reward weights 由 curriculum goal_first_v2 動態控制。

    v4 修正:
    1. reaching_goal 100→500 (terminal reward 必須大於累計 per-step reward)
    2. alive weight=0 (移除存活獎勵)
       NavRL 設計: r_vel + γ折扣 = agent 想盡快到達
       alive 獎勵反而鼓勵「活著比到達更好」，與目標矛盾
    """
    # --- reaching_goal: 100→500 確保 terminal > 累計 per-step reward ---
    reaching_goal = RewTerm(
        func=reaching_goal,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": GOAL_REACH_THRESHOLD, "body_radius": ROBOT_BODY_RADIUS},
        weight=500.0,
    )

    # --- alive: 歸零 — r_vel + γ折扣已提供「盡快完成」的激勵 ---
    alive = RewTerm(
        func=_ground_alive,
        params={},
        weight=0.0,
    )

    dynamic_safety = RewTerm(
        func=_ground_dynamic,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "body_radius": ROBOT_BODY_RADIUS,
            "max_obstacles": 50,
            "mode": "log_distance",
            "risk_sigma": 2.0,
            "b_log": 1.0,
            "b_risk": 1.0,
        },
        weight=2.0,
    )


@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRLGroundV3(ChargeNavigationEnvCfgVLP16Curriculum):
    """VLP-16 Curriculum + NavRL-Ground v3 (20 obstacles)"""
    rewards: RewardsCfgVLP16NavRLGroundV3 = RewardsCfgVLP16NavRLGroundV3()


# ============================================================================
# NavRL-Ground v5: v3(v4) 基礎 + collision_ground 加倍
# 搭配 goal_first_v3 課程（Stage 3-6 局部提高 static_safety）
# ============================================================================

@configclass
class RewardsCfgVLP16NavRLGroundV5(RewardsCfgVLP16NavRLGroundV3):
    """NavRL-Ground v5: collision_ground -50→-100

    搭配 goal_first_v3 課程使用：
    - Stage 3-6 的 static_safety 權重局部提高
    - collision 顯性成本加倍 (-10→-20 per trigger)
    目的: 讓安全側訊號能抗衡前進側推力，促進繞行學習。
    """
    collision_ground = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-100.0,
    )


# ============================================================================
# NavRL-Ground v6: open-ended curriculum 專用
# v3 基礎 + a_front_block=0.4（弱化前向堵塞）+ collision=-50
# ============================================================================

@configclass
class RewardsCfgVLP16NavRLGroundV6(RewardsCfgVLP16NavRLGroundV3):
    """NavRL-Ground v6: open-ended curriculum 專用。

    vs V3/V4:
    - collision_ground: -50（不加倍，v5 加倍反而 side-grade）
    - static_safety: a_front_block 1.0→0.4（弱化前向堵塞，降低 freeze 風險）
    - reward 權重由 open_ended_v1 課程動態控制
    """
    static_safety = RewTerm(
        func=_ground_static,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("lidar"),
            "body_radius": ROBOT_BODY_RADIUS,
            "a_global": 1.0,
            "a_front_block": 0.4,
            "front_half_angle_deg": 30.0,
            "front_nearest_k": 5,
            "front_warn_dist": 1.2,
        },
        weight=2.0,
    )

    collision_ground = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-50.0,
    )


# ============================================================================
# NavRL-Ground v7: 碰撞成本遞增（死亡機制優先）
# ============================================================================

@configclass
class RewardsCfgVLP16NavRLGroundV7(RewardsCfgVLP16NavRLGroundV6):
    """NavRL-Ground v7: 碰撞成本遞增。

    前期 collision=-5（靠死亡機制學避障），後期由 curriculum 遞增到 -80。
    B6 降為 3G 7S 2D（降低 Stage 5→6 難度跳躍）。
    smooth_w_coeff: 1.0→0.6（降低角速度懲罰，讓繞行轉向更自由）。
    """
    smoothness = RewTerm(
        func=_ground_smooth,
        params={"smooth_v_coeff": 1.0, "smooth_w_coeff": 0.6},
        weight=-0.1,
    )

    collision_ground = RewTerm(
        func=collision_terminal_penalty,
        params={"sensor_cfg": SceneEntityCfg("lidar"), "threshold": COLLISION_THRESHOLD},
        weight=-5.0,
    )


# ============================================================================
# NavRL-Ground v8: 修正 safety/goal 比例失衡
# ss+ds weight 降低 + goal_velocity 下限提高
# ============================================================================

@configclass
class RewardsCfgVLP16NavRLGroundV8(RewardsCfgVLP16NavRLGroundV7):
    """NavRL-Ground v8: 修正 (ss+ds)/goal 比例失衡。

    v7 問題: Stage 6 的 (ss+ds)/goal=112%，安全 reward 超過核心。
    v8 修正: 降低 safety weight 上限 + 提高 goal_velocity 下限。
    目標: (ss+ds)/goal < 30%。
    """
    # 修正: V2 把 time_penalty 重設為 0，V3-V7 繼承了 0
    # 恢復 V1 的 -0.2，搭配 bootstrap=True 讓 timeout 有代價
    time_penalty = RewTerm(
        func=per_step_time_penalty,
        params={},
        weight=-0.2,
    )


@configclass
class ChargeNavigationEnvCfgVLP16CurriculumNavRLGroundV8(ChargeNavigationEnvCfgVLP16Curriculum):
    """VLP-16 Curriculum + NavRL-Ground v8 (v21 實驗用)"""
    rewards: RewardsCfgVLP16NavRLGroundV8 = RewardsCfgVLP16NavRLGroundV8()
