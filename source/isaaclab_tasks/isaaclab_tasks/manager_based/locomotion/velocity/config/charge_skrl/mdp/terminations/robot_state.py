"""機器人狀態異常終止條件

VLP16 訓練使用:
    robot_tipped_over: 機器人傾斜角過大 (翻倒)
    robot_flying: 機器人離地過高 (物理異常)
    physics_explosion: **Fix 2** 物理引擎爆炸偵測

physics_explosion 詳細說明:
    偵測 |lin_vel_xy| > 10 m/s 或 |ang_vel_z| > 20 rad/s
    這些超自然速度只會在物理引擎穿牆/碰撞數值不穩定時出現。
    觸發後立即終止 episode，防止 NaN 傳播到 RunningStandardScaler。

    參數:
        max_linear_velocity: float = 10.0  — 最大線速度 [m/s]
        max_angular_velocity: float = 20.0 — 最大角速度 [rad/s]

    背景: 2026-03-07 訓練中，step ~25K 發生物理爆炸
    (lin_vel 達 107,546 m/s)，永久污染了 scaler 統計量，
    導致 agent 後續完全無法學習（learned helplessness）。
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from ..wall_layout import (
    get_all_wall_tensors,
    check_wall_proximity_batch,
    check_wall_proximity_perenv,
    get_combined_wall_data,
)
from .obb_collision import obb_aabb_collision, relative_points_in_obb_frame

# ---- 方向性長方形（OBB）車體常數 ----
# 實車完整量測值（2026-07-20）：車長 0.70m、車寬 0.60m（含輪子）。
# USD 寬度 0.607m 與實車 0.60 僅差 7mm，對齊良好。膨脹後 OBB = 0.90×0.80m。
_ROBOT_HALF_LEN = 0.35     # 車體半長（沿車頭方向）[m]
_ROBOT_HALF_WID = 0.30     # 車體半寬（側向，含輪子）[m]
_OBB_BUFFER = 0.10         # 安全裕量（= COLLISION_BUFFER）[m]
_OBB_OFFSET_X = -0.128     # 碰撞盒中心沿車頭方向相對車體原點偏移 [m]（負=偏後）
_DEFAULT_OBS_PHYS_RADIUS = 0.30  # 障礙物理半徑 fallback（_obstacle_phys_radii 缺席時）[m]

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_tipped_over(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """終止條件：機器人翻倒
    
    當機器人翻倒時終止 episode（異常）。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
    
    Returns:
        shape [num_envs]：布林張量
        True = 翻倒，終止
        False = 正常，繼續
    
    如何判斷翻倒？
    - 檢查機器人的 Z 軸（本地座標系的「上方」）
    - 如果 Z 軸在世界座標系中不向上 → 翻倒了
    
    判斷邏輯：
    - 正常站立：z_axis = [0, 0, 1] → z_axis[:, 2] = 1.0 > 0.5 ✅
    - 側倒 90°：z_axis = [1, 0, 0] → z_axis[:, 2] = 0.0 < 0.5 ❌（翻倒）
    - 倒立：z_axis = [0, 0, -1] → z_axis[:, 2] = -1.0 < 0.5 ❌（翻倒）
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 獲取機器人姿態（四元數）
    quat = asset.data.root_quat_w
    # shape: [num_envs, 4]
    
    # 創建本地 Z 軸向量
    num_envs = quat.shape[0]
    z_vec = torch.zeros(num_envs, 3, device=env.device)
    z_vec[:, 2] = 1.0  # [0, 0, 1] = 垂直向上（本地座標系）
    
    # 轉換到世界座標系
    z_axis = math_utils.quat_apply(quat, z_vec)
    # 如果機器人正常：z_axis ≈ [0, 0, 1]（向上）
    # 如果機器人翻倒：z_axis ≈ [0, 0, -1]（向下）或 [1, 0, 0]（側倒）
    
    # 判斷翻倒
    is_tipped = z_axis[:, 2] < 0.5
    # z_axis[:, 2] 是 Z 軸在世界座標系中的垂直分量
    # < 0.5 表示傾斜超過 60 度（cos(60°) = 0.5）
    
    return is_tipped


def physics_explosion(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    max_linear_velocity: float = 10.0,
    max_angular_velocity: float = 20.0,
) -> torch.Tensor:
    """終止條件：物理爆炸偵測

    當機器人速度超過物理合理範圍時立即終止並重置，
    防止異常數據進入訓練（毒化 RunningStandardScaler）。

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        max_linear_velocity: 最大合理線速度 (m/s)，Charge 上限約 1.5m/s
        max_angular_velocity: 最大合理角速度 (rad/s)

    Returns:
        shape [num_envs]：布林張量
        True = 物理爆炸，終止
        False = 正常，繼續
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 線速度 magnitude (XY 平面)
    lin_vel_xy = asset.data.root_lin_vel_w[:, :2]  # [num_envs, 2]
    lin_speed = torch.norm(lin_vel_xy, dim=-1)  # [num_envs]

    # 角速度 magnitude
    ang_vel_z = asset.data.root_ang_vel_w[:, 2].abs()  # [num_envs]

    # 任一超過閾值 → 物理爆炸
    is_explosion = (lin_speed > max_linear_velocity) | (ang_vel_z > max_angular_velocity)

    # NaN 也算物理爆炸
    is_nan = torch.isnan(lin_speed) | torch.isnan(ang_vel_z)
    is_explosion = is_explosion | is_nan

    return is_explosion


def robot_flying(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """終止條件：機器人飛起來（異常）
    
    當機器人離地太高時終止 episode（物理引擎錯誤）。
    
    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
    
    Returns:
        shape [num_envs]：布林張量
        True = 飛起來了，終止
        False = 在地上，繼續
    
    為什麼需要這個？
    - 物理引擎有時會出錯（穿模、爆炸）
    - 機器人可能「升天」
    - 這種情況下應該終止並重置
    
    注意：
    - 正常情況下，Charge 底盤高度約 0.1-0.2 米
    - 高度 > 1.0 米視為異常
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 獲取高度
    height = asset.data.root_pos_w[:, 2]
    # shape: [num_envs]：Z 座標（高度）
    
    # 判斷是否飛起來
    is_flying = height > 1.0
    # 高度 > 1 米 → True（異常，終止）
    
    return is_flying


_wall_diag_count = 0


def robot_obb_wall_collision(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    half_len: float = _ROBOT_HALF_LEN,
    half_wid: float = _ROBOT_HALF_WID,
    buffer: float = _OBB_BUFFER,
) -> torch.Tensor:
    """方向性長方形車體 vs 牆 AABB 碰撞（撞牆 OBB 專用函式）。

    ★只由 ``wall_collision_termination(use_obb=True)`` 呼叫；**不改動**共用的
    ``check_wall_proximity_perenv``（該函式同時用於目標採樣/出生淨空/障礙擺放/牆反彈，
    改成 OBB 會破壞那些語意）。

    用車中心 + yaw + 牆 AABB 做 SAT 判定：車頭對齊時可通過 0.85m 窄縫、真正碰撞仍終止。
    純幾何在 obb_collision.py，已離線單元測試（test_obb_collision.py）。

    Returns: shape [num_envs] bool，True = 與任一啟用牆碰撞。
    """
    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
    env_origins = env.scene.env_origins[:, :2]
    robot_pos_local = robot_pos - env_origins
    # yaw：車頭相對世界 +x（local/world 旋轉一致，不受平移影響）
    yaw = torch.nan_to_num(math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2], nan=0.0)

    wall_c, wall_s, wall_mask = get_combined_wall_data(env)
    return obb_aabb_collision(
        robot_pos_local, yaw, wall_c, wall_s, wall_mask, half_len, half_wid, buffer, _OBB_OFFSET_X
    )


def wall_collision_termination(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 0.45,
    use_obb: bool = False,
) -> torch.Tensor:
    """分析式牆壁碰撞偵測 — 不依賴 LiDAR/contact sensor。

    ``use_obb=True`` → 走方向性長方形（OBB）判定（見 ``robot_obb_wall_collision``），
    讓車頭對齊時可通過 0.85m 窄縫。預設 ``False`` 維持圓/點模型（既有 run 不受影響）。

    用機器人 2D 位置 vs 所有牆壁 AABB 做距離計算。
    使用 wall_layout.py 的 check_wall_proximity_batch()。

    作為 LiDAR collision 的互補安全網：
    - LiDAR 可能因 kinematic wall 穿入、更新間隔跳過等邊緣情況遺漏牆壁碰撞
    - 此函數直接用幾何位置判斷，100% 可靠

    牆壁來源：
    - get_combined_wall_data(env): per-env maze walls + boundary walls
    - fallback: 靜態 20x20 牆壁（env._maze_wall_centers 不存在時）

    Args:
        env: 環境實例
        asset_cfg: 資產配置（引用機器人）
        threshold: 碰撞距離閾值 [m]（應與 COLLISION_THRESHOLD 一致）

    Returns:
        shape [num_envs]：布林張量
        True = 碰撞（距離 < threshold），終止
        False = 安全，繼續
    """
    global _wall_diag_count

    # 方向性長方形（OBB）路徑：交給專用函式，不動共用 check_wall_proximity_perenv
    if use_obb:
        return robot_obb_wall_collision(env, asset_cfg)

    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
    env_origins = env.scene.env_origins[:, :2]
    robot_pos_local = robot_pos - env_origins

    # 取得所有牆壁（per-env maze + boundary）
    wall_c, wall_s, wall_mask = get_combined_wall_data(env)
    result = check_wall_proximity_perenv(robot_pos_local, wall_c, wall_s, wall_mask, threshold)

    # 啟動診斷（僅第 1 步，摘要統計）
    _wall_diag_count += 1
    if _wall_diag_count == 1:
        pos = robot_pos_local.unsqueeze(1)          # [N, 1, 2]
        delta = (pos - wall_c).abs() - wall_s * 0.5  # [N, W, 2]
        delta = delta.clamp(min=0.0)
        dists = torch.norm(delta, dim=2)             # [N, W]
        # Mask out inactive walls
        dists = torch.where(wall_mask, dists, torch.full_like(dists, 1e6))
        d_min_per_env = dists.min(dim=1).values
        wall_src = "perenv" if hasattr(env, '_maze_wall_centers') else "fallback"
        print(
            f"[牆壁碰撞] 來源={wall_src} 牆數={wall_c.shape[1]} 門檻={threshold:.2f}m "
            f"環境數={d_min_per_env.shape[0]} | "
            f"最近距離: 最小={d_min_per_env.min():.2f} 平均={d_min_per_env.mean():.2f} 最大={d_min_per_env.max():.2f}",
            flush=True,
        )

    return result


_obs_collision_diag_count = 0


def obstacle_collision_geometric(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    collision_distance: float = 0.9,
    max_obstacles: int = 50,
    use_obb: bool = False,
) -> torch.Tensor:
    """幾何式障礙物碰撞偵測 — 不依賴 contact sensor force。

    Bug 修復: 原本依賴 ``collision_contact_occurred`` 透過 PhysX contact
    sensor 的 force_matrix_w 判定碰撞，但因為障礙物是 ``kinematic_enabled=True``
    且透過 ``write_root_pose_to_sim`` teleport 移動，PhysX 經常不產生足夠
    的接觸力 (< 0.1 N)，導致明顯穿透卻 CR=0%。

    判定邏輯：robot 中心 與 obstacle 中心 的距離 < collision_distance：

        碰撞 ⇔ ‖robot_pos - obs_pos‖ < collision_distance

    預設 collision_distance = 0.9m **與真實 LiDAR 最小偵測距離匹配**：
    比 0.9m 近的障礙物，真實感測器看不到 → robot 無法反應 → 視為碰撞失敗。
    這比「物理重疊」更保守，能讓 sim 訓練的 policy 在真實機器人上更安全。

    隱藏障礙物 (Z ≤ 0) 自動排除。

    Args:
        env:                Isaac Lab 環境實例
        asset_cfg:          機器人 SceneEntityCfg
        collision_distance: 中心距離碰撞門檻 (m)，預設 0.9（= 真實 LiDAR 盲區）
        max_obstacles:      場景中障礙物 entity 的最大編號

    Returns:
        shape [num_envs] bool tensor，True = 碰撞 (任一障礙物進入 0.9m 內)
    """
    global _obs_collision_diag_count

    robot = env.scene[asset_cfg.name]
    robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)  # [N, 2]
    N = robot_pos.shape[0]
    device = robot_pos.device

    overlap = torch.zeros(N, dtype=torch.bool, device=device)
    static_overlap = torch.zeros(N, dtype=torch.bool, device=device)
    dynamic_overlap = torch.zeros(N, dtype=torch.bool, device=device)

    # Eval-only [env, slot] hit mask for the corridor motion-phase audit.
    # Allocated only when the audit flag is set so the training path allocates
    # nothing and behaves exactly as before.
    _slot_hit_mask = None
    if getattr(env, "_corridor_phase_audit_enabled", False):
        _slot_hit_mask = torch.zeros(
            N, max_obstacles, dtype=torch.bool, device=device
        )

    # Per-env per-obstacle collision radii (from obs_size_rand randomization)
    has_per_obs_radii = hasattr(env, "_obstacle_radii")

    # OBB 車身碰撞路徑：用車 yaw + 障礙「物理半徑」(_obstacle_phys_radii)，非膨脹門檻 _obstacle_radii
    has_phys_radii = hasattr(env, "_obstacle_phys_radii")
    if use_obb:
        _yaw = torch.nan_to_num(math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2], nan=0.0)
        _cos_yaw = torch.cos(_yaw)  # [N]
        _sin_yaw = torch.sin(_yaw)

    # Static/dynamic classification by env difficulty + obstacle index
    has_difficulty = hasattr(env, "_env_difficulty")
    num_static_mixed = getattr(env, "_num_obstacles_static_mixed", 0)

    # rule_based(BehaviorScheduler)模式: 用每-slot behavior_type 做精確歸因
    # (2026-07-03: 修 behavior/collision_* 分項計數器沒 caller + rule_based 下
    #  _env_difficulty 是 mixed_parallel 概念,歸因不精確 → 有 scheduler 就用它)
    _sched = getattr(env, "_behavior_scheduler", None)
    _btype_static = None
    if _sched is not None:
        try:
            from ..events.behavior_scheduler import BEHAVIOR_STATIC as _btype_static
        except ImportError:
            _sched = None

    for i in range(max_obstacles):
        name = f"obstacle_{i}"
        if name not in env.scene.keys():
            continue

        obs_entity = env.scene[name]
        pos_w = obs_entity.data.root_pos_w  # [N, 3]
        pos_xy = torch.nan_to_num(pos_w[:, :2], nan=0.0)

        # 隱藏障礙物 (Z ≤ 0，例如 z=-10) → 跳過
        visible = pos_w[:, 2] > 0.0

        delta = pos_xy - robot_pos

        if use_obb:
            # 方向性長方形車身 vs 障礙圓（物理半徑）：把障礙轉進車體系、夾到長方形邊界求最近點
            lx, ly = relative_points_in_obb_frame(
                delta.unsqueeze(1),
                _cos_yaw.unsqueeze(1),
                _sin_yaw.unsqueeze(1),
                _OBB_OFFSET_X,
            )
            lx = lx.squeeze(1)
            ly = ly.squeeze(1)
            cx = lx.clamp(-_ROBOT_HALF_LEN, _ROBOT_HALF_LEN)
            cy = ly.clamp(-_ROBOT_HALF_WID, _ROBOT_HALF_WID)
            d_obb = torch.sqrt((lx - cx) ** 2 + (ly - cy) ** 2)
            if has_phys_radii and i < env._obstacle_phys_radii.shape[1]:
                r_phys = env._obstacle_phys_radii[:, i]  # [N] 物理半徑
            else:
                r_phys = _DEFAULT_OBS_PHYS_RADIUS
            hit = visible & (d_obb < (r_phys + _OBB_BUFFER))
        else:
            # 中心距離平方 vs threshold 平方（圓模型，避免 sqrt 計算）
            dist_sq = (delta * delta).sum(dim=1)
            if has_per_obs_radii and i < env._obstacle_radii.shape[1]:
                threshold_sq = env._obstacle_radii[:, i] ** 2  # [N]
            else:
                threshold_sq = collision_distance ** 2
            hit = visible & (dist_sq < threshold_sq)

        overlap = overlap | hit

        # Eval-only: preserve the per-slot hit mask for the corridor motion-phase
        # audit. Read-only bookkeeping — `hit` is already decided above and is
        # not modified here, so collision determination is untouched.
        if _slot_hit_mask is not None and i < _slot_hit_mask.shape[1]:
            _slot_hit_mask[:, i] = hit

        # Attribute hit to static or dynamic per-env
        if hit.any() and _sched is not None and i < _sched.behavior_type.shape[1]:
            # rule_based: 逐 slot behavior_type 精確歸因 + 分項計數(哪種行為造成碰撞)
            hit_ids = hit.nonzero(as_tuple=False).squeeze(-1)
            _sched.record_collisions(hit_ids, i)
            _bt = _sched.behavior_type[hit_ids, i]
            _is_static_b = (_bt == _btype_static)
            static_overlap[hit_ids] = static_overlap[hit_ids] | _is_static_b
            dynamic_overlap[hit_ids] = dynamic_overlap[hit_ids] | ~_is_static_b
        elif hit.any() and has_difficulty:
            diff = env._env_difficulty  # [N]
            # difficulty 1 = all static, 2 = all dynamic
            # difficulty 3 (mixed): i < num_static_mixed → static
            is_static_env = (diff == 1) | ((diff == 3) & (i < num_static_mixed))
            static_overlap = static_overlap | (hit & is_static_env)
            dynamic_overlap = dynamic_overlap | (hit & ~is_static_env)
        elif hit.any():
            static_overlap = static_overlap | hit

    # Store attribution for downstream metrics
    env._obs_collision_static_mask = static_overlap
    env._obs_collision_dynamic_mask = dynamic_overlap
    if _slot_hit_mask is not None:
        env._obs_collision_slot_mask = _slot_hit_mask

    # 一次性診斷（第 1 次呼叫，確認有抓到障礙物）
    _obs_collision_diag_count += 1
    if _obs_collision_diag_count == 1:
        n_obs_in_scene = sum(
            1 for i in range(max_obstacles)
            if f"obstacle_{i}" in env.scene.keys()
        )
        radii_info = ""
        if use_obb and has_phys_radii:
            r = env._obstacle_phys_radii
            radii_info = f" phys_radii=[{r.min():.2f}, {r.max():.2f}]"
        elif has_per_obs_radii:
            r = env._obstacle_radii
            radii_info = f" per_obs_radii=[{r.min():.2f}, {r.max():.2f}]"
        mode_info = (
            f"OBB half=({_ROBOT_HALF_LEN:.2f},{_ROBOT_HALF_WID:.2f}) "
            f"offset_x={_OBB_OFFSET_X:.3f} buffer={_OBB_BUFFER:.2f}"
            if use_obb else f"circle fallback={collision_distance:.2f}m"
        )
        print(
            f"[障礙物幾何碰撞] 實體={n_obs_in_scene}/{max_obstacles} "
            f"mode={mode_info}{radii_info} "
            f"碰撞數={int(overlap.sum().item())}/{N}",
            flush=True,
        )

    return overlap
