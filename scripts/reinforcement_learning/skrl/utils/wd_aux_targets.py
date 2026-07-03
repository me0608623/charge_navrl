"""
wd_aux_targets.py — WD-original dynamic 7D aux target + module loss

═══════════════════════════════════════════════════════════════════════════
Target 語意 — 時間動態預測（WD 原版）
═══════════════════════════════════════════════════════════════════════════

WD 原始 preprocess target 是讓 RNN 學會「單一最近動態障礙物」的時間軌跡：
  - 過去在哪 (t-3)
  - 現在在哪 (t0)
  - 未來會到哪 (v × Δt 外推)

這迫使 RNN hidden state 編碼：
  1. 障礙物速度/方向估計 → 預測 next
  2. 時間序列記憶 → 回憶 t-3
  3. 當前空間感知 → 感知 t0

7D target layout（每個 env 一筆）：
  [0] t0_obs_x      — 障礙物 NOW 的 body-frame surface x
  [1] t0_obs_y      — 障礙物 NOW 的 body-frame surface y
  [2] next_obs_x    — 障礙物 FUTURE (v×Δt) 的 body-frame surface x
  [3] next_obs_y    — 障礙物 FUTURE 的 body-frame surface y
  [4] hist_obs_x    — 障礙物 PAST (t-3) 的 body-frame surface x
  [5] hist_obs_y    — 障礙物 PAST (t-3) 的 body-frame surface y
  [6] total_dis     — 障礙物 PAST (t-3) 的 center distance

所有 vector 都相對「robot 當前位置 + 當前朝向」(body-frame)。
障礙物選擇：最近的 active dynamic obstacle（|v| > 閾值）。

WD CUDA 原始碼對照：
  - spot_3dmodule_step.cu 的 preprocess_data_arr (lines 683-760)
  - loc_x_arr_t3 歷史 cache (lines 1398-1413)
  - collision radius shrinking (lines 696-709)

與前版差異：
  前版 (spatial geometry): nearest-2 障礙物的 [x, y, d] × 2 + timestep
  本版 (temporal dynamics): 單一最近動態障礙物的 past/now/future + distance
═══════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn


# WD 原版 module loss 權重
# dim 0-1: t0 (now) 權重最高 — RNN 必須準確感知當前位置
# dim 2-3: next (future) — 預測能力核心
# dim 4-5: hist (past) — 記憶能力，權重略低
# dim 6: total_dis — 預設不參與訓練（weight=0）
WD_DEFAULT_WEIGHT = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.0]

# 13D weight: 7D original + 6D velocity (top-3 obstacles body-frame vx/vy)
WD_DEFAULT_WEIGHT_13D = [
    1.0, 1.0,    # dim 0-1: t0 (now)
    1.0, 1.0,    # dim 2-3: next (future)
    0.7, 0.7,    # dim 4-5: hist (past)
    0.0,         # dim 6: total_dis (disabled)
    0.8, 0.8,    # dim 7-8: obs1 velocity (nearest)
    0.5, 0.5,    # dim 9-10: obs2 velocity
    0.3, 0.3,    # dim 11-12: obs3 velocity
]

_VELOCITY_SCALE = 1.0 / 2.0  # max obstacle speed ~2 m/s

# 當沒有合適的動態障礙物時，用 FAR_DEFAULT 填充
_FAR_DEFAULT = 10.0

# 動態障礙物速度閾值：|v| > 此值才視為「動態」
_DYNAMIC_SPEED_THRESHOLD = 0.01

# 歷史 cache 深度（t-3，與 WD CUDA 的 loc_x_arr_t3 對齊）
_HISTORY_DEPTH = 3


def _wd_collision_shrink(
    dx: torch.Tensor,
    dy: torch.Tensor,
    obs_size: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """WD-style collision radius shrinking — 把 center vector 縮短到 surface point。

    WD CUDA 原版邏輯 (spot_3dmodule_step.cu lines 696-709):
    把 obstacle 的碰撞半徑沿 dx/dy 方向分別投影並扣除，
    使得 output vector 指向 obstacle 表面而非中心。
    若已在碰撞距離內 (total_dis < obs_size)，output 歸零。

    Args:
        dx: [E] world-frame 相對 x
        dy: [E] world-frame 相對 y
        obs_size: 碰撞半徑（robot + obstacle）

    Returns:
        shrunk_dx, shrunk_dy, total_dis
    """
    total_dis = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-6)

    # 沿 x/y 方向分別投影碰撞半徑
    obs_size_x = obs_size * torch.abs(dx) / total_dis
    obs_size_y = obs_size * torch.abs(dy) / total_dis

    # X 軸 shrink
    out_dx = torch.where(dx > obs_size_x, dx - obs_size_x,
             torch.where(dx < -obs_size_x, dx + obs_size_x,
             torch.zeros_like(dx)))

    # Y 軸 shrink
    out_dy = torch.where(dy > obs_size_y, dy - obs_size_y,
             torch.where(dy < -obs_size_y, dy + obs_size_y,
             torch.zeros_like(dy)))

    # 碰撞距離內全部歸零
    inside = total_dis < obs_size
    out_dx = torch.where(inside, torch.zeros_like(out_dx), out_dx)
    out_dy = torch.where(inside, torch.zeros_like(out_dy), out_dy)

    return out_dx, out_dy, total_dis


def _to_body_frame(
    dx_w: torch.Tensor,
    dy_w: torch.Tensor,
    cos_yaw: torch.Tensor,
    sin_yaw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """World-frame vector → robot body-frame vector.

    WD CUDA: x_body = dif_x * cos_dir + dif_y * sin_dir
             y_body = -dif_x * sin_dir + dif_y * cos_dir
    """
    bx = cos_yaw * dx_w + sin_yaw * dy_w
    by = -sin_yaw * dx_w + cos_yaw * dy_w
    return bx, by


def _update_history_cache(
    env_unwrapped,
    max_obstacles: int,
    device: torch.device,
) -> None:
    """更新 WD-style 3-step 歷史 position cache。

    WD CUDA (lines 1398-1413):
      t3 ← t2, t2 ← t1, t1 ← current
      episode_start 時全部初始化為 current

    Cache 存在 env_unwrapped 上，跨 step 保持。
    """
    num_envs = env_unwrapped.num_envs

    # 第一次呼叫：建立 cache
    if not hasattr(env_unwrapped, "_wd_aux_obs_pos_t1"):
        env_unwrapped._wd_aux_obs_pos_t1 = torch.zeros(
            num_envs, max_obstacles, 2, device=device)
        env_unwrapped._wd_aux_obs_pos_t2 = torch.zeros(
            num_envs, max_obstacles, 2, device=device)
        env_unwrapped._wd_aux_obs_pos_t3 = torch.zeros(
            num_envs, max_obstacles, 2, device=device)
        env_unwrapped._wd_aux_cache_initialized = torch.zeros(
            num_envs, dtype=torch.bool, device=device)

    # 讀取所有障礙物當前 world position
    current_pos = torch.zeros(num_envs, max_obstacles, 2, device=device)
    if not hasattr(env_unwrapped, "_obs_policy_cache"):
        env_unwrapped._obs_policy_cache = []
        for i in range(max_obstacles):
            name = f"obstacle_{i}"
            if name in env_unwrapped.scene.keys():
                env_unwrapped._obs_policy_cache.append(env_unwrapped.scene[name])
            else:
                env_unwrapped._obs_policy_cache.append(None)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None or i >= max_obstacles:
            break
        pos_w = obstacle.data.root_pos_w
        active = pos_w[:, 2] > 0.0
        current_pos[:, i, 0] = torch.where(active, pos_w[:, 0], current_pos[:, i, 0])
        current_pos[:, i, 1] = torch.where(active, pos_w[:, 1], current_pos[:, i, 1])

    # Episode 起點：所有 cache 初始化為 current
    # 用 episode_length_buf <= 1 偵測新 episode（WD: env_timestep_arr <= 1）
    is_new = env_unwrapped.episode_length_buf <= 1
    new_mask = is_new.unsqueeze(-1).unsqueeze(-1)  # [E, 1, 1]

    # 先做 rolling update（非新 episode 的 env）
    # t3 ← t2, t2 ← t1, t1 ← current（WD 原版順序）
    env_unwrapped._wd_aux_obs_pos_t3 = torch.where(
        new_mask, current_pos, env_unwrapped._wd_aux_obs_pos_t2)
    env_unwrapped._wd_aux_obs_pos_t2 = torch.where(
        new_mask, current_pos, env_unwrapped._wd_aux_obs_pos_t1)
    env_unwrapped._wd_aux_obs_pos_t1 = torch.where(
        new_mask, current_pos, current_pos)  # t1 always ← current


def _finite_diff_obstacle_vel(env_unwrapped, all_pos_now: torch.Tensor) -> torch.Tensor:
    """Per-slot 位置 finite-diff 速度 (rule_based 修復, 2026-07-03)。

    `_obstacle_velocities` 只有 mixed_parallel 事件路徑在寫；rule_based 的
    BehaviorScheduler 走 kinematic `write_root_pose_to_sim`,速度存私有 buffer
    從不橋接 → 公共 buffer 恆 0 → velocity target / next(=t0+v·Δt) / dynamic
    判別全部退化 (歷史 velocity-aux 負面結論的 zero-label 汙染源)。
    修法同 privileged_obs.py: Δpos/dt + teleport guard (>0.5 m/step 視為 reset)。
    每個 sim step 只能呼叫一次 (cache 依呼叫更新)。
    """
    # Per-step memo: play/train 迴圈同一步可能多處呼叫 build_wd_preprocess_targets
    # (probe 收集/oracle 收集/aux 診斷)。若每呼叫都更新 cache,第二次 Δpos=0 → 速度歸 0。
    # 用 common_step_counter 判「同一步」→ 回傳上次算好的速度,不動 cache。
    step_id = getattr(env_unwrapped, "common_step_counter", None)
    if step_id is not None and getattr(env_unwrapped, "_wdaux_fd_step", None) == step_id:
        memo = getattr(env_unwrapped, "_wdaux_fd_vel", None)
        if memo is not None and memo.shape == all_pos_now.shape:
            return memo
    dt = float(getattr(env_unwrapped, "step_dt", 0.2)) or 0.2
    prev = getattr(env_unwrapped, "_wdaux_prev_obs_xy", None)
    if prev is not None and prev.shape == all_pos_now.shape:
        delta = all_pos_now - prev
        ok = (delta.norm(dim=-1, keepdim=True) <= 0.5)
        vel_fd = torch.where(ok, delta / dt, torch.zeros_like(delta))
    else:
        vel_fd = torch.zeros_like(all_pos_now)
    env_unwrapped._wdaux_prev_obs_xy = all_pos_now.detach().clone()
    if step_id is not None:
        env_unwrapped._wdaux_fd_step = step_id
        env_unwrapped._wdaux_fd_vel = vel_fd.detach().clone()
    return vel_fd


def _build_velocity_targets(
    env_unwrapped,
    top_k: int,
    cos_yaw: torch.Tensor,
    sin_yaw: torch.Tensor,
    robot_pos_w: torch.Tensor,
    max_obstacles: int,
    device: torch.device,
    all_vel_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build body-frame velocity targets for the nearest K dynamic obstacles.

    Returns:
        [num_envs, top_k * 2] — (vbx, vby) per obstacle, scaled by _VELOCITY_SCALE.
        Slots without active dynamic obstacles are zero-filled.
    """
    num_envs = robot_pos_w.shape[0]
    result = torch.zeros(num_envs, top_k, 2, device=device)

    all_pos = torch.zeros(num_envs, max_obstacles, 2, device=device)
    all_vel = torch.zeros(num_envs, max_obstacles, 2, device=device)
    all_active = torch.zeros(num_envs, max_obstacles, dtype=torch.bool, device=device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None or i >= max_obstacles:
            break
        pos_w = obstacle.data.root_pos_w
        active = pos_w[:, 2] > 0.0
        all_active[:, i] = active
        all_pos[:, i, 0] = pos_w[:, 0]
        all_pos[:, i, 1] = pos_w[:, 1]

    if all_vel_override is not None:
        all_vel = all_vel_override  # 由 build_wd_preprocess_targets 傳入 (buffer+finite-diff 已合併)
    elif hasattr(env_unwrapped, "_obstacle_velocities"):
        n_vel = min(env_unwrapped._obstacle_velocities.shape[1], max_obstacles)
        all_vel[:, :n_vel, :] = env_unwrapped._obstacle_velocities[:, :n_vel, :]

    all_speed = torch.linalg.norm(all_vel, dim=-1)
    is_dynamic = all_active & (all_speed > _DYNAMIC_SPEED_THRESHOLD)

    dx = all_pos[:, :, 0] - robot_pos_w[:, 0:1]
    dy = all_pos[:, :, 1] - robot_pos_w[:, 1:2]
    dist = torch.sqrt(dx ** 2 + dy ** 2)
    dist_for_sort = torch.where(is_dynamic, dist,
                                torch.full_like(dist, _FAR_DEFAULT * 10))
    _, sorted_idx = torch.sort(dist_for_sort, dim=1)  # [E, N]
    batch_idx = torch.arange(num_envs, device=device)

    for k in range(top_k):
        idx = sorted_idx[:, k]  # [E]
        vx_w = all_vel[batch_idx, idx, 0]
        vy_w = all_vel[batch_idx, idx, 1]
        valid = is_dynamic[batch_idx, idx]
        vbx = cos_yaw * vx_w + sin_yaw * vy_w
        vby = -sin_yaw * vx_w + cos_yaw * vy_w
        result[:, k, 0] = torch.where(valid, vbx * _VELOCITY_SCALE, torch.zeros_like(vbx))
        result[:, k, 1] = torch.where(valid, vby * _VELOCITY_SCALE, torch.zeros_like(vby))

    return result.reshape(num_envs, top_k * 2)


def build_wd_preprocess_targets(
    env_unwrapped,
    max_obstacles: int,
    device: torch.device,
    robot_radius: float = 0.33,
    prediction_horizon_s: float = 0.2,
    top_k_velocity: int = 0,
    pos_scale: float = 1.0,
) -> torch.Tensor:
    """建立 WD-original dynamic 7D temporal aux target，可選擴充 velocity targets。

    流程：
    1. 更新 3-step 歷史 cache
    2. 識別最近的 active dynamic obstacle
    3. 對 t0 (now), next (future), hist (t-3) 分別計算 body-frame surface vector
    4. 組成 7D target
    5. (可選) 附加 top-K 障礙物的 body-frame velocity (top_k_velocity × 2D)

    Args:
        env_unwrapped: IsaacLab unwrapped env
        max_obstacles: 場景最大障礙物數量
        device: CUDA device
        robot_radius: 機器人半徑（用於 collision shrinking）
        prediction_horizon_s: 外推時間（秒），用於 next position
        top_k_velocity: 附加最近 K 個動態障礙物的 body-frame velocity。
            0 = 不附加（回傳 7D），3 = 回傳 13D。

    Returns:
        target: [num_envs, 7 + top_k_velocity*2] — temporal target
    """
    num_envs = env_unwrapped.num_envs

    # ------------------------------------------------------------------
    # 1. 更新歷史 cache
    # ------------------------------------------------------------------
    _update_history_cache(env_unwrapped, max_obstacles, device)

    # ------------------------------------------------------------------
    # 2. Robot pose → yaw → cos/sin（body-frame 旋轉）
    # ------------------------------------------------------------------
    robot_pos_w = env_unwrapped.scene["robot"].data.root_pos_w[:, :2]  # [E, 2]
    robot_quat = env_unwrapped.scene["robot"].data.root_quat_w         # [E, 4] wxyz
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)

    # ------------------------------------------------------------------
    # 3. 收集所有障礙物的 current pos / velocity / active status
    # ------------------------------------------------------------------
    all_pos_now = torch.zeros(num_envs, max_obstacles, 2, device=device)
    all_vel = torch.zeros(num_envs, max_obstacles, 2, device=device)
    all_active = torch.zeros(num_envs, max_obstacles, dtype=torch.bool, device=device)
    all_speed = torch.zeros(num_envs, max_obstacles, device=device)

    # 碰撞半徑
    has_radii = hasattr(env_unwrapped, "_obstacle_radii")
    all_col_radius = torch.full(
        (num_envs, max_obstacles), 0.5 + robot_radius, device=device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None or i >= max_obstacles:
            break
        pos_w = obstacle.data.root_pos_w
        active = pos_w[:, 2] > 0.0
        all_active[:, i] = active
        all_pos_now[:, i, 0] = pos_w[:, 0]
        all_pos_now[:, i, 1] = pos_w[:, 1]
        if has_radii:
            all_col_radius[:, i] = env_unwrapped._obstacle_radii[:, i]

    # Velocity: buffer(mixed_parallel 有寫) + per-slot finite-diff fallback(rule_based 修復 2026-07-03)
    # rule_based 下 buffer 恆 0 → 舊版 next(=t0+v·Δt)≡t0、velocity target 全 0、
    # dynamic 判別全 False (zero-label 汙染, 見 finding_zero_velocity_buffer_contamination)
    if hasattr(env_unwrapped, "_obstacle_velocities"):
        n_vel = min(env_unwrapped._obstacle_velocities.shape[1], max_obstacles)
        all_vel[:, :n_vel, :] = env_unwrapped._obstacle_velocities[:, :n_vel, :]
    _vel_fd = _finite_diff_obstacle_vel(env_unwrapped, all_pos_now)
    _buf_has = all_vel.norm(dim=-1, keepdim=True) > 1e-3
    all_vel = torch.where(_buf_has, all_vel, _vel_fd)
    all_speed = torch.sqrt(all_vel[:, :, 0] ** 2 + all_vel[:, :, 1] ** 2)

    # ------------------------------------------------------------------
    # 4. 選擇最近的 active dynamic obstacle
    # ------------------------------------------------------------------
    # 計算每個障礙物到 robot 的 center distance
    dx_all = all_pos_now[:, :, 0] - robot_pos_w[:, 0:1]  # [E, N]
    dy_all = all_pos_now[:, :, 1] - robot_pos_w[:, 1:2]  # [E, N]
    center_dist = torch.sqrt(dx_all ** 2 + dy_all ** 2)   # [E, N]

    # dynamic mask: active AND speed > threshold
    is_dynamic = all_active & (all_speed > _DYNAMIC_SPEED_THRESHOLD)

    # 對非 dynamic obstacle 設距離為 FAR，讓它排在最後
    dist_for_select = torch.where(is_dynamic, center_dist,
                                  torch.full_like(center_dist, _FAR_DEFAULT * 2))

    # 取最近 dynamic obstacle 的 index
    nearest_idx = dist_for_select.argmin(dim=1)  # [E]
    batch_idx = torch.arange(num_envs, device=device)

    # 若此 env 完全沒有 dynamic obstacle，fallback 到最近的 active obstacle
    no_dynamic = ~is_dynamic.any(dim=1)  # [E]
    if no_dynamic.any():
        fallback_dist = torch.where(all_active, center_dist,
                                    torch.full_like(center_dist, _FAR_DEFAULT * 2))
        fallback_idx = fallback_dist.argmin(dim=1)
        nearest_idx = torch.where(no_dynamic, fallback_idx, nearest_idx)

    # 取 selected obstacle 的屬性
    sel_pos_now = all_pos_now[batch_idx, nearest_idx]      # [E, 2]
    sel_vel = all_vel[batch_idx, nearest_idx]               # [E, 2]
    sel_col_r = all_col_radius[batch_idx, nearest_idx]      # [E]
    sel_active = all_active[batch_idx, nearest_idx]         # [E]

    # ------------------------------------------------------------------
    # 5. t0: obstacle NOW 相對 robot NOW（WD CUDA: t0_dx/t0_dy）
    # ------------------------------------------------------------------
    t0_dx_w = sel_pos_now[:, 0] - robot_pos_w[:, 0]
    t0_dy_w = sel_pos_now[:, 1] - robot_pos_w[:, 1]
    t0_sx, t0_sy, _ = _wd_collision_shrink(t0_dx_w, t0_dy_w, sel_col_r)
    t0_bx, t0_by = _to_body_frame(t0_sx, t0_sy, cos_yaw, sin_yaw)

    # ------------------------------------------------------------------
    # 6. next: obstacle FUTURE 相對 robot NOW（WD CUDA: next_dx/next_dy）
    # ------------------------------------------------------------------
    # WD: next_dx = loc[other] + speed * cos(dir) - loc[this]
    # IsaacLab: 直接用 velocity × horizon
    next_pos_x = sel_pos_now[:, 0] + sel_vel[:, 0] * prediction_horizon_s
    next_pos_y = sel_pos_now[:, 1] + sel_vel[:, 1] * prediction_horizon_s
    next_dx_w = next_pos_x - robot_pos_w[:, 0]
    next_dy_w = next_pos_y - robot_pos_w[:, 1]
    next_sx, next_sy, _ = _wd_collision_shrink(next_dx_w, next_dy_w, sel_col_r)
    next_bx, next_by = _to_body_frame(next_sx, next_sy, cos_yaw, sin_yaw)

    # ------------------------------------------------------------------
    # 7. hist: obstacle PAST (t-3) 相對 robot NOW（WD CUDA: current_dx/current_dy）
    # ------------------------------------------------------------------
    # WD 命名: "current" = loc_x_arr_t3（歷史 cache），confusingly named
    hist_pos = env_unwrapped._wd_aux_obs_pos_t3[batch_idx, nearest_idx]  # [E, 2]
    hist_dx_w = hist_pos[:, 0] - robot_pos_w[:, 0]
    hist_dy_w = hist_pos[:, 1] - robot_pos_w[:, 1]
    hist_sx, hist_sy, hist_total_dis = _wd_collision_shrink(
        hist_dx_w, hist_dy_w, sel_col_r)
    hist_bx, hist_by = _to_body_frame(hist_sx, hist_sy, cos_yaw, sin_yaw)

    # ------------------------------------------------------------------
    # 8. Inactive env → FAR_DEFAULT（沒有合適障礙物的 env）
    # ------------------------------------------------------------------
    inactive = ~sel_active
    far = torch.full_like(t0_bx, _FAR_DEFAULT)
    t0_bx = torch.where(inactive, far, t0_bx)
    t0_by = torch.where(inactive, far, t0_by)
    next_bx = torch.where(inactive, far, next_bx)
    next_by = torch.where(inactive, far, next_by)
    hist_bx = torch.where(inactive, far, hist_bx)
    hist_by = torch.where(inactive, far, hist_by)
    hist_total_dis = torch.where(inactive, far, hist_total_dis)

    # ------------------------------------------------------------------
    # 9. 組成 [E, 7] target
    # ------------------------------------------------------------------
    target = torch.stack([
        t0_bx,           # [0] obstacle NOW body-frame surface x
        t0_by,           # [1] obstacle NOW body-frame surface y
        next_bx,         # [2] obstacle FUTURE body-frame surface x
        next_by,         # [3] obstacle FUTURE body-frame surface y
        hist_bx,         # [4] obstacle PAST (t-3) body-frame surface x
        hist_by,         # [5] obstacle PAST (t-3) body-frame surface y
        hist_total_dis,  # [6] obstacle PAST center distance
    ], dim=-1)

    # WD-diff #2 修正:位置 target 是原始公尺(std~3m),但 RNN 輸入是正規化 obs(std~1)→
    # input/target 尺度錯配,最省力解=輸出常數。pos_scale 把位置 target 縮到 ~unit std,
    # 與正規化 obs 一致,且配合 huber 讓誤差落進二次梯度區(梯度∝e,逼模型追蹤)。
    # probe/live metric 都是 scale-invariant(rel_err=e/‖t‖、r2),不受影響。
    if pos_scale != 1.0:
        target = target * pos_scale

    # ------------------------------------------------------------------
    # 10. (可選) 附加 top-K 障礙物 body-frame velocity
    # ------------------------------------------------------------------
    if top_k_velocity > 0:
        vel_targets = _build_velocity_targets(
            env_unwrapped, top_k_velocity,
            cos_yaw, sin_yaw, robot_pos_w,
            max_obstacles, device,
            all_vel_override=all_vel,  # 已含 finite-diff 修復,避免重讀零 buffer/雙更新 cache
        )  # [E, top_k_velocity * 2]
        target = torch.cat([target, vel_targets], dim=-1)

    return target


def compute_wd_module_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: list[float] | None = None,
    loss_type: str = "log",
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """WD-style preprocess module aux loss。

    公式（dim 0~5）依 loss_type：
      "log"（WD 原版）  : L_i = mean( w_i × log(clamp(|e|, 0.01)) )
          ⚠️ 梯度 ∝ 1/|e| — 誤差越大梯度越小 → 大錯幾乎不修 → 鼓勵常數陷阱。
      "huber"（修正）  : L_i = mean( w_i × Huber(e, δ) )
          梯度 = e（|e|<δ）/ δ·sign(e)（|e|≥δ）— 大誤差大梯度 → 逼模型用 input 追蹤。

    dim 6: L_i = mean( w_i × |pred_i - target_i|² )  但預設 w_6=0，不影響梯度。

    Args:
        pred: [B, D] — model predict_head 輸出
        target: [B, D] — privileged geometry target
        weight: 每維 loss 權重，None → WD_DEFAULT_WEIGHT
        loss_type: "log"（WD 原版）或 "huber"（smooth-L1，修正常數陷阱）
        huber_delta: Huber 轉折點（target 量級 ~1）

    Returns:
        (loss_scalar, display_dict)
    """
    if weight is None:
        weight = WD_DEFAULT_WEIGHT

    weight_len = pred.size(-1)
    assert len(weight) == weight_len, (
        f"Weight length {len(weight)} must match pred dim {weight_len}"
    )

    display_loss: dict[str, float | torch.Tensor] = {}
    preprocess_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    for idx in range(weight_len):
        input_flat = pred[..., idx].reshape(-1)
        target_flat = target[..., idx].reshape(-1).detach()
        loss_ = nn.L1Loss(reduction="none")(input_flat, target_flat)

        if idx < 6:
            if loss_type == "huber":
                # Huber/smooth-L1: 大誤差大梯度（與 log 相反，避免常數陷阱）
                _abs = loss_
                loss_ = torch.where(
                    _abs < huber_delta,
                    0.5 * _abs * _abs,
                    huber_delta * (_abs - 0.5 * huber_delta),
                )
            else:
                loss_ = torch.clamp(loss_, min=0.01)
                loss_ = torch.log(loss_)
        else:
            loss_ = loss_ * loss_

        loss_ = loss_ * weight[idx]
        display_loss[f"module_feture_{idx}_loss"] = loss_.mean().item()
        preprocess_loss = preprocess_loss + loss_.mean()

    display_loss["preprcess_loss"] = preprocess_loss
    return preprocess_loss, display_loss
