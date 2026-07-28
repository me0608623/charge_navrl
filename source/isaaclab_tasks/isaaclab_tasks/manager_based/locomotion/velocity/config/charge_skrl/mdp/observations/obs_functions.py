"""VLP-16 專用觀測管線 — 純函數、GPU 向量化

VLP16 訓練使用的函數:
    lidar_vlp16_to_2d_bins: VLP-16 16ch×360 點雲 → 72-bin 2D 最小距離 — 72D (×3帧=216D)
    topk_obstacles_body_frame: 最近 K 個障礙物的 body-frame 特徵 — 50D (10×5)
    robot_heading_normalized: 朝向 → (sin θ, cos θ) — 2D
    robot_position_local: 歸一化位置 (x/8, y/8) — 2D
    discrete_applied_action: 上一步離散動作歸一化 — 2D

lidar_vlp16_to_2d_bins 詳細說明:
    輸入: VLP-16 原始射線 [num_envs, 16×360, 3] hit_points_w
    處理:
        1. 投影到 2D (忽略 Z)
        2. 計算距離 + 方位角
        3. 分成 72 個 5° bins，每 bin 取 min 距離
        4. 歸一化 → [0, 1] (距離/max_distance)
    輸出: [num_envs, 72]
    感測器噪聲 (域隨機化):
        - displacement_std=0.02: 距離高斯雜訊 ±2cm
        - hole_rate=0.005: 射線丟失 0.5%
        - distractor_rate=0.002: 幽靈點 0.2%

topk_obstacles_body_frame 詳細說明:
    輸入: 場景中所有障礙物的世界座標位置和速度
    處理:
        1. 轉換到機器人 body frame
        2. 按距離排序取 Top-K
        3. Line-of-Sight 遮擋檢查 (使用 wall_layout.check_los_batch)
        4. 被遮擋的障礙物特徵歸零
    輸出: [num_envs, K×5] — (dx, dy, vx, vy, size) per obstacle
"""

from __future__ import annotations

import math
import os
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


# ====================================================================
# 觀測函數一：VLP-16 LiDAR 16×360 → N Bins
# ====================================================================

def lidar_vlp16_to_2d_bins(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    num_channels: int = 16,
    num_horizontal: int = 360,
    num_bins: int = 72,
    r_max: float = 20.0,
    r_robot: float = 0.3,
    r_min: float = 0.0,
    z_filter: float = 0.0,
    # Point cloud corruption (Sim-to-Real domain randomization)
    displacement_std: float = 0.0,
    hole_rate: float = 0.0,
    distractor_rate: float = 0.0,
    distractor_range: tuple[float, float] = (0.2, 2.0),
) -> torch.Tensor:
    """VLP-16 LiDAR 3D 點雲降維至 2D N-bin 距離向量。

    物理層：RayCaster 以 16 條垂直線 × 360 個水平採樣，
            發射 5760 條 3D 射線，精準捕捉微小障礙物。

    演算法層：透過向量化前處理，壓縮為 N 個角度區間。
    支援點雲級 Sim-to-Real corruption，於 min pooling 前注入。

    處理流程::

        ray_hits_w [N, 5760, 3]
          │
          ├── 投影至 2D 水平面，計算距離 → [N, 5760]
          │
          ├── Step 1: Clipping (inf/NaN/> r_max → r_max)
          │
          ├── Step 1.5: Point Cloud Corruption (Sim-to-Real)
          │     (a) Random Displacement: N(0, σ²) per ray
          │     (b) Strategic Holes: random rays → r_max
          │     (c) Distractors: random close-range false readings
          │
          ├── Step 2: Min Pooling
          │     view → [N, 16, num_bins, rays_per_bin]
          │     min(dim=3) → [N, 16, num_bins]  (rays_per_bin 取最近)
          │     min(dim=1) → [N, num_bins]       (16 channels 取最近)
          │
          ├── Step 3: Safe Margin (− r_robot, clamp ≥ 0)
          │
          └── Step 4: Normalization (÷ r_max → [0, 1])

    Args:
        env:              Isaac Lab 環境實例
        sensor_cfg:       RayCaster 感測器的 SceneEntityCfg 參照
        num_channels:     垂直通道數 (VLP-16 = 16)
        num_horizontal:   每通道水平採樣數 (360° / 1° = 360)
        num_bins:         最終輸出的角度區間數 (預設 72 = 5° 解析度)
        r_max:            最大探測距離 (m)
        r_robot:          車體半徑 (m)
        r_min:            ★ 真實 LiDAR 最小偵測距離 (m)，0=關閉
                          模擬實際感測器盲區。距離 < r_min 的命中 → r_max
                          (例如真實 VLP-16 的 0.9m 盲區)
        z_filter:         ★ Z 軸過濾門檻 (m)，0=關閉
                          忽略高度差超過此值的命中 (排除隱藏障礙物 z=-10 鬼影)
        displacement_std: 個別射線距離高斯雜訊標準差 (m)，0=關閉
        hole_rate:        射線隨機遺失機率 (→ r_max)，0=關閉
        distractor_rate:  虛假近距離讀數機率，0=關閉
        distractor_range: 虛假讀數距離範圍 (min_m, max_m)

    Returns:
        [num_envs, num_bins] 正規化距離張量，數值範圍 [0, 1]
    """
    # ----------------------------------------------------------
    # 從 Isaac Lab Scene 取得 RayCaster 感測器資料
    # ----------------------------------------------------------
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_pos = sensor.data.pos_w          # [N, 3] 感測器世界座標
    hit_points = sensor.data.ray_hits_w     # [N, total_rays, 3] 射線命中點

    N = sensor_pos.shape[0]                 # num_envs
    device = sensor_pos.device

    # ----------------------------------------------------------
    # 投影至 2D 水平面：計算每條射線的水平距離
    # ----------------------------------------------------------
    # 只取 XY 座標（捨棄 Z），計算 2D 歐式距離
    sensor_xy = sensor_pos[:, :2].unsqueeze(1)  # [N, 1, 2]
    hits_xy = hit_points[:, :, :2]               # [N, total_rays, 2]
    dist_2d = torch.norm(hits_xy - sensor_xy, dim=-1)  # [N, total_rays]

    # ==========================================================
    # Step 0: Z-filter — 排除 Z 異常的命中（修復 Bug B「鬼影」問題）
    #
    # 隱藏障礙物 (z=-10) 投影到 2D 後會偽裝成「近距離物體」，
    # 造成 lidar.min 永遠 ≈ 0。
    # 使用 per-ray origin z (包含 OffsetCfg z=1.6) 而非 pos_w (base_link z≈0)。
    # ==========================================================
    if z_filter > 0:
        # _ray_starts_w includes OffsetCfg; pos_w reports parent prim z≈0
        if hasattr(sensor, "_ray_starts_w") and sensor._ray_starts_w is not None:
            origin_z = sensor._ray_starts_w[:, :, 2:3]  # [N, R, 1]
        else:
            import warnings
            warnings.warn(
                "lidar_vlp16_to_2d_bins: sensor._ray_starts_w unavailable, "
                "falling back to sensor.data.pos_w for z_filter",
                stacklevel=2,
            )
            origin_z = sensor_pos[:, 2:3].unsqueeze(1)  # [N, 1, 1]
        hit_z = hit_points[:, :, 2:3]                    # [N, R, 1]
        z_diff = (hit_z - origin_z).abs().squeeze(-1)    # [N, R]
        # 高度差超過 z_filter 的命中視為無效 → r_max
        invalid_z = z_diff > z_filter
        dist_2d = torch.where(
            invalid_z,
            torch.tensor(r_max, device=device, dtype=dist_2d.dtype),
            dist_2d,
        )

    # ==========================================================
    # Step 1: Clipping
    #   inf / NaN / 超過 r_max → 截斷為 r_max
    # ==========================================================
    dist_2d = torch.where(
        torch.isfinite(dist_2d),
        dist_2d,
        torch.tensor(r_max, device=device, dtype=dist_2d.dtype),
    )
    dist_2d = torch.clamp(dist_2d, max=r_max)

    # ==========================================================
    # Step 1.1: 真實 LiDAR Min Range — 模擬感測器盲區
    #
    # 真實 VLP-16 / RPLidar 等雷達都有「最小偵測距離」，
    # 比這個距離更近的物體 → 雷達看不到 (no return / NaN)。
    # 設 r_min=0.25 對應實測值 (v3, 2026-06-08)：
    #   實測 VLP-16 表面→人物中心最近 0.2m，加上 LiDAR 物理半徑 0.0515m
    #   = sim 內 optical center→人物中心 0.2515m，向上取整 0.25m。
    # 對策：將距離 < r_min 的命中視為「無偵測」→ 設為 r_max。
    # ==========================================================
    if r_min > 0:
        too_close = dist_2d < r_min
        dist_2d = torch.where(
            too_close,
            torch.tensor(r_max, device=device, dtype=dist_2d.dtype),
            dist_2d,
        )

    # ==========================================================
    # Step 1.5: Point Cloud Corruption (Sim-to-Real)
    #   在 min pooling 前對原始 [N, 5760] 點雲注入感測器雜訊
    # ==========================================================

    # (a) Random Displacement: 每條射線加入高斯雜訊，模擬量測不確定性
    if displacement_std > 0:
        noise = torch.randn_like(dist_2d) * displacement_std
        dist_2d = torch.clamp(dist_2d + noise, min=0.0, max=r_max)

    # (b) Strategic Holes: 隨機射線遺失 → r_max，模擬 LiDAR 漏點
    if hole_rate > 0:
        hole_mask = torch.rand_like(dist_2d) < hole_rate
        dist_2d = torch.where(hole_mask, r_max, dist_2d)

    # (c) Distractors: 虛假近距離讀數，模擬感測器缺陷/多路徑反射
    if distractor_rate > 0:
        distr_mask = torch.rand_like(dist_2d) < distractor_rate
        min_r, max_r = distractor_range
        distr_values = torch.rand_like(dist_2d) * (max_r - min_r) + min_r
        dist_2d = torch.where(distr_mask, distr_values, dist_2d)

    # ==========================================================
    # Step 2: Min Pooling (16 × 360 → num_bins)
    #
    #   RayCaster 射線排列順序：
    #     channel_0: h0, h1, ..., h359
    #     channel_1: h0, h1, ..., h359
    #     ...
    #     channel_15: h0, h1, ..., h359
    #
    #   view 為 [N, 16, num_bins, rays_per_bin]：
    #     dim=1: 16 個垂直通道
    #     dim=2: num_bins 個角度區間
    #     dim=3: 每 bin 內 rays_per_bin 條射線
    # ==========================================================
    rays_per_bin = num_horizontal // num_bins   # 360 / 72 = 5

    x = dist_2d.view(N, num_channels, num_bins, rays_per_bin)
    # [N, 16, num_bins, rays_per_bin]

    # 水平 min (dim=3)：每 bin 內 rays_per_bin 條射線取最近距離
    x = x.min(dim=3).values     # [N, 16, num_bins]

    # 垂直 min (dim=1)：16 層取最近距離（最保守估計）
    x = x.min(dim=1).values     # [N, num_bins]

    # ==========================================================
    # Step 3: Safe Margin（扣除車體半徑，保證非負）
    # ==========================================================
    x = torch.clamp(x - r_robot, min=0.0)

    # ==========================================================
    # Step 4: Normalization（除以 r_max，壓縮至 [0, 1]）
    # ==========================================================
    x = x / r_max

    return x   # [N, num_bins]


def _get_episode_noise_scale(
    env: ManagerBasedRLEnv,
    key: str,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """Return per-env noise scale, resampling only for newly-reset envs.

    Stores state in env._lidar_ep_noise_cache to avoid extra env fields.
    Resample condition: episode_length_buf == 0 (first step of each episode).
    """
    cache_attr = "_lidar_ep_noise_cache"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, {})
    cache: dict = getattr(env, cache_attr)

    device = env.device
    if key not in cache:
        mid = (lo + hi) * 0.5
        cache[key] = torch.full((env.num_envs,), mid, device=device, dtype=torch.float32)

    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).squeeze(-1)
    if len(reset_ids) > 0:
        new_vals = torch.rand(len(reset_ids), device=device) * (hi - lo) + lo
        cache[key][reset_ids] = new_vals

    return cache[key]  # [N]


def _dynamic_human_ray_mask(
    env,
    hit_points_w: torch.Tensor,   # [N, R, 3] world-frame ray hits
    valid: torch.Tensor,          # [N, R] bool
    margin: float,
    move_eps: float,
    move_max: float,
    max_obstacles: int = 10,
) -> torch.Tensor:
    """Mark rays whose world hit-point lies inside a visible, MOVING obstacle (=human material).

    Dynamic (human) is detected by **per-step position finite-difference** — robust because
    the rule-based BehaviorScheduler moves obstacles via ``write_root_pose_to_sim`` (kinematic
    pose writes), so ``_obstacle_velocities`` / physics ``root_vel_w`` stay ~0 for them.
    An obstacle counts as human when it is visible (root z > 0) AND its centre moved between
    ``move_eps`` and ``move_max`` metres since the last call (upper bound excludes reset
    teleports). Geometric in-circle test; no mesh-index-layout dependency. Missed rays hold
    NaN hit points → the ``<`` comparisons are False for them (and ``& valid`` re-guards).

    Returns:
        ``[N, R]`` bool mask of rays that hit a dynamic (human) obstacle.
    """
    N, R, _ = hit_points_w.shape
    device = hit_points_w.device
    hit_xy = hit_points_w[..., :2]  # [N, R, 2]

    # 1) Gather current obstacle centres / visibility / radii
    cur_xy = torch.zeros(N, max_obstacles, 2, device=device)
    visible = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
    radius = torch.full((N, max_obstacles), 0.3, device=device)
    sizes = getattr(env, "_obstacle_sizes", None)
    for i in range(max_obstacles):
        name = f"obstacle_{i}"
        if name not in env.scene.keys():
            continue
        pos_w = env.scene[name].data.root_pos_w      # [N, 3]
        cur_xy[:, i] = pos_w[:, :2]
        visible[:, i] = pos_w[:, 2] > 0.0            # z = -10 → hidden
        if sizes is not None:
            try:
                radius[:, i] = float(sizes[i])
            except (IndexError, TypeError):
                pass

    # 2) Dynamic = moved this step (finite-diff), excluding reset teleports (> move_max)
    # Per-step memo (2026-07-03 fix, 審計1-D2): policy 與 critic 兩個 obs group 都掛
    # wd_like_sweep_72 → 同一 sim 步呼叫兩次 → 第二次 Δ=0 → 該 group 的 human dropout
    # 恆 no-op(且 sanity log 被第一 group 掩蓋)。同一步內重複呼叫直接回 memo、不動 cache。
    _step_id = getattr(env, "common_step_counter", None)
    if _step_id is not None and getattr(env, "_matnoise_fd_step", None) == _step_id:
        _memo = getattr(env, "_matnoise_fd_isdyn", None)
        if _memo is not None and _memo.shape == (N, max_obstacles):
            is_dyn = _memo
        else:
            is_dyn = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
    else:
        prev = getattr(env, "_matnoise_prev_obs_xy", None)
        if prev is not None and prev.shape == cur_xy.shape:
            disp = torch.linalg.norm(cur_xy - prev, dim=-1)   # [N, max_obs]
            is_dyn = (disp > move_eps) & (disp < move_max)
        else:
            is_dyn = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
        env._matnoise_prev_obs_xy = cur_xy.detach().clone()
        if _step_id is not None:
            env._matnoise_fd_step = _step_id
            env._matnoise_fd_isdyn = is_dyn.detach().clone()

    # 3) Rays inside any visible+moving obstacle circle
    active = visible & is_dyn                        # [N, max_obs]
    human = torch.zeros(N, R, dtype=torch.bool, device=device)
    if bool(active.any()):
        for i in range(max_obstacles):
            a = active[:, i]
            if not bool(a.any()):
                continue
            d = torch.linalg.norm(hit_xy - cur_xy[:, i:i + 1, :], dim=-1)   # [N, R]
            human |= (d < (radius[:, i:i + 1] + margin)) & a.unsqueeze(1)

    return human & valid


def wd_like_sweep_72(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    num_bins: int = 72,
    r_max: float = 20.0,
    r_robot: float = 0.3,
    r_min: float = 0.0,
    z_filter: float = 0.0,
    # --- L1 noise: fixed-σ (legacy) ---
    displacement_std: float = 0.0,
    hole_rate: float = 0.0,
    distractor_rate: float = 0.0,
    distractor_range: tuple[float, float] = (0.2, 2.0),
    # --- L1 noise: distance-dependent σ(r) = std_per_meter × r ---
    displacement_std_per_meter: float = 0.0,
    # --- L1 noise: fixed-σ soft target (human/clothing), independent of distance ---
    displacement_std_soft: float = 0.0,
    # --- L3 per-episode DR: displacement (per-meter scale) ---
    displacement_std_per_meter_dr_min: float = 0.0,
    displacement_std_per_meter_dr_max: float = 0.0,
    # --- L3 per-episode DR: soft-target fixed σ ---
    displacement_std_soft_dr_min: float = 0.0,
    displacement_std_soft_dr_max: float = 0.0,
    # --- L3 per-episode DR: hole_rate ---
    hole_rate_dr_min: float = 0.0,
    hole_rate_dr_max: float = 0.0,
    # --- L1 systematic range bias: bias(d) = k·d + b ---
    # Scalar values (fixed across envs/episodes). Use DR ranges below for per-episode sampling.
    distance_bias_k: float = 0.0,
    distance_bias_b: float = 0.0,
    # --- L1 distance bias DR (per-episode sampling, range covers measured ±36mm) ---
    # k 範圍涵蓋 ±10mm/m slope；b 範圍涵蓋 ±40mm offset
    # 每 episode 採樣一組 (k, b)，不假設特定模型形狀（避免 R²=0.58 線性假設誤導 policy）
    distance_bias_k_dr_min: float = 0.0,
    distance_bias_k_dr_max: float = 0.0,
    distance_bias_b_dr_min: float = 0.0,
    distance_bias_b_dr_max: float = 0.0,
    # --- L1 per-ring bias (VLP-16, 16ch ±22.7mm fixed offsets) ---
    per_ring_bias: bool = False,
    per_ring_bias_scale: float = 1.0,
    per_ring_bias_scale_dr_min: float = 0.0,
    per_ring_bias_scale_dr_max: float = 0.0,
    # --- legacy L3 params (absorbed, now no-op to prevent TypeError) ---
    displacement_std_dr_min: float = 0.0,
    displacement_std_dr_max: float = 0.0,
    # --- L2 Per-Bin block dropout (absorbed for forward-compat) ---
    block_dropout_prob: float = 0.0,
    block_dropout_width_min: int = 3,
    block_dropout_width_max: int = 8,
    # --- Per-material: human noise on DYNAMIC-obstacle rays (Phase 2, full_material) ---
    # Rays whose world hit-point falls inside a visible, moving obstacle (velocity≠0) get
    # measured human dropout(d) + higher mixed-pixel, and NO σ (human σ is null). Geometric
    # classification (ray-hit inside obstacle circle) — no mesh-index-layout dependency.
    human_dynamic_dropout: bool = False,
    human_dropout_slope: float = 0.0135,      # dropout(d) = slope·d + intercept (measured 1m .209 / 3m .236)
    human_dropout_intercept: float = 0.196,
    human_mixed_pixel_rate: float = 0.178,    # human ghost/mixed-pixel (high but noisy; provisional)
    human_obs_margin: float = 0.15,           # extra margin (m) added to obstacle radius for the in-circle test
    human_move_eps: float = 0.01,             # min per-step obstacle displacement (m) to count as moving (=human)
    human_move_max: float = 0.5,              # max per-step displacement (m); above → reset teleport, not motion
) -> torch.Tensor:
    """Build a WD-style 72-bin sweep from raw ray hits in the robot yaw frame.

    This term does not rely on the original ray ordering. Instead, it:

    1. Computes hit vectors in world frame from ``ray_hits_w - ray_starts_w``.
    2. Rotates them into the sensor/robot yaw-aligned frame.
    3. Quantizes angles into a fixed 360-degree, 72-bin layout.
    4. Uses the minimum distance in each bin as the sweep value.
    5. Supports multi-channel LiDAR implicitly because all channels project into
       the same 72 azimuth bins and are reduced with ``amin``.

    Returns:
        Tensor of shape ``[num_envs, 72]`` in normalized distance units ``[0, 1]``.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_quat_w = sensor.data.quat_w      # [N, 4]
    hit_points_w = sensor.data.ray_hits_w   # [N, R, 3]

    num_envs, num_rays, _ = hit_points_w.shape
    device = hit_points_w.device
    dtype = hit_points_w.dtype

    # Per-ray world-space origins (includes OffsetCfg, e.g. z=1.6)
    # sensor.data.pos_w reports parent prim (base_link z≈0), NOT ray origin.
    if hasattr(sensor, "_ray_starts_w") and sensor._ray_starts_w is not None:
        ray_starts_w = sensor._ray_starts_w  # [N, R, 3]
    else:
        import warnings
        warnings.warn(
            "wd_like_sweep_72: sensor._ray_starts_w unavailable, "
            "falling back to sensor.data.pos_w (z_filter may be inaccurate)",
            stacklevel=2,
        )
        ray_starts_w = sensor.data.pos_w.unsqueeze(1).expand_as(hit_points_w)

    rel_hits_w = hit_points_w - ray_starts_w  # [N, R, 3]
    valid = torch.isfinite(hit_points_w).all(dim=-1)

    if z_filter > 0:
        valid &= rel_hits_w[..., 2].abs() <= z_filter

    yaw_quat_w = math_utils.yaw_quat(sensor_quat_w)
    yaw_quat_w = yaw_quat_w.unsqueeze(1).expand(-1, num_rays, -1)
    rel_hits_b = math_utils.quat_apply_inverse(yaw_quat_w, rel_hits_w)

    distances_2d = torch.linalg.norm(rel_hits_b[..., :2], dim=-1)
    valid &= torch.isfinite(distances_2d)

    if r_min > 0:
        valid &= distances_2d >= r_min

    distances_2d = torch.where(
        valid,
        distances_2d,
        torch.full_like(distances_2d, r_max),
    )
    distances_2d = torch.clamp(distances_2d, min=0.0, max=r_max)

    # --- Per-material: classify rays hitting DYNAMIC obstacles (=human) ---
    human_mask = None
    if human_dynamic_dropout:
        human_mask = _dynamic_human_ray_mask(
            env, hit_points_w, valid, human_obs_margin, human_move_eps, human_move_max,
        )  # [N, R] bool
        # Runtime sanity log: finite-diff needs a prior step, so log the first NONZERO
        # fraction (confirms the classifier fires) — or after 50 calls if still zero
        # (surfaces a permanent no-op instead of silently applying nothing).
        if not getattr(env, "_vlp16_human_logged", False):
            frac = human_mask.float().mean().item()
            calls = getattr(env, "_vlp16_human_calls", 0) + 1
            env._vlp16_human_calls = calls
            if frac > 0.0 or calls >= 50:
                envs_with = (human_mask.any(dim=1)).float().mean().item()
                print(
                    f"[SIM2REAL][full_material] human-ray classifier: "
                    f"frac={frac:.4f} of rays, envs-with-human={envs_with:.1%} "
                    f"(call #{calls}{'' if frac > 0 else ' — STILL ZERO, check obstacle motion'})"
                )
                env._vlp16_human_logged = True

    # --- L3 per-episode DR: resample noise scales at episode reset ---
    if displacement_std_per_meter_dr_min > 0 and displacement_std_per_meter_dr_max > displacement_std_per_meter_dr_min:
        ep_std_scale = _get_episode_noise_scale(
            env, "disp_per_meter",
            displacement_std_per_meter_dr_min,
            displacement_std_per_meter_dr_max,
        )  # [N]
        active_std_per_meter = ep_std_scale
    else:
        active_std_per_meter = displacement_std_per_meter

    if hole_rate_dr_min > 0 and hole_rate_dr_max > hole_rate_dr_min:
        ep_hole = _get_episode_noise_scale(
            env, "hole_rate",
            hole_rate_dr_min,
            hole_rate_dr_max,
        )  # [N]
        active_hole_rate = ep_hole  # [N]
    else:
        active_hole_rate = hole_rate  # scalar

    # --- L3 per-episode DR: soft-target fixed σ ---
    if displacement_std_soft_dr_min > 0 and displacement_std_soft_dr_max > displacement_std_soft_dr_min:
        ep_soft = _get_episode_noise_scale(
            env, "disp_soft",
            displacement_std_soft_dr_min,
            displacement_std_soft_dr_max,
        )  # [N]
        active_std_soft = ep_soft  # [N]
    else:
        active_std_soft = displacement_std_soft  # scalar

    # --- L1 systematic range bias: bias(d) = k·d + b ---
    # Physical model: ToF clock offset (linear with distance) + constant offset
    # Applied BEFORE displacement noise (bias acts on true distance)
    # L3 DR: per-episode 採樣 k, b（涵蓋實測 ±36mm；不假設模型形狀）
    has_k_dr = distance_bias_k_dr_min != 0.0 or distance_bias_k_dr_max != 0.0
    has_b_dr = distance_bias_b_dr_min != 0.0 or distance_bias_b_dr_max != 0.0
    has_scalar_bias = distance_bias_k != 0.0 or distance_bias_b != 0.0

    if has_k_dr or has_b_dr or has_scalar_bias:
        # 每 episode 採樣 k_eff, b_eff per env
        if has_k_dr:
            k_eff = _get_episode_noise_scale(
                env, "distance_bias_k",
                distance_bias_k_dr_min, distance_bias_k_dr_max,
            )  # [N]
        else:
            k_eff = torch.full(
                (env.num_envs,), distance_bias_k,
                device=device, dtype=dtype,
            )
        if has_b_dr:
            b_eff = _get_episode_noise_scale(
                env, "distance_bias_b",
                distance_bias_b_dr_min, distance_bias_b_dr_max,
            )  # [N]
        else:
            b_eff = torch.full(
                (env.num_envs,), distance_bias_b,
                device=device, dtype=dtype,
            )
        # [N, 1] × [N, R] + [N, 1] broadcast
        bias = k_eff.unsqueeze(1) * distances_2d + b_eff.unsqueeze(1)
        distances_2d = torch.clamp(distances_2d + bias, min=0.0, max=r_max)

    # --- L1 per-ring bias: VLP-16 16-channel calibration offsets ---
    # Measured white_wall calibration (2026-07-01, source: vlp16_noise/isaac_lab_noise_params.py,
    # LIDAR_PER_RING_BIAS). Each value = that ring's absolute systematic offset vs GT.
    # The mean (~+12.9mm) carries the common-mode ToF bias; the spread (-0.5..+23.6mm) is the
    # ring-to-ring variation. This array is the AUTHORITATIVE systematic bias — do NOT also add a
    # fixed global distance_bias_b on top (that double-counts the common-mode ~1.3-1.5cm).
    if per_ring_bias:
        cache_attr = "_lidar_per_ring_bias_cache"
        if not hasattr(env, cache_attr):
            _PER_RING_BIAS_M = torch.tensor([
                +0.004389, +0.009268, +0.018511, +0.023619, +0.004456, +0.013587, +0.014694, +0.001912,
                +0.014077, +0.023182, +0.019564, +0.019096, +0.019010, +0.016875, -0.000476, +0.005355,
            ], device=device, dtype=dtype)
            # Ray→ring mapping: LidarPatternCfg builds rays with
            # meshgrid(vertical, horizontal, indexing="ij").reshape(-1,3) → row-major, i.e.
            # ray_idx = ring * H + azimuth, so ring_id = ray_idx // H  (H = num_rays // 16).
            # Verified against isaaclab/.../ray_caster/patterns/patterns.py:lidar_pattern (2026-07-01).
            # (Was `% 16`, which scrambled each ring's azimuths across all 16 offsets — fixed.)
            H = max(1, num_rays // 16)
            ring_ids = (torch.arange(num_rays, device=device) // H).clamp_(max=15)
            ring_offsets = _PER_RING_BIAS_M[ring_ids]  # [R]
            setattr(env, cache_attr, ring_offsets)
        ring_offsets = getattr(env, cache_attr)  # [R]

        # L3 per-episode scale (e.g. U(0.5, 1.5))
        if per_ring_bias_scale_dr_min > 0 and per_ring_bias_scale_dr_max > per_ring_bias_scale_dr_min:
            ep_ring_scale = _get_episode_noise_scale(
                env, "ring_bias_scale",
                per_ring_bias_scale_dr_min,
                per_ring_bias_scale_dr_max,
            )  # [N]
            # [N, 1] × [1, R] = [N, R]
            distances_2d = distances_2d + ep_ring_scale.unsqueeze(1) * ring_offsets.unsqueeze(0)
        else:
            # scalar scale broadcast
            distances_2d = distances_2d + per_ring_bias_scale * ring_offsets.unsqueeze(0)
        distances_2d = torch.clamp(distances_2d, min=0.0, max=r_max)

    # --- L1 displacement noise (hard + soft combined via RSS) ---
    has_per_meter = displacement_std_per_meter > 0 or displacement_std_per_meter_dr_min > 0
    has_soft = displacement_std_soft > 0 or displacement_std_soft_dr_min > 0
    has_legacy = displacement_std > 0

    if has_per_meter or has_soft:
        # σ_total = sqrt(σ_per_meter(r)² + σ_soft²)
        if has_per_meter:
            if isinstance(active_std_per_meter, torch.Tensor):
                sigma_hard = active_std_per_meter.unsqueeze(1) * distances_2d  # [N, R]
            else:
                sigma_hard = active_std_per_meter * distances_2d
        else:
            sigma_hard = torch.zeros_like(distances_2d)

        if has_soft:
            if isinstance(active_std_soft, torch.Tensor):
                sigma_soft_val = active_std_soft.unsqueeze(1).expand_as(distances_2d)
            else:
                sigma_soft_val = torch.full_like(distances_2d, active_std_soft)
        else:
            sigma_soft_val = torch.zeros_like(distances_2d)

        sigma = torch.sqrt(sigma_hard ** 2 + sigma_soft_val ** 2)
        if human_mask is not None:  # human σ = null → no range jitter on human rays
            sigma = torch.where(human_mask, torch.zeros_like(sigma), sigma)
        noise = torch.randn_like(distances_2d) * sigma
        distances_2d = torch.clamp(distances_2d + noise, min=0.0, max=r_max)
    elif has_legacy:
        # legacy fixed-σ fallback
        noise = torch.randn_like(distances_2d) * displacement_std
        if human_mask is not None:
            noise = torch.where(human_mask, torch.zeros_like(noise), noise)
        distances_2d = torch.clamp(distances_2d + noise, min=0.0, max=r_max)

    if human_mask is None:
        # Uniform path (unchanged for ideal/sigma/bias/dropout/full)
        if isinstance(active_hole_rate, torch.Tensor):
            hole_mask = torch.rand_like(distances_2d) < active_hole_rate.unsqueeze(1)
        elif active_hole_rate > 0:
            hole_mask = torch.rand_like(distances_2d) < active_hole_rate
        else:
            hole_mask = None
        if hole_mask is not None:
            distances_2d = torch.where(hole_mask, r_max, distances_2d)
    else:
        # Per-ray dropout: white_wall base rate, human dropout(d)=slope·d+intercept on human rays
        if isinstance(active_hole_rate, torch.Tensor):
            p_hole = active_hole_rate.unsqueeze(1).expand_as(distances_2d).clone()
        else:
            p_hole = torch.full_like(distances_2d, float(active_hole_rate))
        human_p = torch.clamp(
            human_dropout_slope * distances_2d + human_dropout_intercept, 0.0, 1.0
        )
        p_hole = torch.where(human_mask, human_p, p_hole)
        hole_mask = torch.rand_like(distances_2d) < p_hole
        distances_2d = torch.where(hole_mask, r_max, distances_2d)

    # --- Mixed-pixel / distractor (ghost): white_wall base, higher rate on human rays ---
    has_ghost = distractor_rate > 0 or (human_mask is not None and human_mixed_pixel_rate > 0)
    if has_ghost:
        if human_mask is not None:
            p_ghost = torch.full_like(distances_2d, float(distractor_rate))
            p_ghost = torch.where(
                human_mask,
                torch.full_like(distances_2d, float(human_mixed_pixel_rate)),
                p_ghost,
            )
            distractor_mask = torch.rand_like(distances_2d) < p_ghost
        else:
            distractor_mask = torch.rand_like(distances_2d) < distractor_rate
        min_r, max_r = distractor_range
        distractor_values = torch.rand_like(distances_2d) * (max_r - min_r) + min_r
        distances_2d = torch.where(distractor_mask, distractor_values, distances_2d)

    angles = torch.atan2(rel_hits_b[..., 1], rel_hits_b[..., 0])  # [-pi, pi]
    bin_size = 2.0 * math.pi / num_bins
    bin_indices = torch.floor((angles + math.pi) / bin_size).long()
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)

    sweep = torch.full((num_envs, num_bins), r_max, device=device, dtype=dtype)
    sweep.scatter_reduce_(1, bin_indices, distances_2d, reduce="amin", include_self=True)

    # --- L2 Per-Bin block dropout: contiguous sector occlusion ---
    # Models large obstacle (pillar, leg) blocking a 15°-40° sector
    if block_dropout_prob > 0:
        trigger = torch.rand(num_envs, device=device) < block_dropout_prob
        if trigger.any():
            n_trig = int(trigger.sum().item())
            w = torch.randint(
                block_dropout_width_min, block_dropout_width_max + 1,
                (n_trig,), device=device,
            )
            start = torch.randint(0, num_bins, (n_trig,), device=device)
            trig_envs = trigger.nonzero(as_tuple=False).squeeze(-1)
            # Build a [n_trig, num_bins] mask via cumulative comparison
            bin_range = torch.arange(num_bins, device=device).unsqueeze(0)  # [1, B]
            # bin in [start, start+w) mod num_bins
            offs = (bin_range - start.unsqueeze(1)) % num_bins  # [n_trig, B]
            mask = offs < w.unsqueeze(1)  # [n_trig, B]
            sweep[trig_envs] = torch.where(mask, torch.full_like(sweep[trig_envs], r_max), sweep[trig_envs])

    sweep = torch.clamp(sweep - r_robot, min=0.0, max=r_max)
    sweep = sweep / r_max

    return sweep


# ====================================================================
# 觀測函數二：動態障礙物 Top-5 × 3D 極簡特徵
# ====================================================================

def topk_obstacles_simplified(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 5,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
    noise_std: float = 0.02,
    drop_rate: float = 0.05,
) -> torch.Tensor:
    """極簡化動態障礙物觀測：每物件僅 3 維 (dx, dy, r)。

    完全捨棄 ORCA、goal-centric 座標系等過度工程化的特徵。
    速度與加速度的推斷交給 Frame Stacking (history_length=3)。

    處理流程::

        Scene 中所有障礙物 Ground Truth
          │
          ├── 計算 Δx, Δy → 歐式距離排序
          │
          ├── Top-K = 5（不足補零）
          │
          ├── 構建 3 維特徵: [dx, dy, radius]
          │
          ├── Sim-to-Real 雜訊:
          │     • 高斯位移 N(0, σ²) on dx, dy
          │     • 5% 掉幀 → 整列歸零
          │
          └── Flatten → [N, 15]

    Note:
        障礙物索引的 for 迴圈 (max_obstacles ≤ 10) 是遍歷場景實體，
        非遍歷環境。每次迭代內部是完全向量化的 [num_envs] 維度運算。

    Args:
        env:            Isaac Lab 環境實例
        robot_cfg:      自車的 SceneEntityCfg 參照
        top_k:          保留最近的 K 個障礙物
        max_obstacles:  場景中的最大障礙物數量
        max_distance:   超過此距離的障礙物視為不存在 (m)
        noise_std:      dx, dy 高斯雜訊標準差 (m)，0 = 關閉
        drop_rate:      隨機掉幀機率，0 = 關閉

    Returns:
        [num_envs, top_k * 3] = [num_envs, 15] 扁平化特徵張量
    """
    # ----------------------------------------------------------
    # 從 Scene 取得自車位置
    # ----------------------------------------------------------
    robot = env.scene[robot_cfg.name]
    robot_pos_xy = robot.data.root_pos_w[:, :2]   # [N, 2]
    N = robot_pos_xy.shape[0]
    device = robot_pos_xy.device
    K = top_k

    # ----------------------------------------------------------
    # 收集所有障礙物的 XY 位置、可見性、半徑
    # （遍歷障礙物索引，非遍歷環境 — 每次迭代內皆向量化）
    # ----------------------------------------------------------
    all_pos = torch.zeros(N, max_obstacles, 2, device=device)
    all_radius = torch.full((N, max_obstacles), 0.3, device=device)
    all_valid = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
    num_found = 0

    for i in range(max_obstacles):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue

        obs_entity = env.scene[obs_name]
        pos_w = obs_entity.data.root_pos_w     # [N, 3] or [N, 7]

        # 可見性判斷：Z > 0 表示可見（Z = -10 表示被隱藏）
        visible = pos_w[:, 2] > 0.0            # [N]

        all_pos[:, num_found, :] = pos_w[:, :2]
        all_valid[:, num_found] = visible

        # 取得半徑（優先使用環境快取，否則預設 0.3m）
        if hasattr(env, "_obstacle_sizes") and env._obstacle_sizes is not None:
            try:
                all_radius[:, num_found] = env._obstacle_sizes[i]
            except (IndexError, TypeError):
                pass  # 保持預設值

        num_found += 1

    # ----------------------------------------------------------
    # 邊界條件：場景中沒有任何障礙物
    # ----------------------------------------------------------
    if num_found == 0:
        return torch.zeros(N, K * 3, device=device)

    # 裁剪至實際找到的數量
    pos = all_pos[:, :num_found, :]           # [N, F, 2]
    radius = all_radius[:, :num_found]        # [N, F]
    valid = all_valid[:, :num_found]          # [N, F]
    F = num_found

    # ==========================================================
    # 計算相對位置 (dx, dy) 與歐式距離
    # ==========================================================
    delta = pos - robot_pos_xy.unsqueeze(1)    # [N, F, 2]
    dx = delta[:, :, 0]                         # [N, F]
    dy = delta[:, :, 1]                         # [N, F]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # [N, F]

    # 無效 / 超距 → 排序時排到最後
    dist_for_sort = dist.clone()
    dist_for_sort[~valid] = 1e6
    dist_for_sort[dist > max_distance] = 1e6

    # ==========================================================
    # Top-K 選取（由近到遠）
    # ==========================================================
    if F >= K:
        _, topk_idx = torch.topk(dist_for_sort, k=K, dim=1, largest=False)
    else:
        # 不足 K 個：排序後用首索引填充（稍後以 valid mask 覆寫）
        _, sort_idx = torch.sort(dist_for_sort, dim=1)
        pad_idx = sort_idx[:, :1].expand(-1, K - F)
        topk_idx = torch.cat([sort_idx, pad_idx], dim=1)   # [N, K]

    # ==========================================================
    # Gather Top-K 特徵
    # ==========================================================
    topk_dx = torch.gather(dx, 1, topk_idx)         # [N, K]
    topk_dy = torch.gather(dy, 1, topk_idx)         # [N, K]
    topk_r = torch.gather(radius, 1, topk_idx)      # [N, K]
    topk_valid = torch.gather(valid.long(), 1, topk_idx).bool()  # [N, K]

    # 補齊的位置標記為無效
    if F < K:
        topk_valid = topk_valid.clone()
        topk_valid[:, F:] = False

    # 清零無效（Padding）位置的所有特徵
    v = topk_valid.float()                           # [N, K]
    topk_dx = topk_dx * v
    topk_dy = topk_dy * v
    topk_r = topk_r * v

    # ==========================================================
    # Sim-to-Real 雜訊注入
    # ==========================================================
    # (a) 高斯位移：模擬 MOT 追蹤器的估計誤差
    if noise_std > 0:
        noise_xy = torch.randn(N, K, 2, device=device) * noise_std
        topk_dx = topk_dx + noise_xy[:, :, 0] * v
        topk_dy = topk_dy + noise_xy[:, :, 1] * v

    # (b) 隨機掉幀：5% 機率將整個物件特徵歸零
    if drop_rate > 0:
        drop_mask = (torch.rand(N, K, device=device) < drop_rate) & topk_valid
        keep = (~drop_mask).float()                  # 1=保留, 0=掉幀
        topk_dx = topk_dx * keep
        topk_dy = topk_dy * keep
        topk_r = topk_r * keep

    # ==========================================================
    # 組裝並扁平化：[N, K, 3] → [N, K*3]
    # ==========================================================
    features = torch.stack([topk_dx, topk_dy, topk_r], dim=2)  # [N, K, 3]
    return features.reshape(N, -1)  # [N, 15]


# ====================================================================
# 觀測函數三：論文式 4D Ego-Centric 障礙物觀測 + LOS 遮擋
# ====================================================================

def topk_obstacles_ego_centric(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 10,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
    v_max: float = 1.5,
    wall_occlusion: bool = True,
    speed_threshold: float = 0.01,
) -> torch.Tensor:
    """論文式 ego-centric 障礙物觀測 + 牆壁遮擋檢測。

    每障礙物 4 維（論文 Eq. 3.11 / 3.15）::

        ŝ_Δx = (obs_x - robot_x) / max_distance   歸一化相對位移 X (robot frame)
        ŝ_Δy = (obs_y - robot_y) / max_distance   歸一化相對位移 Y (robot frame)
        ŝ_v  = |rel_vel| / v_max                   歸一化相對速度大小
        ŝ_θ  = atan2(rel_vy, rel_vx) / π          歸一化相對運動方向
              （速度 < speed_threshold 時強制歸零）

    座標系：
        - 位置 (dx, dy) 旋轉至機器人航向座標系（與 LiDAR 一致）
        - 速度 (vx, vy) 使用相對速度 (obs_vel - robot_vel) 並旋轉至機器人航向
        - LOS 遮擋檢測使用局部座標系（扣除 env_origins）

    處理流程::

        Scene 中所有障礙物
          │
          ├── 1. 收集 pos/vel（世界座標）
          ├── 2. 可見性掩碼: Z > 0
          ├── 3. 轉換至局部座標系 → LOS 遮擋檢測
          ├── 4. 計算相對位置 → 旋轉至 robot frame
          ├── 5. 歐式距離排序 → Top-K 選取
          ├── 6. 計算 4 維歸一化特徵（相對速度 + 航向旋轉）
          ├── 7. 靜態障礙物 heading 歸零 + 無效位置清零
          └── 8. Flatten → [N, top_k * 4]

    Args:
        env:              Isaac Lab 環境實例
        robot_cfg:        自車的 SceneEntityCfg 參照
        top_k:            保留最近的 K 個障礙物
        max_obstacles:    場景中的最大障礙物數量
        max_distance:     超過此距離的障礙物視為不存在 (m)
        v_max:            速度歸一化的最大值 (m/s)
        wall_occlusion:   是否啟用牆壁 LOS 遮擋檢測
        speed_threshold:  低於此速度的障礙物 heading 強制歸零 (m/s)

    Returns:
        [num_envs, top_k * 4] 扁平化特徵張量（預設 [N, 40]）
    """
    from ..wall_layout import check_los_perenv, get_combined_wall_data

    # ----------------------------------------------------------
    # 從 Scene 取得自車位置、速度、航向
    # ----------------------------------------------------------
    robot = env.scene[robot_cfg.name]
    robot_pos_xy = robot.data.root_pos_w[:, :2]     # [N, 2] 世界座標
    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]  # [N, 2] 世界速度
    N = robot_pos_xy.shape[0]
    device = robot_pos_xy.device
    K = top_k

    # 提取 yaw 角用於 ego-centric 旋轉
    robot_quat = robot.data.root_quat_w              # [N, 4] (w, x, y, z)
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # [N]
    cos_yaw = torch.cos(robot_yaw)                   # [N]
    sin_yaw = torch.sin(robot_yaw)                   # [N]

    # 環境原點偏移（世界座標 → 局部座標）
    env_origins = env.scene.env_origins[:, :2]        # [N, 2]

    # ----------------------------------------------------------
    # 1. 收集所有障礙物的 XY 位置、速度、可見性
    # ----------------------------------------------------------
    all_pos = torch.zeros(N, max_obstacles, 2, device=device)
    all_vel = torch.zeros(N, max_obstacles, 2, device=device)
    all_valid = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
    num_found = 0

    for i in range(max_obstacles):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue

        obs_entity = env.scene[obs_name]
        pos_w = obs_entity.data.root_pos_w     # [N, 3] or [N, 7]

        # 2. 可見性判斷：Z > 0 表示可見（Z = -10 表示被隱藏）
        visible = pos_w[:, 2] > 0.0            # [N]

        all_pos[:, num_found, :] = pos_w[:, :2]
        all_valid[:, num_found] = visible

        # 速度來源（按優先順序）
        if hasattr(env, "_obstacle_velocities") and env._obstacle_velocities is not None:
            try:
                all_vel[:, num_found, :] = env._obstacle_velocities[:, i, :2]
            except (IndexError, TypeError, RuntimeError):
                try:
                    all_vel[:, num_found, :] = obs_entity.data.root_lin_vel_w[:, :2]
                except (AttributeError, IndexError):
                    pass
        else:
            try:
                all_vel[:, num_found, :] = obs_entity.data.root_lin_vel_w[:, :2]
            except (AttributeError, IndexError):
                pass  # 保持零速度（靜態障礙物）

        num_found += 1

    # ----------------------------------------------------------
    # 邊界條件：場景中沒有任何障礙物
    # ----------------------------------------------------------
    if num_found == 0:
        return torch.zeros(N, K * 4, device=device)

    # 裁剪至實際找到的數量
    pos = all_pos[:, :num_found, :]           # [N, F, 2] 世界座標
    vel = all_vel[:, :num_found, :]           # [N, F, 2] 世界速度
    valid = all_valid[:, :num_found]          # [N, F]
    F = num_found

    # ----------------------------------------------------------
    # 3. 牆壁 LOS 遮擋檢測（per-env 牆壁, 局部座標系）
    # ----------------------------------------------------------
    if wall_occlusion:
        robot_pos_local = robot_pos_xy - env_origins           # [N, 2]
        pos_local = pos - env_origins.unsqueeze(1)             # [N, F, 2]
        wall_c, wall_s, wall_mask = get_combined_wall_data(env)
        los_visible = check_los_perenv(robot_pos_local, pos_local, wall_c, wall_s, wall_mask)
        valid = valid & los_visible                            # [N, F]

    # ----------------------------------------------------------
    # 4. 計算世界座標相對位置，再旋轉至 robot frame
    #    論文 Eq. 3.11: Δ = obs_pos - robot_pos
    #    旋轉公式: dx_r =  Δx·cos(yaw) + Δy·sin(yaw)
    #              dy_r = -Δx·sin(yaw) + Δy·cos(yaw)
    # ----------------------------------------------------------
    delta_w = pos - robot_pos_xy.unsqueeze(1)                  # [N, F, 2]
    cos_y = cos_yaw.unsqueeze(1)                               # [N, 1]
    sin_y = sin_yaw.unsqueeze(1)                               # [N, 1]

    dx = delta_w[:, :, 0] * cos_y + delta_w[:, :, 1] * sin_y   # [N, F]
    dy = -delta_w[:, :, 0] * sin_y + delta_w[:, :, 1] * cos_y  # [N, F]

    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)               # [N, F]

    # 無效 / 超距 → 排序時排到最後
    dist_for_sort = dist.clone()
    dist_for_sort[~valid] = 1e6
    dist_for_sort[dist > max_distance] = 1e6

    # ----------------------------------------------------------
    # 5. Top-K 選取（由近到遠）
    # ----------------------------------------------------------
    if F >= K:
        _, topk_idx = torch.topk(dist_for_sort, k=K, dim=1, largest=False)
    else:
        _, sort_idx = torch.sort(dist_for_sort, dim=1)
        pad_idx = sort_idx[:, :1].expand(-1, K - F)
        topk_idx = torch.cat([sort_idx, pad_idx], dim=1)      # [N, K]

    # ----------------------------------------------------------
    # 6. 計算 4 維歸一化特徵
    #    相對速度 (obs_vel - robot_vel) 旋轉至 robot frame
    # ----------------------------------------------------------
    topk_dx = torch.gather(dx, 1, topk_idx)                    # [N, K]
    topk_dy = torch.gather(dy, 1, topk_idx)                    # [N, K]

    # 相對速度（世界座標）
    rel_vel_w = vel - robot_vel_xy.unsqueeze(1)                # [N, F, 2]

    # 旋轉至 robot frame
    rel_vx_r = rel_vel_w[:, :, 0] * cos_y + rel_vel_w[:, :, 1] * sin_y   # [N, F]
    rel_vy_r = -rel_vel_w[:, :, 0] * sin_y + rel_vel_w[:, :, 1] * cos_y  # [N, F]

    topk_vx = torch.gather(rel_vx_r, 1, topk_idx)             # [N, K]
    topk_vy = torch.gather(rel_vy_r, 1, topk_idx)             # [N, K]

    topk_valid = torch.gather(valid.long(), 1, topk_idx).bool()  # [N, K]

    # 補齊的位置標記為無效
    if F < K:
        topk_valid = topk_valid.clone()
        topk_valid[:, F:] = False

    # 超距離標記為無效
    topk_dist = torch.gather(dist_for_sort, 1, topk_idx)
    topk_valid = topk_valid & (topk_dist < max_distance)

    # 歸一化
    dx_norm = topk_dx / max_distance                                   # [-1, 1]
    dy_norm = topk_dy / max_distance                                   # [-1, 1]
    speed = torch.sqrt(topk_vx ** 2 + topk_vy ** 2 + 1e-8)            # [N, K]
    speed_norm = speed / v_max                                         # [0, ~1]

    # heading: 速度 < threshold 時強制歸零（靜態障礙物無方向）
    raw_heading = torch.atan2(topk_vy, topk_vx) / torch.pi            # [-1, 1]
    heading_norm = torch.where(
        speed > speed_threshold,
        raw_heading,
        torch.zeros_like(raw_heading),
    )

    # ----------------------------------------------------------
    # 7. 無效位置清零
    # ----------------------------------------------------------
    v = topk_valid.float()                                     # [N, K]
    dx_norm = dx_norm * v
    dy_norm = dy_norm * v
    speed_norm = speed_norm * v
    heading_norm = heading_norm * v

    # ----------------------------------------------------------
    # 8. 組裝並扁平化：[N, K, 4] → [N, K*4]
    # ----------------------------------------------------------
    features = torch.stack([dx_norm, dy_norm, speed_norm, heading_norm], dim=2)
    return features.reshape(N, -1)  # [N, 40]


# ====================================================================
# 觀測函數四：論文式 Global Frame 7D 障礙物觀測
# ====================================================================

def topk_obstacles_global_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 10,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
    v_max: float = 1.5,
    wall_occlusion: bool = True,
    speed_threshold: float = 0.01,
) -> torch.Tensor:
    """論文式 Global Frame 障礙物觀測 + 牆壁遮擋檢測。

    採用固定座標系（Global/Odom Frame），不旋轉至機器人航向。
    機器人航向 θ 作為獨立觀測項輸入，讓 FC 神經網路自行學習
    「全局差值 Δx,Δy」與「車頭航向 θ」之間的空間相對關係。

    障礙物速度為絕對速度（不扣除機器人速度），與論文一致：
    速度由物件自身的位置差分計算（模擬器中等效於 root_lin_vel_w）。

    每障礙物 7 維::

        s_Δx: 歸一化全局相對位移 X = (obs_x - robot_x) / max_distance
        s_Δy: 歸一化全局相對位移 Y = (obs_y - robot_y) / max_distance
        s_θ̄:  歸一化絕對航向 = atan2(abs_vy, abs_vx) / π
        s_ρ:  障礙物類型 (0.0=靜態, 1.0=動態)
        s_o:  障礙物狀態 (1.0=可見追蹤中, 0.0=遺失/填充)
        s_r:  障礙物半徑 (m)
        s_v̄:  歸一化絕對速度 = |v| / v_max

    搭配獨立觀測項使用::

        robot_heading_normalized:  θ/π ∈ [-1, 1]  (1D)
        robot_position_local:     (px, py)/room   (2D)

    Args:
        env:              Isaac Lab 環境實例
        robot_cfg:        自車的 SceneEntityCfg 參照
        top_k:            保留最近的 K 個障礙物
        max_obstacles:    場景中的最大障礙物數量
        max_distance:     超過此距離的障礙物視為不存在 (m)
        v_max:            速度歸一化的最大值 (m/s)
        wall_occlusion:   是否啟用牆壁 LOS 遮擋檢測
        speed_threshold:  低於此速度的障礙物視為靜態 (m/s)

    Returns:
        [num_envs, top_k * 7] 扁平化特徵張量（預設 [N, 70]）
    """
    from ..wall_layout import check_los_perenv, get_combined_wall_data

    # ----------------------------------------------------------
    # 從 Scene 取得自車位置
    # ----------------------------------------------------------
    robot = env.scene[robot_cfg.name]
    robot_pos_xy = robot.data.root_pos_w[:, :2]     # [N, 2] 世界座標
    N = robot_pos_xy.shape[0]
    device = robot_pos_xy.device
    K = top_k

    # 環境原點偏移（世界座標 → 局部座標，LOS 檢測用）
    env_origins = env.scene.env_origins[:, :2]        # [N, 2]

    # ----------------------------------------------------------
    # 1. 收集所有障礙物的 XY 位置、絕對速度、半徑、可見性
    # ----------------------------------------------------------
    all_pos = torch.zeros(N, max_obstacles, 2, device=device)
    all_vel = torch.zeros(N, max_obstacles, 2, device=device)
    all_radius = torch.full((N, max_obstacles), 0.3, device=device)
    all_valid = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
    num_found = 0

    for i in range(max_obstacles):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue

        obs_entity = env.scene[obs_name]
        pos_w = obs_entity.data.root_pos_w

        # 2. 可見性判斷：Z > 0 表示可見（Z = -10 表示被隱藏）
        visible = pos_w[:, 2] > 0.0

        all_pos[:, num_found, :] = pos_w[:, :2]
        all_valid[:, num_found] = visible

        # 絕對速度（不扣除機器人速度）
        if hasattr(env, "_obstacle_velocities") and env._obstacle_velocities is not None:
            try:
                all_vel[:, num_found, :] = env._obstacle_velocities[:, i, :2]
            except (IndexError, TypeError, RuntimeError):
                try:
                    all_vel[:, num_found, :] = obs_entity.data.root_lin_vel_w[:, :2]
                except (AttributeError, IndexError):
                    pass
        else:
            try:
                all_vel[:, num_found, :] = obs_entity.data.root_lin_vel_w[:, :2]
            except (AttributeError, IndexError):
                pass

        # 半徑
        if hasattr(env, "_obstacle_sizes") and env._obstacle_sizes is not None:
            try:
                all_radius[:, num_found] = env._obstacle_sizes[i]
            except (IndexError, TypeError):
                pass

        num_found += 1

    # ----------------------------------------------------------
    # 邊界條件：場景中沒有任何障礙物
    # ----------------------------------------------------------
    if num_found == 0:
        return torch.zeros(N, K * 7, device=device)

    # 裁剪至實際找到的數量
    pos = all_pos[:, :num_found, :]           # [N, F, 2] 世界座標
    vel = all_vel[:, :num_found, :]           # [N, F, 2] 絕對速度
    radius = all_radius[:, :num_found]        # [N, F]
    valid = all_valid[:, :num_found]          # [N, F]
    F = num_found

    # ----------------------------------------------------------
    # 3. 牆壁 LOS 遮擋檢測（per-env 牆壁, 局部座標系）
    # ----------------------------------------------------------
    if wall_occlusion:
        robot_pos_local = robot_pos_xy - env_origins
        pos_local = pos - env_origins.unsqueeze(1)
        wall_c, wall_s, wall_mask = get_combined_wall_data(env)
        los_visible = check_los_perenv(robot_pos_local, pos_local, wall_c, wall_s, wall_mask)
        valid = valid & los_visible

    # ----------------------------------------------------------
    # 4. 全局座標相對位置（不旋轉）
    # ----------------------------------------------------------
    delta = pos - robot_pos_xy.unsqueeze(1)   # [N, F, 2]
    dx = delta[:, :, 0]                        # [N, F]
    dy = delta[:, :, 1]                        # [N, F]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

    dist_for_sort = dist.clone()
    dist_for_sort[~valid] = 1e6
    dist_for_sort[dist > max_distance] = 1e6

    # ----------------------------------------------------------
    # 5. Top-K 選取（由近到遠）
    # ----------------------------------------------------------
    if F >= K:
        _, topk_idx = torch.topk(dist_for_sort, k=K, dim=1, largest=False)
    else:
        _, sort_idx = torch.sort(dist_for_sort, dim=1)
        pad_idx = sort_idx[:, :1].expand(-1, K - F)
        topk_idx = torch.cat([sort_idx, pad_idx], dim=1)

    # ----------------------------------------------------------
    # 6. 計算 7 維特徵
    # ----------------------------------------------------------
    topk_dx = torch.gather(dx, 1, topk_idx)
    topk_dy = torch.gather(dy, 1, topk_idx)
    topk_vx = torch.gather(vel[:, :, 0], 1, topk_idx)
    topk_vy = torch.gather(vel[:, :, 1], 1, topk_idx)
    topk_r = torch.gather(radius, 1, topk_idx)
    topk_valid = torch.gather(valid.long(), 1, topk_idx).bool()

    if F < K:
        topk_valid = topk_valid.clone()
        topk_valid[:, F:] = False

    topk_dist = torch.gather(dist_for_sort, 1, topk_idx)
    topk_valid = topk_valid & (topk_dist < max_distance)

    # s_Δx, s_Δy: 歸一化全局差值（clamp 保證 [-1, 1]）
    dx_norm = (topk_dx / max_distance).clamp(-1.0, 1.0)
    dy_norm = (topk_dy / max_distance).clamp(-1.0, 1.0)

    # s_v̄: 絕對速度大小歸一化
    speed = torch.sqrt(topk_vx ** 2 + topk_vy ** 2 + 1e-8)
    speed_norm = speed / v_max

    # s_θ̄: 絕對航向（靜態障礙物強制歸零）
    raw_heading = torch.atan2(topk_vy, topk_vx) / torch.pi
    heading_norm = torch.where(
        speed > speed_threshold,
        raw_heading,
        torch.zeros_like(raw_heading),
    )

    # s_ρ: 障礙物類型（靜態=0, 動態=1）
    obs_type = torch.where(
        speed > speed_threshold,
        torch.ones_like(speed),
        torch.zeros_like(speed),
    )

    # s_o: 障礙物狀態（1.0=可見追蹤中, 0.0=遺失/填充）
    obs_status = topk_valid.float()

    # ----------------------------------------------------------
    # 7. 無效位置清零（s_o 本身已為 0.0）
    # ----------------------------------------------------------
    v = topk_valid.float()
    dx_norm = dx_norm * v
    dy_norm = dy_norm * v
    heading_norm = heading_norm * v
    obs_type = obs_type * v
    topk_r = topk_r * v
    speed_norm = speed_norm * v

    # ----------------------------------------------------------
    # 8. 組裝並扁平化：[N, K, 7] → [N, K*7]
    #    順序: [s_Δx, s_Δy, s_θ̄, s_ρ, s_o, s_r, s_v̄]
    # ----------------------------------------------------------
    features = torch.stack([
        dx_norm,       # s_Δx
        dy_norm,       # s_Δy
        heading_norm,  # s_θ̄
        obs_type,      # s_ρ
        obs_status,    # s_o
        topk_r,        # s_r
        speed_norm,    # s_v̄
    ], dim=2)
    return features.reshape(N, -1)  # [N, 70]


# ====================================================================
# 觀測函數四-B：Body Frame 6D 障礙物觀測
# ====================================================================

def topk_obstacles_body_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 10,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
    v_max: float = 1.5,
    wall_occlusion: bool = True,
    speed_threshold: float = 0.01,
) -> torch.Tensor:
    """Body Frame 障礙物觀測 — 位置與速度皆轉至車體座標系。

    所有位置、速度特徵以車體為原點、車頭為 +x 方向：
    - 位置：世界座標差值 → 旋轉至 body frame
    - 速度：扣除機器人自身速度後 → 旋轉至 body frame
    - 半徑、動靜態標記不變
    - s_o 有效位元：區分「真實障礙物」與「填充空位」

    每障礙物 7 維::

        p_x^body:  縱向距離 (+前方, -後方)，歸一化 / max_distance
        p_y^body:  橫向距離 (+左側, -右側)，歸一化 / max_distance
        v_x^body:  縱向相對速度 (-代表正在接近)，歸一化 / v_max
        v_y^body:  橫向相對速度 (側向切入威脅)，歸一化 / v_max
        r_j:       障礙物半徑 (m)
        s_j:       動靜態標記 (0.0=靜態, 1.0=動態)
        s_o:       有效位元 (1.0=可見追蹤中, 0.0=遺失/填充/遮擋)

    s_o 的必要性（Body Frame 特有）::

        在 Body Frame 中，[0,0,0,0,0,0] 既可能是「空位」也可能是
        「障礙物在 (0,0) = 正在碰撞」。s_o 消除此歧義。

        此外，permutation invariant MaxPool 會把 padding 的 0 值
        蓋過合法的負值（如後方牆壁 p_x=-5），s_o 讓 NN 學會
        在 s_o=0 時抑制該 slot 的貢獻。

    座標轉換::

        Body Frame 旋轉矩陣 (θ = robot yaw):
          p_x^body =  cos(θ)·Δx + sin(θ)·Δy
          p_y^body = -sin(θ)·Δx + cos(θ)·Δy

        相對速度:
          rel_vx = obs_vx - robot_vx
          rel_vy = obs_vy - robot_vy
          v_x^body =  cos(θ)·rel_vx + sin(θ)·rel_vy
          v_y^body = -sin(θ)·rel_vx + cos(θ)·rel_vy

    Args:
        env:              Isaac Lab 環境實例
        robot_cfg:        自車的 SceneEntityCfg 參照
        top_k:            保留最近的 K 個障礙物
        max_obstacles:    場景中的最大障礙物數量
        max_distance:     超過此距離的障礙物視為不存在 (m)
        v_max:            速度歸一化的最大值 (m/s)
        wall_occlusion:   是否啟用牆壁 LOS 遮擋檢測
        speed_threshold:  低於此速度的障礙物視為靜態 (m/s)

    Returns:
        [num_envs, top_k * 7] 扁平化特徵張量（預設 [N, 70]）
    """
    from ..wall_layout import check_los_perenv, get_combined_wall_data

    # ----------------------------------------------------------
    # 從 Scene 取得自車狀態
    # ----------------------------------------------------------
    robot = env.scene[robot_cfg.name]
    robot_pos_xy = robot.data.root_pos_w[:, :2]     # [N, 2] 世界座標
    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]  # [N, 2] 世界座標速度
    N = robot_pos_xy.shape[0]
    device = robot_pos_xy.device
    K = top_k

    # 機器人航向 θ（yaw）
    quat = robot.data.root_quat_w                    # [N, 4] (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))  # [N]
    cos_yaw = torch.cos(yaw)                          # [N]
    sin_yaw = torch.sin(yaw)                          # [N]

    # 環境原點偏移（LOS 檢測用）
    env_origins = env.scene.env_origins[:, :2]        # [N, 2]

    # ----------------------------------------------------------
    # 1. 收集所有障礙物的 XY 位置、絕對速度、半徑、可見性
    # ----------------------------------------------------------
    all_pos = torch.zeros(N, max_obstacles, 2, device=device)
    all_vel = torch.zeros(N, max_obstacles, 2, device=device)
    all_radius = torch.full((N, max_obstacles), 0.3, device=device)
    all_valid = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
    num_found = 0

    for i in range(max_obstacles):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue

        obs_entity = env.scene[obs_name]
        pos_w = obs_entity.data.root_pos_w

        # 可見性判斷：Z > 0 表示可見（Z = -10 表示被隱藏）
        visible = pos_w[:, 2] > 0.0

        # NaN protection: scene entity data may contain NaN after resets/collisions
        pos_xy = torch.nan_to_num(pos_w[:, :2], nan=0.0)
        all_pos[:, num_found, :] = pos_xy
        all_valid[:, num_found] = visible & ~torch.isnan(pos_w[:, 0]) & ~torch.isnan(pos_w[:, 1])

        # 絕對速度
        if hasattr(env, "_obstacle_velocities") and env._obstacle_velocities is not None:
            try:
                vel_data = env._obstacle_velocities[:, i, :2]
                all_vel[:, num_found, :] = torch.nan_to_num(vel_data, nan=0.0)
            except (IndexError, TypeError, RuntimeError):
                try:
                    vel_data = obs_entity.data.root_lin_vel_w[:, :2]
                    all_vel[:, num_found, :] = torch.nan_to_num(vel_data, nan=0.0)
                except (AttributeError, IndexError):
                    pass
        else:
            try:
                vel_data = obs_entity.data.root_lin_vel_w[:, :2]
                all_vel[:, num_found, :] = torch.nan_to_num(vel_data, nan=0.0)
            except (AttributeError, IndexError):
                pass

        # 半徑
        if hasattr(env, "_obstacle_sizes") and env._obstacle_sizes is not None:
            try:
                all_radius[:, num_found] = env._obstacle_sizes[i]
            except (IndexError, TypeError):
                pass

        num_found += 1

    # ----------------------------------------------------------
    # 邊界條件：場景中沒有任何障礙物
    # ----------------------------------------------------------
    if num_found == 0:
        return torch.zeros(N, K * 7, device=device)

    # 裁剪至實際找到的數量
    pos = all_pos[:, :num_found, :]           # [N, F, 2]
    vel = all_vel[:, :num_found, :]           # [N, F, 2]
    radius = all_radius[:, :num_found]        # [N, F]
    valid = all_valid[:, :num_found]          # [N, F]
    F = num_found

    # ----------------------------------------------------------
    # 2. 牆壁 LOS 遮擋檢測（per-env 牆壁, 局部座標系）
    # ----------------------------------------------------------
    if wall_occlusion:
        robot_pos_local = robot_pos_xy - env_origins
        pos_local = pos - env_origins.unsqueeze(1)
        wall_c, wall_s, wall_mask = get_combined_wall_data(env)
        los_visible = check_los_perenv(robot_pos_local, pos_local, wall_c, wall_s, wall_mask)
        valid = valid & los_visible

    # ----------------------------------------------------------
    # 3. 世界座標相對位置（用於排序）
    # ----------------------------------------------------------
    delta = pos - robot_pos_xy.unsqueeze(1)   # [N, F, 2]
    dx = delta[:, :, 0]                        # [N, F]
    dy = delta[:, :, 1]                        # [N, F]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

    dist_for_sort = dist.clone()
    dist_for_sort[~valid] = 1e6
    dist_for_sort[dist > max_distance] = 1e6

    # ----------------------------------------------------------
    # 4. Top-K 選取（由近到遠）
    # ----------------------------------------------------------
    if F >= K:
        _, topk_idx = torch.topk(dist_for_sort, k=K, dim=1, largest=False)
    else:
        _, sort_idx = torch.sort(dist_for_sort, dim=1)
        pad_idx = sort_idx[:, :1].expand(-1, K - F)
        topk_idx = torch.cat([sort_idx, pad_idx], dim=1)

    # ----------------------------------------------------------
    # 5. Gather Top-K 的原始數據
    # ----------------------------------------------------------
    topk_dx = torch.gather(dx, 1, topk_idx)           # [N, K]
    topk_dy = torch.gather(dy, 1, topk_idx)           # [N, K]
    topk_vx = torch.gather(vel[:, :, 0], 1, topk_idx) # [N, K] 絕對速度
    topk_vy = torch.gather(vel[:, :, 1], 1, topk_idx) # [N, K]
    topk_r = torch.gather(radius, 1, topk_idx)        # [N, K]
    topk_valid = torch.gather(valid.long(), 1, topk_idx).bool()

    if F < K:
        topk_valid = topk_valid.clone()
        topk_valid[:, F:] = False

    topk_dist = torch.gather(dist_for_sort, 1, topk_idx)
    topk_valid = topk_valid & (topk_dist < max_distance)

    # ----------------------------------------------------------
    # 6. 旋轉至 Body Frame
    # ----------------------------------------------------------
    # 位置：世界差值 → body frame
    cos_y = cos_yaw.unsqueeze(1)   # [N, 1]
    sin_y = sin_yaw.unsqueeze(1)   # [N, 1]

    px_body = cos_y * topk_dx + sin_y * topk_dy       # [N, K] 縱向 (+前)
    py_body = -sin_y * topk_dx + cos_y * topk_dy      # [N, K] 橫向 (+左)

    # 相對速度：扣除機器人速度 → body frame
    rel_vx = topk_vx - robot_vel_xy[:, 0:1]           # [N, K] 世界相對 vx
    rel_vy = topk_vy - robot_vel_xy[:, 1:2]           # [N, K] 世界相對 vy

    vx_body = cos_y * rel_vx + sin_y * rel_vy         # [N, K] 縱向 (-接近)
    vy_body = -sin_y * rel_vx + cos_y * rel_vy        # [N, K] 橫向 (切入)

    # ----------------------------------------------------------
    # 7. 歸一化
    # ----------------------------------------------------------
    px_norm = (px_body / max_distance).clamp(-1.0, 1.0)
    py_norm = (py_body / max_distance).clamp(-1.0, 1.0)
    vx_norm = (vx_body / v_max).clamp(-2.0, 2.0)
    vy_norm = (vy_body / v_max).clamp(-2.0, 2.0)

    # s_j: 動靜態標記（依據絕對速度判斷）
    abs_speed = torch.sqrt(topk_vx ** 2 + topk_vy ** 2 + 1e-8)
    type_flag = torch.where(
        abs_speed > speed_threshold,
        torch.ones_like(abs_speed),
        torch.zeros_like(abs_speed),
    )

    # ----------------------------------------------------------
    # 8. s_o 有效位元（不清零！這是給 NN 的遮罩信號）
    #    s_o=1 → 真實障礙物，s_o=0 → 填充/遮擋/遺失
    # ----------------------------------------------------------
    obs_status = topk_valid.float()                   # [N, K]

    # 無效 slot 的物理特徵清零（但 s_o 保留原值）
    # 使用 torch.where 而非乘法，因為 NaN * 0 = NaN (IEEE 754)
    valid_mask = obs_status.bool()
    zero = torch.zeros_like(px_norm)
    px_norm = torch.where(valid_mask, px_norm, zero)
    py_norm = torch.where(valid_mask, py_norm, zero)
    vx_norm = torch.where(valid_mask, vx_norm, zero)
    vy_norm = torch.where(valid_mask, vy_norm, zero)
    topk_r = torch.where(valid_mask, topk_r, zero)
    type_flag = torch.where(valid_mask, type_flag, zero)

    # ----------------------------------------------------------
    # 9. 組裝並扁平化：[N, K, 7] → [N, K*7]
    #    順序: [p_x^body, p_y^body, v_x^body, v_y^body, r_j, s_j, s_o]
    # ----------------------------------------------------------
    features = torch.stack([
        px_norm,      # p_x^body (縱向距離)
        py_norm,      # p_y^body (橫向距離)
        vx_norm,      # v_x^body (縱向相對速度)
        vy_norm,      # v_y^body (橫向相對速度)
        topk_r,       # r_j      (半徑)
        type_flag,    # s_j      (動靜態標記)
        obs_status,   # s_o      (有效位元 — 不清零！)
    ], dim=2)
    # Final NaN safety net — any residual NaN gets zeroed
    return torch.nan_to_num(features.reshape(N, -1), nan=0.0)  # [N, 70]


def topk_obstacles_6d(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 10,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
    v_max: float = 1.5,
    wall_occlusion: bool = True,
    speed_threshold: float = 0.01,
) -> torch.Tensor:
    """v2 動態障礙物觀測 — 每物件 6 維 (x, y, vx, vy, r, m)。

    封裝 topk_obstacles_body_frame 並去掉 s_j (動靜態標記)，
    保留 6 維特徵：

        o_i = [x_i^robot, y_i^robot, vx_i^robot, vy_i^robot, r_i, m_i]

    其中 m_i 為有效位元 (1=可見, 0=填充/遮擋)。
    不可見的物件特徵全部為 0（m_i=0 時 x,y,vx,vy,r 也為 0）。

    Args:
        env:              Isaac Lab 環境實例
        robot_cfg:        自車的 SceneEntityCfg 參照
        top_k:            保留最近的 K 個障礙物
        max_obstacles:    場景中的最大障礙物數量
        max_distance:     最大觀測距離 (m)
        v_max:            速度歸一化上限 (m/s)
        wall_occlusion:   是否啟用牆壁 LOS 遮擋
        speed_threshold:  靜態判定閾值 (m/s)

    Returns:
        [num_envs, top_k * 6] = [num_envs, 60] 扁平化特徵張量
    """
    # 取得 7D body-frame 特徵 [N, K*7]
    feat_7d = topk_obstacles_body_frame(
        env, robot_cfg, top_k, max_obstacles,
        max_distance, v_max, wall_occlusion, speed_threshold,
    )
    N = feat_7d.shape[0]
    K = top_k
    feat = feat_7d.view(N, K, 7)  # [N, K, 7]

    # 取出 6D：[px, py, vx, vy, r, s_o] — 跳過 index 5 (s_j 動靜態標記)
    feat_6d = torch.cat([
        feat[:, :, :5],   # px, py, vx, vy, r
        feat[:, :, 6:7],  # s_o (有效位元 = mask m_i)
    ], dim=2)  # [N, K, 6]

    return feat_6d.reshape(N, K * 6)  # [N, 60]


def dynamic_obstacles_lvdot(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 5,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
    v_max: float = 1.5,
    wall_occlusion: bool = True,
    dynamic_speed_threshold: float = 0.3,   # LV-DOT dynamic_velocity_threshold=0.3 m/s (實測 config)
    fov_deg: float = 360.0,                 # 360 = 不 cull（LiDAR 全向）；camera 前向可縮
    # ── 中度 DR（SA1 全 0；SA2+ 由 env._lvdot_dr 覆寫，仿 _apply_lidar_noise_config）──
    pos_noise_std: float = 0.0,             # 位置高斯噪 σ (m)
    vel_noise_std: float = 0.0,             # 速度高斯噪 σ (m/s)
    dropout_prob: float = 0.0,              # per-slot 漏偵機率（valid→0）
    false_neg_prob: float = 0.0,            # per-candidate 整幀漏偵機率
) -> torch.Tensor:
    """LV-DOT 動態障礙 channel — K 最近『動態』物 × [px,py,vx,vy,r,valid] body frame。

    對齊車端管線 LV-DOT(onboard_detector) → vo_interface → /vo_interface/tracked_obstacles：
      - **只報動態物**（LV-DOT dynamic_bboxes 只出 >0.2 m/s 移動物；靜態走 LiDAR）
      - 含真速度（vo_interface CV-Kalman 平滑後絕對速度）
      - body frame [縱向 px(+前), 橫向 py(+左), 相對 vx(-接近), 相對 vy(切入), 半徑 r, valid]

    與 topk_obstacles_6d 差異（LV-DOT 保真）：
      1. 候選池先過濾『動態』（abs_speed > dynamic_speed_threshold），靜態不佔 slot。
      2. 中度 DR：位置/速度高斯噪 + per-slot dropout + FOV/range cull + false-negative。
         SA1 全 clean；SA2+ 由 `env._lvdot_dr`（dict）覆寫，curriculum-gated。

    Returns:
        [num_envs, top_k * 6] 扁平化（預設 [N, 30]，K=5）。
    """
    # ── env-level DR 覆寫（curriculum 設定，仿 lidar noise config）──
    _dr = getattr(env, "_lvdot_dr", None)
    if isinstance(_dr, dict):
        pos_noise_std = _dr.get("pos_noise_std", pos_noise_std)
        vel_noise_std = _dr.get("vel_noise_std", vel_noise_std)
        dropout_prob = _dr.get("dropout_prob", dropout_prob)
        false_neg_prob = _dr.get("false_neg_prob", false_neg_prob)
        max_distance = _dr.get("max_distance", max_distance)
        fov_deg = _dr.get("fov_deg", fov_deg)

    robot = env.scene[robot_cfg.name]
    robot_pos_xy = robot.data.root_pos_w[:, :2]
    robot_vel_xy = robot.data.root_lin_vel_w[:, :2]
    N = robot_pos_xy.shape[0]
    device = robot_pos_xy.device
    K = top_k

    quat = robot.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    env_origins = env.scene.env_origins[:, :2]

    # ── 1. 收集所有障礙物 ──
    all_pos = torch.zeros(N, max_obstacles, 2, device=device)
    all_radius = torch.full((N, max_obstacles), 0.3, device=device)
    all_valid = torch.zeros(N, max_obstacles, dtype=torch.bool, device=device)
    num_found = 0
    for i in range(max_obstacles):
        obs_name = f"obstacle_{i}"
        if obs_name not in env.scene.keys():
            continue
        obs_entity = env.scene[obs_name]
        pos_w = obs_entity.data.root_pos_w
        visible = pos_w[:, 2] > 0.0
        all_pos[:, num_found, :] = torch.nan_to_num(pos_w[:, :2], nan=0.0)
        all_valid[:, num_found] = visible & ~torch.isnan(pos_w[:, 0]) & ~torch.isnan(pos_w[:, 1])
        # 速度改用 finite-diff (見迴圈後)：rule_based/BehaviorScheduler 的 kinematic 障礙
        # 是 pose-write 驅動 → _obstacle_velocities / root_lin_vel_w 恆 ~0，讀了會讓動態過濾全滅。
        # 半徑優先讀 per-env 物理半徑 (_obstacle_phys_radii，尺寸隨機化)，否則 fallback scalar sizes。
        _phys = getattr(env, "_obstacle_phys_radii", None)
        if _phys is not None:
            try:
                all_radius[:, num_found] = _phys[:, i]
            except (IndexError, TypeError, RuntimeError):
                pass
        elif hasattr(env, "_obstacle_sizes") and env._obstacle_sizes is not None:
            try:
                all_radius[:, num_found] = env._obstacle_sizes[i]
            except (IndexError, TypeError):
                pass
        num_found += 1

    if num_found == 0:
        return torch.zeros(N, K * 6, device=device)

    pos = all_pos[:, :num_found, :]
    radius = all_radius[:, :num_found]
    valid = all_valid[:, :num_found]
    F = num_found

    # ── 1b. finite-diff 速度（kinematic 障礙 _obstacle_velocities≈0 → 自算）──
    # 用自有 prev-pos cache（wd_like_sweep_72 的 _matnoise_prev_obs_xy 已被它更新成本步，
    # 且 shape 不同）。step-memo 防 policy+critic 同步雙呼叫時第二次 Δ=0。
    # teleport 排除：單步位移 > move_max 視為 reset 傳送 → vel=0。
    _dt = float(getattr(env, "step_dt", 0.2))
    _move_max = 2.0
    _sid = getattr(env, "common_step_counter", None)
    if _sid is not None and getattr(env, "_lvdot_fd_step", None) == _sid:
        _vm = getattr(env, "_lvdot_fd_vel", None)
        vel = _vm if (_vm is not None and _vm.shape == pos.shape) else torch.zeros_like(pos)
    else:
        _prev = getattr(env, "_lvdot_prev_obs_xy", None)
        if _prev is not None and _prev.shape == pos.shape:
            _disp = pos - _prev
            _dn = torch.linalg.norm(_disp, dim=-1, keepdim=True)
            vel = torch.where(_dn < _move_max, _disp / _dt, torch.zeros_like(_disp))
        else:
            vel = torch.zeros_like(pos)
        env._lvdot_prev_obs_xy = pos.detach().clone()
        if _sid is not None:
            env._lvdot_fd_step = _sid
            env._lvdot_fd_vel = vel.detach().clone()

    # ── 2. 動態過濾（LV-DOT 只報移動物）──
    abs_speed_cand = torch.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2 + 1e-8)
    valid = valid & (abs_speed_cand > dynamic_speed_threshold)

    # ── 3. 牆壁 LOS 遮擋 ──
    if wall_occlusion:
        from ..wall_layout import check_los_perenv, get_combined_wall_data
        robot_pos_local = robot_pos_xy - env_origins
        pos_local = pos - env_origins.unsqueeze(1)
        wall_c, wall_s, wall_mask = get_combined_wall_data(env)
        los_visible = check_los_perenv(robot_pos_local, pos_local, wall_c, wall_s, wall_mask)
        valid = valid & los_visible

    # ── 4. 相對位置 + FOV cull ──
    delta = pos - robot_pos_xy.unsqueeze(1)
    dx, dy = delta[:, :, 0], delta[:, :, 1]
    dist = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
    if fov_deg < 360.0:
        bearing = torch.atan2(dy, dx) - yaw.unsqueeze(1)
        bearing = torch.atan2(torch.sin(bearing), torch.cos(bearing))  # wrap [-π,π]
        valid = valid & (bearing.abs() <= math.radians(fov_deg * 0.5))

    # ── 5. false-negative（整候選漏偵）──
    if false_neg_prob > 0.0:
        valid = valid & (torch.rand(N, F, device=device) >= false_neg_prob)

    dist_for_sort = dist.clone()
    dist_for_sort[~valid] = 1e6
    dist_for_sort[dist > max_distance] = 1e6

    # ── 6. Top-K 最近 ──
    if F >= K:
        _, topk_idx = torch.topk(dist_for_sort, k=K, dim=1, largest=False)
    else:
        _, sort_idx = torch.sort(dist_for_sort, dim=1)
        pad_idx = sort_idx[:, :1].expand(-1, K - F)
        topk_idx = torch.cat([sort_idx, pad_idx], dim=1)

    topk_dx = torch.gather(dx, 1, topk_idx)
    topk_dy = torch.gather(dy, 1, topk_idx)
    topk_vx = torch.gather(vel[:, :, 0], 1, topk_idx)
    topk_vy = torch.gather(vel[:, :, 1], 1, topk_idx)
    topk_r = torch.gather(radius, 1, topk_idx)
    topk_valid = torch.gather(valid.long(), 1, topk_idx).bool()
    if F < K:
        topk_valid = topk_valid.clone()
        topk_valid[:, F:] = False
    topk_dist = torch.gather(dist_for_sort, 1, topk_idx)
    topk_valid = topk_valid & (topk_dist < max_distance)

    # ── 7. 中度 DR：位置/速度高斯噪（world 差值 & 絕對速度上加，隨後轉 body）──
    if pos_noise_std > 0.0:
        topk_dx = topk_dx + torch.randn_like(topk_dx) * pos_noise_std
        topk_dy = topk_dy + torch.randn_like(topk_dy) * pos_noise_std
    if vel_noise_std > 0.0:
        topk_vx = topk_vx + torch.randn_like(topk_vx) * vel_noise_std
        topk_vy = topk_vy + torch.randn_like(topk_vy) * vel_noise_std

    # ── 8. 轉 body frame ──
    cos_y, sin_y = cos_yaw.unsqueeze(1), sin_yaw.unsqueeze(1)
    px_body = cos_y * topk_dx + sin_y * topk_dy
    py_body = -sin_y * topk_dx + cos_y * topk_dy
    rel_vx = topk_vx - robot_vel_xy[:, 0:1]
    rel_vy = topk_vy - robot_vel_xy[:, 1:2]
    vx_body = cos_y * rel_vx + sin_y * rel_vy
    vy_body = -sin_y * rel_vx + cos_y * rel_vy

    # ── 9. 歸一化 ──
    px_norm = (px_body / max_distance).clamp(-1.0, 1.0)
    py_norm = (py_body / max_distance).clamp(-1.0, 1.0)
    vx_norm = (vx_body / v_max).clamp(-2.0, 2.0)
    vy_norm = (vy_body / v_max).clamp(-2.0, 2.0)

    # ── 10. per-slot dropout（漏偵：valid→0）──
    obs_status = topk_valid.float()
    if dropout_prob > 0.0:
        keep = (torch.rand(N, K, device=device) >= dropout_prob).float()
        obs_status = obs_status * keep

    valid_mask = obs_status.bool()
    zero = torch.zeros_like(px_norm)
    px_norm = torch.where(valid_mask, px_norm, zero)
    py_norm = torch.where(valid_mask, py_norm, zero)
    vx_norm = torch.where(valid_mask, vx_norm, zero)
    vy_norm = torch.where(valid_mask, vy_norm, zero)
    topk_r = torch.where(valid_mask, topk_r, zero)

    # ── 11. 組裝 [N, K, 6]: [px, py, vx, vy, r, valid] ──
    features = torch.stack([px_norm, py_norm, vx_norm, vy_norm, topk_r, obs_status], dim=2)
    return torch.nan_to_num(features.reshape(N, -1), nan=0.0)  # [N, K*6]


# ====================================================================
# 觀測函數五：機器人航向角（Global Frame 必要輸入）
# ====================================================================

def robot_heading_normalized(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """機器人絕對航向角，歸一化至 [-1, 1]。

    Global Frame 障礙物觀測的必要搭配項：
    神經網路需要知道車頭朝向，才能理解全局差值 Δx,Δy 的空間意義。

    Returns:
        [num_envs, 1] 歸一化航向 θ/π ∈ [-1, 1]
    """
    robot = env.scene[asset_cfg.name]
    quat = robot.data.root_quat_w              # [N, 4] (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return (yaw / torch.pi).unsqueeze(-1)      # [N, 1]


# ====================================================================
# 觀測函數六：機器人局部座標位置（Global Frame 必要輸入）
# ====================================================================

def robot_position_local(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    room_half_size: float = 8.0,
) -> torch.Tensor:
    """機器人在局部座標系中的位置，歸一化至 [-1, 1]。

    扣除 env_origins 得到 16×16m 房間內的座標，
    除以 room_half_size 歸一化。

    Returns:
        [num_envs, 2] 歸一化位置 (px, py) ∈ [-1, 1]
    """
    robot = env.scene[asset_cfg.name]
    pos_w = robot.data.root_pos_w[:, :2]       # [N, 2]
    env_origins = env.scene.env_origins[:, :2]  # [N, 2]
    pos_local = pos_w - env_origins             # [N, 2]
    return pos_local / room_half_size           # [N, 2]


# ====================================================================
# 觀測函數七：離散動作空間裁切後動作
# ====================================================================

def discrete_applied_action(
    env: ManagerBasedRLEnv,
    a_max: float = 0.2,
    omega_max: float = math.pi / 15.0,
) -> torch.Tensor:
    """讀取離散動作空間的裁切後動作 [applied_a, applied_ω]。

    從 ActionTerm.processed_actions 讀取（非 raw NN 輸出），
    歸一化至約 [-1, 1] 範圍。

    歸一化方式::

        a_norm = applied_a / a_max        (a_max = 0.2 → max accel)
        ω_norm = applied_ω / omega_max    (omega_max = π/15)

    注意：加速度 -0.1 歸一化後為 -0.5，不完美對稱但保留物理意義。

    Args:
        env:       Isaac Lab 環境實例
        a_max:     線加速度最大絕對值（用於歸一化）
        omega_max: 角速度最大絕對值（用於歸一化）

    Returns:
        [num_envs, 2] — 歸一化後的 [a_norm, ω_norm]
    """
    term = list(env.action_manager._terms.values())[0]

    # 優先讀取 applied_accelerations（離散動作空間專用）
    # 若不存在（連續動作空間），則 fallback 到 processed_actions
    if hasattr(term, "applied_accelerations"):
        pa = term.applied_accelerations  # [N, 2] = [applied_a, ω]
    else:
        pa = term.processed_actions      # [N, 2] fallback

    result = torch.zeros_like(pa)
    result[:, 0] = pa[:, 0] / a_max          # [-0.1, 0.2] / 0.2 → [-0.5, 1.0]
    result[:, 1] = pa[:, 1] / omega_max      # [-π/15, π/15] / (π/15) → [-1, 1]

    return torch.nan_to_num(result, nan=0.0).clamp(-2.0, 2.0)


# ====================================================================
# 觀測函數七.b：動作歷史堆疊 (v3c, 2026-06-09)
# ====================================================================

def discrete_applied_action_history(
    env: ManagerBasedRLEnv,
    stack_size: int = 2,
    a_max: float = 0.2,
    omega_max: float = math.pi / 15.0,
) -> torch.Tensor:
    """讀取最近 stack_size 步的 issued actions 並堆疊成時序矩陣。

    動機 (v3c anti-jitter):
        Vanilla RNN64 容量有限 (139D obs → 64D hidden) 難以同時過濾 LiDAR 噪聲
        + 精確記住前 N 步動作意圖。把 a_{t-1}, a_{t-2} 顯式塞入 obs，給 policy
        「剎車訊號」: 它能直接看見自己剛下達的指令，做出阻尼修正 (damping)。

    布局：
        返回 [N, stack_size * 2] = [N, 2*stack_size] 維度
        每 2D 為 [a_norm_t-i, ω_norm_t-i]，最新→最舊排列
        例如 stack_size=2: [a_t-1, ω_t-1, a_t-2, ω_t-2]

    機制：
        - 維護 env._action_history_buffer (deque maxlen=stack_size)
        - 每次 step 結束後讀取 term.commanded_accelerations 並 push
        - episode reset 時對應 env 的歷史清零
        - 首次 call 用 zeros 初始化（policy 看到 [0, 0, 0, 0] 沒問題）

    Args:
        env:        Isaac Lab 環境
        stack_size: 堆疊步數，預設 2（共 4D）
        a_max:      線加速度 normalize 上限
        omega_max:  角速度 normalize 上限

    Returns:
        [num_envs, stack_size * 2] — 歸一化後的動作時序
    """
    term = list(env.action_manager._terms.values())[0]

    # 延遲 MDP 需要的是最近發出的命令 queue，而不是已延遲執行的輸出。
    # 無 actuator DR 時 commanded == applied，既有 83D checkpoint 行為不變。
    if hasattr(term, "commanded_accelerations"):
        pa = term.commanded_accelerations  # [N, 2] = [issued_a, issued_omega]
    elif hasattr(term, "applied_accelerations"):
        pa = term.applied_accelerations  # [N, 2] = [applied_a, ω]
    else:
        pa = term.processed_actions

    N = pa.shape[0]
    device = pa.device

    # 歸一化（與 discrete_applied_action 一致）
    current = torch.zeros_like(pa)
    current[:, 0] = pa[:, 0] / a_max
    current[:, 1] = pa[:, 1] / omega_max
    current = torch.nan_to_num(current, nan=0.0).clamp(-2.0, 2.0)

    # 初始化 buffer (首次 call)
    if not hasattr(env, "_action_history_buffer") or env._action_history_buffer is None:
        env._action_history_buffer = [
            torch.zeros((N, 2), device=device) for _ in range(stack_size)
        ]
        env._action_history_buffer_size = stack_size

    # episode reset 時對應 env 的歷史清零
    just_reset = env.episode_length_buf == 0
    if just_reset.any():
        for buf in env._action_history_buffer:
            buf[just_reset] = 0.0

    # 取出歷史（最新 → 最舊）
    # buffer[0] = t-1, buffer[1] = t-2, ...
    #
    # v3d (2026-06-12): action history 編碼模式，由環境變數 CHARGE_ACT_HIST_MODE 控制
    #   "raw"          : [a_{t-1}, ω_{t-1}, a_{t-2}, ω_{t-2}]（v3c 預設，向後相容舊 ckpt）
    #   "delta"        : [a_{t-1}, ω_{t-1}, a_{t-1}-a_{t-2}, ω_{t-1}-ω_{t-2}]（動作變化量）
    #   "action_error" : [a_{t-1}, ω_{t-1}, err_lin, err_ang]
    #                    err = actuator tracking error（指令 pre-DR − 實際 post-DR）。
    #                    - a_{t-1}, ω_{t-1}: 上一步指令 → 補償「感測延遲」(obs_delay)
    #                    - err: 馬達沒跟上的量 → 補償「致動延遲」，且穩定時 ≈0 斷 copy shortcut
    #                    這是訓練端顯式延遲建模（論文 §34），v3d 預設。
    #   ⚠️ train 與 play 必須設成相同 mode，否則 obs 分布不符。
    if not hasattr(env, "_act_hist_mode"):
        env._act_hist_mode = os.environ.get("CHARGE_ACT_HIST_MODE", "raw").strip().lower()

    if env._act_hist_mode == "action_error" and len(env._action_history_buffer) >= 1:
        # a_{t-1}, ω_{t-1} 來自 buffer[0]（上一步 applied 指令，1 步前）
        a_tm1 = env._action_history_buffer[0]   # [N, 2]
        # actuator tracking error（從 action term 讀新鮮值），用 buffer 延後 1 步與 a_{t-1} 對齊
        if hasattr(term, "actuator_tracking_error"):
            err_now = torch.nan_to_num(term.actuator_tracking_error, nan=0.0).clamp(-2.0, 2.0)
        else:
            err_now = torch.zeros_like(a_tm1)
        if (not hasattr(env, "_act_err_prev")) or env._act_err_prev is None \
                or env._act_err_prev.shape != err_now.shape:
            env._act_err_prev = torch.zeros_like(err_now)
        if just_reset.any():
            env._act_err_prev[just_reset] = 0.0
        stacked = torch.cat([a_tm1, env._act_err_prev], dim=-1)  # [N, 4]
        env._act_err_prev = err_now.clone()
    elif env._act_hist_mode == "delta" and len(env._action_history_buffer) >= 2:
        a_tm1 = env._action_history_buffer[0]   # [N, 2] = (a_{t-1}, ω_{t-1})
        a_tm2 = env._action_history_buffer[1]   # [N, 2] = (a_{t-2}, ω_{t-2})
        stacked = torch.cat([a_tm1, a_tm1 - a_tm2], dim=-1)  # [N, 4]
    else:
        stacked = torch.cat(env._action_history_buffer, dim=-1)  # [N, stack_size*2]

    # 更新 buffer: 把當前 push 進 buffer[0]，舊資料往後推
    # 注意：observation 函數在 step 結束後被叫，這時 current = a_t (剛 apply 的)
    # 下次 obs 讀取時，這個 current 已經是 a_{t-1}
    # 所以 push 順序：保留 stack_size 個最新的，丟最舊
    new_buffer = [current.clone()] + env._action_history_buffer[:-1]
    env._action_history_buffer = new_buffer

    return stacked


def topk_obstacles_goal_centric(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    top_k: int = 5,
    max_obstacles: int = 10,
    max_distance: float = 8.0,
) -> torch.Tensor:
    """Top-K 障礙物觀測（Goal-Centric Frame，NavRL 風格）

    選擇最近 K 個可見障礙物，在 goal-centric 座標系下計算相對位置和速度。
    Goal-centric frame: 目標方向為 X 軸，垂直方向為 Y 軸。

    每個障礙物 6 維: [rel_x_goal, rel_y_goal, distance, vel_x_goal, vel_y_goal, size]

    Args:
        env: 環境實例
        robot_cfg: 機器人配置
        top_k: 選取最近的 K 個障礙物
        max_obstacles: 場景中最大障礙物數量
        max_distance: 最大觀測距離（超過此距離視為不可見）

    Returns:
        [num_envs, top_k * 6] 障礙物觀測（30 維 for K=5）
    """
    robot: Articulation = env.scene[robot_cfg.name]
    num_envs = env.num_envs
    device = env.device

    # --- 收集所有障礙物位置、速度、大小（同一迴圈，索引一致） ---
    all_pos = torch.zeros(num_envs, max_obstacles, 2, device=device)
    all_z = torch.zeros(num_envs, max_obstacles, device=device)
    all_vel = torch.zeros(num_envs, max_obstacles, 2, device=device)
    all_size = torch.zeros(num_envs, max_obstacles, device=device)

    vel_cache = getattr(env, "_obstacle_velocities", None)
    sizes_raw = getattr(env, "_obstacle_sizes", None)
    sizes_tensor = None
    if sizes_raw is not None:
        sizes_tensor = torch.as_tensor(sizes_raw, device=device, dtype=torch.float32)

    num_found = 0
    for i in range(max_obstacles):
        obstacle_name = f"obstacle_{i}"
        if obstacle_name not in env.scene.keys():
            continue
        obstacle = env.scene[obstacle_name]
        pos_w = torch.nan_to_num(obstacle.data.root_pos_w, nan=0.0)
        all_pos[:, num_found, :] = pos_w[:, :2]
        all_z[:, num_found] = pos_w[:, 2]
        if vel_cache is not None and num_found < vel_cache.shape[1]:
            all_vel[:, num_found, :] = torch.nan_to_num(vel_cache[:, num_found, :], nan=0.0)
        if sizes_tensor is not None and num_found < len(sizes_tensor):
            all_size[:, num_found] = sizes_tensor[num_found]
        num_found += 1

    if num_found == 0:
        return torch.zeros(num_envs, top_k * 6, device=device)

    all_pos = all_pos[:, :num_found, :]
    all_z = all_z[:, :num_found]
    all_vel = all_vel[:, :num_found, :]
    all_size = all_size[:, :num_found]

    # --- 可見性判斷：Z > 0 表示可見（隱藏的在 Z = -10） ---
    visible = all_z > 0.0

    # --- 機器人位置和目標位置 ---
    robot_pos_xy = robot.data.root_pos_w[:, :2]

    try:
        if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
            goal_xy = env._local_goal_world[:, :2]
        else:
            goal_pos = env.command_manager.get_command("goal_command")
            goal_xy = goal_pos[:, :2]
    except (AttributeError, KeyError, IndexError):
        robot_quat = robot.data.root_quat_w
        w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
        robot_yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        goal_xy = robot_pos_xy + torch.stack(
            [torch.cos(robot_yaw), torch.sin(robot_yaw)], dim=1
        )

    # --- 計算 goal-centric 座標系 ---
    goal_dir = goal_xy - robot_pos_xy
    goal_norm = torch.norm(goal_dir, dim=1, keepdim=True).clamp(min=1e-6)
    x_axis = goal_dir / goal_norm
    y_axis = torch.stack([-x_axis[:, 1], x_axis[:, 0]], dim=1)

    # --- 計算障礙物相對位置 ---
    rel_pos = all_pos - robot_pos_xy.unsqueeze(1)
    distances = torch.norm(rel_pos, dim=2)

    large_val = max_distance * 10.0
    masked_distances = torch.where(
        visible & (distances < max_distance),
        distances,
        torch.full_like(distances, large_val),
    )

    # --- 選取最近 K 個 ---
    actual_k = min(top_k, num_found)
    topk_distances, topk_indices = torch.topk(
        masked_distances, k=actual_k, dim=1, largest=False
    )
    topk_valid = topk_distances < large_val  # [N, K]

    # --- batch gather ---
    idx2 = topk_indices.unsqueeze(2).expand(-1, -1, 2)
    topk_rel_pos = torch.gather(rel_pos, 1, idx2)
    topk_vel = torch.gather(all_vel, 1, idx2)
    topk_size = torch.gather(all_size, 1, topk_indices)

    # --- 投影到 goal-centric frame（向量化） ---
    x_axis_exp = x_axis.unsqueeze(1)  # [N, 1, 2] — broadcasts to [N, K, 2]
    y_axis_exp = y_axis.unsqueeze(1)

    rel_x_goal = (topk_rel_pos * x_axis_exp).sum(dim=2)  # [N, K]
    rel_y_goal = (topk_rel_pos * y_axis_exp).sum(dim=2)
    vel_x_goal = (topk_vel * x_axis_exp).sum(dim=2)
    vel_y_goal = (topk_vel * y_axis_exp).sum(dim=2)

    # --- 組裝輸出（向量化，無 Python for-loop） ---
    features = torch.stack(
        [rel_x_goal, rel_y_goal, topk_distances, vel_x_goal, vel_y_goal, topk_size],
        dim=2,
    )  # [N, K, 6]
    features = features * topk_valid.unsqueeze(2)  # 無效 slot 歸零

    # 填充到 top_k（actual_k 可能 < top_k）
    if actual_k < top_k:
        pad = torch.zeros(num_envs, top_k - actual_k, 6, device=device)
        features = torch.cat([features, pad], dim=1)

    return torch.nan_to_num(features.reshape(num_envs, -1), nan=0.0)
