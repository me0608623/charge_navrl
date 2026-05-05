"""牆壁隨機化事件 — per-env 隨機位置 + 數量

init_perenv_walls (startup):
    初始化 env._maze_wall_centers/sizes/mask 和 env._boundary_wall_centers/sizes。
    所有 wall slots 初始在 Z=-10 (隱藏)。

randomize_walls (reset):
    Per-env 隨機化牆壁位置和數量。
    使用 WALL_SLOT_SPECS 定義的 8 個不同長度 wall slots。
    Rejection sampling 確保牆壁不重疊、不太近機器人/邊界。
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ..wall_layout import (
    WALL_SLOT_SPECS,
    MAX_WALL_SLOTS,
    BOUNDARY_WALLS_20x20,
)


def init_perenv_walls(env: ManagerBasedRLEnv, env_ids):
    """Startup event: 初始化 per-env 牆壁數據結構。

    在環境建立時呼叫一次，建立 env 上的牆壁 tensor 屬性。
    所有 wall slots 初始設為隱藏 (mask=False, Z=-10)。
    """
    N = env.num_envs
    device = env.device

    # Per-env maze wall data (8 slots)
    env._maze_wall_centers = torch.zeros(N, MAX_WALL_SLOTS, 2, device=device)
    env._maze_wall_sizes = torch.zeros(N, MAX_WALL_SLOTS, 2, device=device)
    env._maze_wall_mask = torch.zeros(N, MAX_WALL_SLOTS, dtype=torch.bool, device=device)

    # 預設每個 slot 的 AABB size (未旋轉: L=spec[0], W=spec[1])
    for i, (length, width, _height) in enumerate(WALL_SLOT_SPECS):
        env._maze_wall_sizes[:, i, 0] = length
        env._maze_wall_sizes[:, i, 1] = width

    # Global boundary walls (1.0m thickness, shared by all envs)
    bw_data = torch.tensor(BOUNDARY_WALLS_20x20, dtype=torch.float32, device=device)
    env._boundary_wall_centers = bw_data[:, :2]   # [4, 2]
    env._boundary_wall_sizes = bw_data[:, 2:]     # [4, 2]

    print(
        f"[init_perenv_walls] N={N} | {MAX_WALL_SLOTS} wall slots | "
        f"boundary thickness=1.0m | all slots hidden (Z=-10)",
        flush=True,
    )


def randomize_walls(
    env: ManagerBasedRLEnv,
    env_ids,
    min_walls: int = 2,
    max_walls: int = 5,
    boundary: float = 8.5,
    min_wall_spacing: float = 2.0,
    max_spawn_attempts: int = 30,
    robot_safe_dist: float = 1.5,
    target_wall_length: float = 0.0,
):
    """Reset event: per-env 隨機化牆壁位置和數量。

    Args:
        env: 環境實例
        env_ids: 需要重置牆壁的環境 ID
        min_walls: 每個環境的最少牆壁數量
        max_walls: 每個環境的最多牆壁數量 (≤ MAX_WALL_SLOTS)
        boundary: 牆壁中心的最大 |x|, |y| 座標
        min_wall_spacing: 牆壁之間的最小距離 (AABB 間距)
        max_spawn_attempts: 每個 wall slot 的最大嘗試次數
        robot_safe_dist: 牆壁與機器人的最小距離
        target_wall_length: 目標牆壁長度 (m)。> 0 時優先選擇接近此長度的 slot。
                           0 = 不篩選（使用原始 slot 順序）。
                           WD 對照: Phase 1-3=6.0, Phase 4=4.5, Phase 5=5.0, Phase 7+=3.5
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    N = len(env_ids)
    device = env.device
    max_walls = min(max_walls, MAX_WALL_SLOTS)

    # 依 target_wall_length 排序 slot 優先順序（最接近目標長度的 slot 優先啟用）
    if target_wall_length > 0:
        slot_lengths = [spec[0] for spec in WALL_SLOT_SPECS]
        slot_order = sorted(range(MAX_WALL_SLOTS),
                            key=lambda i: abs(slot_lengths[i] - target_wall_length))
    else:
        slot_order = list(range(MAX_WALL_SLOTS))

    # 每個環境隨機決定啟用的牆壁數量
    num_active = torch.randint(min_walls, max_walls + 1, (N,), device=device)

    # 獲取機器人局部座標（用於安全距離檢查）
    robot_pos_w = env.scene["robot"].data.root_pos_w[env_ids, :2]
    env_origins = env.scene.env_origins[env_ids, :2]
    robot_pos_local = robot_pos_w - env_origins

    # 環境原點用於轉世界座標
    env_origins_3d = env.scene.env_origins[env_ids]  # [N, 3]

    # 準備每個 slot 的隱藏 pose
    HIDDEN_Z = -10.0
    wall_height = 3.0   # ★ 對齊 WALL_SLOT_SPECS height，高於 VLP16

    for priority_rank, slot_idx in enumerate(slot_order):
        spec_length, spec_width, spec_height = WALL_SLOT_SPECS[slot_idx]

        # 哪些 env 的這個 slot 是 active（按優先順序，前 num_active 個啟用）
        is_active = priority_rank < num_active  # [N] bool

        # 隨機方向: 0=原始(水平), 1=旋轉90°(垂直)
        rotate_90 = torch.randint(0, 2, (N,), device=device).bool()

        # AABB size (考慮旋轉)
        aabb_x = torch.where(rotate_90,
                             torch.full((N,), spec_width, device=device),
                             torch.full((N,), spec_length, device=device))
        aabb_y = torch.where(rotate_90,
                             torch.full((N,), spec_length, device=device),
                             torch.full((N,), spec_width, device=device))

        # 隨機位置 (局部座標)
        pos_x = torch.rand(N, device=device) * (2 * boundary) - boundary
        pos_y = torch.rand(N, device=device) * (2 * boundary) - boundary

        # Rejection sampling
        needs_resample = is_active.clone()

        for attempt in range(max_spawn_attempts):
            if not needs_resample.any():
                break

            n_resample = needs_resample.sum().item()

            # 檢查 1: 邊界安全距離 (牆壁 AABB 邊緣不超出 boundary)
            half_x = aabb_x * 0.5
            half_y = aabb_y * 0.5
            valid_boundary = (
                (pos_x - half_x > -boundary) &
                (pos_x + half_x < boundary) &
                (pos_y - half_y > -boundary) &
                (pos_y + half_y < boundary)
            )

            # 檢查 2: 與機器人的距離
            robot_delta = torch.stack([pos_x - robot_pos_local[:, 0],
                                       pos_y - robot_pos_local[:, 1]], dim=-1)
            # AABB distance to robot
            robot_dx = (robot_delta[:, 0].abs() - half_x).clamp(min=0.0)
            robot_dy = (robot_delta[:, 1].abs() - half_y).clamp(min=0.0)
            robot_dist = torch.sqrt(robot_dx ** 2 + robot_dy ** 2)
            valid_robot = robot_dist >= robot_safe_dist

            # 檢查 3: 與已放置牆壁的距離（只看本輪已放置的 slots）
            valid_spacing = torch.ones(N, dtype=torch.bool, device=device)
            for prev_rank in range(priority_rank):
                prev_slot = slot_order[prev_rank]
                prev_mask = env._maze_wall_mask[env_ids, prev_slot]
                prev_cx = env._maze_wall_centers[env_ids, prev_slot, 0]
                prev_cy = env._maze_wall_centers[env_ids, prev_slot, 1]
                prev_sx = env._maze_wall_sizes[env_ids, prev_slot, 0]
                prev_sy = env._maze_wall_sizes[env_ids, prev_slot, 1]

                # AABB-to-AABB distance
                dx = (pos_x - prev_cx).abs() - (half_x + prev_sx * 0.5)
                dy = (pos_y - prev_cy).abs() - (half_y + prev_sy * 0.5)
                dx = dx.clamp(min=0.0)
                dy = dy.clamp(min=0.0)
                gap = torch.sqrt(dx ** 2 + dy ** 2)

                # Only check spacing against active previous walls
                too_close_prev = prev_mask & (gap < min_wall_spacing)
                valid_spacing = valid_spacing & ~too_close_prev

            all_valid = valid_boundary & valid_robot & valid_spacing & needs_resample
            needs_resample = needs_resample & ~all_valid

            if needs_resample.any():
                n = needs_resample.sum().item()
                pos_x[needs_resample] = torch.rand(n, device=device) * (2 * boundary) - boundary
                pos_y[needs_resample] = torch.rand(n, device=device) * (2 * boundary) - boundary

        # 更新 per-env wall data
        env._maze_wall_centers[env_ids, slot_idx, 0] = pos_x
        env._maze_wall_centers[env_ids, slot_idx, 1] = pos_y
        env._maze_wall_sizes[env_ids, slot_idx, 0] = aabb_x
        env._maze_wall_sizes[env_ids, slot_idx, 1] = aabb_y
        env._maze_wall_mask[env_ids, slot_idx] = is_active

        # 建立物理 pose 並寫入模擬器
        wall_name = f"wall_internal_{slot_idx}"
        try:
            wall = env.scene[wall_name]
        except KeyError:
            continue

        pose = torch.zeros(N, 7, device=device)
        # Active walls: local pos → world pos
        world_x = pos_x + env_origins[:, 0]
        world_y = pos_y + env_origins[:, 1]
        pose[:, 0] = torch.where(is_active, world_x, env_origins[:, 0])
        pose[:, 1] = torch.where(is_active, world_y, env_origins[:, 1])
        pose[:, 2] = torch.where(is_active,
                                 torch.full((N,), wall_height / 2, device=device),
                                 torch.full((N,), HIDDEN_Z, device=device))

        # Quaternion: identity (no rotation) or 90° around Z
        # quat(90° Z) = (cos(45°), 0, 0, sin(45°)) = (0.7071, 0, 0, 0.7071)
        cos45 = math.cos(math.pi / 4)
        sin45 = math.sin(math.pi / 4)
        pose[:, 3] = torch.where(rotate_90 & is_active,
                                 torch.full((N,), cos45, device=device),
                                 torch.ones(N, device=device))
        pose[:, 4] = 0.0
        pose[:, 5] = 0.0
        pose[:, 6] = torch.where(rotate_90 & is_active,
                                 torch.full((N,), sin45, device=device),
                                 torch.zeros(N, device=device))

        wall.write_root_pose_to_sim(pose, env_ids=env_ids)

    # 診斷 (僅首次)
    if not hasattr(env, "_wall_randomize_count"):
        env._wall_randomize_count = 0
    env._wall_randomize_count += 1
    if env._wall_randomize_count <= 2:
        active_per_env = env._maze_wall_mask[env_ids].sum(dim=1).float()
        print(
            f"[隨機牆壁] 環境數={N} | "
            f"牆數範圍=({min_walls},{max_walls}) | "
            f"實際: 平均={active_per_env.mean():.1f} "
            f"最小={active_per_env.min().item():.0f} "
            f"最大={active_per_env.max().item():.0f}",
            flush=True,
        )
