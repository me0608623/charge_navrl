"""迷宮牆壁幾何定義 + GPU 並行空間查詢工具

場景:
  - 16x16m: MAZE_WALLS + BOUNDARY_WALLS（舊版，供非課程 task 使用）
  - 20x20m: WALL_SLOT_SPECS (per-env 隨機) + BOUNDARY_WALLS_20x20（課程 task 使用）

資料格式: (center_x, center_y, size_x, size_y) — AABB 表示

函數 (global wall data — 16x16):
    get_wall_tensors(device):        內部牆 (6 段) → GPU tensor
    get_all_wall_tensors(device):    全部牆 (10 段) → GPU tensor
    check_wall_proximity_batch():    GPU AABB proximity
    check_los_batch():               GPU slab method LOS

函數 (per-env — 20x20 randomized walls):
    check_wall_proximity_perenv():   per-env AABB proximity — [N,2] vs [N,W,2]
    check_los_perenv():              per-env LOS — [N,2] origins, [N,F,2] targets
    get_combined_wall_data(env):     合併 per-env maze walls + global boundary walls

使用者:
    - events/reset.py: 重置位置時避免放在牆內
    - events/mixed_parallel.py: 障礙物放置時避開牆壁
    - events/walls.py: per-env 牆壁初始化（讀取 WALL_SLOT_SPECS + BOUNDARY_WALLS_20x20）
    - observations/obs_functions.py: topk_obstacles 的 LOS 遮擋判斷
    - goal_command.py: 目標生成時避開牆壁
    - terminations/robot_state.py: 牆壁碰撞終止

效能: 全部 GPU 向量化，O(N×W) 複雜度。
"""

from __future__ import annotations

import torch
from torch import Tensor

# Wall segment definitions: (center_x, center_y, size_x, size_y)
# All walls: thickness=0.2m, height=3.0m (★ VLP16 可觀測)
# 6 面內部牆，均勻分佈在 4 個象限，形成走廊結構
MAZE_WALLS: list[tuple[float, float, float, float]] = [
    (-6.25, 4.0, 3.5, 0.2),  # 0: Top-left horizontal (x: -8 to -4.5)
    ( 3.0,  6.5, 0.2, 3.0),  # 1: Top-right vertical (y: 5 to 8)
    ( 4.0,  0.0, 3.0, 0.2),  # 2: Right-middle horizontal
    (-3.0, -1.0, 0.2, 4.0),  # 3: Center-left vertical (y: -3 to 1)
    ( 0.5, -5.0, 3.0, 0.2),  # 4: Bottom-center horizontal
    ( 6.5, -5.5, 0.2, 3.0),  # 5: Bottom-right vertical (y: -7 to -4)
]

# Boundary walls: 16x16m room outer walls (thickness 0.2m)
BOUNDARY_WALLS: list[tuple[float, float, float, float]] = [
    ( 0.0,  8.0, 16.2,  0.2),  # North
    ( 0.0, -8.0, 16.2,  0.2),  # South
    ( 8.0,  0.0,  0.2, 16.2),  # East
    (-8.0,  0.0,  0.2, 16.2),  # West
]

# All wall segments: 6 internal + 4 boundary = 10 total
ALL_WALLS = MAZE_WALLS + BOUNDARY_WALLS

# 20×20m 場景：4 面內部牆（比 16×16 的 6 面少，密度更低）
MAZE_WALLS_20x20: list[tuple[float, float, float, float]] = [
    (-7.5,  5.0, 4.0, 0.2),   # Top-left horizontal
    ( 4.0,  7.5, 0.2, 3.5),   # Top-right vertical
    (-4.0, -1.5, 0.2, 4.5),   # Center-left vertical
    ( 2.0, -6.0, 3.5, 0.2),   # Bottom-center horizontal
]

BOUNDARY_WALLS_20x20: list[tuple[float, float, float, float]] = [
    # 必須與 charge_env_cfg_vlp16_curriculum.py MySceneCfgVLP16_20x20 的
    # room_size=10.0, wall_thickness=1.0 一致
    ( 0.0,  10.0, 21.0, 1.0),  # North  (center_x, center_y, size_x, size_y)
    ( 0.0, -10.0, 21.0, 1.0),  # South
    ( 10.0,  0.0,  1.0, 21.0), # East
    (-10.0,  0.0,  1.0, 21.0), # West
]

ALL_WALLS_20x20 = MAZE_WALLS_20x20 + BOUNDARY_WALLS_20x20

# T 字型走廊：58×20m 場景 + 2 個填充塊創造 T 形走道
# 走廊規格: 水平 bar 寬 3.5m (y 方向), 垂直 stem 寬 5.0m (x 方向)
# 橫向 bar 長度 = 57m（原 arena 寬 19m × 3 倍）
# 可行走區域:
#   Bar:  x ∈ [-28.5, +28.5], y ∈ [+6.0, +9.5]  → 57m × 3.5m
#   Stem: x ∈ [-2.5, +2.5],  y ∈ [-9.5, +6.0]   → 5m × 15.5m
BOUNDARY_WALLS_T_CORRIDOR: list[tuple[float, float, float, float]] = [
    # 4 面外牆（58m × 20m 場景，wall_thickness=1.0）
    ( 0.0,  10.0, 59.0, 1.0),   # North  (x: -29.5 to +29.5)
    ( 0.0, -10.0, 59.0, 1.0),   # South
    ( 29.0,  0.0,  1.0, 21.0),  # East   (y: -10.5 to +10.5)
    (-29.0,  0.0,  1.0, 21.0),  # West
    # 2 個填充塊（填滿 T 形走道以外的死區）
    # Left fill:  x ∈ [-28.5, -2.5], y ∈ [-9.5, +6.0]
    (-15.5, -1.75, 26.0, 15.5),  # Left dead zone fill
    ( 15.5, -1.75, 26.0, 15.5),  # Right dead zone fill
]

# ============================================================================
# Per-env randomized wall slot specifications
# ============================================================================
# 8 wall slots with different lengths, all 1.0m thick, 3.0m tall
# ★ height 3.0m: 高於 VLP16 LiDAR (z=1.6m)，確保所有 phase 牆壁可觀測
# Mesh size is fixed at spawn time (Isaac Sim limitation); mask controls visibility.
WALL_SLOT_SPECS: list[tuple[float, float, float]] = [
    (4.0, 1.0, 3.0),   # slot 0: 4m
    (3.0, 1.0, 3.0),   # slot 1: 3m
    (5.0, 1.0, 3.0),   # slot 2: 5m
    (3.5, 1.0, 3.0),   # slot 3: 3.5m
    (4.5, 1.0, 3.0),   # slot 4: 4.5m
    (2.5, 1.0, 3.0),   # slot 5: 2.5m
    (4.0, 1.0, 3.0),   # slot 6: 4m
    (3.0, 1.0, 3.0),   # slot 7: 3m
]
MAX_WALL_SLOTS = 8

# Module-level cache: {device_str: (wall_centers, wall_sizes)}
_wall_tensor_cache: dict[str, tuple[Tensor, Tensor]] = {}
_all_wall_cache: dict[str, tuple[Tensor, Tensor]] = {}


def get_wall_tensors(device) -> tuple[Tensor, Tensor]:
    """Return cached (wall_centers [W, 2], wall_sizes [W, 2]) on the given device.

    Args:
        device: torch device (e.g. "cuda:0", "cpu").

    Returns:
        Tuple of (wall_centers, wall_sizes) tensors.
    """
    key = str(device)
    if key not in _wall_tensor_cache:
        data = torch.tensor(MAZE_WALLS, dtype=torch.float32, device=device)
        centers = data[:, :2]  # [W, 2]
        sizes = data[:, 2:]    # [W, 2]
        _wall_tensor_cache[key] = (centers, sizes)
    return _wall_tensor_cache[key]


def get_all_wall_tensors(device) -> tuple[Tensor, Tensor]:
    """Return cached (wall_centers [10, 2], wall_sizes [10, 2]) for all walls.

    Includes both internal maze walls (6) and boundary walls (4).

    Args:
        device: torch device (e.g. "cuda:0", "cpu").

    Returns:
        Tuple of (wall_centers, wall_sizes) tensors.
    """
    key = str(device)
    if key not in _all_wall_cache:
        data = torch.tensor(ALL_WALLS, dtype=torch.float32, device=device)
        _all_wall_cache[key] = (data[:, :2], data[:, 2:])
    return _all_wall_cache[key]


# NOTE: MAZE_WALLS_20x20 / ALL_WALLS_20x20 kept for reference/debugging.
# Per-env wall spawning uses WALL_SLOT_SPECS (randomized), not these static coords.
# get_wall_tensors_20x20 / get_all_wall_tensors_20x20 removed (never called at runtime).


def check_wall_proximity_batch(
    positions: Tensor,     # [N, 2]
    wall_centers: Tensor,  # [W, 2]
    wall_sizes: Tensor,    # [W, 2]
    min_dist: float,
) -> Tensor:
    """GPU-vectorized AABB proximity check.

    Returns True for each position that is within min_dist of any wall segment.

    Uses signed distance to axis-aligned rectangle:
        dx = max(|px - cx| - sx/2, 0)
        dy = max(|py - cy| - sy/2, 0)
        dist = sqrt(dx^2 + dy^2)

    Args:
        positions: [N, 2] query positions in local frame.
        wall_centers: [W, 2] wall center coordinates.
        wall_sizes: [W, 2] wall (size_x, size_y).
        min_dist: minimum clearance distance.

    Returns:
        [N] boolean tensor — True if position is too close to any wall.
    """
    # positions: [N, 2] -> [N, 1, 2]
    # wall_centers: [W, 2] -> [1, W, 2]
    pos = positions.unsqueeze(1)          # [N, 1, 2]
    wc = wall_centers.unsqueeze(0)        # [1, W, 2]
    ws = wall_sizes.unsqueeze(0)          # [1, W, 2]

    # Signed distance to each wall AABB
    delta = (pos - wc).abs() - ws * 0.5   # [N, W, 2]
    delta = delta.clamp(min=0.0)           # [N, W, 2]
    dist = torch.norm(delta, dim=2)        # [N, W]

    # True if within min_dist of ANY wall
    too_close = (dist < min_dist).any(dim=1)  # [N]
    return too_close


def check_los_batch(
    origins: Tensor,       # [N, 2] robot positions
    targets: Tensor,       # [N, F, 2] obstacle positions
    wall_centers: Tensor,  # [W, 2]
    wall_sizes: Tensor,    # [W, 2]
) -> Tensor:
    """Vectorized 2D line-of-sight check using slab method.

    For each (origin, target) pair, checks whether the line segment
    intersects any wall AABB. Returns True if the target is visible
    (no wall blocks the line of sight).

    Complexity: O(N × F × W) = 256 × 10 × 10 = 25,600 — GPU trivial.

    Args:
        origins: [N, 2] robot positions (XY).
        targets: [N, F, 2] obstacle positions (XY).
        wall_centers: [W, 2] wall center coordinates.
        wall_sizes: [W, 2] wall (size_x, size_y).

    Returns:
        [N, F] boolean tensor — True = visible (no wall blocks LOS).
    """
    # AABB bounds: [W, 2]
    wall_min = wall_centers - wall_sizes * 0.5  # [W, 2]
    wall_max = wall_centers + wall_sizes * 0.5  # [W, 2]

    # Direction vectors: [N, F, 2]
    A = origins.unsqueeze(1)          # [N, 1, 2]
    d = targets - A                   # [N, F, 2]

    # Expand for broadcasting against walls
    # A: [N, 1, 1, 2],  d: [N, F, 1, 2]
    A = A.unsqueeze(2)                # [N, 1, 1, 2]
    d = d.unsqueeze(2)                # [N, F, 1, 2]

    # wall bounds: [1, 1, W, 2]
    wmin = wall_min.unsqueeze(0).unsqueeze(0)  # [1, 1, W, 2]
    wmax = wall_max.unsqueeze(0).unsqueeze(0)  # [1, 1, W, 2]

    # Slab parameters for X and Y axes
    # t1 = (wall_min - A) / d,  t2 = (wall_max - A) / d
    eps = 1e-8
    d_safe = d.clone()
    parallel = d.abs() < eps          # [N, F, 1, 2] — degenerate axes

    # Replace near-zero directions with eps to avoid division by zero
    d_safe[parallel] = eps

    t1 = (wmin - A) / d_safe          # [N, F, W, 2]
    t2 = (wmax - A) / d_safe          # [N, F, W, 2]

    # Ensure t_enter < t_leave per axis
    t_min = torch.min(t1, t2)         # [N, F, W, 2]
    t_max = torch.max(t1, t2)         # [N, F, W, 2]

    # Handle parallel rays: if origin is outside slab, no intersection
    # For parallel axis: if A_i < wmin_i or A_i > wmax_i → no hit
    A_broad = A.expand_as(t1)         # [N, F, W, 2]
    outside_slab = (A_broad < wmin) | (A_broad > wmax)  # [N, F, W, 2]
    parallel_broad = parallel.expand_as(t1)

    # Parallel + outside → force no intersection (t_min > t_max)
    t_min = torch.where(parallel_broad & outside_slab,
                        torch.tensor(1e6, device=origins.device), t_min)
    t_max = torch.where(parallel_broad & outside_slab,
                        torch.tensor(-1e6, device=origins.device), t_max)
    # Parallel + inside → don't constrain (full range)
    t_min = torch.where(parallel_broad & ~outside_slab,
                        torch.tensor(-1e6, device=origins.device), t_min)
    t_max = torch.where(parallel_broad & ~outside_slab,
                        torch.tensor(1e6, device=origins.device), t_max)

    # Global slab intersection: t_enter = max across axes, t_leave = min across axes
    t_enter = t_min.max(dim=-1).values   # [N, F, W]
    t_leave = t_max.min(dim=-1).values   # [N, F, W]

    # Intersection condition: overlapping slab AND segment [0, 1]
    hit = (t_enter <= t_leave) & (t_leave > 0.0) & (t_enter < 1.0)  # [N, F, W]

    # Occluded if ANY wall blocks the ray
    occluded = hit.any(dim=-1)        # [N, F]

    # Visible = NOT occluded
    return ~occluded


# ============================================================================
# Per-env wall query functions (randomized walls)
# ============================================================================

def check_wall_proximity_perenv(
    positions: Tensor,     # [N, 2]
    wall_centers: Tensor,  # [N, W, 2]
    wall_sizes: Tensor,    # [N, W, 2]
    wall_mask: Tensor,     # [N, W]
    min_dist: float,
) -> Tensor:
    """Per-env GPU-vectorized AABB proximity check.

    Same logic as check_wall_proximity_batch, but each env has its own
    set of walls (different positions, different active mask).

    Args:
        positions: [N, 2] query positions in local frame.
        wall_centers: [N, W, 2] per-env wall center coordinates.
        wall_sizes: [N, W, 2] per-env wall (size_x, size_y).
        wall_mask: [N, W] boolean — True = wall is active.
        min_dist: minimum clearance distance.

    Returns:
        [N] boolean tensor — True if position is too close to any active wall.
    """
    pos = positions.unsqueeze(1)                      # [N, 1, 2]
    delta = (pos - wall_centers).abs() - wall_sizes * 0.5  # [N, W, 2]
    delta = delta.clamp(min=0.0)                       # [N, W, 2]
    dist = torch.norm(delta, dim=2)                    # [N, W]

    # Inactive walls → large distance (never trigger proximity)
    dist = torch.where(wall_mask, dist, torch.full_like(dist, 1e6))

    too_close = (dist < min_dist).any(dim=1)           # [N]
    return too_close


def check_los_perenv(
    origins: Tensor,       # [N, 2] robot positions
    targets: Tensor,       # [N, F, 2] obstacle positions
    wall_centers: Tensor,  # [N, W, 2]
    wall_sizes: Tensor,    # [N, W, 2]
    wall_mask: Tensor,     # [N, W]
) -> Tensor:
    """Per-env vectorized 2D line-of-sight check using slab method.

    Same logic as check_los_batch, but each env has its own walls.

    Args:
        origins: [N, 2] robot positions (XY).
        targets: [N, F, 2] obstacle positions (XY).
        wall_centers: [N, W, 2] per-env wall center coordinates.
        wall_sizes: [N, W, 2] per-env wall (size_x, size_y).
        wall_mask: [N, W] boolean — True = wall is active.

    Returns:
        [N, F] boolean tensor — True = visible (no active wall blocks LOS).
    """
    # AABB bounds: [N, W, 2]
    wall_min = wall_centers - wall_sizes * 0.5
    wall_max = wall_centers + wall_sizes * 0.5

    # Direction vectors: [N, F, 2]
    A = origins.unsqueeze(1)          # [N, 1, 2]
    d = targets - A                   # [N, F, 2]

    # Expand for broadcasting against walls
    A = A.unsqueeze(2)                # [N, 1, 1, 2]
    d = d.unsqueeze(2)                # [N, F, 1, 2]

    # wall bounds: [N, 1, W, 2]
    wmin = wall_min.unsqueeze(1)      # [N, 1, W, 2]
    wmax = wall_max.unsqueeze(1)      # [N, 1, W, 2]

    # Slab parameters
    eps = 1e-8
    d_safe = d.clone()
    parallel = d.abs() < eps

    d_safe[parallel] = eps

    t1 = (wmin - A) / d_safe          # [N, F, W, 2]
    t2 = (wmax - A) / d_safe          # [N, F, W, 2]

    t_min = torch.min(t1, t2)
    t_max = torch.max(t1, t2)

    A_broad = A.expand_as(t1)
    outside_slab = (A_broad < wmin) | (A_broad > wmax)
    parallel_broad = parallel.expand_as(t1)

    big = torch.tensor(1e6, device=origins.device)
    neg_big = torch.tensor(-1e6, device=origins.device)

    t_min = torch.where(parallel_broad & outside_slab, big, t_min)
    t_max = torch.where(parallel_broad & outside_slab, neg_big, t_max)
    t_min = torch.where(parallel_broad & ~outside_slab, neg_big, t_min)
    t_max = torch.where(parallel_broad & ~outside_slab, big, t_max)

    t_enter = t_min.max(dim=-1).values   # [N, F, W]
    t_leave = t_max.min(dim=-1).values   # [N, F, W]

    hit = (t_enter <= t_leave) & (t_leave > 0.0) & (t_enter < 1.0)  # [N, F, W]

    # Apply wall mask: inactive walls can't block LOS
    # wall_mask: [N, W] → [N, 1, W]
    hit = hit & wall_mask.unsqueeze(1)

    occluded = hit.any(dim=-1)        # [N, F]
    return ~occluded


def get_combined_wall_data(env) -> tuple[Tensor, Tensor, Tensor]:
    """合併 per-env maze walls + global boundary walls.

    Returns:
        (centers [N, W_total, 2], sizes [N, W_total, 2], mask [N, W_total])
        where W_total = MAX_WALL_SLOTS + 4 (boundary).
    """
    if hasattr(env, '_maze_wall_centers') and env._maze_wall_centers is not None:
        mc = env._maze_wall_centers   # [N, 8, 2]
        ms = env._maze_wall_sizes     # [N, 8, 2]
        mm = env._maze_wall_mask      # [N, 8]
        bc = env._boundary_wall_centers  # [4, 2]
        bs = env._boundary_wall_sizes    # [4, 2]

        N = mc.shape[0]
        B = bc.shape[0]  # boundary wall count (4 for arena, 6 for T-corridor)
        bc_exp = bc.unsqueeze(0).expand(N, -1, -1)   # [N, B, 2]
        bs_exp = bs.unsqueeze(0).expand(N, -1, -1)   # [N, B, 2]
        bm = torch.ones(N, B, dtype=torch.bool, device=mc.device)

        centers = torch.cat([mc, bc_exp], dim=1)  # [N, W_total, 2]
        sizes = torch.cat([ms, bs_exp], dim=1)
        mask = torch.cat([mm, bm], dim=1)

        return centers, sizes, mask

    # Fallback: use static ALL_WALLS_20x20 when per-env walls not initialized
    # (should not happen in curriculum training — events/walls.py always sets _maze_wall_centers)
    device = env.device
    N = env.num_envs
    data = torch.tensor(ALL_WALLS_20x20, dtype=torch.float32, device=device)
    c = data[:, :2]   # [W, 2]
    s = data[:, 2:]   # [W, 2]
    W = c.shape[0]
    centers = c.unsqueeze(0).expand(N, -1, -1)   # [N, W, 2]
    sizes = s.unsqueeze(0).expand(N, -1, -1)
    mask = torch.ones(N, W, dtype=torch.bool, device=device)
    return centers, sizes, mask
