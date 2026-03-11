"""迷宮牆壁幾何定義 + GPU 並行空間查詢工具

原位置: core/wall_layout.py → 遷移至 mdp/wall_layout.py (共用工具)

場景: 16x16m 訓練房間 (center origin)
    - 4 面外牆 (BOUNDARY_WALLS): 封閉房間
    - 6 面內部牆 (MAZE_WALLS): 迷宮結構，製造遮擋和走廊

資料格式: (center_x, center_y, size_x, size_y) — AABB 表示

函數:
    get_wall_tensors(device):        內部牆 (6 段) → GPU tensor
    get_all_wall_tensors(device):    全部牆 (10 段) → GPU tensor
    check_wall_proximity_batch():    GPU AABB proximity — 批量檢查 N 個點是否太近牆壁
    check_los_batch():               GPU slab method — 批量 Line-of-Sight 遮擋檢查

使用者:
    - events/reset.py: 重置位置時避免放在牆內
    - events/obstacles.py: 障礙物放置時避開牆壁
    - observations/obs_functions.py: topk_obstacles 的 LOS 遮擋判斷
    - goal_command.py: 目標生成時避開牆壁

效能: 全部 GPU 向量化，O(N×W) 複雜度，256×14=3,584 次比較在 GPU 上微不足道。
"""

from __future__ import annotations

import torch
from torch import Tensor

# Wall segment definitions: (center_x, center_y, size_x, size_y)
# All walls: thickness=0.2m, height=1.5m
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
