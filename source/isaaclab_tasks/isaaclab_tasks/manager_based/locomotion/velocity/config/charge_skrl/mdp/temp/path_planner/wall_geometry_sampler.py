"""
牆壁幾何採樣器 (Wall Geometry Sampler)

直接從環境配置參數生成障礙物，不依賴 USD Mesh 讀取。

關鍵架構理解：
- 牆是用 CuboidCfg 生成的，尺寸和位置都在配置中
- 不需要讀 USD Mesh，直接用數學參數更精準、更快速
- 這是研究級論文的標準做法
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Optional, Dict
from dataclasses import dataclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class WallConfig:
    """單面牆的配置（從環境配置中提取）"""
    name: str  # 牆的名稱，如 "wall_north"
    center: Tuple[float, float]  # (x, y) 中心位置（環境局部坐標）
    size: Tuple[float, float]  # (length, thickness) 長度和厚度
    yaw: float = 0.0  # 旋轉角度（弧度），默認 0


def get_phase0_wall_configs() -> List[WallConfig]:
    """獲取 Phase 0 的牆壁配置（只有四面邊界牆）

    Phase 0 是車輛動力學校準階段：
    - 16x16m 房間，四面有牆
    - 無內部障礙物（專注於運動學）
    """
    room_size = 8.0
    wall_thickness = 0.2

    walls = []

    # 四面邊界牆
    # 北牆 (Y = +8m)
    walls.append(WallConfig(
        name="wall_north",
        center=(0.0, room_size),
        size=(16.0, wall_thickness),
        yaw=0.0,
    ))

    # 南牆 (Y = -8m)
    walls.append(WallConfig(
        name="wall_south",
        center=(0.0, -room_size),
        size=(16.0, wall_thickness),
        yaw=0.0,
    ))

    # 東牆 (X = +8m)
    walls.append(WallConfig(
        name="wall_east",
        center=(room_size, 0.0),
        size=(wall_thickness, 16.0),
        yaw=0.0,
    ))

    # 西牆 (X = -8m)
    walls.append(WallConfig(
        name="wall_west",
        center=(-room_size, 0.0),
        size=(wall_thickness, 16.0),
        yaw=0.0,
    ))

    return walls


def get_phase1_wall_configs() -> List[WallConfig]:
    """獲取 Phase 1 的所有牆壁配置（硬編碼，從配置文件提取）

    這些參數直接來自 charge_env_cfg_phase1.py
    """
    room_size = 8.0
    wall_thickness = 0.2

    walls = []

    # ============================================================================
    # 外牆
    # ============================================================================
    # 北牆 (Y = +8m)
    walls.append(WallConfig(
        name="wall_north",
        center=(0.0, room_size),
        size=(16.0, wall_thickness),  # 長 16m，厚 0.2m
        yaw=0.0,
    ))

    # 南牆 (Y = -8m)
    walls.append(WallConfig(
        name="wall_south",
        center=(0.0, -room_size),
        size=(16.0, wall_thickness),
        yaw=0.0,
    ))

    # 東牆 (X = +8m)
    walls.append(WallConfig(
        name="wall_east",
        center=(room_size, 0.0),
        size=(wall_thickness, 16.0),  # 厚 0.2m，長 16m
        yaw=0.0,
    ))

    # 西牆 (X = -8m)
    walls.append(WallConfig(
        name="wall_west",
        center=(-room_size, 0.0),
        size=(wall_thickness, 16.0),
        yaw=0.0,
    ))

    # ============================================================================
    # U 型牆
    # ============================================================================
    # U 型牆 - 上段（橫向）
    walls.append(WallConfig(
        name="u_wall_top",
        center=(-2.0, 3.0),
        size=(4.0, wall_thickness),
        yaw=0.0,
    ))

    # U 型牆 - 左段（縱向）
    walls.append(WallConfig(
        name="u_wall_left",
        center=(-4.0, 1.5),
        size=(wall_thickness, 3.0),
        yaw=0.0,
    ))

    # U 型牆 - 下段（橫向，留出開口）
    walls.append(WallConfig(
        name="u_wall_bottom",
        center=(-2.5, 0.0),
        size=(3.0, wall_thickness),
        yaw=0.0,
    ))

    # ============================================================================
    # 隔間牆
    # ============================================================================
    door_width = 1.5
    partition_top_length = 2.5
    partition_bottom_length = 2.5

    # 隔間上牆（帶門口）
    walls.append(WallConfig(
        name="partition_top",
        center=(4.0 + partition_top_length / 2, 2.0),
        size=(partition_top_length, wall_thickness),
        yaw=0.0,
    ))

    # 隔間下牆（帶門口）
    walls.append(WallConfig(
        name="partition_bottom",
        center=(4.0 + partition_bottom_length / 2, -2.0),
        size=(partition_bottom_length, wall_thickness),
        yaw=0.0,
    ))

    # 隔間側牆（封閉右側）
    walls.append(WallConfig(
        name="partition_side",
        center=(4.0 + 2.5 + door_width + wall_thickness / 2, 0.0),
        size=(wall_thickness, 5.0),
        yaw=0.0,
    ))

    return walls


def get_phase2_wall_configs() -> List[WallConfig]:
    """獲取 Phase 2 的所有牆壁配置

    Phase 2 有 T 型路口和窄走廊
    """
    room_size = 8.0
    wall_thickness = 0.2

    walls = []

    # 外牆（與 Phase 1 相同）
    walls.append(WallConfig("wall_north", (0.0, room_size), (16.0, wall_thickness)))
    walls.append(WallConfig("wall_south", (0.0, -room_size), (16.0, wall_thickness)))
    walls.append(WallConfig("wall_east", (room_size, 0.0), (wall_thickness, 16.0)))
    walls.append(WallConfig("wall_west", (-room_size, 0.0), (wall_thickness, 16.0)))

    # TODO: 添加 Phase 2 特有的牆壁（T 型路口、窄走廊）

    return walls


def get_phase3_wall_configs() -> List[WallConfig]:
    """獲取 Phase 3 的所有牆壁配置

    Phase 3 有動態障礙物，但牆壁結構與 Phase 1 類似
    """
    # Phase 3 牆壁與 Phase 1 相同
    return get_phase1_wall_configs()


def add_rectangle_to_occupancy_grid(
    occupancy_grid: torch.Tensor,
    map_origin: torch.Tensor,
    grid_resolution: float,
    grid_width: int,
    grid_depth: int,
    center: Tuple[float, float],
    size: Tuple[float, float],
    yaw: float = 0.0,
    env_origin: Optional[np.ndarray] = None,
) -> int:
    """直接在 occupancy grid 中畫長方形障礙物（數學方法，精準且快速）

    這是研究級論文的標準做法：
    - 不依賴 USD Mesh 讀取
    - 不需要採樣點
    - 完全精準的數學幾何
    - 超快速

    Args:
        occupancy_grid: [W, H] 佔用網格
        map_origin: [2] 地圖原點（地圖左下角，世界坐標）
        grid_resolution: 網格解析度（米/格）
        grid_width: 網格寬度（格數）
        grid_depth: 網格深度（格數）
        center: (x, y) 長方形中心（環境局部坐標）
        size: (length, thickness) 長方形的長度和厚度（米）
        yaw: 旋轉角度（弧度），暫不支持旋轉
        env_origin: 環境在世界坐標系中的偏移（僅在 center 是世界坐標時使用）

    Returns:
        標記的網格數量

    坐標系說明：
        地圖原點 map_origin = (-8, -8) 是地圖左下角的世界坐標
        牆壁 center = (0, 8) 是相對於環境中心的局部坐標

        環境中心 = env_origin（例如 (9, 0)）
        牆壁世界坐標 = env_origin + center = (9, 8)
        地圖網格坐標 = (牆壁世界坐標 - 地圖原點) / 解析度
                       = ((9, 8) - (-8, -8)) / 0.1 = (170, 160)

        但牆壁配置的 center 已經考慮了環境中心的位置，
        所以牆壁在地圖中的位置應該是：
        地圖網格坐標 = (center - map_origin - env_origin) / 解析度
                       = ((0, 8) - (-8, -8) - (9, 0)) / 0.1 = (-1, 160)
    """
    # 🔥 關鍵：坐標系轉換
    # map_origin = (-8, -8) 是地圖左下角
    # center = (0, 8) 是牆壁相對於環境中心的位置
    # 我們需要計算牆壁在「地圖坐標系」中的位置

    # 方法 1：如果提供了 env_origin（環境世界坐標），則：
    # 牆壁世界坐標 = env_origin + center
    # 地圖坐標 = (牆壁世界坐標 - map_origin) / resolution
    if env_origin is not None:
        center_world = (center[0] + env_origin[0], center[1] + env_origin[1])
        origin_np = map_origin.cpu().numpy()

        min_grid_x = int((center_world[0] - origin_np[0] - size[0]/2) / grid_resolution)
        max_grid_x = int((center_world[0] - origin_np[0] + size[0]/2) / grid_resolution)
        min_grid_y = int((center_world[1] - origin_np[1] - size[1]/2) / grid_resolution)
        max_grid_y = int((center_world[1] - origin_np[1] + size[1]/2) / grid_resolution)
    else:
        # 方法 2：如果沒有 env_origin，假設 center 已經是地圖坐標
        center_local = center
        origin_np = map_origin.cpu().numpy()

        min_grid_x = int((center_local[0] - size[0]/2 - origin_np[0]) / grid_resolution)
        max_grid_x = int((center_local[0] + size[0]/2 - origin_np[0]) / grid_resolution)
        min_grid_y = int((center_local[1] - size[1]/2 - origin_np[1]) / grid_resolution)
        max_grid_y = int((center_local[1] + size[1]/2 - origin_np[1]) / grid_resolution)

    # 標記網格
    marked_count = 0
    for gx in range(min_grid_x, max_grid_x + 1):
        for gy in range(min_grid_y, max_grid_y + 1):
            if 0 <= gx < grid_width and 0 <= gy < grid_depth:
                occupancy_grid[gx, gy] = 1
                marked_count += 1

    return marked_count


def add_walls_from_config(
    env: ManagerBasedRLEnv,
    env_id: int,
    occupancy_grid: torch.Tensor,
    map_origin: torch.Tensor,
    grid_resolution: float,
    grid_width: int,
    grid_depth: int,
    phase: str = "phase1",
    debug: bool = False,
) -> int:
    """從牆壁配置直接添加障礙物到 occupancy grid（推薦方法）

    這是最穩定、最精準的方法：
    - 直接使用配置文件中的牆壁參數
    - 數學幾何計算，無誤差
    - 不依賴 USD API
    - 多環境更穩定

    Args:
        env: Isaac Lab 環境實例
        env_id: 環境 ID
        occupancy_grid: [W, H] 佔用網格
        map_origin: [2] 地圖原點
        grid_resolution: 網格解析度
        grid_width: 網格寬度
        grid_depth: 網格深度
        phase: phase 名稱 ("phase1", "phase2", "phase3")
        debug: 是否輸出調試信息

    Returns:
        標記的網格總數
    """
    # 獲取環境原點
    env_origin = env.scene.env_origins[env_id, :2].cpu().numpy()

    # 獲取對應 phase 的牆壁配置
    if phase == "phase0":
        wall_configs = get_phase0_wall_configs()
    elif phase == "phase1":
        wall_configs = get_phase1_wall_configs()
    elif phase == "phase2":
        wall_configs = get_phase2_wall_configs()
    elif phase == "phase3":
        wall_configs = get_phase3_wall_configs()
    else:
        raise ValueError(f"未知的 phase: {phase}")

    if debug:
        print(f"\n{'='*60}")
        print(f"Wall Geometry Sampler - Phase {phase}")
        print(f"{'='*60}")
        print(f"環境世界原點: ({env_origin[0]:.2f}, {env_origin[1]:.2f})")
        print(f"地圖原點（相對環境中心）: ({map_origin[0]:.2f}, {map_origin[1]:.2f})")
        print(f"網格解析度: {grid_resolution}m")
        print(f"牆壁數量: {len(wall_configs)}")
        print(f"坐標系說明: 牆壁 center 已經是環境局部坐標，不需要減去 env_origin")

    # 添加每面牆
    total_marked = 0
    for wall in wall_configs:
        marked = add_rectangle_to_occupancy_grid(
            occupancy_grid=occupancy_grid,
            map_origin=map_origin,
            grid_resolution=grid_resolution,
            grid_width=grid_width,
            grid_depth=grid_depth,
            center=wall.center,
            size=wall.size,
            yaw=wall.yaw,
            env_origin=None,  # 🔥 牆壁 center 已經是環境局部坐標，不需要再次減去 env_origin
        )
        total_marked += marked

        if debug:
            print(f"  {wall.name}: center=({wall.center[0]:.2f}, {wall.center[1]:.2f}), "
                  f"size=({wall.size[0]:.2f}m x {wall.size[1]:.2f}m), marked={marked} grids")

    if debug:
        print(f"{'='*60}")
        print(f"總計: {total_marked} 個網格被標記為障礙物")
        print(f"{'='*60}\n")

    return total_marked


# ============================================================================
# 舊的 USD Mesh 讀取方法（保留用於向後兼容，但不推薦使用）
# ============================================================================

def get_wall_prims_from_usd(
    env: ManagerBasedRLEnv,
    env_id: int,
    debug: bool = False,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """從 USD stage 直接獲取牆壁 prims

    ⚠️ 不推薦使用這個方法！
    推薦使用 add_walls_from_config()，它更精準、更快速。

    這是「正統作法」：
    - 直接遍歷 USD stage
    - 找到所有有 CollisionAPI 的 prims
    - 獲取它們的世界位置

    Args:
        env: Isaac Lab 環境實例
        env_id: 環境 ID
        debug: 是否輸出調試信息

    Returns:
        List of (prim_path, position_local, size) tuples
    """
    import omni.usd
    from pxr import UsdPhysics, UsdGeom, Gf, Usd

    env_origin = env.scene.env_origins[env_id, :2].cpu().numpy()
    walls = []

    if debug:
        print(f"[WallGeometrySampler] env_{env_id}: 從 USD stage 獲取牆壁...")
        print(f"[WallGeometrySampler] env_origin: {env_origin}")

    # 獲取 USD stage
    stage = omni.usd.get_context().get_stage()

    # 構建該環境的命名空間前綴
    env_ns_prefix = f"/World/envs/env_{env_id}/"

    # 遍歷整個 USD stage
    count = 0
    collision_count = 0
    collision_prims = []  # 記錄所有碰撞物名稱（用於調試）

    for prim in stage.Traverse():
        count += 1

        # 只處理當前環境的 prims
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(env_ns_prefix):
            continue

        # 跳過 robot 和 sensor
        if "robot" in prim_path.lower() or "sensor" in prim_path.lower():
            continue

        # 檢查是否是牆壁（有 CollisionAPI）
        is_wall = False
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            is_wall = True
            collision_count += 1
            collision_prims.append(prim.GetName())  # 記錄用於調試

        if not is_wall:
            continue

        # 檢查是否匹配我們關心的牆壁命名模式
        # 🔥 重要：prim.GetName() 返回的是 'mesh'，需要從路徑提取名稱
        # 路徑格式: /World/envs/env_0/Wall_North/mesh
        prim_path = str(prim.GetPath())
        path_parts = prim_path.split("/")

        # 從路徑中提取牆壁名稱（通常在 mesh 上一層）
        wall_name = ""
        if len(path_parts) >= 2:
            # 倒數第二個元素通常是牆壁名稱
            potential_name = path_parts[-2] if path_parts[-1] == "mesh" else path_parts[-1]
            wall_name = potential_name.lower()

        # 🔥 寬鬆匹配：只要路徑包含關鍵詞就算
        prim_path_lower = prim_path.lower()
        is_target_wall = (
            "wall" in prim_path_lower or
            "u_wall" in prim_path_lower or
            "corridor" in prim_path_lower or
            "partition" in prim_path_lower or
            "tee" in prim_path_lower or
            "narrow" in prim_path_lower
        )

        if not is_target_wall:
            continue

        if debug:
            print(f"[WallGeometrySampler]   ✓ 找到牆壁: {prim_path}")

        # 獲取世界位置
        try:
            # 🔥 關鍵修復：對於 mesh prim，需要找到父 Xform prim
            # 路徑結構: /World/envs/env_0/Wall_North/geometry/mesh
            # 我們需要向上遍歷找到可計算 bounding box 的 prim

            # 方法 1: 嘗試將 prim 包裝為 Imageable（這是 Boundable 的基類）
            from pxr import UsdGeomImageable
            imageable = UsdGeomImageable(prim)
            if imageable:
                bbox = imageable.ComputeWorldBound(Usd.TimeCode.Default())
            else:
                # 方法 2: 向上找到父 prim
                parent = prim.GetParent()
                while parent and parent.IsValid():
                    parent_imageable = UsdGeomImageable(parent)
                    if parent_imageable:
                        bbox = parent_imageable.ComputeWorldBound(Usd.TimeCode.Default())
                        break
                    parent = parent.GetParent()
                else:
                    # 都找不到，跳過
                    continue

            range_obj = bbox.GetRange()
            size_3d = range_obj.GetSize()  # Gf.Vec3d

            # 獲取世界變換
            xformable = UsdGeom.Xformable(prim)
            world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = world_transform.ExtractTranslation()

            # 提取 X, Y 尺寸
            size_x = abs(size_3d[0])
            size_y = abs(size_3d[1])

            # 世界座標位置
            pos_w = np.array([translation[0], translation[1]])

            # 轉換為局部座標
            pos_local = pos_w - env_origin

            # 牆壁尺寸（X, Y）
            size = np.array([size_x, size_y])

            walls.append((prim_path, pos_local, size))

            if debug:
                print(f"[WallGeometrySampler]     位置: local=({pos_local[0]:.2f}, {pos_local[1]:.2f}), size=[{size[0]:.2f}, {size[1]:.2f}]")

        except Exception as e:
            if debug:
                print(f"[WallGeometrySampler]   處理失敗 {prim_path}: {e}")

    if debug:
        print(f"[WallGeometrySampler] 遍歷了 {count} 個 prims，找到 {collision_count} 個碰撞物，其中 {len(walls)} 面牆壁")
        if collision_count > 0 and len(walls) == 0:
            print(f"[WallGeometrySampler] 碰撞物名稱列表: {collision_prims}")
            print(f"[WallGeometrySampler] 嘗試匹配 'wall' 關鍵詞...")
            # 檢查是否有任何包含 'wall' 的
            wall_matching = [name for name in collision_prims if "wall" in name.lower()]
            if wall_matching:
                print(f"[WallGeometrySampler] 找到包含 'wall' 的: {wall_matching}")

    return walls


def get_wall_bounding_boxes(
    env: ManagerBasedRLEnv,
    env_id: int,
    debug: bool = False,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """從環境獲取牆壁的 bounding boxes（向後兼容的包裝函數）

    ⚠️ 不推薦使用！推薦使用 add_walls_from_config()。

    Args:
        env: Isaac Lab 環境實例
        env_id: 環境 ID
        debug: 是否輸出調試信息

    Returns:
        List of (center, size) tuples for each wall, in local coordinates
    """
    walls_from_usd = get_wall_prims_from_usd(env, env_id, debug)
    # 轉換為舊格式 (center, size)
    return [(pos, size) for _, pos, size in walls_from_usd]


def sample_wall_points_from_bboxes(
    wall_bboxes: List[Tuple[np.ndarray, np.ndarray]],
    sample_spacing: float = 0.05,
) -> np.ndarray:
    """從牆壁 bounding boxes 採樣點

    ⚠️ 不推薦使用！推薦使用 add_walls_from_config()。

    Args:
        wall_bboxes: List of (center, size) tuples
        sample_spacing: 採樣間距（米）

    Returns:
        [N, 2] 採樣點坐標（局部座標）
    """
    sampled_points = []

    for center, size in wall_bboxes:
        half_size = size / 2
        min_x, max_x = center[0] - half_size[0], center[0] + half_size[0]
        min_y, max_y = center[1] - half_size[1], center[1] + half_size[1]

        # 在牆壁邊界採樣點
        x_samples = max(2, int(np.ceil((max_x - min_x) / sample_spacing)))
        y_samples = max(2, int(np.ceil((max_y - min_y) / sample_spacing)))

        # 只採樣邊界（不是內部所有點）
        for i in range(x_samples):
            x = min_x + i * (max_x - min_x) / (x_samples - 1)
            sampled_points.append([x, min_y])  # 下邊

        for i in range(x_samples):
            x = min_x + i * (max_x - min_x) / (x_samples - 1)
            sampled_points.append([x, max_y])  # 上邊

        for i in range(y_samples):
            y = min_y + i * (max_y - min_y) / (y_samples - 1)
            sampled_points.append([min_x, y])  # 左邊

        for i in range(y_samples):
            y = min_y + i * (max_y - min_y) / (y_samples - 1)
            sampled_points.append([max_x, y])  # 右邊

    if len(sampled_points) == 0:
        return np.array([]).reshape(0, 2)

    return np.array(sampled_points, dtype=np.float32)


def add_wall_points_to_occupancy_grid(
    occupancy_grid: torch.Tensor,
    sampled_points_local: np.ndarray,
    map_origin: torch.Tensor,
    grid_resolution: float,
    grid_width: int,
    grid_depth: int,
) -> int:
    """將採樣的牆壁點添加到 occupancy grid

    ⚠️ 不推薦使用！推薦使用 add_walls_from_config()。
    """
    if len(sampled_points_local) == 0:
        return 0

    marked_count = 0

    for point in sampled_points_local:
        relative_pos = point - map_origin.cpu().numpy()
        grid_x = int(relative_pos[0] / grid_resolution)
        grid_y = int(relative_pos[1] / grid_resolution)

        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_depth:
            occupancy_grid[grid_x, grid_y] = 1
            marked_count += 1

    return marked_count


@dataclass
class WallGeometryVisualizer:
    """牆壁幾何可視化器（用於調試）"""

    def __init__(self, num_envs: int, prim_path: str = "/Visuals/WallPoints"):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils

        self.num_envs = num_envs
        self.markers: list = []

        for env_idx in range(min(4, num_envs)):
            marker_cfg = VisualizationMarkersCfg(
                prim_path=f"{prim_path}/env_{env_idx}",
                markers={
                    # 用方塊表示牆壁網格點（每個標記一個網格）
                    "wall_grid_point": sim_utils.CuboidCfg(
                        size=(0.1, 0.1, 0.1),  # 10cm 立方體
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.5, 0.0, 1.0),  # 紫色（與 LiDAR 區分）
                        ),
                    ),
                },
            )
            self.markers.append(VisualizationMarkers(marker_cfg))

    def visualize_from_grid(
        self,
        env_idx: int,
        occupancy_grid: torch.Tensor,
        map_origin: torch.Tensor,
        grid_resolution: float,
    ):
        """從 occupancy grid 可視化牆壁（顯示所有標記的網格）"""
        if env_idx >= len(self.markers):
            return

        # 找到所有被標記的網格點
        marked_indices = torch.nonzero(occupancy_grid)
        if len(marked_indices) == 0:
            # 清除可視化
            self.markers[env_idx].visualize(
                translations=np.zeros((0, 3))
            )
            return

        # 計算世界坐標
        origin_np = map_origin.cpu().numpy()
        grid_width = occupancy_grid.shape[0]
        grid_depth = occupancy_grid.shape[1]

        # 計算每個標記點的世界坐標
        translations = []
        for idx in marked_indices:
            gx, gy = idx[0].item(), idx[1].item()
            # 網格中心的世界坐標
            world_x = origin_np[0] + (gx + 0.5) * grid_resolution
            world_y = origin_np[1] + (gy + 0.5) * grid_resolution
            world_z = 0.5  # 牆壁高度一半
            translations.append([world_x, world_y, world_z])

        translations = np.array(translations, dtype=np.float32)

        # 方向（默認）
        orientations = np.zeros((len(translations), 4))
        orientations[:, 0] = 1.0

        self.markers[env_idx].visualize(
            translations=translations,
            orientations=orientations,
        )

    def visualize(
        self,
        env_idx: int,
        points_world: np.ndarray,
    ):
        """可視化採樣點（舊方法，保留向後兼容）"""
        if env_idx >= len(self.markers):
            return

        if len(points_world) == 0:
            self.markers[env_idx].visualize(
                translations=np.zeros((0, 3))
            )
            return

        points_3d = np.concatenate([
            points_world,
            np.full((len(points_world), 1), 0.2),
        ], axis=1)

        orientations = np.zeros((len(points_3d), 4))
        orientations[:, 0] = 1.0

        self.markers[env_idx].visualize(
            translations=points_3d,
            orientations=orientations,
        )

    def clear(self, env_idx: int):
        if env_idx < len(self.markers):
            self.markers[env_idx].visualize(
                translations=np.zeros((0, 3))
            )


__all__ = [
    # 推薦使用的新方法（數學幾何，精準且快速）
    "WallConfig",
    "get_phase1_wall_configs",
    "get_phase2_wall_configs",
    "get_phase3_wall_configs",
    "add_rectangle_to_occupancy_grid",
    "add_walls_from_config",

    # 舊方法（保留用於向後兼容，但不推薦使用）
    "add_wall_points_to_occupancy_grid",
    "WallGeometryVisualizer",
    "get_wall_bounding_boxes",
    "get_wall_prims_from_usd",
    "sample_wall_points_from_bboxes",
]
