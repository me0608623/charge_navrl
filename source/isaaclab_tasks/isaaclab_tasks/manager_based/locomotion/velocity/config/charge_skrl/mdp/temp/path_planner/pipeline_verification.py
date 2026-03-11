"""
USD → AIT* Pipeline 驗收測試 (Pipeline Verification Tests)

三個層級、八個測試，全面驗證 USD 牆壁幾何是否正確進入 AIT* 規劃器。

A. 資料層驗收（牆真的進到 grid）
A1. 牆點數量統計（穩定且合理）
A2. 範圍檢查（grid index 不越界）
A3. Probe 測試（牆點在 grid 中標記為 1）

B. 幾何層驗收（紫球與牆對齊，門口沒被封）
B1. Doorway Probe（門口位置必須是 0）
B2. 膨脹策略驗收（inflation 半徑正確）

C. 行為層驗收（AIT* 路徑真的改變）
C1. A/B 測試（清空 grid vs 有牆 grid）
C2. 牆兩側測試（start/goal 分在牆兩側）
C3. hit-wall 計數器（統計無效狀態）
C4. 失敗模式檢查（路徑震盪、找不到路）

使用方式：
    from mdp.path_planner import run_pipeline_verification

    # 在環境重置時運行驗收測試
    results = run_pipeline_verification(env, env_id=0, phase="phase1")
    results.print_summary()
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TestStatus(Enum):
    """測試狀態"""
    PASS = "✓ PASS"
    FAIL = "✗ FAIL"
    WARN = "⚠ WARN"


@dataclass
class TestResult:
    """單個測試結果"""
    name: str
    status: TestStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResults:
    """驗收測試結果集合"""
    phase: str
    env_id: int
    tests: List[TestResult] = field(default_factory=list)

    def add_test(self, name: str, status: TestStatus, message: str, **details):
        """添加測試結果"""
        self.tests.append(TestResult(name, status, message, details))

    def print_summary(self):
        """印出測試摘要"""
        print(f"\n{'='*70}")
        print(f"Pipeline 驗收測試結果 - {self.phase.upper()} (env_{self.env_id})")
        print(f"{'='*70}")

        pass_count = sum(1 for t in self.tests if t.status == TestStatus.PASS)
        fail_count = sum(1 for t in self.tests if t.status == TestStatus.FAIL)
        warn_count = sum(1 for t in self.tests if t.status == TestStatus.WARN)

        for test in self.tests:
            print(f"\n{test.status.value} {test.name}")
            print(f"    {test.message}")
            if test.details:
                for key, value in test.details.items():
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value}")
                    elif isinstance(value, (list, tuple)) and len(value) > 0:
                        if len(value) <= 5:
                            print(f"    {key}: {value}")
                        else:
                            print(f"    {key}: [{value[0]}, ..., {value[-1]}] (n={len(value)})")

        print(f"\n{'='*70}")
        print(f"總計: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")
        print(f"{'='*70}\n")


# ============================================================================
# A. 資料層驗收
# ============================================================================

def test_a1_wall_point_count(
    wall_bboxes: List[Tuple[np.ndarray, np.ndarray]],
    sampled_points: np.ndarray,
    grid_marked_count: int,
) -> TestResult:
    """A1. 牆點數量統計（穩定且合理）"""
    num_walls = len(wall_bboxes)
    num_sampled = len(sampled_points)
    num_grid_marked = grid_marked_count

    # 合理性檢查
    if num_walls == 0:
        return TestResult(
            "A1. 牆點數量統計",
            TestStatus.FAIL,
            "沒有檢測到任何牆壁！",
            details={"num_walls": 0, "num_sampled": 0, "num_grid_marked": 0}
        )

    if num_sampled < 50:
        return TestResult(
            "A1. 牆點數量統計",
            TestStatus.WARN,
            "採樣點數量過少，可能採樣間距太大或牆壁太小",
            details={"num_walls": num_walls, "num_sampled": num_sampled, "num_grid_marked": num_grid_marked}
        )

    if num_sampled > 10000:
        return TestResult(
            "A1. 牆點數量統計",
            TestStatus.WARN,
            "採樣點數量過多，可能採樣間距太小或有重複",
            details={"num_walls": num_walls, "num_sampled": num_sampled, "num_grid_marked": num_grid_marked}
        )

    if num_grid_marked == 0:
        return TestResult(
            "A1. 牆點數量統計",
            TestStatus.FAIL,
            "沒有標記任何 grid cell，座標轉換可能有問題",
            details={"num_walls": num_walls, "num_sampled": num_sampled, "num_grid_marked": num_grid_marked}
        )

    if num_grid_marked > 5000:
        return TestResult(
            "A1. 牆點數量統計",
            TestStatus.WARN,
            "標記的 grid cell 過多，可能採樣範圍錯誤",
            details={"num_walls": num_walls, "num_sampled": num_sampled, "num_grid_marked": num_grid_marked}
        )

    return TestResult(
        "A1. 牆點數量統計",
        TestStatus.PASS,
        f"檢測到 {num_walls} 面牆，採樣 {num_sampled} 點，標記 {num_grid_marked} 個 grid cell",
        details={"num_walls": num_walls, "num_sampled": num_sampled, "num_grid_marked": num_grid_marked}
    )


def test_a2_range_check(
    sampled_points_local: np.ndarray,
    grid_resolution: float,
    grid_width: int,
    grid_depth: int,
    map_origin: torch.Tensor,
) -> TestResult:
    """A2. 範圍檢查（grid index 不越界）"""
    if len(sampled_points_local) == 0:
        return TestResult(
            "A2. 範圍檢查",
            TestStatus.FAIL,
            "沒有採樣點，無法檢查範圍",
            details={}
        )

    # 計算 grid index
    origin_np = map_origin.cpu().numpy()
    relative_pos = sampled_points_local - origin_np
    grid_x = (relative_pos[:, 0] / grid_resolution).astype(int)
    grid_y = (relative_pos[:, 1] / grid_resolution).astype(int)

    x_min, x_max = grid_x.min(), grid_x.max()
    y_min, y_max = grid_y.min(), grid_y.max()

    # 檢查是否越界
    out_of_bounds = ((grid_x < 0) | (grid_x >= grid_width) |
                     (grid_y < 0) | (grid_y >= grid_depth)).sum()

    if out_of_bounds > 0:
        return TestResult(
            "A2. 範圍檢查",
            TestStatus.FAIL,
            f"有 {out_of_bounds} 個點的 grid index 越界！",
            details={
                "grid_x_min": x_min, "grid_x_max": x_max,
                "grid_y_min": y_min, "grid_y_max": y_max,
                "grid_width": grid_width, "grid_depth": grid_depth,
                "out_of_bounds": out_of_bounds
            }
        )

    # 檢查範圍是否合理（對於 16x16 房間，local 範圍應該在 [-8, 8] 左右）
    local_x_min, local_x_max = sampled_points_local[:, 0].min(), sampled_points_local[:, 0].max()
    local_y_min, local_y_max = sampled_points_local[:, 1].min(), sampled_points_local[:, 1].max()

    if local_x_min < -9 or local_x_max > 9 or local_y_min < -9 or local_y_max > 9:
        return TestResult(
            "A2. 範圍檢查",
            TestStatus.WARN,
            "採樣點範圍超出 16x16 房間邊界，可能 env_origin 錯誤",
            details={
                "local_x_range": (local_x_min, local_x_max),
                "local_y_range": (local_y_min, local_y_max),
                "grid_x_range": (x_min, x_max),
                "grid_y_range": (y_min, y_max),
            }
        )

    return TestResult(
        "A2. 範圍檢查",
        TestStatus.PASS,
        f"Grid index 範圍正常: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]",
        details={
            "grid_x_min": x_min, "grid_x_max": x_max,
            "grid_y_min": y_min, "grid_y_max": y_max,
            "local_x_range": (round(local_x_min, 2), round(local_x_max, 2)),
            "local_y_range": (round(local_y_min, 2), round(local_y_max, 2)),
        }
    )


def test_a3_probe_test(
    sampled_points_local: np.ndarray,
    occupancy_grid: torch.Tensor,
    map_origin: torch.Tensor,
    grid_resolution: float,
    grid_width: int,
    grid_depth: int,
    num_probes: int = 20,
) -> TestResult:
    """A3. Probe 測試（牆點在 grid 中標記為 1）"""
    if len(sampled_points_local) == 0:
        return TestResult(
            "A3. Probe 測試",
            TestStatus.FAIL,
            "沒有採樣點，無法進行 probe 測試",
            details={}
        )

    # 隨機抽樣 probe 點
    num_points = len(sampled_points_local)
    num_probes = min(num_probes, num_points)
    indices = np.random.choice(num_points, num_probes, replace=False)

    origin_np = map_origin.cpu().numpy()
    probe_points = sampled_points_local[indices]

    marked_count = 0
    unmarked_count = 0

    for point in probe_points:
        relative_pos = point - origin_np
        grid_x = int(relative_pos[0] / grid_resolution)
        grid_y = int(relative_pos[1] / grid_resolution)

        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_depth:
            if occupancy_grid[grid_x, grid_y].item() > 0:
                marked_count += 1
            else:
                unmarked_count += 1

    if unmarked_count > 0:
        return TestResult(
            "A3. Probe 測試",
            TestStatus.FAIL,
            f"有 {unmarked_count}/{num_probes} 個牆點在 grid 中未被標記！",
            details={
                "num_probes": num_probes,
                "marked_count": marked_count,
                "unmarked_count": unmarked_count
            }
        )

    return TestResult(
        "A3. Probe 測試",
        TestStatus.PASS,
        f"所有 {num_probes} 個隨機 probe 點都在 grid 中正確標記為 1",
        details={
            "num_probes": num_probes,
            "marked_count": marked_count,
            "unmarked_count": unmarked_count
        }
    )


# ============================================================================
# B. 幾何層驗收
# ============================================================================

# 各 Phase 的門口位置（局部座標）
DOORWAY_POSITIONS = {
    "phase1": [
        # U-Wall 開口（朝東）
        (0.0, 0.0),  # U 型牆開口中心
        # Partition 門口（1.5m 寬，中心位置）
        (4.0, 0.0),  # 隔間門口中心
    ],
    "phase2": [
        # 走廊門口（1.3m 寬）
        (-2.0, 3.0),  # 左側門口
        (2.0, 3.0),   # 中間門口
        (6.0, 3.0),   # 右側門口
        # T 型路口
        (0.0, 4.5),   # T 型走廊上段
        # 窄通道
        (6.0, 6.1),   # Narrow1 門口
        (-6.0, -5.4), # Narrow2 門口
    ],
    "phase3": [
        # Phase 3 結合 Phase 1 和 2，使用主要門口
        (0.0, 0.0),   # U-Wall 開口
        (4.0, 0.0),   # 隔間門口
    ],
}


def test_b1_doorway_probe(
    occupancy_grid: torch.Tensor,
    map_origin: torch.Tensor,
    grid_resolution: float,
    grid_width: int,
    grid_depth: int,
    phase: str = "phase1",
    robot_radius: float = 0.3,
) -> TestResult:
    """B1. Doorway Probe（門口位置必須是 0）"""
    doorways = DOORWAY_POSITIONS.get(phase, DOORWAY_POSITIONS["phase1"])

    origin_np = map_origin.cpu().numpy()
    blocked_count = 0
    tested_count = 0
    blocked_doorways = []

    for doorway_pos in doorways:
        # 在門口周圍採樣多個點（考慮門寬）
        doorway_x, doorway_y = doorway_pos
        probe_radius = 0.5  # 門口半徑
        probe_points = [
            (doorway_x, doorway_y),
            (doorway_x + probe_radius, doorway_y),
            (doorway_x - probe_radius, doorway_y),
            (doorway_x, doorway_y + probe_radius),
            (doorway_x, doorway_y - probe_radius),
        ]

        doorway_blocked = False
        for px, py in probe_points:
            relative_pos = np.array([px, py]) - origin_np
            grid_x = int(relative_pos[0] / grid_resolution)
            grid_y = int(relative_pos[1] / grid_resolution)

            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_depth:
                tested_count += 1
                if occupancy_grid[grid_x, grid_y].item() > 0:
                    doorway_blocked = True
                    blocked_count += 1

        if doorway_blocked:
            blocked_doorways.append(doorway_pos)

    if blocked_count > tested_count * 0.5:
        return TestResult(
            "B1. Doorway Probe",
            TestStatus.FAIL,
            f"門口被封死！{blocked_count}/{tested_count} 個門口 probe 點被標記為障礙",
            details={
                "tested_count": tested_count,
                "blocked_count": blocked_count,
                "blocked_doorways": blocked_doorways
            }
        )

    if blocked_count > 0:
        return TestResult(
            "B1. Doorway Probe",
            TestStatus.WARN,
            f"部分門口 probe 點被標記為障礙 ({blocked_count}/{tested_count})，可能膨脹半徑過大",
            details={
                "tested_count": tested_count,
                "blocked_count": blocked_count,
                "blocked_doorways": blocked_doorways
            }
        )

    return TestResult(
        "B1. Doorway Probe",
        TestStatus.PASS,
        f"所有 {len(doorways)} 個門口通過 probe 測試，可通行",
        details={
            "num_doorways": len(doorways),
            "tested_count": tested_count,
            "blocked_count": blocked_count
        }
    )


def test_b2_inflation_check(
    robot_radius: float,
    grid_resolution: float,
    obstacle_inflation: float,
) -> TestResult:
    """B2. 膨脹策略驗收（inflation 半徑正確）"""
    expected_inflation_cells = int(np.ceil(robot_radius / grid_resolution))
    actual_inflation_cells = int(np.ceil(obstacle_inflation / grid_resolution))

    # 檢查膨脹半徑是否合理
    if actual_inflation_cells == 0:
        return TestResult(
            "B2. 膨脹策略驗收",
            TestStatus.WARN,
            "沒有膨脹！機器人可能會擦牆",
            details={
                "robot_radius": robot_radius,
                "obstacle_inflation": obstacle_inflation,
                "expected_cells": expected_inflation_cells,
                "actual_cells": actual_inflation_cells
            }
        )

    # 膨脹不應該超過機器人寬度的 2 倍
    max_inflation_cells = int(np.ceil(robot_radius * 2 / grid_resolution))
    if actual_inflation_cells > max_inflation_cells:
        return TestResult(
            "B2. 膨脹策略驗收",
            TestStatus.WARN,
            f"膨脹半徑過大 ({actual_inflation_cells} cells)，可能會封死門口",
            details={
                "robot_radius": robot_radius,
                "obstacle_inflation": obstacle_inflation,
                "expected_cells": expected_inflation_cells,
                "actual_cells": actual_inflation_cells,
                "max_cells": max_inflation_cells
            }
        )

    return TestResult(
        "B2. 膨脹策略驗收",
        TestStatus.PASS,
        f"膨脹半徑正確: {actual_inflation_cells} cells ({obstacle_inflation}m)",
        details={
            "robot_radius": robot_radius,
            "obstacle_inflation": obstacle_inflation,
            "inflation_cells": actual_inflation_cells
        }
    )


# ============================================================================
# C. 行為層驗收
# ============================================================================

def test_c1_ab_test(
    planner,
    start: np.ndarray,
    goal: np.ndarray,
    env: ManagerBasedRLEnv,
) -> TestResult:
    """C1. A/B 測試（清空 grid vs 有牆 grid）"""
    # 備份原始 occupancy grid
    original_grid = planner.map.occupancy_grid.clone()

    # 測試 A: 清空 grid
    planner.map.occupancy_grid.zero_()
    path_empty = planner.plan_path(torch.from_numpy(start), torch.from_numpy(goal), env=None)

    # 測試 B: 恢復 grid
    planner.map.occupancy_grid.copy_(original_grid)
    path_with_walls = planner.plan_path(torch.from_numpy(start), torch.from_numpy(goal), env=None)

    # 比較路徑
    empty_length = len(path_empty)
    walls_length = len(path_with_walls)

    if empty_length == 0:
        return TestResult(
            "C1. A/B 測試",
            TestStatus.WARN,
            "清空 grid 後仍找不到路徑，start/goal 可能無效",
            details={
                "start": start.tolist(),
                "goal": goal.tolist(),
                "empty_path_length": empty_length,
                "walls_path_length": walls_length
            }
        )

    if walls_length == 0:
        return TestResult(
            "C1. A/B 測試",
            TestStatus.WARN,
            "有牆 grid 找不到路徑，門口可能被封死",
            details={
                "start": start.tolist(),
                "goal": goal.tolist(),
                "empty_path_length": empty_length,
                "walls_path_length": walls_length
            }
        )

    # 計算路徑直線距離
    if empty_length >= 2 and walls_length >= 2:
        empty_distance = torch.norm(path_empty[-1] - path_empty[0]).item()
        walls_distance = torch.norm(path_with_walls[-1] - path_with_walls[0]).item()

        if walls_length <= empty_length + 1:
            return TestResult(
                "C1. A/B 測試",
                TestStatus.WARN,
                f"有牆路徑 ({walls_length} nodes) 與清空路徑 ({empty_length} nodes) 長度相近，牆壁可能沒有生效",
                details={
                    "empty_path_length": empty_length,
                    "walls_path_length": walls_length,
                    "empty_distance": round(empty_distance, 2),
                    "walls_distance": round(walls_distance, 2)
                }
            )

    return TestResult(
        "C1. A/B 測試",
        TestStatus.PASS,
        f"路徑差異明顯: 清空 {empty_length} nodes vs 有牆 {walls_length} nodes",
        details={
            "empty_path_length": empty_length,
            "walls_path_length": walls_length
        }
    )


# 各 Phase 的牆兩側測試位置
WALL_SIDES_TESTS = {
    "phase1": [
        # U-Wall 兩側測試
        {"start": (-5.0, 1.5), "goal": (0.0, 1.5), "description": "U-Wall 左→右"},
        # Partition 兩側測試
        {"start": (3.0, 0.0), "goal": (6.0, 0.0), "description": "Partition 左→右（需穿過門口）"},
    ],
    "phase2": [
        # 走廊測試
        {"start": (-5.0, 3.0), "goal": (5.0, 3.0), "description": "走廊左→右（需穿過門口）"},
        # T 型路口測試
        {"start": (0.0, 6.0), "goal": (0.0, -6.0), "description": "T型路口上→下"},
    ],
    "phase3": [
        # 綜合測試
        {"start": (-6.0, 2.0), "goal": (2.0, 2.0), "description": "繞過 U-Wall"},
    ],
}


def test_c2_wall_sides_test(
    planner,
    phase: str,
    env: ManagerBasedRLEnv,
) -> TestResult:
    """C2. 牆兩側測試（start/goal 分在牆兩側）"""
    tests = WALL_SIDES_TESTS.get(phase, WALL_SIDES_TESTS["phase1"])

    results = []
    for test_case in tests:
        start = np.array(test_case["start"])
        goal = np.array(test_case["goal"])
        description = test_case["description"]

        path = planner.plan_path(torch.from_numpy(start), torch.from_numpy(goal), env=None)

        if len(path) <= 2:
            results.append({
                "description": description,
                "path_length": len(path),
                "status": "FAIL" if len(path) == 0 else "WARN",
                "message": f"路徑過短，可能沒有繞行"
            })
        else:
            results.append({
                "description": description,
                "path_length": len(path),
                "status": "PASS",
                "message": f"路徑有 {len(path)} 個節點"
            })

    # 統計結果
    pass_count = sum(1 for r in results if r["status"] == "PASS")
    total_count = len(results)

    if pass_count == 0:
        return TestResult(
            "C2. 牆兩側測試",
            TestStatus.FAIL,
            f"所有 {total_count} 個測試失敗，牆壁可能沒有生效",
            details={"results": results}
        )

    if pass_count < total_count:
        return TestResult(
            "C2. 牆兩側測試",
            TestStatus.WARN,
            f"部分測試失敗: {pass_count}/{total_count} 通過",
            details={"results": results}
        )

    return TestResult(
        "C2. 牆兩側測試",
        TestStatus.PASS,
        f"所有 {total_count} 個牆兩側測試通過",
        details={"results": results}
    )


@dataclass
class CollisionStats:
    """碰撞統計"""
    num_invalid_due_to_wall: int = 0
    num_valid_states: int = 0
    num_total_checks: int = 0


def test_c3_hit_wall_counter(
    planner,
    stats: CollisionStats,
) -> TestResult:
    """C3. hit-wall 計數器（統計無效狀態）"""
    if stats.num_total_checks == 0:
        return TestResult(
            "C3. hit-wall 計數器",
            TestStatus.WARN,
            "沒有進行任何狀態檢查",
            details={}
        )

    invalid_ratio = stats.num_invalid_due_to_wall / stats.num_total_checks

    if stats.num_invalid_due_to_wall == 0:
        return TestResult(
            "C3. hit-wall 計數器",
            TestStatus.WARN,
            "沒有檢測到任何因牆壁而無效的狀態，牆壁可能沒有在規劃時被查到",
            details={
                "num_invalid": 0,
                "num_valid": stats.num_valid_states,
                "num_total": stats.num_total_checks
            }
        )

    return TestResult(
        "C3. hit-wall 計數器",
        TestStatus.PASS,
        f"檢測到 {stats.num_invalid_due_to_wall} 個因牆壁無效的狀態 ({invalid_ratio*100:.1f}%)",
        details={
            "num_invalid": stats.num_invalid_due_to_wall,
            "num_valid": stats.num_valid_states,
            "num_total": stats.num_total_checks,
            "invalid_ratio": round(invalid_ratio * 100, 1)
        }
    )


def test_c4_failure_mode_check(
    path: torch.Tensor,
) -> TestResult:
    """C4. 失敗模式檢查（路徑震盪、找不到路）"""
    if len(path) == 0:
        return TestResult(
            "C4. 失敗模式檢查",
            TestStatus.FAIL,
            "找不到路徑！門口可能被封死或 start/goal 無效",
            details={"path_length": 0}
        )

    # 檢查路徑震盪（相鄰點之間距離過小）
    if len(path) > 3:
        distances = []
        for i in range(len(path) - 1):
            dist = torch.norm(path[i+1] - path[i]).item()
            distances.append(dist)

        # 計算小於閾值的距離數量
        tiny_jumps = sum(1 for d in distances if d < 0.1)
        tiny_ratio = tiny_jumps / len(distances)

        if tiny_ratio > 0.3:
            return TestResult(
                "C4. 失敗模式檢查",
                TestStatus.WARN,
                f"路徑有 {tiny_ratio*100:.1f}% 的微小跳躍，可能有震盪現象",
                details={
                    "path_length": len(path),
                    "tiny_jumps": tiny_jumps,
                    "tiny_ratio": round(tiny_ratio * 100, 1)
                }
            )

    # 檢查路徑是否過度曲折（平均步長太小）
    if len(path) > 10:
        total_length = 0.0
        for i in range(len(path) - 1):
            total_length += torch.norm(path[i+1] - path[i]).item()

        direct_distance = torch.norm(path[-1] - path[0]).item()
        if direct_distance > 0:
            winding_ratio = total_length / direct_distance
            if winding_ratio > 3.0:
                return TestResult(
                    "C4. 失敗模式檢查",
                    TestStatus.WARN,
                    f"路徑過度曲折 (繞路比 {winding_ratio:.1f}x)，grid 可能太粗",
                    details={
                        "path_length": len(path),
                        "total_distance": round(total_length, 2),
                        "direct_distance": round(direct_distance, 2),
                        "winding_ratio": round(winding_ratio, 2)
                    }
                )

    return TestResult(
        "C4. 失敗模式檢查",
        TestStatus.PASS,
        f"路徑正常: {len(path)} 個節點",
        details={"path_length": len(path)}
    )


# ============================================================================
# 完整驗收測試
# ============================================================================

@dataclass
class PipelineVerifier:
    """Pipeline 驗收測試器"""

    phase: str = "phase1"
    robot_radius: float = 0.3
    grid_resolution: float = 0.1

    # 統計數據
    collision_stats: CollisionStats = field(default_factory=CollisionStats)

    def run_all_tests(
        self,
        env: ManagerBasedRLEnv,
        env_id: int,
        wall_bboxes: List[Tuple[np.ndarray, np.ndarray]],
        sampled_points_local: np.ndarray,
        grid_marked_count: int,
    ) -> VerificationResults:
        """運行所有驗收測試"""
        results = VerificationResults(phase=self.phase, env_id=env_id)

        # 檢查 AIT* 規劃器是否存在
        if not hasattr(env, "_aitstar_planner"):
            results.add_test(
                "AIT* 規劃器",
                TestStatus.FAIL,
                "AIT* 規劃器未初始化"
            )
            return results

        planner = env._aitstar_planner
        map_origin = planner.map.map_origin
        occupancy_grid = planner.map.occupancy_grid
        grid_width = planner.map.grid_width
        grid_depth = planner.map.grid_depth

        # ================================================================
        # A. 資料層驗收
        # ================================================================

        # A1. 牆點數量統計
        result_a1 = test_a1_wall_point_count(
            wall_bboxes, sampled_points_local, grid_marked_count
        )
        results.tests.append(result_a1)

        # A2. 範圍檢查
        result_a2 = test_a2_range_check(
            sampled_points_local, self.grid_resolution, grid_width, grid_depth, map_origin
        )
        results.tests.append(result_a2)

        # A3. Probe 測試
        result_a3 = test_a3_probe_test(
            sampled_points_local, occupancy_grid, map_origin,
            self.grid_resolution, grid_width, grid_depth
        )
        results.tests.append(result_a3)

        # ================================================================
        # B. 幾何層驗收
        # ================================================================

        # B1. Doorway Probe
        result_b1 = test_b1_doorway_probe(
            occupancy_grid, map_origin, self.grid_resolution,
            grid_width, grid_depth, self.phase, self.robot_radius
        )
        results.tests.append(result_b1)

        # B2. 膨脹策略驗收
        result_b2 = test_b2_inflation_check(
            self.robot_radius, self.grid_resolution, planner.map.cfg.obstacle_inflation
        )
        results.tests.append(result_b2)

        # ================================================================
        # C. 行為層驗收
        # ================================================================

        # 獲取當前 start/goal
        robot = env.scene["robot"]
        robot_pos_world = robot.data.root_pos_w[env_id, :2].cpu().numpy()
        goal_pos_world = env.command_manager.get_command("goal_command")[env_id, :2].cpu().numpy()
        env_origin = env.scene.env_origins[env_id, :2].cpu().numpy()

        start = robot_pos_world - env_origin
        goal = goal_pos_world - env_origin
        goal = np.clip(goal, -7.0, 7.0)

        # C1. A/B 測試
        result_c1 = test_c1_ab_test(planner, start, goal, env)
        results.tests.append(result_c1)

        # C2. 牆兩側測試
        result_c2 = test_c2_wall_sides_test(planner, self.phase, env)
        results.tests.append(result_c2)

        # C3. hit-wall 計數器
        result_c3 = test_c3_hit_wall_counter(planner, self.collision_stats)
        results.tests.append(result_c3)

        # C4. 失敗模式檢查
        path = planner.plan_path(torch.from_numpy(start), torch.from_numpy(goal), env=None)
        result_c4 = test_c4_failure_mode_check(path)
        results.tests.append(result_c4)

        return results


# 創建驗收測試器的便利函數
def create_pipeline_verifier(
    phase: str = "phase1",
    robot_radius: float = 0.3,
    grid_resolution: float = 0.1,
) -> PipelineVerifier:
    """創建 Pipeline 驗收測試器"""
    return PipelineVerifier(
        phase=phase,
        robot_radius=robot_radius,
        grid_resolution=grid_resolution,
    )


__all__ = [
    "PipelineVerifier",
    "VerificationResults",
    "TestResult",
    "TestStatus",
    "CollisionStats",
    "create_pipeline_verifier",
    # 測試函數
    "test_a1_wall_point_count",
    "test_a2_range_check",
    "test_a3_probe_test",
    "test_b1_doorway_probe",
    "test_b2_inflation_check",
    "test_c1_ab_test",
    "test_c2_wall_sides_test",
    "test_c3_hit_wall_counter",
    "test_c4_failure_mode_check",
]
