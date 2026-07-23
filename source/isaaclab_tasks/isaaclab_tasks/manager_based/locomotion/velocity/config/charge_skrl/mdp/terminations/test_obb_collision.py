"""OBB 碰撞幾何純函式單元測試（無 Isaac Sim 依賴）。

實車完整量測值（2026-07-20）：
- 車長 0.70m → half_len = 0.35
- 車寬 0.60m（含輪子）→ half_wid = 0.30
- buffer = 0.10
- center_offset_x = -0.128m（碰撞盒中心沿車頭方向相對車體原點的偏移，負=偏後）
膨脹後 OBB = 0.90×0.80m。0.85m 縫每側剩 2.5cm；計入 root→OBB center
偏移在旋轉時造成的橫向掃掠後，理論容許偏航約 ±2.52°。

驗收（Codex 計劃步驟3）：無雜訊 deterministic
- 0.85m 窄縫：車頭對齊 → 可通過；偏航過大 / 方向不對 / 縫太窄 → 仍終止
- offset 造成前/後不對稱（車體原點偏前，前伸短、後伸長）

以直接檔案路徑載入 obb_collision.py，避免觸發 package __init__（其會 import isaaclab）。
"""
import importlib.util
import math
import pathlib

import torch

_MOD = pathlib.Path(__file__).parent / "obb_collision.py"
_spec = importlib.util.spec_from_file_location("obb_collision", _MOD)
obb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(obb)

L = 0.35        # half length
W = 0.30        # half width（含輪子）
BUF = 0.10
OFF = -0.128    # center_offset_x（車體系，負=盒中心偏後）

HALF_PI = math.pi / 2.0


def _walls_for_gap(gap: float, thickness: float = 1.0, length: float = 4.0):
    """兩面垂直牆形成沿 x 的走廊，內面間距 = gap。回傳 (centers[1,2,2], sizes[1,2,2], mask[1,2])。"""
    inner = gap / 2.0
    cx = inner + thickness / 2.0
    centers = torch.tensor([[[-cx, 0.0], [cx, 0.0]]])
    sizes = torch.tensor([[[thickness, length], [thickness, length]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    return centers, sizes, mask


def _wall_hit(robot_xy, yaw, centers, sizes, mask):
    return obb.obb_aabb_collision(robot_xy, yaw, centers, sizes, mask, L, W, BUF, OFF)


def _circle_hit(robot_xy, yaw, centers, radii, mask):
    return obb.obb_circle_collision(robot_xy, yaw, centers, radii, mask, L, W, BUF, OFF)


# ---------- 撞牆 OBB (OBB vs AABB) ----------

def test_wall_085_gap_aligned_passes():
    """0.85m 縫、車頭直行對齊（寬 0.60 面對牆）→ 不判碰撞（每側剩 2.5cm）。圓模型會拒走，關鍵改善。"""
    centers, sizes, mask = _walls_for_gap(0.85)
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI])   # 前向 = +y（穿過走廊），寬向面 ±x 牆
    assert _wall_hit(robot_xy, yaw, centers, sizes, mask).item() is False


def test_wall_085_gap_yaw_2deg_passes():
    """計入 OBB center offset 後，0.85m 縫的偏航邊界約 2.52°。"""
    centers, sizes, mask = _walls_for_gap(0.85)
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI + math.radians(2.0)])
    assert _wall_hit(robot_xy, yaw, centers, sizes, mask).item() is False


def test_wall_085_gap_yaw_3deg_collides():
    centers, sizes, mask = _walls_for_gap(0.85)
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI + math.radians(3.0)])
    assert _wall_hit(robot_xy, yaw, centers, sizes, mask).item() is True


def test_wall_085_gap_lateral_offset_2cm_passes():
    """置中時每側 2.5cm；橫移 2cm 可通，3cm 應撞。"""
    centers, sizes, mask = _walls_for_gap(0.85)
    robot_xy = torch.tensor([[0.02, 0.0]])
    yaw = torch.tensor([HALF_PI])
    assert _wall_hit(robot_xy, yaw, centers, sizes, mask).item() is False


def test_wall_085_gap_lateral_offset_3cm_collides():
    centers, sizes, mask = _walls_for_gap(0.85)
    robot_xy = torch.tensor([[0.03, 0.0]])
    yaw = torch.tensor([HALF_PI])
    assert _wall_hit(robot_xy, yaw, centers, sizes, mask).item() is True


def test_wall_085_gap_misaligned_length_across_collides():
    """0.85m 縫、車身轉 90°（長 0.70 橫跨縫）→ 判碰撞。"""
    centers, sizes, mask = _walls_for_gap(0.85)
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([0.0])       # 前向 = +x，長向面 ±x 牆
    assert _wall_hit(robot_xy, yaw, centers, sizes, mask).item() is True


def test_wall_too_narrow_gap_collides_even_aligned():
    """0.35m 縫、即使對齊（寬向 0.40 > 內面 0.175）→ 仍判碰撞。"""
    centers, sizes, mask = _walls_for_gap(0.35)
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI])
    assert _wall_hit(robot_xy, yaw, centers, sizes, mask).item() is True


def test_wall_inactive_mask_never_collides():
    """牆 mask=False → 不論多近都不判碰撞。"""
    centers, sizes, _ = _walls_for_gap(0.10)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI])
    assert _wall_hit(robot_xy, yaw, centers, sizes, mask).item() is False


# ---------- 撞柱子 OBB (OBB vs circle，用物理半徑) ----------

def _obs(px, py, r=0.3):
    centers = torch.tensor([[[px, py]]])
    radii = torch.tensor([[r]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    return centers, radii, mask


def test_obstacle_front_collides_within_reach():
    """障礙正前 d=0.55 < 前伸門檻 0.622（原點偏前，前伸=0.35-0.128=0.222）→ 碰撞。"""
    centers, radii, mask = _obs(0.0, 0.55)
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI])   # 前向 +y
    assert _circle_hit(robot_xy, yaw, centers, radii, mask).item() is True


def test_obstacle_front_clear_beyond_reach():
    """障礙正前 d=0.70 > 0.622 → 不碰撞。"""
    centers, radii, mask = _obs(0.0, 0.70)
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI])
    assert _circle_hit(robot_xy, yaw, centers, radii, mask).item() is False


def test_obstacle_rear_reaches_further_than_front():
    """offset 造成後伸(0.478)>前伸(0.222)：正後 d=0.80 撞，但正前 d=0.80 不撞。"""
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI])
    c_rear, r_rear, m_rear = _obs(0.0, -0.80)   # 車後方
    c_front, r_front, m_front = _obs(0.0, 0.80)  # 車前方
    assert _circle_hit(robot_xy, yaw, c_rear, r_rear, m_rear).item() is True
    assert _circle_hit(robot_xy, yaw, c_front, r_front, m_front).item() is False


def test_obstacle_side_reach():
    """側向門檻 0.70（寬 0.30 + 障礙 0.3 + buffer 0.10）：x=0.60 撞、x=0.80 不撞。"""
    robot_xy = torch.tensor([[0.0, 0.0]])
    yaw = torch.tensor([HALF_PI])   # 寬向面 x
    c1, r1, m1 = _obs(0.60, 0.0)
    c2, r2, m2 = _obs(0.80, 0.0)
    assert _circle_hit(robot_xy, yaw, c1, r1, m1).item() is True
    assert _circle_hit(robot_xy, yaw, c2, r2, m2).item() is False


# 純 torch 幾何、無 Isaac Sim 依賴 → 支援獨立執行（避免 pytest 觸發 isaaclab package import）
if __name__ == "__main__":
    import sys

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
