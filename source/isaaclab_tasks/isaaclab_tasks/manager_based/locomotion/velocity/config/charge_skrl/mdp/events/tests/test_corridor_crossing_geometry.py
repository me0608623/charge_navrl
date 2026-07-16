"""Task 1 pytest suite — corridor crossing geometry helpers.

Brief 要求優先使用完整套件路徑匯入。實際執行時 isaaclab_tasks.__init__ 會觸發
pxr (Isaac Sim) 而失敗，因此改用 importlib 直接載入模組檔案（sim-free 保證）。
所有 brief 指定的 assert 完全保留不變。
"""
import math
import importlib.util
import pathlib

# ── file-path loader（sim-free）──────────────────────────────────────────────
# 完整套件路徑 (isaaclab_tasks...) 在純 pytest 環境中會透過 pxr 拉入 Isaac Sim
# 而失敗，因此直接從檔案載入以確保測試環境獨立。
_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "corridor_crossing_geometry.py"
)
_spec = importlib.util.spec_from_file_location("corridor_crossing_geometry", _MODULE_PATH)
g = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(g)  # type: ignore[union-attr]

import torch


def test_select_corridor_envs_uses_rand_threshold():
    env_ids = torch.tensor([10, 11, 12, 13])
    rand = torch.tensor([0.05, 0.5, 0.11, 0.9])
    sel = g.select_corridor_envs(env_ids, fraction=0.12, rand=rand)
    assert sel.tolist() == [10, 12]


def test_select_corridor_envs_zero_fraction_empty():
    env_ids = torch.tensor([1, 2, 3])
    rand = torch.zeros(3)
    sel = g.select_corridor_envs(env_ids, fraction=0.0, rand=rand)
    assert sel.numel() == 0


def test_robot_pose_faces_plus_y():
    origins = torch.zeros(2, 3)
    pose = g.robot_corridor_pose(origins, spawn_y=-3.0)
    assert pose.shape == (2, 7)
    assert torch.allclose(pose[:, :3], torch.tensor([0.0, -3.0, 0.0]).expand(2, 3))
    # yaw +90deg about Z -> qw=qz=cos/sin(45deg), qx=qy=0
    expected = torch.tensor([math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)])
    assert torch.allclose(pose[:, 3:], expected.expand(2, 4), atol=1e-6)


def test_goal_pos_ahead_in_y():
    origins = torch.tensor([[5.0, 7.0, 0.0]])
    goal = g.goal_corridor_pos(origins, goal_y=3.0)
    assert torch.allclose(goal, torch.tensor([[5.0, 10.0, 0.0]]))


def test_wall_pose_rotated_90():
    origins = torch.zeros(1, 3)
    pose = g.wall_corridor_pose(origins, wall_x=-2.0, wall_z=1.5)
    assert torch.allclose(pose[:, :3], torch.tensor([[-2.0, 0.0, 1.5]]))
    assert torch.allclose(pose[:, 3:], torch.tensor([[0.7071068, 0.0, 0.0, 0.7071068]]), atol=1e-6)


def test_wall_aabb_swaps_length_into_y():
    assert g.wall_corridor_aabb(4.0) == (1.0, 4.0)
