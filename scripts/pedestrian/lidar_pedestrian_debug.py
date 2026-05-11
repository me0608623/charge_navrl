"""lidar_pedestrian_debug.py — 驗證 PhysX raycast LiDAR 能命中行人 Collider

每個模擬 step 列印:
    - 最近的行人命中距離
    - 命中的 prim 路徑
    - 未命中時的可能原因

用法 (在 pedestrian_scene_setup.py 之後執行，或整合到既有訓練腳本):
    ./isaaclab.sh -p scripts/pedestrian/lidar_pedestrian_debug.py \
        --robot_prim /World/Robot/base_link \
        --pedestrian_root /World

本腳本假設 pedestrian_scene_setup.py 已執行，場景中存在 Pedestrian_* prims。
"""

from __future__ import annotations

import argparse
import math
from typing import NamedTuple

# ── 啟動 SimulationApp ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--robot_prim", type=str, default="/World/Robot/base_link",
                    help="LiDAR 掛載點 prim 路徑")
parser.add_argument("--pedestrian_root", type=str, default="/World",
                    help="行人 prims 的父路徑（用於搜尋 Pedestrian_*）")
parser.add_argument("--lidar_height", type=float, default=1.6,
                    help="LiDAR 相對於 robot_prim 的 Z 高度 (m)")
parser.add_argument("--max_range", type=float, default=20.0,
                    help="LiDAR 最大偵測距離 (m)")
parser.add_argument("--num_rays", type=int, default=36,
                    help="水平掃描 ray 數量（debug 用，減少輸出）")
parser.add_argument("--steps", type=int, default=100,
                    help="驗證步數（-1 = 無限）")
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless})

import omni.usd
import carb
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf


# ── PhysX BatchRaycast interface ───────────────────────────────────────────────
try:
    import omni.physx as physx_ext
    from omni.physx import get_physx_scene_query_interface
    physx_sq = get_physx_scene_query_interface()
    PHYSX_AVAILABLE = True
    print("[Debug] PhysX SceneQuery interface 取得成功")
except Exception as e:
    print(f"[Debug] PhysX SceneQuery 不可用: {e}")
    PHYSX_AVAILABLE = False


class RayHit(NamedTuple):
    distance: float
    prim_path: str
    position: tuple[float, float, float]
    normal: tuple[float, float, float]


def cast_debug_rays(
    origin: Gf.Vec3d,
    num_rays: int = 36,
    height: float = 0.0,
    max_range: float = 20.0,
) -> list[RayHit]:
    """
    從 origin 發射水平掃描 rays，回傳命中列表。
    使用 PhysX SceneQuery（與 RayCaster sensor 相同底層）。
    """
    hits: list[RayHit] = []
    if not PHYSX_AVAILABLE:
        return hits

    o = carb.Float3(float(origin[0]), float(origin[1]), float(origin[2] + height))
    angle_step = 2 * math.pi / num_rays

    for i in range(num_rays):
        angle = i * angle_step
        direction = carb.Float3(math.cos(angle), math.sin(angle), 0.0)

        # PhysX raycast — 回傳最近命中
        hit = physx_sq.raycast_closest(o, direction, max_range)

        if hit and hit.collision:
            # hit.collision 是命中的 physics body path
            # 取得實際 USD prim path（可能是 body or shape）
            prim_path = str(hit.rigidBody) if hasattr(hit, 'rigidBody') else "unknown"
            hits.append(RayHit(
                distance=float(hit.distance),
                prim_path=prim_path,
                position=(float(hit.position.x), float(hit.position.y),
                          float(hit.position.z)),
                normal=(float(hit.normal.x), float(hit.normal.y),
                        float(hit.normal.z)),
            ))

    return hits


def find_pedestrian_paths(root: str) -> list[str]:
    """在場景中搜尋 Pedestrian_* prims。"""
    stage = omni.usd.get_context().get_stage()
    peds = []
    root_prim = stage.GetPrimAtPath(root)
    if not root_prim.IsValid():
        return peds
    for prim in root_prim.GetChildren():
        if prim.GetName().startswith("Pedestrian_"):
            peds.append(str(prim.GetPath()))
    return peds


def get_prim_world_position(prim_path: str) -> Gf.Vec3d | None:
    """取得 prim 的世界座標。"""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    xform = UsdGeom.Xformable(prim)
    xform_cache = UsdGeom.XformCache()
    mat = xform_cache.GetLocalToWorldTransform(prim)
    return Gf.Vec3d(mat[3][0], mat[3][1], mat[3][2])


def diagnose_pedestrian_collider(ped_path: str) -> list[str]:
    """
    診斷行人碰撞體設定，回傳可能原因列表。
    若設定正確，回傳空列表。
    """
    stage = omni.usd.get_context().get_stage()
    issues: list[str] = []

    cap_path = f"{ped_path}/collision_capsule"
    cap_prim = stage.GetPrimAtPath(cap_path)

    if not cap_prim.IsValid():
        issues.append(f"找不到 collision_capsule: {cap_path}")
        issues.append("=> 請先執行 pedestrian_scene_setup.py")
        return issues

    if not cap_prim.HasAPI(UsdPhysics.CollisionAPI):
        issues.append(f"缺少 CollisionAPI: {cap_path}")
        issues.append("=> UsdPhysics.CollisionAPI.Apply(cap_prim) 未執行")

    if not cap_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        issues.append(f"缺少 RigidBodyAPI: {cap_path}")
        issues.append("=> 需要 kinematic RigidBodyAPI 才能讓 PhysX 追蹤位置")

    # 確認碰撞體不在 z=-10（隱藏位置）
    pos = get_prim_world_position(cap_path)
    if pos and pos[2] < -5.0:
        issues.append(f"碰撞體在隱藏位置 z={pos[2]:.1f} — 父 prim 未正確移動")
        issues.append("=> 確認父 prim translate 已更新")

    # 確認 imageable visibility 不影響碰撞（即使 invisible，碰撞應仍有效）
    imageable = UsdGeom.Imageable(cap_prim)
    vis = imageable.ComputeVisibility(Usd.TimeCode.Default())
    if vis == UsdGeom.Tokens.invisible:
        # 不影響 PhysX 碰撞，但記錄供參考
        issues.append(f"注意: capsule 為 invisible（影響視覺，不影響碰撞）")

    return issues


def check_collision_group_filter(ped_path: str) -> None:
    """
    PhysX Collision Group / Filter 檢查。
    若行人 collider 在不同 collision group，raycast 可能跳過。
    """
    stage = omni.usd.get_context().get_stage()
    cap_path = f"{ped_path}/collision_capsule"
    prim = stage.GetPrimAtPath(cap_path)
    if not prim.IsValid():
        return

    # 取得 collision group（若有設定）
    coll_group = None
    for rel in prim.GetRelationships():
        if "collisionGroup" in rel.GetName():
            targets = rel.GetTargets()
            if targets:
                coll_group = str(targets[0])
    if coll_group:
        print(f"  [Filter] {cap_path} 屬於 collision group: {coll_group}")
        print(f"  [Filter] 確認 raycast collision mask 包含此 group")
    else:
        print(f"  [Filter] {cap_path} 無特定 collision group — PhysX 預設命中全部")


# ── 主驗證迴圈 ────────────────────────────────────────────────────────────────
app.update()
stage = omni.usd.get_context().get_stage()

# 取得機器人位置
robot_pos = get_prim_world_position(args.robot_prim)
if robot_pos is None:
    print(f"[Debug] WARNING: 找不到 robot_prim: {args.robot_prim}")
    print("[Debug] 使用原點 (0, 0, 0) 作為 LiDAR 起點")
    robot_pos = Gf.Vec3d(0.0, 0.0, 0.0)
else:
    print(f"[Debug] Robot 位置: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})")

# 尋找場景中的行人
ped_paths = find_pedestrian_paths(args.pedestrian_root)
print(f"[Debug] 找到 {len(ped_paths)} 個行人: {ped_paths}")

# 診斷碰撞體
for ped_path in ped_paths:
    issues = diagnose_pedestrian_collider(ped_path)
    if issues:
        print(f"\n[Diagnose] {ped_path} 碰撞體問題:")
        for issue in issues:
            print(f"  {issue}")
        check_collision_group_filter(ped_path)
    else:
        print(f"[Diagnose] {ped_path} 碰撞體設定正確 ✓")

if not ped_paths:
    print("\n[FAIL] 場景中沒有 Pedestrian_* prims！")
    print("請先執行: ./isaaclab.sh -p scripts/pedestrian/pedestrian_scene_setup.py")
    app.close()
    raise SystemExit(1)

print(f"\n[Debug] 開始 {args.steps} 步驗證迴圈...")
print("-" * 60)

step = 0
any_hit_ped = False

while app.is_running() and (args.steps < 0 or step < args.steps):
    app.update()
    step += 1

    # 更新機器人位置
    pos = get_prim_world_position(args.robot_prim)
    if pos:
        robot_pos = pos

    # 發射 debug rays
    hits = cast_debug_rays(
        origin=robot_pos,
        num_rays=args.num_rays,
        height=args.lidar_height,
        max_range=args.max_range,
    )

    # 分析命中結果
    ped_hits = []
    other_hits = []
    for hit in hits:
        is_ped = any(p in hit.prim_path for p in ped_paths) or \
                 "Pedestrian" in hit.prim_path or \
                 "collision_capsule" in hit.prim_path
        if is_ped:
            ped_hits.append(hit)
            any_hit_ped = True
        else:
            other_hits.append(hit)

    # 每 10 步輸出一次摘要
    if step % 10 == 0 or len(ped_hits) > 0:
        print(f"\n[Step {step:4d}] Robot @ ({robot_pos[0]:.1f}, {robot_pos[1]:.1f})")
        print(f"  Total hits: {len(hits)} | Pedestrian hits: {len(ped_hits)} | Other: {len(other_hits)}")

        if ped_hits:
            ped_hits_sorted = sorted(ped_hits, key=lambda h: h.distance)
            closest = ped_hits_sorted[0]
            print(f"  ✓ 最近行人命中:")
            print(f"    distance={closest.distance:.3f}m")
            print(f"    prim={closest.prim_path}")
            print(f"    position=({closest.position[0]:.2f}, "
                  f"{closest.position[1]:.2f}, "
                  f"{closest.position[2]:.2f})")
        else:
            print(f"  ✗ 沒有命中行人 — 可能原因:")
            # 計算到最近行人的距離
            for ped_path in ped_paths:
                ped_pos = get_prim_world_position(ped_path)
                if ped_pos:
                    dx = ped_pos[0] - robot_pos[0]
                    dy = ped_pos[1] - robot_pos[1]
                    dist = math.hypot(dx, dy)
                    if dist > args.max_range:
                        print(f"    行人 {ped_path} 距離 {dist:.1f}m > max_range {args.max_range}m")
                    elif dist < 0.1:
                        print(f"    行人 {ped_path} 距離 {dist:.3f}m — 可能在機器人正下方")
                    else:
                        print(f"    行人 {ped_path} 距離 {dist:.1f}m（範圍內但未命中）")
                        print(f"    => 可能原因: collision_capsule 無 CollisionAPI / "
                              f"PhysX collision group filter")

        if other_hits:
            # 最近非行人命中
            closest_other = min(other_hits, key=lambda h: h.distance)
            print(f"  最近其他命中: {closest_other.prim_path[:50]} d={closest_other.distance:.2f}m")

print("\n" + "=" * 60)
print(f"[Result] 驗證完成 ({step} steps)")
print(f"[Result] 行人 LiDAR 命中: {'✓ YES' if any_hit_ped else '✗ NO'}")

if not any_hit_ped:
    print("\n[Result] 未命中，請確認以下條件:")
    print("  1. pedestrian_scene_setup.py 已執行，場景存在 Pedestrian_*/collision_capsule")
    print("  2. collision_capsule 有 UsdPhysics.CollisionAPI")
    print("  3. collision_capsule 有 UsdPhysics.RigidBodyAPI (kinematic=True)")
    print("  4. 行人在 LiDAR 的射線範圍內（非遮蔽、非過遠）")
    print("  5. PhysX simulation 已啟動（app.update() 跑過物理步）")
    print("  6. 若使用 MultiMeshRayCaster，確認 mesh_prim_paths 包含:")
    print('     prim_expr="/World/Pedestrian_.*" 或 prim_expr="{ENV_REGEX_NS}/Pedestrian_.*"')
else:
    print("\n[Result] LiDAR 行人偵測正常 ✓")

app.close()
