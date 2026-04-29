"""pedestrian_scene_setup.py — 在既有 USD 場景加入可偵測行人

Isaac Sim 5.1.0 相容。
使用 isaacsim.replicator.agent (IRA) + omni.anim.people 產生行人，
為每個行人附加 Capsule Collider，讓 PhysX raycast LiDAR 可命中。

用法:
    ./isaaclab.sh -p scripts/pedestrian/pedestrian_scene_setup.py \
        --scene /path/to/your_scene.usd \
        --num_pedestrians 3 \
        --headless

需要啟用的 Extensions:
    isaacsim.replicator.agent
    omni.anim.people
    omni.anim.navigation.core
    omni.physx
"""

from __future__ import annotations

import argparse
import time
import math
from pathlib import Path

# ── 1. 啟動 SimulationApp（必須在所有 omni imports 之前）──────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=str, default="",
                    help="既有 USD 場景路徑（空白 = 建立新空場景）")
parser.add_argument("--num_pedestrians", type=int, default=3)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--debug_vis", action="store_true", default=True,
                    help="顯示 capsule collider 的 debug 視覺化")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp
app = SimulationApp({
    "headless": args.headless,
})

# ── 2. 啟用必要 Extensions ────────────────────────────────────────────────────
import omni.ext
_ext_mgr = omni.ext.get_extension_manager()

REQUIRED_EXTENSIONS = [
    "isaacsim.replicator.agent.core",  # IRA core (5.1.0 正確名稱)
    "isaacsim.replicator.agent.ui",    # IRA UI panel
    "omni.anim.navigation.core",       # NavMesh
    "omni.physx",                      # PhysX
    # 注意: omni.anim.people 在 5.1.0 已整合進 IRA，不需單獨啟用
]

for ext in REQUIRED_EXTENSIONS:
    if not _ext_mgr.is_extension_enabled(ext):
        ok = _ext_mgr.set_extension_enabled_immediate(ext, True)
        status = "OK" if ok else "FAILED"
        print(f"[Pedestrian] Extension {ext}: {status}")
    else:
        print(f"[Pedestrian] Extension {ext}: already enabled")

# ── 3. 基本 imports（Extension 啟用後才能 import）────────────────────────────
import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf, Sdf
import omni.kit.app
import carb

# ── 4. 載入 / 建立場景 ────────────────────────────────────────────────────────
stage = omni.usd.get_context().get_stage()

if args.scene and Path(args.scene).exists():
    print(f"[Pedestrian] 載入既有場景: {args.scene}")
    omni.usd.get_context().open_stage(args.scene)
    # 等待場景載入
    for _ in range(50):
        app.update()
    stage = omni.usd.get_context().get_stage()
    print(f"[Pedestrian] 場景載入完成，root prim: {stage.GetDefaultPrim()}")
else:
    print("[Pedestrian] 使用當前場景（未指定 --scene）")
    # 確保有 default prim
    if not stage.GetDefaultPrim().IsValid():
        world_prim = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world_prim.GetPrim())

    # 添加基本地面（若場景沒有地面，行人 NavMesh 需要）
    if not stage.GetPrimAtPath("/World/Ground").IsValid():
        ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
        ground.CreatePointsAttr([
            Gf.Vec3f(-50, -50, 0), Gf.Vec3f(50, -50, 0),
            Gf.Vec3f(50, 50, 0),   Gf.Vec3f(-50, 50, 0),
        ])
        ground.CreateFaceVertexCountsAttr([4])
        ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        ground.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
        UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(ground.GetPrim())
        ground_rb = UsdPhysics.RigidBodyAPI(ground.GetPrim())
        ground_rb.CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxCollisionAPI.Apply(ground.GetPrim())

app.update()
stage = omni.usd.get_context().get_stage()


# ── 5. 建立 Physics Scene（若場景無 PhysicsScene）────────────────────────────
PHYSICS_SCENE_PATH = "/World/PhysicsScene"
if not stage.GetPrimAtPath(PHYSICS_SCENE_PATH).IsValid():
    scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    print(f"[Pedestrian] 建立 PhysicsScene: {PHYSICS_SCENE_PATH}")
app.update()


# ── 6. 建立 NavMesh ───────────────────────────────────────────────────────────
def build_navmesh() -> bool:
    """Bake NavMesh for the current scene. Returns True on success."""
    try:
        import omni.anim.navigation.core as nav_core
        iface = nav_core.acquire_interface()
        print("[Pedestrian] 開始 bake NavMesh...")
        iface.start_navmesh_baking()

        # 等待 bake 完成（最多 30 秒）
        for i in range(300):
            app.update()
            if iface.is_navmesh_baking_done():
                print(f"[Pedestrian] NavMesh bake 完成 (steps={i+1})")
                return True
            time.sleep(0.1)
        print("[Pedestrian] WARNING: NavMesh bake 逾時，繼續使用直線移動")
        return False
    except Exception as e:
        print(f"[Pedestrian] WARNING: NavMesh bake 失敗: {e}")
        print("[Pedestrian] 行人將使用直線 waypoint 移動（不依賴 NavMesh）")
        return False

navmesh_ok = build_navmesh()


# ── 7. 行人 spawn 設定 ────────────────────────────────────────────────────────
# 行人 waypoints — 在場景中巡邏的位置列表（可自訂）
PEDESTRIAN_WAYPOINTS = [
    # (spawn_pos, patrol_waypoints)
    ((2.0, 2.0, 0.0),  [(2.0, 2.0), (-2.0, 2.0), (-2.0, -2.0), (2.0, -2.0)]),
    ((5.0, 0.0, 0.0),  [(5.0, 0.0), (3.0, 3.0), (0.0, 5.0), (-3.0, 3.0)]),
    ((-4.0, 3.0, 0.0), [(-4.0, 3.0), (-4.0, -3.0), (4.0, -3.0), (4.0, 3.0)]),
]

# IRA 標準角色 USD（Isaac Sim Nucleus 提供的 biped 角色）
# 若 Nucleus 不可用，可改用本地的簡單 USD（capsule 代替人形）
CHARACTER_USD_OPTIONS = [
    # Isaac Sim 5.x Nucleus 路徑
    "omniverse://localhost/NVIDIA/Assets/Characters/Biped_Setup/biped_demo/biped_demo.usd",
    # Isaac Sim 本地 fallback（若沒有角色 USD，使用 capsule 代替）
    None,
]


def _find_character_usd() -> str | None:
    """嘗試找到可用的角色 USD。"""
    import omni.client
    for usd_path in CHARACTER_USD_OPTIONS:
        if usd_path is None:
            return None
        result, _ = omni.client.stat(usd_path)
        if result == omni.client.Result.OK:
            return usd_path
    return None


CHARACTER_USD = _find_character_usd()
if CHARACTER_USD:
    print(f"[Pedestrian] 使用角色 USD: {CHARACTER_USD}")
else:
    print("[Pedestrian] 找不到角色 USD，使用 capsule 代替人形")


# ── 8. 生成行人 Prim + Capsule Collider ──────────────────────────────────────
PEDESTRIAN_RADIUS = 0.30   # 膠囊半徑 30cm（比人形保守）
PEDESTRIAN_HEIGHT = 1.20   # 膠囊中段高度 1.2m
PEDESTRIAN_TOTAL_HEIGHT = PEDESTRIAN_HEIGHT + 2 * PEDESTRIAN_RADIUS  # ≈ 1.8m

spawned_pedestrian_paths: list[str] = []


def spawn_pedestrian_capsule(
    ped_index: int,
    position: tuple[float, float, float],
) -> str:
    """
    在 position 生成一個以 Capsule 為可見體 + Collider 的行人替代物。
    當沒有角色 USD 可用時使用此函數。
    返回根 prim 的 USD 路徑。
    """
    root_path = f"/World/Pedestrian_{ped_index}"

    # -- 根 Xform（用於位置控制）--
    xform = UsdGeom.Xform.Define(stage, root_path)
    xform_op = xform.AddTranslateOp()
    xform_op.Set(Gf.Vec3d(position[0], position[1], position[2]))

    # -- 視覺 capsule（作為臨時人形視覺）--
    vis_path = f"{root_path}/visual"
    vis_capsule = UsdGeom.Capsule.Define(stage, vis_path)
    vis_capsule.CreateRadiusAttr(PEDESTRIAN_RADIUS)
    vis_capsule.CreateHeightAttr(PEDESTRIAN_HEIGHT)
    vis_capsule.CreateAxisAttr("Z")
    # 讓 capsule 底部在 z=0（預設中心在 z=0，offset 向上）
    vis_xform = vis_capsule.AddTranslateOp()
    vis_xform.Set(Gf.Vec3d(0.0, 0.0, PEDESTRIAN_TOTAL_HEIGHT / 2.0))
    # 添加顏色（橘色 = 行人）
    vis_prim = vis_capsule.GetPrim()
    display_color = vis_capsule.CreateDisplayColorAttr()
    display_color.Set([Gf.Vec3f(1.0, 0.5, 0.1)])

    # -- 碰撞 capsule（PhysX 可命中）--
    _apply_collision_capsule(root_path)

    return root_path


def spawn_pedestrian_ira(
    ped_index: int,
    position: tuple[float, float, float],
    character_usd: str,
) -> str | None:
    """
    使用 IRA omni.anim.people 生成角色行人。
    返回角色根 prim 路徑，或 None（失敗時）。
    """
    try:
        from omni.anim.people.scripts.character_spawner import CharacterSpawner

        root_path = f"/World/Pedestrian_{ped_index}"
        print(f"[Pedestrian] 生成 IRA 角色 at {root_path}")

        spawner = CharacterSpawner()
        spawner.spawn(
            character_usd,
            root_path,
            position=Gf.Vec3d(position[0], position[1], position[2]),
        )
        app.update()

        if not stage.GetPrimAtPath(root_path).IsValid():
            print(f"[Pedestrian] IRA spawn 失敗: {root_path} 不存在")
            return None

        return root_path
    except Exception as e:
        print(f"[Pedestrian] IRA spawn 失敗: {e}")
        return None


def _apply_collision_capsule(character_root_path: str) -> str:
    """
    在 character_root_path 下建立 /collision_capsule 子 prim。
    套用 PhysX CollisionAPI + kinematic RigidBodyAPI。
    這個 capsule 會繼承父 prim 的 world transform，隨行人移動。
    返回 capsule prim 路徑。
    """
    capsule_path = f"{character_root_path}/collision_capsule"

    if stage.GetPrimAtPath(capsule_path).IsValid():
        print(f"[Pedestrian] Capsule 已存在: {capsule_path}")
        return capsule_path

    capsule = UsdGeom.Capsule.Define(stage, capsule_path)
    capsule.CreateRadiusAttr(PEDESTRIAN_RADIUS)
    capsule.CreateHeightAttr(PEDESTRIAN_HEIGHT)
    capsule.CreateAxisAttr("Z")

    # 高度偏移：讓底部貼地
    t_op = capsule.AddTranslateOp()
    t_op.Set(Gf.Vec3d(0.0, 0.0, PEDESTRIAN_TOTAL_HEIGHT / 2.0))

    # 讓 capsule 不可見（純碰撞用）
    UsdGeom.Imageable(capsule.GetPrim()).MakeInvisible()

    prim = capsule.GetPrim()

    # PhysX Collision API（raycast 命中需要這個）
    UsdPhysics.CollisionAPI.Apply(prim)
    PhysxSchema.PhysxCollisionAPI.Apply(prim)

    # Kinematic RigidBody（跟隨父 prim，不被物理推動）
    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rb_api.CreateKinematicEnabledAttr(True)

    # 碰撞 approximation = capsule（效能最佳）
    mesh_coll_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
    # Capsule 不需要 mesh approx，但套用後確保 PhysX 使用 capsule shape

    print(f"[Pedestrian] 建立碰撞 Capsule: {capsule_path} "
          f"(r={PEDESTRIAN_RADIUS}, h={PEDESTRIAN_HEIGHT})")
    return capsule_path


# 生成所有行人
num_to_spawn = min(args.num_pedestrians, len(PEDESTRIAN_WAYPOINTS))
for i in range(num_to_spawn):
    spawn_pos, waypoints = PEDESTRIAN_WAYPOINTS[i]

    if CHARACTER_USD:
        ped_path = spawn_pedestrian_ira(i, spawn_pos, CHARACTER_USD)
        if ped_path is None:
            print(f"[Pedestrian] IRA 失敗，改用 capsule 替代")
            ped_path = spawn_pedestrian_capsule(i, spawn_pos)
        else:
            # IRA 角色成功生成，補上 collision capsule
            _apply_collision_capsule(ped_path)
    else:
        ped_path = spawn_pedestrian_capsule(i, spawn_pos)

    spawned_pedestrian_paths.append(ped_path)
    print(f"[Pedestrian] 行人 {i}: {ped_path} @ {spawn_pos}")

app.update()
print(f"[Pedestrian] 共生成 {len(spawned_pedestrian_paths)} 個行人")


# ── 9. 設定 GoTo 行為（IRA 行人）─────────────────────────────────────────────
def setup_goto_behaviors(pedestrian_paths: list[str]) -> None:
    """為 IRA 角色設定 GoTo 巡邏行為。"""
    try:
        from omni.anim.people.scripts.pedestrian_manager import PedestrianManager

        for i, ped_path in enumerate(pedestrian_paths):
            _, waypoints = PEDESTRIAN_WAYPOINTS[i]
            # GoTo command 格式: {"command": "GoTo", "args": {"target": [x, y, z]}}
            commands = []
            for wp in waypoints:
                commands.append({
                    "command": "GoTo",
                    "args": {"target": [wp[0], wp[1], 0.0]},
                })

            mgr = PedestrianManager()
            mgr.set_agent_commands(ped_path, commands)
            print(f"[Pedestrian] 設定 {len(commands)} 個 GoTo waypoints for {ped_path}")

    except ImportError as e:
        print(f"[Pedestrian] GoTo API 不可用 ({e})，改用腳本化移動")
        _setup_scripted_movement(pedestrian_paths)
    except Exception as e:
        print(f"[Pedestrian] GoTo 設定失敗 ({e})，改用腳本化移動")
        _setup_scripted_movement(pedestrian_paths)


# 備援：腳本化直線移動（不需要 IRA / NavMesh）
class ScriptedPedestrian:
    def __init__(self, root_path: str, waypoints: list[tuple[float, float]],
                 speed: float = 0.8):
        self.root_path = root_path
        self.waypoints = waypoints
        self.speed = speed
        self.current_wp_idx = 0
        self.pos = list(waypoints[0]) + [0.0]

    def step(self, dt: float) -> None:
        """每個模擬步更新位置。"""
        prim = stage.GetPrimAtPath(self.root_path)
        if not prim.IsValid():
            return

        target = self.waypoints[self.current_wp_idx]
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dist = math.hypot(dx, dy)

        if dist < 0.1:
            # 到達 waypoint，前進到下一個
            self.current_wp_idx = (self.current_wp_idx + 1) % len(self.waypoints)
            return

        move = min(self.speed * dt, dist)
        self.pos[0] += (dx / dist) * move
        self.pos[1] += (dy / dist) * move

        # 更新 USD translate
        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(self.pos[0], self.pos[1], self.pos[2]))
                break


_scripted_peds: list[ScriptedPedestrian] = []


def _setup_scripted_movement(pedestrian_paths: list[str]) -> None:
    global _scripted_peds
    for i, ped_path in enumerate(pedestrian_paths):
        _, waypoints = PEDESTRIAN_WAYPOINTS[i]
        _scripted_peds.append(ScriptedPedestrian(ped_path, waypoints))
    print(f"[Pedestrian] 腳本化移動：{len(_scripted_peds)} 個行人")


if CHARACTER_USD and navmesh_ok:
    setup_goto_behaviors(spawned_pedestrian_paths)
else:
    _setup_scripted_movement(spawned_pedestrian_paths)


# ── 10. Debug Visualization（顯示 Capsule 輪廓）──────────────────────────────
if args.debug_vis:
    try:
        # 讓碰撞體可見（debug 用）
        for ped_path in spawned_pedestrian_paths:
            cap_path = f"{ped_path}/collision_capsule"
            cap_prim = stage.GetPrimAtPath(cap_path)
            if cap_prim.IsValid():
                UsdGeom.Imageable(cap_prim).MakeVisible()
                # 設為半透明紅色以區分
                capsule = UsdGeom.Capsule(cap_prim)
                capsule.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.2, 0.2)])
                capsule.CreateDisplayOpacityAttr([0.5])
        print("[Pedestrian] Debug visualization 啟用 — 紅色半透明 capsule = 碰撞體")
    except Exception as e:
        print(f"[Pedestrian] Debug vis 設定失敗: {e}")


# ── 11. 驗證碰撞體設定 ─────────────────────────────────────────────────────────
def verify_collision_setup() -> dict[str, bool]:
    """驗證每個行人的碰撞體是否正確設定。"""
    results = {}
    for ped_path in spawned_pedestrian_paths:
        cap_path = f"{ped_path}/collision_capsule"
        prim = stage.GetPrimAtPath(cap_path)

        if not prim.IsValid():
            results[ped_path] = False
            print(f"[VERIFY] FAIL: {cap_path} 不存在")
            continue

        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)
        has_rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        is_kinematic = False
        if has_rigid:
            rb = UsdPhysics.RigidBodyAPI(prim)
            kin_attr = rb.GetKinematicEnabledAttr()
            is_kinematic = kin_attr.Get() if kin_attr else False

        ok = has_collision and has_rigid and is_kinematic
        results[ped_path] = ok
        status = "OK" if ok else "FAIL"
        print(f"[VERIFY] {status}: {cap_path}")
        print(f"         CollisionAPI={has_collision}, "
              f"RigidBodyAPI={has_rigid}, "
              f"kinematic={is_kinematic}")

    return results

verify_results = verify_collision_setup()
print(f"\n[VERIFY] 碰撞體設定: "
      f"{sum(verify_results.values())}/{len(verify_results)} 通過")

# 輸出如何讓 LiDAR 偵測到行人的提示
print("\n" + "="*60)
print("[INFO] LiDAR 偵測行人 — 設定重點:")
print("  1. Pedestrian_*/collision_capsule 已有 CollisionAPI ✓")
print("  2. PhysX raycast 自動命中所有 CollisionAPI prim")
print("  3. 如使用 MultiMeshRayCaster，在 mesh_prim_paths 加入:")
print('     MultiMeshRayCasterCfg.RaycastTargetCfg(')
print('         prim_expr="/World/Pedestrian_.*",')
print('         track_mesh_transforms=True,')
print('     )')
print("  4. 如使用 PhysX BatchRaycast，無需額外設定，直接命中")
print("="*60 + "\n")

app.update()

# ── 12. 主模擬迴圈（含腳本化行人移動）────────────────────────────────────────
print("[Pedestrian] 開始模擬迴圈（Ctrl+C 結束）")
PHYSICS_DT = 1.0 / 60.0  # 60Hz physics
try:
    while app.is_running():
        # 腳本化行人移動（若無 IRA GoTo）
        for ped in _scripted_peds:
            ped.step(PHYSICS_DT)
        app.update()
except KeyboardInterrupt:
    print("[Pedestrian] 使用者中斷")

app.close()
