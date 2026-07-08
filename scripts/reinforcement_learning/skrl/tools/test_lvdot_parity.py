"""LV-DOT channel train↔deploy parity test.

驗證訓練端 torch 函數 `dynamic_obstacles_lvdot` 與部署端 numpy 參考實作
（=車端 adapter 該做的計算）在**相同合成情境**下輸出逐位相同。

這把「風險②：train/deploy 座標約定不一致」確定性打掉，不需要 live LV-DOT。
DR 全關（noise/dropout/false_neg=0）→ 結果決定性，兩端應 bit-parity。

跑法（IsaacLab side，不需啟 sim）:
    conda activate env_isaaclab
    python scripts/reinforcement_learning/skrl/tools/test_lvdot_parity.py
"""
from __future__ import annotations

import math
import sys
import types

import numpy as np
import torch

# ── 載入訓練端 torch 函數，但避開 Isaac 啟動 ──
# obs_functions.py 頂層只 import isaaclab.{envs,managers,utils.math}（僅型別提示/預設參數用）；
# dynamic_obstacles_lvdot 執行期完全不需要它們。故 stub 這些 module 後直接以檔案載入單檔，
# 不走套件 __init__ 鏈（那會 import env cfg → 觸發 SimulationApp 要求）。
import importlib.util  # noqa: E402


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _SceneEntityCfg:  # 預設參數 SceneEntityCfg("robot") 需可呼叫且有 .name
    def __init__(self, name="robot", **kw):
        self.name = name


_stub("isaaclab")
_stub("isaaclab.envs", ManagerBasedRLEnv=object)
_stub("isaaclab.managers", SceneEntityCfg=_SceneEntityCfg)
_stub("isaaclab.utils")
_stub("isaaclab.utils.math")  # dynamic_obstacles_lvdot 不用，其他函數才用

_OBS_PATH = (
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/observations/obs_functions.py"
)
_spec = importlib.util.spec_from_file_location("obs_functions_standalone", _OBS_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
dynamic_obstacles_lvdot = _mod.dynamic_obstacles_lvdot

DEVICE = "cpu"
MAX_DIST = 8.0
V_MAX = 1.5
TOP_K = 5
DYN_TH = 0.3  # 對齊 LV-DOT dynamic_velocity_threshold=0.3


# ══════════════════════════════════════════════════════════════════════════
# Mock env（最小化：只提供 dynamic_obstacles_lvdot 會存取的欄位）
# ══════════════════════════════════════════════════════════════════════════
class _Data:
    def __init__(self, pos_w, quat_w=None, vel_w=None):
        self.root_pos_w = pos_w
        self.root_quat_w = quat_w
        self.root_lin_vel_w = vel_w


class _Entity:
    def __init__(self, data):
        self.data = data


class _Scene:
    def __init__(self, entities, env_origins):
        self._e = entities
        self.env_origins = env_origins

    def __getitem__(self, k):
        return self._e[k]

    def keys(self):
        return self._e.keys()


def _yaw_to_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]  # (w,x,y,z)


def make_env(robot, obstacles, env_origin=(0.0, 0.0)):
    """robot=(x,y,yaw,vx,vy); obstacles=list of (x,y,vx,vy,r,z)."""
    rx, ry, ryaw, rvx, rvy = robot
    ents = {}
    ents["robot"] = _Entity(_Data(
        torch.tensor([[rx, ry, 0.0]], dtype=torch.float32),
        torch.tensor([_yaw_to_quat(ryaw)], dtype=torch.float32),
        torch.tensor([[rvx, rvy, 0.0]], dtype=torch.float32),
    ))
    n = len(obstacles)
    sizes = []
    dt = 0.2
    # finite-diff prev-pos：訓練函數用 (cur − prev)/dt 算速度，故設 prev = cur − vel·dt 以還原意圖速度。
    prev = torch.zeros(1, n, 2)
    for i, (ox, oy, ovx, ovy, r, oz) in enumerate(obstacles):
        ents[f"obstacle_{i}"] = _Entity(_Data(
            torch.tensor([[ox, oy, oz]], dtype=torch.float32),
            vel_w=torch.tensor([[ovx, ovy, 0.0]], dtype=torch.float32),
        ))
        prev[0, i, 0] = ox - ovx * dt
        prev[0, i, 1] = oy - ovy * dt
        sizes.append(r)
    env = types.SimpleNamespace()
    env.scene = _Scene(ents, torch.tensor([list(env_origin)], dtype=torch.float32))
    env.device = DEVICE
    env._obstacle_sizes = sizes
    env.num_envs = 1
    env.step_dt = dt
    env.common_step_counter = 1          # != 未設的 _lvdot_fd_step → 走 finite-diff 計算路徑
    env._lvdot_prev_obs_xy = prev        # [1, n, 2]，與 pos 同 shape (max_obstacles=n)
    return env


# ══════════════════════════════════════════════════════════════════════════
# 部署端 numpy 參考實作（= 車端 adapter 該做的計算，逐步鏡射 torch 函數）
# ══════════════════════════════════════════════════════════════════════════
def deploy_reference(robot, obstacles, top_k=TOP_K, max_distance=MAX_DIST, v_max=V_MAX):
    """robot=(x,y,yaw,vx,vy) odom frame; obstacles=list (x,y,vx,vy,r,z) odom frame,
    速度為絕對速度（LV-DOT 給的）。回傳 [top_k*6] numpy，與訓練端逐位對齊。"""
    rx, ry, ryaw, rvx, rvy = robot
    cos_y, sin_y = math.cos(ryaw), math.sin(ryaw)
    cand = []  # (dist, px_n, py_n, vx_n, vy_n, r)
    for (ox, oy, ovx, ovy, r, oz) in obstacles:
        visible = oz > 0.0
        abs_speed = math.hypot(ovx, ovy)
        dynamic = abs_speed > DYN_TH
        dx, dy = ox - rx, oy - ry
        dist = math.sqrt(dx * dx + dy * dy + 1e-8)
        valid = visible and dynamic and (dist <= max_distance)
        if not valid:
            continue
        # body frame
        px = cos_y * dx + sin_y * dy
        py = -sin_y * dx + cos_y * dy
        rel_vx, rel_vy = ovx - rvx, ovy - rvy
        vxb = cos_y * rel_vx + sin_y * rel_vy
        vyb = -sin_y * rel_vx + cos_y * rel_vy
        px_n = float(np.clip(px / max_distance, -1.0, 1.0))
        py_n = float(np.clip(py / max_distance, -1.0, 1.0))
        vx_n = float(np.clip(vxb / v_max, -2.0, 2.0))
        vy_n = float(np.clip(vyb / v_max, -2.0, 2.0))
        cand.append((dist, px_n, py_n, vx_n, vy_n, r))
    cand.sort(key=lambda t: t[0])  # 由近到遠
    out = np.zeros(top_k * 6, dtype=np.float32)
    for slot in range(min(top_k, len(cand))):
        _, px_n, py_n, vx_n, vy_n, r = cand[slot]
        out[slot * 6:slot * 6 + 6] = [px_n, py_n, vx_n, vy_n, r, 1.0]  # valid=1
    return out


# ══════════════════════════════════════════════════════════════════════════
# 測試情境
# ══════════════════════════════════════════════════════════════════════════
def run_case(name, robot, obstacles):
    env = make_env(robot, obstacles)
    train_out = dynamic_obstacles_lvdot(
        env, top_k=TOP_K, max_obstacles=len(obstacles), max_distance=MAX_DIST,
        v_max=V_MAX, wall_occlusion=False, dynamic_speed_threshold=DYN_TH, fov_deg=360.0,
    ).cpu().numpy().reshape(-1)
    deploy_out = deploy_reference(robot, obstacles)
    ok = np.allclose(train_out, deploy_out, atol=1e-5)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        diff = np.abs(train_out - deploy_out)
        bad = np.where(diff > 1e-5)[0]
        for j in bad[:12]:
            print(f"    idx {j}: train={train_out[j]:+.5f}  deploy={deploy_out[j]:+.5f}")
    return ok


def main():
    all_ok = True
    # Case 1: yaw≠0 + 機器人有速度 + 1靜態(排除)+2動態+1隱藏+1超距(排除)
    all_ok &= run_case(
        "yaw+vel, mixed static/dynamic/hidden/far",
        robot=(2.0, 3.0, 0.9, 0.4, 0.1),
        obstacles=[
            (4.0, 3.0, 0.0, 0.0, 0.30, 0.5),    # 靜態 → 排除
            (3.0, 4.0, -0.5, 0.3, 0.35, 0.5),   # 動態 近 → 收
            (2.5, 2.0, 0.2, -0.6, 0.28, 0.5),   # 動態 近 → 收
            (2.2, 3.2, 0.4, 0.4, 0.4, -10.0),   # 隱藏 z<0 → 排除
            (15.0, 15.0, 1.0, 1.0, 0.5, 0.5),   # 超距 → 排除
        ],
    )
    # Case 2: 正前方逼近（yaw=0），驗證 vx_body 符號（接近應為負）
    all_ok &= run_case(
        "head-on approach, yaw=0",
        robot=(0.0, 0.0, 0.0, 0.5, 0.0),
        obstacles=[(3.0, 0.0, -1.0, 0.0, 0.35, 0.5)],  # 正前方，朝車走
    )
    # Case 3: 全靜態 → channel 應全 0
    all_ok &= run_case(
        "all static → empty channel",
        robot=(1.0, 1.0, 0.3, 0.0, 0.0),
        obstacles=[(2.0, 1.0, 0.0, 0.0, 0.3, 0.5), (1.0, 2.0, 0.01, 0.0, 0.3, 0.5)],
    )
    # Case 4: 超過 K 個動態 → 只收最近 5
    all_ok &= run_case(
        "more than K dynamic → nearest 5",
        robot=(0.0, 0.0, 1.2, 0.2, -0.3),
        obstacles=[(d * 0.9, d * 0.4, -0.4, 0.2, 0.3, 0.5) for d in range(1, 8)],
    )
    print("\n" + ("✅ ALL PARITY CHECKS PASS" if all_ok else "❌ PARITY MISMATCH"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
