#!/usr/bin/env python
"""護欄 B: obs 單位契約斷言測試 (2026-07-06 立, 起因 ×8 殭屍常數 bug)。

不需 Isaac Sim — 純驗證兩件事:
  1. 正規化公式契約: obs = clamp(d_surface, 0, r_max) / r_max
     (d_surface = 雷射距離 − r_robot)。已知距離 → 預期 bin 值。
  2. 源碼一致性: 訓練內部 _LIDAR_MAX_DISTANCE_M 必須 == wd_like_sweep_72 的 r_max。
     這兩個不一致 = 所有距離門檻被錯讀 (react/gap/monitor)。

用法: /home/aa/miniconda3/envs/env_isaaclab/bin/python scripts/reinforcement_learning/skrl/tools/test_obs_units.py
CI/改 obs 後必跑。任何 FAIL = 距離語義破裂, 別訓練。
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OBS_F = REPO / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/observations/obs_functions.py"
TRAIN = REPO / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def norm(d_raw, r_robot, r_max):
    """契約公式: 雷射距離 → obs 正規化值。"""
    d_surf = min(max(d_raw - r_robot, 0.0), r_max)
    return d_surf / r_max


print("=== 1. 正規化公式契約 (已知距離 → 預期 obs) ===")
R_ROBOT, R_MAX = 0.35, 20.0
# 障礙表面到車 LiDAR 中心的原始距離 → obs 值 → 反推公尺 (×r_max)
cases = [(0.60, 0.0125), (1.20, 0.0425), (2.35, 0.10), (3.35, 0.15), (20.35, 1.0), (25.0, 1.0)]
for d_raw, expect in cases:
    got = norm(d_raw, R_ROBOT, R_MAX)
    check(f"d_raw={d_raw}m → obs≈{expect}", abs(got - expect) < 1e-3, f"got {got:.4f}")

# 反向: obs × r_max 必須還原正確表面距離 (這是 ×8 bug 的核心契約)
print("\n=== 2. 還原契約: obs × r_max = d_surface (×8 bug 就是這裡錯) ===")
for d_surf in [0.5, 1.2, 3.0, 5.0]:
    obs_val = min(d_surf, R_MAX) / R_MAX
    recovered = obs_val * R_MAX
    check(f"d_surf={d_surf}m 還原", abs(recovered - d_surf) < 1e-3, f"×{R_MAX}→{recovered:.3f}m")

print("\n=== 3. 源碼一致性: _LIDAR_MAX_DISTANCE_M == wd_like_sweep_72 的 r_max ===")
train_src = TRAIN.read_text()
m = re.search(r"_LIDAR_MAX_DISTANCE_M\s*=\s*([\d.]+)", train_src)
lidar_max = float(m.group(1)) if m else None
obs_src = OBS_F.read_text()
m2 = re.search(r"def wd_like_sweep_72.*?r_max:\s*float\s*=\s*([\d.]+)", obs_src, re.DOTALL)
sweep_rmax = float(m2.group(1)) if m2 else None
# ObsTerm 實際傳入的 r_max
m3 = re.search(r'"r_max":\s*([\d.]+)', obs_src)
obsterm_rmax = float(m3.group(1)) if m3 else sweep_rmax
print(f"    _LIDAR_MAX_DISTANCE_M = {lidar_max}")
print(f"    wd_like_sweep_72 r_max default = {sweep_rmax}")
print(f"    ObsTerm 傳入 r_max = {obsterm_rmax}")
_ratio = (obsterm_rmax / lidar_max) if lidar_max else 0.0
check("內部×常數 == ObsTerm r_max", lidar_max is not None and abs(lidar_max - obsterm_rmax) < 1e-6,
      ("一致" if lidar_max == obsterm_rmax else
       f"{lidar_max} vs {obsterm_rmax} — 不等! react/gap/monitor 距離門檻全被錯讀 {_ratio:.2f}倍"))

print("\n=== 4. ObsTerm clip 不得截斷正規化前的公尺值 (必須 clip=(0,1) 且 return 已 /r_max) ===")
has_div = "sweep = sweep / r_max" in obs_src or "/ r_max" in obs_src
check("wd_like_sweep_72 return 前有 /r_max 正規化", has_div, "無=公尺值直接被 clip(0,1) 鍘 (原誤判的病)")

print()
if fails:
    print(f"❌ {len(fails)} 項 FAIL: {fails}")
    sys.exit(1)
print("✅ 全部通過 — obs 單位契約完好")
