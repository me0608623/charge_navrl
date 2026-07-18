"""swept_arc_audit.py — r_arc 訓練前的 offline counterfactual audit（用戶 07-14 定的試金石）。

目的：在花任何訓練成本前，證明「同一份 79D LiDAR 是否真能分辨安全方向」。
方法：載入實收的 SA4 72-beam LiDAR frames，對每個 frame 固定 LiDAR、掃描假想動作網格
      v ∈ {0.4,0.6,0.8} × ω ∈ {-1.0..+1.0}，用 swept_arc.arc_clearance 算 c_arc，
      看 Δarc = max_ω c_arc(v,ω) − c_arc(v, ω=0)（＝轉彎相對直走能多清出的空間）。

判定（用戶定）：
  - SA4 frames 的 Δarc 明顯 > 0（且左右不對稱有方向性）→ reward 有可學梯度 → 可訓練。
  - 若所有 ω 都被判會撞（c_arc 全 < 0、Δarc≈0）→ 問題在 arc 幾何/膨脹/觀測鑑別力，訓練只會浪費時間。

用法：
  python swept_arc_audit.py --npz /tmp/lidar_sa4.npz [--near_thresh 2.0] [--horizon 2.7]
  # 可同時比 stage3：--npz_ref /tmp/lidar_sa3.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swept_arc as sa  # noqa: E402


def load_frames(npz_path: str, body_radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Load sensor ranges; repair legacy dumps that stored policy clearances."""
    with np.load(npz_path) as d:
        lidar_m = d["lidar_m"].astype(np.float32)
        va = d["va"].astype(np.float32)
        semantics = str(d["lidar_semantics"].item()) if "lidar_semantics" in d else "policy_clearance_m"
        stored_radius = float(d["body_radius"].item()) if "body_radius" in d else body_radius

    if semantics == "policy_clearance_m":
        print(
            f"[audit] Legacy dump detected: converting policy clearance to sensor range "
            f"with body_radius={stored_radius:.3f}m"
        )
        lidar_m = lidar_m + stored_radius
    elif semantics != "sensor_range_m":
        raise ValueError(f"Unsupported lidar_semantics={semantics!r} in {npz_path}")
    return lidar_m, va


def audit(lidar_m: np.ndarray, args) -> dict:
    """回傳 audit 統計 dict。lidar_m: [N,72] sensor-to-hit ranges in meters。"""
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    lm = torch.from_numpy(lidar_m).to(dev)                       # [N,72]
    N = lm.shape[0]

    # 只看「有障礙靠近」的 frame（空曠場沒有方向可言）：任一 beam < near_thresh
    valid_beam = (lm > args.hole_thresh_m) & (lm < args.max_range)
    clearance = lm - args.body_radius
    nearest = torch.where(valid_beam, clearance, torch.full_like(lm, args.max_range)).amin(dim=1)  # [N]
    near_mask = nearest < args.near_thresh
    idx = near_mask.nonzero(as_tuple=False).reshape(-1)
    n_near = int(idx.numel())
    if n_near == 0:
        return {"n_frames": N, "n_near": 0}

    lm_near = lm[idx]                                            # [M,72]
    M = lm_near.shape[0]

    v_grid = [float(x) for x in args.v_grid.split(",")]
    w_grid = [float(x) for x in args.w_grid.split(",")]
    W = len(w_grid)
    w_tensor = torch.tensor(w_grid, device=dev)
    zero_wi = int(np.argmin([abs(w) for w in w_grid]))          # ω=0 的 index

    per_v = {}
    for v in v_grid:
        # tile: [M*W, 72]、[M*W]
        lm_rep = lm_near.repeat_interleave(W, dim=0)            # [M*W,72]
        v_rep = torch.full((M * W,), v, device=dev)
        w_rep = w_tensor.repeat(M)                              # [M*W]
        c = sa.arc_clearance(lm_rep, v_rep, w_rep, horizon=args.horizon,
                             body_radius=args.body_radius, max_range=args.max_range,
                             hole_thresh_m=args.hole_thresh_m, chunk=args.chunk)
        c = c.reshape(M, W)                                     # [M,W]
        c_straight = c[:, zero_wi]                              # [M]
        c_best = c.amax(dim=1)                                  # [M] 最佳 ω 的間距
        best_wi = c.argmax(dim=1)                              # [M]
        delta = c_best - c_straight                             # [M] Δarc
        # 方向性：最佳 ω 落在左(>0)/右(<0)/直(0)
        best_w = w_tensor[best_wi]
        per_v[v] = {
            "delta_mean": float(delta.mean()),
            "delta_median": float(delta.median()),
            "frac_directional": float((delta > args.dir_thresh).float().mean()),  # Δarc>門檻比例
            "frac_all_hit": float((c.amax(dim=1) < 0).float().mean()),  # 連最佳 ω 都撞(c<0)比例
            "frac_straight_hit": float((c_straight < 0).float().mean()),
            "best_turn_left_frac": float((best_w > 0.05).float().mean()),
            "best_turn_right_frac": float((best_w < -0.05).float().mean()),
            "best_straight_frac": float((best_w.abs() <= 0.05).float().mean()),
            "c_straight_mean": float(c_straight.mean()),
            "c_best_mean": float(c_best.mean()),
        }
    return {"n_frames": N, "n_near": n_near, "near_thresh": args.near_thresh,
            "v_grid": v_grid, "w_grid": w_grid, "per_v": per_v}


def print_report(tag: str, r: dict, dir_thresh: float):
    print(f"\n{'='*66}\n[{tag}] frames={r['n_frames']}  近障 frames={r.get('n_near',0)} "
          f"(<{r.get('near_thresh','?')}m)")
    if r.get("n_near", 0) == 0:
        print("  ⚠ 無近障 frame，無法判方向性（場景太空或門檻太嚴）")
        return
    print(f"{'v':>5} | {'Δarc中位':>8} {'Δarc均':>7} | {'方向性%':>7} {'全撞%':>6} {'直走撞%':>7} "
          f"| {'best:左/右/直%':>14}")
    print("-" * 66)
    for v, s in r["per_v"].items():
        print(f"{v:>5.1f} | {s['delta_median']:>+8.3f} {s['delta_mean']:>+7.3f} | "
              f"{s['frac_directional']*100:>6.1f}% {s['frac_all_hit']*100:>5.1f}% "
              f"{s['frac_straight_hit']*100:>6.1f}% | "
              f"{s['best_turn_left_frac']*100:>4.0f}/{s['best_turn_right_frac']*100:>3.0f}/"
              f"{s['best_straight_frac']*100:>3.0f}%")
    # 綜合裁決（取 v=0.6 或中間值）
    vs = list(r["per_v"].keys())
    vmid = vs[len(vs) // 2]
    s = r["per_v"][vmid]
    print(f"\n  裁決依據 v={vmid}: 方向性比例={s['frac_directional']*100:.1f}% "
          f"(Δarc>{dir_thresh}m), 全撞比例={s['frac_all_hit']*100:.1f}%")
    if s["frac_directional"] > 0.30 and s["frac_all_hit"] < 0.40:
        print("  ✅ PASS：多數近障 frame 有可清出空間的方向 → r_arc 有可學梯度，可訓練。")
    elif s["frac_all_hit"] > 0.60:
        print("  ❌ FAIL：多數 frame 連最佳 ω 都撞 → 幾何/膨脹/觀測鑑別力問題，先別訓練。")
    else:
        print("  ⚠ 邊界：方向性偏弱，建議調 horizon/c_safe/body_radius 再看，或小心訓練。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="SA4 LiDAR frames (CHARGE_LIDAR_DUMP 產)")
    p.add_argument("--npz_ref", default=None, help="對照(如 stage3) frames")
    p.add_argument("--near_thresh", type=float, default=2.0, help="只看最近障礙<此(m)的 frame")
    p.add_argument("--dir_thresh", type=float, default=0.2, help="Δarc>此(m)算有方向性")
    p.add_argument("--horizon", type=float, default=2.7)
    p.add_argument("--body_radius", type=float, default=0.35)
    p.add_argument("--max_range", type=float, default=20.0)
    p.add_argument("--hole_thresh_m", type=float, default=0.4)
    p.add_argument("--v_grid", default="0.4,0.6,0.8")
    p.add_argument("--w_grid", default="-1.0,-0.8,-0.6,-0.4,-0.2,0.0,0.2,0.4,0.6,0.8,1.0")
    p.add_argument("--chunk", type=int, default=20000)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    lm, _va = load_frames(args.npz, args.body_radius)
    print_report(f"SA4  {os.path.basename(args.npz)}", audit(lm, args), args.dir_thresh)
    if args.npz_ref:
        lm2, _ = load_frames(args.npz_ref, args.body_radius)
        print_report(f"REF  {os.path.basename(args.npz_ref)}", audit(lm2, args), args.dir_thresh)


if __name__ == "__main__":
    main()
