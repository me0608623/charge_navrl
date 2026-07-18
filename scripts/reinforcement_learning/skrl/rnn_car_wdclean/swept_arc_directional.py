"""swept_arc_directional.py — 方向性反應診斷（07-15 用戶定的正確裁決標準）。

★核心問題（不是「有沒有減速」，而是「有沒有在 1.5–2.0m 就朝正確方向早轉」）：
  1. 轉向 onset 距離：goal 正前方場景，|ω| 是否從 1.5–2.0m 起 > 空曠基線。
  2. 安全方向一致率：LiDAR 算出的最佳安全側(swept-arc)，policy 是否真的往那邊轉（sign(ω) 一致），
     不能只看 |ω| 大小——大而亂轉沒用。
  3. 有效橫移 vs 震盪：提早轉後是否真的橫移繞開，還是左右抖動。

輸入＝ play_rnn_car.py 的 CHARGE_LIDAR_DUMP（已擴充 goal_te/pose_te）。
用法：
  python swept_arc_directional.py --control ctrl.npz --arc arc.npz [--ahead_deg 20] [--wp_grid ...]

判準（用戶定）：
  - Arc 在 1.5–2.0m 確朝**正確方向**提早轉（onset 左移 + sign-agreement 明顯 >50%），只是轉過頭/震盪
    → 調轉向 reward/平滑度（非判死）。
  - 仍到 ~0.5m 才朝正確方向轉（onset 不左移 or sign-agreement≈亂猜）→ r_arc 真正失敗。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swept_arc as sa  # noqa: E402

# 前錐 bin36 ±6 = beams 30:43（對齊 play_rnn_car.py 反應曲線）
FRONT_LO, FRONT_HI = 30, 43
# 前障距離分箱（m），由遠到近
DIST_BINS = [(5.0, 99.0, ">5.0 空曠"), (3.0, 5.0, "3.0-5.0"), (2.5, 3.0, "2.5-3.0"),
             (2.0, 2.5, "2.0-2.5"), (1.5, 2.0, "★1.5-2.0"), (1.0, 1.5, "1.0-1.5"),
             (0.5, 1.0, "0.5-1.0"), (0.0, 0.5, "0.0-0.5")]


def load_dump(path: str):
    """回傳 dict：lidar[T,E,72], v[T,E], omega[T,E](帶符號), goal[T,E,2], pose[T,E,3]。"""
    with np.load(path) as d:
        T = int(d["T"]) if "T" in d else None
        E = int(d["E"]) if "E" in d else None
        lidar = d["lidar_m"].astype(np.float32)
        va = d["va"].astype(np.float32)
        if T is None or "goal_te" not in d:
            raise ValueError(f"{path} 缺 T/E/goal_te —— 需用擴充後的 play_rnn_car.py 重新收 dump")
        lidar = lidar.reshape(T, E, 72)
        va = va.reshape(T, E, 2)
        goal = d["goal_te"].astype(np.float32)          # [T,E,2]
        pose = d["pose_te"].astype(np.float32)          # [T,E,3]
    return {"lidar": lidar, "v": va[..., 0], "omega": va[..., 1],
            "goal": goal, "pose": pose, "T": T, "E": E}


def front_dist(lidar_te: np.ndarray, hole_thresh: float = 0.4, max_range: float = 20.0) -> np.ndarray:
    """前錐最近障礙距離(m)；破洞/超範圍→inf。lidar_te:[...,72] → [...]"""
    f = lidar_te[..., FRONT_LO:FRONT_HI]
    f = np.where((f > hole_thresh) & (f < max_range), f, np.inf)
    return f.min(axis=-1)


def best_safe_omega(lidar_flat: torch.Tensor, v_flat: torch.Tensor, wp_grid, horizon, body_radius, device):
    """對每個 frame 掃 ω 網格用 arc_clearance 找最清弧的 ω → 回傳 best_omega[B]、best_clear[B]。

    v 用實際速度但夾 ≥0.4 讓弧有前伸（v≈0 無方向性）。"""
    B = lidar_flat.shape[0]
    W = len(wp_grid)
    v_probe = v_flat.clamp(min=0.4)
    lm_rep = lidar_flat.repeat_interleave(W, dim=0)                      # [B*W,72]
    v_rep = v_probe.repeat_interleave(W, dim=0)                          # [B*W]
    w_rep = torch.tensor(wp_grid, device=device, dtype=torch.float32).repeat(B)  # [B*W]
    c = sa.arc_clearance(lm_rep, v_rep, w_rep, horizon=horizon,
                         body_radius=body_radius, max_range=20.0, hole_thresh_m=0.4,
                         chunk=20000).reshape(B, W)
    best_wi = c.argmax(dim=1)
    w_tensor = torch.tensor(wp_grid, device=device, dtype=torch.float32)
    return w_tensor[best_wi], c.amax(dim=1)


def analyze(dump, tag, args, device):
    lidar = dump["lidar"]; omega = dump["omega"]; goal = dump["goal"]
    T, E = dump["T"], dump["E"]
    fd = front_dist(lidar)                                   # [T,E]
    goal_brg = np.abs(np.arctan2(goal[..., 1], goal[..., 0]))  # [T,E] rad
    ahead = goal_brg < np.deg2rad(args.ahead_deg)           # [T,E] 目標正前方
    absw = np.abs(omega)

    # --- [1] goal-ahead |ω| vs 前障距離（onset 曲線）---
    curve = []
    for lo, hi, name in DIST_BINS:
        m = ahead & (fd >= lo) & (fd < hi)
        n = int(m.sum())
        wbar = float(absw[m].mean()) if n > 0 else float("nan")
        curve.append((name, lo, hi, wbar, n))
    baseline = next((w for nm, lo, hi, w, n in curve if lo >= 5.0 and n > 20), float("nan"))

    # --- [2] 安全方向一致率（goal-ahead & 前障 in [0.5,2.5] & |ω|>thr）---
    near = ahead & (fd >= 0.5) & (fd < 2.5) & np.isfinite(fd)
    idx = np.argwhere(near)                                  # [(t,e),...]
    agree_by_bin = {nm: [0, 0] for _, _, nm in DIST_BINS}   # name -> [agree, total]
    overall = [0, 0]
    if len(idx) > 0:
        lm = torch.from_numpy(lidar[idx[:, 0], idx[:, 1]]).to(device)   # [B,72]
        vv = torch.from_numpy(dump["v"][idx[:, 0], idx[:, 1]]).to(device)
        ww = omega[idx[:, 0], idx[:, 1]]                                # signed actual ω
        fdi = fd[idx[:, 0], idx[:, 1]]
        best_w, _ = best_safe_omega(lm, vv, args.wp_grid, args.horizon, args.body_radius, device)
        best_w = best_w.cpu().numpy()
        safe_side = np.sign(best_w) * (np.abs(best_w) > args.side_thresh)  # -1/0/+1
        act_side = np.sign(ww) * (np.abs(ww) > args.turn_thresh)
        directional = (safe_side != 0) & (act_side != 0)     # 兩邊都真的有方向
        agree = directional & (safe_side == act_side)
        overall = [int(agree.sum()), int(directional.sum())]
        for k, (lo, hi, nm) in enumerate([(lo, hi, nm) for lo, hi, nm in DIST_BINS]):
            bm = directional & (fdi >= lo) & (fdi < hi)
            if int(bm.sum()) > 0:
                agree_by_bin[nm] = [int((agree & bm).sum()), int(bm.sum())]

    # --- [3] 有效橫移 vs 震盪（分 episode，goal-ahead 近障 approach 段）---
    pose = dump["pose"]                                       # [T,E,3] x,y,yaw
    lat_list, flip_list = [], []
    for e in range(E):
        # episode 分段：pose 跳躍 >1.5m = reset teleport
        p = pose[:, e, :]                                     # [T,3]
        jumps = np.concatenate([[0], np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)])
        seg_start = 0
        for t in range(1, T + 1):
            if t == T or jumps[t] > 1.5:
                s, en = seg_start, t
                seg_start = t
                if en - s < 5:
                    continue
                # approach 窗：goal-ahead 且 前障從 <2.0 掉到 >0.5
                fseg = fd[s:en, e]; aseg = ahead[s:en, e]; wseg = omega[s:en, e]
                appr = aseg & (fseg < 2.0) & (fseg > 0.4) & np.isfinite(fseg)
                if int(appr.sum()) < 3:
                    continue
                # 淨橫移：approach 段起點 heading 為軸，算垂直位移
                ii = np.argwhere(appr).reshape(-1)
                a0, a1 = ii[0], ii[-1]
                yaw0 = p[s + a0, 2]
                disp = p[s + a1, :2] - p[s + a0, :2]
                lat = float(-np.sin(yaw0) * disp[0] + np.cos(yaw0) * disp[1])  # body-frame 橫移(左+)
                lat_list.append(abs(lat))
                # 震盪：approach 段內 ω sign 翻轉次數 / 步數
                wsa = wseg[appr]
                sgn = np.sign(wsa[np.abs(wsa) > args.turn_thresh])
                flips = int((np.diff(sgn) != 0).sum()) if len(sgn) > 1 else 0
                flip_list.append(flips / max(1, len(wsa)))

    return {"tag": tag, "curve": curve, "baseline": baseline,
            "agree_overall": overall, "agree_by_bin": agree_by_bin,
            "lat_mean": float(np.mean(lat_list)) if lat_list else float("nan"),
            "lat_n": len(lat_list),
            "flip_mean": float(np.mean(flip_list)) if flip_list else float("nan")}


def onset_distance(curve, baseline, margin):
    """|ω| 首次(由遠到近) > baseline+margin 的距離箱上緣。"""
    for name, lo, hi, w, n in curve:
        if lo >= 5.0:
            continue
        if n > 20 and np.isfinite(w) and w > baseline + margin:
            return hi, name
    return None, None


def print_compare(rc, ra, args):
    print("\n" + "=" * 74)
    print("方向性反應診斷：Control vs Arc（判「早轉朝正確方向」非「減速」）")
    print("=" * 74)

    print(f"\n[1] 目標正前方 |ω| vs 前障距離 (onset)   goal_ahead<{args.ahead_deg}°")
    print(f"{'前障距離':>12} | {'Control |ω|':>12} {'(n)':>7} | {'Arc |ω|':>10} {'(n)':>7} | {'Δ(Arc-Ctrl)':>11}")
    print("-" * 74)
    cmap = {nm: (w, n) for nm, lo, hi, w, n in rc["curve"]}
    amap = {nm: (w, n) for nm, lo, hi, w, n in ra["curve"]}
    for nm, lo, hi, w, n in rc["curve"]:
        cw, cn = cmap[nm]; aw, an = amap[nm]
        d = (aw - cw) if (np.isfinite(aw) and np.isfinite(cw)) else float("nan")
        print(f"{nm:>12} | {cw:>12.3f} {cn:>7} | {aw:>10.3f} {an:>7} | {d:>+11.3f}")
    co, cn = onset_distance(rc["curve"], rc["baseline"], args.onset_margin)
    ao, an = onset_distance(ra["curve"], ra["baseline"], args.onset_margin)
    print(f"  空曠基線 |ω|: Control {rc['baseline']:.3f} / Arc {ra['baseline']:.3f}  "
          f"(onset 門檻=baseline+{args.onset_margin})")
    print(f"  ★轉向 onset 距離: Control {co}m ({cn}) / Arc {ao}m ({an})")
    if ao and co and ao > co:
        print(f"    → Arc onset 更遠({ao}>{co}) = 提早轉 ✔（左移）")
    elif ao and co and ao == co:
        print(f"    → onset 同箱，看 [2] 方向對不對才知有沒有用")
    else:
        print(f"    → Arc onset 未比 Control 遠 = 沒提早（除非 [2] 一致率高）")

    print(f"\n[2] 安全方向一致率 sign(ω)==best-safe-side  (goal-ahead & 前障0.5-2.5m & 雙側都轉)")
    print(f"    best-safe-side = swept-arc arc_clearance 掃 ω 網格取最清弧方向")
    print(f"{'前障距離':>12} | {'Control 一致率':>16} | {'Arc 一致率':>16}")
    print("-" * 74)
    for _, _, nm in [(lo, hi, nm) for lo, hi, nm in DIST_BINS]:
        ca, ct = rc["agree_by_bin"][nm]; aa, at = ra["agree_by_bin"][nm]
        cs = f"{100*ca/ct:5.1f}% ({ct})" if ct > 0 else "   -  "
        as_ = f"{100*aa/at:5.1f}% ({at})" if at > 0 else "   -  "
        print(f"{nm:>12} | {cs:>16} | {as_:>16}")
    co_a, co_t = rc["agree_overall"]; ao_a, ao_t = ra["agree_overall"]
    cpct = 100 * co_a / co_t if co_t else float("nan")
    apct = 100 * ao_a / ao_t if ao_t else float("nan")
    print(f"  ★總一致率: Control {cpct:.1f}% (n={co_t}) / Arc {apct:.1f}% (n={ao_t})   (50%=亂猜基準)")

    print(f"\n[3] 有效橫移 vs 震盪 (goal-ahead 近障 approach 段, 前障<2.0m)")
    print(f"  淨橫移(m,越大越有效繞開): Control {rc['lat_mean']:.3f} (n={rc['lat_n']}) / "
          f"Arc {ra['lat_mean']:.3f} (n={ra['lat_n']})")
    print(f"  ω sign 翻轉率(越高越左右抖): Control {rc['flip_mean']:.3f} / Arc {ra['flip_mean']:.3f}")

    print(f"\n{'='*74}\n★裁決指引:")
    print("  - Arc onset 左移 + 一致率明顯>50% + 橫移↑ → 有效早轉(即使震盪也是調 reward/平滑不是判死)")
    print("  - onset 不左移 OR 一致率≈50%(亂轉) → r_arc 真的沒學會朝正確方向早轉 = 失敗")
    print("  - 一致率高但橫移沒增/翻轉率高 → 方向對但轉過頭/震盪 → 調平滑度")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", required=True, help="Control checkpoint 的 CHARGE_LIDAR_DUMP npz")
    p.add_argument("--arc", required=True, help="Arc checkpoint 的 CHARGE_LIDAR_DUMP npz")
    p.add_argument("--ahead_deg", type=float, default=20.0, help="目標方位角<此(°)算正前方")
    p.add_argument("--horizon", type=float, default=2.7)
    p.add_argument("--body_radius", type=float, default=0.35)
    p.add_argument("--turn_thresh", type=float, default=0.10, help="|ω|>此才算「有在轉」(rad/s)")
    p.add_argument("--side_thresh", type=float, default=0.10, help="|best_ω|>此才算「有安全側」")
    p.add_argument("--onset_margin", type=float, default=0.08, help="|ω|>baseline+此算 onset")
    p.add_argument("--wp_grid", type=lambda s: [float(x) for x in s.split(",")],
                   default=[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dc = load_dump(args.control)
    da = load_dump(args.arc)
    print(f"[dir] Control dump T×E={dc['T']}×{dc['E']}  Arc dump T×E={da['T']}×{da['E']}  device={device}")
    rc = analyze(dc, "Control", args, device)
    ra = analyze(da, "Arc", args, device)
    print_compare(rc, ra, args)


if __name__ == "__main__":
    main()
