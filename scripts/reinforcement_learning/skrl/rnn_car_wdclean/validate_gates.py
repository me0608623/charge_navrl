"""validate_gates.py — checkpoint 晉級驗收分析器（用戶 2026-07-15 定的四閘協定）。

由 validate_checkpoint.sh 呼叫；讀各 gate 的 play det log + 擋路場 CHARGE_LIDAR_DUMP，
算指標、對逐階段門檻判 PASS/FAIL。不用 training stochastic SR 代替 det。

四閘（SA1 門檻示例，其餘見 STAGE_THRESH）：
  #1 訓練健康  : approx_kl 中位≤0.015(不持續>0.0225) / clip_fraction<0.25 / enc_g中位>1e-3 / vf無3×爆
  #2 held-out det: 3 seed×≥1000ep, SR≥0.98 / CR≤0.015 / TO≤0.005
  #3 擋路測試  : SR≥0.90 / CR≤0.07 / TO≤0.03 + clear|ω|中位≤0.15 + Δ|ω|@1.5-2m≥0.10
                 + 實際閉迴路安全軌跡率≥0.85 + clear停止率<0.05
  #4 SA2 preview: SA1 ckpt 丟 SA2 不訓, det SR≥0.85
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swept_arc as sa  # noqa: E402

# ── 逐階段門檻 ─────────────────────────────────────────────
STAGE_THRESH = {
    1: dict(det_sr=0.98, det_cr=0.015, det_to=0.005,
            blk_sr=0.90, blk_cr=0.07, blk_to=0.03,
            clear_omega_max=0.15, delta_omega_min=0.10, safe_arc_min=0.85, clear_stop_max=0.05,
            preview_stage=2, preview_sr=0.85),
    2: dict(det_sr=0.96, det_cr=0.03, det_to=0.02, blk_sr=0.88, blk_cr=0.08, blk_to=0.03,
            clear_omega_max=0.20, delta_omega_min=0.10, safe_arc_min=0.83, clear_stop_max=0.06,
            preview_stage=3, preview_sr=0.80),
    3: dict(det_sr=0.94, det_cr=0.05, det_to=0.03, blk_sr=0.85, blk_cr=0.10, blk_to=0.04,
            clear_omega_max=0.25, delta_omega_min=0.10, safe_arc_min=0.80, clear_stop_max=0.06,
            preview_stage=4, preview_sr=0.78),
    4: dict(det_sr=0.92, det_cr=0.07, det_to=0.035, blk_sr=0.83, blk_cr=0.12, blk_to=0.045,
            clear_omega_max=0.27, delta_omega_min=0.10, safe_arc_min=0.79, clear_stop_max=0.065,
            preview_stage=5, preview_sr=0.77),
    5: dict(det_sr=0.90, det_cr=0.09, det_to=0.04, blk_sr=0.81, blk_cr=0.13, blk_to=0.05,
            clear_omega_max=0.28, delta_omega_min=0.09, safe_arc_min=0.78, clear_stop_max=0.07,
            preview_stage=6, preview_sr=0.76),
    6: dict(det_sr=0.88, det_cr=0.11, det_to=0.045, blk_sr=0.79, blk_cr=0.15, blk_to=0.055,
            clear_omega_max=0.29, delta_omega_min=0.09, safe_arc_min=0.76, clear_stop_max=0.075,
            preview_stage=7, preview_sr=0.75),
    7: dict(det_sr=0.86, det_cr=0.13, det_to=0.05, blk_sr=0.78, blk_cr=0.16, blk_to=0.06,
            clear_omega_max=0.30, delta_omega_min=0.08, safe_arc_min=0.75, clear_stop_max=0.08,
            preview_stage=8, preview_sr=0.75),
    8: dict(det_sr=0.90, det_cr=0.10, det_to=0.05, blk_sr=0.82, blk_cr=0.12, blk_to=0.05,
            clear_omega_max=0.30, delta_omega_min=0.08, safe_arc_min=0.78, clear_stop_max=0.06,
            preview_stage=None, preview_sr=None),  # SA8 = 畢業(部署標準),無 preview
}
FRONT_LO, FRONT_HI = 30, 43  # 前錐 bin36±6
NARROW_DEPLOY_WIDTH_M = 1.2
NARROW_STRESS_WIDTH_M = 1.0
NARROW_DIAGNOSTIC_YAW_DEG = 10.0
NARROW_DEPLOY_THRESH = dict(
    sr=0.90,
    cr=0.05,
    crossing=0.95,
    direct_crossing=0.95,
)


# ── log 解析 ─────────────────────────────────────────────
def parse_play_summary(path: str) -> dict | None:
    """從 play 的『PLAY 統計摘要』抓 SR/CR/TO/回合數。找不到→None。"""
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return None
    def pct(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) / 100.0 if m else None
    n = re.search(r"總回合數[:：]\s*(\d+)", txt)
    return {
        "n": int(n.group(1)) if n else 0,
        "sr": pct(r"成功率.*?\((\d+\.?\d*)%\)"),
        "cr": pct(r"碰撞率\s*\(總計\).*?\((\d+\.?\d*)%\)"),
        "to": pct(r"超時率.*?\((\d+\.?\d*)%\)"),
    }


def agg_seeds(paths: list[str]) -> dict:
    """多 seed det log 聚合(以回合數加權)。"""
    rs = [parse_play_summary(p) for p in paths]
    rs = [r for r in rs if r and r["sr"] is not None]
    if not rs:
        return {"n": 0, "sr": None, "cr": None, "to": None, "seeds": 0}
    N = sum(r["n"] for r in rs) or 1
    w = lambda k: sum(r[k] * r["n"] for r in rs) / N
    return {"n": N, "sr": w("sr"), "cr": w("cr"), "to": w("to"), "seeds": len(rs)}


def parse_narrow_gap(path: str) -> dict | None:
    """Parse policy outcomes plus throat-yaw statistics from narrow-gap play."""
    summary = parse_play_summary(path)
    try:
        txt = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return None
    line = re.search(r"\[NARROW-GAP-METRICS\](.*)", txt)
    if not summary or not line:
        return None
    values = {
        key: float(value)
        for key, value in re.findall(r"([A-Za-z0-9_.]+)=([+-]?(?:nan|\d+(?:\.\d+)?))", line.group(1))
    }
    return {**summary, **values}


def narrow_gap_checks(metrics: dict) -> dict:
    """Build hard deployment checks; yaw remains a non-gating diagnostic."""
    required = ("sr", "cr", "crossing_rate", "direct_crossing_rate")
    if any(metrics.get(key) is None for key in required):
        return {}
    return {
        "到達SR": (
            metrics["sr"] >= NARROW_DEPLOY_THRESH["sr"],
            f"{metrics['sr']:.3f}≥{NARROW_DEPLOY_THRESH['sr']}",
        ),
        "碰撞CR": (
            metrics["cr"] <= NARROW_DEPLOY_THRESH["cr"],
            f"{metrics['cr']:.3f}≤{NARROW_DEPLOY_THRESH['cr']}",
        ),
        "穿越率": (
            metrics["crossing_rate"] >= NARROW_DEPLOY_THRESH["crossing"],
            f"{metrics['crossing_rate']:.3f}≥{NARROW_DEPLOY_THRESH['crossing']}",
        ),
        "直接穿越率": (
            metrics["direct_crossing_rate"]
            >= NARROW_DEPLOY_THRESH["direct_crossing"],
            f"{metrics['direct_crossing_rate']:.3f}"
            f"≥{NARROW_DEPLOY_THRESH['direct_crossing']}",
        ),
    }


# ── gate #3 行為指標(擋路場 dump) ────────────────────────
def _closed_loop_safe_path(lidar: np.ndarray, pose: np.ndarray, mask: np.ndarray, *,
                           horizon: float, control_dt: float, body_radius: float,
                           safe_clearance: float, max_step_jump: float = 0.5) -> tuple[float, int]:
    """Measure clearance along the policy's recorded closed-loop path.

    The policy replans every ``control_dt``. Holding one sampled action for the full
    horizon measures an open-loop controller that is not deployed. This instead
    compares the current LiDAR point cloud with the actual future robot poses and
    rejects windows crossing an episode reset (a pose jump over ``max_step_jump``).
    """
    steps = max(1, int(np.floor(horizon / control_dt)))
    if pose.shape[0] <= steps:
        return float("nan"), 0

    idx = np.argwhere(mask[:-steps])
    if not len(idx):
        return float("nan"), 0
    ti, ei = idx[:, 0], idx[:, 1]
    seq = np.stack([pose[ti + j, ei] for j in range(steps + 1)], axis=1)
    continuous = (np.linalg.norm(np.diff(seq[..., :2], axis=1), axis=-1) < max_step_jump).all(axis=1)
    idx, seq = idx[continuous], seq[continuous]
    if not len(idx):
        return float("nan"), 0

    angles = np.deg2rad(np.arange(lidar.shape[-1]) * 5.0 - 180.0).astype(np.float32)
    safe, total = 0, 0
    for start in range(0, len(idx), 3000):
        batch_idx = idx[start:start + 3000]
        batch_seq = seq[start:start + 3000]
        delta = batch_seq[:, 1:, :2] - batch_seq[:, None, 0, :2]
        yaw = batch_seq[:, 0, 2]
        cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
        rx = cy * delta[..., 0] + sy * delta[..., 1]
        ry = -sy * delta[..., 0] + cy * delta[..., 1]

        lm = lidar[batch_idx[:, 0], batch_idx[:, 1]]
        px = lm * np.cos(angles)[None, :]
        py = lm * np.sin(angles)[None, :]
        valid = (lm > 0.4) & (lm < 20.0)
        distance = np.sqrt(
            (rx[..., None] - px[:, None, :]) ** 2
            + (ry[..., None] - py[:, None, :]) ** 2
        )
        distance = np.where(valid[:, None, :], distance, np.inf)
        clearance = distance.min(axis=(1, 2)) - body_radius
        safe += int((clearance >= safe_clearance).sum())
        total += len(clearance)
    return safe / total, total


def blocking_behavior(dump_path: str, ahead_deg=20.0, horizon=2.7, body_radius=0.35,
                      safe_clearance=0.10, stop_v=0.05, control_dt=0.2) -> dict:
    """從擋路場 dump 算提前轉向、閉迴路安全軌跡率與 clear 停止率。"""
    import torch
    with np.load(dump_path) as d:
        T, E = int(d["T"]), int(d["E"])
        lidar = d["lidar_m"].astype(np.float32).reshape(T, E, 72)
        va = d["va"].astype(np.float32).reshape(T, E, 2)
        goal = d["goal_te"].astype(np.float32)
        pose = d["pose_te"].astype(np.float32) if "pose_te" in d else None
    v, omega = va[..., 0], va[..., 1]
    f = lidar[..., FRONT_LO:FRONT_HI]
    fd = np.where((f > 0.4) & (f < 20), f, np.inf).min(axis=-1)   # 前障距離
    ahead = np.abs(np.arctan2(goal[..., 1], goal[..., 0])) < np.deg2rad(ahead_deg)
    absw = np.abs(omega)

    clear = ahead & (fd > 3.0)                                    # goal 正前方 + 無近障 = clear
    recovery_clear = (
        (np.abs(np.arctan2(goal[..., 1], goal[..., 0])) < np.deg2rad(15.0))
        & (fd > 2.5)
    )                                                             # 繞障後應回正的開闊直行幀
    near = ahead & (fd >= 1.5) & (fd < 2.0)                       # 1.5-2.0m 擋路
    clear_omega = float(np.median(absw[clear])) if clear.any() else float("nan")
    recovery_clear_omega = (
        float(np.median(absw[recovery_clear])) if recovery_clear.any() else float("nan")
    )
    near_omega = float(np.mean(absw[near])) if near.any() else float("nan")
    delta_omega = near_omega - clear_omega if np.isfinite(near_omega) and np.isfinite(clear_omega) else float("nan")
    clear_stop = float((v[clear] < stop_v).mean()) if clear.any() else float("nan")

    # 實際 action-conditioned 弧線是否有淨空。中央單障礙的左/右常同時可解，
    # 因此不用「和 argmax 側的符號一致」，避免把另一個等價安全側誤判為錯。
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nmask = ahead & (fd >= 0.5) & (fd < 2.5) & np.isfinite(fd)
    idx = np.argwhere(nmask)
    open_loop_safe_arc = float("nan")
    if len(idx) > 30:
        lm = torch.from_numpy(lidar[idx[:, 0], idx[:, 1]]).to(dev)
        vv = torch.from_numpy(v[idx[:, 0], idx[:, 1]]).to(dev)
        ww = torch.from_numpy(omega[idx[:, 0], idx[:, 1]]).to(dev)
        c_actual = sa.arc_clearance(lm, vv, ww, horizon=horizon, body_radius=body_radius,
                                    chunk=20000)
        open_loop_safe_arc = float((c_actual >= safe_clearance).float().mean().item())
    closed_loop_safe_arc, closed_loop_n = (float("nan"), 0)
    if pose is not None:
        closed_loop_safe_arc, closed_loop_n = _closed_loop_safe_path(
            lidar, pose, nmask, horizon=horizon, control_dt=control_dt,
            body_radius=body_radius, safe_clearance=safe_clearance,
        )
    return dict(clear_omega=clear_omega, recovery_clear_omega=recovery_clear_omega,
                near_omega=near_omega, delta_omega=delta_omega,
                clear_stop=clear_stop, open_loop_safe_arc=open_loop_safe_arc,
                closed_loop_safe_arc=closed_loop_safe_arc, closed_loop_n=closed_loop_n)


# ── gate #1 訓練健康(wandb) ──────────────────────────────
def training_health(run_id: str, last_n=20) -> dict:
    try:
        import wandb
        api = wandb.Api(timeout=30)
        run_path = run_id if "/" in run_id else f"charge_skrl/{run_id}"
        r = api.run(run_path)
        hist = r.history(keys=["rl/approx_kl", "rl/clip_fraction", "rl_encoder/grad_norm_pre_clip",
                               "rl/value_loss"], samples=10000)
        def col(k):
            if k not in hist:
                return np.array([])
            return np.array([x for x in hist[k].values[-last_n:] if x is not None], dtype=float)
        kl, cf, eg, vf = col("rl/approx_kl"), col("rl/clip_fraction"), col("rl_encoder/grad_norm_pre_clip"), col("rl/value_loss")
        vf_spike = bool(len(vf) >= 4 and vf[-1] > 3.0 * np.median(vf[:-1])) if len(vf) else False
        return dict(
            kl_med=float(np.median(kl)) if len(kl) else None,
            kl_sustained=float(np.percentile(kl, 90)) if len(kl) else None,
            clip_frac=float(np.mean(cf)) if len(cf) else None,
            enc_g_med=float(np.median(eg)) if len(eg) else None,
            vf_spike=vf_spike,
        )
    except Exception as e:
        return {"error": str(e)}


# ── 判定 + 報告 ─────────────────────────────────────────
def _p(ok):
    return "✅PASS" if ok else "❌FAIL"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--wandb_id", default="gtefxx15")
    p.add_argument("--det_glob", required=True, help="held-out det logs glob(3 seed)")
    p.add_argument("--blk_log", default="", help="SA5+ 擋路場 det log")
    p.add_argument("--blk_dump", default="", help="SA5+ 擋路場 CHARGE_LIDAR_DUMP npz")
    p.add_argument("--preview_log", default="", help="SA(N+1) preview det log")
    p.add_argument("--narrow_deploy_log", default="", help="OBB lineage 1.2m deployment narrow-gap det log")
    p.add_argument("--narrow_stress_log", default="", help="OBB lineage 1.0m stress narrow-gap det log")
    p.add_argument("--narrow_log", default="", help=argparse.SUPPRESS)
    p.add_argument("--ckpt", default="")
    args = p.parse_args()
    if args.stage not in STAGE_THRESH:
        p.error(f"unsupported stage {args.stage}; expected one of {sorted(STAGE_THRESH)}")
    th = STAGE_THRESH[args.stage]
    advanced_gates = args.stage >= 5

    print(f"\n{'='*68}\ncheckpoint 晉級驗收 — Stage {args.stage}  {os.path.basename(args.ckpt)}\n{'='*68}")
    results = []

    # gate #1
    h = training_health(args.wandb_id)
    if "error" in h:
        print(f"[Gate1 訓練健康] ⚠ 無法取 wandb: {h['error']}")
        g1 = None
    else:
        kl_ok = (h["kl_med"] is not None and h["kl_med"] <= 0.015 and h["kl_sustained"] <= 0.0225)
        cf_ok = (h["clip_frac"] is None or h["clip_frac"] < 0.25)
        eg_ok = (h["enc_g_med"] is not None and h["enc_g_med"] > 1e-3)
        g1 = kl_ok and cf_ok and eg_ok and not h["vf_spike"]
        print(f"[Gate1 訓練健康] {_p(g1)}  kl中位={h['kl_med']} p90={h['kl_sustained']} "
              f"clip_frac={h['clip_frac']} enc_g中位={h['enc_g_med']} vf爆={h['vf_spike']}")
    results.append(("訓練健康", g1))

    # gate #2 held-out det
    det = agg_seeds(sorted(glob.glob(args.det_glob)))
    if det["sr"] is None:
        print(f"[Gate2 held-out det] ⚠ 無 det log({args.det_glob})"); g2 = None
    else:
        g2 = det["sr"] >= th["det_sr"] and det["cr"] <= th["det_cr"] and det["to"] <= th["det_to"]
        print(f"[Gate2 held-out det] {_p(g2)}  {det['seeds']}seed/{det['n']}ep  "
              f"SR={det['sr']:.3f}(≥{th['det_sr']}) CR={det['cr']:.3f}(≤{th['det_cr']}) TO={det['to']:.3f}(≤{th['det_to']})")
    results.append(("held-out det", g2))

    # gate #3 擋路測試
    if not advanced_gates:
        print(f"[Gate3 擋路測試] ↷ DEFERRED — 依協定自 SA5 起啟用")
        g3 = None
    else:
        blk = parse_play_summary(args.blk_log)
        beh = blocking_behavior(args.blk_dump) if os.path.exists(args.blk_dump) else {}
        if not blk or blk["sr"] is None or not beh:
            print(f"[Gate3 擋路測試] ⚠ 缺 log/dump"); g3 = None
        else:
            checks = {
                "SR": (blk["sr"] >= th["blk_sr"], f"{blk['sr']:.3f}≥{th['blk_sr']}"),
                "CR": (blk["cr"] <= th["blk_cr"], f"{blk['cr']:.3f}≤{th['blk_cr']}"),
                "TO": (blk["to"] <= th["blk_to"], f"{blk['to']:.3f}≤{th['blk_to']}"),
                "clear|ω|": (beh["clear_omega"] <= th["clear_omega_max"], f"{beh['clear_omega']:.3f}≤{th['clear_omega_max']}"),
                "Δ|ω|@1.5-2m": (beh["delta_omega"] >= th["delta_omega_min"], f"{beh['delta_omega']:+.3f}≥{th['delta_omega_min']}"),
                "閉迴路安全弧%": (beh["closed_loop_safe_arc"] >= th["safe_arc_min"],
                                  f"{beh['closed_loop_safe_arc']:.3f}≥{th['safe_arc_min']}"),
                "clear停止率": (beh["clear_stop"] <= th["clear_stop_max"], f"{beh['clear_stop']:.3f}≤{th['clear_stop_max']}"),
            }
            g3 = all(ok for ok, _ in checks.values())
            print(f"[Gate3 擋路測試] {_p(g3)}")
            for k, (ok, s) in checks.items():
                print(f"    {_p(ok)} {k:12s} {s}")
            print(f"    診斷 固定動作2.7s安全弧={beh['open_loop_safe_arc']:.3f} "
                  f"(非部署控制器；閉迴路樣本={beh['closed_loop_n']})")
            print(f"    診斷 path-recovery |ω|中位={beh['recovery_clear_omega']:.3f} rad/s "
                  "(goal<15deg 且前錐>2.5m；僅觀察、不參與 PASS/FAIL)")
        results.append(("擋路測試", g3))

    # gate #4 preview
    if th["preview_stage"] is None:
        print(f"[Gate4 preview] — SA{args.stage} 為畢業階段,無 preview")
        g4 = True
    elif not args.preview_log:
        print(f"[Gate4 preview] ⚠ 無 preview log"); g4 = None
    else:
        pv = parse_play_summary(args.preview_log)
        g4 = pv and pv["sr"] is not None and pv["sr"] >= th["preview_sr"]
        print(f"[Gate4 SA{th['preview_stage']} preview] {_p(g4)}  det SR={pv['sr'] if pv else None}(≥{th['preview_sr']})")
    results.append((f"SA{th['preview_stage']} preview" if th["preview_stage"] else "preview", g4))

    # OBB lineage extra gate: 1.2 m is the deployment hard gate. The 1.0 m
    # result is diagnostic only; 0.85 m remains a geometry-level test.
    #
    # 2026-07-27 正名：這個閘一律在 10 m 場地執行，而中央牆長度寫死按 10 m 算
    # （`NarrowGapSpec.arena_half_extent` 未接 `--arena_size`），因此牆剛好封死到
    # 外牆，牆端與外牆之間形成死角。實測 D0 在此死角撞牆 89.4% / direct 0.000，
    # 但同一顆 checkpoint 在**真正封死且尺寸相符**的 Gate5a（`--narrow_gap_mode
    # sealed`）上是 direct 1.000 / 零碰撞 / 橫偏中位 0.052 m。
    # 故此閘量的是「小場地邊界死角下的行為」，**不能代表純窄縫直穿能力**。
    # 純直穿請跑 Gate5a。此閘保留為壓測，不刪除。
    deploy_log = args.narrow_deploy_log or args.narrow_log
    if not advanced_gates:
        print(
            f"[Gate5a {NARROW_DEPLOY_WIDTH_M:.1f}m 封死窄縫·純直穿] "
            "↷ DEFERRED — 依協定自 SA5 起啟用"
        )
    elif deploy_log:
        ng = parse_narrow_gap(deploy_log)
        if not ng or ng.get("sr") is None:
            print(
                f"[Gate5a {NARROW_DEPLOY_WIDTH_M:.1f}m 封死窄縫·純直穿] "
                "⚠ 缺摘要或 NARROW-GAP-METRICS"
            )
            g5 = None
        else:
            checks = narrow_gap_checks(ng)
            if not checks:
                print(
                    f"[Gate5a {NARROW_DEPLOY_WIDTH_M:.1f}m 封死窄縫·純直穿] "
                    "⚠ 缺 SR/CR/穿越指標"
                )
                g5 = None
            else:
                g5 = all(ok for ok, _ in checks.values())
                print(
                    f"[Gate5 {NARROW_DEPLOY_WIDTH_M:.1f}m部署窄縫] {_p(g5)}  "
                    f"n={ng['n']} yaw_frames={int(ng['yaw_frames'])}"
                )
                for key, (ok, value) in checks.items():
                    print(f"    {_p(ok)} {key:12s} {value}")
                yaw_key = f"yaw_within_{NARROW_DIAGNOSTIC_YAW_DEG:.2f}deg"
                yaw_within = ng.get(yaw_key, float("nan"))
                print(
                    f"    診斷（不影響PASS） yaw分布 "
                    f"p50={ng.get('yaw_abs_p50_deg', float('nan')):.3f}° "
                    f"p90={ng.get('yaw_abs_p90_deg', float('nan')):.3f}° "
                    f"p95={ng.get('yaw_abs_p95_deg', float('nan')):.3f}° "
                    f"yaw≤{NARROW_DIAGNOSTIC_YAW_DEG:g}°率={yaw_within:.3f}"
                )
        results.append((f"{NARROW_DEPLOY_WIDTH_M:.1f}m部署窄縫", g5))

    if advanced_gates and args.narrow_stress_log:
        stress = parse_narrow_gap(args.narrow_stress_log)
        if not stress or stress.get("sr") is None:
            print(f"[Stress {NARROW_STRESS_WIDTH_M:.1f}m窄縫] ⚠ 缺摘要或 NARROW-GAP-METRICS")
        else:
            yaw_key = f"yaw_within_{NARROW_DIAGNOSTIC_YAW_DEG:.2f}deg"
            yaw_within = stress.get(yaw_key, float("nan"))
            print(
                f"[Stress {NARROW_STRESS_WIDTH_M:.1f}m窄縫] 診斷（不影響晉級）  "
                f"n={stress['n']} SR={stress['sr']:.3f} CR={stress['cr']:.3f} "
                f"穿越率={stress['crossing_rate']:.3f} "
                f"yaw_p95={stress['yaw_abs_p95_deg']:.3f}° "
                f"yaw≤{NARROW_DIAGNOSTIC_YAW_DEG:g}°率={yaw_within:.3f}"
            )

    # 綜合
    hard = [r for _, r in results if r is not None]
    overall = bool(hard) and all(hard)
    missing = [n for n, r in results if r is None]
    print(f"\n{'─'*68}")
    print(f"★綜合裁決: {'✅ 全 PASS → 可晉級' if overall and not missing else ('❌ 有 FAIL → 不晉級' if not overall else '⚠ 部分缺資料')}")
    if missing:
        print(f"  缺資料 gate: {missing}")
    print(f"  逐 gate: " + " | ".join(f"{n}={_p(r) if r is not None else '⚠缺'}" for n, r in results))
    if g3 is False:
        print(f"  ※ 擋路測試失敗 → 增加 SA{args.stage} 障礙落 goal-path 比例"
              "(非改 reward/空跑); det 用 argmax 非 training stochastic。")
    sys.exit(0 if overall and not missing else 1)


if __name__ == "__main__":
    main()
