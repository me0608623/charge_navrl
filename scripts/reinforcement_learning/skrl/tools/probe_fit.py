#!/usr/bin/env python
"""Offline linear probe 擬合 — 裁決「RNN hidden 是否編碼障礙運動」。

用法:
    env python scripts/reinforcement_learning/skrl/tools/probe_fit.py <dump.npz>

輸入 npz (play_rnn_car --probe_dump 產生):
    X     [N,12]  preprocess_feat (12D 瓶頸)
    XH    [N,64]  RNN hidden
    Y     [N,6]   最近 3 個動態障礙的 body-frame 速度 target (scaled, dims 7-12)
    valid [N,3]   各障礙「有真動態」旗標 (speed > 1e-3)
    TGT   [N,13]  完整 aux target (t0/next/hist 位置 + 速度)
    EXT   [N,96]  extractor 輸出 (可選)

⚠ 2026-07-03 前的 dump 速度標籤來自零 buffer(汙染) — 只信本日修復後收集的 dump
  (sanity: Y 的 std 應 > 0, valid 比例應 > 0)。

方法: 80/20 train/test split + ridge-lite 線性迴歸 (lstsq), 報 test R²。
判準: 最近障礙 velocity R² 明顯 >0 (如 >0.2) = hidden 有編碼運動;
      ≈0 = 真的沒編碼 (這次是用好尺量的)。
"""
from __future__ import annotations

import sys

import numpy as np


def fit_r2(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Ridge-lite linear fit, return test R² (80/20 split)."""
    n = X.shape[0]
    if n < 200:
        return float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]
    Xb = np.concatenate([X, np.ones((n, 1))], axis=1)
    A = Xb[tr].T @ Xb[tr] + 1e-3 * np.eye(Xb.shape[1])
    w = np.linalg.solve(A, Xb[tr].T @ y[tr])
    pred = Xb[te] @ w
    ss_res = float(((y[te] - pred) ** 2).sum())
    ss_tot = float(((y[te] - y[te].mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-9)


def main(path: str) -> None:
    d = np.load(path)
    X, XH, Y, V, TGT = d["X"], d["XH"], d["Y"], d["valid"], d["TGT"]
    EXT = d["EXT"] if "EXT" in d.files and d["EXT"].size else None
    print(f"dump: {path}")
    print(f"  N={X.shape[0]}  feat12={X.shape}  hidden={XH.shape}  Y={Y.shape}")

    # ── 標籤 sanity(汙染檢查): 舊 dump 這裡會是 0 ──
    print(f"  [sanity] Y std={Y.std():.5f}  |Y| max={np.abs(Y).max():.4f}  "
          f"valid_frac(最近障礙)={V[:, 0].mean():.3f}")
    if Y.std() < 1e-6:
        print("  ⚠ 速度標籤全 0 — 這是汙染 dump（修復前收集），裁決無效")
        return

    # 只取「最近障礙真的是動態」的樣本
    m = V[:, 0].astype(bool)
    y_vel = Y[m][:, 0:2]      # 最近障礙 body-frame (vbx, vby)
    y_pos = TGT[m][:, 0:2]    # 最近障礙 t0 位置 (sanity, 位置一向可解)
    y_disp = TGT[m][:, 2:4] - TGT[m][:, 0:2]  # next - t0 (位移訊號, 修復後非零)
    print(f"  動態樣本 n={m.sum()}  disp std={y_disp.std():.5f} (修復後應>0)")

    print(f"\n{'':16s}{'velocity R²':>12s}{'position R²':>13s}{'disp R²':>10s}")
    for name, feats in (("hidden(64D)", XH[m]), ("feat12", X[m]),
                        ("extractor96" if EXT is not None else "", EXT[m] if EXT is not None else None)):
        if not name:
            continue
        rv = fit_r2(feats, y_vel)
        rp = fit_r2(feats, y_pos)
        rd = fit_r2(feats, y_disp)
        print(f"  {name:14s}{rv:12.3f}{rp:13.3f}{rd:10.3f}")

    print("\n判準: velocity R²(hidden) 明顯>0 → RNN 有編碼運動(舊負面結論翻案);"
          "≈0 → 用好尺量仍沒有(負面結論保留,但證據換成乾淨的)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe_mf4_90k_reallabels.npz")
