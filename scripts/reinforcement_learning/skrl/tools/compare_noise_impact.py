"""對比兩個 WandB run 的訓練指標 — 用於量化 LiDAR 噪聲對訓練的影響。

預設對比 SA1_v2_lidar2 (clean) vs SA2_v2_lidar2 (noisy)。

用法：
    # 預設對比（SA1_v2_lidar2 vs SA2_v2_lidar2，最後 20 iter 平均）
    python compare_noise_impact.py

    # 指定 run name + 輸出 plot 與 CSV
    python compare_noise_impact.py --sa1 sa1_v2_lidar2_ne1024_s42 \\
        --sa2 sa2_v2_lidar2_ne1024_s42 --plot --save_csv

    # 對比最後 50 iter
    python compare_noise_impact.py --last_n 50

輸出：
    1. 終端：兩 run 在訓練尾段（last_n iter）的指標對比表
    2. 噪聲衝擊摘要（SR/CR/lin_idx 三項變化）
    3. 可選：./noise_impact_table.csv（完整對比表）
    4. 可選：./noise_impact_curves.png（六大指標 iter-curve overlay）
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import wandb


PROJECT = "me0608623-none/charge_skrl"

# 需要從 WandB 拉出的 metric → 顯示名稱（短名）
METRICS: dict[str, str] = {
    # Performance
    "charge/goal_reach_rate": "SR",
    "charge/obstacle_collision_rate": "obsCR",
    "charge/wall_collision_rate": "wallCR",
    "charge/timeout_rate": "TO",
    # Policy / RL
    "rl/entropy": "ent",
    "rl/entropy_angular": "ent_ang",
    "rl_critic/variance_explained": "vexp",
    # Behavior (action)
    "charge/action_linear_idx_mean": "lin_idx",
    "charge/speed_x_mean": "v_x",
    "charge/omega_std_over_steps": "omega_std",
    "charge/omega_mean": "omega_mean",
    "goal_diagnostics/heading_error_abs_mean_deg": "hdg_err",
    # 新增（如果 SA2 已部署新 metric 就會有）
    "charge/lidar_min_step_mean": "lidar_min",
    "charge/lidar_zero_count_step_mean": "lidar_zero",
    "charge/speed_x_near_obs_mean": "v_near_obs",
}


class CompareResult(NamedTuple):
    """單一 metric 的對比結果。"""

    metric: str
    sa1: float
    sa2: float
    delta: float
    pct: float


def fetch_run(api: wandb.Api, name: str) -> pd.DataFrame:
    """從 WandB 拉指定 display_name 的最新 run history。"""
    runs = list(api.runs(PROJECT, filters={"display_name": name}, order="-created_at"))
    if not runs:
        raise ValueError(f"No run found with display_name='{name}' in {PROJECT}")
    run = runs[0]
    print(f"  ↳ {name}: id={run.id} state={run.state} (loaded {len(run.history(samples=1)) >= 0})")
    rows = list(run.history(samples=20000, pandas=False))
    if not rows:
        raise ValueError(f"Run {name} has no history rows")
    df = pd.DataFrame(rows)
    df["iter"] = df["_step"] // 300
    keep_cols = [c for c in METRICS if c in df.columns]
    if "iter" not in df.columns or not keep_cols:
        raise ValueError(f"Run {name} missing required columns")
    df = df[["iter"] + keep_cols].copy()
    df = df.rename(columns={c: METRICS[c] for c in keep_cols})
    return df.dropna(how="all", subset=list(METRICS.values()) & set(df.columns))


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """加入衍生指標 — obsCR/SR 比、lin_idx 飽和差距。"""
    df = df.copy()
    if "obsCR" in df.columns and "SR" in df.columns:
        df["obsCR_per_SR"] = df["obsCR"] / (df["SR"] + 1e-6)
    if "lin_idx" in df.columns:
        df["lin_idx_from_max"] = 17.99 - df["lin_idx"]
    return df


def compare_tail(sa1: pd.DataFrame, sa2: pd.DataFrame, last_n: int) -> list[CompareResult]:
    """對比兩 run 在最後 last_n iter 的指標平均。"""
    sa1_end = sa1.tail(last_n).mean(numeric_only=True)
    sa2_end = sa2.tail(last_n).mean(numeric_only=True)
    results = []
    for col in sa1_end.index:
        if col == "iter":
            continue
        v1, v2 = sa1_end[col], sa2_end[col]
        if pd.isna(v1) or pd.isna(v2):
            continue
        delta = v2 - v1
        pct = (delta / (abs(v1) + 1e-6)) * 100
        results.append(CompareResult(col, v1, v2, delta, pct))
    return results


def render_table(results: list[CompareResult], sa1_name: str, sa2_name: str, last_n: int) -> str:
    """產生 markdown-friendly 對比表。"""
    lines = [
        "=" * 90,
        f"End-state comparison (mean over last {last_n} iter)",
        f"  SA1 (clean): {sa1_name}",
        f"  SA2 (noisy): {sa2_name}",
        "=" * 90,
        f"{'metric':<22} {'SA1 (clean)':>14} {'SA2 (noisy)':>14} {'Δ':>12} {'Δ%':>10}",
        "-" * 90,
    ]
    for r in results:
        lines.append(
            f"{r.metric:<22} {r.sa1:>+14.5f} {r.sa2:>+14.5f} {r.delta:>+12.5f} {r.pct:>+9.2f}%"
        )
    return "\n".join(lines)


def noise_impact_summary(results: list[CompareResult]) -> str:
    """產生「噪聲衝擊」三大關鍵指標摘要。"""
    by_name = {r.metric: r for r in results}
    lines = ["", "=" * 90, "噪聲衝擊摘要 (Noise Impact Summary)", "=" * 90]

    if "SR" in by_name:
        r = by_name["SR"]
        verdict = "🟢 微影響" if abs(r.delta) < 0.02 else "🟡 中等" if abs(r.delta) < 0.05 else "🔴 大幅退化"
        lines.append(f"SR     : {r.sa1 * 100:>6.2f}% → {r.sa2 * 100:>6.2f}%  (Δ={r.delta * 100:+.2f}pp)   {verdict}")

    if "obsCR" in by_name:
        r = by_name["obsCR"]
        verdict = "🟢 微升" if r.delta < 0.005 else "🟡 中等" if r.delta < 0.02 else "🔴 大幅升"
        lines.append(f"obsCR  : {r.sa1 * 100:>6.2f}% → {r.sa2 * 100:>6.2f}%  (Δ={r.delta * 100:+.2f}pp)   {verdict}")

    if "obsCR_per_SR" in by_name:
        r = by_name["obsCR_per_SR"]
        lines.append(f"CR/SR  : {r.sa1:>6.4f} → {r.sa2:>6.4f}  (Δ={r.delta:+.4f})   ← 噪聲對碰撞的純貢獻")

    if "vexp" in by_name:
        r = by_name["vexp"]
        verdict = "🟢 critic 穩" if abs(r.delta) < 0.02 else "🟡 下滑" if r.delta < 0 else "🟢 上升"
        lines.append(f"vexp   : {r.sa1:>6.4f} → {r.sa2:>6.4f}  (Δ={r.delta:+.4f})   {verdict}")

    if "lin_idx" in by_name:
        r = by_name["lin_idx"]
        slowdown = r.sa1 - r.sa2
        if slowdown > 0.5:
            verdict = "🟢 學會減速（robust）"
        elif slowdown > 0.1:
            verdict = "🟡 微減速"
        else:
            verdict = "🔴 沒減速（可能過擬合）"
        lines.append(f"lin_idx: {r.sa1:>6.2f}  → {r.sa2:>6.2f}   (Δ={-slowdown:+.2f})   {verdict}")

    if "hdg_err" in by_name:
        r = by_name["hdg_err"]
        verdict = "🟢 維持" if abs(r.delta) < 2 else "🟡 升高" if r.delta > 0 else "🟢 改善"
        lines.append(f"hdg_err: {r.sa1:>6.2f}° → {r.sa2:>6.2f}°  (Δ={r.delta:+.2f}°)   {verdict}")

    if "omega_std" in by_name:
        r = by_name["omega_std"]
        verdict = "🟢 穩定" if abs(r.delta) < 0.01 else "🟡 增加"
        lines.append(f"ω_std  : {r.sa1:>6.4f} → {r.sa2:>6.4f}  (Δ={r.delta:+.4f})   {verdict}")

    return "\n".join(lines)


def save_curves(
    sa1: pd.DataFrame,
    sa2: pd.DataFrame,
    sa1_name: str,
    sa2_name: str,
    out_path: Path,
) -> None:
    """畫六大指標的 iter-curve overlay。"""
    import matplotlib.pyplot as plt

    plots = [
        ("SR", "Success Rate"),
        ("obsCR", "Obstacle Collision Rate"),
        ("vexp", "Variance Explained"),
        ("lin_idx", "Linear Action Index (max=18)"),
        ("omega_std", "Omega Std Over Steps"),
        ("hdg_err", "Heading Error (deg)"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, (col, title) in zip(axes.flat, plots):
        if col in sa1.columns:
            ax.plot(sa1["iter"], sa1[col], label=f"SA1: {sa1_name}", color="C0", alpha=0.85, linewidth=1.5)
        if col in sa2.columns:
            ax.plot(sa2["iter"], sa2[col], label=f"SA2: {sa2_name}", color="C1", alpha=0.85, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("iter")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("LiDAR Noise Impact: SA1 (clean) vs SA2 (noisy)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sa1", default="sa1_v2_lidar2_ne1024_s42", help="SA1 (clean) run display_name")
    parser.add_argument("--sa2", default="sa2_v2_lidar2_ne1024_s42", help="SA2 (noisy) run display_name")
    parser.add_argument("--last_n", type=int, default=20, help="末段平均 iter 數（default: 20）")
    parser.add_argument("--save_csv", action="store_true", help="輸出對比表到 ./noise_impact_table.csv")
    parser.add_argument("--plot", action="store_true", help="輸出曲線到 ./noise_impact_curves.png")
    parser.add_argument("--out_dir", type=Path, default=Path.cwd(), help="輸出目錄（default: cwd）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading runs from WandB...")
    api = wandb.Api()
    sa1 = add_derived(fetch_run(api, args.sa1))
    sa2 = add_derived(fetch_run(api, args.sa2))

    print(f"\nSA1 history: {len(sa1)} rows (iter {sa1['iter'].min()}–{sa1['iter'].max()})")
    print(f"SA2 history: {len(sa2)} rows (iter {sa2['iter'].min()}–{sa2['iter'].max()})")

    results = compare_tail(sa1, sa2, args.last_n)
    print()
    print(render_table(results, args.sa1, args.sa2, args.last_n))
    print(noise_impact_summary(results))

    if args.save_csv:
        out = args.out_dir / "noise_impact_table.csv"
        pd.DataFrame([r._asdict() for r in results]).to_csv(out, index=False)
        print(f"\n  ✓ Saved CSV → {out}")

    if args.plot:
        out = args.out_dir / "noise_impact_curves.png"
        save_curves(sa1, sa2, args.sa1, args.sa2, out)
        print(f"  ✓ Saved plot → {out}")


if __name__ == "__main__":
    main()
