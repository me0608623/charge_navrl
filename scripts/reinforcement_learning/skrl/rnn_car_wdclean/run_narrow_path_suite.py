"""Run the fixed-seed narrow-gap gate with direct-path diagnostics.

2026-07-27 正名。這支腳本歷來跑的是 **Gate5b：10 m 邊界死角壓測**，不是純窄縫閘。

`NarrowGapSpec.arena_half_extent` 寫死 5.0 且 `--arena_size` 從未傳進去，
所以中央牆永遠只長到 y=±4.5 m。本腳本又固定用 `--arena_size 10`，
外牆內緣剛好也在 ±4.5 m —— 牆封死到外牆，牆端與外牆之間形成**死角**。

實測（D0, gap 1.2 m）：

| 場地 | 牆端側口 | direct crossing |
|---|---|---|
| 10 m（本腳本預設）| 0 m | **0.000**（撞牆 89.4%）|
| 12 m | 1.0 m | 0.0042（繞牆端，橫偏中位 5.02 m）|
| 14 m | 2.0 m | 1.000 |

同一顆 D0 在**真正封死且尺寸相符**的 Gate5a（`--narrow_gap_mode sealed`）上是
**direct 1.000 / 零碰撞 / 橫偏中位 0.052 m**（n=2304）。

所以：
* `--mode legacy`（預設，維持歷史行為）= **Gate5b**，量的是小場地邊界死角下的行為
* `--mode sealed` = **Gate5a**，牆隨場地延伸到外牆，才是純測直穿能力
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import (  # noqa: E402
    NARROW_DEPLOY_THRESH,
    narrow_gap_checks,
    parse_narrow_gap,
)


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def _aggregate(reports: list[dict]) -> dict:
    episodes = sum(int(report["episodes"]) for report in reports)
    outcome_episodes = sum(int(report["n"]) for report in reports)
    if episodes <= 0 or outcome_episodes <= 0:
        raise RuntimeError("narrow suite produced no completed episodes")

    def weighted_outcome(key: str) -> float:
        return sum(
            float(report[key]) * int(report["n"])
            for report in reports
        ) / outcome_episodes

    aggregate = {
        "n": outcome_episodes,
        "episodes": episodes,
        "sr": weighted_outcome("sr"),
        "cr": weighted_outcome("cr"),
        "to": weighted_outcome("to"),
        "crossed": int(sum(report["crossed"] for report in reports)),
        "direct_crossed": int(
            sum(report["direct_crossed"] for report in reports)
        ),
    }
    aggregate["crossing_rate"] = aggregate["crossed"] / episodes
    aggregate["direct_crossing_rate"] = (
        aggregate["direct_crossed"] / episodes
    )
    for key in (
        "path_length_ratio_p95",
        "first_cross_time_s_p95",
        "max_pre_cross_abs_y_m_p95",
        "backtrack_distance_m_p95",
    ):
        aggregate[f"worst_seed_{key}"] = max(
            float(report[key]) for report in reports
        )
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--seeds",
        type=_parse_csv_ints,
        default=(404, 505, 606, 707, 808),
    )
    parser.add_argument(
        "--mode",
        choices=("legacy", "sealed"),
        default="legacy",
        help="legacy=Gate5b 10m 邊界死角壓測（歷史行為，預設）；"
             "sealed=Gate5a 牆隨場地封死到外牆，純測直穿能力",
    )
    parser.add_argument(
        "--arena-size",
        type=float,
        default=10.0,
        help="場地邊長 m（預設 10 對齊歷史 Gate5b；Gate5a 應改用訓練場地尺寸）",
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=5,
        help="課程階段（預設 5 對齊歷史 Gate5b；Gate5a 應用該 checkpoint 的訓練階段，"
             "例如 SA7 用 7）",
    )
    args = parser.parse_args()

    if args.mode == "sealed" and args.arena_size == 10.0:
        print(
            "[NARROW-PATH-SUITE] ⚠️ sealed 模式仍在用預設的 10 m 場地 —— "
            "Gate5a 應使用該 checkpoint 的訓練場地尺寸（SA7=13、SA8=12），"
            "否則量到的仍是場地錯配下的行為。",
            flush=True,
        )

    gate_name = "Gate5a 純直穿" if args.mode == "sealed" else "Gate5b 10m邊界死角壓測"
    side_opening = (
        0.0 if args.mode == "sealed" else max(0.0, 0.5 * args.arena_size - 5.0)
    )
    print(
        f"[NARROW-PATH-SUITE] {gate_name}｜arena={args.arena_size:.1f}m "
        f"牆端側口={side_opening:.2f}m",
        flush=True,
    )
    if side_opening > 1e-6:
        print(
            "[NARROW-PATH-SUITE] ⚠️ 側口 > 0：機器人可繞過中央缺口，"
            "crossing 只代表越過牆的 x 平面。純直穿請用 --mode sealed。",
            flush=True,
        )

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    reports: list[dict] = []
    for seed in args.seeds:
        log_path = output / f"narrow_s{seed}.log"
        print(f"[NARROW-PATH-SUITE] seed={seed}", flush=True)
        _run_play(
            checkpoint,
            log_path,
            num_envs=args.num_envs,
            steps=args.steps,
            extra=[
                "--stage",
                str(args.stage),
                "--arena_size",
                f"{args.arena_size:g}",
                "--narrow_gap_mode",
                args.mode,
                "--num_static_obs",
                "0",
                "--num_dynamic_obs",
                "0",
                "--obs_near_goal_count",
                "0",
                "--narrow_gap_eval",
                "--narrow_gap_width",
                "1.2",
                "--narrow_gap_yaw_limit_deg",
                "10",
                "--seed",
                str(seed),
            ],
        )
        report = parse_narrow_gap(str(log_path))
        if report is None:
            raise RuntimeError(f"missing narrow metrics: {log_path}")
        report["seed"] = seed
        reports.append(report)
        print(
            "[NARROW-PATH-SUITE] "
            f"seed={seed} SR={report['sr']:.3f} CR={report['cr']:.3f} "
            f"crossing={report['crossing_rate']:.3f} "
            f"direct={report['direct_crossing_rate']:.3f} "
            f"path_p95={report['path_length_ratio_p95']:.3f} "
            f"cross_t_p95={report['first_cross_time_s_p95']:.2f}s "
            f"pre_y_p95={report['max_pre_cross_abs_y_m_p95']:.2f}m "
            f"backtrack_p95={report['backtrack_distance_m_p95']:.2f}m",
            flush=True,
        )

    aggregate = _aggregate(reports)
    checks = narrow_gap_checks(aggregate)
    passed = bool(checks) and all(ok for ok, _ in checks.values())
    suite = {
        "checkpoint": str(checkpoint),
        "seeds": list(args.seeds),
        "thresholds": NARROW_DEPLOY_THRESH,
        "aggregate": aggregate,
        "checks": {
            name: {"pass": ok, "detail": detail}
            for name, (ok, detail) in checks.items()
        },
        "pass": passed,
        "per_seed": reports,
    }
    report_path = output / "narrow_path_suite.json"
    report_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[NARROW-PATH-SUITE] "
        f"ALL={'PASS' if passed else 'FAIL'} n={aggregate['episodes']} "
        f"SR={aggregate['sr']:.3f} CR={aggregate['cr']:.3f} "
        f"crossing={aggregate['crossing_rate']:.3f} "
        f"direct={aggregate['direct_crossing_rate']:.3f} "
        f"report={report_path}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
