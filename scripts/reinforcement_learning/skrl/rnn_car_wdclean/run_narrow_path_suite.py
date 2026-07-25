"""Run fixed-seed Gate5 with direct-path diagnostics."""

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
    args = parser.parse_args()

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
                "5",
                "--arena_size",
                "10",
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
