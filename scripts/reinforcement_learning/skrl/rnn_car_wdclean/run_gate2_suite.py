"""Run the fixed three-seed Stage-5 Gate2 deployment evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixed_actuator_eval import (  # noqa: E402
    ACTUATOR_PROFILES,
    fixed_actuator_cli_args,
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import STAGE_THRESH, agg_seeds  # noqa: E402


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def _aggregate_jitter(reports: list[dict]) -> dict:
    samples = sum(int(report["samples"]) for report in reports)
    if samples <= 0:
        raise RuntimeError("jitter audit produced no samples")

    def weighted(key: str) -> float:
        return sum(
            float(report[key]) * int(report["samples"])
            for report in reports
        ) / samples

    omega_squared_mean = weighted("omega_squared_mean_rad2_s2")
    return {
        "samples": samples,
        "omega_rms_rad_s": omega_squared_mean ** 0.5,
        "full_steer_fraction": weighted("full_steer_fraction"),
        "ratio_flip_rate_mean": weighted("ratio_flip_rate_mean"),
        "worst_seed_ratio_flip_rate_p95": max(
            float(report["ratio_flip_rate_p95"]) for report in reports
        ),
        "worst_seed_omega_abs_std_p95_rad_s": max(
            float(report["omega_abs_std_p95_rad_s"]) for report in reports
        ),
        "worst_seed_dominant_frequency_p95_hz": max(
            float(report["dominant_frequency_p95_hz"])
            for report in reports
        ),
        "worst_seed_max_same_sign_turn_p95_s": max(
            float(report["max_same_sign_turn_p95_s"])
            for report in reports
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--seeds", type=_parse_csv_ints, default=(101, 202, 303)
    )
    parser.add_argument(
        "--actuator-delay-steps",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help=(
            "fixed action delay for deployment evaluation: "
            "0/1/2 steps = 0/200/400 ms. Velocity scale remains U(0.9,1.1) "
            "per episode and motor lag remains alpha=0.3."
        ),
    )
    parser.add_argument(
        "--actuator-profile",
        choices=ACTUATOR_PROFILES,
        default="bridge",
        help=(
            "bridge keeps historical U(0.9,1.1)+alpha=0.3; "
            "sa1_delay_only keeps scale/lag neutral"
        ),
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    actuator_args = fixed_actuator_cli_args(
        args.actuator_delay_steps, args.actuator_profile
    )

    logs: list[str] = []
    jitter_reports: list[dict] = []
    for seed in args.seeds:
        log_path = output / f"det_s{seed}.log"
        jitter_path = output / f"det_s{seed}_jitter.json"
        print(
            f"[GATE2-SUITE] seed={seed} "
            f"delay_steps={args.actuator_delay_steps} "
            f"actuator_profile={args.actuator_profile}",
            flush=True,
        )
        _run_play(
            checkpoint,
            log_path,
            num_envs=args.num_envs,
            steps=args.steps,
            extra=[
                "--stage",
                "5",
                "--arena_size",
                "15",
                "--num_static_obs",
                "10",
                "--num_dynamic_obs",
                "3",
                "--obs_near_goal_count",
                "0",
                "--seed",
                str(seed),
                "--jitter_eval",
                "--jitter_eval_output",
                str(jitter_path),
                *actuator_args,
            ],
        )
        verify_fixed_actuator_runtime(
            log_path, args.actuator_delay_steps, args.actuator_profile
        )
        logs.append(str(log_path))
        jitter_reports.append(
            json.loads(jitter_path.read_text(encoding="utf-8"))
        )

    aggregate = agg_seeds(logs)
    jitter_aggregate = _aggregate_jitter(jitter_reports)
    thresholds = STAGE_THRESH[5]
    passed = bool(
        aggregate["sr"] is not None
        and aggregate["sr"] >= thresholds["det_sr"]
        and aggregate["cr"] <= thresholds["det_cr"]
        and aggregate["to"] <= thresholds["det_to"]
    )
    report = {
        "checkpoint": str(checkpoint),
        "seeds": list(args.seeds),
        "actuator_eval": fixed_actuator_metadata(
            args.actuator_delay_steps, args.actuator_profile
        ),
        "thresholds": {
            "success_rate_min": thresholds["det_sr"],
            "collision_rate_max": thresholds["det_cr"],
            "timeout_rate_max": thresholds["det_to"],
        },
        "aggregate": aggregate,
        "action_stability": {
            "aggregate": jitter_aggregate,
            "per_seed": jitter_reports,
        },
        "pass": passed,
        "logs": logs,
    }
    report_path = output / "gate2_suite.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        "[GATE2-SUITE] "
        f"ALL={'PASS' if passed else 'FAIL'} n={aggregate['n']} "
        f"SR={aggregate['sr']:.3f} CR={aggregate['cr']:.3f} "
        f"TO={aggregate['to']:.3f} report={report_path}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
