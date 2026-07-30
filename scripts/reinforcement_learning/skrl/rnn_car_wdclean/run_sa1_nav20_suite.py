"""Run the stage-aligned SA1 20 x 20 m navigation evaluation.

``clean`` is the blocking SA1 gate: open 20 x 20 m room, random spawn/yaw,
one stationary random goal, and no obstacles.

``native`` is advisory: the same setup with two sparse static obstacles.
It measures basic bypass behaviour but does not turn deployment-grade obstacle
avoidance into an SA1 promotion requirement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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


NAV20_MODES = ("clean", "native")
ARENA_SIZE_M = 20.0
GOAL_DISTANCE_RANGE_M = (2.0, 9.0)
EPISODE_LENGTH_S = 60.0
STATIC_OBSTACLES = {
    "clean": 0,
    "native": 2,
}
BLOCKING = {
    "clean": True,
    "native": False,
}

# "Nearly always navigates" retains the historical SA1 deterministic target.
CLEAN_THRESHOLDS = {
    "sr": STAGE_THRESH[1]["det_sr"],
    "cr": STAGE_THRESH[1]["det_cr"],
    "to": STAGE_THRESH[1]["det_to"],
}

# Advisory only. These values label a result; they never block SA1 promotion.
NATIVE_ADVISORY_THRESHOLDS = {
    "sr": 0.90,
    "cr": 0.08,
    "to": 0.03,
}


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


def thresholds_for(mode: str) -> dict[str, float]:
    if mode == "clean":
        return dict(CLEAN_THRESHOLDS)
    if mode == "native":
        return dict(NATIVE_ADVISORY_THRESHOLDS)
    raise ValueError(f"unknown Nav20 mode {mode!r}")


def build_scene_args(
    mode: str,
    *,
    seed: int,
    jitter_path: Path,
    actuator_args: list[str],
) -> list[str]:
    if mode not in NAV20_MODES:
        raise ValueError(f"unknown Nav20 mode {mode!r}")
    return [
        "--stage",
        "1",
        "--arena_size",
        str(ARENA_SIZE_M),
        "--num_static_obs",
        str(STATIC_OBSTACLES[mode]),
        "--num_dynamic_obs",
        "0",
        "--num_walls",
        "0",
        "--goal_distance_min",
        str(GOAL_DISTANCE_RANGE_M[0]),
        "--goal_distance_max",
        str(GOAL_DISTANCE_RANGE_M[1]),
        "--episode_length_s",
        str(EPISODE_LENGTH_S),
        "--obs_near_goal_count",
        "0",
        "--obstacle_behavior",
        "static",
        "--seed",
        str(seed),
        "--jitter_eval",
        "--jitter_eval_output",
        str(jitter_path),
        *actuator_args,
    ]


def verify_nav20_runtime(log_path: Path, mode: str) -> dict:
    """Fail closed when play did not build the requested Nav20 scene."""
    if mode not in NAV20_MODES:
        raise ValueError(f"unknown Nav20 mode {mode!r}")
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    expected_static = STATIC_OBSTACLES[mode]
    required = (
        "[PLAY] 場景配置",
        f"目標數=1  靜態障礙={expected_static}  動態障礙=0",
        "牆壁=0~0",
        "目標距離=2.0~9.0m",
        "episode=60.0s",
        "[PLAY] Arena 縮放: 20×20m → 20×20m",
        "stage=1 name=SA1_nav_bootstrap",
        "[SIM2REAL][VLP16-ablation] mode=full",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            f"Nav20 {mode} runtime contract incomplete in {log_path}: "
            f"missing {missing}"
        )

    match = re.search(
        r"目標數=(\d+)\s+靜態障礙=(\d+)\s+動態障礙=(\d+)",
        text,
    )
    if match is None:
        raise RuntimeError(f"Nav20 {mode} scene counts are not parseable")
    counts = tuple(int(value) for value in match.groups())
    expected_counts = (1, expected_static, 0)
    if counts != expected_counts:
        raise RuntimeError(
            f"Nav20 {mode} scene counts {counts}, expected {expected_counts}"
        )

    return {
        "mode": mode,
        "blocking": BLOCKING[mode],
        "arena_size_m": ARENA_SIZE_M,
        "goals": 1,
        "goal_movement": False,
        "goal_distance_range_m": list(GOAL_DISTANCE_RANGE_M),
        "static_obstacles": expected_static,
        "dynamic_obstacles": 0,
        "internal_walls": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=NAV20_MODES, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--seeds",
        type=_parse_csv_ints,
        default=(515, 616, 717),
    )
    parser.add_argument(
        "--actuator-delay-steps",
        type=int,
        choices=(0, 1, 2),
        default=None,
    )
    parser.add_argument(
        "--actuator-profile",
        choices=ACTUATOR_PROFILES,
        default="sa1_delay_only",
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    actuator_args = fixed_actuator_cli_args(
        args.actuator_delay_steps,
        args.actuator_profile,
    )

    logs: list[str] = []
    jitter_reports: list[dict] = []
    scene_contract: dict | None = None
    for seed in args.seeds:
        log_path = output / f"{args.mode}_s{seed}.log"
        jitter_path = output / f"{args.mode}_s{seed}_jitter.json"
        print(
            f"[NAV20-SUITE] mode={args.mode} seed={seed} "
            f"delay_steps={args.actuator_delay_steps} "
            f"actuator_profile={args.actuator_profile}",
            flush=True,
        )
        _run_play(
            checkpoint,
            log_path,
            num_envs=args.num_envs,
            steps=args.steps,
            extra=build_scene_args(
                args.mode,
                seed=seed,
                jitter_path=jitter_path,
                actuator_args=actuator_args,
            ),
        )
        verify_fixed_actuator_runtime(
            log_path,
            args.actuator_delay_steps,
            args.actuator_profile,
        )
        current_contract = verify_nav20_runtime(log_path, args.mode)
        if scene_contract is not None and current_contract != scene_contract:
            raise RuntimeError("Nav20 scene contract changed across seeds")
        scene_contract = current_contract
        logs.append(str(log_path))
        jitter_reports.append(
            json.loads(jitter_path.read_text(encoding="utf-8"))
        )

    aggregate = agg_seeds(logs)
    thresholds = thresholds_for(args.mode)
    threshold_pass = bool(
        aggregate["sr"] is not None
        and aggregate["sr"] >= thresholds["sr"]
        and aggregate["cr"] <= thresholds["cr"]
        and aggregate["to"] <= thresholds["to"]
    )
    report = {
        "checkpoint": str(checkpoint),
        "mode": args.mode,
        "blocking": BLOCKING[args.mode],
        "seeds": list(args.seeds),
        "scene_contract": scene_contract,
        "actuator_eval": fixed_actuator_metadata(
            args.actuator_delay_steps,
            args.actuator_profile,
        ),
        "thresholds": {
            "success_rate_min": thresholds["sr"],
            "collision_rate_max": thresholds["cr"],
            "timeout_rate_max": thresholds["to"],
        },
        "aggregate": aggregate,
        "action_stability": {
            "aggregate": _aggregate_jitter(jitter_reports),
            "per_seed": jitter_reports,
        },
        "pass": threshold_pass,
        "logs": logs,
    }
    report_path = output / f"nav20_{args.mode}_suite.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    verdict = "PASS" if threshold_pass else "FAIL"
    role = "BLOCKING" if BLOCKING[args.mode] else "ADVISORY"
    print(
        f"[NAV20-SUITE] {args.mode}={verdict} role={role} "
        f"n={aggregate['n']} SR={aggregate['sr']:.3f} "
        f"CR={aggregate['cr']:.3f} TO={aggregate['to']:.3f} "
        f"report={report_path}",
        flush=True,
    )
    return 0 if threshold_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
