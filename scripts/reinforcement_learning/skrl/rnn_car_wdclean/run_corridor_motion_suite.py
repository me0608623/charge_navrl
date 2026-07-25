"""Run fixed-seed deployment-corridor gates for every dynamic motion family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sa5_joint_retention_gates import _run_play  # noqa: E402


MODES = ("lateral", "longitudinal", "random_2d", "mixed")
SR_MIN = 0.90
CR_MAX = 0.10
TO_MAX = 0.05


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return values


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = set(modes) - set(MODES)
    if not modes or invalid:
        raise argparse.ArgumentTypeError(
            f"modes must be a subset of {MODES}; invalid={sorted(invalid)}"
        )
    return modes


def _aggregate(reports: list[dict]) -> dict:
    episodes = sum(int(report["episodes"]) for report in reports)
    if episodes <= 0:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "timeout_rate": 0.0,
        }

    def weighted(key: str) -> float:
        return sum(
            float(report[key]) * int(report["episodes"])
            for report in reports
        ) / episodes

    return {
        "episodes": episodes,
        "success_rate": weighted("success_rate"),
        "collision_rate": weighted("collision_rate"),
        "timeout_rate": weighted("timeout_rate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--seeds", type=_parse_csv_ints, default=(515, 616, 717)
    )
    parser.add_argument("--modes", type=_parse_modes, default=MODES)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    suite: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "seeds": list(args.seeds),
        "thresholds": {
            "success_rate_min": SR_MIN,
            "collision_rate_max": CR_MAX,
            "timeout_rate_max": TO_MAX,
        },
        "modes": {},
    }
    all_pass = True
    for mode in args.modes:
        reports: list[dict] = []
        for seed in args.seeds:
            stem = f"{mode}_s{seed}"
            log_path = output / f"{stem}.log"
            json_path = output / f"{stem}.json"
            print(
                f"[CORRIDOR-SUITE] mode={mode} seed={seed}",
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
                    "12",
                    "--obs_near_goal_count",
                    "0",
                    "--long_corridor_eval",
                    "--long_corridor_static_obstacles",
                    "4",
                    "--long_corridor_dynamic_obstacles",
                    "2",
                    "--long_corridor_motion_mode",
                    mode,
                    "--long_corridor_output",
                    str(json_path),
                    "--seed",
                    str(seed),
                ],
            )
            report = json.loads(json_path.read_text(encoding="utf-8"))
            report["seed"] = seed
            reports.append(report)

        aggregate = _aggregate(reports)
        structural_pass = all(
            bool(report.get("geometry_pass"))
            and bool(report.get("movement_pass"))
            and bool(report.get("motion_mode_pass"))
            and bool(report.get("obstacle_mix_pass"))
            and bool(report.get("goal_alignment_pass"))
            and int(report.get("constructive_unsolvable_count", -1)) == 0
            for report in reports
        )
        performance_pass = bool(
            aggregate["success_rate"] >= SR_MIN
            and aggregate["collision_rate"] <= CR_MAX
            and aggregate["timeout_rate"] <= TO_MAX
        )
        mode_pass = structural_pass and performance_pass
        suite["modes"][mode] = {
            "aggregate": aggregate,
            "structural_pass": structural_pass,
            "performance_pass": performance_pass,
            "pass": mode_pass,
            "seeds": reports,
        }
        all_pass &= mode_pass
        print(
            "[CORRIDOR-SUITE] "
            f"{mode}={'PASS' if mode_pass else 'FAIL'} "
            f"n={aggregate['episodes']} "
            f"SR={aggregate['success_rate']:.3f} "
            f"CR={aggregate['collision_rate']:.3f} "
            f"TO={aggregate['timeout_rate']:.3f} "
            f"structural={structural_pass}",
            flush=True,
        )

    suite["all_modes_pass"] = all_pass
    report_path = output / "corridor_motion_suite.json"
    report_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[CORRIDOR-SUITE] ALL={'PASS' if all_pass else 'FAIL'} "
        f"report={report_path}",
        flush=True,
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
