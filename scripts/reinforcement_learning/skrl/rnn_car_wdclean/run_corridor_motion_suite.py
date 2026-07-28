"""Run fixed-seed deployment-corridor gates for every dynamic motion family."""

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


#: Default suite. ``mixed`` is the legacy balanced-deal gate and stays the
#: default so historical numbers remain comparable; ``mixed_iid`` is the
#: unbiased per-obstacle variant and must be requested explicitly via --modes.
MODES = ("lateral", "longitudinal", "random_2d", "mixed")
SELECTABLE_MODES = MODES + ("mixed_iid",)

#: Named suite profiles. The bare defaults stay the legacy patrol gate so no
#: historical invocation changes meaning; the deployment profile must be
#: requested explicitly, which prevents accidentally grading a candidate on
#: the wrong kinematics in either direction.
PROFILES = {
    "legacy_patrol_v1": {
        "modes": MODES,
        "random_2d_kinematics": "patrol",
    },
    # Deployment definition per the wander ruling: random_2d runs the bounded
    # random walk, and mixed is the unbiased per-obstacle mixture whose random
    # legs are also wander.
    "deployment_wander_v1": {
        "modes": ("lateral", "longitudinal", "random_2d", "mixed_iid"),
        "random_2d_kinematics": "wander",
    },
}


def resolve_profile(name, modes, kinematics):
    """Resolve a named profile into (modes, kinematics, profile_complete).

    A profile is a frozen contract, not a set of defaults:

    * A kinematics override that contradicts the profile is an error — a JSON
      stamped ``deployment_wander_v1`` must never contain patrol episodes.
    * ``--modes`` may select a *subset* of the profile's modes (sentinel
      runs); anything outside the profile is an error.
    * ``profile_complete`` is True only when every profile mode ran, so a
      subset run can never be mistaken for a full deployment JOINT PASS.

    Without a profile the caller's values pass through untouched and
    ``profile_complete`` is None (the concept does not apply).
    """
    if name is None:
        return modes, kinematics, None
    profile = PROFILES[name]
    if (
        kinematics is not None
        and kinematics != profile["random_2d_kinematics"]
    ):
        raise ValueError(
            f"profile {name!r} pins random_2d_kinematics="
            f"{profile['random_2d_kinematics']!r}; explicit "
            f"--random-2d-kinematics {kinematics!r} contradicts it. Drop the "
            "flag or drop the profile."
        )
    if modes is not None:
        invalid = set(modes) - set(profile["modes"])
        if invalid:
            raise ValueError(
                f"modes {sorted(invalid)} are not part of profile {name!r} "
                f"(allowed: {profile['modes']})"
            )
        resolved_modes = tuple(modes)
    else:
        resolved_modes = tuple(profile["modes"])
    complete = set(resolved_modes) == set(profile["modes"])
    return resolved_modes, profile["random_2d_kinematics"], complete


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
    invalid = set(modes) - set(SELECTABLE_MODES)
    if not modes or invalid:
        raise argparse.ArgumentTypeError(
            f"modes must be a subset of {SELECTABLE_MODES}; invalid={sorted(invalid)}"
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


def summarize_verdict(all_pass, profile_complete):
    """Fold selected-mode results and profile coverage into one verdict.

    PASS semantics must not outrun coverage: a sentinel running a subset of a
    profile may pass *its selected modes*, but its output must never be
    readable — by a human or a supervisor grepping ``ALL=PASS`` — as a full
    deployment JOINT PASS. The banner for a subset run therefore uses
    ``SELECTED=``/``JOINT=N/A`` and never contains the ``ALL=`` token.
    """
    joint = (
        (bool(all_pass) and profile_complete)
        if profile_complete is not None
        else None
    )
    verdict = {
        "all_selected_modes_pass": bool(all_pass),
        "profile_joint_pass": joint,
        # Back-compat key: True only when the result may be read as complete.
        # A deliberate subset (profile_complete is False) can never set it.
        "all_modes_pass": bool(all_pass) and profile_complete is not False,
    }
    if profile_complete is False:
        banner = (
            f"SELECTED={'PASS' if all_pass else 'FAIL'} "
            "PROFILE_COMPLETE=false JOINT=N/A"
        )
    elif profile_complete is True:
        banner = (
            f"ALL={'PASS' if all_pass else 'FAIL'} "
            f"JOINT={'PASS' if joint else 'FAIL'}"
        )
    else:
        banner = f"ALL={'PASS' if all_pass else 'FAIL'}"
    verdict["banner"] = banner
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--seeds", type=_parse_csv_ints, default=(515, 616, 717)
    )
    parser.add_argument("--modes", type=_parse_modes, default=None)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default=None,
        help=(
            "named gate profile; deployment_wander_v1 = lateral/longitudinal/"
            "random_2d(wander)/mixed_iid(wander). Explicit --modes or "
            "--random-2d-kinematics override the profile field-by-field."
        ),
    )
    parser.add_argument(
        "--pause-mode",
        choices=("default", "zero"),
        default="default",
        help=(
            "eval-only patrol pause override. 'default' keeps the stock 0-5 step "
            "behaviour used by training and every historical gate."
        ),
    )
    parser.add_argument(
        "--random-2d-kinematics",
        choices=("patrol", "wander"),
        default=None,
        help="random_2d motion implementation; 'patrol' is the frozen default",
    )
    parser.add_argument(
        "--phase-audit",
        action="store_true",
        default=False,
        help="emit per-mode motion-phase attribution JSON alongside each run",
    )
    parser.add_argument(
        "--actuator-delay-steps",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help=(
            "fixed action delay: 0/1/2 steps = 0/200/400 ms; velocity scale "
            "remains U(0.9,1.1) per episode and motor lag remains alpha=0.3"
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

    try:
        modes, kinematics, profile_complete = resolve_profile(
            args.profile, args.modes, args.random_2d_kinematics
        )
    except ValueError as error:
        parser.error(str(error))
    if modes is None:
        modes = MODES
    if kinematics is None:
        kinematics = "patrol"
    args.modes = modes
    args.random_2d_kinematics = kinematics

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    actuator_args = fixed_actuator_cli_args(
        args.actuator_delay_steps, args.actuator_profile
    )
    actuator_metadata = fixed_actuator_metadata(
        args.actuator_delay_steps, args.actuator_profile
    )

    suite: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "seeds": list(args.seeds),
        "actuator_eval": actuator_metadata,
        "pause_mode": args.pause_mode,
        "profile": args.profile,
        # False = deliberate subset run (sentinel); such a run must never be
        # read as a full deployment JOINT PASS. None = no profile requested.
        "profile_complete": profile_complete,
        "random_2d_kinematics": args.random_2d_kinematics,
        "phase_audit": bool(args.phase_audit),
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
                    "--long_corridor_pause_mode",
                    args.pause_mode,
                    "--long_corridor_random_2d_kinematics",
                    args.random_2d_kinematics,
                    *(
                        [
                            "--long_corridor_phase_audit",
                            "--long_corridor_phase_audit_output",
                            str(output / f"{stem}_phase.json"),
                        ]
                        if args.phase_audit
                        else []
                    ),
                    "--seed",
                    str(seed),
                    *actuator_args,
                ],
            )
            verify_fixed_actuator_runtime(
                log_path,
                args.actuator_delay_steps,
                args.actuator_profile,
            )
            report = json.loads(json_path.read_text(encoding="utf-8"))
            report["seed"] = seed
            report["actuator_eval"] = actuator_metadata
            json_path.write_text(
                json.dumps(report, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            reports.append(report)

        aggregate = _aggregate(reports)
        structural_pass = all(
            bool(report.get("geometry_pass"))
            and bool(report.get("movement_pass"))
            and bool(report.get("motion_mode_pass"))
            and bool(report.get("obstacle_mix_pass"))
            and bool(report.get("goal_alignment_pass"))
            and int(report.get("constructive_unsolvable_count", -1)) == 0
            and bool(report.get("penetration_pass"))
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

    verdict = summarize_verdict(all_pass, profile_complete)
    suite["all_selected_modes_pass"] = verdict["all_selected_modes_pass"]
    suite["profile_joint_pass"] = verdict["profile_joint_pass"]
    suite["all_modes_pass"] = verdict["all_modes_pass"]
    report_path = output / "corridor_motion_suite.json"
    report_path.write_text(
        json.dumps(suite, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        f"[CORRIDOR-SUITE] {verdict['banner']} report={report_path}",
        flush=True,
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
