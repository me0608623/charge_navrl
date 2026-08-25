"""Rerun the fixed SA4-D4 cell with realized winner-ray LiDAR angles.

This D4-r2 suite preserves the original checkpoint, scene, seed, actuator and
rollout budget. It runs a fresh identity baseline followed by the corrected
``geometry_feasible_argmin`` arm. On every active intervention frame the
corrected selector also evaluates the legacy 5-degree bin-center geometry as a
paired shadow; only the realized-angle grid may modify actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_d4_geometry_suite as d4_v1  # noqa: E402
from d3_yield_recorder import shield_protocol as baseline_protocol  # noqa: E402
from d4_geometry_selector import (  # noqa: E402
    ARGMIN_MODE,
    geometry_argmin_selector_protocol,
)


ARMS = ("baseline", ARGMIN_MODE)
OBS_FUNCTIONS = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py"
)
D7_TRACE = HERE / "d7_lidar_residual_origin.py"
NET_RECOVERY_MIN_FRACTION = 0.05
NET_RECOVERY_MIN_FRAMES = 100


def source_fingerprint() -> dict[str, str]:
    files = {
        **{
            f"d4_v1_{name}": value
            for name, value in d4_v1.source_fingerprint().items()
        },
        "runner_r2": d4_v1.sha256_of(Path(__file__).resolve()),
        "observation_trace": d4_v1.sha256_of(OBS_FUNCTIONS),
        "trace_matcher": d4_v1.sha256_of(D7_TRACE),
    }
    return files


def _validate_argmin_ledger(result: dict) -> dict:
    runtime = result["selector_runtime"]
    problems: list[str] = []
    if runtime.get("mode") != ARGMIN_MODE:
        problems.append("runtime mode mismatch")
    if runtime.get("lidar_angle_source") != "winning_raw_ray":
        problems.append("selector did not use winning raw-ray angles")

    active = int(runtime.get("active_environment_frames", -1))
    paired = int(runtime.get("paired_center_shadow_active_frames", -1))
    actual_feasible = int(runtime.get("jointly_feasible_active_frames", -1))
    center_feasible = int(
        runtime.get("center_jointly_feasible_active_frames", -1)
    )
    both = int(runtime.get("both_jointly_feasible_active_frames", -1))
    argmin_only = int(runtime.get("argmin_only_feasible_active_frames", -1))
    center_only = int(runtime.get("center_only_feasible_active_frames", -1))
    neither = int(runtime.get("both_no_feasible_active_frames", -1))
    counts = (active, paired, actual_feasible, center_feasible, both,
              argmin_only, center_only, neither)
    if min(counts) < 0:
        problems.append("paired feasibility ledger field missing")
    else:
        if paired != active:
            problems.append("paired center shadow does not cover every active frame")
        if both + argmin_only + center_only + neither != active:
            problems.append("paired feasibility quadrants do not reconcile")
        if actual_feasible != both + argmin_only:
            problems.append("actual-angle feasible count does not reconcile")
        if center_feasible != both + center_only:
            problems.append("center-angle feasible count does not reconcile")
    if problems:
        raise RuntimeError("invalid D4-r2 argmin ledger: " + "; ".join(problems))

    denominator = max(active, 1)
    actual_fraction = actual_feasible / denominator
    center_fraction = center_feasible / denominator
    net_frames = argmin_only - center_only
    net_fraction = net_frames / denominator
    clear_recovery = (
        net_frames >= NET_RECOVERY_MIN_FRAMES
        and net_fraction >= NET_RECOVERY_MIN_FRACTION
    )
    return {
        "active_frames": active,
        "actual_angle_feasible_fraction": actual_fraction,
        "center_angle_feasible_fraction": center_fraction,
        "net_feasible_recovery_frames": net_frames,
        "net_feasible_recovery_fraction": net_fraction,
        "clear_candidate_recovery": clear_recovery,
        "descriptive_rule": {
            "net_fraction_min": NET_RECOVERY_MIN_FRACTION,
            "net_frames_min": NET_RECOVERY_MIN_FRAMES,
            "inferential_claim": False,
        },
    }


def _render(arms: list[dict], recovery: dict) -> str:
    base = arms[0]["metrics"]
    fixed = arms[1]["metrics"]
    return "\n".join(
        [
            "# SA4-D4-r2 realized winner-angle comparison",
            "",
            "> Fresh baseline and corrected argmin-angle selector; fixed c6400, "
            "SA4 lateral, seed 818, d1=200 ms, 64 env x 2500 steps.",
            "",
            "| Arm | episodes | SR | obstacle CR | wall CR | total CR | TO |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| baseline | {base['n']:,} | {base['sr']:.2%} | "
                f"{base['obstacle_cr']:.2%} | {base['wall_cr']:.2%} | "
                f"{base['cr']:.2%} | {base['to']:.2%} |"
            ),
            (
                f"| {ARGMIN_MODE} | {fixed['n']:,} | {fixed['sr']:.2%} | "
                f"{fixed['obstacle_cr']:.2%} | {fixed['wall_cr']:.2%} | "
                f"{fixed['cr']:.2%} | {fixed['to']:.2%} |"
            ),
            "",
            "## Paired candidate geometry",
            "",
            (
                f"- Actual winner-angle feasible: "
                f"{recovery['actual_angle_feasible_fraction']:.2%}"
            ),
            (
                f"- Legacy bin-center feasible: "
                f"{recovery['center_angle_feasible_fraction']:.2%}"
            ),
            (
                f"- Net paired recovery: "
                f"{recovery['net_feasible_recovery_fraction']:+.2%} "
                f"({recovery['net_feasible_recovery_frames']:+,} frames)"
            ),
            (
                f"- Clear descriptive recovery: "
                f"{recovery['clear_candidate_recovery']}"
            ),
            "",
            "Single checkpoint and evaluator seed: diagnostic evidence only.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = d4_v1.sha256_of(checkpoint)
    if checkpoint_sha256 != args.expect_checkpoint_sha256:
        raise RuntimeError(
            "checkpoint hash mismatch: "
            f"expected {args.expect_checkpoint_sha256}, got {checkpoint_sha256}"
        )

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        output / "PROTOCOL.json",
        output / "suite_manifest.json",
        output / "comparison.json",
        output / "COMPARISON.md",
    ]
    d4_v1._require_new_targets(targets)

    planned_fingerprint = source_fingerprint()
    protocol = {
        "schema": "sa4_d4_argmin_suite_preregistration/v1",
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "purpose": (
            "isolate the D4 5-degree-center localization artifact by using "
            "the realized raw-ray angle that won each 72-bin amin reduction"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "fixed_cell": {
            "geometry_stage": d4_v1.baseline.GEOMETRY_STAGE,
            "scenario": d4_v1.baseline.SCENARIO,
            "seed": d4_v1.baseline.SEED,
            "delay_steps": d4_v1.baseline.DELAY_STEPS,
            "delay_ms": 200,
            "actuator_profile": d4_v1.baseline.ACTUATOR_PROFILE,
            "num_envs": d4_v1.baseline.NUM_ENVS,
            "steps": d4_v1.baseline.ROLLOUT_STEPS,
        },
        "arms_in_fixed_order": list(ARMS),
        "baseline_identity_protocol": baseline_protocol(),
        "argmin_selector_protocol": geometry_argmin_selector_protocol(),
        "noise_model_change_in_this_suite": None,
        "clear_candidate_recovery_rule": {
            "paired_net_feasible_fraction_min": NET_RECOVERY_MIN_FRACTION,
            "paired_net_feasible_frames_min": NET_RECOVERY_MIN_FRAMES,
            "descriptive_only": True,
        },
        "source_fingerprint": planned_fingerprint,
    }
    targets[0].write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[SA4-D4-r2] protocol frozen: {targets[0]}", flush=True)

    values = d4_v1.baseline.sa4.scene_values()
    arms = [
        d4_v1._run_arm(mode, checkpoint, output, values) for mode in ARMS
    ]
    recovery = _validate_argmin_ledger(arms[1])
    final_fingerprint = source_fingerprint()
    if final_fingerprint != planned_fingerprint:
        raise RuntimeError("D4-r2 source changed during the comparison")

    comparison = {
        "schema": "sa4_d4_argmin_comparison/v1",
        "checkpoint_sha256": checkpoint_sha256,
        "protocol_sha256": geometry_argmin_selector_protocol()["sha256"],
        "arms": arms,
        "delta_argmin_minus_baseline": {
            key: arms[1]["metrics"][key] - arms[0]["metrics"][key]
            for key in ("sr", "cr", "obstacle_cr", "wall_cr", "to")
        },
        "paired_candidate_recovery": recovery,
    }
    targets[2].write_text(
        json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8"
    )
    targets[3].write_text(_render(arms, recovery), encoding="utf-8")
    manifest = {
        "schema": "sa4_d4_argmin_suite/v1",
        "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": str(targets[0]),
        "protocol_sha256": geometry_argmin_selector_protocol()["sha256"],
        "source_fingerprint": final_fingerprint,
        "arms": arms,
        "paired_candidate_recovery": recovery,
        "comparison_json": str(targets[2]),
        "comparison_markdown": str(targets[3]),
        "training_started": False,
        "next_stage_started": False,
    }
    targets[1].write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[SA4-D4-r2] COMPLETE comparison={targets[3]}", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
