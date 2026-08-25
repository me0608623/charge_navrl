"""Run one frozen SA4-v3 c550/c600 retention cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa3_gate_bc as base  # noqa: E402
import run_sa3_v3_retention_screen_cell as retention_helpers  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import run_sa4_v3_checkpoint_screen_cell as corridor_helpers  # noqa: E402
import sa4_v3_c550_c600_retention as protocol  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_cli_args,
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


# Reuse the already-tested parsers and runtime verifiers with this frozen
# protocol. No historical evaluator source is edited.
retention_helpers.protocol = protocol
corridor_helpers.protocol = protocol


def scene_values() -> dict:
    return corridor_helpers.scene_values()


def build_scene_args(scenario: dict, corridor_path: Path) -> list[str]:
    values = scene_values()
    actuator = fixed_actuator_cli_args(
        protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    common = [
        "--stage",
        str(protocol.STAGE),
        "--vlp16_noise_mode",
        protocol.LIDAR_NOISE_MODE,
        "--lidar-distractor-eligibility",
        protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "--arena_size",
        f"{values['arena_size_m']:g}",
        "--obs_near_goal_count",
        "0",
        "--seed",
        str(protocol.SEED),
        "--speed_rate",
        f"{protocol.SPEED_RATE:g}",
        "--speed_rate_obs",
        protocol.SPEED_RATE_OBS,
        "--deployment_speed_scale",
        f"{protocol.DEPLOYMENT_SPEED_SCALE:g}",
    ]
    if scenario["kind"] == "native":
        return [*common, *actuator]
    if scenario["kind"] == "narrow":
        return [
            *common,
            "--num_static_obs",
            "0",
            "--num_dynamic_obs",
            "0",
            "--num_walls",
            "0",
            "--narrow_replay_eval",
            "--narrow_replay_width_range",
            f"{values['narrow_width_range_m'][0]:g}",
            f"{values['narrow_width_range_m'][1]:g}",
            "--narrow_replay_yaw_limit_deg",
            f"{values['narrow_yaw_limit_deg']:g}",
            "--narrow_replay_segment_length",
            f"{values['narrow_segment_length_m']:g}",
            *actuator,
        ]
    if scenario["kind"] == "corridor":
        return corridor_helpers.build_scene_args(scenario, corridor_path)
    raise ValueError(f"unsupported retention scenario kind {scenario['kind']!r}")


def verify_native_runtime(stage_banner: dict) -> dict:
    problems = []
    if stage_banner.get("stage_name") != protocol.NATIVE_SCENE["stage_name"]:
        problems.append(f"stage_name={stage_banner.get('stage_name')}")
    if int(stage_banner.get("native_static_obstacles", -1)) != int(
        protocol.NATIVE_SCENE["static_obstacles"]
    ):
        problems.append(f"static={stage_banner.get('native_static_obstacles')}")
    if int(stage_banner.get("native_dynamic_obstacles", -1)) != int(
        protocol.NATIVE_SCENE["dynamic_obstacles"]
    ):
        problems.append(f"dynamic={stage_banner.get('native_dynamic_obstacles')}")
    if problems:
        raise RuntimeError(f"native stage scene mismatch: {problems}")
    return {
        "scenario": "nav_native",
        "mechanism": "stage scene via STAGE_PARAMETER",
        "stage_name": stage_banner["stage_name"],
        "static_obstacles": int(stage_banner["native_static_obstacles"]),
        "dynamic_obstacles": int(stage_banner["native_dynamic_obstacles"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--expect-protocol-sha256", required=True)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args(argv)

    frozen = protocol.retention_protocol()
    candidate = protocol.candidate_by_name(args.checkpoint_name)
    scenario = protocol.scenario_by_name(args.scenario)
    if args.expect_protocol_sha256 != frozen["sha256"]:
        raise ValueError("CLI protocol hash differs from frozen retention protocol")
    if args.expect_checkpoint_sha256 != candidate["sha256"]:
        raise ValueError("CLI checkpoint hash differs from frozen retention protocol")
    if args.smoke_steps is not None and not 1 <= args.smoke_steps <= 128:
        raise ValueError("smoke steps must be within [1, 128]")
    steps = args.smoke_steps or protocol.STEPS
    is_smoke = args.smoke_steps is not None

    checkpoint = args.checkpoint.expanduser().resolve()
    if checkpoint != protocol.checkpoint_path(candidate).resolve():
        raise ValueError("checkpoint path differs from frozen SA4-v3 lineage")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = protocol.sha256_of(checkpoint)
    if digest != candidate["sha256"]:
        raise RuntimeError("checkpoint content hash mismatch")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = f"_smoke{steps}" if is_smoke else ""
    stem = f"{protocol.cell_label(candidate['name'], scenario['name'])}{suffix}"
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    cell_path = output / f"{stem}_cell.json"
    targets = [log_path, cell_path]
    if scenario["kind"] == "corridor":
        targets.append(corridor_path)
    d9_runner._require_new_targets(targets)

    print(
        f"[SA4-V3-RETENTION-CELL] checkpoint={candidate['name']} "
        f"scenario={scenario['name']} steps={steps} smoke={is_smoke} "
        f"sha={digest[:12]}",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=protocol.NUM_ENVS,
        steps=steps,
        extra=build_scene_args(scenario, corridor_path),
    )

    d9_runner._validate_runtime_log(
        log_path, protocol.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        log_path, protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
    )
    values = scene_values()
    stage_banner = base.verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError("play outcome summary is missing")
    corridor_report = None
    if scenario["kind"] == "native":
        metrics = base.reconcile_console_metrics(
            dict(summary), base.parse_exact_outcome_counts(log_path)
        )
        scene_contract = verify_native_runtime(stage_banner)
        speed_rate_runtime = retention_helpers.verify_speed_rate_log(log_path)
    elif scenario["kind"] == "narrow":
        narrow = base.parse_narrow_replay_metrics(log_path)
        if int(narrow["episodes"]) != int(summary["n"]):
            raise RuntimeError("narrow episode ledgers disagree")
        metrics = base.reconcile_console_metrics(
            dict(summary), base.parse_exact_outcome_counts(log_path)
        )
        metrics.update(
            {key: value for key, value in narrow.items() if key != "episodes"}
        )
        scene_contract = base.verify_narrow_runtime(log_path, values)
        speed_rate_runtime = retention_helpers.verify_speed_rate_log(log_path)
    else:
        corridor_report = json.loads(corridor_path.read_text(encoding="utf-8"))
        scene_contract = corridor_helpers.verify_corridor_runtime(
            corridor_report, scenario, values
        )
        speed_rate_runtime = corridor_helpers.verify_speed_rate_runtime(
            corridor_report
        )
        metrics = base.reconcile_corridor_metrics(dict(summary), corridor_report)

    if not is_smoke and int(metrics["n"]) < protocol.MIN_EPISODES:
        raise RuntimeError(
            f"cell completed only {metrics['n']} episodes; "
            f"minimum is {protocol.MIN_EPISODES}"
        )
    gate = protocol.evaluate_metrics(scenario["name"], metrics)
    cell = {
        "schema": "sa4_v3_c550_c600_retention_cell/v1",
        "cell_valid": not is_smoke,
        "formal_evidence": not is_smoke,
        "cell": protocol.cell_label(candidate["name"], scenario["name"]),
        "checkpoint_name": candidate["name"],
        "conceptual_iteration": candidate["conceptual_iteration"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "protocol_sha256": frozen["sha256"],
        "scenario": scenario["name"],
        "scenario_kind": scenario["kind"],
        "stage": protocol.STAGE,
        "seed": protocol.SEED,
        "num_envs": protocol.NUM_ENVS,
        "steps": steps,
        "delay_steps": protocol.DELAY_STEPS,
        "actuator_eval": fixed_actuator_metadata(
            protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
        ),
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "speed_rate": protocol.SPEED_RATE,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        "speed_rate_runtime": speed_rate_runtime,
        "stage_banner": stage_banner,
        "scene_contract": scene_contract,
        "metrics": metrics,
        "gate": gate,
        "corridor_report": corridor_report,
        "log": str(log_path),
    }
    if is_smoke:
        cell["schema"] = "sa4_v3_c550_c600_retention_smoke/v1"
    else:
        protocol.validate_cell(cell)
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-V3-RETENTION-CELL] {cell['cell']} "
        f"{'PASS' if gate['pass'] else 'FAIL'} n={metrics['n']} "
        f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f}",
        flush=True,
    )
    # A valid threshold failure is formal evidence, not a runner failure.
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
