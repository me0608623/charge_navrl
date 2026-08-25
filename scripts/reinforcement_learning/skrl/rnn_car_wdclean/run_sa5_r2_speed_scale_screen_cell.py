"""Run one frozen SA5-R2 c250 vehicle speed-rate cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sa3_gate_bc as base  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
import run_sa5_r2_checkpoint_screen_cell as parent_runner  # noqa: E402
import sa5_r2_checkpoint_screen as parent_protocol  # noqa: E402
import sa5_r2_speed_scale_screen as protocol  # noqa: E402
from fixed_actuator_eval import (  # noqa: E402
    fixed_actuator_metadata,
    verify_fixed_actuator_runtime,
)
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


def verify_speed_rate_runtime(report: dict, scale: float) -> dict:
    observed = float(report.get("speed_rate", float("nan")))
    error = float(report.get("deployment_scale_max_abs_error", float("nan")))
    samples = int(report.get("deployment_scale_samples", 0))
    if not math.isfinite(observed) or abs(observed - scale) > 1.0e-9:
        raise RuntimeError("vehicle speed-rate runtime mismatch")
    if report.get("speed_rate_obs_mode") != protocol.SPEED_RATE_OBS_MODE:
        raise RuntimeError("vehicle speed-rate observation mode mismatch")
    if float(report.get("deployment_speed_scale", float("nan"))) != 1.0:
        raise RuntimeError("downstream deployment scaling was stacked")
    if not math.isfinite(error) or error > 1.0e-6 or samples <= 0:
        raise RuntimeError("identity deployment output failed reconciliation")
    limits = report.get("speed_rate_action_limits") or {}
    for key in (
        "max_linear_velocity",
        "max_linear_accel",
        "max_angular_vel",
        "max_angular_accel",
    ):
        if key not in limits or not math.isfinite(float(limits[key])):
            raise RuntimeError(f"speed-rate runtime lacks {key}")
    return {
        "speed_rate": observed,
        "speed_rate_obs": protocol.SPEED_RATE_OBS_MODE,
        "action_limits": limits,
        "samples": samples,
        "identity_deployment_max_abs_error": error,
        "lidar_scaled": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--speed-rate", type=float, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--expect-protocol-sha256", required=True)
    args = parser.parse_args(argv)

    frozen = protocol.screen_protocol()
    scale = float(args.speed_rate)
    label = protocol.scale_label(scale)
    if args.expect_protocol_sha256 != frozen["sha256"]:
        raise ValueError("CLI protocol hash differs from frozen protocol")
    if args.expect_checkpoint_sha256 != protocol.CHECKPOINT["sha256"]:
        raise ValueError("CLI checkpoint hash differs from frozen protocol")

    checkpoint = parent_protocol.checkpoint_path(protocol.CHECKPOINT).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = base.sha256_of(checkpoint)
    if digest != protocol.CHECKPOINT["sha256"]:
        raise RuntimeError("checkpoint content hash mismatch")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = (
        f"c250_lateral_{label}_g5_d{parent_protocol.DELAY_STEPS}_"
        f"s{parent_protocol.SEED}"
    )
    log_path = output / f"{stem}.log"
    corridor_path = output / f"{stem}_corridor.json"
    timing_path = output / f"{stem}_timing.json"
    cell_path = output / f"{stem}_cell.json"
    d9_runner._require_new_targets(
        [log_path, corridor_path, timing_path, cell_path]
    )

    extra = [
        *parent_runner.build_scene_args(protocol.SCENARIO, corridor_path),
        "--speed_rate", f"{scale:g}",
        "--speed_rate_obs", protocol.SPEED_RATE_OBS_MODE,
        "--deployment_speed_scale", "1.0",
        "--d3_yield_audit",
        "--d3_yield_output",
        str(timing_path),
        "--d3_shield_mode",
        "baseline",
    ]
    print(
        f"[SA5-R2-SPEED-CELL] c250 scale={scale:g} "
        f"steps={protocol.STEPS} sha={digest[:12]}",
        flush=True,
    )
    _run_play(
        checkpoint,
        log_path,
        num_envs=parent_protocol.NUM_ENVS,
        steps=protocol.STEPS,
        extra=extra,
    )

    d9_runner._validate_runtime_log(
        log_path, parent_protocol.LIDAR_DISTRACTOR_ELIGIBILITY
    )
    verify_fixed_actuator_runtime(
        log_path,
        parent_protocol.DELAY_STEPS,
        parent_protocol.ACTUATOR_PROFILE,
    )
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    runtime_marker = (
        f"[VEHICLE-SPEED-RATE] rate={scale:g} obs="
        f"{protocol.SPEED_RATE_OBS_MODE} lidar_scaled=False "
        "deployment_scale=1"
    )
    if runtime_marker not in text:
        raise RuntimeError("vehicle speed-rate runtime marker is missing")
    values = parent_runner.stage_values()
    stage_banner = base.verify_common_runtime(log_path, values)

    summary = parse_play_summary(str(log_path))
    if not summary or summary.get("sr") is None:
        raise RuntimeError("play outcome summary is missing")
    corridor_report = json.loads(corridor_path.read_text(encoding="utf-8"))
    scene_contract = parent_runner.verify_corridor_runtime(
        corridor_report, protocol.SCENARIO, values
    )
    metrics = base.reconcile_corridor_metrics(dict(summary), corridor_report)
    if int(metrics["n"]) < protocol.MIN_EPISODES:
        raise RuntimeError(
            f"cell completed only {metrics['n']} episodes; "
            f"minimum is {protocol.MIN_EPISODES}"
        )
    speed_rate_runtime = verify_speed_rate_runtime(corridor_report, scale)

    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_meta = timing_payload["metadata"]
    if float(timing_meta["speed_rate"]) != scale:
        raise RuntimeError("D3 timing metadata speed-rate mismatch")
    if timing_meta["speed_rate_obs_mode"] != protocol.SPEED_RATE_OBS_MODE:
        raise RuntimeError("D3 timing metadata observation mode mismatch")
    reaction_timing = protocol.reaction_timing_summary(timing_payload)
    gate = parent_protocol.evaluate_metrics(protocol.SCENARIO, metrics)
    cell = {
        "schema": "sa5_r2_c250_speed_scale_screen_cell/v2",
        "cell_valid": True,
        "cell": label,
        "scale_label": label,
        "speed_rate": scale,
        "speed_rate_obs_mode": protocol.SPEED_RATE_OBS_MODE,
        "checkpoint_name": protocol.CHECKPOINT_NAME,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "protocol_sha256": frozen["sha256"],
        "scenario": protocol.SCENARIO,
        "stage": parent_protocol.STAGE,
        "seed": parent_protocol.SEED,
        "num_envs": parent_protocol.NUM_ENVS,
        "steps": protocol.STEPS,
        "actuator_eval": fixed_actuator_metadata(
            parent_protocol.DELAY_STEPS,
            parent_protocol.ACTUATOR_PROFILE,
        ),
        "speed_rate_runtime": speed_rate_runtime,
        "lidar_noise_mode": parent_protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": (
            parent_protocol.LIDAR_DISTRACTOR_ELIGIBILITY
        ),
        "scene_contract": scene_contract,
        "stage_banner": stage_banner,
        "metrics": metrics,
        "corridor_report": corridor_report,
        "reaction_timing": reaction_timing,
        "historical_gate": gate,
        "log": str(log_path),
        "timing_log": str(timing_path),
    }
    protocol.validate_cell(cell)
    cell_path.write_text(
        json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA5-R2-SPEED-CELL] {label} n={metrics['n']} "
        f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
        f"TO={metrics['to']:.4f} speed="
        f"{corridor_report['linear_speed_abs_mean_mps']:.4f} "
        f"stop={corridor_report['stop_command_fraction']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
