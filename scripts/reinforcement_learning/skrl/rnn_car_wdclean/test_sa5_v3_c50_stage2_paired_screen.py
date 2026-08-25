"""Contracts and decision tests for the paired fixed screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_v3_c50_stage2_paired_screen as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa5_v3_c50_stage2_paired_screen_v1.json"
QUEUE = HERE / "sa5_v3_c50_stage2_paired_screen_queue.py"


def _cell(candidate: dict, mode: str, sr: float, cr: float, to: float) -> dict:
    spec = next(row for row in protocol.SCENARIO_SPECS if row["motion_mode"] == mode)
    n = 2000
    metrics = {"n": n, "sr": sr, "cr": cr, "to": to}
    verdict = protocol.evaluate_metrics(spec["name"], metrics)
    return {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoint_name": candidate["name"],
        "checkpoint_sha256": candidate["expected_sha256"],
        "scenario": spec["name"],
        "geometry_stage": protocol.STAGE,
        "seed": protocol.SEED,
        "delay_steps": protocol.DELAY_STEPS,
        "num_envs": protocol.NUM_ENVS,
        "steps": protocol.STEPS,
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "speed_rate_runtime": {
            "speed_rate": protocol.SPEED_RATE,
            "speed_rate_obs": protocol.SPEED_RATE_OBS,
            "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        },
        "scene_contract": {"fixed_density": "4S2D", "motion_mode": mode},
        "corridor_report": {
            "configured_static_obstacles": 4,
            "configured_dynamic_obstacles": 2,
            "requested_dynamic_speed_range_m_s": list(protocol.DYNAMIC_SPEED_RANGE),
            "static_layout_episodes_total": n,
            "static_layout_outcomes": {
                "405": {"episodes": n // 2},
                "410": {"episodes": n // 2},
            },
            "wall_collision_rate": min(0.01, cr),
            "obstacle_collision_rate": max(0.0, cr - 0.01),
            "stop_command_fraction": 0.15,
            "linear_speed_abs_mean_mps": 0.30,
            "reverse_command_fraction": 0.10,
        },
        "metrics": metrics,
        "threshold_pass": verdict["threshold_pass"],
    }


def _matrix(*, target_gain: bool, retention_loss: bool = False) -> list[dict]:
    a2 = {
        "lateral": (0.80, 0.20, 0.00),
        "longitudinal": (0.90, 0.10, 0.00),
        "random_2d": (0.70, 0.28, 0.02),
        "mixed": (0.78, 0.21, 0.01),
    }
    b2 = dict(a2)
    b2["random_2d"] = (
        (0.74, 0.24, 0.02) if target_gain else (0.71, 0.27, 0.02)
    )
    if retention_loss:
        b2["lateral"] = (0.75, 0.25, 0.00)
    rows = []
    for candidate, data in ((protocol.A2_EQUAL, a2), (protocol.B2_MILD, b2)):
        for mode in protocol.MOTION_MODES:
            rows.append(_cell(candidate, mode, *data[mode]))
    return rows


def test_protocol_is_exactly_frozen() -> None:
    frozen = json.loads(FREEZE.read_text())
    assert frozen["protocol_sha256"] == protocol.screen_protocol()["sha256"]
    assert frozen["cell_count"] == 8
    assert frozen["screen_starts_training"] is False
    assert frozen["screen_starts_sa6"] is False


def test_matrix_and_runtime_contract_are_fixed() -> None:
    frozen = protocol.screen_protocol()
    assert frozen["matrix"]["cell_count"] == 8
    assert frozen["fixed_evaluation"]["seed"] == 818
    assert frozen["fixed_evaluation"]["actuator_delay_steps"] == 1
    assert frozen["fixed_evaluation"]["speed_rate"] == 0.7
    assert frozen["fixed_evaluation"]["minimum_episodes"] == 1000
    assert frozen["acceptance"]["minimum_target_sr_improvement"] == 0.02
    assert frozen["acceptance"]["minimum_target_cr_reduction"] == 0.02


def test_target_and_retention_pass_only_authorizes_more_retention() -> None:
    verdict = protocol.final_verdict(_matrix(target_gain=True))
    assert verdict["target_pass"] is True
    assert verdict["retention_pass"] is True
    assert verdict["mild_weight_acceptance_observed"] is True
    assert verdict["next_action"] == (
        "RUN_NATIVE_NARROW_P060_RETENTION_BEFORE_ANY_EXTENSION"
    )
    assert verdict["extension_authorized"] is False
    assert verdict["graduation_authorized"] is False
    assert verdict["sa6_started"] is False


def test_small_target_change_fails_even_when_direction_is_favorable() -> None:
    verdict = protocol.final_verdict(_matrix(target_gain=False))
    assert verdict["target_pass"] is False
    assert verdict["mild_weight_acceptance_observed"] is False
    assert verdict["next_action"] == (
        "MILD_030_030_040_TARGET_NOT_SUPPORTED_STOP_WEIGHT_TUNING"
    )


def test_retention_tradeoff_rejects_mild_weights() -> None:
    verdict = protocol.final_verdict(
        _matrix(target_gain=True, retention_loss=True)
    )
    assert verdict["target_pass"] is True
    assert verdict["retention_pass"] is False
    assert verdict["mild_weight_acceptance_observed"] is False
    assert verdict["next_action"] == (
        "MILD_030_030_040_CAPABILITY_REDISTRIBUTION_REJECT"
    )


def test_partial_or_duplicate_matrix_fails_closed() -> None:
    matrix = _matrix(target_gain=True)
    with pytest.raises(ValueError, match="exactly 8"):
        protocol.final_verdict(matrix[:-1])
    with pytest.raises(ValueError, match="duplicated or missing"):
        protocol.final_verdict(matrix[:-1] + [matrix[0]])


def test_queue_cannot_start_training_or_sa6() -> None:
    source = QUEUE.read_text()
    assert "INCOMPLETE" in (HERE / "sa5_v3_provisional_pilot_screen_queue.py").read_text()
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemd-run" not in source

