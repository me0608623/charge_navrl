"""Contracts for the c50 versus weighted-it25 fixed 4S2D screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa5_v3_c50_random2d_weighted_screen_cell as runner  # noqa: E402
import sa5_v3_c50_random2d_weighted_screen as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa5_v3_c50_random2d_weighted_screen_v1.json"
AUTHORIZATION = protocol.AUTHORIZATION
QUEUE = HERE / "sa5_v3_c50_random2d_weighted_screen_queue.py"


def _metrics(sr: float, cr: float, to: float = 0.0, n: int = 2000) -> dict:
    return {"n": n, "sr": sr, "cr": cr, "to": to}


def _cell(name: str, mode: str, metrics: dict, *, stop: float = 0.15) -> dict:
    scenario = next(
        row["name"] for row in protocol.SCENARIO_SPECS if row["motion_mode"] == mode
    )
    spec = protocol.candidate_by_name(name)
    verdict = protocol.evaluate_metrics(scenario, metrics)
    n = int(metrics["n"])
    return {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoint_name": name,
        "checkpoint_sha256": spec["sha256"],
        "scenario": scenario,
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
        "scene_contract": {
            "fixed_density": "4S2D",
            "motion_mode": mode,
        },
        "corridor_report": {
            "configured_static_obstacles": 4,
            "configured_dynamic_obstacles": 2,
            "requested_dynamic_speed_range_m_s": list(protocol.DYNAMIC_SPEED_RANGE),
            "static_layout_episodes_total": n,
            "static_layout_outcomes": {
                "405": {"episodes": n // 2},
                "410": {"episodes": n - n // 2},
            },
            "wall_collision_rate": 0.01,
            "obstacle_collision_rate": metrics["cr"] - 0.01,
            "stop_command_fraction": stop,
            "linear_speed_abs_mean_mps": 0.30,
            "reverse_command_fraction": 0.10,
        },
        "metrics": metrics,
        "threshold_pass": verdict["threshold_pass"],
    }


def _matrix(
    *,
    candidate_random: tuple[float, float, float] = (0.34, 0.64, 0.02),
    candidate_lateral: tuple[float, float, float] = (0.40, 0.60, 0.0),
) -> list[dict]:
    anchor_values = {
        "lateral": (0.40, 0.60, 0.0),
        "longitudinal": (0.18, 0.81, 0.01),
        "random_2d": (0.24, 0.74, 0.02),
        "mixed": (0.44, 0.56, 0.0),
    }
    candidate_values = dict(anchor_values)
    candidate_values["random_2d"] = candidate_random
    candidate_values["lateral"] = candidate_lateral
    result = []
    for name, values in (
        ("anchor_c50", anchor_values),
        ("weighted_it25", candidate_values),
    ):
        for mode in protocol.MOTION_MODES:
            result.append(_cell(name, mode, _metrics(*values[mode])))
    return result


def test_authorization_and_frozen_protocol_match() -> None:
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    assert authorization["matrix"]["cell_count"] == 8
    assert authorization["acceptance"]["target_motion_mode"] == "random_2d"
    assert authorization["acceptance"]["minimum_target_sr_improvement_pp"] == 5.0
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == protocol.screen_protocol()


def test_lineage_and_candidate_hashes_are_exact() -> None:
    assert protocol.ANCHOR["expected_sha256"] == (
        "4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c"
    )
    assert protocol.WEIGHTED_IT25["expected_sha256"] == (
        "cbc8a91a43eb0674dcd33118b3436f73f5d611b33860ebab5f7c83e6048f670f"
    )
    frozen = protocol.screen_protocol()
    assert frozen["screen_starts_training"] is False
    assert frozen["screen_starts_sa6"] is False
    assert frozen["screen_authorizes_extension"] is False


def test_runner_pins_density_noise_delay_speed_and_motion() -> None:
    for row in protocol.SCENARIO_SPECS:
        joined = " ".join(runner.build_scene_args(row["name"], Path("out.json")))
        assert "--long_corridor_static_obstacles 4" in joined
        assert "--long_corridor_dynamic_obstacles 2" in joined
        assert f"--long_corridor_motion_mode {row['motion_mode']}" in joined
        assert "--long_corridor_dynamic_speed_range 0.25 0.45" in joined
        assert "--speed_rate 0.7" in joined
        assert "--speed_rate_obs ego" in joined
        assert "--actuator_delay_range 1 1" in joined
        assert "--vlp16_noise_mode full" in joined


def test_target_improvement_with_direction_retention_passes() -> None:
    verdict = protocol.final_verdict(_matrix())
    assert verdict["target_pass"] is True
    assert verdict["retention_pass"] is True
    assert verdict["pilot_acceptance_observed"] is True
    assert verdict["selected_development_checkpoint"] == "weighted_it25"
    assert verdict["selected_extension_checkpoint"] is None
    assert verdict["extension_authorized"] is False
    assert verdict["sa6_started"] is False


def test_collision_to_timeout_conversion_is_rejected() -> None:
    verdict = protocol.final_verdict(
        _matrix(candidate_random=(0.24, 0.64, 0.12))
    )
    assert verdict["target_checks"]["cr_reduces"] is True
    assert verdict["target_checks"]["sr_improves"] is False
    assert verdict["target_checks"]["to_absolute"] is False
    assert verdict["target_pass"] is False
    assert verdict["selected_development_checkpoint"] is None


def test_two_point_direction_regression_is_rejected() -> None:
    verdict = protocol.final_verdict(
        _matrix(candidate_lateral=(0.37, 0.63, 0.0))
    )
    lateral = next(
        row for row in verdict["retention"] if row["motion_mode"] == "lateral"
    )
    assert lateral["checks"]["sr"] is False
    assert lateral["checks"]["cr"] is False
    assert verdict["retention_pass"] is False
    assert verdict["pilot_acceptance_observed"] is False


def test_partial_matrix_fails_closed() -> None:
    with pytest.raises(ValueError, match="exactly 8"):
        protocol.final_verdict(_matrix()[:-1])


def test_queue_cannot_launch_training_or_sa6() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    base_source = (
        HERE / "sa5_v3_provisional_pilot_screen_queue.py"
    ).read_text(encoding="utf-8")
    combined = source + base_source
    assert "INCOMPLETE_FAIL_CLOSED" in combined
    assert "source fingerprint drifted during screen" in combined
    assert "train_rnn_car_wdclip.py" not in combined
    assert "systemd-run" not in combined
