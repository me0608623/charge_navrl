"""Contracts for the matched equal-weight-control fixed screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa5_v3_c50_equalweight_control_screen_cell as runner  # noqa: E402
import sa5_v3_c50_equalweight_control_screen as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa5_v3_c50_equalweight_control_screen_v1.json"
QUEUE = HERE / "sa5_v3_c50_equalweight_control_screen_queue.py"


def _metrics(sr: float, cr: float, to: float, n: int) -> dict:
    return {"n": n, "sr": sr, "cr": cr, "to": to}


def _cell(mode: str, metrics: dict) -> dict:
    scenario = next(
        row["name"] for row in protocol.SCENARIO_SPECS if row["motion_mode"] == mode
    )
    verdict = protocol.evaluate_metrics(scenario, metrics)
    n = int(metrics["n"])
    return {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoint_name": protocol.CONTROL_IT25["name"],
        "checkpoint_sha256": protocol.CONTROL_IT25["expected_sha256"],
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
        "scene_contract": {"fixed_density": "4S2D", "motion_mode": mode},
        "corridor_report": {
            "configured_static_obstacles": 4,
            "configured_dynamic_obstacles": 2,
            "requested_dynamic_speed_range_m_s": list(protocol.DYNAMIC_SPEED_RANGE),
            "static_layout_episodes_total": n,
            "static_layout_outcomes": {
                "405": {"episodes": n // 2},
                "410": {"episodes": n - n // 2},
            },
            "wall_collision_rate": min(0.01, metrics["cr"]),
            "obstacle_collision_rate": max(0.0, metrics["cr"] - 0.01),
            "stop_command_fraction": 0.15,
            "linear_speed_abs_mean_mps": 0.30,
            "reverse_command_fraction": 0.10,
        },
        "metrics": metrics,
        "threshold_pass": verdict["threshold_pass"],
    }


def _control_matrix(*, target_support: bool, lateral_tradeoff: bool = False) -> list[dict]:
    prior = {
        (row["checkpoint_name"], row["motion_mode"]): row
        for row in protocol._frozen_prior_rows()
    }
    cells = []
    for mode in protocol.MOTION_MODES:
        weighted = prior[(protocol.WEIGHTED_IT25["name"], mode)]
        sr, cr, to = weighted["sr"], weighted["cr"], weighted["to"]
        if mode == "random_2d" and target_support:
            sr -= 0.08
            cr += 0.08
        if mode == "lateral" and lateral_tradeoff:
            sr += 0.05
            cr -= 0.05
        cells.append(_cell(mode, _metrics(sr, cr, to, weighted["n"])))
    return cells


def test_authorization_and_frozen_protocol_match() -> None:
    authorization = json.loads(protocol.AUTHORIZATION.read_text(encoding="utf-8"))
    assert authorization["new_matrix"]["new_cell_count"] == 4
    assert authorization["new_matrix"]["reuse_existing_cell_count"] == 8
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == protocol.screen_protocol()


def test_lineage_and_hashes_are_exact() -> None:
    assert protocol.CONTROL_IT25["expected_sha256"] == (
        "d9095e7df21633056ef3f491d460520b99dbd72d9924355821c1e48783b3419a"
    )
    assert protocol.screen_protocol()["screen_starts_training"] is False
    assert protocol.screen_protocol()["screen_starts_sa6"] is False


def test_runner_pins_density_noise_delay_speed_and_motion() -> None:
    for row in protocol.SCENARIO_SPECS:
        joined = " ".join(runner.base_runner.build_scene_args(row["name"], Path("x.json")))
        assert "--long_corridor_static_obstacles 4" in joined
        assert "--long_corridor_dynamic_obstacles 2" in joined
        assert "--actuator_delay_range 1 1" in joined
        assert "--vlp16_noise_mode full" in joined


def test_weight_specific_target_support_can_be_detected() -> None:
    verdict = protocol.final_verdict(_control_matrix(target_support=True))
    assert verdict["weight_specific_target_pass"] is True
    assert verdict["weight_specific_retention_pass"] is True
    assert verdict["weight_intervention_accepted"] is True
    assert verdict["selected_extension_checkpoint"] is None
    assert verdict["sa6_started"] is False


def test_weight_specific_tradeoff_rejects_current_weights() -> None:
    verdict = protocol.final_verdict(
        _control_matrix(target_support=True, lateral_tradeoff=True)
    )
    assert verdict["weight_specific_retention_pass"] is False
    assert verdict["weight_intervention_accepted"] is False
    assert verdict["next_action"] == (
        "WEIGHT_SPECIFIC_CAPABILITY_REDISTRIBUTION_OBSERVED_REJECT_CURRENT_WEIGHTS"
    )


def test_partial_matrix_fails_closed() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        protocol.final_verdict(_control_matrix(target_support=True)[:-1])


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
