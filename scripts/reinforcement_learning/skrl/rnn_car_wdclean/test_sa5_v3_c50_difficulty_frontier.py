"""Contract tests for the frozen SA5-v3 c50 difficulty frontier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import run_sa5_v3_c50_difficulty_frontier_cell as runner
import sa5_v3_c50_difficulty_frontier as protocol
import sa5_v3_c50_difficulty_frontier_queue as queue


REPO = Path(__file__).resolve().parents[4]
FREEZE = REPO / "docs/freeze/sa5_v3_c50_difficulty_frontier_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(spec: dict, cr: float = 0.05) -> dict:
    n = 1200
    layouts = {
        "405": {
            "episodes": 600,
            "success_rate": 0.95,
            "collision_rate": 0.05,
            "wall_collision_rate": 0.0,
            "obstacle_collision_rate": 0.05,
            "timeout_rate": 0.0,
        },
        "410": {
            "episodes": 600,
            "success_rate": 0.95,
            "collision_rate": 0.05,
            "wall_collision_rate": 0.0,
            "obstacle_collision_rate": 0.05,
            "timeout_rate": 0.0,
        },
    }
    if spec["static_obstacles"] != 4:
        layouts = {
            str(spec["static_obstacles"] * 100): {
                "episodes": n,
                "success_rate": 1.0 - cr,
                "collision_rate": cr,
                "wall_collision_rate": 0.0,
                "obstacle_collision_rate": cr,
                "timeout_rate": 0.0,
            }
        }
    metrics = {"n": n, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    verdict = protocol.evaluate_metrics(spec["name"], metrics)
    return {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "checkpoint_name": "c50",
        "checkpoint_sha256": protocol.ANCHOR["expected_sha256"],
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "geometry_stage": protocol.STAGE,
        "scenario": spec["name"],
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
            "fixed_density": (
                f"{spec['static_obstacles']}S{spec['dynamic_obstacles']}D"
            ),
            "motion_mode": spec["motion_mode"],
        },
        "corridor_report": {
            "configured_static_obstacles": spec["static_obstacles"],
            "configured_dynamic_obstacles": spec["dynamic_obstacles"],
            "requested_dynamic_speed_range_m_s": list(
                protocol.DYNAMIC_SPEED_RANGE
            ),
            "static_layout_episodes_total": n,
            "static_layout_outcomes": layouts,
            "wall_collision_rate": 0.0,
            "obstacle_collision_rate": cr,
        },
        "metrics": metrics,
        **verdict,
    }


def test_authorization_freezes_ungraduated_anchor_and_sa6_hold() -> None:
    record = json.loads(protocol.AUTHORIZATION.read_text(encoding="utf-8"))
    assert _sha256(protocol.AUTHORIZATION) == protocol.AUTHORIZATION_SHA256
    assert record["lineage"]["c50_status"] == (
        "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT"
    )
    assert record["lineage"]["c100_status"] == "DO_NOT_CONTINUE"
    assert record["lineage"]["c150_status"] == "DO_NOT_CONTINUE"
    assert record["lineage"]["sa6"] == "HOLD_NOT_AUTHORIZED"


def test_exact_c50_checkpoint_and_optimizer_are_present() -> None:
    path = protocol.checkpoint_path()
    assert _sha256(path) == protocol.ANCHOR["expected_sha256"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["iteration"] == 49
    assert payload["total_steps"] == 6400
    assert len(payload["charge_opt_rl"]["state"]) == 38


def test_matrix_is_exactly_three_by_two_by_four() -> None:
    assert len(protocol.SCENARIO_SPECS) == 24
    observed = {
        (row["static_obstacles"], row["dynamic_obstacles"], row["motion_mode"])
        for row in protocol.SCENARIO_SPECS
    }
    expected = {
        (s, d, mode)
        for s in (1, 2, 4)
        for d in (1, 2)
        for mode in ("lateral", "longitudinal", "random_2d", "mixed")
    }
    assert observed == expected


def test_protocol_and_freeze_lock_all_fixed_conditions() -> None:
    frozen = protocol.screen_protocol()
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == frozen
    fixed = frozen["fixed_evaluation"]
    assert fixed["seed"] == 818
    assert fixed["speed_rate"] == 0.7
    assert fixed["actuator_delay_steps"] == 1
    assert fixed["lidar_noise_mode"] == "full"
    assert fixed["lidar_distractor_eligibility"] == "valid_return_only"
    assert fixed["dynamic_speed_range_m_s"] == [0.25, 0.45]


@pytest.mark.parametrize("spec", protocol.SCENARIO_SPECS)
def test_runner_pins_each_density_and_motion_mode(spec: dict) -> None:
    joined = " ".join(runner.build_scene_args(spec["name"], Path("c.json")))
    assert (
        f"--long_corridor_static_obstacles {spec['static_obstacles']}" in joined
    )
    assert (
        f"--long_corridor_dynamic_obstacles {spec['dynamic_obstacles']}" in joined
    )
    assert f"--long_corridor_motion_mode {spec['motion_mode']}" in joined
    assert "--long_corridor_dynamic_speed_range 0.25 0.45" in joined
    assert "--speed_rate 0.7" in joined
    assert "--actuator_delay_range 1 1" in joined
    assert "--vlp16_noise_mode full" in joined
    assert "--lidar-distractor-eligibility valid_return_only" in joined


def test_dynamic_gate_crossing_is_detected() -> None:
    payloads = [_payload(spec) for spec in protocol.SCENARIO_SPECS]
    target = next(
        payload for payload in payloads
        if payload["scenario"] == "s1_d2_lateral"
    )
    target["metrics"] = {"n": 1200, "sr": 0.45, "cr": 0.55, "to": 0.0}
    target["corridor_report"]["obstacle_collision_rate"] = 0.55
    target.update(protocol.evaluate_metrics(target["scenario"], target["metrics"]))
    summary = protocol.analyze_cells(payloads)
    matches = [
        edge for edge in summary["severe_jumps"]
        if edge["source"] == "s1_d1_lateral"
        and edge["destination"] == "s1_d2_lateral"
    ]
    assert len(matches) == 1
    assert summary["recommended_next_action"] == (
        "BUILD_1D_TO_2D_CURRICULUM_PILOT_FROM_C50"
    )


def test_partial_screen_fails_closed() -> None:
    payloads = [_payload(spec) for spec in protocol.SCENARIO_SPECS]
    with pytest.raises(ValueError, match="exactly 24 cells"):
        protocol.analyze_cells(payloads[:-1])


def test_layout_410_ledger_is_required_for_4s() -> None:
    spec = next(row for row in protocol.SCENARIO_SPECS if row["name"] == "s4_d1_lateral")
    payload = _payload(spec)
    payload["corridor_report"]["static_layout_outcomes"].pop("410")
    with pytest.raises(ValueError, match="layouts 405 and 410"):
        protocol.validate_cell(payload)


def test_queue_cannot_launch_training_or_sa6() -> None:
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert "INCOMPLETE_FAIL_CLOSED" in source
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemd-run" not in source
    assert "pilot_started\": False" in source
    assert "sa6_started\": False" in source
