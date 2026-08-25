"""Contract tests for the frozen staged SA4-v3 c500 retention screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_v3_c500_staged_retention_cell as runner  # noqa: E402
import sa4_v3_c500_staged_retention as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa4_v3_c500_staged_retention_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(scenario_name: str, cr: float = 0.02) -> dict:
    candidate = protocol.candidate_by_name("c500")
    scenario = protocol.scenario_by_name(scenario_name)
    metrics = {"n": 1500, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    if scenario["kind"] == "narrow":
        metrics.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    payload = {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "formal_evidence": True,
        "cell": protocol.cell_label("c500", scenario_name),
        "checkpoint_name": "c500",
        "conceptual_iteration": 500,
        "checkpoint": str(protocol.checkpoint_path(candidate)),
        "checkpoint_sha256": candidate["sha256"],
        "protocol_sha256": protocol.retention_protocol()["sha256"],
        "scenario": scenario_name,
        "scenario_kind": scenario["kind"],
        "stage": protocol.STAGE,
        "seed": protocol.SEED,
        "num_envs": protocol.NUM_ENVS,
        "steps": protocol.STEPS,
        "delay_steps": protocol.DELAY_STEPS,
        "actuator_eval": runner.implementation.fixed_actuator_metadata(
            protocol.DELAY_STEPS, protocol.ACTUATOR_PROFILE
        ),
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "speed_rate": protocol.SPEED_RATE,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        "speed_rate_runtime": {
            "speed_rate": protocol.SPEED_RATE,
            "speed_rate_obs": protocol.SPEED_RATE_OBS,
            "action_limits": {
                "max_linear_velocity": 0.7,
                "max_linear_accel": 0.35,
                "max_angular_vel": 0.84,
                "max_angular_accel": 2.1,
            },
        },
        "stage_banner": {
            "stage_built": protocol.STAGE,
            "stage_name": protocol.NATIVE_SCENE["stage_name"],
            "native_static_obstacles": protocol.NATIVE_SCENE[
                "static_obstacles"
            ],
            "native_dynamic_obstacles": protocol.NATIVE_SCENE[
                "dynamic_obstacles"
            ],
        },
        "scene_contract": {"scenario": scenario_name},
        "metrics": metrics,
        "gate": protocol.evaluate_metrics(scenario_name, metrics),
        "corridor_report": None,
    }
    if scenario["kind"] == "corridor":
        payload["corridor_report"] = {
            "configured_static_obstacles": scenario["static_obstacles"],
            "configured_dynamic_obstacles": scenario["dynamic_obstacles"],
            "requested_dynamic_speed_range_m_s": list(
                protocol.P060_SPEED_RANGE_M_S
            ),
            "dynamic_motion_mode": protocol.MOTION_MODE,
        }
    return payload


def test_candidate_is_exact_gate_rank_three_c500() -> None:
    assert len(protocol.CANDIDATES) == 1
    candidate = protocol.CANDIDATES[0]
    assert candidate["name"] == "c500"
    assert candidate["conceptual_iteration"] == 500
    assert candidate["gate_rank"] == 3
    assert candidate["gate_hard_pass"] is False


def test_locked_sources_and_checkpoint_match_disk() -> None:
    assert _sha256(protocol.checkpoint_path("c500")) == (
        protocol.CANDIDATES[0]["sha256"]
    )
    assert _sha256(protocol.GATE_SUMMARY) == protocol.GATE_SUMMARY_SHA256
    assert _sha256(protocol.GATE_FREEZE) == protocol.GATE_FREEZE_SHA256
    assert _sha256(protocol.PREVIOUS_RETENTION_SUMMARY) == (
        protocol.PREVIOUS_RETENTION_SUMMARY_SHA256
    )
    assert _sha256(protocol.PREVIOUS_RETENTION_FREEZE) == (
        protocol.PREVIOUS_RETENTION_FREEZE_SHA256
    )
    gate = json.loads(protocol.GATE_SUMMARY.read_text(encoding="utf-8"))
    previous = json.loads(
        protocol.PREVIOUS_RETENTION_SUMMARY.read_text(encoding="utf-8")
    )
    protocol.validate_sources(gate, previous)


def test_frozen_json_matches_runtime_protocol() -> None:
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == (
        protocol.retention_protocol()
    )


def test_staging_order_is_native_then_three_followups() -> None:
    assert [cell["scenario"] for cell in protocol.cells()] == [
        "nav_native",
        "narrow_range",
        "p060_0s1d_lateral",
        "p060_1s1d_lateral",
    ]
    assert protocol.native_cell()["scenario"] == "nav_native"
    assert [cell["scenario"] for cell in protocol.followup_cells()] == [
        "narrow_range",
        "p060_0s1d_lateral",
        "p060_1s1d_lateral",
    ]


def test_native_uses_stage_four_without_count_overrides(tmp_path: Path) -> None:
    args = runner.implementation.build_scene_args(
        protocol.scenario_by_name("nav_native"), tmp_path / "unused.json"
    )
    assert "--stage" in args and args[args.index("--stage") + 1] == "4"
    assert "--num_static_obs" not in args
    assert "--num_dynamic_obs" not in args
    assert "--num_walls" not in args
    assert "--long_corridor_eval" not in args


def test_p060_contract_is_frozen(tmp_path: Path) -> None:
    args = runner.implementation.build_scene_args(
        protocol.scenario_by_name("p060_1s1d_lateral"),
        tmp_path / "corridor.json",
    )
    joined = " ".join(args)
    for fragment in (
        "--vlp16_noise_mode full",
        "--lidar-distractor-eligibility valid_return_only",
        "--long_corridor_dynamic_speed_range 0.5 0.7",
        "--long_corridor_static_obstacles 1",
        "--long_corridor_dynamic_obstacles 1",
        "--long_corridor_motion_mode lateral",
        "--speed_rate 0.7",
        "--speed_rate_obs ego",
        "--deployment_speed_scale 1",
        "--actuator_delay_range 1 1",
    ):
        assert fragment in joined


def test_thresholds_match_prior_retention_contract() -> None:
    assert protocol.THRESHOLDS["nav_native"] == {
        "episodes_min": 1000,
        "sr_min": 0.92,
        "cr_max": 0.07,
        "to_max": 0.035,
    }
    assert protocol.THRESHOLDS["narrow_range"]["cr_max"] == 0.05
    assert protocol.THRESHOLDS["narrow_range"]["crossing_min"] == 0.95
    assert protocol.THRESHOLDS["p060_0s1d_lateral"]["cr_max"] == 0.10


def test_valid_native_fail_is_complete_and_stops_followups() -> None:
    summary = protocol.summarize_staged([_cell("nav_native", cr=0.085)])
    assert summary["status"] == (
        "COMPLETE_VALID_NATIVE_BLOCKED_STAGED_RETENTION"
    )
    assert summary["native_pass"] is False
    assert summary["followup_authorized"] is False
    assert len(summary["remaining_cells_not_run"]) == 3
    assert summary["retention_pass"] is False


def test_followup_after_native_fail_is_rejected() -> None:
    with pytest.raises(ValueError, match="after native failed"):
        protocol.summarize_staged(
            [_cell("nav_native", cr=0.085), _cell("narrow_range")]
        )


def test_native_pass_without_all_followups_is_incomplete() -> None:
    with pytest.raises(ValueError, match="three follow-up cells are incomplete"):
        protocol.summarize_staged([_cell("nav_native")])


def test_all_four_pass_make_c500_eligible_for_bounded_pilot_only() -> None:
    payloads = [_cell(spec["scenario"]) for spec in protocol.cells()]
    summary = protocol.summarize_staged(payloads)
    assert summary["status"] == "COMPLETE_VALID_FULL_STAGED_RETENTION"
    assert summary["retention_pass"] is True
    assert summary["c500_exposure_parent_decision"] == (
        "ELIGIBLE_FOR_BOUNDED_EXPOSURE_CURRICULUM_PILOT"
    )
    assert summary["sa4_graduated"] is False
    assert summary["accepted_exposure_parent"] is None
    assert summary["exposure_curriculum_started"] is False
    assert summary["sa5_started"] is False


def test_followup_fail_blocks_c500_parent_eligibility() -> None:
    payloads = [
        _cell(spec["scenario"], cr=0.12 if spec["scenario"] == (
            "p060_1s1d_lateral"
        ) else 0.02)
        for spec in protocol.cells()
    ]
    summary = protocol.summarize_staged(payloads)
    assert summary["native_pass"] is True
    assert summary["retention_pass"] is False
    assert summary["c500_exposure_parent_decision"] == (
        "NOT_ELIGIBLE_DUE_TO_RETENTION_FAILURE"
    )


def test_runner_adapts_only_historical_schema_literal() -> None:
    source = (HERE / "run_sa4_v3_c500_staged_retention_cell.py").read_text(
        encoding="utf-8"
    )
    assert "implementation.protocol = protocol" in source
    assert "implementation.retention_helpers.protocol = protocol" in source
    assert "implementation.corridor_helpers.protocol = protocol" in source
    assert 'adapted["schema"] = protocol.CELL_SCHEMA' in source
    assert 'payload["schema"] = protocol.SMOKE_SCHEMA if is_smoke else' in source


def test_queue_has_staged_branch_and_no_downstream_launch() -> None:
    source = (HERE / "sa4_v3_c500_staged_retention_queue.py").read_text(
        encoding="utf-8"
    )
    assert 'if native["gate"]["pass"]:' in source
    assert "for spec in protocol.followup_cells():" in source
    assert "follow-up cells will not run" in source
    assert '"exposure_curriculum_started": False' in source
    assert '"sa5_started": False' in source
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemd-run" not in source
