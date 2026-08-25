"""Contract tests for the frozen SA4-v3 c550/c600 retention screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_v3_c550_c600_retention_cell as runner  # noqa: E402
import sa4_v3_c550_c600_retention as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa4_v3_c550_c600_retention_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(checkpoint: str, scenario_name: str, cr: float = 0.02) -> dict:
    candidate = protocol.candidate_by_name(checkpoint)
    scenario = protocol.scenario_by_name(scenario_name)
    metrics = {"n": 1500, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    if scenario["kind"] == "narrow":
        metrics.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    payload = {
        "schema": "sa4_v3_c550_c600_retention_cell/v1",
        "cell_valid": True,
        "formal_evidence": True,
        "cell": protocol.cell_label(checkpoint, scenario_name),
        "checkpoint_name": checkpoint,
        "conceptual_iteration": candidate["conceptual_iteration"],
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
        "actuator_eval": runner.fixed_actuator_metadata(
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


def test_candidates_are_exactly_gate_promoted_order() -> None:
    assert [candidate["name"] for candidate in protocol.CANDIDATES] == [
        "c550",
        "c600",
    ]
    assert [candidate["gate_rank"] for candidate in protocol.CANDIDATES] == [1, 2]
    assert all(not candidate["gate_hard_pass"] for candidate in protocol.CANDIDATES)


def test_eight_cells_cover_native_narrow_and_p060_low_density() -> None:
    cells = protocol.cells()
    assert len(cells) == 8
    assert {cell["scenario"] for cell in cells} == {
        "nav_native",
        "narrow_range",
        "p060_0s1d_lateral",
        "p060_1s1d_lateral",
    }
    assert protocol.P060_SPEED_RANGE_M_S == (0.50, 0.70)
    assert protocol.MOTION_MODE == "lateral"


def test_candidate_and_gate_hashes_match_disk() -> None:
    for candidate in protocol.CANDIDATES:
        assert _sha256(protocol.checkpoint_path(candidate)) == candidate["sha256"]
    assert _sha256(protocol.GATE_SUMMARY) == protocol.GATE_SUMMARY_SHA256
    assert _sha256(protocol.GATE_FREEZE) == protocol.GATE_FREEZE_SHA256
    gate = json.loads(protocol.GATE_SUMMARY.read_text(encoding="utf-8"))
    protocol.validate_gate_summary(gate)


def test_gate_candidates_need_not_have_passed_graduation() -> None:
    gate = json.loads(protocol.GATE_SUMMARY.read_text(encoding="utf-8"))
    assert not any(row["hard_gate_pass"] for row in gate["ranked_candidates"])
    protocol.validate_gate_summary(gate)


def test_frozen_json_matches_runtime_protocol() -> None:
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == (
        protocol.retention_protocol()
    )


def test_native_uses_stage_four_scene_without_count_overrides(tmp_path: Path) -> None:
    args = runner.build_scene_args(
        protocol.scenario_by_name("nav_native"), tmp_path / "unused.json"
    )
    joined = " ".join(args)
    assert "--stage 4" in joined
    assert "--num_static_obs" not in args
    assert "--num_dynamic_obs" not in args
    assert "--num_walls" not in args
    assert "--long_corridor_eval" not in args


def test_narrow_geometry_comes_from_stage_four_source(tmp_path: Path) -> None:
    values = runner.scene_values()
    args = runner.build_scene_args(
        protocol.scenario_by_name("narrow_range"), tmp_path / "unused.json"
    )
    joined = " ".join(args)
    assert (
        "--narrow_replay_width_range "
        f"{values['narrow_width_range_m'][0]:g} "
        f"{values['narrow_width_range_m'][1]:g}"
    ) in joined
    assert (
        f"--narrow_replay_yaw_limit_deg {values['narrow_yaw_limit_deg']:g}"
        in joined
    )


def test_p060_cell_pins_density_speed_motion_and_common_contract(
    tmp_path: Path,
) -> None:
    args = runner.build_scene_args(
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


def test_thresholds_use_sa4_native_and_frozen_retention_contract() -> None:
    assert protocol.THRESHOLDS["nav_native"] == {
        "episodes_min": 1000,
        "sr_min": 0.92,
        "cr_max": 0.07,
        "to_max": 0.035,
    }
    assert protocol.THRESHOLDS["narrow_range"]["crossing_min"] == 0.95
    assert protocol.THRESHOLDS["narrow_range"]["direct_crossing_min"] == 0.95
    assert protocol.THRESHOLDS["p060_0s1d_lateral"]["cr_max"] == 0.10


def test_native_runtime_count_drift_invalidates_cell() -> None:
    payload = _cell("c550", "nav_native")
    payload["stage_banner"]["native_static_obstacles"] = 7
    with pytest.raises(ValueError, match="native runtime static count mismatch"):
        protocol.validate_cell(payload)


def test_valid_threshold_fail_remains_a_valid_cell() -> None:
    payload = _cell("c550", "p060_1s1d_lateral", cr=0.20)
    protocol.validate_cell(payload)
    assert payload["gate"]["pass"] is False


def test_all_pass_recommends_c550_for_bounded_exposure_only() -> None:
    payloads = [
        _cell(spec["checkpoint_name"], spec["scenario"])
        for spec in protocol.cells()
    ]
    summary = protocol.summarize_cells(payloads)
    assert summary["recommended_exposure_parent_candidate"] == "c550"
    assert summary["c550_exposure_parent_decision"] == (
        "ELIGIBLE_FOR_BOUNDED_EXPOSURE_CURRICULUM_PILOT"
    )
    assert summary["sa4_graduation_gate_pass"] is False
    assert summary["sa4_graduated"] is False
    assert summary["accepted_exposure_parent"] is None
    assert summary["sa5_started"] is False


def test_c550_failure_falls_back_to_c600_without_accepting_it() -> None:
    payloads = []
    for spec in protocol.cells():
        cr = 0.20 if (
            spec["checkpoint_name"] == "c550"
            and spec["scenario"] == "p060_1s1d_lateral"
        ) else 0.02
        payloads.append(_cell(spec["checkpoint_name"], spec["scenario"], cr))
    summary = protocol.summarize_cells(payloads)
    assert summary["recommended_exposure_parent_candidate"] == "c600"
    assert summary["c550_exposure_parent_decision"] == (
        "NOT_ELIGIBLE_DUE_TO_RETENTION_FAILURE"
    )
    assert summary["accepted_exposure_parent"] is None


def test_partial_screen_is_fail_closed_before_summary() -> None:
    payloads = [
        _cell(spec["checkpoint_name"], spec["scenario"])
        for spec in protocol.cells()[:-1]
    ]
    with pytest.raises(ValueError, match="exactly 8 cells"):
        protocol.summarize_cells(payloads)


def test_queue_has_no_training_or_parent_acceptance_path() -> None:
    source = (HERE / "sa4_v3_c550_c600_retention_queue.py").read_text(
        encoding="utf-8"
    )
    assert "for spec in protocol.cells()" in source
    assert '"accepted_exposure_parent": False' in source
    assert '"exposure_curriculum_started": False' in source
    assert '"sa5_started": False' in source
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemd-run" not in source


def test_native_source_still_defines_eight_static_two_dynamic() -> None:
    source = (REPO / protocol.NATIVE_SCENE["source"]).read_text(
        encoding="utf-8"
    )
    assert '"SA4_spatial_plan": (8, 2)' in source


def test_runner_treats_valid_gate_fail_as_evidence() -> None:
    source = (HERE / "run_sa4_v3_c550_c600_retention_cell.py").read_text(
        encoding="utf-8"
    )
    assert "A valid threshold failure is formal evidence" in source
    assert source.count("return 0") >= 1
