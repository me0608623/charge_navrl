"""Contract tests for the frozen SA3-v3 c300/c200 retention screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa3_v3_retention_screen_cell as runner  # noqa: E402
import sa3_v3_retention_screen as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa3_v3_retention_screen_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(checkpoint: str, scenario_name: str, cr: float = 0.02) -> dict:
    candidate = protocol.candidate_by_name(checkpoint)
    scenario = protocol.scenario_by_name(scenario_name)
    metrics = {"n": 1500, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    if scenario["kind"] == "narrow":
        metrics.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    payload = {
        "schema": "sa3_v3_retention_screen_cell/v1",
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
        "lidar_distractor_eligibility": (
            protocol.LIDAR_DISTRACTOR_ELIGIBILITY
        ),
        "speed_rate": protocol.SPEED_RATE,
        "speed_rate_obs": protocol.SPEED_RATE_OBS,
        "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        "speed_rate_runtime": {
            "speed_rate": protocol.SPEED_RATE,
            "speed_rate_obs": protocol.SPEED_RATE_OBS,
            "lidar_scaled": False,
            "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
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
                protocol.P035_SPEED_RANGE_M_S
            ),
            "dynamic_motion_mode": protocol.P035_MOTION_MODE,
        }
    return payload


def test_candidates_are_exactly_phase_a_promoted_order() -> None:
    assert [candidate["name"] for candidate in protocol.CANDIDATES] == [
        "c300",
        "c200",
    ]
    assert [candidate["phase_a_rank"] for candidate in protocol.CANDIDATES] == [
        1,
        2,
    ]


def test_eight_cells_cover_native_narrow_and_two_p035_arms() -> None:
    cells = protocol.cells()
    assert len(cells) == 8
    assert {cell["scenario"] for cell in cells} == {
        "nav_native",
        "narrow_range",
        "p035_0s1d_lateral",
        "p035_1s1d_lateral",
    }
    assert protocol.P035_SPEED_RANGE_M_S == (0.25, 0.45)
    assert protocol.P035_MOTION_MODE == "lateral"


def test_candidate_and_phase_a_hashes_match_disk() -> None:
    for candidate in protocol.CANDIDATES:
        assert _sha256(protocol.checkpoint_path(candidate)) == candidate["sha256"]
    assert _sha256(protocol.PHASE_A_SUMMARY) == protocol.PHASE_A_SUMMARY_SHA256
    phase_a = json.loads(protocol.PHASE_A_SUMMARY.read_text(encoding="utf-8"))
    protocol.validate_phase_a_summary(phase_a)


def test_frozen_json_matches_runtime_protocol() -> None:
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == (
        protocol.retention_protocol()
    )


def test_native_uses_stage_scene_without_count_overrides(tmp_path: Path) -> None:
    args = runner.build_scene_args(
        protocol.scenario_by_name("nav_native"), tmp_path / "unused.json"
    )
    joined = " ".join(args)
    assert "--stage 3" in joined
    assert "--num_static_obs" not in args
    assert "--num_dynamic_obs" not in args
    assert "--num_walls" not in args
    assert "--long_corridor_eval" not in args


def test_narrow_geometry_comes_from_stage_three_source(tmp_path: Path) -> None:
    values = runner.base.scene_values(protocol.STAGE)
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
    assert (
        f"--narrow_replay_segment_length {values['narrow_segment_length_m']:g}"
        in joined
    )


def test_p035_cell_pins_density_speed_motion_and_common_contract(
    tmp_path: Path,
) -> None:
    args = runner.build_scene_args(
        protocol.scenario_by_name("p035_1s1d_lateral"),
        tmp_path / "corridor.json",
    )
    joined = " ".join(args)
    for fragment in (
        "--vlp16_noise_mode full",
        "--lidar-distractor-eligibility valid_return_only",
        "--long_corridor_dynamic_speed_range 0.25 0.45",
        "--long_corridor_static_obstacles 1",
        "--long_corridor_dynamic_obstacles 1",
        "--long_corridor_motion_mode lateral",
        "--speed_rate 0.7",
        "--speed_rate_obs ego",
        "--deployment_speed_scale 1",
        "--actuator_delay_range 1 1",
    ):
        assert fragment in joined


def test_thresholds_match_frozen_acceptance_contract() -> None:
    assert protocol.THRESHOLDS["nav_native"] == {
        "episodes_min": 1000,
        "sr_min": 0.94,
        "cr_max": 0.05,
        "to_max": 0.03,
    }
    assert protocol.THRESHOLDS["narrow_range"]["crossing_min"] == 0.95
    assert protocol.THRESHOLDS["narrow_range"]["direct_crossing_min"] == 0.95
    assert protocol.THRESHOLDS["p035_0s1d_lateral"]["cr_max"] == 0.10


def test_speed_rate_runtime_marker_is_checked(tmp_path: Path) -> None:
    log = tmp_path / "speed.log"
    log.write_text(
        "\n".join(
            [
                "[SPEED_RATE] max_linear_velocity: 1.0 -> 0.7000",
                "[SPEED_RATE] max_linear_accel: 0.5 -> 0.3500",
                "[SPEED_RATE] max_angular_vel: 1.2 -> 0.8400",
                "[SPEED_RATE] max_angular_accel: 3.0 -> 2.1000",
                "[VEHICLE-SPEED-RATE] rate=0.7 obs=ego "
                "lidar_scaled=False deployment_scale=1",
            ]
        ),
        encoding="utf-8",
    )
    result = runner.verify_speed_rate_log(log)
    assert result["speed_rate"] == 0.7
    assert result["deployment_speed_scale"] == 1.0
    assert len(result["action_limits"]) == 4


def test_native_runtime_count_drift_invalidates_cell() -> None:
    payload = _cell("c300", "nav_native")
    payload["stage_banner"]["native_static_obstacles"] = 5
    with pytest.raises(ValueError, match="native runtime static count mismatch"):
        protocol.validate_cell(payload)


def test_valid_threshold_fail_remains_a_valid_cell() -> None:
    payload = _cell("c300", "p035_1s1d_lateral", cr=0.20)
    protocol.validate_cell(payload)
    assert payload["gate"]["pass"] is False


def test_summary_respects_phase_a_order_and_does_not_accept_parent() -> None:
    payloads = [
        _cell(spec["checkpoint_name"], spec["scenario"])
        for spec in protocol.cells()
    ]
    summary = protocol.summarize_cells(payloads)
    assert summary["recommended_parent_candidate"] == "c300"
    assert summary["accepted_parent"] is None
    assert summary["sa4_parent_replaced"] is False
    assert summary["sa4_started"] is False


def test_summary_falls_back_to_c200_only_when_c300_fails() -> None:
    payloads = []
    for spec in protocol.cells():
        cr = 0.20 if (
            spec["checkpoint_name"] == "c300"
            and spec["scenario"] == "p035_1s1d_lateral"
        ) else 0.02
        payloads.append(_cell(spec["checkpoint_name"], spec["scenario"], cr))
    summary = protocol.summarize_cells(payloads)
    assert summary["recommended_parent_candidate"] == "c200"


def test_partial_screen_is_fail_closed_before_summary() -> None:
    payloads = [
        _cell(spec["checkpoint_name"], spec["scenario"])
        for spec in protocol.cells()[:-1]
    ]
    with pytest.raises(ValueError, match="exactly 8 cells"):
        protocol.summarize_cells(payloads)


def test_queue_has_no_parent_acceptance_or_sa4_execution_path() -> None:
    source = (HERE / "sa3_v3_retention_screen_queue.py").read_text(
        encoding="utf-8"
    )
    assert "for spec in protocol.cells()" in source
    assert '"accepted_parent": False' in source
    assert '"sa4_started": False' in source
    assert "import run_sa4" not in source
    assert "from run_sa4" not in source


def test_native_source_still_defines_six_static_one_dynamic() -> None:
    source = (REPO / protocol.NATIVE_SCENE["source"]).read_text(
        encoding="utf-8"
    )
    assert '"SA3_walls_crossing": (6, 1)' in source


def test_runner_treats_valid_gate_fail_as_evidence() -> None:
    source = (HERE / "run_sa3_v3_retention_screen_cell.py").read_text(
        encoding="utf-8"
    )
    assert "A valid threshold failure is evidence" in source
    assert source.count("return 0") >= 1
