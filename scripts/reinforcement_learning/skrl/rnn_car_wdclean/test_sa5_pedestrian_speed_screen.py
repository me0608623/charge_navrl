"""CPU contracts for the no-training SA5 pedestrian-speed screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rnn_car_wdclean import run_sa5_r2_checkpoint_screen_cell as parent_runner
from rnn_car_wdclean import sa5_pedestrian_speed_screen as screen


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FREEZE = REPO / "docs/freeze/sa5_pedestrian_speed_screen_v1.json"


def _d3_payload(closing: float = 0.6) -> dict:
    return {
        "schema": "sa4_d3_yield_timing/v2",
        "mode": "baseline",
        "self_check": {"reconciliation_ok": True},
        "events": [
            {
                "event_type": "dynamic_collision",
                "tracked_obstacle_slot": 0,
                "records": [
                    {
                        "dynamic_obstacle_relative_closing_speeds_mps": [
                            closing
                        ]
                    }
                ],
            }
        ],
    }


def _d5_payload(lead: float | None = 0.8) -> dict:
    return {
        "schema": "sa4_d5_feasibility_frontier/v1",
        "self_check": {"reconciliation_ok": True},
        "events": [
            {
                "event_type": "dynamic_collision",
                "frontier": {
                    "closest_dynamic_feasible_s_before_event": lead,
                },
            }
        ],
        "frontier_summary": {
            "dynamic_collision": {
                "event_no_dynamic_feasible_fraction": 0.0,
                "persistent_dynamic_collapse_observed_fraction": 0.5,
                "persistent_dynamic_collapse_lead_s": {
                    "p10": 0.2,
                    "p25": 0.4,
                    "p50": 0.6,
                    "p75": 0.8,
                    "p90": 1.0,
                },
            }
        },
    }


def _cell(arm: dict, *, cr: float, feasible: float) -> dict:
    return {
        "schema": "sa5_c250_pedestrian_speed_screen_cell/v1",
        "cell_valid": True,
        "cell": arm["label"].lower(),
        "arm": arm["label"],
        "pedestrian_speed_range_m_s": list(arm["range_m_s"]),
        "speed_rate": 1.0,
        "checkpoint_sha256": screen.CHECKPOINT["sha256"],
        "protocol_sha256": screen.screen_protocol()["sha256"],
        "metrics": {"n": 1200, "sr": 1.0 - cr, "cr": cr, "to": 0.0},
        "corridor_report": {
            "requested_dynamic_speed_range_m_s": list(arm["range_m_s"]),
            "obstacle_collision_rate": cr,
            "wall_collision_rate": 0.0,
            "stop_command_fraction": 0.2,
            "actual_body_stop_fraction": 0.1,
        },
        "impact_severity": {
            "events": 10,
            "observed": 10,
            "p50": 0.5,
            "p90": 0.9,
        },
        "dynamic_feasibility": {
            "events": 10,
            "dynamic_feasible_within_1s_fraction": feasible,
            "event_no_dynamic_feasible_fraction": 1.0 - feasible,
            "persistent_dynamic_collapse_observed_fraction": 0.5,
        },
        "reaction_timing": {
            "dynamic_collision": {
                "first_actual_decel": {
                    "observed_fraction": 0.8,
                    "lead_s": {"p50": 1.2},
                },
                "first_actual_stop": {
                    "observed_fraction": 0.4,
                    "lead_s": {"p50": 1.0},
                },
            }
        },
    }


def test_protocol_changes_only_pedestrian_speed():
    frozen = screen.screen_protocol()
    assert frozen["only_independent_variable"] == (
        "long_corridor_dynamic_speed_range"
    )
    assert frozen["fixed_evaluation"]["speed_rate"] == 1.0
    assert frozen["fixed_evaluation"]["scenario"] == "corridor_lateral"
    assert frozen["fixed_evaluation"]["corridor_density"] == "4S2D"
    assert frozen["fixed_evaluation"]["steps"] == 3000
    assert [arm["label"] for arm in screen.PEDESTRIAN_SPEED_ARMS] == [
        "P035",
        "P060",
        "P080",
        "P100",
    ]


def test_scene_builder_passes_exact_arm_range_without_changing_default():
    default_args = parent_runner.build_scene_args(
        screen.SCENARIO, Path("/tmp/default.json")
    )
    arm_args = parent_runner.build_scene_args(
        screen.SCENARIO,
        Path("/tmp/arm.json"),
        dynamic_speed_range=(0.90, 1.10),
    )
    marker = "--long_corridor_dynamic_speed_range"
    default_index = default_args.index(marker)
    arm_index = arm_args.index(marker)
    assert default_args[default_index + 1 : default_index + 3] == [
        "0.25",
        "0.45",
    ]
    assert arm_args[arm_index + 1 : arm_index + 3] == ["0.9", "1.1"]
    assert arm_args.count(marker) == 1


def test_collision_severity_is_radial_component_with_full_coverage():
    result = screen.collision_radial_closing_summary(_d3_payload(0.75))
    assert result["events"] == 1
    assert result["observed"] == 1
    assert result["coverage"] == 1.0
    assert result["p50"] == pytest.approx(0.75)
    assert "not full 2D" in result["definition"]


def test_dynamic_feasibility_uses_dynamic_only_lead_window():
    result = screen.dynamic_feasibility_summary(_d5_payload(0.8))
    assert result["events"] == 1
    assert result["dynamic_feasible_within_1s_fraction"] == 1.0
    result = screen.dynamic_feasibility_summary(_d5_payload(1.2))
    assert result["dynamic_feasible_within_1s_fraction"] == 0.0


def test_summary_classifies_policy_limited_only_after_two_se_cr_rise():
    cr = {"P035": 0.20, "P060": 0.24, "P080": 0.28, "P100": 0.32}
    cells = [
        _cell(arm, cr=cr[arm["label"]], feasible=0.7)
        for arm in screen.PEDESTRIAN_SPEED_ARMS
    ]
    result = screen.summarize_cells(cells)
    assert result["mechanism_classification"] == (
        "POLICY_LIMITED_MAJORITY_DYNAMICALLY_FEASIBLE"
    )
    assert result["accepted_parent"] is False
    assert result["training_started"] is False
    assert result["sa6_started"] is False


def test_summary_names_d5_model_and_final_window_when_feasibility_is_low():
    cr = {"P035": 0.20, "P060": 0.24, "P080": 0.28, "P100": 0.32}
    cells = [
        _cell(arm, cr=cr[arm["label"]], feasible=0.3)
        for arm in screen.PEDESTRIAN_SPEED_ARMS
    ]
    result = screen.summarize_cells(cells)
    assert result["mechanism_classification"] == (
        "D5_MODEL_FEASIBILITY_LIMITED_MAJORITY_WITHOUT_"
        "FINAL_1S_DYNAMIC_SOLUTION"
    )


def test_missing_arm_fails_closed():
    cells = [
        _cell(arm, cr=0.20, feasible=0.7)
        for arm in screen.PEDESTRIAN_SPEED_ARMS
    ]
    with pytest.raises(ValueError, match="exactly four"):
        screen.summarize_cells(cells[:-1])


def test_checkpoint_and_freeze_match_runtime():
    checkpoint = screen.checkpoint_path()
    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        screen.CHECKPOINT["sha256"]
    )
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen == screen.screen_protocol()
