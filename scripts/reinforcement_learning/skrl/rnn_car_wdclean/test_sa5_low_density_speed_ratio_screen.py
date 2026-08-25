"""CPU contracts for the low-density P100 x robot speed-rate screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa5_r2_checkpoint_screen_cell as parent_runner  # noqa: E402
import sa5_low_density_speed_ratio_screen as screen  # noqa: E402


FREEZE = REPO / "docs/freeze/sa5_low_density_speed_ratio_screen_v1.json"


def _cell(density: str, rate: float, cr: float, *, n: int = 1200) -> dict:
    spec = screen.density_by_label(density)
    collisions = 12
    return {
        "schema": "sa5_low_density_speed_ratio_cell/v1",
        "cell_valid": True,
        "cell": screen.cell_label(density, rate),
        "density": density,
        "scenario": spec["scenario"],
        "pedestrian_speed_range_m_s": list(
            screen.PEDESTRIAN_SPEED_RANGE_M_S
        ),
        "speed_rate": rate,
        "checkpoint_sha256": screen.CHECKPOINT["sha256"],
        "protocol_sha256": screen.screen_protocol()["sha256"],
        "metrics": {"n": n, "sr": 1.0 - cr, "cr": cr, "to": 0.0},
        "corridor_report": {
            "configured_static_obstacles": spec["static_obstacles"],
            "configured_dynamic_obstacles": spec["dynamic_obstacles"],
            "requested_dynamic_speed_range_m_s": list(
                screen.PEDESTRIAN_SPEED_RANGE_M_S
            ),
            "speed_rate": rate,
            "obstacle_collision_rate": cr,
            "wall_collision_rate": 0.0,
            "stop_command_fraction": 0.2,
            "actual_body_stop_fraction": 0.2,
        },
        "reaction_timing": {
            "dynamic_collision": {
                "first_actual_decel": {"observed_fraction": 0.8},
                "first_actual_stop": {
                    "observed_fraction": 0.6,
                    "lead_s": {"p50": 2.0},
                },
            }
        },
        "impact_severity": {
            "events": collisions,
            "observed": collisions,
            "p50": 0.4,
            "p90": 0.8,
        },
        "dynamic_feasibility": {
            "events": collisions,
            "dynamic_feasible_within_1s_fraction": 0.6,
        },
    }


def _four_cells(cr_by_cell: dict[str, float]) -> list[dict]:
    return [
        _cell(spec["label"], spec["speed_rate"], cr_by_cell[spec["cell"]])
        for spec in screen.cells()
    ]


def test_protocol_is_exact_four_cell_cross_and_fixes_p100():
    protocol = screen.screen_protocol()
    assert len(protocol["cells"]) == 4
    assert {cell["label"] for cell in protocol["cells"]} == {
        "0S1D",
        "1S1D",
    }
    assert {cell["speed_rate"] for cell in protocol["cells"]} == {1.0, 0.7}
    assert protocol["fixed_evaluation"]["pedestrian_speed_range_m_s"] == [
        0.9,
        1.1,
    ]
    assert protocol["fixed_evaluation"]["lidar_distractor_eligibility"] == (
        "valid_return_only"
    )
    assert protocol["fixed_evaluation"]["actuator_delay_steps"] == 1
    assert protocol["fixed_evaluation"]["motion_mode"] == "lateral"


def test_scene_builder_uses_exact_low_density_and_p100_range():
    for density in screen.DENSITY_ARMS:
        args = parent_runner.build_scene_args(
            density["scenario"],
            Path("/tmp/corridor.json"),
            dynamic_speed_range=screen.PEDESTRIAN_SPEED_RANGE_M_S,
            corridor_motion_mode=screen.MOTION_MODE,
        )
        static_at = args.index("--long_corridor_static_obstacles")
        dynamic_at = args.index("--long_corridor_dynamic_obstacles")
        speed_at = args.index("--long_corridor_dynamic_speed_range")
        motion_at = args.index("--long_corridor_motion_mode")
        assert int(args[static_at + 1]) == density["static_obstacles"]
        assert int(args[dynamic_at + 1]) == density["dynamic_obstacles"]
        assert args[speed_at + 1 : speed_at + 3] == ["0.9", "1.1"]
        assert args[motion_at + 1] == "lateral"


def test_summary_requires_two_se_before_naming_speed_rate_effect():
    result = screen.summarize_cells(
        _four_cells(
            {
                "0s1d_s100": 0.20,
                "0s1d_s070": 0.21,
                "1s1d_s100": 0.22,
                "1s1d_s070": 0.23,
            }
        )
    )
    assert all(
        value["classification"] == "NO_ESTABLISHED_CR_DIFFERENCE"
        for value in result["within_density_comparisons"].values()
    )


def test_curriculum_boundary_uses_deployment_rate_gate():
    result = screen.summarize_cells(
        _four_cells(
            {
                "0s1d_s100": 0.04,
                "0s1d_s070": 0.05,
                "1s1d_s100": 0.07,
                "1s1d_s070": 0.09,
            }
        )
    )
    assert result["curriculum_boundary"] == "P100_LOW_DENSITY_READY"
    result = screen.summarize_cells(
        _four_cells(
            {
                "0s1d_s100": 0.05,
                "0s1d_s070": 0.08,
                "1s1d_s100": 0.12,
                "1s1d_s070": 0.14,
            }
        )
    )
    assert result["curriculum_boundary"] == "P100_0S1D_ONLY"
    result = screen.summarize_cells(
        _four_cells(
            {
                "0s1d_s100": 0.12,
                "0s1d_s070": 0.15,
                "1s1d_s100": 0.20,
                "1s1d_s070": 0.25,
            }
        )
    )
    assert result["curriculum_boundary"] == (
        "P100_NOT_ENTRY_READY_BEGIN_WITH_P060_P080"
    )


def test_missing_or_duplicate_cell_fails_closed():
    payloads = _four_cells(
        {
            "0s1d_s100": 0.05,
            "0s1d_s070": 0.05,
            "1s1d_s100": 0.05,
            "1s1d_s070": 0.05,
        }
    )
    with pytest.raises(ValueError, match="exactly four"):
        screen.summarize_cells(payloads[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        screen.summarize_cells([*payloads[:-1], payloads[0]])


def test_runner_is_baseline_only_and_forbids_training():
    source = (HERE / "run_sa5_low_density_speed_ratio_cell.py").read_text(
        encoding="utf-8"
    )
    for marker in (
        '"--speed_rate"',
        '"--speed_rate_obs"',
        '"--d3_yield_audit"',
        '"--d3_shield_mode"',
        '"baseline"',
        '"--d5_feasibility_shadow"',
        "verify_fixed_actuator_runtime",
        "verify_corridor_runtime",
    ):
        assert marker in source
    assert "train_rnn_car_wdclip" not in source


def test_checkpoint_and_freeze_match_runtime():
    checkpoint = screen.checkpoint_path()
    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        screen.CHECKPOINT["sha256"]
    )
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert frozen == screen.screen_protocol()
