"""Contract tests for the frozen SA3-v3 P060 checkpoint screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa3_v3_p060_checkpoint_screen_cell as runner  # noqa: E402
import sa3_v3_p060_checkpoint_screen as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa3_v3_p060_checkpoint_screen_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(checkpoint: str, density: str, cr: float) -> dict:
    candidate = protocol.candidate_by_name(checkpoint)
    arm = protocol.density_by_label(density)
    metrics = {"n": 1500, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    return {
        "schema": "sa3_v3_p060_checkpoint_screen_cell/v1",
        "cell_valid": True,
        "formal_evidence": True,
        "cell": protocol.cell_label(checkpoint, density),
        "checkpoint_name": checkpoint,
        "conceptual_iteration": candidate["conceptual_iteration"],
        "checkpoint": str(protocol.checkpoint_path(candidate)),
        "checkpoint_sha256": candidate["sha256"],
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "density": density,
        "scenario": arm["scenario"],
        "stage": protocol.STAGE,
        "seed": protocol.SEED,
        "num_envs": protocol.NUM_ENVS,
        "steps": protocol.STEPS,
        "delay_steps": protocol.DELAY_STEPS,
        "speed_rate": protocol.SPEED_RATE,
        "pedestrian_speed_range_m_s": list(
            protocol.PEDESTRIAN_SPEED_RANGE_M_S
        ),
        "motion_mode": protocol.MOTION_MODE,
        "metrics": metrics,
        "gate": protocol.evaluate_metrics(metrics),
        "corridor_report": {
            "configured_static_obstacles": arm["static_obstacles"],
            "configured_dynamic_obstacles": arm["dynamic_obstacles"],
            "requested_dynamic_speed_range_m_s": list(
                protocol.PEDESTRIAN_SPEED_RANGE_M_S
            ),
            "dynamic_motion_mode": protocol.MOTION_MODE,
            "speed_rate": protocol.SPEED_RATE,
        },
    }


def test_candidate_subset_excludes_only_c50() -> None:
    assert [row["name"] for row in protocol.CANDIDATES] == [
        "c100",
        "c150",
        "c200",
        "c250",
        "c300",
    ]
    assert [row["name"] for row in protocol.EXCLUDED_CANDIDATES] == ["c50"]
    assert "36 percentage points" in protocol.EXCLUDED_CANDIDATES[0]["reason"]


def test_ten_cells_are_exactly_two_p060_lateral_arms_per_candidate() -> None:
    cells = protocol.cells()
    assert len(cells) == 10
    assert {cell["density"] for cell in cells} == {"0S1D", "1S1D"}
    assert protocol.PEDESTRIAN_SPEED_RANGE_M_S == (0.50, 0.70)
    assert protocol.MOTION_MODE == "lateral"
    assert protocol.SPEED_RATE == 0.7
    assert protocol.STAGE == 3


def test_candidate_hashes_match_the_completed_run() -> None:
    for candidate in protocol.CANDIDATES:
        path = protocol.checkpoint_path(candidate)
        assert path.is_file()
        assert _sha256(path) == candidate["sha256"]
    excluded = protocol.EXCLUDED_CANDIDATES[0]
    assert _sha256(protocol.RUN_DIR / excluded["filename"]) == excluded["sha256"]


def test_frozen_json_matches_runtime_protocol() -> None:
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == protocol.screen_protocol()


def test_runner_scene_args_pin_every_independent_variable(tmp_path: Path) -> None:
    args = runner.build_scene_args(
        protocol.density_by_label("1S1D"), tmp_path / "corridor.json"
    )
    joined = " ".join(args)
    for fragment in (
        "--stage 3",
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


def test_cell_validation_rejects_runtime_density_drift() -> None:
    payload = _cell("c200", "1S1D", 0.05)
    payload["corridor_report"]["configured_static_obstacles"] = 0
    with pytest.raises(ValueError, match="static density mismatch"):
        protocol.validate_cell(payload)


def test_ranking_is_worst_cell_first_not_symmetry_first() -> None:
    balanced_but_bad = {
        "checkpoint_name": "a",
        "conceptual_iteration": 100,
        "worst_fixed_cell_cr": 0.09,
        "mean_fixed_cell_cr": 0.09,
    }
    lopsided_but_good = {
        "checkpoint_name": "b",
        "conceptual_iteration": 50,
        "worst_fixed_cell_cr": 0.05,
        "mean_fixed_cell_cr": 0.035,
    }
    ranked = protocol.rank_summaries([balanced_but_bad, lopsided_but_good])
    assert ranked[0]["checkpoint_name"] == "b"


def test_exact_score_tie_prefers_later_checkpoint_only_last() -> None:
    rows = [
        {
            "checkpoint_name": "c100",
            "conceptual_iteration": 100,
            "worst_fixed_cell_cr": 0.05,
            "mean_fixed_cell_cr": 0.04,
        },
        {
            "checkpoint_name": "c150",
            "conceptual_iteration": 150,
            "worst_fixed_cell_cr": 0.05,
            "mean_fixed_cell_cr": 0.04,
        },
    ]
    assert protocol.rank_summaries(rows)[0]["checkpoint_name"] == "c150"


def test_summary_promotes_exactly_two_without_accepting_parent() -> None:
    cr = {
        "c100": (0.09, 0.08),
        "c150": (0.05, 0.05),
        "c200": (0.02, 0.05),
        "c250": (0.02, 0.03),
        "c300": (0.025, 0.035),
    }
    payloads = [
        _cell(candidate, density, cr[candidate][index])
        for candidate in cr
        for index, density in enumerate(("0S1D", "1S1D"))
    ]
    summary = protocol.summarize_cells(payloads)
    assert summary["promoted_for_retention"] == ["c250", "c300"]
    assert summary["accepted_parent"] is None
    assert summary["retention_started"] is False
    assert summary["sa4_parent_replaced"] is False
    assert summary["sa4_started"] is False


def test_partial_screen_is_fail_closed_before_ranking() -> None:
    payloads = [
        _cell(spec["checkpoint_name"], spec["density"], 0.05)
        for spec in protocol.cells()[:-1]
    ]
    with pytest.raises(ValueError, match="exactly 10 valid cells"):
        protocol.summarize_cells(payloads)


def test_queue_has_no_retention_or_sa4_execution_path() -> None:
    source = (
        HERE / "sa3_v3_p060_checkpoint_screen_queue.py"
    ).read_text(encoding="utf-8")
    assert "for spec in protocol.cells()" in source
    assert '"retention_started": False' in source
    assert '"sa4_started": False' in source
    assert "import run_sa4" not in source
    assert "from run_sa4" not in source
    assert "import run_sa5_joint_retention" not in source
    assert "from run_sa5_joint_retention" not in source


def test_valid_gate_fail_is_an_outcome_not_an_invalid_cell() -> None:
    payload = _cell("c100", "0S1D", 0.25)
    protocol.validate_cell(payload)
    assert payload["gate"]["pass"] is False
