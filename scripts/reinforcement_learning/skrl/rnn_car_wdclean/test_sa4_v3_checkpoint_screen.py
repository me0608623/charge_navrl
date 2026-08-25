"""Contract tests for the frozen SA4-v3 checkpoint screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa4_v3_checkpoint_screen_cell as runner  # noqa: E402
import sa4_v3_checkpoint_screen as protocol  # noqa: E402


FREEZE = REPO / "docs/freeze/sa4_v3_checkpoint_screen_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(checkpoint: str, fixed_cell: str, cr: float) -> dict:
    candidate = protocol.candidate_by_name(checkpoint)
    spec = protocol.fixed_cell_by_label(fixed_cell)
    metrics = {"n": 1500, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    return {
        "schema": "sa4_v3_checkpoint_screen_cell/v1",
        "cell_valid": True,
        "formal_evidence": True,
        "cell": protocol.cell_label(checkpoint, fixed_cell),
        "checkpoint_name": checkpoint,
        "conceptual_iteration": candidate["conceptual_iteration"],
        "checkpoint": str(protocol.checkpoint_path(candidate)),
        "checkpoint_sha256": candidate["sha256"],
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "fixed_cell": fixed_cell,
        "density": spec["density"],
        "scenario": spec["scenario"],
        "stage": protocol.STAGE,
        "seed": protocol.SEED,
        "num_envs": protocol.NUM_ENVS,
        "steps": protocol.STEPS,
        "delay_steps": protocol.DELAY_STEPS,
        "speed_rate": protocol.SPEED_RATE,
        "pedestrian_speed_range_m_s": list(
            spec["pedestrian_speed_range_m_s"]
        ),
        "motion_mode": protocol.MOTION_MODE,
        "metrics": metrics,
        "gate": protocol.evaluate_metrics(metrics),
        "corridor_report": {
            "configured_static_obstacles": spec["static_obstacles"],
            "configured_dynamic_obstacles": spec["dynamic_obstacles"],
            "requested_dynamic_speed_range_m_s": list(
                spec["pedestrian_speed_range_m_s"]
            ),
            "dynamic_motion_mode": protocol.MOTION_MODE,
            "speed_rate": protocol.SPEED_RATE,
        },
    }


def test_all_six_checkpoints_are_included() -> None:
    assert [row["name"] for row in protocol.CANDIDATES] == [
        "c50",
        "c100",
        "c150",
        "c200",
        "c250",
        "c300",
    ]


def test_eighteen_cells_match_the_frozen_sa4_gate() -> None:
    assert len(protocol.cells()) == 18
    assert [row["label"] for row in protocol.FIXED_CELLS] == [
        "0S1D_P080",
        "1S1D_P080",
        "2S1D_P060",
    ]
    assert [row["pedestrian_speed_range_m_s"] for row in protocol.FIXED_CELLS] == [
        (0.70, 0.90),
        (0.70, 0.90),
        (0.50, 0.70),
    ]
    assert protocol.MOTION_MODE == "lateral"
    assert protocol.SPEED_RATE == 0.7
    assert protocol.STAGE == 4


def test_candidate_hashes_match_the_completed_run() -> None:
    for candidate in protocol.CANDIDATES:
        path = protocol.checkpoint_path(candidate)
        assert path.is_file()
        assert _sha256(path) == candidate["sha256"]


def test_frozen_json_matches_runtime_protocol() -> None:
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == protocol.screen_protocol()


@pytest.mark.parametrize("fixed_cell", ["0S1D_P080", "1S1D_P080", "2S1D_P060"])
def test_runner_scene_args_pin_every_independent_variable(
    tmp_path: Path, fixed_cell: str
) -> None:
    spec = protocol.fixed_cell_by_label(fixed_cell)
    args = runner.build_scene_args(spec, tmp_path / "corridor.json")
    joined = " ".join(args)
    speed = spec["pedestrian_speed_range_m_s"]
    for fragment in (
        "--stage 4",
        "--vlp16_noise_mode full",
        "--lidar-distractor-eligibility valid_return_only",
        f"--long_corridor_dynamic_speed_range {speed[0]:g} {speed[1]:g}",
        f"--long_corridor_static_obstacles {spec['static_obstacles']}",
        "--long_corridor_dynamic_obstacles 1",
        "--long_corridor_motion_mode lateral",
        "--speed_rate 0.7",
        "--speed_rate_obs ego",
        "--deployment_speed_scale 1",
        "--actuator_delay_range 1 1",
    ):
        assert fragment in joined


def test_stage4_helper_extension_does_not_leak() -> None:
    original = runner.base.SUPPORTED_GEOMETRY_STAGES
    values = runner.scene_values()
    assert values["arena_size_m"] > 0
    assert runner.base.SUPPORTED_GEOMETRY_STAGES == original


def test_cell_validation_rejects_runtime_density_drift() -> None:
    payload = _cell("c200", "2S1D_P060", 0.05)
    payload["corridor_report"]["configured_static_obstacles"] = 1
    with pytest.raises(ValueError, match="static density mismatch"):
        protocol.validate_cell(payload)


def test_cell_validation_rejects_speed_range_drift() -> None:
    payload = _cell("c200", "0S1D_P080", 0.05)
    payload["corridor_report"]["requested_dynamic_speed_range_m_s"] = [0.5, 0.7]
    with pytest.raises(ValueError, match="runtime pedestrian-speed range mismatch"):
        protocol.validate_cell(payload)


def test_ranking_is_worst_cell_first() -> None:
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
            "checkpoint_name": "c250",
            "conceptual_iteration": 250,
            "worst_fixed_cell_cr": 0.05,
            "mean_fixed_cell_cr": 0.04,
        },
        {
            "checkpoint_name": "c300",
            "conceptual_iteration": 300,
            "worst_fixed_cell_cr": 0.05,
            "mean_fixed_cell_cr": 0.04,
        },
    ]
    assert protocol.rank_summaries(rows)[0]["checkpoint_name"] == "c300"


def test_summary_promotes_two_without_authorizing_continuation() -> None:
    values = {
        "c50": (0.20, 0.22, 0.18),
        "c100": (0.12, 0.13, 0.11),
        "c150": (0.08, 0.09, 0.07),
        "c200": (0.04, 0.05, 0.03),
        "c250": (0.03, 0.04, 0.02),
        "c300": (0.02, 0.03, 0.025),
    }
    labels = [row["label"] for row in protocol.FIXED_CELLS]
    payloads = [
        _cell(candidate, label, values[candidate][index])
        for candidate in values
        for index, label in enumerate(labels)
    ]
    summary = protocol.summarize_cells(payloads)
    assert summary["promoted_for_retention"] == ["c300", "c250"]
    assert summary["retention_started"] is False
    assert summary["continuation_authorized"] is False
    assert summary["continuation_started"] is False
    assert summary["accepted_sa5_parent"] is None
    assert summary["sa5_started"] is False


def test_partial_screen_is_fail_closed_before_ranking() -> None:
    payloads = [
        _cell(spec["checkpoint_name"], spec["fixed_cell"], 0.05)
        for spec in protocol.cells()[:-1]
    ]
    with pytest.raises(ValueError, match="exactly 18 valid cells"):
        protocol.summarize_cells(payloads)


def test_valid_gate_fail_is_an_outcome_not_an_invalid_cell() -> None:
    payload = _cell("c50", "0S1D_P080", 0.25)
    protocol.validate_cell(payload)
    assert payload["gate"]["pass"] is False


def test_queue_cannot_start_retention_continuation_or_sa5() -> None:
    source = (HERE / "sa4_v3_checkpoint_screen_queue.py").read_text(
        encoding="utf-8"
    )
    assert "for index, spec in enumerate(protocol.cells(), start=1)" in source
    assert '"retention_started": False' in source
    assert '"continuation_authorized": False' in source
    assert '"continuation_started": False' in source
    assert '"sa5_started": False' in source
    assert "systemd-run" not in source
    assert "train_rnn_car_wdclip.py" not in source
