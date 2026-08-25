from __future__ import annotations

from pathlib import Path

import run_sa4_checkpoint_screen as runner
import sa4_checkpoint_screen_queue as queue
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
)


def _metrics(sr: float, cr: float) -> dict:
    return {"n": 1200, "sr": sr, "cr": cr, "to": 0.0}


def _payload(checkpoint: str, scenario: str, sr: float, cr: float) -> dict:
    return {
        "checkpoint_name": checkpoint,
        "scenario": scenario,
        "metrics": _metrics(sr, cr),
    }


def _comparison_payloads(
    *,
    c50_native_cr: float = 0.05,
    c100_native_cr: float = 0.06,
    c50_crossing_cr: float = 0.06,
    c100_crossing_cr: float = 0.07,
    c50_corridor_cr: float = 0.40,
    c100_corridor_cr: float = 0.20,
) -> list[dict]:
    return [
        _payload("sa4_c50", "nav_native", 1.0 - c50_native_cr, c50_native_cr),
        _payload("sa4_c50", "native_crossing", 1.0 - c50_crossing_cr, c50_crossing_cr),
        _payload("sa4_c50", "corridor_lateral", 1.0 - c50_corridor_cr, c50_corridor_cr),
        _payload("sa4_c50", "corridor_longitudinal", 0.90, 0.10),
        _payload("sa4_c100", "nav_native", 1.0 - c100_native_cr, c100_native_cr),
        _payload(
            "sa4_c100", "native_crossing", 1.0 - c100_crossing_cr, c100_crossing_cr
        ),
        _payload(
            "sa4_c100", "corridor_lateral", 1.0 - c100_corridor_cr, c100_corridor_cr
        ),
        _payload("sa4_c100", "corridor_longitudinal", 0.95, 0.05),
    ]


def test_runner_is_stage4_only() -> None:
    assert runner.GEOMETRY_STAGE == 4
    assert runner.base.SUPPORTED_GEOMETRY_STAGES == (2, 3)
    assert set(runner.SCENARIOS) == {
        "nav_native",
        "native_crossing",
        "corridor_lateral",
        "corridor_longitudinal",
    }


def test_scene_values_come_from_stage4_contract() -> None:
    values = runner.scene_values()
    spec = STAGE_SCENE_CURRICULUM[4]
    assert values["room_half_extent_m"] == spec.room_half_extent
    assert values["corridor_free_width_m"] == spec.corridor_free_width
    assert values["corridor_static"] == spec.corridor_static_obstacles
    assert values["corridor_dynamic"] == spec.corridor_dynamic_obstacles


def test_queue_hashes_match_checkpoint_files() -> None:
    for checkpoint in queue.CHECKPOINTS:
        assert Path(checkpoint["path"]).is_file()
        assert runner.base.sha256_of(checkpoint["path"]) == checkpoint["sha256"]


def test_longitudinal_budget_is_matched_and_above_the_invalid_r1_budget() -> None:
    planned = queue.cells()
    longitudinal = [
        cell for cell in planned if cell["scenario"] == "corridor_longitudinal"
    ]
    assert len(longitudinal) == 2
    assert {cell["steps"] for cell in longitudinal} == {4000}
    assert all(
        cell["steps"] == 1200
        for cell in planned
        if cell["scenario"] != "corridor_longitudinal"
    )


def test_comparison_selects_c100_when_native_retained_and_corridor_improves() -> None:
    result = queue.compare_checkpoints(_comparison_payloads())
    assert result["native_retention"]["pass"] is True
    assert result["corridor"]["worst_cr_reduction"] == 0.20
    assert result["resume_candidate"] == "sa4_c100"
    assert result["training_extension_authorized"] is False


def test_comparison_holds_when_native_regresses_materially() -> None:
    result = queue.compare_checkpoints(
        _comparison_payloads(c100_native_cr=0.09)
    )
    assert result["native_retention"]["pass"] is False
    assert result["resume_candidate"] is None


def test_comparison_holds_when_corridor_does_not_improve() -> None:
    result = queue.compare_checkpoints(
        _comparison_payloads(c50_corridor_cr=0.20, c100_corridor_cr=0.20)
    )
    assert result["corridor"]["worst_cr_reduction"] == 0.0
    assert result["resume_candidate"] is None


def test_queue_never_authorizes_training_extension() -> None:
    source = Path(queue.__file__).read_text(encoding="utf-8")
    assert '"training_extension_authorized": False' in source
    assert "systemctl" not in source
    assert "train_rnn_car_wdclip" not in source
