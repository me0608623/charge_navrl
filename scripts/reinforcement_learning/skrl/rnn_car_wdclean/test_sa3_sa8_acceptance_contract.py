"""CPU-only tests for the frozen SA3-SA8 acceptance contract."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import sa3_sa8_acceptance_contract as contract


def test_formal_axes_are_fixed():
    assert contract.SCREEN_SEED == 818
    assert contract.SCREEN_DELAY_STEPS == 1
    assert contract.FORMAL_SEEDS == (515, 616, 717)
    assert contract.FORMAL_DELAY_STEPS == (0, 1, 2)
    assert contract.ACTUATOR_PROFILE == "sa1_delay_only"


def test_scenario_matrix_covers_exactly_sa3_to_sa8():
    assert set(contract.SCENARIOS_BY_STAGE) == set(range(3, 9))
    assert set(contract.NATIVE_THRESHOLDS) == set(range(3, 9))


@pytest.mark.parametrize(
    ("stage", "scenarios", "cells"),
    [
        (3, 6, 54),
        (4, 6, 54),
        (5, 9, 81),
        (6, 9, 81),
        (7, 10, 90),
        (8, 10, 90),
    ],
)
def test_formal_cell_counts(stage, scenarios, cells):
    assert len(contract.SCENARIOS_BY_STAGE[stage]) == scenarios
    assert contract.formal_cells_per_candidate(stage) == cells


def test_invalid_formal_stage_is_rejected():
    with pytest.raises(ValueError, match="3..8"):
        contract.formal_cells_per_candidate(2)


def test_navigation_core_remains_strict_at_every_stage():
    t = contract.NAV_CLEAN_THRESHOLDS
    assert t.episodes_min == 1000
    assert (t.sr_min, t.cr_max, t.to_max) == (0.98, 0.015, 0.005)


def test_sa8_native_gate_tightens_again():
    sa7 = contract.NATIVE_THRESHOLDS[7]
    sa8 = contract.NATIVE_THRESHOLDS[8]
    assert sa8.sr_min > sa7.sr_min
    assert sa8.cr_max < sa7.cr_max


def test_narrow_and_corridor_bars_are_frozen():
    narrow = contract.NARROW_THRESHOLDS
    corridor = contract.CORRIDOR_THRESHOLDS
    assert narrow.to_dict() == {
        "episodes_min": 1000,
        "sr_min": 0.90,
        "cr_max": 0.05,
        "to_max": 0.05,
        "crossing_min": 0.95,
        "direct_crossing_min": 0.95,
    }
    assert corridor.to_dict() == {
        "episodes_min": 1000,
        "sr_min": 0.90,
        "cr_max": 0.10,
        "to_max": 0.05,
        "crossing_min": None,
        "direct_crossing_min": None,
    }


@pytest.mark.parametrize(
    ("dynamic_count", "expected"),
    [
        (0, (1.0, 0.0, 0.0)),
        (1, (1.0, 0.0, 0.0)),
        (2, (0.60, 0.20, 0.20)),
        (3, (0.75, 0.25, 0.0)),
        (4, (0.60, 0.20, 0.20)),
        (5, (0.60, 0.20, 0.20)),
    ],
)
def test_interaction_targets_match_sampler_policy(dynamic_count, expected):
    assert contract.interaction_targets(dynamic_count) == expected


def test_negative_dynamic_count_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        contract.interaction_targets(-1)


def test_sampling_tolerance_uses_floor_when_sampling_error_is_small():
    assert contract.density_family_tolerance(0.5, 100_000) == 0.03
    assert contract.interaction_tolerance(0.2, 100_000) == 0.05


def test_sampling_tolerance_expands_for_small_groups():
    tolerance = contract.interaction_tolerance(0.2, 10)
    assert tolerance > 0.05


@pytest.mark.parametrize(
    ("probability", "samples", "floor"),
    [(-0.1, 10, 0.03), (1.1, 10, 0.03), (0.5, 0, 0.03), (0.5, 10, -0.1)],
)
def test_invalid_sampling_inputs_are_rejected(probability, samples, floor):
    with pytest.raises(ValueError):
        contract.sampling_tolerance(
            probability, samples, absolute_floor=floor
        )


@pytest.mark.parametrize("stage", range(3, 9))
def test_scene_contract_is_read_from_training_source(stage):
    scene = contract.stage_scene_contract(stage)
    spec = contract.STAGE_SCENE_CURRICULUM[stage]
    assert scene["arena_size_m"] == 2.0 * spec.room_half_extent
    assert scene["narrow_width_range_m"] == list(spec.narrow_width_range)
    assert scene["corridor_free_width_m"] == spec.corridor_free_width
    assert scene["corridor_speed_range_m_s"] == list(
        spec.corridor_speed_range
    )


@pytest.mark.parametrize("stage", (3, 4))
def test_sa3_sa4_have_no_random2d_or_count_mix(stage):
    scenarios = contract.SCENARIOS_BY_STAGE[stage]
    assert not any("random2d" in name for name in scenarios)
    assert "corridor_exact_mix" not in scenarios


@pytest.mark.parametrize("stage", (5, 6, 7, 8))
def test_sa5_plus_cover_random2d_mix_and_interactions(stage):
    scenarios = contract.SCENARIOS_BY_STAGE[stage]
    assert any("random2d" in name for name in scenarios)
    assert "corridor_exact_mix" in scenarios
    assert any("corridor_crossing" in name for name in scenarios)
    assert any("corridor_side_by_side" in name for name in scenarios)


@pytest.mark.parametrize("stage", (7, 8))
def test_sa7_sa8_cover_exact_narrow_and_wander(stage):
    scenarios = contract.SCENARIOS_BY_STAGE[stage]
    assert "narrow_exact_1p20" in scenarios
    assert any("random2d_wander" in name for name in scenarios)


def test_contract_self_validation_passes():
    contract.validate_contract()

