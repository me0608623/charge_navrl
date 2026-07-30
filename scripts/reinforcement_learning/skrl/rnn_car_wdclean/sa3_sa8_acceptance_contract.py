"""Machine-readable SA3-SA8 acceptance contract.

The prose rationale lives in
``docs/freeze/sa3_sa8_acceptance_matrix_20260730.md``. This module contains
only immutable values and pure validation helpers so a future queue can build
the exact matrix without parsing Markdown or importing Isaac Lab.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
)


SCREEN_SEED = 818
SCREEN_DELAY_STEPS = 1
FORMAL_SEEDS = (515, 616, 717)
FORMAL_DELAY_STEPS = (0, 1, 2)
ACTUATOR_PROFILE = "sa1_delay_only"


@dataclass(frozen=True)
class RateThresholds:
    episodes_min: int
    sr_min: float
    cr_max: float
    to_max: float
    crossing_min: float | None = None
    direct_crossing_min: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


NAV_CLEAN_THRESHOLDS = RateThresholds(
    episodes_min=1000,
    sr_min=0.98,
    cr_max=0.015,
    to_max=0.005,
)

NATIVE_THRESHOLDS = {
    3: RateThresholds(1000, 0.94, 0.05, 0.03),
    4: RateThresholds(1000, 0.92, 0.07, 0.035),
    5: RateThresholds(1000, 0.90, 0.09, 0.04),
    6: RateThresholds(1000, 0.88, 0.11, 0.045),
    7: RateThresholds(1000, 0.86, 0.13, 0.05),
    8: RateThresholds(1000, 0.90, 0.10, 0.05),
}

NARROW_THRESHOLDS = RateThresholds(
    episodes_min=1000,
    sr_min=0.90,
    cr_max=0.05,
    to_max=0.05,
    crossing_min=0.95,
    direct_crossing_min=0.95,
)

CORRIDOR_THRESHOLDS = RateThresholds(
    episodes_min=1000,
    sr_min=0.90,
    cr_max=0.10,
    to_max=0.05,
)

PREVIOUS_STAGE_RETENTION = {
    "sr_drop_max": 0.02,
    "cr_increase_max": 0.02,
}


SCENARIOS_BY_STAGE = {
    3: (
        "nav_clean",
        "nav_native",
        "native_crossing",
        "narrow_range",
        "corridor_lateral_3S1D",
        "corridor_longitudinal_3S1D",
    ),
    4: (
        "nav_clean",
        "nav_native",
        "native_crossing",
        "narrow_range",
        "corridor_lateral_3S2D",
        "corridor_longitudinal_3S2D",
    ),
    5: (
        "nav_clean",
        "nav_native",
        "narrow_range",
        "corridor_lateral_4S2D",
        "corridor_longitudinal_4S2D",
        "corridor_random2d_patrol_4S2D",
        "corridor_exact_mix",
        "corridor_crossing_4S2D",
        "corridor_side_by_side_4S2D",
    ),
    6: (
        "nav_clean",
        "nav_native",
        "narrow_range",
        "corridor_lateral_4S3D",
        "corridor_longitudinal_4S3D",
        "corridor_random2d_patrol_4S3D",
        "corridor_exact_mix",
        "corridor_crossing_4S3D",
        "corridor_side_by_side_4S2D",
    ),
    7: (
        "nav_clean",
        "nav_native",
        "narrow_range",
        "narrow_exact_1p20",
        "corridor_lateral_5S5D",
        "corridor_longitudinal_5S5D",
        "corridor_random2d_wander_5S5D",
        "corridor_exact_mix",
        "corridor_crossing_5S5D",
        "corridor_side_by_side_5S5D",
    ),
    8: (
        "nav_clean",
        "nav_native",
        "narrow_range",
        "narrow_exact_1p20",
        "corridor_lateral_5S5D",
        "corridor_longitudinal_5S5D",
        "corridor_random2d_wander_5S5D",
        "corridor_exact_mix",
        "corridor_crossing_5S5D",
        "corridor_side_by_side_5S5D",
    ),
}


def formal_cells_per_candidate(stage: int) -> int:
    """Return scenario x seed x delay cells for one formal candidate."""
    try:
        scenarios = SCENARIOS_BY_STAGE[int(stage)]
    except KeyError as exc:
        raise ValueError(f"formal stage must be in 3..8, got {stage}") from exc
    return len(scenarios) * len(FORMAL_SEEDS) * len(FORMAL_DELAY_STEPS)


def sampling_tolerance(
    probability: float,
    samples: int,
    *,
    absolute_floor: float,
) -> float:
    """Three-standard-error tolerance with a frozen absolute floor."""
    p = float(probability)
    n = int(samples)
    floor = float(absolute_floor)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0,1], got {probability}")
    if n <= 0:
        raise ValueError(f"samples must be positive, got {samples}")
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError(
            f"absolute_floor must be finite and non-negative, got {absolute_floor}"
        )
    return max(floor, 3.0 * math.sqrt(p * (1.0 - p) / n))


def density_family_tolerance(probability: float, samples: int) -> float:
    return sampling_tolerance(
        probability, samples, absolute_floor=0.03
    )


def interaction_tolerance(probability: float, samples: int) -> float:
    return sampling_tolerance(
        probability, samples, absolute_floor=0.05
    )


def interaction_targets(dynamic_count: int) -> tuple[float, float, float]:
    """Return independent/crossing/side-by-side target probabilities."""
    count = int(dynamic_count)
    if count < 0:
        raise ValueError(f"dynamic_count must be non-negative, got {count}")
    if count < 2:
        return (1.0, 0.0, 0.0)
    if count == 3:
        return (0.75, 0.25, 0.0)
    return (0.60, 0.20, 0.20)


def stage_scene_contract(stage: int) -> dict:
    """Return the training-source scene values used to construct gate cells."""
    try:
        spec = STAGE_SCENE_CURRICULUM[int(stage)]
    except KeyError as exc:
        raise ValueError(f"stage must be in 3..8, got {stage}") from exc
    if int(stage) not in SCENARIOS_BY_STAGE:
        raise ValueError(f"stage must be in 3..8, got {stage}")
    return {
        "stage": spec.stage,
        "arena_size_m": 2.0 * spec.room_half_extent,
        "narrow_width_range_m": list(spec.narrow_width_range),
        "narrow_yaw_limit_deg": spec.narrow_yaw_limit_deg,
        "narrow_exact_width_m": spec.narrow_exact_width,
        "narrow_exact_width_ratio": spec.narrow_exact_width_ratio,
        "corridor_free_width_m": spec.corridor_free_width,
        "corridor_length_m": 10.0,
        "corridor_static_obstacles": spec.corridor_static_obstacles,
        "corridor_dynamic_obstacles": spec.corridor_dynamic_obstacles,
        "corridor_speed_range_m_s": list(spec.corridor_speed_range),
        "corridor_density_mix": spec.corridor_density_mix,
        "corridor_motion_mode": spec.corridor_motion_mode,
        "corridor_motion_weights": spec.corridor_motion_weights,
        "random_2d_kinematics": spec.random_2d_kinematics,
    }


def validate_contract() -> None:
    stages = set(range(3, 9))
    if set(SCENARIOS_BY_STAGE) != stages:
        raise RuntimeError("scenario matrix must cover exactly SA3-SA8")
    if set(NATIVE_THRESHOLDS) != stages:
        raise RuntimeError("native thresholds must cover exactly SA3-SA8")

    expected_counts = {3: 54, 4: 54, 5: 81, 6: 81, 7: 90, 8: 90}
    actual_counts = {
        stage: formal_cells_per_candidate(stage) for stage in sorted(stages)
    }
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"formal cell matrix changed: {actual_counts} != {expected_counts}"
        )

    for stage in sorted(stages):
        spec = STAGE_SCENE_CURRICULUM[stage]
        scenarios = SCENARIOS_BY_STAGE[stage]
        has_random = any("random2d" in name for name in scenarios)
        if has_random != (stage >= 5):
            raise RuntimeError(f"SA{stage} random-2D scenario coverage drifted")
        has_exact = "narrow_exact_1p20" in scenarios
        if has_exact != (spec.narrow_exact_width is not None):
            raise RuntimeError(f"SA{stage} exact-width coverage drifted")
        has_mix = "corridor_exact_mix" in scenarios
        if has_mix != (spec.corridor_density_mix is not None):
            raise RuntimeError(f"SA{stage} density-mix coverage drifted")
        if stage < 5 and spec.corridor_motion_weights != (0.50, 0.50, 0.0):
            raise RuntimeError(f"SA{stage} fixed-family weights drifted")
        if stage >= 5:
            expected_kinematics = "wander" if stage >= 7 else "patrol"
            if spec.random_2d_kinematics != expected_kinematics:
                raise RuntimeError(f"SA{stage} random-2D kinematics drifted")


validate_contract()

