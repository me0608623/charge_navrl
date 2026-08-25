"""Joint vehicle-speed and corridor-density curriculum for rebuilt SA3-SA4.

This v3 branch is intentionally separate from the frozen v2 lineage. The
2026-08-21 low-density cross-screen showed that P100 pedestrians are not a
valid entry task for the existing policy at either vehicle speed rate. V3
therefore fixes the deployment vehicle contract at 0.7 from SA3 onward and
couples faster pedestrians only to low-density corridor profiles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    make_sim2real_curriculum_config,
)
from rnn_car_modular.experiment_config import ExperimentConfig


SpeedDensityMix = tuple[
    tuple[tuple[int, int], tuple[float, float], float], ...
]

P035 = (0.25, 0.45)
P060 = (0.50, 0.70)
P080 = (0.70, 0.90)
P100 = (0.90, 1.10)

# SA3: establish the deployment vehicle contract and learn P060 only where a
# single pedestrian is not compounded by a dense static layout.
SA3_SPEED_DENSITY_MIX: SpeedDensityMix = (
    ((0, 1), P035, 0.15),
    ((1, 1), P035, 0.15),
    ((0, 1), P060, 0.25),
    ((1, 1), P060, 0.25),
    ((2, 1), P035, 0.10),
    ((3, 1), P035, 0.10),
)

# SA4: add P080 only at 0S1D/1S1D and introduce moderate density at P035/P060.
# P100 remains excluded until the low-density P080 graduation gate passes.
SA4_SPEED_DENSITY_MIX: SpeedDensityMix = (
    ((0, 1), P060, 0.15),
    ((1, 1), P060, 0.20),
    ((0, 1), P080, 0.10),
    ((1, 1), P080, 0.15),
    ((2, 1), P035, 0.15),
    ((2, 1), P060, 0.10),
    ((3, 1), P035, 0.15),
)

VEHICLE_SPEED_RATE = 0.7
VEHICLE_SPEED_OBS_MODE = "ego"
ACTUATOR_DELAY_RANGE = (1, 2)
VLP16_ELIGIBILITY = "valid_return_only"


@dataclass(frozen=True)
class SpeedDensityStageSpec:
    stage: int
    mix: SpeedDensityMix
    entry_gate: str
    graduation_gate: str

    @property
    def pedestrian_speed_envelope(self) -> tuple[float, float]:
        return (
            min(speed_range[0] for _, speed_range, _ in self.mix),
            max(speed_range[1] for _, speed_range, _ in self.mix),
        )


SPEED_DENSITY_CURRICULUM: dict[int, SpeedDensityStageSpec] = {
    3: SpeedDensityStageSpec(
        stage=3,
        mix=SA3_SPEED_DENSITY_MIX,
        entry_gate="accepted operational SA2 c100 checkpoint",
        graduation_gate=(
            "speed_rate=0.7 fixed P060 0S1D and 1S1D each require "
            "SR>=0.90, CR<=0.10, TO<=0.05 plus native/narrow retention"
        ),
    ),
    4: SpeedDensityStageSpec(
        stage=4,
        mix=SA4_SPEED_DENSITY_MIX,
        entry_gate="accepted SA3-v3 checkpoint passing the P060 low-density gate",
        graduation_gate=(
            "speed_rate=0.7 fixed P080 0S1D and 1S1D plus P060 2S1D each "
            "require SR>=0.90, CR<=0.10, TO<=0.05 and native/narrow retention"
        ),
    ),
}


def pending_parent_checkpoint(stage: int) -> str:
    if stage != 4:
        raise ValueError(f"v3 pending parent is defined only for SA4, got SA{stage}")
    return "__PENDING_ACCEPTED_SA3_SPEED_DENSITY_V3_CHECKPOINT__.pt"


def make_speed_density_curriculum_config(
    stage: int,
    *,
    checkpoint: str,
) -> ExperimentConfig:
    """Build a fixed-stage v3 config without changing the frozen v2 table."""
    try:
        spec = SPEED_DENSITY_CURRICULUM[int(stage)]
    except KeyError as exc:
        raise ValueError(f"v3 supports only SA3 and SA4, got SA{stage}") from exc
    if not checkpoint:
        raise ValueError(f"SA{stage} v3 requires an explicit parent checkpoint")

    base = make_sim2real_curriculum_config(stage, checkpoint=checkpoint)
    speed_min, speed_max = spec.pedestrian_speed_envelope
    return replace(
        base,
        name=f"e2e_sa{stage}_k8_obb_speed_density_v3",
        description=(
            f"Rebuilt SA{stage} K8 OBB lineage with deployment speed_rate=0.7 "
            "and a joint pedestrian-speed/corridor-density curriculum."
        ),
        speed_rate=VEHICLE_SPEED_RATE,
        speed_rate_obs=VEHICLE_SPEED_OBS_MODE,
        actuator_delay_range=ACTUATOR_DELAY_RANGE,
        actuator_velocity_scale=(1.0, 1.0),
        actuator_motor_lag=1.0,
        actuator_motor_lag_by_channel=None,
        lidar_distractor_eligibility=VLP16_ELIGIBILITY,
        long_corridor_obstacle_count_mix=None,
        long_corridor_speed_density_mix=spec.mix,
        # Envelope is retained for banners/fallback validation; the joint mix
        # owns the actual per-environment speed draw.
        long_corridor_dynamic_speed_range=(speed_min, speed_max),
        tags=base.tags
        + (
            "sim2real_speed_density_v3",
            "vehicle_speed_rate_0p7_from_sa3",
            "joint_speed_density_profiles",
            "p100_excluded_from_sa3_sa4",
            "valid_return_only",
            "actdelay_u1_2",
        ),
        notes=(
            f"{base.notes} V3 evidence basis: the 2026-08-21 P100 low-density "
            "cross-screen failed even at 0S1D, so P100 is not an entry task. "
            f"This SA{stage} stage uses joint profiles {spec.mix!r}. Faster "
            "pedestrians are coupled only to lower density; density and speed "
            "must never be sampled independently. "
            f"Entry gate: {spec.entry_gate}. Graduation gate: "
            f"{spec.graduation_gate}."
        ),
    )


for _stage, _spec in SPEED_DENSITY_CURRICULUM.items():
    assert _stage == _spec.stage
    assert abs(sum(weight for _, _, weight in _spec.mix) - 1.0) < 1e-12
    assert all(speed_range != P100 for _, speed_range, _ in _spec.mix)

