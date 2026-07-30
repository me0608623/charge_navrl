"""Stage-wise scene curriculum for the fresh OBB sim-to-real lineage.

Sensor and actuator realism stay fixed across SA1-SA8. Scene difficulty changes
only at stage boundaries: the arena shrinks, narrow passages tighten, corridor
replay becomes denser and faster, and random-2D motion is introduced at SA5.

The running ``sim2real_v1`` SA1 experiment predates this table and remains a
frozen all-at-once robustness arm. This module defines a separate v2 lineage;
it must not be used to relabel the v1 run after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from rnn_car_modular.configs.e2e_k8_future_frozen_base import (
    ROOM_SIZE_BY_STAGE,
)
from rnn_car_modular.configs.e2e_sa1_k8_obb_sim2real_v1 import (
    CONFIG as _SIM2REAL_V1,
)
from rnn_car_modular.experiment_config import ExperimentConfig


DensityMix = tuple[tuple[tuple[int, int], float], ...]


SA5_MODERATE_DENSITY_MIX: DensityMix = (
    ((3, 2), 0.50),
    ((4, 2), 0.50),
)

SA6_NEAR_DEPLOYMENT_DENSITY_MIX: DensityMix = (
    ((3, 2), 0.30),
    ((4, 2), 0.50),
    ((4, 3), 0.20),
)

SA7_DEPLOYMENT_DENSITY_MIX: DensityMix = (
    ((3, 1), 0.25),
    ((4, 2), 0.35),
    ((4, 3), 0.20),
    ((5, 3), 0.15),
    ((5, 5), 0.05),
)

SA8_STRESS_DENSITY_MIX: DensityMix = (
    ((3, 1), 0.15),
    ((4, 2), 0.25),
    ((4, 3), 0.20),
    ((5, 3), 0.20),
    ((5, 4), 0.10),
    ((5, 5), 0.10),
)


@dataclass(frozen=True)
class StageSceneSpec:
    """One stage's replay distribution and geometry."""

    stage: int
    narrow_fraction: float
    narrow_width_range: tuple[float, float]
    narrow_yaw_limit_deg: float
    narrow_exact_width_ratio: float
    corridor_fraction: float
    corridor_free_width: float
    corridor_static_obstacles: int
    corridor_dynamic_obstacles: int
    corridor_speed_range: tuple[float, float]
    corridor_density_mix: DensityMix | None
    corridor_motion_mode: str
    corridor_motion_weights: tuple[float, float, float] | None
    random_2d_kinematics: str

    @property
    def room_half_extent(self) -> float:
        return ROOM_SIZE_BY_STAGE[self.stage]

    @property
    def native_fraction(self) -> float:
        return 1.0 - self.narrow_fraction - self.corridor_fraction

    @property
    def narrow_exact_width(self) -> float | None:
        if self.narrow_exact_width_ratio <= 0.0:
            return None
        return self.narrow_width_range[0]


STAGE_SCENE_CURRICULUM: dict[int, StageSceneSpec] = {
    1: StageSceneSpec(
        stage=1,
        narrow_fraction=0.04,
        narrow_width_range=(1.80, 2.00),
        narrow_yaw_limit_deg=2.0,
        narrow_exact_width_ratio=0.0,
        corridor_fraction=0.02,
        corridor_free_width=5.0,
        corridor_static_obstacles=2,
        corridor_dynamic_obstacles=1,
        corridor_speed_range=(0.15, 0.25),
        corridor_density_mix=None,
        corridor_motion_mode="lateral",
        corridor_motion_weights=None,
        random_2d_kinematics="patrol",
    ),
    2: StageSceneSpec(
        stage=2,
        narrow_fraction=0.06,
        narrow_width_range=(1.60, 1.80),
        narrow_yaw_limit_deg=3.0,
        narrow_exact_width_ratio=0.0,
        corridor_fraction=0.04,
        corridor_free_width=4.8,
        corridor_static_obstacles=3,
        corridor_dynamic_obstacles=1,
        corridor_speed_range=(0.18, 0.30),
        corridor_density_mix=None,
        corridor_motion_mode="env_stratified",
        corridor_motion_weights=(0.50, 0.50, 0.0),
        random_2d_kinematics="patrol",
    ),
    3: StageSceneSpec(
        stage=3,
        narrow_fraction=0.08,
        narrow_width_range=(1.50, 1.70),
        narrow_yaw_limit_deg=4.0,
        narrow_exact_width_ratio=0.0,
        corridor_fraction=0.06,
        corridor_free_width=4.6,
        corridor_static_obstacles=3,
        corridor_dynamic_obstacles=1,
        corridor_speed_range=(0.20, 0.35),
        corridor_density_mix=None,
        corridor_motion_mode="env_stratified",
        corridor_motion_weights=(0.50, 0.50, 0.0),
        random_2d_kinematics="patrol",
    ),
    4: StageSceneSpec(
        stage=4,
        narrow_fraction=0.10,
        narrow_width_range=(1.40, 1.60),
        narrow_yaw_limit_deg=4.0,
        narrow_exact_width_ratio=0.0,
        corridor_fraction=0.08,
        corridor_free_width=4.4,
        corridor_static_obstacles=3,
        corridor_dynamic_obstacles=2,
        corridor_speed_range=(0.22, 0.40),
        corridor_density_mix=None,
        corridor_motion_mode="env_stratified",
        corridor_motion_weights=(0.50, 0.50, 0.0),
        random_2d_kinematics="patrol",
    ),
    5: StageSceneSpec(
        stage=5,
        narrow_fraction=0.12,
        narrow_width_range=(1.30, 1.50),
        narrow_yaw_limit_deg=4.0,
        narrow_exact_width_ratio=0.0,
        corridor_fraction=0.10,
        corridor_free_width=4.2,
        corridor_static_obstacles=4,
        corridor_dynamic_obstacles=2,
        corridor_speed_range=(0.25, 0.45),
        corridor_density_mix=SA5_MODERATE_DENSITY_MIX,
        corridor_motion_mode="env_stratified",
        corridor_motion_weights=None,
        random_2d_kinematics="patrol",
    ),
    6: StageSceneSpec(
        stage=6,
        narrow_fraction=0.12,
        narrow_width_range=(1.20, 1.40),
        narrow_yaw_limit_deg=4.0,
        narrow_exact_width_ratio=0.0,
        corridor_fraction=0.10,
        corridor_free_width=4.0,
        corridor_static_obstacles=4,
        corridor_dynamic_obstacles=2,
        corridor_speed_range=(0.27, 0.50),
        corridor_density_mix=SA6_NEAR_DEPLOYMENT_DENSITY_MIX,
        corridor_motion_mode="env_stratified",
        corridor_motion_weights=None,
        random_2d_kinematics="patrol",
    ),
    7: StageSceneSpec(
        stage=7,
        narrow_fraction=0.12,
        narrow_width_range=(1.20, 1.40),
        narrow_yaw_limit_deg=6.0,
        narrow_exact_width_ratio=0.25,
        corridor_fraction=0.10,
        corridor_free_width=4.0,
        corridor_static_obstacles=4,
        corridor_dynamic_obstacles=2,
        corridor_speed_range=(0.30, 0.60),
        corridor_density_mix=SA7_DEPLOYMENT_DENSITY_MIX,
        corridor_motion_mode="env_stratified",
        corridor_motion_weights=None,
        random_2d_kinematics="wander",
    ),
    8: StageSceneSpec(
        stage=8,
        narrow_fraction=0.12,
        narrow_width_range=(1.20, 1.40),
        narrow_yaw_limit_deg=8.0,
        narrow_exact_width_ratio=0.50,
        corridor_fraction=0.10,
        corridor_free_width=4.0,
        corridor_static_obstacles=4,
        corridor_dynamic_obstacles=2,
        corridor_speed_range=(0.30, 0.60),
        corridor_density_mix=SA8_STRESS_DENSITY_MIX,
        corridor_motion_mode="env_stratified",
        corridor_motion_weights=None,
        random_2d_kinematics="wander",
    ),
}


_DROP_BASE_TAGS = frozenset(
    {
        "sa1",
        "from_scratch",
        "actuator_delay_only",
        "native_78pct",
        "narrow_retention_12pct",
        "deployment_corridor_10pct",
        "wander_corridor_kinematics",
        "audited_corridor_density_mix",
    }
)
_INVARIANT_TAGS = tuple(
    tag for tag in _SIM2REAL_V1.tags if tag not in _DROP_BASE_TAGS
)


def pending_parent_checkpoint(stage: int) -> str:
    """Return the fail-closed checkpoint sentinel for SA2-SA8."""
    if stage not in range(2, 9):
        raise ValueError(f"parent checkpoint sentinel requires SA2-SA8, got SA{stage}")
    return (
        f"__PENDING_ACCEPTED_SA{stage - 1}_SIM2REAL_CURRICULUM_V2_"
        "CHECKPOINT__.pt"
    )


def make_sim2real_curriculum_config(
    stage: int,
    *,
    checkpoint: str | None,
) -> ExperimentConfig:
    """Build one fixed-stage config while preserving the sim-to-real contract."""
    try:
        spec = STAGE_SCENE_CURRICULUM[int(stage)]
    except KeyError as exc:
        raise ValueError(f"stage must be in 1..8, got {stage}") from exc

    expected_checkpoint = None if stage == 1 else pending_parent_checkpoint(stage)
    if stage == 1 and checkpoint is not None:
        raise ValueError("SA1 curriculum v2 must start from scratch")
    if stage > 1 and not checkpoint:
        raise ValueError(f"SA{stage} curriculum v2 requires a parent checkpoint")

    density_label = (
        "fixed"
        if spec.corridor_density_mix is None
        else "mixed_density"
    )
    stage_tag = f"sa{stage}"
    reset_tag = (
        f"scene_reset_{round(spec.native_fraction * 100):02d}_"
        f"{round(spec.narrow_fraction * 100):02d}_"
        f"{round(spec.corridor_fraction * 100):02d}"
    )
    notes = (
        f"Canonical sim-to-real curriculum v2 SA{stage}. Nominal reset shares "
        f"are native={spec.native_fraction:.2f}, narrow={spec.narrow_fraction:.2f}, "
        f"corridor={spec.corridor_fraction:.2f}; rollout shares may differ because "
        "episode lengths differ. Narrow replay uses width "
        f"{spec.narrow_width_range[0]:.2f}-{spec.narrow_width_range[1]:.2f} m "
        f"and yaw +/-{spec.narrow_yaw_limit_deg:.1f} deg. Corridor replay uses "
        f"{spec.corridor_free_width:.1f} m free width, "
        f"{spec.corridor_static_obstacles}S+{spec.corridor_dynamic_obstacles}D "
        f"legacy counts ({density_label}), speed "
        f"{spec.corridor_speed_range[0]:.2f}-{spec.corridor_speed_range[1]:.2f} "
        f"m/s, motion={spec.corridor_motion_mode}, and random_2d="
        f"{spec.random_2d_kinematics}. Measured VLP-16 full noise, decoded-command "
        "delay U{0,1,2}=0/200/400 ms, K8/83D issued-action history, OBB collision, "
        "reward, model and optimizer settings remain fixed. SA2-SA8 are fail-closed "
        "until the preceding stage passes its graduation gates."
    )

    return replace(
        _SIM2REAL_V1,
        name=f"e2e_sa{stage}_k8_obb_sim2real_curriculum_v2",
        description=(
            f"Canonical SA{stage} K8 OBB sim-to-real curriculum v2 with "
            "stage-wise narrow-passage and corridor difficulty."
        ),
        initial_stage=stage,
        room_size=spec.room_half_extent,
        checkpoint=checkpoint if stage > 1 else expected_checkpoint,
        no_resume_optimizer=True,
        previous_stage_replay_fraction=0.0,
        narrow_passage_fraction=spec.narrow_fraction,
        narrow_passage_segment_length=spec.room_half_extent,
        narrow_passage_final_stress_ratio=0.0,
        narrow_passage_fixed_width_range=spec.narrow_width_range,
        narrow_passage_fixed_yaw_limit_deg=spec.narrow_yaw_limit_deg,
        narrow_passage_exact_width=spec.narrow_exact_width,
        narrow_passage_exact_width_ratio=spec.narrow_exact_width_ratio,
        long_corridor_fraction=spec.corridor_fraction,
        long_corridor_free_width=spec.corridor_free_width,
        long_corridor_length=10.0,
        long_corridor_static_obstacles=spec.corridor_static_obstacles,
        long_corridor_dynamic_obstacles=spec.corridor_dynamic_obstacles,
        long_corridor_obstacle_count_mix=spec.corridor_density_mix,
        long_corridor_gate_aligned_share=0.0,
        long_corridor_dynamic_speed_range=spec.corridor_speed_range,
        long_corridor_dynamic_motion_mode=spec.corridor_motion_mode,
        long_corridor_dynamic_motion_weights=spec.corridor_motion_weights,
        long_corridor_random_2d_kinematics=spec.random_2d_kinematics,
        tags=_INVARIANT_TAGS
        + (
            "sim2real_curriculum_v2",
            stage_tag,
            "from_scratch" if stage == 1 else "warm_start",
            reset_tag,
            density_label,
        ),
        notes=notes,
    )


for _stage, _spec in STAGE_SCENE_CURRICULUM.items():
    assert _stage == _spec.stage
    assert abs(
        _spec.native_fraction
        + _spec.narrow_fraction
        + _spec.corridor_fraction
        - 1.0
    ) < 1e-12

