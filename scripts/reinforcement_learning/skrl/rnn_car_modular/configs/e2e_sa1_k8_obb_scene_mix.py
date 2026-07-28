"""Fresh SA1 OBB control with retained narrow-passage and corridor scenes.

This config is the scene-distribution parent for the new sim-to-real lineage.
It deliberately carries no sensor or actuator randomization so the delay-only
and measured-LiDAR arms can each add one locked factor.

Reset distribution:

    78% native SA1
    12% fixed 1.2-1.4 m narrow passages
    10% 4 m x 10 m deployment corridors

The corridor uses the audited per-env density mix and deployment ``wander``
kinematics. It does not use the failed SA7.1 gate-aligned split.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa1_k8_obb import CONFIG as _CLEAN_SA1
from rnn_car_modular.configs.e2e_sa7_wander_from_w1c10 import (
    SA7_DENSITY_MIX,
)


NATIVE_SCENE_FRACTION = 0.78
NARROW_PASSAGE_FRACTION = 0.12
LONG_CORRIDOR_FRACTION = 0.10
SA1_NARROW_SEGMENT_LENGTH_M = 10.0
SA1_ACTION_HISTORY_ACCEL_NORMALIZER = 0.5
SA1_ACTION_HISTORY_OMEGA_NORMALIZER = 1.2
DEPLOYMENT_CORRIDOR_DENSITY_MIX = SA7_DENSITY_MIX


CONFIG = replace(
    _CLEAN_SA1,
    name="e2e_sa1_k8_obb_scene_mix",
    description=(
        "Fresh SA1 K8 OBB control with 78% native scenes, 12% fixed narrow "
        "passages and 10% deployment corridors."
    ),
    checkpoint=None,
    no_resume_optimizer=True,
    action_history_accel_normalizer=SA1_ACTION_HISTORY_ACCEL_NORMALIZER,
    action_history_omega_normalizer=SA1_ACTION_HISTORY_OMEGA_NORMALIZER,
    previous_stage_replay_fraction=0.0,
    narrow_passage_fraction=NARROW_PASSAGE_FRACTION,
    # SA1 uses a 20 x 20 m room. The historical 9 m segment leaves a bypass
    # at the extreme +/-1 m gap centers; 10 m closes every sampled endpoint.
    narrow_passage_segment_length=SA1_NARROW_SEGMENT_LENGTH_M,
    narrow_passage_final_stress_ratio=0.0,
    narrow_passage_fixed_width_range=(1.2, 1.4),
    narrow_passage_fixed_yaw_limit_deg=4.0,
    long_corridor_fraction=LONG_CORRIDOR_FRACTION,
    long_corridor_free_width=4.0,
    long_corridor_length=10.0,
    long_corridor_static_obstacles=4,
    long_corridor_dynamic_obstacles=2,
    long_corridor_dynamic_speed_range=(0.30, 0.60),
    long_corridor_obstacle_count_mix=DEPLOYMENT_CORRIDOR_DENSITY_MIX,
    long_corridor_dynamic_motion_mode="env_stratified",
    long_corridor_random_2d_kinematics="wander",
    long_corridor_gate_aligned_share=0.0,
    tags=_CLEAN_SA1.tags
    + (
        "native_78pct",
        "narrow_retention_12pct",
        "deployment_corridor_10pct",
        "wander_corridor_kinematics",
        "audited_corridor_density_mix",
    ),
    notes=(
        "Scene-distribution control for the fresh SA1 sim-to-real lineage. "
        "The reset classes are disjoint: 78% native SA1, 12% fixed 1.2-1.4 m "
        "narrow passages and 10% 4 m x 10 m deployment corridors. Narrow "
        "barrier segments are 10 m long so every sampled gap in the 20 m room "
        "is non-bypassable. The two-step issued-action history is normalized "
        "by the actual decoder limits (0.5 m/s^2 and 1.2 rad/s), rather than "
        "the saturated legacy divisors. Corridor "
        "obstacle counts use the audited 25% 3S1D / 35% 4S2D / 20% 4S3D / "
        "15% 5S3D / 5% 5S5D mix, with deployment wander kinematics and "
        "env-stratified motion families. Gate-aligned scenes stay off because "
        "the paired SA7.1 experiment failed; no SA7 policy, optimizer or "
        "teacher checkpoint is inherited. LiDAR noise and actuator DR remain "
        "disabled in this control."
    ),
)


assert abs(
    NATIVE_SCENE_FRACTION
    + NARROW_PASSAGE_FRACTION
    + LONG_CORRIDOR_FRACTION
    - 1.0
) < 1e-12
