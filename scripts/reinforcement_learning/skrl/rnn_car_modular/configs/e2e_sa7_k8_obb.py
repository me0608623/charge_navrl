"""SA7 entrypoint for the measured OBB collision lineage."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_k8_future_frozen_base import make_stage_config


_BASE = make_stage_config(7)
_NARROW_TEACHER = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt"
)

CONFIG = replace(
    _BASE,
    name="e2e_sa7_k8_obb",
    description="K=8 + future occupancy with measured 0.70x0.60m OBB collision on SA7.",
    use_obb_collision=True,
    use_action_history=True,
    previous_stage_replay_fraction=0.10,
    narrow_passage_fraction=0.12,
    narrow_passage_final_stress_ratio=0.0,
    narrow_passage_fixed_width_range=(1.2, 1.4),
    narrow_passage_fixed_yaw_limit_deg=4.0,
    long_corridor_fraction=0.10,
    long_corridor_free_width=4.0,
    long_corridor_length=10.0,
    long_corridor_static_obstacles=4,
    long_corridor_dynamic_obstacles=2,
    long_corridor_dynamic_speed_range=(0.30, 0.60),
    teacher_retention_checkpoint=_NARROW_TEACHER,
    teacher_retention_weight=0.30,
    tags=_BASE.tags + (
        "obb_collision",
        "physical_footprint",
        "action_history_2step",
        "sa5_general_replay_10pct",
        "narrow_retention_12pct",
        "deployment_corridor_10pct",
    ),
    notes=(
        "OBB lineage warm-started from an accepted SA6 checkpoint. "
        "Collision footprint: half_length=0.35m, half_width=0.30m including wheels, "
        "center_offset_x=-0.128m, buffer=0.10m. Observation includes two applied actions (4D). "
        "LiDAR noise and actuator DR remain disabled. Replay mix is 68% native SA7, "
        "10% SA5-general (10S+3D, 2-3 walls at 4m), 12% fixed 1.2-1.4m narrow "
        "passages and 10% 4m x 10m deployment corridors. "
        "The calibrated beta=0.30 c20 forward-KL applies only to narrow replay. Gates 1-5 plus "
        "the deployment-corridor gate are hard graduation gates."
    ),
)
