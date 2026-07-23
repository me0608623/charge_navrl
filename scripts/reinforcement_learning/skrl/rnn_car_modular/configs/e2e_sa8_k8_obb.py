"""SA8 entrypoint for the measured OBB collision lineage."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_k8_future_frozen_base import make_stage_config


_BASE = make_stage_config(8)
_NARROW_TEACHER = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt"
)

CONFIG = replace(
    _BASE,
    name="e2e_sa8_k8_obb",
    description="K=8 + future occupancy with measured 0.70x0.60m OBB collision on SA8.",
    use_obb_collision=True,
    use_action_history=True,
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
    teacher_retention_weight=0.10,
    tags=_BASE.tags + (
        "obb_collision",
        "physical_footprint",
        "action_history_2step",
        "narrow_retention_12pct",
        "deployment_corridor_10pct",
    ),
    notes=(
        "OBB lineage warm-started from an accepted SA7 checkpoint. "
        "Collision footprint: half_length=0.35m, half_width=0.30m including wheels, "
        "center_offset_x=-0.128m, buffer=0.10m. Observation includes two applied actions (4D). "
        "LiDAR noise and actuator DR remain disabled. Replay mix is 78% native SA8, "
        "12% fixed 1.2-1.4m narrow passages and 10% 4m x 10m deployment corridors. "
        "The c20 teacher forward-KL applies only to narrow replay. Gates 1-5 plus "
        "the deployment-corridor gate are hard graduation gates."
    ),
)
