"""SA5 joint recovery: Gate2 freedom, narrow retention and corridor replay."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_narrow_recovery import (
    CONFIG as _RECOVERY,
)


_C20 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_narrow_recovery_seg2_c1280_s42/checkpoint_1280.pt"
)


CONFIG = replace(
    _RECOVERY,
    name="e2e_sa5_k8_obb_teacher_corridor_recovery",
    description=(
        "Ten-iteration SA5 recovery from c20: 78% original, 12% fixed "
        "1.2-1.4m narrow replay with c20 forward-KL retention, and 10% "
        "4m x 10m deployment corridor replay with 4 static + 2 dynamic obstacles."
    ),
    checkpoint=_C20,
    no_resume_optimizer=False,
    timesteps=1280,
    save_interval=2,
    narrow_passage_fraction=0.12,
    narrow_passage_schedule_steps=1280,
    narrow_passage_final_stress_ratio=0.0,
    narrow_passage_fixed_width_range=(1.2, 1.4),
    narrow_passage_fixed_yaw_limit_deg=4.0,
    narrow_passage_exact_width=1.2,
    narrow_passage_exact_width_ratio=0.50,
    long_corridor_fraction=0.10,
    long_corridor_free_width=4.0,
    long_corridor_length=10.0,
    long_corridor_static_obstacles=4,
    long_corridor_dynamic_obstacles=2,
    long_corridor_dynamic_speed_range=(0.30, 0.60),
    teacher_retention_checkpoint=_C20,
    teacher_retention_weight=0.10,
    tags=_RECOVERY.tags
    + (
        "teacher_retention_beta0p1",
        "deployment_corridor_4x10",
        "scene_mix_78_12_10",
    ),
    notes=(
        "Reward, K8 network, OBB collision, action-history observation and DR are "
        "unchanged. Forward KL(c20||current) is applied only to narrow replay "
        "frames. Save every two iterations; evaluate Gate2, 1.2m Gate5 and the "
        "4m x 10m corridor gate at each checkpoint, stopping at the first joint pass."
    ),
)
