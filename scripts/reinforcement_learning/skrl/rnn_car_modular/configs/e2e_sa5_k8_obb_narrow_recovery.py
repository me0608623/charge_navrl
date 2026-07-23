"""Short SA5 retention recovery from the best narrow-passage checkpoint."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_narrow_bridge import CONFIG as _BRIDGE


CONFIG = replace(
    _BRIDGE,
    name="e2e_sa5_k8_obb_narrow_recovery",
    description=(
        "Ten-iteration SA5 recovery segment: 88% original SA5 plus 12% fixed "
        "1.2-1.4m narrow-passage retention, without stress or schedule ramp."
    ),
    checkpoint=(
        "/home/aa/IsaacLab/logs/rnn_car/"
        "sa5_e2e_k8_obb_narrow_bridge_c64000_s42/checkpoint_6400.pt"
    ),
    no_resume_optimizer=False,
    timesteps=1280,
    save_interval=10,
    narrow_passage_fraction=0.12,
    narrow_passage_schedule_steps=1280,
    narrow_passage_final_stress_ratio=0.0,
    narrow_passage_fixed_width_range=(1.2, 1.4),
    narrow_passage_fixed_yaw_limit_deg=4.0,
    tags=_BRIDGE.tags + ("retention_recovery", "fixed_gap_1p2_1p4", "early_stop"),
    notes=(
        "Run in at most three ten-iteration segments. After each segment, evaluate "
        "three-seed SA5 Gate2 and deterministic 1.2m Gate5. Stop at the first "
        "checkpoint where both pass. Reward, PPO, network, DR and optimizer lineage "
        "remain unchanged; 1.0m stress is disabled."
    ),
)
