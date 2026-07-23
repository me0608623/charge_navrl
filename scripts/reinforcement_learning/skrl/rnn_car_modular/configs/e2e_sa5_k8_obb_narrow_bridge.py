"""SA5 OBB narrow-passage bridge warm-started from the accepted c64000."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb import CONFIG as _SA5


CONFIG = replace(
    _SA5,
    name="e2e_sa5_k8_obb_narrow_bridge",
    description=(
        "SA5 warm-start bridge: 88% original SA5 and 12% guaranteed-solvable "
        "narrow wall gaps, with no reward or network changes."
    ),
    checkpoint=(
        "/home/aa/IsaacLab/logs/rnn_car/"
        "sa5_e2e_k8_obb_acthist_future_managed_s42/checkpoint_64000.pt"
    ),
    no_resume_optimizer=False,
    timesteps=19200,
    save_interval=50,
    narrow_passage_fraction=0.12,
    narrow_passage_schedule_steps=19200,
    narrow_passage_segment_length=9.0,
    narrow_passage_final_stress_ratio=0.25,
    tags=_SA5.tags + ("narrow_passage_bridge", "warm_start_c64000", "reward_unchanged"),
    notes=(
        "Warm-start from accepted SA5 checkpoint_64000 with optimizer state. "
        "Expected episode mix: 88% unchanged SA5, 12% controlled barrier gaps. "
        "Schedule: 1.6-1.4m -> 1.4-1.2m, then up to 25% of injected episodes "
        "at 1.0-1.2m; yaw grows from +/-2 to +/-10 degrees. Every injected "
        "scene passes constructive measured-OBB solvability validation."
    ),
)
