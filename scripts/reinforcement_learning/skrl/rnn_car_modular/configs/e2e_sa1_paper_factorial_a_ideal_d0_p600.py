"""Paper realism factorial A: ideal LiDAR and zero actuator delay."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa1_k8_obb_scene_mix import (
    CONFIG as _SCENE_MIX,
)


FORMAL_ITERATIONS = 600
FORMAL_TIMESTEPS = FORMAL_ITERATIONS * _SCENE_MIX.rollout_length
CHECKPOINT_INTERVAL_ITERATIONS = 100
MODEL_INIT_SEED_MATCH_RUN = -1
NEUTRAL_VELOCITY_SCALE = (1.0, 1.0)
NEUTRAL_MOTOR_LAG_ALPHA = 1.0


CONFIG = replace(
    _SCENE_MIX,
    name="e2e_sa1_paper_factorial_a_ideal_d0_p600",
    description=(
        "Paper 2x2 cell A: ideal VLP-16 observations and identity d0 "
        "actuation, trained from scratch for 600 iterations."
    ),
    timesteps=FORMAL_TIMESTEPS,
    save_interval=CHECKPOINT_INTERVAL_ITERATIONS,
    checkpoint=None,
    no_resume_optimizer=True,
    model_init_seed=MODEL_INIT_SEED_MATCH_RUN,
    lidar_no_noise=True,
    vlp16_noise_mode="ideal",
    lidar_distractor_eligibility="valid_return_only",
    no_domain_randomization=True,
    enable_actuator_dr=False,
    actuator_delay_range=(0, 0),
    actuator_velocity_scale=NEUTRAL_VELOCITY_SCALE,
    actuator_motor_lag=NEUTRAL_MOTOR_LAG_ALPHA,
    actuator_motor_lag_by_channel=None,
    obs_delay_steps=(0, 0),
    tags=_SCENE_MIX.tags
    + (
        "paper_realism_factorial_v1",
        "cell_a",
        "ideal_lidar",
        "delay_d0",
        "from_scratch",
    ),
    notes=(
        "READY_NOT_RUN paper factorial control. Event-level physics, sensor "
        "marker and external-force DR are disabled. The action term is off "
        "and all actuator parameters are identity values. model_init_seed=-1 "
        "reuses each run seed immediately before network construction so the "
        "four cells in a seed block start from identical network weights."
    ),
)


assert FORMAL_TIMESTEPS == 76_800
