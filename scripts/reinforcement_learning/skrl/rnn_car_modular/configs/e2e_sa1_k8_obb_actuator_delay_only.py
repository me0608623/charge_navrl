"""SA1 scene-mix plus actuator-delay control for the sim-to-real lineage.

This arm starts from the fresh SA1 scene-distribution control, which already
contains native, narrow-passage and deployment-corridor scenes. It introduces
only the measured command dead-time support U{0,1,2} at control_dt=0.2 s.
Uncalibrated velocity scaling and motor response lag are deliberately neutral:

    d ~ U{0,1,2}  ->  0 / 200 / 400 ms
    scale_v = scale_omega = 1
    alpha_v = alpha_omega = 1

The separate delay-only arm is retained even when the combined LiDAR arm is
trained, so a failure can be attributed to sensor noise or command delay.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa1_k8_obb_scene_mix import (
    CONFIG as _SA1_SCENE_MIX,
)


ACTUATOR_DELAY_SUPPORT_STEPS = (0, 2)
NEUTRAL_VELOCITY_SCALE = (1.0, 1.0)
NEUTRAL_MOTOR_LAG_ALPHA = 1.0


CONFIG = replace(
    _SA1_SCENE_MIX,
    name="e2e_sa1_k8_obb_actuator_delay_only",
    description=(
        "Fresh SA1 K8 OBB scene-mix training with decoded-command delay "
        "support U{0,1,2}=0/200/400 ms and no uncalibrated scale or motor lag."
    ),
    checkpoint=None,
    no_resume_optimizer=True,
    enable_actuator_dr=True,
    actuator_delay_range=ACTUATOR_DELAY_SUPPORT_STEPS,
    actuator_velocity_scale=NEUTRAL_VELOCITY_SCALE,
    actuator_motor_lag=NEUTRAL_MOTOR_LAG_ALPHA,
    actuator_motor_lag_by_channel=None,
    obs_delay_steps=(0, 0),
    tags=_SA1_SCENE_MIX.tags
    + (
        "sim2real_new_lineage",
        "actuator_delay_only",
        "delay_u012",
        "from_scratch",
    ),
    notes=(
        "Delay-only arm for the new SA1 sim-to-real lineage. It starts from "
        "random model/optimizer state with the same 78/12/10 native, narrow "
        "and corridor scene mix as its control, then adds only the frozen "
        "delay support U{0,1,2} steps at control_dt=0.2 s. Velocity scale is "
        "exactly one and motor lag alpha is exactly one because neither has "
        "a completed vehicle sysid measurement. Observation delay remains "
        "hard off. The 4D issued-command history reduces the partial "
        "observability introduced by d<=2, but does not expose the sampled "
        "delay value itself."
    ),
)
