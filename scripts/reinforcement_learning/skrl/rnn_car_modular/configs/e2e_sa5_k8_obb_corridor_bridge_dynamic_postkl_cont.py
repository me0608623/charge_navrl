"""Five uninterrupted 2S+1D updates from the best retained c5 policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic_postkl import (
    CONFIG as _POST_KL,
)


_POST_KL_C5 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_dynamic2s1d_postkl_c5_s42/"
    "checkpoint_128.pt"
)


CONFIG = replace(
    _POST_KL,
    name="e2e_sa5_k8_obb_corridor_bridge_dynamic_postkl_cont",
    description=(
        "Continue five updates in one simulator process from the Gate5-perfect "
        "post-KL c5 policy, preserving Adam state and narrow projection."
    ),
    checkpoint=_POST_KL_C5,
    no_resume_optimizer=False,
    timesteps=640,
    save_interval=1,
    tags=_POST_KL.tags
    + ("five_update_continuation", "continuous_scene_rng"),
    notes=(
        "The prior c4/c5/c6 experiments restarted the simulator after every "
        "update, which replayed seed 42's first scene batch even though Adam "
        "state was restored. This run keeps one process alive for five updates "
        "so later rollouts receive new resets. Reward and post-PPO projection "
        "parameters remain unchanged."
    ),
)
