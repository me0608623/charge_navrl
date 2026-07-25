"""Three-update future-occupancy weight A/B from the joint-gate c10."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic_postkl_cont import (
    CONFIG as _CONT,
)


_POST_KL_C10 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_dynamic2s1d_postkl_cont_c10_s42/"
    "checkpoint_640.pt"
)


CONFIG = replace(
    _CONT,
    name="e2e_sa5_k8_obb_corridor_bridge_dynamic_future05",
    description=(
        "Isolated three-update A/B raising future occupancy from 0.10 to "
        "0.50 after c10 passed aggregate Gate2 and retained Gate5."
    ),
    checkpoint=_POST_KL_C10,
    no_resume_optimizer=False,
    future_occupancy_weight=0.50,
    timesteps=384,
    save_interval=1,
    tags=_CONT.tags + ("future_occupancy_w05", "single_parameter_ablation"),
    notes=(
        "At c10, future occupancy detected 96.9% of corridor dynamic-collision "
        "steps with mean collision-step risk 0.523, but its corridor mean "
        "penalty was only -0.00072 versus +0.1157 progress. Change only the "
        "weight from 0.10 to 0.50; preserve the c3 narrow teacher, post-PPO KL "
        "projection, Adam state, scene mix and all other reward terms."
    ),
)
