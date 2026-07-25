"""Continue the fixed SA6 recipe from c100 to c110 with a c105 save point."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_long_cont300 import (
    CONFIG as _LONG_RECIPE,
)


_C100 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_c60_to_c100_s42/checkpoint_5120.pt"
)


CONFIG = replace(
    _LONG_RECIPE,
    name="e2e_sa6_k8_obb_long_c100_to_c110",
    description=(
        "Exact ten-update continuation from c100, saving c105 and c110 to "
        "search the non-monotonic narrow/corridor Pareto frontier."
    ),
    checkpoint=_C100,
    no_resume_optimizer=False,
    timesteps=10 * _LONG_RECIPE.rollout_length,
    save_interval=5,
    tags=_LONG_RECIPE.tags
    + (
        "c100_warm_start",
        "c105_c110_checkpoints",
        "joint_gate_search",
    ),
    notes=(
        "No recipe change: preserve optimizer state, K8+future reward, c20 "
        "narrow retention, SA5 replay, and 4m corridor replay. Corrected "
        "fixed-seed measurements are non-monotonic: c90 narrow crossing "
        "48.02% / corridor CR 9.17%, while c100 reaches narrow 91.02% / "
        "corridor CR 8.01%. Save at five-update resolution and evaluate c105 "
        "then c110. Stop at the first checkpoint satisfying Gate5 and "
        "fixed-goal 4S+2D corridor CR <=10%."
    ),
)
