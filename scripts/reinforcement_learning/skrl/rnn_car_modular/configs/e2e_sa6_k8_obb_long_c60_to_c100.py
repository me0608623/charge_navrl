"""Continue the fixed SA6 recipe from long-continuation c60 to c100."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_long_cont300 import (
    CONFIG as _LONG_RECIPE,
)


_C60 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_cont300_c128_s42/checkpoint_7680.pt"
)


CONFIG = replace(
    _LONG_RECIPE,
    name="e2e_sa6_k8_obb_long_c60_to_c100",
    description=(
        "Uninterrupted forty-iteration continuation from c60, preserving the "
        "fixed SA6 optimizer, reward, network, and replay recipe."
    ),
    checkpoint=_C60,
    no_resume_optimizer=False,
    timesteps=40 * _LONG_RECIPE.rollout_length,
    save_interval=10,
    tags=_LONG_RECIPE.tags
    + (
        "c60_warm_start",
        "c70_to_c100_checkpoints",
    ),
    notes=(
        "Corrected fixed-goal 4S+2D evaluation measured c50 CR 13.20% and "
        "c60 CR 11.04%. Continue the exact recipe in one process and save "
        "c70, c80, c90, and c100 equivalents every ten updates. Do not alter "
        "reward, replay fractions, network, entropy, or c20 beta 0.30. After "
        "training, select the first fixed-seed checkpoint with corridor "
        "CR <= 10%, then repair Gate5 with full-actor projection."
    ),
)
