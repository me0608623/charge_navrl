"""Eight-step full-actor narrow projection from the corridor-improved c50."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_fullactor_projection import (
    CONFIG as _FOUR_STEP,
)


CONFIG = replace(
    _FOUR_STEP,
    name="e2e_sa6_k8_obb_c50_narrow_fullactor_projection8",
    description=(
        "Projection-only c50 diagnostic with eight full-batch updates through "
        "the LiDAR CNN encoder and policy head."
    ),
    teacher_retention_post_kl_epochs=8,
    tags=_FOUR_STEP.tags + ("projection_steps_8",),
    notes=(
        "The matched four-step full-actor projection raised fixed-seed Gate5 "
        "crossing from 78.81% to 93.11%, but missed the 95% crossing and 5% "
        "collision hard gates. This run changes only projection steps from "
        "four to eight and still performs no PPO update."
    ),
)
