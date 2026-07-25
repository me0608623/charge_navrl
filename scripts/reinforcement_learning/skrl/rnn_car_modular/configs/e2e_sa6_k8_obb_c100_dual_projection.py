"""Merge c20 narrow behavior into the stronger c100 corridor policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_dual_projection import (
    CONFIG as _C50_DUAL,
)


_CORRIDOR_C100 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_c60_to_c100_s42/checkpoint_5120.pt"
)


CONFIG = replace(
    _C50_DUAL,
    name="e2e_sa6_k8_obb_c100_dual_projection",
    description=(
        "Projection-only c100 Pareto merge: recover c20 narrow behavior while "
        "functionally anchoring the stronger c100 policy on non-narrow frames."
    ),
    checkpoint=_CORRIDOR_C100,
    previous_stage_teacher_checkpoint=_CORRIDOR_C100,
    teacher_retention_post_anchor_weight=0.75,
    tags=tuple(
        tag for tag in _C50_DUAL.tags
        if tag not in {
            "c50_corridor_candidate",
            "c50_non_narrow_anchor",
            "anchor_lambda_0p5",
        }
    )
    + (
        "c100_corridor_candidate",
        "c100_non_narrow_anchor",
        "anchor_lambda_0p75",
    ),
    notes=(
        "Forty additional uninterrupted PPO updates reduced corrected "
        "fixed-goal 4S+2D corridor CR from c60 11.04% to c90 9.17% and c100 "
        "8.01%. C100 supplies 1.99 percentage points of projection margin. "
        "Use eight full-actor updates, no PPO, c20 on narrow frames, and c100 "
        "on disjoint non-narrow frames. Lambda 0.75 is the bounded midpoint "
        "between the prior c50 lambda 0.5 Gate5-safe result and lambda 1.0 "
        "corridor-preserving but Gate5-failing result."
    ),
)
