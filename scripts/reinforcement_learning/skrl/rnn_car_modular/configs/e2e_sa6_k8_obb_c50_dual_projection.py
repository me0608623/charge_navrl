"""Dual functional projection from c50 toward c20 only on narrow frames."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_fullactor_projection8 import (
    CONFIG as _NARROW_FULL_ACTOR,
)


_CORRIDOR_C50 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_cont300_c128_s42/checkpoint_6400.pt"
)


CONFIG = replace(
    _NARROW_FULL_ACTOR,
    name="e2e_sa6_k8_obb_c50_dual_projection",
    description=(
        "Projection-only Pareto recovery from c50. Match c20 on narrow replay "
        "frames while anchoring c50 on every non-narrow frame."
    ),
    previous_stage_teacher_checkpoint=_CORRIDOR_C50,
    previous_stage_teacher_retention_weight=0.0,
    previous_stage_teacher_scope="non_narrow",
    teacher_retention_post_anchor_weight=0.5,
    tags=_NARROW_FULL_ACTOR.tags
    + (
        "dual_functional_projection",
        "c50_non_narrow_anchor",
        "anchor_lambda_0p5",
    ),
    notes=(
        "The eight-step full-actor c20-only projection recovered Gate5 "
        "crossing to 98.45%, proving the narrow behavior also resides in the "
        "LiDAR CNN encoder, but corridor CR regressed from c50 13.20% to "
        "14.85%. This matched one-rollout run keeps the same c20 objective on "
        "narrow frames and adds 0.5 times forward KL to frozen c50 on the "
        "disjoint non-narrow frames. There is no PPO or value update."
    ),
)
