"""Apply the proven c20-only full-actor projection to the c100 policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_fullactor_projection8 import (
    CONFIG as _C50_PROJECTION,
)


_CORRIDOR_C100 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_c60_to_c100_s42/checkpoint_5120.pt"
)


CONFIG = replace(
    _C50_PROJECTION,
    name="e2e_sa6_k8_obb_c100_narrow_projection",
    description=(
        "Projection-only recovery of c20 narrow behavior from the stronger "
        "c100 corridor checkpoint, without a competing c100 anchor."
    ),
    checkpoint=_CORRIDOR_C100,
    previous_stage_teacher_checkpoint=None,
    previous_stage_teacher_retention_weight=0.0,
    previous_stage_teacher_scope="previous_stage",
    teacher_retention_post_anchor_weight=0.0,
    tags=tuple(
        tag for tag in _C50_PROJECTION.tags
        if tag != "c50_corridor_candidate"
    )
    + (
        "c100_corridor_candidate",
        "c20_only_projection",
        "long_train_then_project",
    ),
    notes=(
        "C100 reached corrected fixed-goal 4S+2D SR 91.99% / CR 8.01%. "
        "The prior c50 c20-only eight-step projection recovered Gate5 to "
        "98.45% crossing and shifted corridor CR by about +1.65 percentage "
        "points, which fits inside c100's 1.99-point margin. Use the exact "
        "proven full-actor projection with no PPO and no c100 anchor."
    ),
)
