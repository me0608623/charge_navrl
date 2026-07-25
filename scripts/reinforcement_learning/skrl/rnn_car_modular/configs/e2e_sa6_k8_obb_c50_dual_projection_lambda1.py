"""Dual c20/c50 projection with an equal-weight non-narrow c50 anchor."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_dual_projection import (
    CONFIG as _LAMBDA_HALF,
)


CONFIG = replace(
    _LAMBDA_HALF,
    name="e2e_sa6_k8_obb_c50_dual_projection_lambda1",
    description=(
        "Matched dual functional projection with c50 non-narrow anchor "
        "weight increased from 0.5 to 1.0."
    ),
    teacher_retention_post_anchor_weight=1.0,
    tags=tuple(
        tag for tag in _LAMBDA_HALF.tags
        if tag != "anchor_lambda_0p5"
    )
    + ("anchor_lambda_1p0",),
    notes=(
        "Lambda 0.5 retained Gate5 crossing at 99.40%, Gate2 at "
        "SR 90.60% / CR 9.33%, and 2S+1D corridor at SR 95.32%, but "
        "4S+2D corridor regressed to SR 84.02% / CR 15.98%. This run changes "
        "only the c50 non-narrow functional anchor from 0.5 to 1.0."
    ),
)
