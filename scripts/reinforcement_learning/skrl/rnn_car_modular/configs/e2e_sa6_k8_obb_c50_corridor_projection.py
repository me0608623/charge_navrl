"""Dual projection with c50 anchored only on deployment-corridor frames."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_dual_projection import (
    CONFIG as _NON_NARROW_ANCHOR,
)


CONFIG = replace(
    _NON_NARROW_ANCHOR,
    name="e2e_sa6_k8_obb_c50_corridor_projection",
    description=(
        "Matched c20 narrow projection with the c50 functional anchor scoped "
        "only to 4S+2D deployment-corridor replay frames."
    ),
    previous_stage_teacher_scope="corridor",
    tags=tuple(
        tag for tag in _NON_NARROW_ANCHOR.tags
        if tag not in {"c50_non_narrow_anchor", "anchor_lambda_0p5"}
    )
    + (
        "c50_corridor_anchor",
        "anchor_lambda_0p5",
        "anchor_scope_ablation",
    ),
    notes=(
        "Broad non-narrow anchoring retained Gate5 and Gate2 but produced "
        "4S+2D corridor CR 15.98% at lambda 0.5. Lambda 1.0 improved corridor "
        "CR only to 14.29% while failing Gate5 crossing at 94.80%. This run "
        "returns to lambda 0.5 and changes only the c50 anchor mask from all "
        "non-narrow frames to corridor replay frames, making corridor KL an "
        "independent domain mean instead of a diluted component."
    ),
)
