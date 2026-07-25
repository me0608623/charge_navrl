"""C100 dual projection with the proven c50 anchor weight."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c100_dual_projection import (
    CONFIG as _LAMBDA_075,
)


CONFIG = replace(
    _LAMBDA_075,
    name="e2e_sa6_k8_obb_c100_dual_projection_lambda0p5",
    description=(
        "Bounded c100 Pareto projection with c20 on narrow frames and a 0.5 "
        "c100 functional anchor on every disjoint non-narrow frame."
    ),
    teacher_retention_post_anchor_weight=0.5,
    tags=tuple(
        tag for tag in _LAMBDA_075.tags if tag != "anchor_lambda_0p75"
    )
    + ("anchor_lambda_0p5",),
    notes=(
        "Fixed-seed c100 baseline passes narrow SR/CR but reaches only 91.02% "
        "crossing. Lambda 0.75 raises crossing to 93.69%, 1.31 points below "
        "the hard gate. This is the single bounded stronger-recovery test: "
        "reduce only the disjoint c100 anchor from 0.75 to the previously "
        "validated 0.5. If Gate5 passes, immediately run corrected fixed-goal "
        "4S+2D and require CR <=10%; otherwise stop projection tuning."
    ),
)
