"""One-update SA6 dual-retention probe at the calibrated upper beta."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_dual_retention_beta0p02 import (
    CONFIG as _BETA02,
)


CONFIG = replace(
    _BETA02,
    name="e2e_sa6_k8_obb_dual_retention_beta0p10",
    description=(
        "One production-size SA6 update from c128 with c20 beta=0.30 "
        "on narrow frames and c12 beta=0.10 on SA5-general frames."
    ),
    previous_stage_teacher_retention_weight=0.10,
    tags=tuple(
        tag
        for tag in _BETA02.tags
        if tag != "sa5_teacher_beta0p02"
    )
    + ("sa5_teacher_beta0p10",),
    notes=(
        "Upper point of the scale-calibrated c12 KL sweep. Beta=0.02 "
        "reached Gate2 CR 9.533%; beta=0.05 reached 9.634%. Beta=0.10 "
        "is the final stronger-anchor test before rejecting this lever. "
        "All other variables remain identical."
    ),
)
