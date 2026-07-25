"""One-update SA6 dual-retention probe with a stronger SA5 teacher."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_dual_retention_beta0p02 import (
    CONFIG as _BETA02,
)


CONFIG = replace(
    _BETA02,
    name="e2e_sa6_k8_obb_dual_retention_beta0p05",
    description=(
        "One production-size SA6 update from c128 with c20 beta=0.30 "
        "on narrow frames and c12 beta=0.05 on SA5-general frames."
    ),
    previous_stage_teacher_retention_weight=0.05,
    tags=tuple(
        tag
        for tag in _BETA02.tags
        if tag != "sa5_teacher_beta0p02"
    )
    + ("sa5_teacher_beta0p05",),
    notes=(
        "Second calibrated dual-teacher point. Beta=0.02 changed pooled "
        "Gate2 CR only from 9.632% to 9.533%; beta=0.05 tests whether the "
        "same c12 direction is simply underweighted. All other variables "
        "remain identical."
    ),
)
