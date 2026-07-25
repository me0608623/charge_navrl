"""Recover narrow behavior using complete trajectories driven by the c20 teacher."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c100_narrow_projection import (
    CONFIG as _STUDENT_VISITED_PROJECTION,
)


CONFIG = replace(
    _STUDENT_VISITED_PROJECTION,
    name="e2e_sa6_k8_obb_c100_teacher_forced_projection",
    description=(
        "Projection-only c100 recovery using c20-driven narrow rollouts so "
        "the student trains on complete successful throat-crossing states."
    ),
    teacher_retention_rollout_override=True,
    tags=_STUDENT_VISITED_PROJECTION.tags
    + (
        "teacher_forced_narrow_rollout",
        "complete_success_trajectory",
    ),
    notes=(
        "C100 passes the corrected fixed-goal 4S+2D corridor gate at 8.01% "
        "CR, but projection on c100-visited narrow states failed because c100 "
        "stalls before the throat and never supplies post-throat observations. "
        "This matched causal run changes only rollout state coverage: on the "
        "12% narrow replay envs, frozen c20 deterministic actions drive the "
        "environment. The c100 student is then projected on those complete "
        "successful trajectories. PPO remains disabled; non-narrow envs keep "
        "c100 actions. Gate5 is evaluated first, followed by the fixed-goal "
        "4S+2D corridor gate only if Gate5 passes."
    ),
)
