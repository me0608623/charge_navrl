"""One conservative intervention-only corridor update from c12."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_teacher_distill import (
    CONFIG as _DISTILL_RECIPE,
)


_C12 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_teacher_distill_c12_stepnorm_s42/"
    "checkpoint_128.pt"
)


CONFIG = replace(
    _DISTILL_RECIPE,
    name="e2e_sa5_k8_obb_corridor_teacher_intervention",
    description=(
        "One conservative update from c12 with privileged supervision only "
        "when the deterministic policy has an unsafe swept trajectory."
    ),
    checkpoint=_C12,
    no_resume_optimizer=False,
    corridor_teacher_distill_lr=0.02,
    corridor_teacher_intervention_only=True,
    corridor_teacher_intervention_clearance_m=0.20,
    timesteps=128,
    save_interval=1,
    tags=_DISTILL_RECIPE.tags
    + (
        "c12_parent",
        "intervention_only",
        "unsafe_policy_trajectory_only",
    ),
    notes=(
        "Keep c12 immutable. Select only corridor states where the current "
        "deterministic action predicts OBB collision or less than 0.20m "
        "clearance, the privileged teacher has a feasible action, and the "
        "teacher differs. Clear-path corridor frames receive no distillation. "
        "Use four full-batch steps at lr=0.02, then fail-fast Gate5, Gate2, "
        "and corridor. The inherited narrow teacher remains c10."
    ),
)
