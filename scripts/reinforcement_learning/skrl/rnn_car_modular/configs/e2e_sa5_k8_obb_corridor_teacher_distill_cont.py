"""One guarded teacher-distillation update from the c12 Pareto candidate."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_teacher_distill import (
    CONFIG as _C12_RECIPE,
)


_C12 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_teacher_distill_c12_stepnorm_s42/"
    "checkpoint_128.pt"
)


CONFIG = replace(
    _C12_RECIPE,
    name="e2e_sa5_k8_obb_corridor_teacher_distill_cont",
    description=(
        "One additional step-normalized corridor teacher update from c12. "
        "The narrow teacher remains the original c10 joint-pass policy."
    ),
    checkpoint=_C12,
    no_resume_optimizer=False,
    timesteps=128,
    save_interval=1,
    tags=_C12_RECIPE.tags + ("c12_guarded_continuation",),
    notes=(
        "c12 preserved Gate2 CR below 9% and Gate5 crossing above 95% while "
        "improving the 2S+1D corridor. This is one guarded continuation only. "
        "Keep the c10 narrow teacher inherited from the base recipe, evaluate "
        "Gate2, Gate5 and corridor immediately, and stop on any regression."
    ),
)
