"""Short mixed-motion corridor continuation from the fixed c19200 candidate."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_deepen_from_iter160 import (
    CONFIG as _C19200_RECIPE,
)


_C19200 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_corridor_deepen_from_iter160_s42/checkpoint_19200.pt"
)


CONFIG = replace(
    _C19200_RECIPE,
    name="e2e_sa6_k8_obb_mixed_corridor_probe_from_c19200",
    description=(
        "Thirty-iteration diagnostic continuation from c19200. The only "
        "training-distribution change is replacing lateral-only motion in the "
        "existing 10% deployment-corridor replay with balanced lateral, "
        "longitudinal and controlled random-2D patrols."
    ),
    checkpoint=_C19200,
    no_resume_optimizer=False,
    timesteps=30 * _C19200_RECIPE.rollout_length,
    save_interval=5,
    long_corridor_dynamic_motion_mode="mixed",
    tags=_C19200_RECIPE.tags
    + (
        "mixed_corridor_motion",
        "c19200_warm_start",
        "thirty_iteration_probe",
    ),
    notes=(
        "Resume c19200 model and Adam. Preserve the 68/10/12/10 replay mix, "
        "K8+future reward, OBB/action-history network, LR and horizon. Change "
        "only the dynamic motion family inside the existing 10% corridor "
        "replay from lateral to balanced mixed. Save every five iterations. "
        "Evaluate Gate2, direct Gate5 and all four corridor motion gates before "
        "extending beyond 30 iterations."
    ),
)
