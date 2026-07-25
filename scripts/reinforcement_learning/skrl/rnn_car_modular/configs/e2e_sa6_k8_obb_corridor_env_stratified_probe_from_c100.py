"""Gate-aligned corridor replay probe from the mixed-motion c100 policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_mixed_corridor_probe_from_c19200 import (
    CONFIG as _FROZEN_RECIPE,
)


_C100 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_corridor_priority_c30_s42/checkpoint_12800.pt"
)


CONFIG = replace(
    _FROZEN_RECIPE,
    name="e2e_sa6_k8_obb_corridor_env_stratified_probe_from_c100",
    description=(
        "Thirty-iteration probe from c100 that fixes the corridor train/eval "
        "distribution mismatch. Each corridor environment now contains one "
        "pure dynamic-motion family, balanced across environments."
    ),
    checkpoint=_C100,
    no_resume_optimizer=False,
    timesteps=30 * _FROZEN_RECIPE.rollout_length,
    save_interval=5,
    long_corridor_dynamic_motion_mode="env_stratified",
    tags=_FROZEN_RECIPE.tags
    + (
        "env_stratified_corridor_motion",
        "gate_aligned_replay",
        "c100_warm_start",
        "thirty_iteration_probe",
    ),
    notes=(
        "Resume c100 model and Adam. Preserve the 68/10/12/10 replay mix, "
        "reward, network, LR and horizon. Change only corridor motion sampling: "
        "both dynamic obstacles in an environment use the same lateral, "
        "longitudinal or random-2D family, balanced 1:1:1 across corridor "
        "environments. Save every five iterations. Continue only if lateral CR "
        "returns toward <=10% while longitudinal, random-2D and mixed do not "
        "regress materially; Gate2 remains mandatory."
    ),
)
