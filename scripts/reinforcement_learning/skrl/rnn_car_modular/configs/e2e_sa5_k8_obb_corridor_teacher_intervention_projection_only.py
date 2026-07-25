"""Projection-only intervention experiment from the frozen c12 policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_teacher_intervention import (
    CONFIG as _INTERVENTION,
)


CONFIG = replace(
    _INTERVENTION,
    name="e2e_sa5_k8_obb_corridor_teacher_intervention_projection_only",
    description=(
        "Collect one c12 rollout but skip PPO, then apply only targeted "
        "corridor action projection and the c10 narrow retention projection."
    ),
    ppo_epochs=0,
    timesteps=128,
    save_interval=1,
    tags=_INTERVENTION.tags + ("projection_only", "ppo_frozen"),
    notes=(
        "This isolates the teacher intervention from the ordinary PPO update. "
        "Actor/critic/extractor receive no PPO optimizer step. The corridor "
        "policy-head projection uses only unsafe intervention states at "
        "lr=0.02, followed by the inherited c10 narrow KL guard. Fail-fast "
        "Gate5 before corridor or Gate2."
    ),
)
