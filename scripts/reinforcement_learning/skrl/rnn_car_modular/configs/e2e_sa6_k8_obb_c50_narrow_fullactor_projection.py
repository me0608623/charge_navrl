"""Full-actor narrow projection diagnostic from the corridor-improved c50."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_postkl_recovery import (
    CONFIG as _HEAD_ONLY,
)


CONFIG = replace(
    _HEAD_ONLY,
    name="e2e_sa6_k8_obb_c50_narrow_fullactor_projection",
    description=(
        "Matched projection-only diagnostic from corridor-improved c50. "
        "Update the LiDAR CNN encoder and policy head together to test whether "
        "narrow-passage behavior was lost in the encoder."
    ),
    teacher_retention_post_policy_head_only=False,
    tags=tuple(
        tag for tag in _HEAD_ONLY.tags if tag != "projection_only"
    )
    + (
        "projection_only",
        "full_actor_projection",
        "encoder_entanglement_test",
    ),
    notes=(
        "Matched against the head-only c50 recovery: same checkpoint, one "
        "1024-env rollout, no PPO, four full-batch SGD steps, forward KL and "
        "argmax margin. The sole causal change is allowing gradients through "
        "the LiDAR CNN encoder as well as the policy head. Gate5 must recover "
        "to at least 95%; then rerun both corridor densities to measure "
        "reverse forgetting."
    ),
)
