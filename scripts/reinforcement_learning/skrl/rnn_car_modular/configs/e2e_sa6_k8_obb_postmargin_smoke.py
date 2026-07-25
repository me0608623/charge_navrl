"""One-update SA6 calibration with post-PPO narrow argmax retention."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_postmargin_smoke",
    description=(
        "One SA6 PPO update followed by a narrow-only c20 KL plus deterministic "
        "argmax-margin projection on the policy head."
    ),
    teacher_retention_weight=0.0,
    teacher_retention_margin_weight=0.0,
    teacher_retention_action_ce_weight=0.0,
    teacher_retention_post_kl_epochs=4,
    teacher_retention_post_kl_lr=1e-3,
    teacher_retention_post_kl_batch_size=65536,
    teacher_retention_post_kl_max_grad_norm=0.5,
    teacher_retention_post_margin_weight=1.0,
    teacher_retention_post_action_ce_weight=0.0,
    teacher_retention_post_policy_head_only=True,
    teacher_retention_argmax_margin=0.2,
    timesteps=128,
    save_interval=1,
    tags=_SA6.tags
    + (
        "post_ppo_narrow_projection",
        "teacher_argmax_margin",
        "one_update_smoke",
    ),
    notes=(
        "Calibration only. Ordinary and corridor frames use unchanged PPO. "
        "After PPO, four full-batch SGD steps act only on narrow replay and "
        "only on the policy head, minimizing forward KL plus a unit-weight "
        "teacher argmax margin. Fail-fast on the fixed 1.2m Gate5 before "
        "Gate2 or corridor."
    ),
)
