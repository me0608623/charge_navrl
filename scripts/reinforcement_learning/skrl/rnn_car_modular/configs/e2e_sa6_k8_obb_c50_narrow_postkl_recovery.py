"""Projection-only narrow recovery from the corridor-improved SA6 c50 policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


_CORRIDOR_C50 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_cont300_c128_s42/checkpoint_6400.pt"
)


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_c50_narrow_postkl_recovery",
    description=(
        "One projection-only rollout from corridor-improved c50. Skip PPO and "
        "restore c20 narrow deterministic behavior with policy-head-only "
        "forward-KL plus argmax-margin projection."
    ),
    checkpoint=_CORRIDOR_C50,
    no_resume_optimizer=False,
    ppo_epochs=0,
    timesteps=_SA6.rollout_length,
    save_interval=1,
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
    tags=_SA6.tags
    + (
        "c50_corridor_candidate",
        "projection_only",
        "narrow_post_kl_recovery",
        "teacher_argmax_margin",
    ),
    notes=(
        "C50 improved fixed-goal 4S+2D corridor SR from 77.57% to 86.80% "
        "but Gate5 crossing regressed from 99.20% to 78.81%. This one-update "
        "causal test performs no PPO/value/encoder optimizer step. It collects "
        "student-visited narrow states, then applies four full-batch SGD steps "
        "only to the policy head against the frozen c20 teacher. Run Gate5 "
        "first; only if crossing recovers to at least 95% evaluate Gate2 and "
        "both corridor densities."
    ),
)
