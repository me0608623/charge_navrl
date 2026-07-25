"""One-update 2S+1D branch with post-PPO KL projection."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic import (
    CONFIG as _DYNAMIC,
)


_DYNAMIC_C3 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_dynamic2s1d_c3c4_s42/"
    "checkpoint_128.pt"
)


CONFIG = replace(
    _DYNAMIC,
    name="e2e_sa5_k8_obb_corridor_bridge_dynamic_postkl",
    description=(
        "One 2S+1D PPO update followed by stateless narrow-only forward-KL "
        "projection to the Gate5-perfect c3 teacher."
    ),
    checkpoint=_DYNAMIC_C3,
    teacher_retention_checkpoint=_DYNAMIC_C3,
    teacher_retention_weight=0.0,
    teacher_retention_margin_weight=0.0,
    teacher_retention_action_ce_weight=0.0,
    teacher_retention_post_kl_epochs=4,
    teacher_retention_post_kl_lr=1e-3,
    teacher_retention_post_kl_batch_size=4096,
    teacher_retention_post_kl_max_grad_norm=0.5,
    timesteps=128,
    save_interval=1,
    tags=_DYNAMIC.tags + ("post_ppo_forward_kl", "one_update_only"),
    notes=(
        "Joint pre-update KL is zero when teacher and student start equal, "
        "while hard teacher-action CE over-sharpens logits and still collapsed "
        "Gate5. This branch leaves PPO untouched, then corrects the non-zero "
        "post-update KL on narrow replay with four stateless SGD passes."
    ),
)
