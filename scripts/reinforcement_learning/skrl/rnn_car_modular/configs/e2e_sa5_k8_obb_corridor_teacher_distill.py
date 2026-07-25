"""One-update privileged corridor distillation from the joint-pass c10 policy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_dynamic_postkl_cont import (
    CONFIG as _C10_LINEAGE,
)


_C10 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_dynamic2s1d_postkl_cont_c10_s42/"
    "checkpoint_640.pt"
)


CONFIG = replace(
    _C10_LINEAGE,
    name="e2e_sa5_k8_obb_corridor_teacher_distill",
    description=(
        "One ordinary PPO update from the Gate2+Gate5 joint-pass c10 policy, "
        "followed by corridor-only privileged action projection and the "
        "existing narrow-only c10 KL projection."
    ),
    checkpoint=_C10,
    no_resume_optimizer=False,
    teacher_retention_checkpoint=_C10,
    teacher_retention_weight=0.0,
    teacher_retention_margin_weight=0.0,
    teacher_retention_action_ce_weight=0.0,
    teacher_retention_post_kl_epochs=4,
    teacher_retention_post_kl_lr=1e-3,
    corridor_teacher_distill_epochs=4,
    corridor_teacher_distill_lr=0.1,
    corridor_teacher_distill_batch_size=65536,
    corridor_teacher_distill_max_grad_norm=0.5,
    corridor_teacher_distill_neighbor_mass=0.20,
    corridor_teacher_distill_stride=2,
    corridor_teacher_distill_chunk_size=32,
    timesteps=128,
    save_interval=1,
    tags=_C10_LINEAGE.tags
    + (
        "privileged_corridor_teacher_v4",
        "corridor_soft_action_projection",
        "one_update_only",
    ),
    notes=(
        "The privileged teacher is used only to label feasible states from "
        "the 20% exact 4m x 10m 2S+1D corridor replay. PPO reward and future "
        "occupancy weight stay frozen. A soft target places 20% mass on "
        "adjacent action bins to avoid hard-CE over-sharpening. An offline "
        "held-out-env sweep selected four full-batch SGD passes at lr=0.1 as "
        "the first setting with measurable teacher agreement. The 65536 batch "
        "cap keeps the 1024-env run at the same four optimizer steps as the "
        "successful 64-env calibration instead of multiplying the update by "
        "the number of minibatches. Projection order is PPO, corridor action "
        "supervision, then narrow c10 forward-KL."
    ),
)
