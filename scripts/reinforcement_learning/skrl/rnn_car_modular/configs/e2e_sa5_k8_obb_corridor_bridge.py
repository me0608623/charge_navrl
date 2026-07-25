"""SA5 deployment-corridor bridge with narrow-passage retention."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_teacher_corridor_recovery import (
    CONFIG as _RETENTION,
)


_C6 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_teacher_corridor_margin1_atom50_detach_c26_s42/"
    "checkpoint_256.pt"
)


CONFIG = replace(
    _RETENTION,
    name="e2e_sa5_k8_obb_corridor_bridge",
    description=(
        "Short SA5 bridge from the Gate5-passing c6 policy: 68% original, "
        "12% narrow replay and 20% target 4m x 10m corridor replay."
    ),
    checkpoint=_C6,
    no_resume_optimizer=True,
    timesteps=1280,
    save_interval=2,
    long_corridor_fraction=0.20,
    teacher_retention_checkpoint=_C6,
    teacher_retention_weight=0.0,
    teacher_retention_margin_weight=0.0,
    teacher_retention_action_ce_weight=1.0,
    critic_detach_encoder=True,
    tags=_RETENTION.tags
    + (
        "c6_teacher_action_ce",
        "corridor_bridge_20pct",
        "scene_mix_68_12_20",
    ),
    notes=(
        "Reward, K8 network, OBB collision and action observations remain "
        "unchanged. This temporary bridge doubles corridor replay to 20%; "
        "accepted SA6-SA8 retention returns to 10%. The c6 teacher-action CE "
        "acts only on narrow replay, and critic value gradients are detached "
        "from the shared policy encoder."
    ),
)
