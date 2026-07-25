"""One-update 2S+1D branch with stronger narrow teacher-action CE."""

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
    name="e2e_sa5_k8_obb_corridor_bridge_dynamic_ce5",
    description=(
        "One-update 2S+1D continuation from the Gate5-perfect c3 policy "
        "with narrow-only teacher-action CE increased from 1 to 5."
    ),
    checkpoint=_DYNAMIC_C3,
    teacher_retention_checkpoint=_DYNAMIC_C3,
    teacher_retention_action_ce_weight=5.0,
    timesteps=128,
    save_interval=1,
    tags=_DYNAMIC.tags + ("teacher_action_ce5", "one_update_only"),
    notes=(
        "CE=1 retained the 1.2m skill for one update but collapsed it on "
        "the second (34.3% crossing, 65.2% timeout). Preserve the c3 Adam "
        "state and change only narrow teacher-action CE to 5. Train exactly "
        "one update, then require Gate5 before any corridor evaluation."
    ),
)
