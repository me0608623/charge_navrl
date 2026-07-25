"""Second corridor bridge rung: fixed 4 m x 10 m with 3S+1D."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_easy import (
    CONFIG as _EASY,
)


_EASY_C2 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_easy2s0d_c6teacher_ce_c2_s42/"
    "checkpoint_256.pt"
)


CONFIG = replace(
    _EASY,
    name="e2e_sa5_k8_obb_corridor_bridge_medium",
    description=(
        "Second 4m x 10m corridor rung from the verified easy policy: "
        "20% corridor replay with 3 static and 1 dynamic obstacle."
    ),
    checkpoint=_EASY_C2,
    teacher_retention_checkpoint=_EASY_C2,
    long_corridor_static_obstacles=3,
    long_corridor_dynamic_obstacles=1,
    tags=tuple(
        tag for tag in _EASY.tags if tag != "corridor_rung_2s0d"
    )
    + ("corridor_rung_3s1d",),
    notes=(
        "Warm-start from the 2S+0D c2 checkpoint that passed both its "
        "corridor diagnostic and the 1.2m Gate5 at 100%. Reset Adam once "
        "for the rung transition; teacher-action CE is restricted to narrow "
        "replay. Geometry remains exactly 4m x 10m and reward is unchanged."
    ),
)
