"""Gate2 recovery after the accepted 3S+1D corridor update."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_rung_3s1d_sa5replay20 import (
    CONFIG as _RUNG_3S1D_SA5_20,
)


_CORRIDOR_PASS = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_corridor_rung3s1d_sa5replay20_c128_1u_s42/"
    "checkpoint_128.pt"
)


CONFIG = replace(
    _RUNG_3S1D_SA5_20,
    name="e2e_sa6_k8_obb_corridor_rung_3s1d_sa5recovery40",
    description=(
        "One production-size Gate2 recovery update after the 3S+1D "
        "corridor checkpoint, using 40% SA5-general replay."
    ),
    checkpoint=_CORRIDOR_PASS,
    no_resume_optimizer=False,
    timesteps=128,
    save_interval=1,
    previous_stage_replay_fraction=0.40,
    tags=_RUNG_3S1D_SA5_20.tags + (
        "sa5_general_replay_40pct",
        "gate2_recovery",
    ),
    notes=(
        "The input checkpoint passes 3S+1D at 90.94% SR but Gate2 remains "
        "at 9.94% CR. Run one recovery update with 40% SA5-general, 12% "
        "narrow, 10% 3S+1D corridor, and 38% native SA6. Keep c20 "
        "beta=0.30, reward, LR, and optimizer state unchanged. Gate2 is "
        "evaluated first; corridor and Gate5 must then be retained."
    ),
)
