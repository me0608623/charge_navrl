"""Second one-update continuation on the c12 2S+1D corridor rung."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c12_corridor_rung_2s1d import (
    CONFIG as _RUNG_C1,
)


_C1 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_c12_corridor_rung2s1d_1u_s42/checkpoint_128.pt"
)


CONFIG = replace(
    _RUNG_C1,
    name="e2e_sa6_k8_obb_c12_corridor_rung_2s1d_cont2",
    description=(
        "Second production-size update at the c12 2S+1D corridor rung."
    ),
    checkpoint=_C1,
    no_resume_optimizer=False,
    timesteps=128,
    save_interval=1,
    tags=_RUNG_C1.tags + ("corridor_rung_cont2",),
    notes=(
        "Continue exactly one update from c1 with its Adam state. The first "
        "update improved fixed 2S+1D SR from 84.12% to 85.57% but did not "
        "pass. Keep scene mix, reward, c20 beta, LR, and obstacle rung "
        "unchanged; re-run the corridor gate before any other gate."
    ),
)
