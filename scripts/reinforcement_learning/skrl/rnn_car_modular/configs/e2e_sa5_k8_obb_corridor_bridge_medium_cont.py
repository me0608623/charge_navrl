"""Continue the 3S+1D corridor rung without resetting Adam."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_medium import (
    CONFIG as _MEDIUM,
)


_MEDIUM_C2 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_medium3s1d_easyteacher_ce_c2_s42/"
    "checkpoint_256.pt"
)


CONFIG = replace(
    _MEDIUM,
    name="e2e_sa5_k8_obb_corridor_bridge_medium_cont",
    description=(
        "Continue the 3S+1D corridor rung in two-update windows while "
        "preserving Adam state and the verified c2 narrow teacher."
    ),
    checkpoint=_MEDIUM_C2,
    no_resume_optimizer=False,
    teacher_retention_checkpoint=_MEDIUM_C2,
    tags=_MEDIUM.tags + ("resume_optimizer", "two_update_window"),
    notes=(
        "The medium c2 policy passed 1.2m Gate5 at 100% but reached only "
        "76.6% SR / 23.4% CR on the 3S+1D corridor. Continue on the same "
        "rung without optimizer reset; evaluate every two updates and stop "
        "as soon as corridor SR>=90%, CR<=10%, TO<=5%."
    ),
)
