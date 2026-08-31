"""Paper realism factorial D: measured LiDAR corruption plus delay DR."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa1_paper_factorial_c_ideal_u012_p600 import (
    CONFIG as _CELL_C,
)


CONFIG = replace(
    _CELL_C,
    name="e2e_sa1_paper_factorial_d_full_u012_p600",
    description=(
        "Paper 2x2 cell D: fixed measured VLP-16 full corruption plus "
        "decoded-command delay U{0,1,2}, trained from scratch for 600 iterations."
    ),
    lidar_no_noise=False,
    vlp16_noise_mode="full",
    tags=tuple(tag for tag in _CELL_C.tags if tag not in {"cell_c", "ideal_lidar"})
    + ("cell_d", "measured_full_lidar"),
    notes=(
        "READY_NOT_RUN complete realism arm for the paper factorial. It "
        "combines fixed measurement-calibrated VLP-16 corruption with only "
        "the randomized decoded-command delay. It must not be described as "
        "full physics or full actuator domain randomization."
    ),
)
