"""Paper realism factorial C: ideal LiDAR and randomized command delay."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa1_paper_factorial_a_ideal_d0_p600 import (
    CONFIG as _CELL_A,
)


DELAY_SUPPORT_STEPS = (0, 2)


CONFIG = replace(
    _CELL_A,
    name="e2e_sa1_paper_factorial_c_ideal_u012_p600",
    description=(
        "Paper 2x2 cell C: ideal VLP-16 observations and decoded-command "
        "delay U{0,1,2}, trained from scratch for 600 iterations."
    ),
    enable_actuator_dr=True,
    actuator_delay_range=DELAY_SUPPORT_STEPS,
    tags=tuple(tag for tag in _CELL_A.tags if tag not in {"cell_a", "delay_d0"})
    + ("cell_c", "delay_u012"),
    notes=(
        "READY_NOT_RUN delay main-effect arm. Relative to cell A, the only "
        "behavioral factor is decoded-command delay U{0,1,2} at 0.2 s per "
        "step. Velocity scale and motor lag remain exactly one; physics and "
        "external-force DR remain disabled."
    ),
)
