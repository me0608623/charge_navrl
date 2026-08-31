"""Paper realism factorial B: measured VLP-16 corruption and zero delay."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa1_paper_factorial_a_ideal_d0_p600 import (
    CONFIG as _CELL_A,
)


CONFIG = replace(
    _CELL_A,
    name="e2e_sa1_paper_factorial_b_full_d0_p600",
    description=(
        "Paper 2x2 cell B: fixed measured VLP-16 full corruption and identity "
        "d0 actuation, trained from scratch for 600 iterations."
    ),
    lidar_no_noise=False,
    vlp16_noise_mode="full",
    tags=tuple(tag for tag in _CELL_A.tags if tag not in {"cell_a", "ideal_lidar"})
    + ("cell_b", "measured_full_lidar"),
    notes=(
        "READY_NOT_RUN LiDAR main-effect arm. Relative to cell A, the only "
        "behavioral factor is the fixed measured full VLP-16 preset: 8.672 mm "
        "sigma, per-ring bias, measured holes and valid-return-only mixed-pixel "
        "outliers. This is calibrated corruption, not per-episode sensor DR."
    ),
)
