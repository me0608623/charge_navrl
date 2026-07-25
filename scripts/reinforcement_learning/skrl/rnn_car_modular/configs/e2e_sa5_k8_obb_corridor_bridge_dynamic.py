"""Isolate the first dynamic-corridor skill on a 2S+1D rung."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge_medium_cont import (
    CONFIG as _MEDIUM_CONT,
)


CONFIG = replace(
    _MEDIUM_CONT,
    name="e2e_sa5_k8_obb_corridor_bridge_dynamic",
    description=(
        "Isolate dynamic crossing avoidance in the deployment corridor: "
        "20% exact 4m x 10m replay with 2 static and 1 dynamic obstacle."
    ),
    save_interval=1,
    long_corridor_static_obstacles=2,
    long_corridor_dynamic_obstacles=1,
    tags=tuple(
        tag for tag in _MEDIUM_CONT.tags if tag != "corridor_rung_3s1d"
    )
    + ("corridor_rung_2s1d", "save_every_update"),
    notes=(
        "The 3S+1D c2 policy collided predominantly with patrol obstacles, "
        "and its 2S+1D diagnostic still reached only 75.1% SR / 24.9% CR. "
        "Keep reward, LR, teacher, optimizer and corridor geometry fixed; "
        "change only the obstacle mix and save every update to catch narrow "
        "deterministic optima."
    ),
)
