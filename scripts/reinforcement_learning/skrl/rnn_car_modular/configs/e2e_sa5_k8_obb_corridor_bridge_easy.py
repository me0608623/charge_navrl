"""First corridor bridge rung: fixed 4 m x 10 m with 2 static obstacles."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa5_k8_obb_corridor_bridge import (
    CONFIG as _BRIDGE,
)


CONFIG = replace(
    _BRIDGE,
    name="e2e_sa5_k8_obb_corridor_bridge_easy",
    description=(
        "First 4m x 10m corridor rung from c6: 20% corridor replay with "
        "2 static and 0 dynamic obstacles; geometry and retention stay frozen."
    ),
    long_corridor_static_obstacles=2,
    long_corridor_dynamic_obstacles=0,
    tags=_BRIDGE.tags + ("corridor_rung_2s0d",),
    notes=(
        "Temporary constructive curriculum rung. Corridor width and length "
        "remain the deployment 4m x 10m; only obstacle count is reduced to "
        "create successful confined-navigation trajectories before 3S+1D and "
        "4S+2D."
    ),
)
