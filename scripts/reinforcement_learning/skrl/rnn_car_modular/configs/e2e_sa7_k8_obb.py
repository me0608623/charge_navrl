"""SA7 entrypoint for the measured OBB collision lineage."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_k8_future_frozen_base import make_stage_config


_BASE = make_stage_config(7)

CONFIG = replace(
    _BASE,
    name="e2e_sa7_k8_obb",
    description="K=8 + future occupancy with measured 0.70x0.60m OBB collision on SA7.",
    use_obb_collision=True,
    use_action_history=True,
    tags=_BASE.tags + ("obb_collision", "physical_footprint", "action_history_2step"),
    notes=(
        "OBB lineage warm-started from an accepted SA6 checkpoint. "
        "Collision footprint: half_length=0.35m, half_width=0.30m including wheels, "
        "center_offset_x=-0.128m, buffer=0.10m. Observation includes two applied actions (4D). "
        "LiDAR noise and actuator DR remain disabled. Gate1-5 are hard graduation gates."
    ),
)
