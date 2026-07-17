"""K=8 treatment arm for action-conditioned dynamic future occupancy."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa4_fs8_clean_antispin_control import CONFIG as _CONTROL


CONFIG = replace(
    _CONTROL,
    name="e2e_sa4_fs8_clean_antispin_future",
    description="K=8 + A v2 + dynamic future occupancy treatment on fixed SA4.",
    future_occupancy_weight=0.10,
    future_occupancy_horizon_s=1.5,
    future_occupancy_samples=8,
    future_occupancy_safe_distance_m=1.0,
    future_occupancy_near_distance_m=3.0,
    future_occupancy_move_threshold_mps=0.1,
    tags=("e2e", "frame_stack8", "ppo", "clean_progress", "sa4", "anti_spin_a", "future_occupancy"),
    notes="Treatment differs from K=8 control only by capped dynamic swept-occupancy penalty (max -0.10/step).",
)
