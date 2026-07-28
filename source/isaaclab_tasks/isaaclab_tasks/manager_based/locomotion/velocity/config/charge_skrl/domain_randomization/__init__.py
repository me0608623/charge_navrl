# 域隨機化模組 — 部分子模組可能有外部依賴未安裝
import warnings as _warnings

try:
    from .dr_events import (
        apply_domain_randomization,
        domain_randomization_pre_step,
    )
except ImportError as _e:
    _warnings.warn(f"[domain_randomization] dr_events 載入失敗: {_e}", stacklevel=1)

try:
    from .sensor_dr import (
        apply_lidar_ray_dropout,
        apply_lidar_distance_noise,
        apply_lidar_systematic_bias,
    )
except ImportError:
    pass

try:
    from .physics_dr import (
        randomize_robot_mass,
        randomize_ground_friction,
        randomize_com_offset,
    )
except ImportError:
    pass

try:
    from .actuator_dr import (
        apply_action_delay,
        apply_velocity_scaling,
        apply_motor_response_lag,
        apply_actuator_dynamics,
    )
except ImportError:
    pass

try:
    from .disturbance_dr import (
        apply_random_push,
        apply_continuous_wind,
    )
except ImportError:
    pass
