"""Ready-to-run rebuilt SA3-v3 from the locked operational SA2 c100 parent."""

from dataclasses import replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.sim2real_speed_density_curriculum_v3 import (
    ACTUATOR_DELAY_RANGE,
    SA3_SPEED_DENSITY_MIX,
    VEHICLE_SPEED_RATE,
    make_speed_density_curriculum_config,
)


PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa2_sim2real_v2_from_sa1r1_c1700_ne1024_s42_p300_r1/"
    "checkpoint_12800.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "a5ea893446c69257caf7e1d5e55dd3dfc6bfaeeb384429552f727f8ef524188c"
)
TRAINING_ITERATIONS = 300
ROLLOUT_LENGTH = 128
SAVE_INTERVAL = 50

_parent_path = Path(PARENT_CHECKPOINT)
if not _parent_path.is_file():
    raise FileNotFoundError(f"SA3-v3 parent is missing: {PARENT_CHECKPOINT}")
with _parent_path.open("rb") as _checkpoint_file:
    _actual_sha256 = hashlib.file_digest(_checkpoint_file, "sha256").hexdigest()
if _actual_sha256 != PARENT_CHECKPOINT_SHA256:
    raise RuntimeError(
        "SA3-v3 parent hash mismatch: "
        f"expected {PARENT_CHECKPOINT_SHA256}, got {_actual_sha256}"
    )

_BASE = make_speed_density_curriculum_config(
    3,
    checkpoint=PARENT_CHECKPOINT,
)

CONFIG = replace(
    _BASE,
    name="e2e_sa3_k8_obb_speed_density_v3_from_sa2_r1_c100",
    description=(
        "Rebuilt SA3-v3 from locked SA2 c100 with speed_rate=0.7 and joint "
        "P035/P060 low-density corridor profiles."
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=True,
    timesteps=TRAINING_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=SAVE_INTERVAL,
    tags=_BASE.tags + ("from_locked_sa2_r1_c100", "pilot300_fail_closed"),
    notes=(
        f"{_BASE.notes} The optimizer is reset because the vehicle action "
        "contract changes at the SA3 boundary. Checkpoints are saved every 50 "
        "iterations; no checkpoint authorizes SA4 until the fixed P060 "
        "low-density and retention gates pass."
    ),
)

assert CONFIG.initial_stage == 3
assert CONFIG.fixed_stage is True
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True
assert CONFIG.timesteps == 38_400
assert CONFIG.save_interval == 50
assert CONFIG.speed_rate == VEHICLE_SPEED_RATE == 0.7
assert CONFIG.speed_rate_obs == "ego"
assert CONFIG.long_corridor_speed_density_mix == SA3_SPEED_DENSITY_MIX
assert CONFIG.long_corridor_obstacle_count_mix is None
assert CONFIG.actuator_delay_range == ACTUATOR_DELAY_RANGE == (1, 2)
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"

