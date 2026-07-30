"""SA3 pilot warm-started from the operational SA2 R1 c100 checkpoint.

The SA2 qualification-strength matrix was intentionally stopped and retained
as diagnostic evidence. This continuation therefore records c100 as a
pragmatic, balanced parent rather than a formal Stage B winner.
"""

from dataclasses import replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.e2e_sa3_k8_obb_sim2real_curriculum_v2 import (
    CONFIG as _SA3,
)


PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa2_sim2real_v2_from_sa1r1_c1700_ne1024_s42_p300_r1/"
    "checkpoint_12800.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "a5ea893446c69257caf7e1d5e55dd3dfc6bfaeeb384429552f727f8ef524188c"
)
PILOT_ITERATIONS = 300
ROLLOUT_LENGTH = 128

_parent_path = Path(PARENT_CHECKPOINT)
if not _parent_path.is_file():
    raise FileNotFoundError(
        f"operational SA2 R1 c100 parent checkpoint is missing: {PARENT_CHECKPOINT}"
    )
with _parent_path.open("rb") as _checkpoint_file:
    _parent_sha256 = hashlib.file_digest(_checkpoint_file, "sha256").hexdigest()
if _parent_sha256 != PARENT_CHECKPOINT_SHA256:
    raise RuntimeError(
        "operational SA2 R1 c100 parent checkpoint hash mismatch: "
        f"expected {PARENT_CHECKPOINT_SHA256}, got {_parent_sha256}"
    )


CONFIG = replace(
    _SA3,
    name="e2e_sa3_k8_obb_sim2real_from_sa2_r1_c100",
    description=(
        "SA3 300-iteration pilot on the 17x17 m stage, warm-started from "
        "the operationally selected SA2 R1 c100 checkpoint."
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=True,
    timesteps=PILOT_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=50,
    tags=_SA3.tags
    + (
        "from_sa2_sim2real_v2_r1_c100",
        "noncanonical_lineage",
        "operational_parent",
        "pilot300",
        "lateral_retention_watch",
    ),
    notes=(
        f"{_SA3.notes} This pilot continues from SA2 R1 c100 because it had "
        "the best observed balance between lateral and longitudinal corridor "
        "collision rates in the retained diagnostic evidence. The incomplete "
        "SA2 Stage B matrix is not treated as a formal verdict, and c100 must "
        "not be described as a qualification winner. The initial budget is "
        "exactly 300 iterations and does not auto-extend. Lateral corridor "
        "retention is the primary regression watch."
    ),
)


assert CONFIG.initial_stage == 3
assert CONFIG.fixed_stage is True
assert CONFIG.room_size == 8.5
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 38_400
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True
assert CONFIG.save_interval == 50
assert CONFIG.narrow_passage_fraction == 0.08
assert CONFIG.narrow_passage_fixed_width_range == (1.5, 1.7)
assert CONFIG.narrow_passage_fixed_yaw_limit_deg == 4.0
assert CONFIG.long_corridor_fraction == 0.06
assert CONFIG.long_corridor_free_width == 4.6
assert CONFIG.long_corridor_static_obstacles == 3
assert CONFIG.long_corridor_dynamic_obstacles == 1
assert CONFIG.long_corridor_dynamic_speed_range == (0.20, 0.35)
assert CONFIG.long_corridor_dynamic_motion_weights == (0.5, 0.5, 0.0)
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.end_to_end_frame_stack is True
assert CONFIG.use_action_history is True
