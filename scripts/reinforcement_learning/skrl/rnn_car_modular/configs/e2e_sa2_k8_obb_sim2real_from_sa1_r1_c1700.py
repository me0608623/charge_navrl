"""SA2 pilot warm-started from the accepted SA1 R1 c1700 checkpoint.

This is a pragmatic continuation of the all-at-once ``sim2real_v1`` SA1 arm,
not the canonical curriculum-v2 lineage. SA2 keeps the native phase's ten
stationary goals so reaching a goal remains the dominant, frequently sampled
training outcome.
"""

from dataclasses import replace
from pathlib import Path

from rnn_car_modular.configs.e2e_sa2_k8_obb_sim2real_curriculum_v2 import (
    CONFIG as _SA2,
)


PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/sa1_sim2real_v1_ne1024_s42_r1/"
    "checkpoint_217600.pt"
)
PILOT_ITERATIONS = 300
ROLLOUT_LENGTH = 128

if not Path(PARENT_CHECKPOINT).is_file():
    raise FileNotFoundError(
        f"accepted SA1 c1700 parent checkpoint is missing: {PARENT_CHECKPOINT}"
    )


CONFIG = replace(
    _SA2,
    name="e2e_sa2_k8_obb_sim2real_from_sa1_r1_c1700",
    description=(
        "SA2 300-iteration pilot on the 18x18 m stage, warm-started from "
        "the accepted non-canonical SA1 R1 c1700 checkpoint."
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=True,
    timesteps=PILOT_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=50,
    tags=_SA2.tags
    + (
        "from_sa1_sim2real_v1_r1_c1700",
        "noncanonical_parent",
        "pilot300",
        "ten_stationary_goals",
    ),
    notes=(
        f"{_SA2.notes} This run deliberately uses the accepted SA1 R1 c1700 "
        "checkpoint as a pragmatic parent, so it must not be relabelled as a "
        "canonical curriculum-v2 lineage. Native SA2 retains ten stationary "
        "goals at 2-9 m to keep successful goal reaching frequent in PPO; "
        "single-goal navigation remains an external graduation gate. The "
        "initial budget is exactly 300 iterations and does not auto-extend."
    ),
)


assert CONFIG.initial_stage == 2
assert CONFIG.fixed_stage is True
assert CONFIG.room_size == 9.0
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == 38_400
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.end_to_end_frame_stack is True
