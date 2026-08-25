"""SA4 pilot warm-started from the SA3 R1 c100 checkpoint.

Unlike the SA3 continuation, this parent *did* clear a graduation gate before
being used:

* Phase A -- 12/12, 6 checkpoints x lateral/longitudinal on SA3 geometry.
  ``checkpoint_12800`` (c100) had the lowest worst-direction corridor CR.
  The final checkpoint c300 was **not** selected: its lateral CR was the worst
  of the six (0.0472 vs c100's 0.0171), which the merged training metric hid.
* Phase B -- 4/4, nav_clean / nav_native / native_crossing / narrow_range.
* Phase C -- 6/6 absolute, plus a matched retention comparison against the SA2
  parent on SA2 geometry: SR rose and CR fell in both corridor families, so
  adapting to SA3 cost nothing measurable on the parent's own scene.

One caveat is recorded deliberately: the Phase B ``native_crossing`` cell passed
by a single collision (110/2210 = 0.049774 against a 0.05 bar, margin +0.0002).
It is a valid PASS against a pre-registered threshold, but it is not a robust
one, and any SA4 result that hinges on crossing performance should be read with
that in mind.
"""

from dataclasses import replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.e2e_sa4_k8_obb_sim2real_curriculum_v2 import (
    CONFIG as _SA4,
)


PARENT_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa3_sim2real_v2_from_sa2r1_c100_ne1024_s42_p300_r1/"
    "checkpoint_12800.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "e7da9aa0771252966f4d3cc7081adcbfdb35d9eaf828432ea8cd2fb50e453f88"
)
PILOT_ITERATIONS = 100
ROLLOUT_LENGTH = 128

_parent_path = Path(PARENT_CHECKPOINT)
if not _parent_path.is_file():
    raise FileNotFoundError(
        f"gate-selected SA3 R1 c100 parent checkpoint is missing: {PARENT_CHECKPOINT}"
    )
with _parent_path.open("rb") as _checkpoint_file:
    _parent_sha256 = hashlib.file_digest(_checkpoint_file, "sha256").hexdigest()
if _parent_sha256 != PARENT_CHECKPOINT_SHA256:
    raise RuntimeError(
        "gate-selected SA3 R1 c100 parent checkpoint hash mismatch: "
        f"expected {PARENT_CHECKPOINT_SHA256}, got {_parent_sha256}"
    )


CONFIG = replace(
    _SA4,
    name="e2e_sa4_k8_obb_sim2real_from_sa3_r1_c100",
    description=(
        "SA4 100-iteration pilot on the 16x16 m stage, warm-started from the "
        "gate-selected SA3 R1 c100 checkpoint."
    ),
    checkpoint=PARENT_CHECKPOINT,
    no_resume_optimizer=True,
    timesteps=PILOT_ITERATIONS * ROLLOUT_LENGTH,
    save_interval=50,
    tags=_SA4.tags
    + (
        "from_sa3_sim2real_v2_r1_c100",
        "gate_selected_parent",
        "pilot100",
        "lateral_retention_watch",
    ),
    notes=(
        f"{_SA4.notes} Parent selected by the SA3 minimal graduation gate "
        "(Phase A 12/12, Phase B 4/4, Phase C 6/6 with matched retention). "
        "c300 was rejected as parent: its lateral corridor CR was the worst of "
        "the six SA3 checkpoints while the merged corridor metric still looked "
        "healthy, so lateral corridor CR stays the primary regression watch. "
        "The budget is exactly 100 iterations and does not auto-extend; SA5 is "
        "not started by this run."
    ),
)


# Config lock. Anything here that drifts changes what the pilot means, so it
# fails at import rather than at hour three of a run.
assert CONFIG.initial_stage == 4
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH
assert CONFIG.timesteps == PILOT_ITERATIONS * ROLLOUT_LENGTH == 12_800
assert CONFIG.save_interval == 50
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True

# Sim-to-real mechanisms that must survive the warm start.
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.actuator_delay_range == (0, 2)
assert CONFIG.lidar_hole_rate > 0.0
assert CONFIG.lidar_distractor_rate > 0.0
assert CONFIG.lidar_displacement_std_soft > 0.0
assert CONFIG.lidar_no_noise is False
