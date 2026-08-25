"""SA5-R2 correction run with corridor side walls sealed to the room boundary.

This lineage deliberately restarts from the same SA4-R3 conceptual-it125
checkpoint as the completed legacy SA5 run. Sensor noise, actuator delay,
reward, model, PPO, scene mix and training budget are unchanged. The only
behavioral change is supplied by the frozen environment source: the 10 m
corridor remains the interaction zone while its physical side walls span the
15 m Stage-5 room and overlap the boundary walls by 0.5 m.

The parent itself has earlier exposure to legacy unsealed corridor replay, so
this is a warm-start correction lineage, not a from-scratch clean baseline.
"""

from dataclasses import fields, replace
import hashlib
from pathlib import Path

from rnn_car_modular.configs.e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver_actdelay12 import (
    CHECKPOINTS,
    CONFIG as _LEGACY_SA5,
    PARENT_CHECKPOINT,
    PARENT_CHECKPOINT_SHA256,
    PARENT_CONCEPTUAL_ITERATION,
    ROLLOUT_LENGTH,
    SAVE_INTERVAL,
    TRAINING_ITERATIONS,
)


RUN_NAME = "sa5_r2_sealed_from_sa4r3_it125_actd12_ne1024_s42_p300_r1"
INTERACTION_LENGTH_M = 10.0
PHYSICAL_WALL_SPAN_M = 15.0
ROOM_HALF_EXTENT_M = 7.5
BOUNDARY_WALL_WIDTH_M = 1.0
BOUNDARY_OVERLAP_M = 0.5

REPO = Path(__file__).resolve().parents[5]
SEALED_SOURCE_FILES = {
    "trainer": (
        REPO
        / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"
    ),
    "corridor_replay": (
        REPO
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
        / "config/charge_skrl/mdp/events/long_corridor_replay.py"
    ),
    "corridor_geometry": (
        REPO
        / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
        / "config/charge_skrl/mdp/events/long_corridor_replay_geometry.py"
    ),
    "legacy_sa5_config": (
        REPO
        / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"
        / "e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver_actdelay12.py"
    ),
}
SEALED_SOURCE_SHA256 = {
    "trainer": "7e3c24793e92b0afb749a3299807a423335f846cc4e3b02956821c7f21ba7999",
    "corridor_replay": "954510f4fba331907c8037249d406ae736498244bb62b3c705366dec0b50071c",
    "corridor_geometry": "f57f4351573af93cd3e42d60554ba9d77956a49ed5614a9df5d2b40b646466a4",
    "legacy_sa5_config": "28bad43a3f6f08ec2a7fb93f70a2c95382e7dd54faa29537a2a576ca351206df",
}


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"SA5-R2 sealed source is missing: {path}")
    with path.open("rb") as source_file:
        return hashlib.file_digest(source_file, "sha256").hexdigest()


for _source_name, _source_path in SEALED_SOURCE_FILES.items():
    _actual_sha256 = _sha256(_source_path)
    if _actual_sha256 != SEALED_SOURCE_SHA256[_source_name]:
        raise RuntimeError(
            "SA5-R2 sealed source drift: "
            f"{_source_name} expected {SEALED_SOURCE_SHA256[_source_name]}, "
            f"got {_actual_sha256}"
        )


METADATA_FIELDS = frozenset({"name", "description", "tags", "notes"})

CONFIG = replace(
    _LEGACY_SA5,
    name="e2e_sa5_r2_sealed_from_sa4r3_it125_actdelay12",
    description=(
        "SA5-R2 300-iteration correction from the original SA4-R3 it125 "
        "parent, with the 10 m corridor interaction zone physically sealed "
        "to the 15 m Stage-5 room boundary."
    ),
    tags=_LEGACY_SA5.tags
    + (
        "sa5_r2",
        "sealed_corridor",
        "same_parent_geometry_correction",
        "optimizer_reset",
        "legacy_sa5_not_parent",
    ),
    notes=(
        f"{_LEGACY_SA5.notes} SA5-R2 correction lineage: restart from the "
        f"same SA4-R3 conceptual-it{PARENT_CONCEPTUAL_ITERATION} parent "
        f"(SHA-256 {PARENT_CHECKPOINT_SHA256}), not from any legacy SA5 "
        "checkpoint. Preserve every ExperimentConfig behavior field from the "
        "completed delay-U{1,2} SA5 run. The frozen environment source keeps "
        f"the corridor interaction length at {INTERACTION_LENGTH_M:.1f} m but "
        f"sets physical wall span to {PHYSICAL_WALL_SPAN_M:.1f} m in the "
        f"15 m room, producing {BOUNDARY_OVERLAP_M:.1f} m overlap with each "
        "north/south boundary wall. This corrects INVALID_GEOMETRY_BYPASS; it "
        "does not retroactively validate the legacy SA5 run. Reset optimizer, "
        "train at most 300 iterations, save every 50, require fixed sealed "
        "corridor plus native/narrow retention gates, and never auto-launch "
        "SA6."
    ),
)


assert CONFIG.initial_stage == 5
assert CONFIG.fixed_stage is True
assert CONFIG.num_envs == 1024
assert CONFIG.seed == 42
assert CONFIG.rollout_length == ROLLOUT_LENGTH == 128
assert CONFIG.timesteps == TRAINING_ITERATIONS * ROLLOUT_LENGTH == 38_400
assert CONFIG.save_interval == SAVE_INTERVAL == 50
assert CONFIG.checkpoint == PARENT_CHECKPOINT
assert CONFIG.no_resume_optimizer is True
assert CONFIG.room_size == ROOM_HALF_EXTENT_M
assert CONFIG.long_corridor_length == INTERACTION_LENGTH_M
assert CONFIG.long_corridor_fraction == 0.10
assert CONFIG.lidar_frame_stack == 8
assert CONFIG.use_action_history is True
assert CONFIG.vlp16_noise_mode == "full"
assert CONFIG.lidar_distractor_eligibility == "valid_return_only"
assert CONFIG.actuator_delay_range == (1, 2)
assert CONFIG.actuator_velocity_scale == (1.0, 1.0)
assert CONFIG.actuator_motor_lag == 1.0
assert CONFIG.future_occupancy_weight == 0.15
assert CHECKPOINTS == (
    (50, "checkpoint_6400.pt"),
    (100, "checkpoint_12800.pt"),
    (150, "checkpoint_19200.pt"),
    (200, "checkpoint_25600.pt"),
    (250, "checkpoint_32000.pt"),
    (300, "checkpoint_38400.pt"),
)
assert 0.5 * PHYSICAL_WALL_SPAN_M - (
    ROOM_HALF_EXTENT_M - 0.5 * BOUNDARY_WALL_WIDTH_M
) == BOUNDARY_OVERLAP_M

_behavior_drift = sorted(
    field.name
    for field in fields(CONFIG)
    if field.name not in METADATA_FIELDS
    and getattr(CONFIG, field.name) != getattr(_LEGACY_SA5, field.name)
)
if _behavior_drift:
    raise RuntimeError(
        "SA5-R2 ExperimentConfig must match the legacy SA5 exactly; sealed "
        f"geometry lives in the frozen environment source. Drift: {_behavior_drift}"
    )
