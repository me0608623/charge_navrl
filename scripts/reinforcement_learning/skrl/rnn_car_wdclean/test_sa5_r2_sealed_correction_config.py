"""CPU-only contract tests for the SA5-R2 sealed correction lineage."""

from dataclasses import fields
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from rnn_car_modular.configs import (
    e2e_sa5_k8_obb_sim2real_from_sa4r3_it125_hwaiver_actdelay12 as legacy_mod,
)
from rnn_car_modular.configs import (
    e2e_sa5_r2_sealed_from_sa4r3_it125_actdelay12 as sealed_mod,
)


REPO = Path(__file__).resolve().parents[4]
LEGACY = legacy_mod.CONFIG
SEALED = sealed_mod.CONFIG
FREEZE = REPO / "docs/freeze/sa5_r2_sealed_correction_v1.json"
GEOMETRY = sealed_mod.SEALED_SOURCE_FILES["corridor_geometry"]


def _load_geometry_module():
    spec = importlib.util.spec_from_file_location(
        "sa5_r2_sealed_geometry_contract", GEOMETRY
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parent_is_the_original_sa4_it125_and_not_a_legacy_sa5_checkpoint():
    assert SEALED.checkpoint == legacy_mod.PARENT_CHECKPOINT
    assert sealed_mod.PARENT_CONCEPTUAL_ITERATION == 125
    assert "sa4_r3_cont25_from_it100" in SEALED.checkpoint
    assert "sa5_sim2real" not in SEALED.checkpoint
    assert hashlib.sha256(Path(SEALED.checkpoint).read_bytes()).hexdigest() == (
        sealed_mod.PARENT_CHECKPOINT_SHA256
    )


def test_experiment_config_changes_metadata_only():
    changed = {
        field.name
        for field in fields(SEALED)
        if getattr(SEALED, field.name) != getattr(LEGACY, field.name)
    }
    assert changed == sealed_mod.METADATA_FIELDS


def test_sealed_geometry_is_independent_of_interaction_length():
    geometry = _load_geometry_module()
    corridor = geometry.LongCorridorSpec(
        free_width=SEALED.long_corridor_free_width,
        length=sealed_mod.INTERACTION_LENGTH_M,
        wall_span_length=sealed_mod.PHYSICAL_WALL_SPAN_M,
    )
    assert corridor.length == 10.0
    assert corridor.physical_wall_span == 15.0
    assert geometry.wall_boundary_overlap(
        corridor,
        room_half_extent=sealed_mod.ROOM_HALF_EXTENT_M,
        boundary_wall_width=sealed_mod.BOUNDARY_WALL_WIDTH_M,
    ) == sealed_mod.BOUNDARY_OVERLAP_M == 0.5


def test_training_path_carries_room_extent_and_wall_span_across_resets():
    trainer = sealed_mod.SEALED_SOURCE_FILES["trainer"].read_text(encoding="utf-8")
    replay = sealed_mod.SEALED_SOURCE_FILES["corridor_replay"].read_text(
        encoding="utf-8"
    )
    assert "room_half_extent=float(" in trainer
    assert "boundary_wall_width=1.0" in trainer
    assert "wall_span_length = 2.0 * float(room_half_extent)" in replay
    assert '"wall_span_length": spec.physical_wall_span' in replay
    assert "if boundary_overlap < 0.0:" in replay


def test_measurement_defining_sources_are_hash_locked():
    assert set(sealed_mod.SEALED_SOURCE_FILES) == set(
        sealed_mod.SEALED_SOURCE_SHA256
    )
    for name, path in sealed_mod.SEALED_SOURCE_FILES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == (
            sealed_mod.SEALED_SOURCE_SHA256[name]
        )


def test_budget_optimizer_sensor_actuator_and_scene_are_frozen():
    assert SEALED.no_resume_optimizer is True
    assert SEALED.timesteps == 38_400
    assert SEALED.save_interval == 50
    assert SEALED.num_envs == 1024
    assert SEALED.seed == 42
    assert SEALED.vlp16_noise_mode == "full"
    assert SEALED.lidar_distractor_eligibility == "valid_return_only"
    assert SEALED.actuator_delay_range == (1, 2)
    assert SEALED.actuator_velocity_scale == (1.0, 1.0)
    assert SEALED.long_corridor_fraction == 0.10
    assert SEALED.narrow_passage_fraction == 0.12


def test_freeze_matches_config_and_geometry_contract():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["schema"] == "sa5_r2_sealed_correction/v1"
    assert freeze["run_name"] == sealed_mod.RUN_NAME
    assert freeze["parent"]["checkpoint"] == sealed_mod.PARENT_CHECKPOINT
    assert freeze["parent"]["sha256"] == sealed_mod.PARENT_CHECKPOINT_SHA256
    assert freeze["geometry"]["interaction_length_m"] == 10.0
    assert freeze["geometry"]["physical_wall_span_m"] == 15.0
    assert freeze["geometry"]["boundary_overlap_m"] == 0.5
    assert freeze["source_sha256"] == sealed_mod.SEALED_SOURCE_SHA256


def test_freeze_requires_same_parent_sealed_pretraining_baseline():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    baseline = freeze["pre_training_baseline_requirement"]
    assert baseline["checkpoint"] == (
        "the exact frozen SA4-R3 conceptual-it125 parent"
    )
    assert baseline["geometry"] == "sealed"
    assert baseline["families"] == [
        "lateral",
        "longitudinal",
        "random_2d",
        "mixed",
    ]
    assert set(baseline["required_metrics"]) == {
        "sr",
        "cr",
        "to",
        "wall_cr",
        "obstacle_cr",
    }


def test_config_cannot_launch_sa6_or_other_processes():
    tree = ast.parse(Path(sealed_mod.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for forbidden in ("subprocess", "os", "shutil", "socket"):
        assert forbidden not in imported
    assert "never auto-launch sa6" in SEALED.notes.lower()
