"""Contract tests for the READY_NOT_RUN paper realism factorial."""

from __future__ import annotations

import ast
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import types

from rnn_car_modular.configs import (
    e2e_sa1_paper_factorial_a_ideal_d0_p600 as cell_a,
)
from rnn_car_modular.configs import (
    e2e_sa1_paper_factorial_b_full_d0_p600 as cell_b,
)
from rnn_car_modular.configs import (
    e2e_sa1_paper_factorial_c_ideal_u012_p600 as cell_c,
)
from rnn_car_modular.configs import (
    e2e_sa1_paper_factorial_d_full_u012_p600 as cell_d,
)
from rnn_car_modular.experiment_config import apply_experiment_config


REPO = Path(__file__).resolve().parents[4]
FREEZE = REPO / "docs/freeze/paper_realism_factorial_v1_ready_not_run_20260830.json"
FREEZE_DIGEST = REPO / "docs/freeze/paper_realism_factorial_v1_ready_not_run_20260830.sha256"
ROADMAP = REPO / "docs/freeze/paper_ablation_roadmap_v1_20260830.json"
TRAINER = REPO / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"
METADATA = {"name", "description", "tags", "notes"}
LIDAR_FACTOR = {"lidar_no_noise", "vlp16_noise_mode"}
DELAY_FACTOR = {"enable_actuator_dr", "actuator_delay_range"}
CELLS = (cell_a.CONFIG, cell_b.CONFIG, cell_c.CONFIG, cell_d.CONFIG)


def _behavioral_diffs(left, right):
    return {
        field.name
        for field in fields(left)
        if field.name not in METADATA
        and getattr(left, field.name) != getattr(right, field.name)
    }


def test_factorial_pairs_change_exactly_one_conceptual_factor():
    assert _behavioral_diffs(cell_a.CONFIG, cell_b.CONFIG) == LIDAR_FACTOR
    assert _behavioral_diffs(cell_c.CONFIG, cell_d.CONFIG) == LIDAR_FACTOR
    assert _behavioral_diffs(cell_a.CONFIG, cell_c.CONFIG) == DELAY_FACTOR
    assert _behavioral_diffs(cell_b.CONFIG, cell_d.CONFIG) == DELAY_FACTOR


def test_all_cells_share_the_frozen_training_contract():
    for cfg in CELLS:
        assert cfg.checkpoint is None
        assert cfg.no_resume_optimizer is True
        assert cfg.num_envs == 1024
        assert cfg.rollout_length == 128
        assert cfg.timesteps == 76_800
        assert cfg.timesteps // cfg.rollout_length == 600
        assert cfg.save_interval == 100
        assert cfg.model_init_seed == -1
        assert cfg.no_domain_randomization is True
        assert cfg.lidar_distractor_eligibility == "valid_return_only"
        assert cfg.actuator_velocity_scale == (1.0, 1.0)
        assert cfg.actuator_motor_lag == 1.0
        assert cfg.actuator_motor_lag_by_channel is None
        assert cfg.obs_delay_steps == (0, 0)
        assert cfg.speed_rate == 1.0


def test_factor_levels_are_explicit_and_not_full_domain_randomization():
    assert cell_a.CONFIG.vlp16_noise_mode == cell_c.CONFIG.vlp16_noise_mode == "ideal"
    assert cell_b.CONFIG.vlp16_noise_mode == cell_d.CONFIG.vlp16_noise_mode == "full"
    assert cell_a.CONFIG.enable_actuator_dr is False
    assert cell_b.CONFIG.enable_actuator_dr is False
    assert cell_c.CONFIG.enable_actuator_dr is True
    assert cell_d.CONFIG.enable_actuator_dr is True
    assert cell_c.CONFIG.actuator_delay_range == (0, 2)
    assert cell_d.CONFIG.actuator_delay_range == (0, 2)
    for cfg in CELLS:
        assert cfg.lidar_displacement_std_per_meter_dr is None
        assert cfg.lidar_displacement_std_soft_dr is None
        assert cfg.lidar_displacement_std_dr is None
        assert cfg.lidar_hole_rate_dr is None


def test_cli_training_seed_override_keeps_model_seed_bound_to_the_run_seed():
    args = types.SimpleNamespace(seed=43)
    apply_experiment_config(args, cell_d.CONFIG, argv=["prog", "--seed", "43"])
    assert args.seed == 43
    assert args.model_init_seed == -1


def test_trainer_reseeds_immediately_before_network_construction():
    source = TRAINER.read_text(encoding="utf-8")
    ast.parse(source, filename=str(TRAINER))
    assert "[MODEL-INIT-SEED]" in source
    reseed = source.index("torch.manual_seed(_model_init_seed)")
    extractor = source.index("extractor = LidarStateExtractor(")
    policy = source.index("policy_head = PolicyHead(")
    assert reseed < extractor < policy


def test_freeze_is_ready_not_run_and_uses_training_seed_as_n():
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert payload["status"] == "READY_NOT_RUN"
    assert payload["training_contract"]["training_seeds"] == [42, 43, 44]
    assert payload["training_contract"]["seed_is_statistical_unit"] is True
    assert payload["training_contract"]["runs"] == 12
    assert payload["evaluation_contract"]["evaluation_seeds"] == [515, 616, 717]
    assert payload["analysis_contract"]["do_not_use_episode_count_as_independent_n"] is True
    assert payload["automatic_actions"]["gpu_smoke_authorized"] is False
    assert payload["automatic_actions"]["training_launch_authorized"] is False
    assert payload["automatic_actions"]["fixed_gate_launch_authorized"] is False


def test_freeze_fails_closed_on_any_locked_source_drift():
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    for relative, expected in payload["source_locks"]["files"].items():
        with (REPO / relative).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == expected, relative


def test_freeze_payload_matches_its_external_digest():
    expected, filename = FREEZE_DIGEST.read_text(encoding="utf-8").split()
    assert filename == FREEZE.name
    with FREEZE.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    assert actual == expected


def test_gate_cannot_start_until_all_required_metrics_are_instrumented():
    payload = json.loads(FREEZE.read_text(encoding="utf-8"))
    gate = payload["evaluation_contract"]
    assert gate["status"] == "SPECIFIED_BUT_RUNNER_METRICS_NOT_YET_COMPLETE"
    assert set(gate["required_metrics"]) == {
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "traversal_time_s",
        "path_efficiency",
        "minimum_obstacle_clearance_m",
    }
    assert gate["blocking_instrumentation_gap"]


def test_ablation_roadmap_keeps_expensive_followups_ordered_and_unauthorized():
    payload = json.loads(ROADMAP.read_text(encoding="utf-8"))
    assert payload["status"] == "DESIGN_ONLY_NO_GPU_AUTHORIZATION"
    assert payload["priority_order"][:2] == [
        "P0_unified_gate_metrics",
        "P1_lidar_x_delay_factorial",
    ]
    assert payload["experiments"]["P1_lidar_x_delay_factorial"]["total_training_runs"] == 12
    assert payload["experiments"]["P2_lidar_component_ablation"]["status"] == "DESIGN_ONLY_AFTER_P1"
    assert payload["experiments"]["P3_domain_randomization_source_ablation"]["status"] == "BLOCKED_ON_RUNTIME_PROOF_AND_CALIBRATION"
    assert all(value is False for value in payload["automatic_actions"].values())
