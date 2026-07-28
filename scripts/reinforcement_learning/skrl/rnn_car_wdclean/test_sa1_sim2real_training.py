"""Training-side contract tests for the fresh SA1 sim-to-real lineage."""

import ast
import contextlib
from dataclasses import fields, replace
import importlib.util
import io
import json
import pathlib
import sys
import tempfile
import types
import unittest

import torch


_REPO = pathlib.Path(__file__).resolve().parents[4]
_SKRL = _REPO / "scripts" / "reinforcement_learning" / "skrl"
if str(_SKRL) not in sys.path:
    sys.path.insert(0, str(_SKRL))

_ACTUATOR_DR = (
    _REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/domain_randomization/actuator_dr.py"
)
_ACTION_TERM = (
    _REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py"
)
_OVERRIDES = _SKRL / "utils" / "charge_env_overrides.py"
_FIXED_ACTUATOR_EVAL = (
    _SKRL / "rnn_car_wdclean" / "fixed_actuator_eval.py"
)
_ACTION_CONTRACT_EXPORTER = (
    _SKRL / "rnn_car_wdclean" / "export_action_contract.py"
)
_ACTION_CONTRACT_FIXTURE = (
    _REPO / "docs/freeze/sa1_action_contract_v1.json"
)
_TRAINER = _SKRL / "train" / "train_rnn_car_wdclip.py"
_EVENTS = (
    _REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events"
)
_BEHAVIOR_SCHEDULER = _EVENTS / "behavior_scheduler.py"
_NARROW_GEOMETRY = _EVENTS / "narrow_passage_bridge_geometry.py"


def _load_by_path(alias: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


_actuator_dr = _load_by_path("_sa1_actuator_dr", _ACTUATOR_DR)
_overrides = _load_by_path("_sa1_charge_env_overrides", _OVERRIDES)
_fixed_eval = _load_by_path("_sa1_fixed_actuator_eval", _FIXED_ACTUATOR_EVAL)
_action_contract = _load_by_path(
    "_sa1_action_contract_exporter", _ACTION_CONTRACT_EXPORTER
)
if str(_EVENTS) not in sys.path:
    sys.path.insert(0, str(_EVENTS))
_behavior_scheduler = _load_by_path(
    "_sa1_behavior_scheduler", _BEHAVIOR_SCHEDULER
)
_narrow_geometry = _load_by_path(
    "_sa1_narrow_geometry", _NARROW_GEOMETRY
)

from rnn_car_modular.configs.e2e_sa1_k8_obb import CONFIG as CLEAN_SA1  # noqa: E402
from rnn_car_modular.configs.e2e_sa1_k8_obb_scene_mix import (  # noqa: E402
    CONFIG as SCENE_MIX_SA1,
    DEPLOYMENT_CORRIDOR_DENSITY_MIX,
    LONG_CORRIDOR_FRACTION,
    NARROW_PASSAGE_FRACTION,
    NATIVE_SCENE_FRACTION,
    SA1_ACTION_HISTORY_ACCEL_NORMALIZER,
    SA1_ACTION_HISTORY_OMEGA_NORMALIZER,
    SA1_NARROW_SEGMENT_LENGTH_M,
)
from rnn_car_modular.configs.e2e_sa1_k8_obb_actuator_delay_only import (  # noqa: E402
    ACTUATOR_DELAY_SUPPORT_STEPS,
    CONFIG as DELAY_ONLY_SA1,
    NEUTRAL_MOTOR_LAG_ALPHA,
    NEUTRAL_VELOCITY_SCALE,
)
from rnn_car_modular.configs.e2e_sa1_k8_obb_sim2real_v1 import (  # noqa: E402
    CONFIG as SIM2REAL_SA1,
    VLP16_NOISE_MODE,
)
from rnn_car_modular.experiment_config import apply_experiment_config  # noqa: E402


class _StubEnv:
    def __init__(self, num_envs: int = 4):
        self.num_envs = num_envs
        self.device = "cpu"
        self.episode_length_buf = torch.ones(num_envs, dtype=torch.long)


class _StubDiffDrive:
    def __init__(self):
        self.enable_actuator_dr = False
        self.actuator_delay_range = (0, 0)
        self.actuator_velocity_scale = (1.0, 1.0)
        self.actuator_motor_lag = 1.0
        self.actuator_motor_lag_by_channel = None


class SA1ConfigLockTest(unittest.TestCase):
    def test_scene_mix_changes_exactly_the_intended_fields(self):
        differing = {
            field.name
            for field in fields(CLEAN_SA1)
            if getattr(CLEAN_SA1, field.name)
            != getattr(SCENE_MIX_SA1, field.name)
        }
        self.assertEqual(
            differing,
            {
                "name",
                "description",
                "action_history_accel_normalizer",
                "action_history_omega_normalizer",
                "narrow_passage_fraction",
                "narrow_passage_segment_length",
                "narrow_passage_final_stress_ratio",
                "narrow_passage_fixed_width_range",
                "narrow_passage_fixed_yaw_limit_deg",
                "long_corridor_fraction",
                "long_corridor_obstacle_count_mix",
                "long_corridor_dynamic_motion_mode",
                "long_corridor_random_2d_kinematics",
                "tags",
                "notes",
            },
        )

    def test_scene_mix_is_the_frozen_native_narrow_corridor_distribution(self):
        self.assertEqual(SCENE_MIX_SA1.previous_stage_replay_fraction, 0.0)
        self.assertEqual(SCENE_MIX_SA1.narrow_passage_fraction, 0.12)
        self.assertEqual(SCENE_MIX_SA1.long_corridor_fraction, 0.10)
        self.assertAlmostEqual(
            NATIVE_SCENE_FRACTION
            + NARROW_PASSAGE_FRACTION
            + LONG_CORRIDOR_FRACTION,
            1.0,
        )
        self.assertEqual(SCENE_MIX_SA1.narrow_passage_fixed_width_range, (1.2, 1.4))
        self.assertEqual(SCENE_MIX_SA1.narrow_passage_fixed_yaw_limit_deg, 4.0)
        self.assertEqual(
            SCENE_MIX_SA1.narrow_passage_segment_length,
            SA1_NARROW_SEGMENT_LENGTH_M,
        )
        self.assertEqual(
            SCENE_MIX_SA1.action_history_accel_normalizer,
            SA1_ACTION_HISTORY_ACCEL_NORMALIZER,
        )
        self.assertEqual(
            SCENE_MIX_SA1.action_history_omega_normalizer,
            SA1_ACTION_HISTORY_OMEGA_NORMALIZER,
        )
        self.assertEqual(SA1_ACTION_HISTORY_ACCEL_NORMALIZER, 0.5)
        self.assertEqual(SA1_ACTION_HISTORY_OMEGA_NORMALIZER, 1.2)
        self.assertEqual(SCENE_MIX_SA1.long_corridor_free_width, 4.0)
        self.assertEqual(SCENE_MIX_SA1.long_corridor_length, 10.0)
        self.assertEqual(
            SCENE_MIX_SA1.long_corridor_obstacle_count_mix,
            DEPLOYMENT_CORRIDOR_DENSITY_MIX,
        )
        self.assertEqual(
            SCENE_MIX_SA1.long_corridor_dynamic_motion_mode,
            "env_stratified",
        )
        self.assertEqual(
            SCENE_MIX_SA1.long_corridor_random_2d_kinematics,
            "wander",
        )
        self.assertEqual(SCENE_MIX_SA1.long_corridor_gate_aligned_share, 0.0)
        self.assertIsNone(SCENE_MIX_SA1.teacher_retention_checkpoint)
        self.assertEqual(SCENE_MIX_SA1.teacher_retention_weight, 0.0)

    def test_sa1_narrow_segments_close_every_sampled_gap_endpoint(self):
        common = {
            "barrier_x": 0.0,
            "start_x": -3.0,
            "goal_x": 3.0,
            "goal_y": 0.0,
            "room_half_extent": CLEAN_SA1.room_size,
            "boundary_wall_width": 1.0,
        }
        old_ok, old_reason = _narrow_geometry.validate_constructed_scene(
            gap_width=1.2,
            gap_center=1.0,
            segment_length=9.0,
            **common,
        )
        self.assertIs(old_ok, False)
        self.assertIn("does not reach the boundary", old_reason)

        for gap_center in (-1.0, 1.0):
            for gap_width in (1.2, 1.4):
                with self.subTest(
                    gap_center=gap_center,
                    gap_width=gap_width,
                ):
                    ok, reason = _narrow_geometry.validate_constructed_scene(
                        gap_width=gap_width,
                        gap_center=gap_center,
                        segment_length=SA1_NARROW_SEGMENT_LENGTH_M,
                        **common,
                    )
                    self.assertIs(ok, True, reason)

    def test_delay_only_arm_changes_exactly_the_intended_fields(self):
        differing = {
            field.name
            for field in fields(SCENE_MIX_SA1)
            if getattr(SCENE_MIX_SA1, field.name)
            != getattr(DELAY_ONLY_SA1, field.name)
        }
        self.assertEqual(
            differing,
            {
                "name",
                "description",
                "enable_actuator_dr",
                "actuator_velocity_scale",
                "actuator_motor_lag",
                "tags",
                "notes",
            },
        )

    def test_fresh_lineage_action_history_normalization_reaches_both_groups(self):
        def _term():
            return types.SimpleNamespace(params={"stack_size": 2})

        env_cfg = types.SimpleNamespace(
            observations=types.SimpleNamespace(
                policy=types.SimpleNamespace(past_actions=_term()),
                critic=types.SimpleNamespace(past_actions=_term()),
            )
        )
        args = types.SimpleNamespace()
        apply_experiment_config(args, SCENE_MIX_SA1, argv=["prog"])

        changed = _overrides._apply_action_history_normalization(
            env_cfg, args
        )

        self.assertIs(changed, True)
        for group_name in ("policy", "critic"):
            params = getattr(
                env_cfg.observations, group_name
            ).past_actions.params
            self.assertEqual(
                params,
                {
                    "stack_size": 2,
                    "a_max": 0.5,
                    "omega_max": 1.2,
                },
            )

    def test_legacy_config_keeps_action_history_normalization_untouched(self):
        term = types.SimpleNamespace(params={"stack_size": 2})
        env_cfg = types.SimpleNamespace(
            observations=types.SimpleNamespace(
                policy=types.SimpleNamespace(past_actions=term)
            )
        )
        args = types.SimpleNamespace()
        apply_experiment_config(args, CLEAN_SA1, argv=["prog"])

        changed = _overrides._apply_action_history_normalization(
            env_cfg, args
        )

        self.assertIs(changed, False)
        self.assertEqual(term.params, {"stack_size": 2})

    def test_partial_or_invalid_action_history_contract_fails_closed(self):
        term = types.SimpleNamespace(params={"stack_size": 2})
        env_cfg = types.SimpleNamespace(
            observations=types.SimpleNamespace(
                policy=types.SimpleNamespace(past_actions=term)
            )
        )
        for accel, omega in ((0.5, None), (None, 1.2), (0.0, 1.2), (0.5, -1.0)):
            with self.subTest(accel=accel, omega=omega):
                args = types.SimpleNamespace(
                    action_history_accel_normalizer=accel,
                    action_history_omega_normalizer=omega,
                )
                with self.assertRaises((ValueError, RuntimeError)):
                    _overrides._apply_action_history_normalization(
                        env_cfg, args
                    )

    def test_trainer_calls_action_history_normalization(self):
        tree = ast.parse(
            _TRAINER.read_text(encoding="utf-8"),
            filename=str(_TRAINER),
        )
        main = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        calls = {
            node.func.id
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
        }
        self.assertIn("_apply_action_history_normalization", calls)

    def test_combined_arm_adds_only_measured_lidar_noise(self):
        differing = {
            field.name
            for field in fields(DELAY_ONLY_SA1)
            if getattr(DELAY_ONLY_SA1, field.name)
            != getattr(SIM2REAL_SA1, field.name)
        }
        self.assertEqual(
            differing,
            {
                "name",
                "description",
                "lidar_no_noise",
                "vlp16_noise_mode",
                "tags",
                "notes",
            },
        )

    def test_delay_support_is_the_frozen_zero_200_400_ms_set(self):
        self.assertEqual(
            DELAY_ONLY_SA1.actuator_delay_range,
            ACTUATOR_DELAY_SUPPORT_STEPS,
        )
        lo, hi = DELAY_ONLY_SA1.actuator_delay_range
        self.assertEqual(
            [round(step * 0.2 * 1000) for step in range(lo, hi + 1)],
            [0, 200, 400],
        )
        self.assertNotEqual(lo, hi)

    def test_uncalibrated_actuator_factors_are_neutral(self):
        self.assertEqual(
            DELAY_ONLY_SA1.actuator_velocity_scale,
            NEUTRAL_VELOCITY_SCALE,
        )
        self.assertEqual(
            DELAY_ONLY_SA1.actuator_motor_lag,
            NEUTRAL_MOTOR_LAG_ALPHA,
        )
        self.assertIsNone(DELAY_ONLY_SA1.actuator_motor_lag_by_channel)

    def test_mainline_is_fresh_and_keeps_unmeasured_dr_off(self):
        self.assertIsNone(SIM2REAL_SA1.checkpoint)
        self.assertIs(SIM2REAL_SA1.no_resume_optimizer, True)
        self.assertIs(SIM2REAL_SA1.use_action_history, True)
        self.assertEqual(SIM2REAL_SA1.obs_delay_steps, (0, 0))
        self.assertIs(SIM2REAL_SA1.no_domain_randomization, True)
        self.assertIs(SIM2REAL_SA1.lidar_no_noise, False)
        self.assertEqual(SIM2REAL_SA1.vlp16_noise_mode, VLP16_NOISE_MODE)
        self.assertEqual(VLP16_NOISE_MODE, "full")
        self.assertEqual(SIM2REAL_SA1.narrow_passage_fraction, 0.12)
        self.assertEqual(SIM2REAL_SA1.long_corridor_fraction, 0.10)

    def test_mainline_uses_k8_e2e_without_recurrent_policy_state(self):
        self.assertEqual(SIM2REAL_SA1.lidar_frame_stack, 8)
        self.assertIs(SIM2REAL_SA1.end_to_end_frame_stack, True)
        self.assertEqual(SIM2REAL_SA1.aux_profile, "none")
        self.assertEqual(SIM2REAL_SA1.rnn_lr, 0.0)
        self.assertEqual(SIM2REAL_SA1.aux_lr, 0.0)

    def test_full_lidar_mode_applies_the_measured_values(self):
        params = {
            "displacement_std_per_meter": 99.0,
            "displacement_std": 99.0,
            "displacement_std_soft": 99.0,
            "hole_rate": 99.0,
            "distractor_rate": 99.0,
            "per_ring_bias": False,
            "distance_bias_k": 99.0,
            "distance_bias_b": 99.0,
            "block_dropout_prob": 99.0,
            "example_dr_min": 99.0,
            "example_dr_max": 99.0,
        }
        lidar_term = types.SimpleNamespace(params=params, noise=object())
        env_cfg = types.SimpleNamespace(
            observations=types.SimpleNamespace(
                policy=types.SimpleNamespace(lidar_scan=lidar_term)
            )
        )
        args = types.SimpleNamespace(
            vlp16_noise_mode=SIM2REAL_SA1.vlp16_noise_mode,
            lidar_no_noise=SIM2REAL_SA1.lidar_no_noise,
        )

        _overrides._apply_lidar_noise_config(env_cfg, args)

        self.assertEqual(params["displacement_std_per_meter"], 0.0)
        self.assertEqual(params["displacement_std"], 0.0)
        self.assertAlmostEqual(params["displacement_std_soft"], 0.008672)
        self.assertAlmostEqual(params["hole_rate"], 0.194859)
        self.assertAlmostEqual(params["distractor_rate"], 0.002515)
        self.assertIs(params["per_ring_bias"], True)
        self.assertEqual(params["distance_bias_k"], 0.0)
        self.assertEqual(params["distance_bias_b"], 0.0)
        self.assertEqual(params["block_dropout_prob"], 0.0)
        self.assertEqual(params["example_dr_min"], 0.0)
        self.assertEqual(params["example_dr_max"], 0.0)


class BehaviorSchedulerCapacityTest(unittest.TestCase):
    def test_reserved_capacity_does_not_increase_native_active_count(self):
        scheduler = _behavior_scheduler.BehaviorScheduler(
            stage_config={"behavior_mix": {"static": 1.0}},
            num_envs=4,
            max_obstacles=10,
            active_obstacles=2,
            device="cpu",
        )

        self.assertEqual(scheduler.max_obstacles, 10)
        self.assertEqual(scheduler.active_obstacles, 2)
        self.assertEqual(scheduler.behavior_type.shape, (4, 10))
        self.assertEqual(
            scheduler._allocate_counts(
                scheduler.behavior_mix, scheduler.active_obstacles
            ),
            {"static": 2},
        )

    def test_stage_update_fails_if_active_count_exceeds_capacity(self):
        scheduler = _behavior_scheduler.BehaviorScheduler(
            stage_config={"behavior_mix": {"static": 1.0}},
            num_envs=1,
            max_obstacles=10,
            active_obstacles=2,
            device="cpu",
        )
        scheduler.update_stage(
            {
                "behavior_mix": {"static": 0.5, "patrol": 0.5},
                "num_obstacles_static": 4,
                "num_obstacles_dynamic": 2,
            }
        )
        self.assertEqual(scheduler.active_obstacles, 6)

        with self.assertRaisesRegex(RuntimeError, "reserved.*capacity"):
            scheduler.update_stage(
                {
                    "num_obstacles_static": 6,
                    "num_obstacles_dynamic": 5,
                }
            )


class PerChannelMotorLagTest(unittest.TestCase):
    def test_legacy_scalar_applies_to_both_channels(self):
        env = _StubEnv()
        target = torch.ones(4, 2)
        actual = _actuator_dr.apply_motor_response_lag(env, target, 0.5)
        self.assertTrue(torch.equal(actual, torch.full((4, 2), 0.5)))

    def test_linear_and_angular_channels_have_independent_response(self):
        env = _StubEnv()
        target = torch.ones(4, 2)

        first = _actuator_dr.apply_motor_response_lag(
            env, target, (0.25, 0.75)
        )
        second = _actuator_dr.apply_motor_response_lag(
            env, target, (0.25, 0.75)
        )

        self.assertTrue(
            torch.allclose(first, torch.tensor([0.25, 0.75]).repeat(4, 1))
        )
        self.assertTrue(
            torch.allclose(
                second,
                torch.tensor([0.4375, 0.9375]).repeat(4, 1),
            )
        )

    def test_invalid_channel_lag_fails_closed(self):
        env = _StubEnv()
        target = torch.ones(4, 2)
        for alpha in ((0.3,), (0.3, 0.5, 0.7), (-0.1, 0.5), (0.3, 1.1)):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    _actuator_dr.apply_motor_response_lag(
                        env, target, alpha
                    )

    def test_channel_lag_wires_from_experiment_config_to_action_term(self):
        calibrated = replace(
            DELAY_ONLY_SA1,
            actuator_motor_lag_by_channel=(0.3, 0.5),
        )
        args = types.SimpleNamespace()
        apply_experiment_config(args, calibrated, argv=["prog"])
        env_cfg = types.SimpleNamespace(
            actions=types.SimpleNamespace(diff_drive=_StubDiffDrive())
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            _overrides._apply_actuator_dr_config(env_cfg, args)

        diff = env_cfg.actions.diff_drive
        self.assertEqual(diff.actuator_motor_lag_by_channel, (0.3, 0.5))
        self.assertIn(
            "motor_lag alpha=(v=0.3, omega=0.5)",
            output.getvalue(),
        )

    def test_action_term_prefers_channel_lag_when_present(self):
        tree = ast.parse(
            _ACTION_TERM.read_text(encoding="utf-8"),
            filename=str(_ACTION_TERM),
        )
        process = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DiscreteDifferentialDriveAction"
            for node in node.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "process_actions"
        )
        attrs = {
            node.attr
            for node in ast.walk(process)
            if isinstance(node, ast.Attribute)
        }
        self.assertIn("actuator_motor_lag_by_channel", attrs)
        self.assertIn("actuator_motor_lag", attrs)


class FixedDelayProfileTest(unittest.TestCase):
    def test_sa1_gate_profile_does_not_reintroduce_bridge_dynamics(self):
        self.assertEqual(
            _fixed_eval.fixed_actuator_cli_args(1, "sa1_delay_only"),
            [
                "--enable_actuator_dr",
                "--actuator_delay_range",
                "1",
                "1",
                "--actuator_velocity_scale",
                "1.0",
                "1.0",
                "--actuator_motor_lag",
                "1.0",
            ],
        )
        metadata = _fixed_eval.fixed_actuator_metadata(
            1, "sa1_delay_only"
        )
        self.assertEqual(metadata["profile"], "sa1_delay_only")
        self.assertEqual(metadata["delay_ms"], 200)
        self.assertEqual(metadata["velocity_scale_range"], [1.0, 1.0])
        self.assertEqual(metadata["motor_lag_alpha"], 1.0)
        self.assertEqual(metadata["randomized_components"], [])

    def test_sa1_gate_profile_runtime_marker_is_fail_closed(self):
        complete = (
            "[SIM2REAL] Actuator DR: delay=(1, 1) steps, "
            "vel_scale=(1.0, 1.0), motor_lag alpha=1.0, "
            "pipeline=decode->delay->scale->lag, "
            "history=issued_command_queue\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            log_path = pathlib.Path(tmp) / "play.log"
            log_path.write_text(complete, encoding="utf-8")
            _fixed_eval.verify_fixed_actuator_runtime(
                log_path, 1, "sa1_delay_only"
            )
            with self.assertRaisesRegex(
                RuntimeError, "actuator gate wiring incomplete"
            ):
                _fixed_eval.verify_fixed_actuator_runtime(
                    log_path, 1, "bridge"
                )

    def test_all_gate_runners_expose_the_actuator_profile(self):
        for filename in (
            "run_gate2_suite.py",
            "run_corridor_motion_suite.py",
            "run_narrow_path_suite.py",
        ):
            with self.subTest(filename=filename):
                source = (
                    _SKRL / "rnn_car_wdclean" / filename
                ).read_text(encoding="utf-8")
                self.assertIn('"--actuator-profile"', source)
                self.assertIn("args.actuator_profile", source)


class ActionContractFixtureTest(unittest.TestCase):
    def test_frozen_fixture_matches_the_current_training_decoder(self):
        fixture = json.loads(
            _ACTION_CONTRACT_FIXTURE.read_text(encoding="utf-8")
        )
        self.assertEqual(fixture, _action_contract.build_contract())

    def test_fixture_covers_all_361_zero_state_actions(self):
        fixture = _action_contract.build_contract()
        rows = fixture["zero_state_all_361_actions"]
        self.assertEqual(len(rows), 361)
        pairs = {
            (row["linear_index"], row["angular_index"])
            for row in rows
        }
        self.assertEqual(
            pairs,
            {
                (linear_index, angular_index)
                for linear_index in range(19)
                for angular_index in range(19)
            },
        )

    def test_fixture_locks_reverse_and_angular_slew_boundaries(self):
        fixture = _action_contract.build_contract()
        sequences = {
            item["name"]: item["steps"]
            for item in fixture["stateful_sequences"]
        }
        reverse = sequences["reverse_saturation"]
        self.assertEqual(
            [step["issued_velocity_mps"] for step in reverse],
            [-0.1, -0.2, -0.2, -0.2],
        )
        positive = sequences["positive_omega_slew"]
        self.assertEqual(
            [step["issued_omega_rad_s"] for step in positive],
            [0.6, 1.2, 1.2],
        )
        flip = sequences["omega_sign_flip"]
        self.assertEqual(
            [step["issued_omega_rad_s"] for step in flip],
            [0.6, 1.2, 0.6, 0.0, -0.6, -1.2],
        )


if __name__ == "__main__":
    unittest.main()
