"""CPU-only contract tests for the SA5-R2 exact-parent sealed baseline."""

import json
from pathlib import Path

from rnn_car_wdclean import sa5_checkpoint_screen as screen
from rnn_car_wdclean import sa5_r2_parent_baseline_queue as baseline


def test_frozen_protocol_matches_code_exactly():
    frozen = json.loads(baseline.FREEZE.read_text(encoding="utf-8"))
    assert frozen == baseline.baseline_protocol()
    assert frozen["sha256"] == (
        "a4a8f14d471fa8d7fab02db05011bb9deade8ec9838d468a343c208d34eff362"
    )


def test_exact_parent_identity_is_hash_locked():
    parent = baseline.verify_parent()
    assert parent["conceptual_iteration"] == 125
    assert "sa4_r3_cont25_from_it100" in parent["checkpoint"]
    assert "sa5_sim2real" not in parent["checkpoint"]
    assert parent["sha256"] == baseline.PARENT_SHA256


def test_baseline_uses_all_four_sealed_corridor_families():
    protocol = baseline.baseline_protocol()
    fixed = protocol["fixed_evaluation"]
    assert tuple(fixed["scenarios"]) == screen.CORRIDOR_SCENARIOS
    assert fixed["stage"] == 5
    assert fixed["seed"] == 818
    assert fixed["num_envs"] == 64
    assert fixed["steps"] == 2500
    assert fixed["actuator_delay_steps"] == 1
    assert fixed["lidar_noise_mode"] == "full"
    assert fixed["lidar_distractor_eligibility"] == "valid_return_only"
    assert fixed["scene"]["corridor_length_m"] == 10.0
    assert fixed["scene"]["corridor_wall_span_m"] == 15.0
    assert fixed["scene"]["corridor_expected_boundary_overlap_m"] == 0.5
    assert fixed["scene"]["corridor_sealed_to_boundary"] is True


def test_each_scene_invocation_uses_the_frozen_motion_mode_and_actuator():
    for scenario in screen.CORRIDOR_SCENARIOS:
        args = baseline.screen_cell.build_scene_args(
            scenario, Path("/tmp/baseline-corridor.json")
        )
        assert "--long_corridor_eval" in args
        assert "--long_corridor_motion_mode" in args
        assert "--enable_actuator_dr" in args
        delay = args.index("--actuator_delay_range")
        assert args[delay + 1 : delay + 3] == ["1", "1"]


def test_runner_is_diagnostic_only_and_fail_closed():
    source = Path(baseline.__file__).read_text(encoding="utf-8")
    assert "COMPLETE_VALID_DIAGNOSTIC_BASELINE" in source
    assert "INCOMPLETE_NO_BASELINE" in source
    assert 'if int(metrics["n"]) < screen.MIN_EPISODES:' in source
    assert "source_fingerprint() != before" in source
    assert "screen_before = screen_queue.source_fingerprint()" in source
    assert "expected_fingerprint=screen_before" in source
    assert "expected_fingerprint=before" not in source
    assert '"training_started": False' in source
    assert '"sa6_started": False' in source
    assert "train_rnn_car_wdclip" not in source
    assert "systemctl" not in source
