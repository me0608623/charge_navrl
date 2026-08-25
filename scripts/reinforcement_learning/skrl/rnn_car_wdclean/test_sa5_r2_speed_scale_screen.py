"""CPU contracts for the SA5-R2 c250 vehicle speed-rate screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
import run_sa5_r2_speed_scale_screen_cell as runner  # noqa: E402
import sa5_r2_speed_scale_screen as screen  # noqa: E402


PLAY = HERE.parent / "play_eval/play_rnn_car.py"
VEHICLE = (
    REPO / "rover_rl/src/rover_rl_inference/rover_rl_inference/policy_node.py"
)
FREEZE = REPO / "docs/freeze/sa5_r2_c250_speed_scale_screen_v2.json"


def test_protocol_locks_c250_and_the_four_requested_scales():
    frozen = screen.screen_protocol()
    assert frozen["checkpoint"]["name"] == "c250"
    assert frozen["checkpoint"]["filename"] == "checkpoint_32000.pt"
    assert frozen["checkpoint"]["sha256"] == (
        "df14c45d8dd27b0e327f9447a9a613bf62eaa850af542b474aad682dcd937a4a"
    )
    assert [arm["speed_rate"] for arm in frozen["arms"]] == [
        1.0,
        0.8,
        0.7,
        0.6,
    ]
    assert {arm["speed_rate_obs"] for arm in frozen["arms"]} == {"ego"}
    assert frozen["fixed_evaluation"]["scenario"] == "corridor_lateral"
    assert frozen["fixed_evaluation"]["corridor_density"] == "4S2D"
    assert frozen["fixed_evaluation"]["actuator_delay_steps"] == 1
    assert frozen["fixed_evaluation"]["lidar_distractor_eligibility"] == (
        "valid_return_only"
    )


def test_freeze_matches_runtime_protocol():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == (
        screen.screen_protocol()
    )


def test_play_and_vehicle_implement_the_same_time_dilation_contract():
    play_source = PLAY.read_text(encoding="utf-8")
    vehicle_source = VEHICLE.read_text(encoding="utf-8")
    for key in (
        "max_linear_velocity",
        "max_linear_accel",
        "max_angular_vel",
        "max_angular_accel",
    ):
        assert key in play_source
    assert '"--speed_rate"' in play_source
    assert '"--speed_rate_obs"' in play_source
    assert "[VEHICLE-SPEED-RATE]" in play_source
    assert '"actual_body_stop_fraction"' in play_source
    assert '"actual_body_linear_speed_abs_mean_mps"' in play_source
    assert "last_accel * inv" in vehicle_source
    assert "linear_vel=v * inv" in vehicle_source
    assert "goal_body_x=g_inflate_x" in vehicle_source
    assert "max_linear_velocity * rate" in vehicle_source
    assert "max_angular_accel * rate" in vehicle_source
    assert "self._push_act_hist(accel * inv, cmd_w * inv)" in vehicle_source


def test_vehicle_policy_implementation_hashes_are_frozen():
    frozen = screen.screen_protocol()["vehicle_speed_rate_contract"]
    for key in ("source", "installed_copy"):
        path = Path(frozen[key]["path"])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == frozen[key][
            "sha256"
        ]


def test_runtime_reconciliation_requires_vehicle_semantics_and_no_double_scale():
    report = {
        "speed_rate": 0.7,
        "speed_rate_obs_mode": "ego",
        "speed_rate_action_limits": {
            "max_linear_velocity": 0.7,
            "max_linear_accel": 0.35,
            "max_angular_vel": 0.84,
            "max_angular_accel": 2.1,
        },
        "deployment_speed_scale": 1.0,
        "deployment_scale_samples": 100,
        "deployment_scale_max_abs_error": 0.0,
    }
    observed = runner.verify_speed_rate_runtime(report, 0.7)
    assert observed["speed_rate_obs"] == "ego"
    report["deployment_speed_scale"] = 0.7
    try:
        runner.verify_speed_rate_runtime(report, 0.7)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("stacked downstream scaling was accepted")


def test_runner_enables_identity_timing_audit_and_no_training():
    source = (
        HERE / "run_sa5_r2_speed_scale_screen_cell.py"
    ).read_text(encoding="utf-8")
    for marker in (
        '"--speed_rate"',
        '"--speed_rate_obs"',
        '"--deployment_speed_scale", "1.0"',
        '"--d3_yield_audit"',
        '"--d3_shield_mode"',
        '"baseline"',
        "verify_fixed_actuator_runtime",
        "verify_corridor_runtime",
        "reaction_timing_summary",
    ):
        assert marker in source
    frozen = screen.screen_protocol()
    assert "not authorization to train or start SA6" in frozen[
        "evidence_boundary"
    ]
