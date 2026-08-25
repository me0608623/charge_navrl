from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_checkpoint_screen as screen  # noqa: E402
import sa5_checkpoint_screen_queue as queue  # noqa: E402
import run_sa5_checkpoint_screen_cell as runner  # noqa: E402


FREEZE = REPO / "docs/freeze/sa5_checkpoint_screen_v2.json"
RUNNER = HERE / "run_sa5_checkpoint_screen_cell.py"
QUEUE = HERE / "sa5_checkpoint_screen_queue.py"
PLAY = HERE.parent / "play_eval" / "play_rnn_car.py"


def _cell(name: str, scenario: str, cr: float, *, hard_pass: bool = True) -> dict:
    candidate = screen.candidate_by_name(name)
    metrics = {"n": 2000, "sr": 1.0 - cr, "cr": cr, "to": 0.0}
    if scenario == "narrow_range":
        metrics.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    return {
        "protocol_sha256": screen.screen_protocol()["sha256"],
        "checkpoint_name": name,
        "checkpoint_sha256": candidate["sha256"],
        "scenario": scenario,
        "cell_valid": True,
        "threshold_pass": hard_pass,
        "metrics": metrics,
    }


def _phase_a(scores: dict[str, tuple[float, float, float, float]]) -> list[dict]:
    return [
        _cell(name, scenario, cr)
        for name, values in scores.items()
        for scenario, cr in zip(screen.CORRIDOR_SCENARIOS, values, strict=True)
    ]


def test_frozen_protocol_matches_runtime():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == screen.screen_protocol()


def test_candidate_identity_and_two_phase_matrix_are_frozen():
    protocol = screen.screen_protocol()
    assert [item["name"] for item in protocol["candidates"]] == [
        "c150", "c200", "c250", "c300"
    ]
    assert protocol["phase_a"]["cells"] == 16
    assert protocol["phase_b"]["cells"] == 4
    fixed = protocol["fixed_evaluation"]
    assert fixed["seed"] == 818
    assert fixed["actuator_delay_steps"] == 1
    assert fixed["lidar_distractor_eligibility"] == "valid_return_only"
    assert fixed["scene"]["arena_size_m"] == 15.0
    assert fixed["scene"]["room_half_extent_m"] == 7.5
    assert fixed["scene"]["corridor_free_width_m"] == 4.2
    assert fixed["scene"]["corridor_wall_span_m"] == 15.0
    assert fixed["scene"]["corridor_expected_boundary_overlap_m"] == 0.5
    assert fixed["scene"]["corridor_sealed_to_boundary"] is True
    assert fixed["corridor_fixed_density"] == "4S2D"
    assert fixed["scene"]["corridor_density_mix"] == [[[3, 2], 0.5], [[4, 2], 0.5]]


def test_ranking_uses_the_worst_family_before_the_mean():
    payloads = _phase_a(
        {
            "c150": (0.01, 0.01, 0.01, 0.09),
            "c200": (0.06, 0.06, 0.06, 0.06),
            "c250": (0.07, 0.01, 0.01, 0.01),
            "c300": (0.08, 0.01, 0.01, 0.01),
        }
    )
    result = screen.rank_phase_a(payloads)
    assert result["ranked"] == ["c200", "c250", "c300", "c150"]
    assert result["top_two"] == ["c200", "c250"]


def test_hard_gate_failure_is_reported_but_does_not_erase_ranking_data():
    payloads = _phase_a(
        {
            "c150": (0.11, 0.01, 0.01, 0.01),
            "c200": (0.12, 0.12, 0.12, 0.12),
            "c250": (0.13, 0.13, 0.13, 0.13),
            "c300": (0.14, 0.14, 0.14, 0.14),
        }
    )
    payloads[0]["threshold_pass"] = False
    result = screen.rank_phase_a(payloads)
    assert result["ranked"][0] == "c150"
    row = next(row for row in result["summaries"] if row["checkpoint_name"] == "c150")
    assert row["all_corridor_hard_pass"] is False


def test_missing_or_invalid_phase_a_cell_fails_closed():
    payloads = _phase_a(
        {name: (0.05, 0.05, 0.05, 0.05) for name in ("c150", "c200", "c250", "c300")}
    )
    try:
        screen.rank_phase_a(payloads[:-1])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("missing Phase-A cell must fail closed")
    payloads[0]["cell_valid"] = False
    try:
        screen.rank_phase_a(payloads)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("invalid Phase-A cell must fail closed")


def test_phase_b_only_folds_the_selected_top_two_and_never_accepts_parent():
    phase_a = _phase_a(
        {
            "c150": (0.04, 0.04, 0.04, 0.04),
            "c200": (0.05, 0.05, 0.05, 0.05),
            "c250": (0.06, 0.06, 0.06, 0.06),
            "c300": (0.07, 0.07, 0.07, 0.07),
        }
    )
    phase_b = [
        _cell(name, scenario, 0.01)
        for name in ("c150", "c200")
        for scenario in screen.RETENTION_SCENARIOS
    ]
    verdict = screen.final_verdict(phase_a, phase_b)
    assert verdict["recommended_for_human_consideration"] == "c150"
    assert verdict["accepted_parent"] is False
    assert verdict["training_started"] is False
    assert verdict["sa6_started"] is False


def test_runner_pins_every_measurement_defining_argument():
    source = RUNNER.read_text(encoding="utf-8")
    for marker in (
        '"--vlp16_noise_mode"',
        '"--lidar-distractor-eligibility"',
        '"--long_corridor_free_width"',
        '"--long_corridor_dynamic_speed_range"',
        '"--long_corridor_static_obstacles"',
        '"--long_corridor_dynamic_obstacles"',
        '"--long_corridor_motion_mode"',
        '"--long_corridor_random_2d_kinematics"',
        '"requested_wall_span_m"',
        "verify_fixed_actuator_runtime",
        "verify_corridor_runtime",
        '"sealed_to_boundary_pass"',
        "reconcile_corridor_metrics",
        "parse_exact_outcome_counts",
    ):
        assert marker in source


def _sealed_corridor_report() -> dict:
    values = runner.stage_values()
    return {
        "requested_free_width_m": values["corridor_free_width_m"],
        "requested_length_m": values["corridor_length_m"],
        "requested_wall_span_m": values["arena_size_m"],
        "static_obstacles_per_env": values["corridor_static_obstacles"],
        "dynamic_obstacles_per_env": values["corridor_dynamic_obstacles"],
        "requested_dynamic_speed_range_m_s": values[
            "corridor_speed_range_m_s"
        ],
        "dynamic_motion_mode": "lateral",
        "random_2d_kinematics": values["random_2d_kinematics"],
        "geometry_pass": True,
        "movement_pass": True,
        "motion_mode_pass": True,
        "obstacle_mix_pass": True,
        "goal_alignment_pass": True,
        "penetration_pass": True,
        "speed_upper_bound_pass": True,
        "sealed_to_boundary_pass": True,
        "constructive_unsolvable_count": 0,
        "dynamic_motion_type_fractions": {"lateral": 1.0},
        "free_width_m_mean": values["corridor_free_width_m"],
        "length_m_mean": values["corridor_length_m"],
        "wall_span_m_mean": values["arena_size_m"],
        "wall_boundary_overlap_m_min": values[
            "corridor_expected_boundary_overlap_m"
        ],
    }


def test_runner_accepts_recorded_sealed_corridor_geometry():
    result = runner.verify_corridor_runtime(
        _sealed_corridor_report(),
        "corridor_lateral",
        runner.stage_values(),
    )
    assert result["wall_span_m"] == 15.0
    assert result["wall_boundary_overlap_m"] == 0.5
    assert result["sealed_to_boundary"] is True


def test_runner_rejects_unsealed_corridor_geometry():
    report = _sealed_corridor_report()
    report["sealed_to_boundary_pass"] = False
    report["wall_boundary_overlap_m_min"] = -2.0
    try:
        runner.verify_corridor_runtime(
            report,
            "corridor_lateral",
            runner.stage_values(),
        )
    except RuntimeError as error:
        assert "sealed_to_boundary_pass=false" in str(error)
    else:  # pragma: no cover
        raise AssertionError("unsealed corridor must fail closed")


def test_play_direct_install_carries_the_same_sealed_wall_span():
    source = PLAY.read_text(encoding="utf-8")
    assert "room_half_extent=_corridor_room_half_extent" in source
    assert "wall_span_length=_corridor_wall_span_requested" in source
    assert '"sealed_to_boundary_pass"' in source


def test_queue_is_fail_closed_and_has_no_training_or_sa6_launch_path():
    source = QUEUE.read_text(encoding="utf-8")
    assert "INCOMPLETE_NO_VERDICT" in source
    assert "measurement source drift" in source
    assert "verify_checkpoints()" in source
    assert "planned_phase_b_cells(phase_a_result[\"top_two\"])" in source
    assert "str(RUNNER)" in source
    assert "subprocess.run(command" in source
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemctl" not in source
    assert "subprocess.Popen" not in source
    assert '"sa6_started": False' in source


def test_shared_gpu_mode_keeps_a_hard_memory_floor(monkeypatch):
    outputs = iter(
        [
            type("Result", (), {
                "returncode": 0,
                "stdout": "16900, 90, 61\n",
                "stderr": "",
            })(),
            type("Result", (), {
                "returncode": 0,
                "stdout": "3214308, /home/cm/yolo/python3, 14612\n",
                "stderr": "",
            })(),
        ]
    )
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: next(outputs))
    monkeypatch.setattr(queue, "_mem_available_gib", lambda: 40.0)
    snapshot = queue.resource_snapshot(allow_shared_gpu=True)
    assert snapshot["ready"] is True
    assert snapshot["policy_mode"] == "shared_memory_guarded"
    assert snapshot["requirements"]["min_gpu_free_mib"] == 12000
    assert snapshot["blocking_apps"] == []
    assert len(snapshot["coexisting_heavy_apps"]) == 1


def test_shared_gpu_mode_refuses_less_than_twelve_gib(monkeypatch):
    outputs = iter(
        [
            type("Result", (), {
                "returncode": 0,
                "stdout": "11999, 1, 40\n",
                "stderr": "",
            })(),
            type("Result", (), {
                "returncode": 0,
                "stdout": "\n",
                "stderr": "",
            })(),
        ]
    )
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: next(outputs))
    monkeypatch.setattr(queue, "_mem_available_gib", lambda: 40.0)
    assert queue.resource_snapshot(allow_shared_gpu=True)["ready"] is False
