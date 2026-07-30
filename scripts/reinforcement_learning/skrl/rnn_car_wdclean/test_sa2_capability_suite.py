"""Tests for the SA2 capability suite and the corridor scene it depends on.

Two kinds of test:

* behavioural tests on ``run_sa2_capability_suite``, which imports cleanly
  because it never touches Isaac Lab;
* source-contract tests on ``play_rnn_car.py``, which cannot be imported here
  (Isaac Lab at module scope) and whose bugs are structural: a value threaded
  into one of two call sites, or a measured quantity compared against a
  literal instead of against the request.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

import pytest


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

suite = pytest.importorskip("run_sa2_capability_suite")

PLAY = _HERE.parent / "play_eval" / "play_rnn_car.py"
PLAY_SOURCE = PLAY.read_text(encoding="utf-8")


# --- the scene must come from the training spec, not from this file ---------


def test_scene_values_match_the_sa2_training_spec():
    from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
        STAGE_SCENE_CURRICULUM,
    )

    spec = STAGE_SCENE_CURRICULUM[2]
    assert suite.ARENA_SIZE_M == 2.0 * spec.room_half_extent == 18.0
    assert suite.NARROW_WIDTH_RANGE == spec.narrow_width_range == (1.60, 1.80)
    assert suite.NARROW_YAW_LIMIT_DEG == spec.narrow_yaw_limit_deg == 3.0
    assert suite.CORRIDOR_FREE_WIDTH_M == spec.corridor_free_width == 4.8
    assert suite.CORRIDOR_STATIC == spec.corridor_static_obstacles == 3
    assert suite.CORRIDOR_DYNAMIC == spec.corridor_dynamic_obstacles == 1
    assert suite.CORRIDOR_SPEED_RANGE == spec.corridor_speed_range == (0.18, 0.30)


def test_corridor_length_is_ten_metres_not_the_free_width():
    """4.8 m is the free width. Reading it as the length is a different scene."""
    assert suite.CORRIDOR_LENGTH_M == 10.0
    assert suite.CORRIDOR_FREE_WIDTH_M != suite.CORRIDOR_LENGTH_M


def test_scenarios_are_exactly_sa2s_families():
    assert suite.SCENARIOS == ("narrow", "lateral", "longitudinal")
    assert "random_2d" not in suite.SCENARIOS


def test_module_refuses_to_load_if_sa2_gains_random_2d(monkeypatch):
    """If the curriculum changes, this suite must stop claiming completeness."""
    source = (_HERE / "run_sa2_capability_suite.py").read_text(encoding="utf-8")
    assert "corridor_motion_weights[2] != 0.0" in source
    assert "under-test SA2" in source
    assert "corridor_density_mix is not None" in source
    assert "narrow_exact_width is not None" in source


# --- built CLI arguments ----------------------------------------------------


def _args(scenario: str) -> list[str]:
    return suite.build_scene_args(
        scenario,
        seed=818,
        actuator_args=["--enable_actuator_dr"],
        corridor_json=Path("/tmp/unused.json"),
    )


def _flag_values(args: list[str], flag: str, count: int = 1) -> list[str]:
    index = args.index(flag)
    return args[index + 1 : index + 1 + count]


@pytest.mark.parametrize("scenario", suite.SCENARIOS)
def test_every_scenario_pins_stage_and_sensor_and_arena(scenario):
    args = _args(scenario)
    assert _flag_values(args, "--stage") == ["2"]
    assert _flag_values(args, "--vlp16_noise_mode") == ["full"]
    assert _flag_values(args, "--arena_size") == ["18"]
    assert _flag_values(args, "--seed") == ["818"]


@pytest.mark.parametrize("scenario", suite.SCENARIOS)
def test_stage_is_explicit_so_checkpoint_cannot_override_it(scenario):
    """play only auto-inherits initial_stage when --stage is absent."""
    assert "--stage" in _args(scenario)


def test_narrow_uses_the_training_replay_not_the_fixed_gate5_scene():
    args = _args("narrow")
    assert "--narrow_replay_eval" in args
    assert "--narrow_gap_eval" not in args
    assert _flag_values(args, "--narrow_replay_width_range", 2) == ["1.6", "1.8"]
    assert _flag_values(args, "--narrow_replay_yaw_limit_deg") == ["3"]
    assert _flag_values(args, "--narrow_replay_segment_length") == ["9"]


def test_narrow_zeroes_the_random_wall_slots_but_not_the_barrier():
    args = _args("narrow")
    assert _flag_values(args, "--num_walls") == ["0"]
    assert _flag_values(args, "--num_static_obs") == ["0"]
    assert _flag_values(args, "--num_dynamic_obs") == ["0"]
    # The barrier is a dedicated asset pair, so it survives --num_walls 0.
    assert "--narrow_replay_eval" in args


@pytest.mark.parametrize("family", suite.CORRIDOR_FAMILIES)
def test_corridor_requests_sa2_geometry_counts_and_speed(family):
    args = _args(family)
    assert _flag_values(args, "--long_corridor_free_width") == ["4.8"]
    assert _flag_values(args, "--long_corridor_dynamic_speed_range", 2) == [
        "0.18",
        "0.3",
    ]
    assert _flag_values(args, "--long_corridor_static_obstacles") == ["3"]
    assert _flag_values(args, "--long_corridor_dynamic_obstacles") == ["1"]
    assert _flag_values(args, "--long_corridor_motion_mode") == [family]


def test_corridor_does_not_pass_deployment_defaults():
    """The SA5/deployment values must never appear in an SA2 command."""
    for family in suite.CORRIDOR_FAMILIES:
        args = " ".join(_args(family))
        assert "--long_corridor_free_width 4.0" not in args
        assert "0.3 0.6" not in args
        assert "--long_corridor_static_obstacles 4" not in args


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match="unknown SA2 scenario"):
        _args("random_2d")


# --- thresholds and margins ------------------------------------------------


def test_narrow_blocking_bars_cover_outcomes_and_traversal():
    """crossing_min is implied by direct_crossing_min, so it adds no bar."""
    t = suite.thresholds_for("narrow")
    assert t == {
        "sr_min": 0.90,
        "cr_max": 0.05,
        "to_max": 0.05,
        "crossing_min": 0.95,
        "direct_crossing_min": 0.95,
    }


def test_narrow_timeout_is_blocking():
    verdict = suite.evaluate(
        "narrow",
        {
            "n": 1200,
            "sr": 0.95,
            "cr": 0.0,
            "to": 0.50,
            "crossing_rate": 0.99,
            "direct_crossing_rate": 0.98,
        },
    )
    assert verdict["checks"]["to"]["pass"] is False
    assert verdict["threshold_pass"] is False


@pytest.mark.parametrize("family", suite.CORRIDOR_FAMILIES)
def test_corridor_thresholds_cover_outcomes_and_sample_size(family):
    t = suite.thresholds_for(family)
    assert t == {"sr_min": 0.90, "cr_max": 0.10, "to_max": 0.05}


def test_narrow_gates_outcomes_and_both_traversal_rates():
    verdict = suite.evaluate(
        "narrow",
        {
            "n": 1200,
            "sr": 0.99,
            "cr": 0.0,
            "to": 0.01,
            "crossing_rate": 0.99,
            "direct_crossing_rate": 0.98,
        },
    )
    assert set(verdict["checks"]) == {
        "sr",
        "cr",
        "to",
        "crossing_rate",
        "direct_crossing_rate",
    }
    assert verdict["threshold_pass"] is True


def test_corridor_gates_reach_collision_and_timeout():
    verdict = suite.evaluate(
        "lateral", {"n": 1200, "sr": 0.95, "cr": 0.02, "to": 0.01}
    )
    assert set(verdict["checks"]) == {"sr", "cr", "to"}
    assert verdict["threshold_pass"] is True


def test_sample_size_stays_out_of_the_margin_units():
    """An episode count and a rate cannot share a margin scale.

    Mixing them would make a 1200-episode cell score +200 and a 900-episode one
    score -100, so the Stage A ranking would order candidates by sample size
    rather than by capability.
    """
    verdict = suite.evaluate(
        "lateral", {"n": 999_999, "sr": 0.91, "cr": 0.02, "to": 0.01}
    )
    assert verdict["worst_margin"] == pytest.approx(0.01)
    assert all(
        abs(check["margin"]) <= 1.0 for check in verdict["checks"].values()
    )
    assert "episodes" not in verdict["checks"]


def test_margin_sign_matches_pass_for_lower_bounds():
    verdict = suite.evaluate(
        "narrow",
        {
            "n": 1200,
            "sr": 0.85,
            "cr": 0.0,
            "to": 0.0,
            "crossing_rate": 0.99,
            "direct_crossing_rate": 0.99,
        },
    )
    assert verdict["checks"]["sr"]["pass"] is False
    assert verdict["checks"]["sr"]["margin"] == pytest.approx(-0.05)
    assert verdict["threshold_pass"] is False


def test_margin_sign_matches_pass_for_upper_bounds():
    verdict = suite.evaluate(
        "lateral", {"n": 1200, "sr": 0.99, "cr": 0.20, "to": 0.0}
    )
    assert verdict["checks"]["cr"]["pass"] is False
    assert verdict["checks"]["cr"]["margin"] == pytest.approx(-0.10)
    assert verdict["threshold_pass"] is False


def test_worst_margin_identifies_the_binding_check():
    verdict = suite.evaluate(
        "lateral", {"n": 1200, "sr": 0.93, "cr": 0.09, "to": 0.001}
    )
    assert verdict["worst_check"] == "cr"
    assert verdict["worst_margin"] == pytest.approx(0.01)


def test_a_missing_metric_raises_rather_than_scoring_zero():
    with pytest.raises(RuntimeError, match="was not measured"):
        suite.evaluate("narrow", {"n": 1200, "sr": 0.99, "cr": 0.0})


def test_exactly_on_the_threshold_passes():
    verdict = suite.evaluate(
        "lateral", {"n": 1000, "sr": 0.90, "cr": 0.10, "to": 0.05}
    )
    assert verdict["threshold_pass"] is True
    assert verdict["worst_margin"] == pytest.approx(0.0)


def test_too_few_episodes_voids_the_cell_rather_than_failing_it():
    """Unmeasured is not the same as failed, and must not read as a bad score."""
    with pytest.raises(RuntimeError, match="void rather than failed"):
        suite.evaluate("lateral", {"n": 999, "sr": 1.0, "cr": 0.0, "to": 0.0})


def test_a_cell_at_the_episode_floor_is_measurable():
    verdict = suite.evaluate(
        "lateral", {"n": suite.MIN_EPISODES, "sr": 0.95, "cr": 0.0, "to": 0.0}
    )
    assert verdict["episodes"] == suite.MIN_EPISODES
    assert verdict["threshold_pass"] is True


def test_missing_episode_count_raises():
    with pytest.raises(RuntimeError, match="episode count was not measured"):
        suite.evaluate("lateral", {"sr": 1.0, "cr": 0.0, "to": 0.0})


@pytest.mark.parametrize("name", ["sr", "cr", "to"])
def test_non_finite_outcome_metric_raises(name):
    metrics = {"n": 1200, "sr": 0.99, "cr": 0.0, "to": 0.0}
    metrics[name] = float("nan")
    with pytest.raises(RuntimeError, match="was not measured"):
        suite.evaluate("lateral", metrics)


# --- runtime verification --------------------------------------------------


_NARROW_LAYOUT_OK = (
    "[PLAY] curriculum=warp_drive_e2e_final20_v1 stage=2 name=SA2 目標=10\n"
    "[SIM2REAL][VLP16-ablation] mode=full σ=ON bias=ON dropout=ON\n"
    "[PLAY] 隨機窄縫 replay 評測: width∈(1.6, 1.8) m, gap_y∈(-1.0, 1.0) m, "
    "barrier_x∈(-0.5, 0.5) m, start_dist=3.00m, goal_dist∈(3.0, 3.0)m, "
    "goal_dy∈(0.0, 0.0)m, segment=9.00m, yaw±3.0deg, 左右各半, "
    "room_half_extent=9.0m\n"
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "cell.log"
    path.write_text(text, encoding="utf-8")
    return path


def test_narrow_runtime_accepts_the_sa2_layout(tmp_path):
    contract = suite.verify_narrow_runtime(_write(tmp_path, _NARROW_LAYOUT_OK))
    assert contract["width_range_m_built"] == [1.6, 1.8]
    assert contract["room_half_extent_m_built"] == 9.0


def test_narrow_runtime_rejects_a_different_width(tmp_path):
    text = _NARROW_LAYOUT_OK.replace("width∈(1.6, 1.8)", "width∈(1.2, 1.4)")
    with pytest.raises(RuntimeError, match="width_lo"):
        suite.verify_narrow_runtime(_write(tmp_path, text))


def test_narrow_runtime_rejects_a_clamped_layout(tmp_path):
    text = _NARROW_LAYOUT_OK + "[NARROW-REPLAY] ⚠ gap centre clamped to room\n"
    with pytest.raises(RuntimeError, match="clamped"):
        suite.verify_narrow_runtime(_write(tmp_path, text))


def test_narrow_runtime_rejects_a_missing_layout_line(tmp_path):
    text = _NARROW_LAYOUT_OK.split("[PLAY] 隨機窄縫")[0]
    with pytest.raises(RuntimeError, match="cell is void"):
        suite.verify_narrow_runtime(_write(tmp_path, text))


def test_narrow_runtime_rejects_a_clean_lidar_run(tmp_path):
    text = _NARROW_LAYOUT_OK.replace("mode=full", "mode=ideal")
    with pytest.raises(RuntimeError, match="VLP16-ablation"):
        suite.verify_narrow_runtime(_write(tmp_path, text))


def test_narrow_runtime_rejects_the_wrong_stage(tmp_path):
    text = _NARROW_LAYOUT_OK.replace("stage=2 ", "stage=1 ")
    with pytest.raises(RuntimeError, match="stage=2"):
        suite.verify_narrow_runtime(_write(tmp_path, text))


_CORRIDOR_LOG_OK = (
    "[PLAY] curriculum=warp_drive_e2e_final20_v1 stage=2 name=SA2 目標=10\n"
    "[SIM2REAL][VLP16-ablation] mode=full σ=ON\n"
)


def _corridor_report(**overrides) -> dict:
    report = {
        "requested_free_width_m": 4.8,
        "requested_length_m": 10.0,
        "requested_dynamic_speed_range_m_s": [0.18, 0.30],
        "free_width_m_mean": 4.8,
        "length_m_mean": 10.0,
        "observed_max_dynamic_speed_m_s": 0.28,
        "geometry_pass": True,
        "movement_pass": True,
        "motion_mode_pass": True,
        "obstacle_mix_pass": True,
        "goal_alignment_pass": True,
        "penetration_pass": True,
        "speed_upper_bound_pass": True,
        "constructive_unsolvable_count": 0,
        "static_obstacles_per_env": 3.0,
        "dynamic_obstacles_per_env": 1.0,
        "dynamic_motion_type_fractions": {
            "lateral": 1.0,
            "longitudinal": 0.0,
            "random_2d": 0.0,
        },
    }
    report.update(overrides)
    return report


def test_corridor_runtime_accepts_the_sa2_scene(tmp_path):
    contract = suite.verify_corridor_runtime(
        _write(tmp_path, _CORRIDOR_LOG_OK), _corridor_report(), "lateral"
    )
    assert contract["free_width_m_measured"] == 4.8
    assert contract["length_m_measured"] == 10.0
    assert contract["motion_family"] == "lateral"


def test_corridor_runtime_rejects_the_deployment_width(tmp_path):
    with pytest.raises(RuntimeError, match="requested free width"):
        suite.verify_corridor_runtime(
            _write(tmp_path, _CORRIDOR_LOG_OK),
            _corridor_report(requested_free_width_m=4.0),
            "lateral",
        )


def test_corridor_runtime_rejects_the_deployment_speed(tmp_path):
    with pytest.raises(RuntimeError, match="requested speed"):
        suite.verify_corridor_runtime(
            _write(tmp_path, _CORRIDOR_LOG_OK),
            _corridor_report(requested_dynamic_speed_range_m_s=[0.30, 0.60]),
            "lateral",
        )


def test_corridor_runtime_rejects_a_stale_report_without_the_new_fields(tmp_path):
    stale = _corridor_report()
    del stale["requested_free_width_m"]
    with pytest.raises(RuntimeError, match="predates the parameterised"):
        suite.verify_corridor_runtime(
            _write(tmp_path, _CORRIDOR_LOG_OK), stale, "lateral"
        )


@pytest.mark.parametrize(
    "flag",
    [
        "geometry_pass",
        "movement_pass",
        "motion_mode_pass",
        "obstacle_mix_pass",
        "goal_alignment_pass",
        "penetration_pass",
        "speed_upper_bound_pass",
    ],
)
def test_corridor_runtime_rejects_each_structural_failure(tmp_path, flag):
    with pytest.raises(RuntimeError, match=flag):
        suite.verify_corridor_runtime(
            _write(tmp_path, _CORRIDOR_LOG_OK),
            _corridor_report(**{flag: False}),
            "lateral",
        )


def test_corridor_runtime_rejects_wrong_obstacle_counts(tmp_path):
    with pytest.raises(RuntimeError, match="static per env"):
        suite.verify_corridor_runtime(
            _write(tmp_path, _CORRIDOR_LOG_OK),
            _corridor_report(static_obstacles_per_env=4.0),
            "lateral",
        )


def test_corridor_runtime_rejects_an_unsolvable_construction(tmp_path):
    with pytest.raises(RuntimeError, match="constructive_unsolvable_count"):
        suite.verify_corridor_runtime(
            _write(tmp_path, _CORRIDOR_LOG_OK),
            _corridor_report(constructive_unsolvable_count=3),
            "lateral",
        )


def test_corridor_runtime_rejects_a_family_that_did_not_run(tmp_path):
    """Asking for longitudinal but getting lateral must not silently pass."""
    with pytest.raises(RuntimeError, match="longitudinal motion fraction"):
        suite.verify_corridor_runtime(
            _write(tmp_path, _CORRIDOR_LOG_OK),
            _corridor_report(),
            "longitudinal",
        )


def test_corridor_runtime_rejects_missing_motion_ledger(tmp_path):
    report = _corridor_report()
    del report["dynamic_motion_type_fractions"]
    with pytest.raises(RuntimeError, match="missing requested family"):
        suite.verify_corridor_runtime(
            _write(tmp_path, _CORRIDOR_LOG_OK), report, "lateral"
        )


def test_corridor_metrics_use_precise_json_after_ledger_check():
    summary = {"n": 1200, "sr": 0.923, "cr": 0.071, "to": 0.006}
    report = {
        "episodes": 1200,
        "success_rate": 0.9234,
        "collision_rate": 0.0708,
        "timeout_rate": 0.0058,
    }
    assert suite.reconcile_corridor_metrics(summary, report) == {
        "n": 1200,
        "sr": 0.9234,
        "cr": 0.0708,
        "to": 0.0058,
    }


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("episodes", 1199, "episode ledgers disagree"),
        ("success_rate", 0.90, "sr ledgers disagree"),
        ("collision_rate", 0.05, "cr ledgers disagree"),
        ("timeout_rate", 0.02, "to ledgers disagree"),
    ],
)
def test_corridor_metrics_reject_ledger_disagreement(field, value, match):
    summary = {"n": 1200, "sr": 0.923, "cr": 0.071, "to": 0.006}
    report = {
        "episodes": 1200,
        "success_rate": 0.9234,
        "collision_rate": 0.0708,
        "timeout_rate": 0.0058,
    }
    report[field] = value
    with pytest.raises(RuntimeError, match=match):
        suite.reconcile_corridor_metrics(summary, report)


# --- narrow metric parsing -------------------------------------------------


def test_narrow_metrics_are_read_from_the_replay_banner(tmp_path):
    text = (
        "[NARROW-REPLAY-METRICS] episodes=2304 crossed=2290 "
        "direct_crossed=2250 crossing_rate=0.993924 "
        "direct_crossing_rate=0.976562 path_length_ratio_p95=1.180000\n"
    )
    values = suite.parse_narrow_replay_metrics(_write(tmp_path, text))
    assert values["episodes"] == 2304
    assert values["direct_crossing_rate"] == pytest.approx(0.976562)


def test_gap_banner_is_not_accepted_for_the_replay_scene(tmp_path):
    """NARROW-GAP-METRICS is the fixed Gate5 scene, a different distribution."""
    text = "[NARROW-GAP-METRICS] direct_crossing_rate=1.000000\n"
    with pytest.raises(RuntimeError, match="no narrow replay metrics"):
        suite.parse_narrow_replay_metrics(_write(tmp_path, text))


def test_incomplete_narrow_metrics_raise(tmp_path):
    text = "[NARROW-REPLAY-METRICS] episodes=100 crossing_rate=0.9\n"
    with pytest.raises(RuntimeError, match="direct_crossing_rate"):
        suite.parse_narrow_replay_metrics(_write(tmp_path, text))


# --- play_rnn_car.py source contracts --------------------------------------


def test_play_exposes_the_two_corridor_scene_flags():
    assert '"--long_corridor_free_width"' in PLAY_SOURCE
    assert '"--long_corridor_dynamic_speed_range"' in PLAY_SOURCE


def test_play_rejects_invalid_corridor_geometry_and_speed():
    assert "--long_corridor_free_width must be positive" in PLAY_SOURCE
    assert (
        "--long_corridor_dynamic_speed_range must be positive and ordered"
        in PLAY_SOURCE
    )


def test_new_play_flags_default_to_the_historical_deployment_values():
    """A new flag must not change what any existing command does."""
    width = re.search(
        r'"--long_corridor_free_width", type=float, default=([\d.]+)',
        PLAY_SOURCE,
    )
    speed = re.search(
        r'"--long_corridor_dynamic_speed_range", type=float, nargs=2,\s*'
        r"default=\(([\d.]+), ([\d.]+)\)",
        PLAY_SOURCE,
    )
    assert width and float(width.group(1)) == 4.0
    assert speed and [float(v) for v in speed.groups()] == [0.30, 0.60]


def test_both_corridor_call_sites_are_parameterised():
    """configure_* and setup_* write the same scene from two code paths."""
    assert "free_width=_corridor_free_width" in PLAY_SOURCE
    assert "free_width=_install_free_width" in PLAY_SOURCE
    assert "dynamic_speed_range=_corridor_speed_range" in PLAY_SOURCE
    assert "dynamic_speed_min=_install_speed_range[0]" in PLAY_SOURCE
    assert "dynamic_speed_max=_install_speed_range[1]" in PLAY_SOURCE


def test_no_hardcoded_corridor_scene_values_remain():
    for literal in (
        "free_width=4.0,",
        "dynamic_speed_range=(0.30, 0.60)",
        "dynamic_speed_min=0.30",
        "dynamic_speed_max=0.60",
    ):
        assert literal not in PLAY_SOURCE, f"still hardcoded: {literal}"


def test_geometry_pass_compares_against_the_request_not_a_literal():
    block = PLAY_SOURCE[PLAY_SOURCE.index('"geometry_pass": bool(') :]
    block = block[: block.index('"movement_pass"')]
    assert "args_cli.long_corridor_free_width" in block
    assert "_corridor_inner_width, 4.0" not in block


def test_corridor_report_records_what_was_requested():
    for key in (
        '"requested_free_width_m"',
        '"requested_length_m"',
        '"requested_dynamic_speed_range_m_s"',
        '"observed_max_dynamic_speed_m_s"',
        '"speed_upper_bound_pass"',
    ):
        assert key in PLAY_SOURCE


def test_speed_bound_is_not_folded_into_the_historical_gate_pass():
    """Adding it would change the verdict of every past deployment-gate run."""
    block = PLAY_SOURCE[PLAY_SOURCE.index('_corridor_report["gate_pass"] = bool(') :]
    block = block[: block.index("print(")]
    assert "speed_upper_bound_pass" not in block


def test_corridor_banner_prints_actual_values_not_a_constant():
    """A contract check that reads a constant banner verifies nothing."""
    assert "free_width=4.00m length=10.00m " not in PLAY_SOURCE
    assert "patrol=[0.30,0.60]m/s" not in PLAY_SOURCE
    assert "free_width={_corridor_free_width:.2f}m" in PLAY_SOURCE


def test_play_still_compiles():
    import py_compile

    py_compile.compile(str(PLAY), doraise=True)


# --- cell shape ------------------------------------------------------------


def test_delay_ms_mapping_matches_the_control_period():
    assert suite.DELAY_MS == {0: 0, 1: 200, 2: 400}


def test_seed_and_delay_are_required_so_cells_cannot_be_pooled():
    source = (_HERE / "run_sa2_capability_suite.py").read_text(encoding="utf-8")

    # Restrict to the parser section: the bare flag strings also appear in
    # build_scene_args, where "required" would never be found.
    parser_section = source[
        source.index("argparse.ArgumentParser(") : source.index(
            "args = parser.parse_args"
        )
    ]

    def parser_block(flag: str) -> str:
        start = parser_section.index(f'"{flag}"')
        return parser_section[start : start + 300]

    assert "required=True" in parser_block("--seed")
    assert "required=True" in parser_block("--actuator-delay-steps")
    assert "required=True" in parser_block("--scenario")
    # A plural form would reintroduce a code path that pools seeds.
    assert '"--seeds"' not in source


def test_formal_actuator_profile_is_pinned_to_delay_only():
    source = (_HERE / "run_sa2_capability_suite.py").read_text(encoding="utf-8")
    assert suite.ACTUATOR_PROFILE == "sa1_delay_only"
    parser_section = source[
        source.index("argparse.ArgumentParser(") : source.index(
            "args = parser.parse_args"
        )
    ]
    block = parser_section[
        parser_section.index('"--actuator-profile"') :
        parser_section.index('"--num-envs"')
    ]
    assert "choices=(ACTUATOR_PROFILE,)" in block
    assert "default=ACTUATOR_PROFILE" in block


# --- queue aggregation: path matching must survive symlinks -----------------


def test_checkpoint_fold_matches_through_a_symlink(tmp_path):
    """The run directory is a symlink, so string path comparison folds nothing.

    The queue passes ``IsaacLab/logs/rnn_car/<run>/ckpt.pt``; the runner calls
    ``.resolve()`` and records ``/home/aa/logs/rnn_car/<run>/ckpt.pt``. Comparing
    the two as strings matched zero cells, so a clean 18/18 run was summarised as
    ``cells=0, worst_margin=None`` for every checkpoint.
    """
    queue = pytest.importorskip("sa2_capability_queue")

    real = tmp_path / "real"
    real.mkdir()
    ckpt = real / "checkpoint_6400.pt"
    ckpt.write_bytes(b"x")
    link = tmp_path / "via_symlink"
    link.symlink_to(real)

    payloads = [
        {
            # as the runner records it: resolved through the symlink
            "checkpoint": str(ckpt.resolve()),
            "scenario": scenario,
            "seed": 818,
            "delay_steps": 1,
            "valid": True,
            "threshold_pass": True,
            "worst_margin": 0.05,
            "metrics": {"cr": 0.0, "sr": 1.0, "to": 0.0, "n": 2000},
        }
        for scenario in queue.SCENARIOS
    ]

    # as the queue holds it: the un-resolved path through the symlink
    folded = queue.summarize_checkpoint(
        link / "checkpoint_6400.pt", payloads, expected_cells=len(queue.SCENARIOS)
    )
    assert folded["cells"] == len(queue.SCENARIOS)
    assert folded["valid_cells"] == len(queue.SCENARIOS)
    assert folded["worst_margin"] == pytest.approx(0.05)
    assert folded["all_cells_pass"] is True


def test_checkpoint_fold_raises_when_nothing_matches(tmp_path):
    """Zero matches must be loud: silently folding to None reads as 'no data'."""
    queue = pytest.importorskip("sa2_capability_queue")
    other = tmp_path / "other.pt"
    other.write_bytes(b"y")
    wanted = tmp_path / "wanted.pt"
    wanted.write_bytes(b"z")
    payloads = [
        {
            "checkpoint": str(other),
            "scenario": "narrow",
            "seed": 818,
            "delay_steps": 1,
            "valid": True,
            "threshold_pass": True,
            "worst_margin": 0.05,
            "metrics": {"cr": 0.0, "sr": 1.0, "to": 0.0, "n": 2000},
        }
    ]
    with pytest.raises(RuntimeError, match="no cells matched"):
        queue.summarize_checkpoint(wanted, payloads, expected_cells=1)


# --- Stage A screen: ties expand, unsafe cuts do not pass ------------------


def _summary(name, worst, per=None):
    per = per or {s: worst for s in ("narrow", "lateral", "longitudinal")}
    return {
        "checkpoint": f"/tmp/{name}.pt",
        "cells": 3,
        "valid_cells": 3,
        "invalid_cells": 0,
        "all_cells_pass": True,
        "worst_margin": worst,
        "per_scenario": {k: {"worst_margin": v} for k, v in per.items()},
    }


def _payload(scenario, checks):
    return {"scenario": scenario, "checks": checks}


def test_a_tie_at_the_cut_carries_every_tied_candidate():
    """Truncating a tie to CARRY_FORWARD picks by sort order, not by merit."""
    queue = pytest.importorskip("sa2_capability_queue")
    summaries = [
        _summary("c50", 0.05), _summary("c100", 0.05),
        _summary("c150", 0.05), _summary("c200", 0.05),
        _summary("c250", 0.038), _summary("c300", 0.0327),
    ]
    screen = queue.rank_candidates(summaries, [])
    assert screen["cut_is_safe"] is False
    assert screen["tie_expanded"] is True
    assert [Path(p).stem for p in screen["carried_forward"]] == [
        "c50", "c100", "c150", "c200"
    ]


def test_a_safe_cut_carries_exactly_the_slots():
    queue = pytest.importorskip("sa2_capability_queue")
    summaries = [
        _summary("a", 0.09), _summary("b", 0.08),
        _summary("c", 0.04), _summary("d", 0.03),
    ]
    screen = queue.rank_candidates(summaries, [])
    assert screen["cut_is_safe"] is True
    assert screen["tie_expanded"] is False
    assert len(screen["carried_forward"]) == queue.CARRY_FORWARD


def test_tie_boundary_is_strictly_inside_tie_margin():
    """A candidate exactly TIE_MARGIN below the cut is separable, not tied."""
    queue = pytest.importorskip("sa2_capability_queue")
    m = queue.TIE_MARGIN
    summaries = [
        _summary("a", 0.09), _summary("b", 0.09),
        _summary("c", 0.09 - m), _summary("d", 0.09 - m - 0.01),
    ]
    screen = queue.rank_candidates(summaries, [])
    assert screen["cut_is_safe"] is True
    assert len(screen["carried_forward"]) == queue.CARRY_FORWARD


def test_saturated_checks_are_named_and_stay_blocking():
    queue = pytest.importorskip("sa2_capability_queue")
    payloads = [
        _payload("lateral", {
            "to": {"margin": 0.05, "value": 0.0},          # same for both
            "cr": {"margin": 0.077, "value": 0.023},       # varies
        }),
        _payload("lateral", {
            "to": {"margin": 0.05, "value": 0.0},
            "cr": {"margin": 0.063, "value": 0.037},
        }),
    ]
    report = queue.saturation_report(payloads)
    assert "lateral/to" in report["saturated_checks"]
    assert "lateral/cr" in report["discriminating_checks"]
    assert report["saturated_still_blocking"] is True
    # A saturated check is still a threshold in the runner, not a diagnostic.
    assert "to_max" in suite.thresholds_for("lateral")


def test_screen_states_that_margin_is_not_a_quality_score():
    queue = pytest.importorskip("sa2_capability_queue")
    screen = queue.rank_candidates([_summary("a", 0.05), _summary("b", 0.05)], [])
    assert "not a quality score" in screen["ranking_note"]


def test_exit_gate_requires_a_safe_cut():
    """An unsafe cut must not exit 0: exit 0 is what invites an auto Stage B."""
    source = (_HERE / "sa2_capability_queue.py").read_text(encoding="utf-8")
    block = source[source.index('ok = bool(\n            summary["source_fingerprint_stable"]'):]
    block = block[: block.index(")")]
    assert 'screen["cut_is_safe"]' in block


def test_stage_b_is_never_auto_started():
    source = (_HERE / "sa2_capability_queue.py").read_text(encoding="utf-8")
    assert "Stage B is NEVER started by this command" in source
    # No code path may invoke a Stage B run from inside a Stage A run.
    stage_a = source[source.index('if args.stage == "A":'):]
    stage_a = stage_a[: stage_a.index("    else:")]
    assert "verdict_cells" not in stage_a
