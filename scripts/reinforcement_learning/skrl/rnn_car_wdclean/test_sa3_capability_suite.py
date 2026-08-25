"""run_sa3_capability_suite.py 與 sa3_phase_a_queue.py 的契約測試。

重點不是「程式跑得動」，而是「跑的是不是 SA3 的題目」，以及排名規則在看到
資料之前就已經固定。
"""

import json
from pathlib import Path

import pytest

import run_sa3_capability_suite as suite
import sa3_phase_a_queue as queue
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
)
import sa3_sa8_acceptance_contract as contract


SPEC3 = STAGE_SCENE_CURRICULUM[3]


# --------------------------------------------------------------------------
# 場景常數必須來自 spec，不得在 runner 內重述
# --------------------------------------------------------------------------
def test_stage_is_three():
    assert suite.SA3_STAGE == 3


def test_arena_matches_stage_three_room():
    assert suite.ARENA_SIZE_M == pytest.approx(2.0 * SPEC3.room_half_extent)
    assert suite.ARENA_SIZE_M == pytest.approx(17.0)


def test_narrow_geometry_matches_spec():
    assert suite.NARROW_WIDTH_RANGE == SPEC3.narrow_width_range == (1.50, 1.70)
    assert suite.NARROW_YAW_LIMIT_DEG == SPEC3.narrow_yaw_limit_deg == 4.0


def test_corridor_geometry_matches_spec():
    assert suite.CORRIDOR_FREE_WIDTH_M == SPEC3.corridor_free_width == 4.6
    assert suite.CORRIDOR_STATIC == SPEC3.corridor_static_obstacles == 3
    assert suite.CORRIDOR_DYNAMIC == SPEC3.corridor_dynamic_obstacles == 1
    assert suite.CORRIDOR_SPEED_RANGE == SPEC3.corridor_speed_range == (0.20, 0.35)


def test_corridor_length_is_the_fixed_ten_metres():
    assert suite.CORRIDOR_LENGTH_M == pytest.approx(10.0)


def test_runner_does_not_restate_sa2_geometry():
    """避免複製 SA2 runner 時漏改常數。"""
    source = Path(suite.__file__).read_text()
    for sa2_value in ("18.0", "1.60", "1.80", "4.8", "0.18", "0.30"):
        assert f"= {sa2_value}" not in source


def test_thresholds_come_from_the_frozen_contract():
    assert suite.THRESHOLDS["lateral"]["sr_min"] == contract.CORRIDOR_THRESHOLDS.sr_min
    assert suite.THRESHOLDS["lateral"]["cr_max"] == contract.CORRIDOR_THRESHOLDS.cr_max
    assert suite.THRESHOLDS["narrow"]["cr_max"] == contract.NARROW_THRESHOLDS.cr_max
    assert suite.MIN_EPISODES == contract.CORRIDOR_THRESHOLDS.episodes_min == 1000


# --------------------------------------------------------------------------
# spec 漂移守門
# --------------------------------------------------------------------------
def test_module_guards_reject_random_2d(monkeypatch):
    assert SPEC3.corridor_motion_weights[2] == 0.0, (
        "SA3 若開始抽 random_2d，lateral+longitudinal 就不再是完整場景集，"
        "本套件會靜默少測一種題型"
    )


def test_module_guards_reject_density_mix():
    assert SPEC3.corridor_density_mix is None


def test_module_guards_reject_exact_width_rung():
    assert SPEC3.narrow_exact_width is None


def test_corridor_families_are_the_complete_set():
    assert suite.CORRIDOR_FAMILIES == ("lateral", "longitudinal")


# --------------------------------------------------------------------------
# CLI 參數
# --------------------------------------------------------------------------
def _args(scenario, seed=818):
    return suite.build_scene_args(
        scenario, seed=seed, actuator_args=["--A"], corridor_json=Path("/tmp/c.json")
    )


def test_stage_is_passed_explicitly():
    """play 只在 --stage 不在 argv 時才從 checkpoint 繼承，跨血緣比較必須明寫。"""
    args = _args("lateral")
    assert args[args.index("--stage") + 1] == "3"


def test_vlp16_noise_mode_is_pinned():
    args = _args("lateral")
    assert args[args.index("--vlp16_noise_mode") + 1] == "full"


def test_arena_size_is_seventeen():
    args = _args("lateral")
    assert args[args.index("--arena_size") + 1] == "17"


@pytest.mark.parametrize("family", ["lateral", "longitudinal"])
def test_corridor_args_carry_sa3_values(family):
    args = _args(family)
    assert args[args.index("--long_corridor_free_width") + 1] == "4.6"
    idx = args.index("--long_corridor_dynamic_speed_range")
    assert args[idx + 1 : idx + 3] == ["0.2", "0.35"]
    assert args[args.index("--long_corridor_static_obstacles") + 1] == "3"
    assert args[args.index("--long_corridor_dynamic_obstacles") + 1] == "1"
    assert args[args.index("--long_corridor_motion_mode") + 1] == family


def test_narrow_args_carry_sa3_values():
    args = _args("narrow")
    idx = args.index("--narrow_replay_width_range")
    assert args[idx + 1 : idx + 3] == ["1.5", "1.7"]
    assert args[args.index("--narrow_replay_yaw_limit_deg") + 1] == "4"


def test_narrow_zeroes_the_random_wall_slots():
    args = _args("narrow")
    for flag in ("--num_static_obs", "--num_dynamic_obs", "--num_walls"):
        assert args[args.index(flag) + 1] == "0"


def test_seed_is_a_single_value():
    assert _args("lateral", seed=818)[
        _args("lateral", seed=818).index("--seed") + 1
    ] == "818"


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        _args("corridor_random2d")


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------
def _metrics(n=1200, sr=0.95, cr=0.04, to=0.0):
    return {"n": n, "sr": sr, "cr": cr, "to": to}


def test_passing_corridor_cell():
    v = suite.evaluate("lateral", _metrics())
    assert v["threshold_pass"] is True
    assert v["worst_margin"] > 0


def test_cr_above_bound_fails():
    v = suite.evaluate("lateral", _metrics(cr=0.12, sr=0.88))
    assert v["threshold_pass"] is False
    assert v["worst_margin"] < 0


def test_too_few_episodes_is_void_not_failed():
    with pytest.raises(RuntimeError, match="void rather than failed"):
        suite.evaluate("lateral", _metrics(n=500))


def test_episode_count_never_enters_worst_margin():
    """計數與比率不得共用同一個 margin 尺度。"""
    v = suite.evaluate("lateral", _metrics(n=1200))
    assert "episodes" not in v["checks"]
    assert set(v["checks"]) == {"sr", "cr", "to"}


def test_missing_metric_raises_rather_than_defaulting():
    m = _metrics()
    del m["cr"]
    with pytest.raises(RuntimeError, match="was not measured"):
        suite.evaluate("lateral", m)


def test_cell_id_is_unique_per_axis():
    ids = {
        suite.make_cell_id(Path("/x/checkpoint_6400.pt"), s, d, seed)
        for s in ("lateral", "longitudinal")
        for d in (0, 1, 2)
        for seed in (818, 515)
    }
    assert len(ids) == 12


# --------------------------------------------------------------------------
# Phase A 佇列：規模與固定條件
# --------------------------------------------------------------------------
def test_phase_a_is_twelve_cells():
    cells = queue.build_cells(queue.CANDIDATES)
    assert len(cells) == 12
    assert len({c["checkpoint"] for c in cells}) == 6


def test_phase_a_fixes_seed_and_delay():
    assert queue.SCREEN_SEED == 818
    assert queue.SCREEN_DELAY_STEPS == 1
    for cell in queue.build_cells(queue.CANDIDATES):
        assert cell["seed"] == 818
        assert cell["delay_steps"] == 1


def test_phase_a_candidates_are_the_six_sa3_checkpoints():
    assert [Path(c).name for c in queue.CANDIDATES] == [
        "checkpoint_6400.pt",
        "checkpoint_12800.pt",
        "checkpoint_19200.pt",
        "checkpoint_25600.pt",
        "checkpoint_32000.pt",
        "checkpoint_38400.pt",
    ]


def test_phase_a_only_runs_corridor_families():
    scenarios = {c["scenario"] for c in queue.build_cells(queue.CANDIDATES)}
    assert scenarios == {"lateral", "longitudinal"}


# --------------------------------------------------------------------------
# 排名規則：資料到手之前就固定
# --------------------------------------------------------------------------
def _summary(name, cr_lat, cr_lon, passed=True, valid=True):
    return {
        "checkpoint": f"/x/{name}",
        "name": name,
        "cr_lateral": cr_lat,
        "cr_longitudinal": cr_lon,
        "all_pass": passed,
        "all_valid": valid,
    }


def test_ranking_key_is_worst_direction_first():
    """『最平衡』= 最差方向最好，不是兩方向差距最小。"""
    balanced_but_bad = _summary("a", 0.090, 0.090)   # |Δ|=0    max=0.090
    lopsided_but_good = _summary("b", 0.020, 0.050)  # |Δ|=0.03 max=0.050
    ranked = queue.rank([balanced_but_bad, lopsided_but_good])
    assert ranked[0]["name"] == "b"


def test_mean_cr_breaks_ties_on_worst_direction():
    a = _summary("a", 0.050, 0.050)  # max 0.050, mean 0.050
    b = _summary("b", 0.050, 0.020)  # max 0.050, mean 0.035
    assert queue.rank([a, b])[0]["name"] == "b"


def test_failing_checkpoint_is_excluded_from_ranking():
    good = _summary("good", 0.04, 0.04)
    bad = _summary("bad", 0.01, 0.01, passed=False)
    ranked = queue.rank([good, bad])
    assert [r["name"] for r in ranked] == ["good"]


def test_invalid_checkpoint_is_excluded_from_ranking():
    good = _summary("good", 0.04, 0.04)
    void = _summary("void", 0.01, 0.01, valid=False)
    assert [r["name"] for r in queue.rank([good, void])] == ["good"]


def test_tie_band_groups_close_candidates():
    a = _summary("a", 0.0400, 0.0400)
    b = _summary("b", 0.0430, 0.0430)   # 差 0.003 < 0.005
    c = _summary("c", 0.0600, 0.0600)   # 差 0.020 > 0.005
    tied = queue.tie_group(queue.rank([a, b, c]))
    assert {t["name"] for t in tied} == {"a", "b"}


def test_tie_band_prefers_final_checkpoint_when_inside():
    """c300 在 tie band 內時優先，避免無謂地選中間 checkpoint。"""
    mid = _summary("checkpoint_19200.pt", 0.0400, 0.0400)
    final = _summary("checkpoint_38400.pt", 0.0430, 0.0430)
    chosen = queue.select(queue.rank([mid, final]))
    assert chosen["name"] == "checkpoint_38400.pt"


def test_final_checkpoint_is_not_preferred_outside_the_band():
    mid = _summary("checkpoint_19200.pt", 0.0400, 0.0400)
    final = _summary("checkpoint_38400.pt", 0.0600, 0.0600)
    assert queue.select(queue.rank([mid, final]))["name"] == "checkpoint_19200.pt"


def test_tie_margin_is_the_pre_registered_value():
    assert queue.TIE_MARGIN == 0.005


def test_select_returns_none_when_nothing_qualifies():
    assert queue.select([]) is None


# --------------------------------------------------------------------------
# 不得自動啟動任何後續工作
# --------------------------------------------------------------------------
def test_queue_never_launches_sa4():
    source = Path(queue.__file__).read_text()
    for forbidden in ("train_rnn_car_wdclip", "systemd-run", "systemctl", "sa4"):
        assert forbidden not in source.lower()


def test_queue_never_chains_to_phase_b():
    source = Path(queue.__file__).read_text()
    assert "phase_b" not in source.lower() or "never" in source.lower()


def test_runner_and_queue_do_not_kill_processes():
    for module in (suite, queue):
        source = Path(module.__file__).read_text()
        for forbidden in ("pkill", "killpg", "SIGKILL"):
            assert forbidden not in source
