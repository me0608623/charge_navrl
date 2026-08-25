"""SA3 Phase B/C 畢業 Gate 的契約測試（CPU only）。"""

import ast
from pathlib import Path

import pytest

import run_sa3_gate_bc as runner
import sa3_gate_bc_queue as queue
import sa3_sa8_acceptance_contract as contract
from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    STAGE_SCENE_CURRICULUM,
)


def _args(scenario, stage=3, seed=818):
    return runner.build_scene_args(
        scenario, geometry_stage=stage, seed=seed, actuator_args=["--A"],
        corridor_json=Path("/tmp/c.json"),
    )


# --------------------------------------------------------------------------
# 不得碰 Phase A 產物
# --------------------------------------------------------------------------
def _imported_names(module):
    """真正被 import 的模組名 —— docstring 提到名字不算 import。"""
    tree = ast.parse(Path(module.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_does_not_import_phase_a_modules():
    for module in (runner, queue):
        names = _imported_names(module)
        assert "run_sa3_capability_suite" not in names
        assert "sa3_phase_a_queue" not in names


def test_does_not_write_into_phase_a_output():
    """Phase A 目錄名只准出現在說明文字裡，不得出現在任何字串常值。"""
    for module in (runner, queue):
        tree = ast.parse(Path(module.__file__).read_text())
        # clean=False：get_docstring 預設會 dedent，與原始 Constant 值不相等
        docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                      if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
        literals = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value not in docstrings
        ]
        assert not [s for s in literals if "sa3_phase_a" in s]


# --------------------------------------------------------------------------
# 幾何一律來自 spec，不得重述
# --------------------------------------------------------------------------
def test_scene_values_match_stage_three_spec():
    v = runner.scene_values(3)
    spec = STAGE_SCENE_CURRICULUM[3]
    assert v["arena_size_m"] == pytest.approx(17.0)
    assert v["narrow_width_range_m"] == spec.narrow_width_range == (1.50, 1.70)
    assert v["narrow_yaw_limit_deg"] == 4.0
    assert v["corridor_free_width_m"] == 4.6
    assert v["corridor_static"] == 3 and v["corridor_dynamic"] == 1
    assert v["corridor_speed_range_m_s"] == (0.20, 0.35)


def test_scene_values_match_stage_two_spec():
    v = runner.scene_values(2)
    assert v["arena_size_m"] == pytest.approx(18.0)
    assert v["corridor_free_width_m"] == 4.8
    assert v["corridor_speed_range_m_s"] == (0.18, 0.30)


def test_unsupported_geometry_stage_raises():
    with pytest.raises(ValueError):
        runner.scene_values(5)


def test_runner_restates_no_geometry_constant():
    source = Path(runner.__file__).read_text()
    for literal in ("17.0", "18.0", "4.6", "4.8", "1.50", "1.70", "0.35",
                    "8.5", "9.0"):
        assert f"= {literal}" not in source


# --------------------------------------------------------------------------
# 門檻一律來自 frozen contract
# --------------------------------------------------------------------------
def test_nav_clean_thresholds_from_contract():
    t = runner.thresholds_for("nav_clean", 3)
    assert t["sr_min"] == contract.NAV_CLEAN_THRESHOLDS.sr_min == 0.98
    assert t["cr_max"] == contract.NAV_CLEAN_THRESHOLDS.cr_max == 0.015


def test_native_thresholds_are_stage_specific():
    t = runner.thresholds_for("nav_native", 3)
    assert t["sr_min"] == contract.NATIVE_THRESHOLDS[3].sr_min == 0.94
    assert t["cr_max"] == contract.NATIVE_THRESHOLDS[3].cr_max == 0.05


def test_native_scenario_on_stage_without_thresholds_raises():
    """SA2 幾何沒有 native 門檻 —— 必須吵，不得默默套別的階段。"""
    with pytest.raises(ValueError):
        runner.thresholds_for("nav_native", 2)


def test_narrow_thresholds_include_crossing_bars():
    t = runner.thresholds_for("narrow_range", 3)
    assert t["crossing_min"] == 0.95 and t["direct_crossing_min"] == 0.95


def test_corridor_thresholds_from_contract():
    for scenario in runner.CORRIDOR_SCENARIOS:
        t = runner.thresholds_for(scenario, 3)
        assert t["sr_min"] == 0.90 and t["cr_max"] == 0.10 and t["to_max"] == 0.05


def test_min_episodes_is_one_thousand():
    assert runner.MIN_EPISODES == 1000


# --------------------------------------------------------------------------
# 場景組裝
# --------------------------------------------------------------------------
def test_nav_native_passes_no_scene_override():
    """play 的 STAGE_PARAMETER=True 會載入訓練場景；傳計數等於把它換掉。"""
    args = _args("nav_native")
    for forbidden in ("--num_static_obs", "--num_dynamic_obs", "--num_walls"):
        assert forbidden not in args
    assert args[args.index("--stage") + 1] == "3"


def test_nav_clean_zeroes_everything():
    args = _args("nav_clean")
    for flag in ("--num_static_obs", "--num_dynamic_obs", "--num_walls"):
        assert args[args.index(flag) + 1] == "0"


def test_native_crossing_uses_the_stage_crossing_behaviour():
    args = _args("native_crossing")
    assert args[args.index("--obstacle_behavior") + 1] == "corridor_crossing"
    for forbidden in ("--num_static_obs", "--num_dynamic_obs", "--num_walls"):
        assert forbidden not in args


def test_native_crossing_must_not_use_the_near_wall_probe():
    """near_wall_crossing_eval 會把所有 *collision* termination 設為 None
    （near_wall_crossing_eval.py:49-53），CR 因此結構上恆為 0、TO 吸收了本該
    是碰撞的回合。把 contract 的 SR/CR/TO 門檻套上去等於量錯東西。"""
    for scenario in runner.SCENARIOS:
        assert "--near_wall_crossing_eval" not in _args(scenario)
    # 註解可以提到它（說明為何不用）；只有字串常值算「真的傳了」。
    tree = ast.parse(Path(runner.__file__).read_text())
    docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
    literals = [n.value for n in ast.walk(tree)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)
                and n.value not in docstrings]
    assert not [s for s in literals if "near_wall_crossing" in s]


def test_narrow_uses_stage_three_widths():
    args = _args("narrow_range")
    idx = args.index("--narrow_replay_width_range")
    assert args[idx + 1: idx + 3] == ["1.5", "1.7"]
    assert args[args.index("--narrow_replay_segment_length") + 1] == "8.5"


def test_corridor_args_follow_the_geometry_stage():
    a3 = _args("corridor_lateral", stage=3)
    a2 = _args("corridor_lateral", stage=2)
    assert a3[a3.index("--long_corridor_free_width") + 1] == "4.6"
    assert a2[a2.index("--long_corridor_free_width") + 1] == "4.8"
    assert a3[a3.index("--arena_size") + 1] == "17"
    assert a2[a2.index("--arena_size") + 1] == "18"


def test_corridor_motion_mode_is_the_family_not_the_scenario_name():
    args = _args("corridor_longitudinal")
    assert args[args.index("--long_corridor_motion_mode") + 1] == "longitudinal"


def test_stage_and_noise_mode_always_explicit():
    for scenario in runner.SCENARIOS:
        args = _args(scenario)
        assert "--stage" in args
        assert args[args.index("--vlp16_noise_mode") + 1] == "full"


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        _args("corridor_random2d")


# --------------------------------------------------------------------------
# cell id 必須同時帶血緣與幾何階段
# --------------------------------------------------------------------------
def test_cell_id_separates_policy_from_geometry():
    sa3 = Path("/l/sa3_run/checkpoint_12800.pt")
    sa2 = Path("/l/sa2_run/checkpoint_12800.pt")
    ids = {
        runner.make_cell_id(c, "corridor_lateral", g, 1, 818)
        for c in (sa3, sa2) for g in (2, 3)
    }
    assert len(ids) == 4


# --------------------------------------------------------------------------
# 判定
# --------------------------------------------------------------------------
def test_passing_cell():
    v = runner.evaluate("corridor_lateral", 3,
                        {"n": 1200, "sr": 0.95, "cr": 0.04, "to": 0.0})
    assert v["threshold_pass"] and v["worst_margin"] > 0


def test_failing_cell():
    v = runner.evaluate("nav_clean", 3,
                        {"n": 1200, "sr": 0.95, "cr": 0.04, "to": 0.0})
    assert v["threshold_pass"] is False  # nav_clean 要求 SR>=0.98


def test_too_few_episodes_is_void():
    with pytest.raises(RuntimeError, match="void rather than failed"):
        runner.evaluate("nav_clean", 3, {"n": 999, "sr": 1.0, "cr": 0.0, "to": 0.0})


def test_episode_count_stays_out_of_margin():
    v = runner.evaluate("corridor_lateral", 3,
                        {"n": 5000, "sr": 0.95, "cr": 0.04, "to": 0.0})
    assert set(v["checks"]) == {"sr", "cr", "to"}


# --------------------------------------------------------------------------
# 佇列規模與 fail-closed
# --------------------------------------------------------------------------
def test_phase_b_is_four_cells_on_sa3_geometry():
    cells = queue.cells_for("B")
    assert len(cells) == 4
    assert {c["scenario"] for c in cells} == set(queue.PHASE_B_SCENARIOS)
    assert all(c["geometry_stage"] == 3 for c in cells)
    assert all(c["policy"] == "sa3_c100" for c in cells)


def test_phase_c_is_six_cells_split_two_and_four():
    cells = queue.cells_for("C")
    assert len(cells) == 6
    on_sa3_geom = [c for c in cells if c["geometry_stage"] == 3]
    on_sa2_geom = [c for c in cells if c["geometry_stage"] == 2]
    assert len(on_sa3_geom) == 2 and all(c["policy"] == "sa2_c100" for c in on_sa3_geom)
    assert len(on_sa2_geom) == 4
    assert {c["policy"] for c in on_sa2_geom} == {"sa2_c100", "sa3_c100"}


def test_phase_c_matched_pair_covers_both_families():
    cells = [c for c in queue.cells_for("C") if c["geometry_stage"] == 2]
    for policy in ("sa2_c100", "sa3_c100"):
        got = {c["scenario"] for c in cells if c["policy"] == policy}
        assert got == set(runner.CORRIDOR_SCENARIOS)


def test_every_cell_pins_a_checkpoint_hash():
    for phase in ("B", "C"):
        for cell in queue.cells_for(phase):
            assert len(cell["sha256"]) == 64


def test_seed_and_delay_are_fixed():
    assert queue.GATE_SEED == 818 and queue.GATE_DELAY_STEPS == 1


def test_retention_bars_are_the_specified_values():
    assert queue.RETENTION_SR_DROP_MAX == 0.02
    assert queue.RETENTION_CR_INCREASE_MAX == 0.02


def test_retention_passes_when_child_matches_parent():
    v = queue.retention_verdict([
        ("corridor_lateral", {"sr": 0.95, "cr": 0.05}, {"sr": 0.94, "cr": 0.06}),
    ])
    assert v["pass"]


def test_retention_fails_on_sr_drop():
    v = queue.retention_verdict([
        ("corridor_lateral", {"sr": 0.95, "cr": 0.05}, {"sr": 0.92, "cr": 0.05}),
    ])
    assert not v["pass"] and v["rows"][0]["sr_drop"] == pytest.approx(0.03)


def test_retention_fails_on_cr_increase():
    v = queue.retention_verdict([
        ("corridor_lateral", {"sr": 0.95, "cr": 0.02}, {"sr": 0.95, "cr": 0.05}),
    ])
    assert not v["pass"] and v["rows"][0]["cr_increase"] == pytest.approx(0.03)


def test_phase_c_is_not_reachable_from_the_phase_b_command():
    """--phase 是必填且單選；一次呼叫不會自己接下一個 phase。"""
    source = Path(queue.__file__).read_text()
    assert 'choices=("B", "C"), required=True' in source


def test_queue_never_starts_training():
    source = Path(queue.__file__).read_text().lower()
    for forbidden in ("train_rnn_car_wdclip", "systemd-run", "systemctl", "sa4"):
        assert forbidden not in source


def test_runner_and_queue_do_not_kill_processes():
    for module in (runner, queue):
        source = Path(module.__file__).read_text()
        for forbidden in ("pkill", "killpg", "SIGKILL"):
            assert forbidden not in source


# --------------------------------------------------------------------------
# 精確計數：門檻邊界不得用四捨五入的百分比決定
# --------------------------------------------------------------------------
_LOG_TEMPLATE = """
  總回合數: 2210
  成功率 (到達目標):   2100 (95.0%)
  碰撞率 (總計):       110 (5.0%)
  超時率:                0 (0.0%)
"""


def test_exact_counts_beat_the_rounded_percentage(tmp_path):
    """110/2210 = 0.049773 < 0.05；console 印 5.0% 會讀成剛好 0.05。"""
    log = tmp_path / "cell.log"
    log.write_text(_LOG_TEMPLATE)
    exact = runner.parse_exact_outcome_counts(log)
    assert exact["n"] == 2210
    assert exact["cr"] == pytest.approx(110 / 2210)
    assert exact["cr"] < 0.05
    assert runner.evaluate("native_crossing", 3, exact)["threshold_pass"]


def test_rounded_value_alone_would_sit_exactly_on_the_bar():
    rounded = {"n": 2210, "sr": 0.95, "cr": 0.05, "to": 0.0}
    v = runner.evaluate("native_crossing", 3, rounded)
    assert v["checks"]["cr"]["margin"] == 0.0


def test_missing_exact_counts_raise_rather_than_fall_back(tmp_path):
    log = tmp_path / "cell.log"
    log.write_text("  總回合數: 2210\n")
    with pytest.raises(RuntimeError, match="rounded percentages"):
        runner.parse_exact_outcome_counts(log)


def test_reconcile_rejects_disagreeing_ledgers(tmp_path):
    exact = {"n": 2210, "sr": 0.95, "cr": 110 / 2210, "to": 0.0,
             "counts": {"success": 2100, "collision": 110, "timeout": 0}}
    with pytest.raises(RuntimeError, match="ledgers disagree"):
        runner.reconcile_console_metrics(
            {"n": 2210, "sr": 0.95, "cr": 0.09, "to": 0.0}, exact)


def test_reconcile_keeps_the_exact_counts(tmp_path):
    exact = {"n": 2210, "sr": 2100 / 2210, "cr": 110 / 2210, "to": 0.0,
             "counts": {"success": 2100, "collision": 110, "timeout": 0}}
    out = runner.reconcile_console_metrics(
        {"n": 2210, "sr": 0.95, "cr": 0.05, "to": 0.0}, exact)
    assert out["cr"] == pytest.approx(110 / 2210)
    assert out["outcome_counts"]["collision"] == 110
