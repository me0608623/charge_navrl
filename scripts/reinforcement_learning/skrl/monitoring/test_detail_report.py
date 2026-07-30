"""detail_report 的單元測試。

測試策略：判定邏輯全部是純函式，用假 metric rows 驅動，不碰 GPU、不碰真實
訓練。I/O 外殼只測「找得到正確的 run」與「狀態檔往返」。
"""

import argparse
import json
from pathlib import Path

import pytest

import detail_report as dr


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _row(iteration, **overrides):
    """一列健康的 metrics，需要哪個欄位壞掉再用 overrides 蓋掉。"""
    row = {
        "iteration": iteration,
        "iterations_target": 300,
        "sr": 0.96,
        "cr": 0.04,
        "timeout": 0.0,
        "entropy": 3.70,
        "kl": 0.005,
        "clip_fraction": 0.06,
        "policy_loss": -0.003,
        "vf": 4.0,
        "actor_param_delta_norm": 0.017,
        "critic_param_delta_norm": 0.015,
        "encoder_grad": 5.2,
        "fps": 3450.0,
        "stage": 3,
        "total_episodes": 4100.0,
        "scene/native/episodes": 3500.0,
        "scene/native/sr": 0.97,
        "scene/native/cr": 0.03,
        "scene/native/timeout": 0.0,
        "scene/narrow/episodes": 330.0,
        "scene/narrow/sr": 1.0,
        "scene/narrow/cr": 0.0,
        "scene/narrow/timeout": 0.0,
        "scene/corridor/episodes": 260.0,
        "scene/corridor/sr": 0.95,
        "scene/corridor/cr": 0.05,
        "scene/corridor/timeout": 0.0,
    }
    row.update(overrides)
    return row


def _rows(count=80, **overrides):
    return [_row(i, **overrides) for i in range(1, count + 1)]


def _levels(findings):
    return {f.code: f.level for f in findings}


# --------------------------------------------------------------------------
# 核心健康檢查：NaN / None
# --------------------------------------------------------------------------
def test_core_health_clean_row_has_no_findings():
    assert dr.core_health(_row(10)) == []


@pytest.mark.parametrize("key", ["sr", "cr", "entropy", "kl", "vf", "policy_loss"])
def test_core_health_flags_none_in_each_core_field(key):
    findings = dr.core_health(_row(10, **{key: None}))
    assert [f.level for f in findings] == ["RED"]
    assert key in findings[0].message


@pytest.mark.parametrize("key", ["sr", "entropy", "kl", "vf"])
def test_core_health_flags_nan(key):
    findings = dr.core_health(_row(10, **{key: float("nan")}))
    assert [f.level for f in findings] == ["RED"]
    assert key in findings[0].message


def test_core_health_flags_inf():
    findings = dr.core_health(_row(10, vf=float("inf")))
    assert [f.level for f in findings] == ["RED"]


def test_core_health_missing_field_is_reported_not_silently_passed():
    """欄位不存在必須是 finding，不能因為讀不到就當作通過。"""
    row = _row(10)
    del row["entropy"]
    findings = dr.core_health(row)
    assert [f.level for f in findings] == ["RED"]
    assert "entropy" in findings[0].message


# --------------------------------------------------------------------------
# 趨勢：回報實際跨度，不假裝跨了要求的區間
# --------------------------------------------------------------------------
def test_trend_reports_actual_span_when_history_is_short():
    rows = _rows(10)
    t = dr.trend(rows, "entropy", lookback=50)
    assert t.span == 9  # 只有 10 列，跨度就是 9，不是 50
    assert t.requested_lookback == 50


def test_trend_uses_full_lookback_when_available():
    rows = [_row(i, entropy=4.0 - 0.01 * i) for i in range(1, 81)]
    t = dr.trend(rows, "entropy", lookback=50)
    assert t.span == 50
    assert t.first == pytest.approx(4.0 - 0.01 * 30)
    assert t.last == pytest.approx(4.0 - 0.01 * 80)
    assert t.delta == pytest.approx(-0.50)


def test_trend_returns_none_for_missing_key():
    assert dr.trend(_rows(10), "does_not_exist", lookback=5) is None


def test_trend_skips_non_numeric_values():
    rows = _rows(10)
    rows[-1]["entropy"] = None
    t = dr.trend(rows, "entropy", lookback=5)
    assert t is not None
    assert t.last is not None


# --------------------------------------------------------------------------
# 判定規則
# --------------------------------------------------------------------------
def test_healthy_run_is_green():
    findings = dr.evaluate(_rows(80))
    assert dr.verdict(findings) == "GREEN"


def test_kl_above_red_threshold_is_red():
    rows = _rows(80)
    rows[-1]["kl"] = 0.025
    findings = dr.evaluate(rows)
    assert _levels(findings)["kl_high"] == "RED"
    assert dr.verdict(findings) == "RED"


def test_kl_in_yellow_band_is_yellow():
    rows = _rows(80)
    rows[-1]["kl"] = 0.013
    findings = dr.evaluate(rows)
    assert _levels(findings)["kl_high"] == "YELLOW"
    assert dr.verdict(findings) == "YELLOW"


def test_clip_fraction_spike_is_flagged():
    rows = _rows(80)
    rows[-1]["clip_fraction"] = 0.35
    assert _levels(dr.evaluate(rows))["clip_high"] == "RED"


def test_entropy_collapse_is_red():
    rows = [_row(i, entropy=4.0 - 0.02 * i) for i in range(1, 81)]  # 50 iter 掉 1.0
    findings = dr.evaluate(rows)
    assert _levels(findings)["entropy_drop"] == "RED"


def test_entropy_mild_decline_is_yellow():
    rows = [_row(i, entropy=4.0 - 0.006 * i) for i in range(1, 81)]  # 50 iter 掉 0.30
    findings = dr.evaluate(rows)
    assert _levels(findings)["entropy_drop"] == "YELLOW"


def test_entropy_rising_is_not_flagged():
    rows = [_row(i, entropy=3.0 + 0.01 * i) for i in range(1, 81)]
    assert "entropy_drop" not in _levels(dr.evaluate(rows))


def test_cr_rebound_is_flagged():
    rows = [_row(i, cr=0.03) for i in range(1, 61)]
    rows += [_row(i, cr=0.03 + 0.004 * (i - 60)) for i in range(61, 81)]  # +0.08
    findings = dr.evaluate(rows)
    assert _levels(findings)["cr_rebound"] == "RED"


def test_cr_falling_is_not_flagged():
    rows = [_row(i, cr=0.10 - 0.001 * i) for i in range(1, 81)]
    assert "cr_rebound" not in _levels(dr.evaluate(rows))


def test_fps_drop_is_yellow_not_red():
    """fps 掉不等於壞掉 —— 可能是別人的 GPU job，只能 YELLOW 並要求查 GPU。"""
    rows = _rows(80)
    for row in rows[-3:]:
        row["fps"] = 1200.0
    findings = dr.evaluate(rows)
    assert _levels(findings)["fps_drop"] == "YELLOW"
    assert "nvidia-smi" in next(f for f in findings if f.code == "fps_drop").message


def test_verdict_takes_worst_level():
    findings = [
        dr.Finding("GREEN", "a", "ok"),
        dr.Finding("YELLOW", "b", "meh"),
        dr.Finding("RED", "c", "bad"),
    ]
    assert dr.verdict(findings) == "RED"


def test_verdict_of_empty_findings_is_green():
    assert dr.verdict([]) == "GREEN"


# --------------------------------------------------------------------------
# 進度 / ETA / stall
# --------------------------------------------------------------------------
def test_progress_rate_from_two_observations():
    prev = {"observed_at": 1000.0, "iteration": 10}
    rate = dr.iteration_rate(prev, {"observed_at": 1600.0, "iteration": 30})
    assert rate == pytest.approx(20 / 600.0)


def test_progress_rate_none_without_previous_observation():
    assert dr.iteration_rate(None, {"observed_at": 1.0, "iteration": 5}) is None


def test_progress_rate_none_when_no_iteration_advanced():
    prev = {"observed_at": 1000.0, "iteration": 30}
    assert dr.iteration_rate(prev, {"observed_at": 1600.0, "iteration": 30}) is None


def test_eta_seconds_from_rate():
    assert dr.eta_seconds(rate=0.03, remaining=300) == pytest.approx(10000.0)


def test_eta_none_without_rate():
    assert dr.eta_seconds(rate=None, remaining=300) is None


def test_stall_detected_when_iteration_and_log_size_both_frozen():
    prev = {"observed_at": 1000.0, "iteration": 30, "log_size": 5000}
    cur = {"observed_at": 1600.0, "iteration": 30, "log_size": 5000}
    findings = dr.check_stall(prev, cur, process_alive=True)
    assert _levels(findings)["stall"] == "RED"


def test_no_stall_when_log_grows_even_if_iteration_is_frozen():
    prev = {"observed_at": 1000.0, "iteration": 30, "log_size": 5000}
    cur = {"observed_at": 1600.0, "iteration": 30, "log_size": 9000}
    assert _levels(dr.check_stall(prev, cur, process_alive=True)) == {}


def test_process_gone_before_target_is_red():
    findings = dr.check_process(iteration=120, target=300, process_alive=False)
    assert _levels(findings)["process_gone"] == "RED"


def test_process_gone_at_target_is_completion_not_alarm():
    findings = dr.check_process(iteration=300, target=300, process_alive=False)
    assert _levels(findings)["run_complete"] == "GREEN"


def test_process_alive_produces_no_finding():
    assert dr.check_process(iteration=120, target=300, process_alive=True) == []


# --------------------------------------------------------------------------
# 分場景表
# --------------------------------------------------------------------------
def test_scene_rows_extracted_for_all_present_scenes():
    scenes = dr.scene_snapshot(_row(80))
    assert [s.name for s in scenes] == ["native", "narrow", "corridor"]
    assert scenes[1].episodes == 330.0
    assert scenes[1].sr == 1.0


def test_scene_snapshot_empty_when_run_has_no_scene_keys():
    row = {k: v for k, v in _row(80).items() if not k.startswith("scene/")}
    assert dr.scene_snapshot(row) == []


def test_scene_snapshot_tolerates_zero_episodes():
    row = _row(80, **{"scene/corridor/episodes": 0.0, "scene/corridor/sr": None})
    scenes = dr.scene_snapshot(row)
    corridor = next(s for s in scenes if s.name == "corridor")
    assert corridor.episodes == 0.0
    assert corridor.sr is None


def test_low_episode_scene_is_flagged_as_low_confidence():
    rows = _rows(80)
    rows[-1]["scene/corridor/episodes"] = 3.0
    rows[-1]["scene/corridor/cr"] = 0.33
    findings = dr.evaluate(rows)
    assert _levels(findings)["scene_low_n"] == "YELLOW"


def test_scene_cr_alone_does_not_trigger_red_when_n_is_tiny():
    """3 個 episode 的 33% CR 不是證據，不得升級成 RED。"""
    rows = _rows(80)
    rows[-1]["scene/corridor/episodes"] = 3.0
    rows[-1]["scene/corridor/cr"] = 0.33
    assert dr.verdict(dr.evaluate(rows)) != "RED"


# --------------------------------------------------------------------------
# I/O 外殼
# --------------------------------------------------------------------------
def test_find_active_run_picks_newest_dir_with_metrics(tmp_path):
    old = tmp_path / "old_run"
    new = tmp_path / "new_run"
    decoy = tmp_path / "no_metrics_run"
    for d in (old, new, decoy):
        d.mkdir()
    (old / "supervisor_metrics.jsonl").write_text("{}\n")
    (new / "supervisor_metrics.jsonl").write_text("{}\n")
    (decoy / "console.log").write_text("noise\n")
    import os

    os.utime(old / "supervisor_metrics.jsonl", (1000, 1000))
    os.utime(new / "supervisor_metrics.jsonl", (2000, 2000))
    os.utime(decoy, (3000, 3000))
    assert dr.find_active_run(tmp_path) == new


def test_find_active_run_returns_none_when_nothing_matches(tmp_path):
    assert dr.find_active_run(tmp_path) is None


def test_load_rows_skips_blank_lines(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(_row(1)) + "\n\n" + json.dumps(_row(2)) + "\n")
    assert [r["iteration"] for r in dr.load_rows(p)] == [1, 2]


def test_load_rows_raises_on_corrupt_line(tmp_path):
    """壞掉的 jsonl 必須吵，不能安靜跳過整段歷史。"""
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps(_row(1)) + "\n{not json}\n")
    with pytest.raises(ValueError):
        dr.load_rows(p)


def test_state_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    assert dr.read_state(p) is None
    obs = {"observed_at": 123.0, "iteration": 7, "log_size": 42, "run": "r"}
    dr.write_state(p, obs)
    assert dr.read_state(p) == obs


def test_read_state_returns_none_on_corrupt_file(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{broken")
    assert dr.read_state(p) is None


def test_state_is_scoped_to_run_name(tmp_path):
    """換 run 之後不能拿舊 run 的 iteration 算速率。"""
    p = tmp_path / "state.json"
    dr.write_state(p, {"observed_at": 1.0, "iteration": 200, "log_size": 1, "run": "old"})
    assert dr.previous_for_run(p, "new") is None
    assert dr.previous_for_run(p, "old") is not None


# --------------------------------------------------------------------------
# 渲染
# --------------------------------------------------------------------------
def test_render_contains_verdict_and_iteration():
    text = dr.render_report(
        run_name="demo_run",
        rows=_rows(80),
        process_alive=True,
        gpu_lines=["9151 MiB, 32607 MiB, 26 %, 47"],
        gpu_procs=[("289593", "8084 MiB")],
        errors=[],
        replay_mix="native=77.3% narrow=9.4% corridor=13.3%",
        rate=0.03,
        findings=dr.evaluate(_rows(80)),
        now_text="2026-07-30 17:00:00 CST",
    )
    assert "GREEN" in text
    assert "80/300" in text
    assert "demo_run" in text
    assert "corridor" in text


def test_render_survives_run_without_scene_metrics():
    rows = [
        {k: v for k, v in r.items() if not k.startswith("scene/")} for r in _rows(20)
    ]
    text = dr.render_report(
        run_name="legacy_run",
        rows=rows,
        process_alive=True,
        gpu_lines=[],
        gpu_procs=[],
        errors=[],
        replay_mix=None,
        rate=None,
        findings=[],
        now_text="t",
    )
    assert "legacy_run" in text


def test_render_lists_every_finding():
    findings = [
        dr.Finding("YELLOW", "fps_drop", "fps 掉了"),
        dr.Finding("RED", "kl_high", "KL 太高"),
    ]
    text = dr.render_report(
        run_name="r",
        rows=_rows(20),
        process_alive=True,
        gpu_lines=[],
        gpu_procs=[],
        errors=[],
        replay_mix=None,
        rate=None,
        findings=findings,
        now_text="t",
    )
    assert "fps 掉了" in text
    assert "KL 太高" in text


# --------------------------------------------------------------------------
# 來源契約：這支程式只回報，永不介入
# --------------------------------------------------------------------------
def test_module_never_kills_a_process():
    source = Path(dr.__file__).read_text()
    for forbidden in ("kill(", "pkill", "killpg", "terminate(", "SIGKILL", "SIGTERM"):
        assert forbidden not in source, f"報告程式不得含有介入手段: {forbidden}"


def test_module_does_not_write_outside_declared_outputs():
    """只允許寫報告檔與狀態檔，不得改訓練或凍結來源。"""
    source = Path(dr.__file__).read_text()
    assert "logs/rnn_car" not in source.replace("LOGS_DIR_DEFAULT", "")


def test_build_report_without_any_run_explains_why(tmp_path):
    """找不到 run 時的訊息必須完整說明原因，不能被字串串接截斷。"""
    args = argparse.Namespace(
        logs_dir=str(tmp_path / "empty"),
        run=None,
        state=str(tmp_path / "state.json"),
    )
    observation, text = dr.build_report(args)
    assert observation is None
    assert dr.METRICS_FILENAME in text
    assert "logs_dir" in text


def test_thresholds_are_immutable():
    t = dr.Thresholds()
    with pytest.raises(Exception):
        t.kl_red = 0.5


def test_thresholds_are_all_documented_constants():
    """門檻必須是具名常數，不能散落在判定函式裡。"""
    source = Path(dr.__file__).read_text()
    body = source.split("def evaluate(", 1)[1].split("\ndef ", 1)[0]
    assert "0.02" not in body, "KL 門檻應來自 Thresholds，不得寫死在 evaluate 內"
