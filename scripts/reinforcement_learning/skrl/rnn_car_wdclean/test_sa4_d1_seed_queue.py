"""SA4-D1 三 seed 合併分析的契約測試（CPU only）。"""

import ast
import hashlib
import json
from pathlib import Path

import pytest

import sa4_d1_seed_queue as q


def _cell(policy_stem, scenario, seed, n, collisions):
    return {
        "checkpoint": f"/l/sa4_run/{policy_stem}.pt",
        "scenario": scenario,
        "seed": seed,
        "metrics": {"n": n, "cr": collisions / n, "sr": 1 - collisions / n, "to": 0.0},
    }


# --------------------------------------------------------------------------
# 不得改動支撐既有 818 證據的檔案
# --------------------------------------------------------------------------
def test_does_not_import_the_seed818_queue():
    tree = ast.parse(Path(q.__file__).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert "sa4_checkpoint_screen_queue" not in names
    assert "run_sa4_checkpoint_screen" in names


def test_reuses_the_existing_runner_unchanged():
    assert q.RUNNER.name == "run_sa4_checkpoint_screen.py"
    assert q.RUNNER.is_file()


# --------------------------------------------------------------------------
# 格子網格
# --------------------------------------------------------------------------
def test_grid_is_eight_new_cells():
    cells = q.build_cells()
    assert len(cells) == 8
    assert {c["seed"] for c in cells} == {515, 616}
    assert {c["scenario"] for c in cells} == set(q.CORRIDOR_SCENARIOS)
    assert {c["policy"] for c in cells} == {"sa4_c50", "sa4_c100"}


def test_grid_uses_the_pre_registered_matched_step_budgets():
    cells = q.build_cells()
    for cell in cells:
        assert cell["steps"] == q.STEPS_BY_SCENARIO[cell["scenario"]]
    assert q.STEPS_BY_SCENARIO == {
        "corridor_lateral": 1200,
        "corridor_longitudinal": 4000,
    }


def test_only_corridor_families_are_rerun():
    assert q.CORRIDOR_SCENARIOS == ("corridor_lateral", "corridor_longitudinal")
    for cell in q.build_cells():
        assert cell["scenario"].startswith("corridor_")


def test_stage_and_delay_are_fixed():
    assert q.GEOMETRY_STAGE == 4
    assert q.DELAY_STEPS == 1


def test_every_cell_pins_a_checkpoint_hash():
    for cell in q.build_cells():
        assert len(cell["sha256"]) == 64


def test_checkpoint_hashes_match_the_files_on_disk():
    for _, checkpoint, digest in q.CHECKPOINTS:
        actual = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        assert actual == digest


# --------------------------------------------------------------------------
# 合併規則：按計數，不是平均比率
# --------------------------------------------------------------------------
def test_pooling_uses_counts_not_averaged_rates():
    """兩格 n 差 10 倍時，平均比率與按計數合併會給出不同答案。"""
    payloads = [
        _cell("checkpoint_6400", "corridor_lateral", 818, 100, 50),   # CR 0.50
        _cell("checkpoint_6400", "corridor_lateral", 515, 1000, 100),  # CR 0.10
    ]
    pooled = q.pool(payloads)[("sa4_c50", "corridor_lateral")]
    assert pooled["episodes"] == 1100
    assert pooled["collisions"] == 150
    assert pooled["cr"] == pytest.approx(150 / 1100)
    # 平均比率會是 0.30 —— 明顯不同，證明沒有退化成平均
    assert pooled["cr"] != pytest.approx(0.30)


def test_pool_records_every_seed():
    payloads = [
        _cell("checkpoint_12800", "corridor_lateral", s, 1000, 50)
        for s in (818, 515, 616)
    ]
    pooled = q.pool(payloads)[("sa4_c100", "corridor_lateral")]
    assert pooled["seeds"] == [515, 616, 818]
    assert len(pooled["per_seed"]) == 3


def test_non_integer_collision_count_raises():
    """比率不精確時合併會捏造精度，必須吵。"""
    bad = _cell("checkpoint_6400", "corridor_lateral", 818, 2210, 110)
    bad["metrics"]["cr"] = 0.05          # 四捨五入值，0.05*2210 = 110.5
    with pytest.raises(RuntimeError, match="fabricate precision"):
        q.pool([bad])


def test_policy_is_read_from_the_checkpoint_not_the_directory():
    payload = _cell("checkpoint_12800", "corridor_lateral", 818, 10, 1)
    payload["checkpoint"] = "/l/some_misleading_dir_name/checkpoint_12800.pt"
    assert q.policy_of(payload) == "sa4_c100"


# --------------------------------------------------------------------------
# 判定統計量：最差方向，不是合併平均
# --------------------------------------------------------------------------
def test_worst_direction_takes_the_max_family():
    payloads = [
        _cell("checkpoint_6400", "corridor_lateral", 818, 1000, 200),      # 0.20
        _cell("checkpoint_6400", "corridor_longitudinal", 818, 1000, 50),  # 0.05
    ]
    v = q.worst_direction(q.pool(payloads), "sa4_c50")
    assert v["worst_family"] == "corridor_lateral"
    assert v["worst_cr"] == pytest.approx(0.20)
    # 合併平均會是 0.125，刻意不使用
    assert v["worst_cr"] != pytest.approx(0.125)


def test_worst_direction_raises_when_a_family_is_missing():
    payloads = [_cell("checkpoint_6400", "corridor_lateral", 818, 1000, 100)]
    with pytest.raises(RuntimeError, match="missing pooled families"):
        q.worst_direction(q.pool(payloads), "sa4_c50")


def test_merged_average_never_appears_in_the_verdict():
    v = q.worst_direction(
        q.pool([
            _cell("checkpoint_6400", "corridor_lateral", 818, 1000, 200),
            _cell("checkpoint_6400", "corridor_longitudinal", 818, 1000, 50),
        ]),
        "sa4_c50",
    )
    assert set(v["per_family"]) == set(q.CORRIDOR_SCENARIOS)
    assert "merged" not in v and "mean" not in v


# --------------------------------------------------------------------------
# 既有 seed818 資料讀取
# --------------------------------------------------------------------------
def test_existing_seed_cells_are_corridor_only(tmp_path):
    for scenario in ("corridor_lateral", "nav_native"):
        d = tmp_path / scenario
        d.mkdir()
        (d / f"{scenario}_g4_d1_s818_cell.json").write_text(
            json.dumps(_cell("checkpoint_6400", scenario, 818, 1000, 100))
        )
    loaded = q.load_existing_seed_cells(tmp_path)
    assert [p["scenario"] for p in loaded] == ["corridor_lateral"]


def test_existing_screen_root_exists_on_disk():
    assert q.EXISTING_SCREEN_ROOT.is_dir()


def test_existing_seed818_grid_passes_identity_and_evidence_validation():
    payloads = q.load_existing_seed_cells()
    assert q.validate_payload_set(
        payloads,
        q.expected_existing_cells(),
        source="test",
    ) == []


def test_three_seeds_after_pooling_with_the_existing_batch():
    new = [_cell("checkpoint_6400", s, seed, 1000, 100)
           for s in q.CORRIDOR_SCENARIOS for seed in q.NEW_SEEDS]
    old = [_cell("checkpoint_6400", s, q.EXISTING_SEED, 1000, 100)
           for s in q.CORRIDOR_SCENARIOS]
    pooled = q.pool(new + old)[("sa4_c50", "corridor_lateral")]
    assert pooled["seeds"] == [515, 616, 818]


# --------------------------------------------------------------------------
# 不得延長訓練或啟動下一階段
# --------------------------------------------------------------------------
def test_queue_starts_no_training():
    source = Path(q.__file__).read_text().lower()
    for forbidden in ("train_rnn_car_wdclip", "systemd-run", "systemctl"):
        assert forbidden not in source


def test_queue_does_not_kill_processes():
    source = Path(q.__file__).read_text()
    for forbidden in ("pkill", "killpg", "SIGKILL"):
        assert forbidden not in source


# --------------------------------------------------------------------------
# 部分完成不得產生結論
# --------------------------------------------------------------------------
def test_runner_invocation_passes_every_required_argument():
    """漏傳必填參數會讓每一格靜默失敗，而 queue 仍可能印出看似合理的表。"""
    import inspect
    src = inspect.getsource(q.run_cell)
    for required in ("--scenario", "--seed", "--actuator-delay-steps",
                     "--expect-checkpoint-sha256", "--output-dir", "--steps"):
        assert required in src, f"run_cell 未傳 {required}"


def test_delay_argument_carries_the_module_constant():
    import inspect
    src = inspect.getsource(q.run_cell)
    assert '"--actuator-delay-steps", str(DELAY_STEPS)' in src


def test_steps_argument_carries_the_per_cell_budget():
    import inspect
    src = inspect.getsource(q.run_cell)
    assert '"--steps", str(cell["steps"])' in src


def test_run_cell_removes_a_stale_cell_json_before_launch():
    import inspect
    src = inspect.getsource(q.run_cell)
    assert "cell_path.unlink(missing_ok=True)" in src
    assert src.index("cell_path.unlink") < src.index("subprocess.run")


def test_low_episode_payload_is_invalid():
    cell = q.build_cells()[0]
    payload = {
        "checkpoint": cell["checkpoint"],
        "checkpoint_sha256": cell["sha256"],
        "scenario": cell["scenario"],
        "seed": cell["seed"],
        "delay_steps": q.DELAY_STEPS,
        "geometry_stage": q.GEOMETRY_STAGE,
        "steps": cell["steps"],
        "metrics": {"n": 999, "sr": 0.8, "cr": 0.2, "to": 0.0},
    }
    assert any("minimum=1000" in error for error in q.validate_payload(payload, cell))


def test_payload_identity_mismatch_is_invalid():
    cell = q.build_cells()[0]
    payload = {
        "checkpoint": cell["checkpoint"],
        "checkpoint_sha256": cell["sha256"],
        "scenario": cell["scenario"],
        "seed": cell["seed"],
        "delay_steps": 0,
        "geometry_stage": q.GEOMETRY_STAGE,
        "steps": cell["steps"],
        "metrics": {"n": 1000, "sr": 0.8, "cr": 0.2, "to": 0.0},
    }
    assert any("delay_steps=0" in error for error in q.validate_payload(payload, cell))


def test_partial_run_reports_no_pooled_verdict():
    """有 invalid 時必須 return 非零並且不輸出合併結論。"""
    import inspect
    src = inspect.getsource(q.main)
    marker = src.split("if invalid or len(new_payloads) != len(cells):", 1)
    assert len(marker) == 2, "main 缺少 fail-closed 分支"
    branch = marker[1].split("existing = load_existing_seed_cells()", 1)[0]
    assert "return 1" in branch
    # fail-closed 分支必須早於任何合併
    assert src.index("if invalid or len(new_payloads)") < src.index("buckets = pool(")


def test_source_drift_and_existing_evidence_fail_before_pooling():
    import inspect
    src = inspect.getsource(q.main)
    assert "if evidence_invalid or not stable:" in src
    assert src.index("if evidence_invalid or not stable:") < src.index(
        "buckets = pool("
    )
    branch = src.split("if evidence_invalid or not stable:", 1)[1].split(
        "buckets = pool(", 1
    )[0]
    assert "return 1" in branch
