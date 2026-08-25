"""Contracts for the bounded provisional SA5-v3 c550 pilot."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SKRL = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SKRL))

import run_sa5_v3_provisional_pilot_screen_cell as runner  # noqa: E402
import sa5_v3_provisional_pilot_screen as protocol  # noqa: E402
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa4_k8_obb_speed_density_v3_from_sa3 as sa4,
)
from rnn_car_modular.configs import (  # noqa: E402
    e2e_sa5_v3_provisional_from_sa4v3_c550_p50 as pilot,
)


AUTHORIZATION = (
    REPO
    / "docs/freeze/sa5_v3_provisional_c550_parent_authorization_20260822.json"
)
SCREEN_FREEZE = REPO / "docs/freeze/sa5_v3_provisional_pilot_screen_v1.json"
LAUNCH_FREEZE = (
    REPO / "docs/freeze/sa5_v3_provisional_c550_pilot_launch_v1.json"
)
QUEUE = HERE / "sa5_v3_provisional_pilot_screen_queue.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_candidate_manifest_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        protocol,
        "candidate_by_name",
        lambda name: {"name": name, "sha256": f"sha-{name}"},
    )


def _metrics(
    scenario: str,
    cr: float,
    *,
    n: int = 4000,
    sr: float | None = None,
    to: float = 0.0,
) -> dict:
    result = {
        "n": n,
        "sr": 1.0 - cr - to if sr is None else sr,
        "cr": cr,
        "to": to,
    }
    if scenario == "narrow_range":
        result.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    return result


def _cell(name: str, scenario: str, metrics: dict) -> dict:
    verdict = protocol.evaluate_metrics(scenario, metrics)
    return {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoint_name": name,
        "checkpoint_sha256": f"sha-{name}",
        "scenario": scenario,
        "geometry_stage": protocol.STAGE,
        "seed": protocol.SEED,
        "delay_steps": protocol.DELAY_STEPS,
        "num_envs": protocol.NUM_ENVS,
        "steps": protocol.STEPS_BY_SCENARIO[scenario],
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": (
            protocol.LIDAR_DISTRACTOR_ELIGIBILITY
        ),
        "speed_rate_runtime": {
            "speed_rate": protocol.SPEED_RATE,
            "speed_rate_obs": protocol.SPEED_RATE_OBS,
            "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        },
        "metrics": metrics,
        "threshold_pass": verdict["threshold_pass"],
    }


def _matrix(
    *,
    it25_target_cr: float = 0.09,
    it50_target_cr: float = 0.11,
    it25_native_cr: float = 0.09,
) -> list[dict]:
    parent = {
        protocol.TARGET_SCENARIO: _metrics(protocol.TARGET_SCENARIO, 0.12),
        "nav_native": _metrics("nav_native", 0.085),
        "narrow_range": _metrics("narrow_range", 0.01),
        "corridor_low_0s1d": _metrics("corridor_low_0s1d", 0.04),
        "corridor_low_1s1d": _metrics("corridor_low_1s1d", 0.05),
    }
    it25 = {
        protocol.TARGET_SCENARIO: _metrics(
            protocol.TARGET_SCENARIO, it25_target_cr
        ),
        "nav_native": _metrics("nav_native", it25_native_cr),
        "narrow_range": _metrics("narrow_range", 0.015),
        "corridor_low_0s1d": _metrics("corridor_low_0s1d", 0.045),
        "corridor_low_1s1d": _metrics("corridor_low_1s1d", 0.055),
    }
    it50 = {
        protocol.TARGET_SCENARIO: _metrics(
            protocol.TARGET_SCENARIO, it50_target_cr
        ),
        "nav_native": _metrics("nav_native", 0.09),
        "narrow_range": _metrics("narrow_range", 0.015),
        "corridor_low_0s1d": _metrics("corridor_low_0s1d", 0.045),
        "corridor_low_1s1d": _metrics("corridor_low_1s1d", 0.055),
    }
    values = {"parent_c550": parent, "pilot_it25": it25, "pilot_it50": it50}
    return [
        _cell(name, scenario, values[name][scenario])
        for name in ("parent_c550", "pilot_it25", "pilot_it50")
        for scenario in protocol.ALL_SCENARIOS
    ]


def test_authorization_preserves_failed_sa4_and_provisional_parent() -> None:
    record = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    assert record["lineage_status"] == {
        "sa4": "SA4_NOT_GRADUATED",
        "sa5_parent": "PROVISIONAL_SA5_PARENT_C550",
        "meaning": (
            "Training may continue under a bounded SA5 pilot, but the failed "
            "SA4 graduation verdict remains unchanged."
        ),
    }
    assert record["parent"]["sha256"] == pilot.PARENT_CHECKPOINT_SHA256
    assert record["parent"]["optimizer_resume"] is True


def test_parent_checkpoint_contains_resumable_optimizer() -> None:
    path = Path(pilot.PARENT_CHECKPOINT)
    assert _sha256(path) == pilot.PARENT_CHECKPOINT_SHA256
    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer = payload["charge_opt_rl"]
    assert optimizer["state"]
    assert optimizer["param_groups"]
    assert payload["iteration"] == 249
    assert payload["total_steps"] == 32000


def test_only_preregistered_fields_change_from_sa4() -> None:
    actual = {
        field.name
        for field in fields(pilot.CONFIG)
        if field.name not in pilot.METADATA_FIELDS
        and getattr(pilot.CONFIG, field.name) != getattr(sa4.CONFIG, field.name)
    }
    assert actual == pilot.EXPECTED_CHANGED_FIELDS
    assert pilot.CONFIG.no_resume_optimizer is False


def test_stage5_replay_keeps_native_narrow_and_low_density_corridors() -> None:
    cfg = pilot.CONFIG
    assert cfg.narrow_passage_fraction == pytest.approx(0.12)
    assert cfg.long_corridor_fraction == pytest.approx(0.10)
    assert 1.0 - cfg.narrow_passage_fraction - cfg.long_corridor_fraction == pytest.approx(
        0.78
    )
    assert sum(row[2] for row in pilot.SA5_SPEED_DENSITY_MIX) == pytest.approx(1.0)
    assert [row[0] for row in pilot.SA5_SPEED_DENSITY_MIX[:2]] == [(0, 1), (1, 1)]
    assert [row[1] for row in pilot.SA5_SPEED_DENSITY_MIX[:2]] == [
        (0.70, 0.90),
        (0.70, 0.90),
    ]
    assert all(row[1][1] < 1.0 for row in pilot.SA5_SPEED_DENSITY_MIX)


def test_pilot_budget_and_checkpoints_are_bounded() -> None:
    assert pilot.CONFIG.timesteps == 50 * 128
    assert pilot.CONFIG.save_interval == 25
    assert pilot.CHECKPOINTS == (
        (25, "checkpoint_3200.pt"),
        (50, "checkpoint_6400.pt"),
    )


def test_launch_freeze_matches_current_sources_and_optimizer_decision() -> None:
    freeze = json.loads(LAUNCH_FREEZE.read_text(encoding="utf-8"))
    assert freeze["lineage"]["sa4"] == "SA4_NOT_GRADUATED"
    assert (
        freeze["lineage"]["sa5_parent"]
        == "PROVISIONAL_SA5_PARENT_C550"
    )
    assert freeze["run"]["config_sha256"] == _sha256(Path(pilot.__file__))
    assert freeze["run"]["trainer_sha256"] == _sha256(
        REPO / freeze["run"]["trainer_path"]
    )
    assert freeze["parent"]["checkpoint_sha256"] == pilot.PARENT_CHECKPOINT_SHA256
    assert freeze["parent"]["optimizer_policy"] == "RESUME_EMBEDDED_OPTIMIZER"
    assert freeze["curriculum"]["scene_reset_shares"] == {
        "native": 0.78,
        "narrow": 0.12,
        "corridor": 0.10,
    }
    assert freeze["curriculum"]["low_density_replay_retained"] is True
    assert freeze["pilot"]["auto_extend"] is False
    assert freeze["pilot"]["auto_start_sa6"] is False


def test_frozen_screen_matches_runtime_protocol() -> None:
    assert json.loads(SCREEN_FREEZE.read_text(encoding="utf-8")) == (
        protocol.screen_protocol()
    )
    assert len(protocol.CANDIDATE_SPECS) * len(protocol.ALL_SCENARIOS) == 15


def test_runner_pins_speed_density_noise_and_delay() -> None:
    values = protocol.screen_protocol()["fixed_evaluation"]["scene"]
    for scenario, density, speed, mode in (
        (protocol.TARGET_SCENARIO, (4, 2), protocol.P035, "mixed"),
        ("corridor_low_0s1d", (0, 1), protocol.P060, "lateral"),
        ("corridor_low_1s1d", (1, 1), protocol.P060, "lateral"),
    ):
        args = runner.build_scene_args(scenario, Path("corridor.json"))
        joined = " ".join(args)
        assert f"--long_corridor_static_obstacles {density[0]}" in joined
        assert f"--long_corridor_dynamic_obstacles {density[1]}" in joined
        assert f"--long_corridor_dynamic_speed_range {speed[0]:g} {speed[1]:g}" in joined
        assert f"--long_corridor_motion_mode {mode}" in joined
        assert "--speed_rate 0.7" in joined
        assert "--speed_rate_obs ego" in joined
        assert "--deployment_speed_scale 1" in joined
        assert "--actuator_delay_range 1 1" in joined
        assert "--vlp16_noise_mode full" in joined
        assert values["corridor_sealed_to_boundary"] is True


def test_clear_target_improvement_with_retention_authorizes_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_candidate_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(_matrix())
    assert verdict["selected_extension_checkpoint"] == "pilot_it25"
    assert verdict["extension_authorized"] is True
    assert verdict["next_action"] == "EXTEND_SELECTED_C550_LINEAGE_TO_TOTAL_IT300"
    assert verdict["sa4_graduated"] is False
    assert verdict["sa6_started"] is False


def test_nonsignificant_target_change_routes_to_c500_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_candidate_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(
        _matrix(it25_target_cr=0.115, it50_target_cr=0.116)
    )
    assert verdict["selected_extension_checkpoint"] is None
    assert verdict["extension_authorized"] is False
    assert verdict["next_action"] == "RUN_IDENTICAL_C500_PARENT_CONTROL_P50"


def test_native_above_ten_percent_blocks_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_candidate_manifest_stub(monkeypatch)
    verdict = protocol.final_verdict(
        _matrix(it25_native_cr=0.11, it50_target_cr=0.119)
    )
    row = verdict["candidate_results"][0]
    assert row["target_improvement"]["pass"] is True
    assert row["retention"]["nav_native"]["absolute_pass"] is False
    assert row["extension_eligible"] is False


def test_partial_matrix_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_candidate_manifest_stub(monkeypatch)
    with pytest.raises(ValueError, match="exactly 15 cells"):
        protocol.final_verdict(_matrix()[:-1])


def test_screen_queue_cannot_launch_training_or_sa6() -> None:
    source = QUEUE.read_text(encoding="utf-8")
    assert "INCOMPLETE_FAIL_CLOSED" in source
    assert "source fingerprint drifted during screen" in source
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemd-run" not in source
    assert '"sa6_started": False' in source
