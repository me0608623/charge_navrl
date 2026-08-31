"""CPU locks for the SA5 B3 c500/c550/c600 fixed screen."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import sa5_v3_c50_stage3_b3_c500_c600_screen as protocol
import sa5_v3_c50_stage3_b3_c500_c600_screen_queue as queue
import sa5_v3_c50_stage3_b3_cont25_screen as source


def _payload(name: str, scenario: str, cr: float, to: float = 0.0) -> dict:
    candidate = protocol.candidate_by_name(name)
    speed = protocol.speed_range_for(scenario)
    density = protocol.density_for(scenario)
    sr = 1.0 - cr - to
    return {
        "schema": protocol.CELL_SCHEMA,
        "cell_valid": True,
        "checkpoint_name": name,
        "checkpoint_sha256": candidate["sha256"],
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "geometry_stage": protocol.STAGE,
        "scenario": scenario,
        "seed": protocol.SEED,
        "delay_steps": protocol.DELAY_STEPS,
        "num_envs": protocol.NUM_ENVS,
        "steps": protocol.STEPS_BY_SCENARIO[scenario],
        "lidar_noise_mode": protocol.LIDAR_NOISE_MODE,
        "lidar_distractor_eligibility": protocol.LIDAR_DISTRACTOR_ELIGIBILITY,
        "speed_rate_runtime": {
            "speed_rate": protocol.SPEED_RATE,
            "speed_rate_obs": protocol.SPEED_RATE_OBS,
            "deployment_speed_scale": protocol.DEPLOYMENT_SPEED_SCALE,
        },
        "corridor_report": {
            "requested_dynamic_speed_range_m_s": list(speed),
            "static_obstacles_per_env": density[0],
            "dynamic_obstacles_per_env": density[1],
            "dynamic_motion_mode": "lateral",
        },
        "metrics": {"n": 2000, "sr": sr, "cr": cr, "to": to},
        "threshold_pass": sr >= 0.90 and cr <= 0.10 and to <= 0.05,
    }


def _matrix(values: dict[str, tuple[float, float, float, float]]) -> list[dict]:
    return [
        _payload(spec["name"], scenario, cr)
        for spec in protocol.CANDIDATE_SPECS
        for scenario, cr in zip(protocol.ALL_SCENARIOS, values[spec["name"]])
    ]


def test_fixed_protocol_is_unchanged_from_bounded_continuation_screen():
    assert protocol.ALL_SCENARIOS == source.ALL_SCENARIOS
    assert protocol.STEPS_BY_SCENARIO == source.STEPS_BY_SCENARIO
    assert protocol.THRESHOLDS == source.THRESHOLDS
    for name in (
        "STAGE",
        "SEED",
        "NUM_ENVS",
        "MIN_EPISODES",
        "DELAY_STEPS",
        "DELAY_MS",
        "ACTUATOR_PROFILE",
        "LIDAR_NOISE_MODE",
        "LIDAR_DISTRACTOR_ELIGIBILITY",
        "SPEED_RATE",
        "SPEED_RATE_OBS",
        "DEPLOYMENT_SPEED_SCALE",
        "P060",
        "P080",
        "MAX_DEGRADATION",
        "TIE_BAND",
    ):
        assert getattr(protocol, name) == getattr(source, name), name


def test_candidate_identity_hash_and_checkpoint_ledger_are_locked():
    assert [row["name"] for row in protocol.CANDIDATE_SPECS] == [
        "c500",
        "c550",
        "c600",
    ]
    assert [row["conceptual_iteration"] for row in protocol.CANDIDATE_SPECS] == [
        500,
        550,
        600,
    ]
    assert [row["embedded_iteration"] for row in protocol.CANDIDATE_SPECS] == [
        399,
        449,
        499,
    ]
    assert [row["embedded_total_steps"] for row in protocol.CANDIDATE_SPECS] == [
        51_200,
        57_600,
        64_000,
    ]
    for spec in protocol.CANDIDATE_SPECS:
        path = protocol.checkpoint_path(spec)
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec["expected_sha256"]
    manifest = queue.build_checkpoint_manifest()
    assert set(manifest["checkpoints"]) == {"c500", "c550", "c600"}


def test_exactly_twelve_cells_and_frozen_protocol_match():
    assert len(protocol.CANDIDATE_SPECS) * len(protocol.ALL_SCENARIOS) == 12
    current = protocol.screen_protocol()
    frozen = json.loads(queue.FREEZE.read_text(encoding="utf-8"))
    assert frozen["protocol_sha256"] == current["sha256"]
    assert queue.verify_frozen_protocol() == current


def test_ranking_requires_p060_gate_and_p080_retention():
    values = {
        "c500": (0.09, 0.095, 0.08, 0.08),
        "c550": (0.08, 0.085, 0.09, 0.08),
        "c600": (0.07, 0.075, 0.11, 0.08),
    }
    verdict = protocol.final_verdict(_matrix(values))
    assert verdict["passing_candidates"] == ["c500", "c550"]
    assert verdict["tie_group"] == ["c550"]
    assert verdict["sa5_candidate"] == "c550"
    assert verdict["phase_b_authorized"] is False
    assert verdict["parent_selected"] is False
    assert verdict["sa6_started"] is False


def test_tie_band_prefers_mean_then_earlier_checkpoint():
    values = {
        "c500": (0.08, 0.08, 0.08, 0.08),
        "c550": (0.077, 0.079, 0.08, 0.08),
        "c600": (0.076, 0.080, 0.08, 0.08),
    }
    verdict = protocol.final_verdict(_matrix(values))
    assert verdict["tie_group"] == ["c550", "c600", "c500"]
    assert verdict["sa5_candidate"] == "c550"


def test_fail_closed_on_partial_duplicate_or_tampered_matrix():
    values = {name: (0.08, 0.08, 0.08, 0.08) for name in ("c500", "c550", "c600")}
    payloads = _matrix(values)
    with pytest.raises(ValueError):
        protocol.final_verdict(payloads[:-1])
    with pytest.raises(ValueError):
        protocol.final_verdict(payloads[:-1] + [payloads[0]])
    bad = dict(payloads[0])
    bad["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        protocol.validate_cell(bad)


def test_runner_preserves_fixed_density_speed_and_lateral_mode():
    source_text = Path(__file__).with_name(
        "run_sa5_v3_c50_stage3_b3_c500_c600_screen_cell.py"
    ).read_text(encoding="utf-8")
    assert "protocol.speed_range_for(scenario)" in source_text
    assert "protocol.density_for(" in source_text
    assert 'corridor_motion_mode="lateral"' in source_text


def test_queue_fingerprints_all_checkpoints_and_has_no_training_launch():
    fingerprint = queue.base_queue.source_fingerprint()
    for spec in protocol.CANDIDATE_SPECS:
        key = str(protocol.checkpoint_path(spec).relative_to(protocol.REPO))
        assert fingerprint["files"][key] == spec["expected_sha256"]
    queue_text = Path(queue.__file__).read_text(encoding="utf-8")
    assert "train_rnn_car_wdclip" not in queue_text
    assert "systemd-run" not in queue_text
