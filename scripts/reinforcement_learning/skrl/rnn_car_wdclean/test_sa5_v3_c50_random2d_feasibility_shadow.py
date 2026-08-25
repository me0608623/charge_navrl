"""CPU contracts for the c50 4S2D random-2D feasibility audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import d3_yield_recorder as d3
import run_sa5_v3_c50_random2d_feasibility_shadow as runner
import sa5_v3_c50_random2d_feasibility_shadow as protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FREEZE = (
    REPO
    / "docs/freeze/sa5_v3_c50_random2d_feasibility_shadow_v1.json"
)


def _report(*, feasible_seen: float, no_feasible_entire: float) -> dict:
    collision = {
        "events": 200,
        "feasible_seen_in_window_fraction": feasible_seen,
        "no_feasible_entire_window_fraction": no_feasible_entire,
        "event_no_feasible_fraction": 0.9,
        "persistent_collapse_lead_s": {
            "p10": 0.8,
            "p25": 1.0,
            "p50": 1.4,
            "p75": 1.8,
            "p90": 2.0,
        },
    }
    control = {
        "events": 75,
        "feasible_seen_in_window_fraction": 1.0,
        "no_feasible_entire_window_fraction": 0.0,
        "event_no_feasible_fraction": 0.1,
    }
    return {
        "schema": "sa4_d5_feasibility_frontier/v1",
        "mode": "feasibility_shadow",
        "counts": {
            "dynamic_collision": 200,
            "successful_noncollision_closest_approach": 75,
        },
        "self_check": {"reconciliation_ok": True},
        "frontier_summary": {
            "dynamic_collision": collision,
            "successful_noncollision_closest_approach": control,
        },
        "metadata": {"shadow_runtime": {"environment_frames": 160000}},
    }


def test_checked_in_protocol_matches_generated_contract():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == (
        protocol.protocol_payload()
    )


def test_protocol_locks_lineage_and_forbids_training_and_sa6():
    frozen = protocol.protocol_payload()
    assert frozen["checkpoint"]["sha256"] == protocol.CHECKPOINT["sha256"]
    assert frozen["fixed_cell"]["static_obstacles"] == 4
    assert frozen["fixed_cell"]["dynamic_obstacles"] == 2
    assert frozen["fixed_cell"]["motion_mode"] == "random_2d"
    assert frozen["fixed_cell"]["speed_rate"] == pytest.approx(0.7)
    assert frozen["fixed_cell"]["actuator_delay_steps"] == 1
    assert "training in this audit" in frozen["forbidden"]
    assert "SA6 launch" in frozen["forbidden"]
    assert "physical inevitability" in frozen["interpretation_limit"]


def test_scene_args_carry_every_fixed_runtime_value(tmp_path: Path):
    args = runner.build_scene_args(tmp_path / "corridor.json")

    def value(flag: str, offset: int = 1) -> str:
        return args[args.index(flag) + offset]

    assert value("--stage") == "5"
    assert value("--seed") == "818"
    assert value("--vlp16_noise_mode") == "full"
    assert value("--lidar-distractor-eligibility") == "valid_return_only"
    assert value("--long_corridor_static_obstacles") == "4"
    assert value("--long_corridor_dynamic_obstacles") == "2"
    assert value("--long_corridor_motion_mode") == "random_2d"
    assert value("--long_corridor_random_2d_kinematics") == "patrol"
    assert value("--speed_rate") == "0.7"
    assert value("--speed_rate_obs") == "ego"
    assert value("--deployment_speed_scale") == "1"
    delay = args.index("--actuator_delay_range")
    assert args[delay + 1 : delay + 3] == ["1", "1"]


@pytest.mark.parametrize(
    ("feasible_seen", "no_feasible_entire", "expected"),
    [
        (0.95, 0.05, "SHORT_HORIZON_ACTION_OPPORTUNITY_OBSERVED"),
        (0.40, 0.60, "MOSTLY_NO_CANDIDATE_UNDER_FROZEN_MODEL"),
        (0.70, 0.30, "INDETERMINATE_REQUIRES_MORE_DIAGNOSTIC"),
    ],
)
def test_analysis_branches_are_frozen(
    feasible_seen: float, no_feasible_entire: float, expected: str
):
    result = protocol.analyze_report(
        _report(
            feasible_seen=feasible_seen,
            no_feasible_entire=no_feasible_entire,
        )
    )
    assert result["decision"] == expected
    assert result["training_started"] is False
    assert result["sa6_started"] is False


def test_analysis_fails_closed_on_weak_or_unreconciled_evidence():
    weak = _report(feasible_seen=1.0, no_feasible_entire=0.0)
    weak["counts"]["dynamic_collision"] = 99
    with pytest.raises(ValueError, match="too few dynamic-collision"):
        protocol.analyze_report(weak)

    broken = _report(feasible_seen=1.0, no_feasible_entire=0.0)
    broken["self_check"]["reconciliation_ok"] = False
    with pytest.raises(ValueError, match="reconciliation"):
        protocol.analyze_report(broken)


def test_runner_is_evaluation_only_and_refuses_overwrite():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert '"--d3_shield_mode"' in source
    assert '"baseline"' in source
    assert '"--d5_feasibility_shadow"' in source
    assert "source fingerprint changed during rollout" in source
    assert "require_dynamic_collision=not is_smoke" in source
    for forbidden in ("systemctl start", "train_rnn_car_wdclip.py"):
        assert forbidden not in source
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_require_new_targets" in calls


def test_nonlateral_d3_scope_opens_only_for_identity_d5_shadow():
    d3.validate_audit_motion_scope(
        "random_2d", "baseline", feasibility_shadow=True
    )
    with pytest.raises(ValueError, match="identity baseline"):
        d3.validate_audit_motion_scope(
            "random_2d", "baseline", feasibility_shadow=False
        )
    with pytest.raises(ValueError, match="intervention protocols"):
        d3.validate_audit_motion_scope(
            "random_2d", "geometry_feasible", feasibility_shadow=True
        )


def test_source_fingerprint_covers_contract_and_runtime_sources():
    paths = {path.resolve() for path in runner.source_paths()}
    assert Path(protocol.__file__).resolve() in paths
    assert FREEZE.resolve() in paths
    assert protocol.AUTHORIZATION.resolve() in paths
    assert runner.PLAY.resolve() in paths
    assert runner.ACTION_TERM.resolve() in paths
    assert all(path.is_file() for path in paths)
