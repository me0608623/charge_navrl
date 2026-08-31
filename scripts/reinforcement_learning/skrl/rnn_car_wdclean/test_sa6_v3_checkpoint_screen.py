"""CPU contracts for the frozen SA6-v3 two-phase checkpoint screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import run_sa6_v3_checkpoint_screen_cell as runner  # noqa: E402
import sa6_v3_checkpoint_screen as screen  # noqa: E402
import sa6_v3_checkpoint_screen_queue as queue  # noqa: E402


FREEZE = REPO / "docs/freeze/sa6_v3_checkpoint_screen_v3.json"
RUNNER = HERE / "run_sa6_v3_checkpoint_screen_cell.py"
QUEUE = HERE / "sa6_v3_checkpoint_screen_queue.py"


def _phase_a_cell(
    checkpoint_name: str,
    fixed_cell: str,
    *,
    sr: float,
    cr: float,
    to: float,
    valid: bool = True,
) -> dict:
    candidate = screen.candidate_by_name(checkpoint_name)
    spec = screen.phase_a_spec_by_label(fixed_cell)
    metrics = {"n": 2000, "sr": sr, "cr": cr, "to": to}
    return {
        "schema": screen.PHASE_A_CELL_SCHEMA,
        "cell_valid": valid,
        "protocol_sha256": screen.screen_protocol()["sha256"],
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": candidate["sha256"],
        "cell": screen.phase_a_cell_label(checkpoint_name, fixed_cell),
        "fixed_cell": fixed_cell,
        "cell_kind": spec["cell_kind"],
        "stage": screen.STAGE,
        "seed": screen.SEED,
        "delay_steps": screen.DELAY_STEPS,
        "speed_rate": screen.SPEED_RATE,
        "metrics": metrics,
        "gate": screen.evaluate_metrics("corridor", metrics),
        "runtime": {
            "static_obstacles": spec["static_obstacles"],
            "dynamic_obstacles": spec["dynamic_obstacles"],
            "pedestrian_speed_range_m_s": list(spec["speed_range_m_s"]),
            "motion_mode": spec["motion_mode"],
            "installation": spec["installation"],
        },
    }


def _phase_a_payloads(
    scores: dict[str, tuple[float, float, float]],
) -> list[dict]:
    payloads = []
    for candidate in screen.CANDIDATES:
        sr, cr, to = scores[candidate["name"]]
        for spec in screen.PHASE_A_SPECS:
            payloads.append(
                _phase_a_cell(
                    candidate["name"],
                    spec["label"],
                    sr=sr,
                    cr=cr,
                    to=to,
                )
            )
    return payloads


def _phase_b_cell(checkpoint_name: str, scenario: str) -> dict:
    candidate = screen.candidate_by_name(checkpoint_name)
    metrics = {"n": 2000, "sr": 0.98, "cr": 0.02, "to": 0.0}
    if scenario == "narrow_range":
        metrics.update({"crossing_rate": 1.0, "direct_crossing_rate": 1.0})
    return {
        "schema": screen.PHASE_B_CELL_SCHEMA,
        "cell_valid": True,
        "protocol_sha256": screen.screen_protocol()["sha256"],
        "checkpoint_name": checkpoint_name,
        "checkpoint_sha256": candidate["sha256"],
        "cell": screen.phase_b_cell_label(checkpoint_name, scenario),
        "scenario": scenario,
        "stage": screen.STAGE,
        "seed": screen.SEED,
        "delay_steps": screen.DELAY_STEPS,
        "speed_rate": screen.SPEED_RATE,
        "metrics": metrics,
        "gate": screen.evaluate_metrics(scenario, metrics),
    }


def test_frozen_protocol_matches_runtime():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == screen.screen_protocol()


def test_matrix_separates_target_profiles_from_family_regressions():
    protocol = screen.screen_protocol()
    assert [row["name"] for row in protocol["candidates"]] == [
        "c50",
        "c100",
        "c150",
        "c200",
        "c250",
        "c300",
        "c350",
    ]
    assert [row["label"] for row in screen.PHASE_A_SPECS] == [
        "profile_sa6_3s2d",
        "profile_sa6_4s2d",
        "profile_sa6_4s3d",
        "family_lateral_4s2d",
        "family_longitudinal_4s2d",
        "family_random_2d_4s2d",
        "family_mixed_4s2d",
    ]
    assert screen.CORRIDOR_FAMILIES == (
        "lateral",
        "longitudinal",
        "random_2d",
        "mixed",
    )
    assert len(screen.phase_a_cells()) == 49
    assert protocol["phase_a"]["cells"] == 49
    assert all(
        not (
            row["dynamic_obstacles"] == 3
            and row["motion_mode"] == "lateral"
        )
        for row in screen.PHASE_A_SPECS
    )
    assert screen.PHASE_B_SCENARIOS == (
        "nav_native",
        "narrow_range",
        "retention_p060_0s1d",
        "retention_p060_1s1d",
        "retention_p060_2s1d",
    )
    assert protocol["phase_b"]["cells"] == 10


def test_r3_uniformly_raises_every_phase_a_checkpoint_budget():
    planned = queue.planned_phase_a_cells()
    by_cell = {
        (row["checkpoint_name"], row["fixed_cell"]): row["steps"]
        for row in planned
    }
    for candidate in screen.CANDIDATES:
        name = candidate["name"]
        for spec in screen.PHASE_A_SPECS:
            assert by_cell[(name, spec["label"])] == 5000
    assert screen.phase_b_steps("nav_native") == 2500
    assert screen.phase_b_steps("retention_p060_2s1d") == 2500


def test_r3_records_r2_sample_shortfall_without_changing_gate_threshold():
    protocol = screen.screen_protocol()
    assert protocol["schema"] == "sa6_v3_checkpoint_screen_protocol/v3"
    correction = protocol["fixed_evaluation"]["r2_correction"]
    assert correction["predecessor_status"] == "INCOMPLETE_NO_VERDICT"
    assert correction["failed_cell"] == "c150_profile_sa6_4s2d"
    assert correction["completed_episodes"] == 912
    assert correction["minimum_completed_episodes"] == 1000
    assert protocol["fixed_evaluation"]["minimum_completed_episodes"] == 1000


def test_fixed_runtime_contract_matches_sa6_training_contract():
    fixed = screen.screen_protocol()["fixed_evaluation"]
    assert fixed["stage"] == 6
    assert fixed["seed"] == 818
    assert fixed["actuator_delay_steps"] == 1
    assert fixed["speed_rate"] == 0.7
    assert fixed["speed_rate_obs"] == "ego"
    assert fixed["lidar_noise_mode"] == "full"
    assert fixed["lidar_distractor_eligibility"] == "valid_return_only"
    assert fixed["scene"]["arena_size_m"] == 14.0
    assert fixed["scene"]["corridor_free_width_m"] == 4.0
    assert fixed["scene"]["corridor_wall_span_m"] == 14.0
    assert fixed["scene"]["corridor_sealed_to_boundary"] is True
    assert fixed["corridor_installation"]["profile_cells"].startswith(
        "single-entry production count-mix"
    )
    assert fixed["corridor_installation"]["family_cells"].startswith(
        "legacy fixed-count 4S2D"
    )


def test_runner_separates_profile_and_family_installation_paths(tmp_path):
    profile_spec = screen.phase_a_spec_by_label("profile_sa6_4s3d")
    profile_args = runner.build_phase_a_args(
        profile_spec, tmp_path / "profile.json"
    )
    mix_index = profile_args.index("--long_corridor_count_mix") + 1
    motion_index = profile_args.index("--long_corridor_motion_mode") + 1
    assert profile_args[mix_index] == "4,3:1.0"
    assert profile_args[motion_index] == "env_stratified"

    family_spec = screen.phase_a_spec_by_label("family_lateral_4s2d")
    family_args = runner.build_phase_a_args(
        family_spec, tmp_path / "family.json"
    )
    assert "--long_corridor_count_mix" not in family_args
    motion_index = family_args.index("--long_corridor_motion_mode") + 1
    assert family_args[motion_index] == "lateral"


def test_ranking_prevents_collision_to_timeout_gaming():
    scores = {
        "c50": (0.91, 0.09, 0.00),
        "c100": (0.91, 0.01, 0.08),
        "c150": (0.89, 0.07, 0.04),
        "c200": (0.88, 0.08, 0.04),
        "c250": (0.87, 0.08, 0.05),
        "c300": (0.86, 0.09, 0.05),
        "c350": (0.85, 0.10, 0.05),
    }
    ranked = screen.rank_phase_a(_phase_a_payloads(scores))
    assert ranked["ranked"][:2] == ["c50", "c100"]
    first = ranked["summaries_by_checkpoint"]["c50"]
    second = ranked["summaries_by_checkpoint"]["c100"]
    assert first["worst_failure_rate"] == pytest.approx(0.09)
    assert second["worst_failure_rate"] == pytest.approx(0.09)
    assert first["worst_cr"] > second["worst_cr"]
    # Equal failure is resolved by CR only after failure, not by hiding timeout.
    assert ranked["ranked"][0] == "c50"


def test_missing_duplicate_or_invalid_phase_a_cell_fails_closed():
    payloads = _phase_a_payloads(
        {candidate["name"]: (0.95, 0.05, 0.0) for candidate in screen.CANDIDATES}
    )
    with pytest.raises(ValueError):
        screen.rank_phase_a(payloads[:-1])
    with pytest.raises(ValueError):
        screen.rank_phase_a([*payloads[:-1], payloads[0]])
    payloads[0]["cell_valid"] = False
    with pytest.raises(ValueError):
        screen.rank_phase_a(payloads)


def test_phase_b_folds_only_top_two_and_never_accepts_parent():
    phase_a = _phase_a_payloads(
        {
            "c50": (0.98, 0.02, 0.0),
            "c100": (0.97, 0.03, 0.0),
            "c150": (0.96, 0.04, 0.0),
            "c200": (0.95, 0.05, 0.0),
            "c250": (0.94, 0.06, 0.0),
            "c300": (0.93, 0.07, 0.0),
            "c350": (0.92, 0.08, 0.0),
        }
    )
    phase_b = [
        _phase_b_cell(name, scenario)
        for name in ("c50", "c100")
        for scenario in screen.PHASE_B_SCENARIOS
    ]
    verdict = screen.final_verdict(phase_a, phase_b)
    assert verdict["phase_a"]["top_two"] == ["c50", "c100"]
    assert verdict["recommended_for_human_consideration"] == "c50"
    assert verdict["accepted_parent"] is False
    assert verdict["training_started"] is False
    assert verdict["sa7_started"] is False


def test_runner_pins_every_measurement_defining_argument():
    source = RUNNER.read_text(encoding="utf-8")
    for marker in (
        '"--vlp16_noise_mode"',
        '"--lidar-distractor-eligibility"',
        '"--speed_rate"',
        '"--speed_rate_obs"',
        '"--deployment_speed_scale"',
        '"--long_corridor_free_width"',
        '"--long_corridor_dynamic_speed_range"',
        '"--long_corridor_static_obstacles"',
        '"--long_corridor_dynamic_obstacles"',
        '"--long_corridor_count_mix"',
        '"--long_corridor_motion_mode"',
        "verify_fixed_actuator_runtime",
        "reconcile_corridor_metrics",
        '"sealed_to_boundary_pass"',
    ):
        assert marker in source


def test_queue_is_fail_closed_and_has_no_training_or_sa7_path():
    source = QUEUE.read_text(encoding="utf-8")
    assert "INCOMPLETE_NO_VERDICT" in source
    assert "source fingerprint" in source
    assert "verify_checkpoints()" in source
    assert "planned_phase_b_cells" in source
    assert "train_rnn_car_wdclip.py" not in source
    assert "systemctl" not in source
    assert "subprocess.Popen" not in source
    assert '"sa7_started": False' in source


def test_shared_gpu_mode_keeps_memory_and_ram_floors(monkeypatch):
    outputs = iter(
        [
            type(
                "Result",
                (),
                {"returncode": 0, "stdout": "16900, 90, 61\n", "stderr": ""},
            )(),
            type(
                "Result",
                (),
                {
                    "returncode": 0,
                    "stdout": "3214308, /home/cm/yolo/python3, 14612\n",
                    "stderr": "",
                },
            )(),
        ]
    )
    monkeypatch.setattr(queue.subprocess, "run", lambda *args, **kwargs: next(outputs))
    monkeypatch.setattr(queue, "_mem_available_gib", lambda: 40.0)
    snapshot = queue.resource_snapshot(allow_shared_gpu=True)
    assert snapshot["ready"] is True
    assert snapshot["requirements"]["min_gpu_free_mib"] == 12000
    assert snapshot["requirements"]["min_available_ram_gib"] == 12.0
