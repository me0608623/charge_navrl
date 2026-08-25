"""Pure contract and reducers for the SA5-R2 fixed speed-0.7 screen."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import sa5_r2_checkpoint_screen as base_screen


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

SPEED_RATE = 0.7
SPEED_RATE_OBS = "ego"
DEPLOYMENT_SPEED_SCALE = 1.0
TIE_MARGIN = 0.005
RETENTION_TOLERANCE = 0.02
POSTHOC_REFERENCE_EPISODES = 1200
POSTHOC_EFFECT_SE_THRESHOLD = 2.0
V2_PROTOCOL_SHA256 = "ca68fed6835e64ced6976199ea0fd47ab1292a746c55f84e541fe211cdf9f1e2"

CORRIDOR_SCENARIOS = base_screen.CORRIDOR_SCENARIOS
RETENTION_SCENARIOS = ("nav_native", "narrow_range")
SCENARIOS = CORRIDOR_SCENARIOS + RETENTION_SCENARIOS
STEP_TIERS_BY_SCENARIO = {
    "corridor_lateral": (2500, 5000, 7500),
    "corridor_longitudinal": (2500, 5000, 7500),
    "corridor_random2d": (5000, 7500),
    "corridor_mixed": (5000, 7500),
    "nav_native": (1200, 2400, 3600),
    "narrow_range": (1200, 2400, 3600),
}
STEPS_BY_SCENARIO = {
    scenario: tiers[0] for scenario, tiers in STEP_TIERS_BY_SCENARIO.items()
}

BASE_RUN = base_screen.RUN_NAME
ADAPT_RUN = "sa5_r2_c250_speed0p7_adapt_ne1024_s42_p50_r1"
CANDIDATES = (
    {
        "name": "baseline_c250",
        "role": "rate1_trained_c250_baseline",
        "conceptual_iteration": 250,
        "run_name": BASE_RUN,
        "filename": "checkpoint_32000.pt",
        "sha256": "df14c45d8dd27b0e327f9447a9a613bf62eaa850af542b474aad682dcd937a4a",
    },
    {
        "name": "control_c300",
        "role": "natural_rate1_plus50_control",
        "conceptual_iteration": 300,
        "run_name": BASE_RUN,
        "filename": "checkpoint_38400.pt",
        "sha256": "97a4ccdf929a0fcab9adff5d80ae7a1c7475067e3049f11571e04fd164a524db",
    },
    {
        "name": "adapt_it25",
        "role": "fixed_rate0p7_adaptation_it25",
        "conceptual_iteration": 275,
        "run_name": ADAPT_RUN,
        "filename": "checkpoint_3200.pt",
        "sha256": "e38e7c504316e9a0b7f6799325bc719966a7b896c00b9e562081eae5fa322f77",
    },
    {
        "name": "adapt_it50",
        "role": "fixed_rate0p7_adaptation_it50",
        "conceptual_iteration": 300,
        "run_name": ADAPT_RUN,
        "filename": "checkpoint_6400.pt",
        "sha256": "812a495902f7c54c638096ba113b1b44266ddcc41431d476e6919302ee136f1c",
    },
)


def candidate_by_name(name: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return dict(candidate)
    raise ValueError(f"unknown candidate {name!r}")


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    return REPO / "logs/rnn_car" / data["run_name"] / data["filename"]


def screen_protocol() -> dict:
    parent = base_screen.screen_protocol()
    payload = {
        "schema": "sa5_r2_speed0p7_checkpoint_screen_protocol/v3",
        "purpose": (
            "compare c250 baseline, natural c300 control, speed-0.7 adaptation "
            "it25, and adaptation it50 under one fixed speed-0.7 exam"
        ),
        "candidates": [
            {**dict(candidate), "path": str(checkpoint_path(candidate).resolve())}
            for candidate in CANDIDATES
        ],
        "fixed_evaluation": {
            "stage": base_screen.STAGE,
            "seed": base_screen.SEED,
            "num_envs": base_screen.NUM_ENVS,
            "scenarios": list(SCENARIOS),
            "steps_by_scenario": dict(STEPS_BY_SCENARIO),
            "step_tiers_by_scenario": {
                scenario: list(tiers)
                for scenario, tiers in STEP_TIERS_BY_SCENARIO.items()
            },
            "minimum_completed_episodes": base_screen.MIN_EPISODES,
            "sample_size_stopping_rule": (
                "run the first frozen step tier; only when completed episodes "
                "are below 1000, discard that attempt and rerun from scratch "
                "at the next frozen tier; SR/CR/TO never select a tier"
            ),
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "lidar_scaled": False,
            "actuator_delay_steps": base_screen.DELAY_STEPS,
            "actuator_delay_ms": base_screen.DELAY_MS,
            "actuator_profile": base_screen.ACTUATOR_PROFILE,
            "lidar_noise_mode": base_screen.LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": (
                base_screen.LIDAR_DISTRACTOR_ELIGIBILITY
            ),
            "scene": parent["fixed_evaluation"]["scene"],
            "corridor_density": "4S2D",
        },
        "ranking": {
            "primary": "min max(CR across four corridor families)",
            "secondary": "min episode-unweighted mean corridor-family CR",
            "clear_improvement_margin": TIE_MARGIN,
            "later_checkpoint_preference": False,
        },
        "retention": {
            "scenarios": list(RETENTION_SCENARIOS),
            "absolute_thresholds": {
                "nav_native": base_screen.NATIVE_THRESHOLDS,
                "narrow_range": base_screen.NARROW_THRESHOLDS,
            },
            "relative_reference": "baseline_c250 at speed_rate=0.7",
            "maximum_regression": RETENTION_TOLERANCE,
            "rule": (
                "pass absolute hard gates and remain within 2 pp of baseline "
                "for SR, CR, TO, and narrow crossing metrics"
            ),
        },
        "decision_rules": [
            (
                "extend adapt_it50 by 25-50 iterations only if its worst-family "
                "CR beats every comparator by at least 0.005 and retention passes"
            ),
            (
                "if adapt_it25 beats adapt_it50 by at least 0.005, stop because "
                "the second half shows late forgetting"
            ),
            (
                "if neither adaptation checkpoint beats both rate-1-trained "
                "controls by at least 0.005 with retention, move the speed "
                "contract earlier to the SA3-to-SA4 boundary"
            ),
            "otherwise do not extend because the result is inconclusive",
        ],
        "cells": len(CANDIDATES) * len(SCENARIOS),
        "evidence_boundary": (
            "single evaluator seed fixed screen; selects the next engineering "
            "action but does not graduate SA5, accept a parent, start training, "
            "or authorize SA6"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _finite(value, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def validate_cell(payload: dict, *, require_min_episodes: bool = True) -> None:
    if payload.get("schema") != "sa5_r2_speed0p7_checkpoint_screen_cell/v3":
        raise ValueError("unexpected cell schema")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("cell protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    scenario = str(payload.get("scenario"))
    if scenario not in SCENARIOS:
        raise ValueError("invalid cell scenario")
    tiers = STEP_TIERS_BY_SCENARIO[scenario]
    tier_index = int(payload.get("step_tier_index", -1))
    if tier_index not in range(len(tiers)):
        raise ValueError("invalid cell step tier index")
    if int(payload.get("steps", -1)) != tiers[tier_index]:
        raise ValueError("cell steps differ from frozen tier")
    if _finite(payload.get("speed_rate"), "speed rate") != SPEED_RATE:
        raise ValueError("cell speed rate mismatch")
    if payload.get("speed_rate_obs") != SPEED_RATE_OBS:
        raise ValueError("cell speed-rate observation mode mismatch")
    if _finite(payload.get("deployment_speed_scale"), "deployment scale") != 1.0:
        raise ValueError("cell stacked downstream speed scaling")
    metrics = payload.get("metrics") or {}
    enough_episodes = int(metrics.get("n", 0)) >= base_screen.MIN_EPISODES
    if bool(payload.get("cell_valid")) != enough_episodes:
        raise ValueError("cell validity does not match completed episode count")
    if require_min_episodes and not enough_episodes:
        raise ValueError("cell has fewer than the frozen minimum episodes")
    for key in ("sr", "cr", "to"):
        _finite(metrics.get(key), f"{scenario} {key}")
    expected_gate = base_screen.evaluate_metrics(scenario, metrics)
    if payload.get("outcome_gate") != expected_gate:
        raise ValueError("cell outcome gate does not reconcile")


def migrate_v2_cell(
    payload: dict,
    *,
    origin_path: str,
    origin_cell_sha256: str,
    origin_source_fingerprint_sha256: str,
) -> dict:
    """Wrap a valid v2 first-tier cell in explicit v3 provenance."""

    if payload.get("schema") != "sa5_r2_speed0p7_checkpoint_screen_cell/v2":
        raise ValueError("resume cell is not a v2 cell")
    if payload.get("protocol_sha256") != V2_PROTOCOL_SHA256:
        raise ValueError("resume cell has the wrong v2 protocol hash")
    scenario = str(payload.get("scenario"))
    if scenario not in SCENARIOS:
        raise ValueError("resume cell scenario is unknown")
    if int(payload.get("steps", -1)) != STEP_TIERS_BY_SCENARIO[scenario][0]:
        raise ValueError("resume cell is not a v3 first-tier measurement")
    if int((payload.get("metrics") or {}).get("n", 0)) < base_screen.MIN_EPISODES:
        raise ValueError("resume cell did not satisfy the evidence floor")

    migrated = dict(payload)
    migrated.update(
        {
            "schema": "sa5_r2_speed0p7_checkpoint_screen_cell/v3",
            "protocol_sha256": screen_protocol()["sha256"],
            "step_tier_index": 0,
            "measurement_provenance": {
                "mode": "imported_v2_first_tier",
                "origin_protocol_sha256": V2_PROTOCOL_SHA256,
                "origin_path": origin_path,
                "origin_cell_sha256": origin_cell_sha256,
                "origin_source_fingerprint_sha256": (
                    origin_source_fingerprint_sha256
                ),
            },
        }
    )
    validate_cell(migrated)
    return migrated


def _clear_better(left: dict, right: dict) -> bool:
    return (
        float(left["worst_family_cr"])
        <= float(right["worst_family_cr"]) - TIE_MARGIN
    )


def _conservative_worst_cr_comparison(left: dict, right: dict) -> dict:
    """Describe a worst-family CR difference using a fixed conservative n.

    All completed worst-family cells in the final screen have more than 1,200
    episodes.  Using n=1,200 for both arms gives a slightly larger binomial
    standard error than using their unequal observed counts.  This is a
    post-screen evidence interpretation, not a replacement for the frozen
    ranking rule or a multi-seed significance test.
    """

    left_cr = float(left["worst_family_cr"])
    right_cr = float(right["worst_family_cr"])
    variance = (
        left_cr * (1.0 - left_cr) / POSTHOC_REFERENCE_EPISODES
        + right_cr * (1.0 - right_cr) / POSTHOC_REFERENCE_EPISODES
    )
    standard_error = math.sqrt(variance)
    delta = left_cr - right_cr
    standardized_difference = delta / standard_error
    return {
        "left": left["checkpoint_name"],
        "right": right["checkpoint_name"],
        "delta_cr": delta,
        "standard_error": standard_error,
        "standardized_difference_se": standardized_difference,
        "reference_episodes_per_arm": POSTHOC_REFERENCE_EPISODES,
        "at_least_2se_worse": (
            standardized_difference >= POSTHOC_EFFECT_SE_THRESHOLD
        ),
        "at_least_2se_better": (
            standardized_difference <= -POSTHOC_EFFECT_SE_THRESHOLD
        ),
    }


def _relative_retention_checks(
    candidate_cells: dict[str, dict], baseline_cells: dict[str, dict]
) -> dict:
    checks = {}
    for scenario in RETENTION_SCENARIOS:
        metrics = candidate_cells[scenario]["metrics"]
        baseline = baseline_cells[scenario]["metrics"]
        scenario_checks = {
            "sr": float(metrics["sr"]) >= float(baseline["sr"]) - RETENTION_TOLERANCE,
            "cr": float(metrics["cr"]) <= float(baseline["cr"]) + RETENTION_TOLERANCE,
            "to": float(metrics["to"]) <= float(baseline["to"]) + RETENTION_TOLERANCE,
        }
        if scenario == "narrow_range":
            scenario_checks.update(
                {
                    "crossing_rate": float(metrics["crossing_rate"])
                    >= float(baseline["crossing_rate"]) - RETENTION_TOLERANCE,
                    "direct_crossing_rate": float(metrics["direct_crossing_rate"])
                    >= float(baseline["direct_crossing_rate"])
                    - RETENTION_TOLERANCE,
                }
            )
        checks[scenario] = scenario_checks
    return checks


def summarize(payloads: Iterable[dict]) -> dict:
    cells = list(payloads)
    expected = len(CANDIDATES) * len(SCENARIOS)
    if len(cells) != expected:
        raise ValueError(f"screen needs exactly {expected} cells, got {len(cells)}")
    keyed: dict[str, dict[str, dict]] = {}
    for cell in cells:
        validate_cell(cell)
        name = cell["checkpoint_name"]
        scenario = cell["scenario"]
        if scenario in keyed.setdefault(name, {}):
            raise ValueError(f"duplicate cell {name}/{scenario}")
        keyed[name][scenario] = cell
    if set(keyed) != {candidate["name"] for candidate in CANDIDATES}:
        raise ValueError("candidate set mismatch")
    for name, by_scenario in keyed.items():
        if set(by_scenario) != set(SCENARIOS):
            raise ValueError(f"scenario set mismatch for {name}")

    baseline_cells = keyed["baseline_c250"]
    rows = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        by_scenario = keyed[name]
        family_cr = {
            scenario: float(by_scenario[scenario]["metrics"]["cr"])
            for scenario in CORRIDOR_SCENARIOS
        }
        worst_family = max(family_cr, key=family_cr.get)
        relative_checks = _relative_retention_checks(
            by_scenario, baseline_cells
        )
        hard_retention_pass = all(
            bool(by_scenario[scenario]["outcome_gate"]["threshold_pass"])
            for scenario in RETENTION_SCENARIOS
        )
        relative_retention_pass = all(
            passed
            for scenario_checks in relative_checks.values()
            for passed in scenario_checks.values()
        )
        rows.append(
            {
                "checkpoint_name": name,
                "role": candidate["role"],
                "conceptual_iteration": candidate["conceptual_iteration"],
                "checkpoint_sha256": candidate["sha256"],
                "family_cr": family_cr,
                "family_sr": {
                    scenario: float(by_scenario[scenario]["metrics"]["sr"])
                    for scenario in CORRIDOR_SCENARIOS
                },
                "family_to": {
                    scenario: float(by_scenario[scenario]["metrics"]["to"])
                    for scenario in CORRIDOR_SCENARIOS
                },
                "family_episodes": {
                    scenario: int(by_scenario[scenario]["metrics"]["n"])
                    for scenario in CORRIDOR_SCENARIOS
                },
                "worst_family": worst_family,
                "worst_family_cr": family_cr[worst_family],
                "mean_family_cr": sum(family_cr.values()) / len(family_cr),
                "native": by_scenario["nav_native"]["metrics"],
                "narrow": by_scenario["narrow_range"]["metrics"],
                "hard_retention_pass": hard_retention_pass,
                "relative_retention_checks": relative_checks,
                "relative_retention_pass": relative_retention_pass,
                "retention_acceptable": (
                    hard_retention_pass and relative_retention_pass
                ),
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (row["worst_family_cr"], row["mean_family_cr"]),
    )
    by_name = {row["checkpoint_name"]: row for row in rows}
    it25 = by_name["adapt_it25"]
    it50 = by_name["adapt_it50"]
    controls = [by_name["baseline_c250"], by_name["control_c300"]]
    it50_beats_every_comparator = all(
        _clear_better(it50, other)
        for other in (it25, *controls)
    )
    it25_beats_it50 = _clear_better(it25, it50)
    adaptation_beats_both_controls = {
        row["checkpoint_name"]: (
            row["retention_acceptable"]
            and all(_clear_better(row, control) for control in controls)
        )
        for row in (it25, it50)
    }

    if it50_beats_every_comparator and it50["retention_acceptable"]:
        preregistered_decision = "EXTEND_ADAPT_IT50_BY_25_TO_50"
        preregistered_reason = (
            "adapt_it50 is the clear worst-family leader with retention"
        )
    elif it25_beats_it50:
        preregistered_decision = "STOP_IT50_LATE_FORGETTING"
        preregistered_reason = (
            "adapt_it25 clears the frozen 0.005 raw worst-family margin over "
            "adapt_it50"
        )
    elif not any(adaptation_beats_both_controls.values()):
        preregistered_decision = "MOVE_SPEED_CONTRACT_EARLIER_TO_SA3_SA4"
        preregistered_reason = (
            "neither adaptation checkpoint clearly beats both controls with retention"
        )
    else:
        preregistered_decision = "INCONCLUSIVE_DO_NOT_EXTEND"
        preregistered_reason = (
            "the preregistered clear-leader rules were not satisfied"
        )

    comparisons = {
        "control_vs_baseline": _conservative_worst_cr_comparison(
            by_name["control_c300"], by_name["baseline_c250"]
        ),
        "adapt_it25_vs_baseline": _conservative_worst_cr_comparison(
            it25, by_name["baseline_c250"]
        ),
        "adapt_it50_vs_baseline": _conservative_worst_cr_comparison(
            it50, by_name["baseline_c250"]
        ),
        "adapt_it50_vs_adapt_it25": _conservative_worst_cr_comparison(
            it50, it25
        ),
    }
    it25_improved_over_baseline = comparisons[
        "adapt_it25_vs_baseline"
    ]["at_least_2se_better"]
    it50_degraded_vs_baseline = comparisons[
        "adapt_it50_vs_baseline"
    ]["at_least_2se_worse"]
    it50_degraded_vs_it25 = comparisons[
        "adapt_it50_vs_adapt_it25"
    ]["at_least_2se_worse"]

    if it50_beats_every_comparator and it50["retention_acceptable"]:
        decision = "EXTEND_ADAPT_IT50_BY_25_TO_50"
        reason = "adapt_it50 is the worst-family leader with retention"
    elif it25_improved_over_baseline and it50_degraded_vs_it25:
        decision = "STOP_IT50_LATE_REGRESSION_SUPPORTED"
        reason = (
            "adapt_it25 first improves on baseline and adapt_it50 then becomes "
            "at least 2 conservative SE worse than adapt_it25"
        )
    elif it50_degraded_vs_baseline and not it25_improved_over_baseline:
        decision = "ADAPTATION_DEGRADED_NO_IMPROVEMENT_AT_ANY_CHECKPOINT"
        reason = (
            "adapt_it50 is at least 2 conservative SE worse than baseline, "
            "while adapt_it25 never demonstrates a 2-SE improvement"
        )
    elif not any(adaptation_beats_both_controls.values()):
        decision = "MOVE_SPEED_CONTRACT_EARLIER_TO_SA3_SA4"
        reason = (
            "neither adaptation checkpoint beats both controls with retention"
        )
    else:
        decision = "INCONCLUSIVE_DO_NOT_EXTEND"
        reason = (
            "raw ranking differences do not support a 2-SE evidence claim"
        )

    return {
        "schema": "sa5_r2_speed0p7_checkpoint_screen_summary/v2",
        "status": "COMPLETE_VALID_SINGLE_EVALUATOR_SEED_SCREEN",
        "protocol_sha256": screen_protocol()["sha256"],
        "rows": rows,
        "ranked": [row["checkpoint_name"] for row in ranked],
        "leader": ranked[0]["checkpoint_name"],
        "leader_tie_group": [
            row["checkpoint_name"]
            for row in ranked
            if row["worst_family_cr"] - ranked[0]["worst_family_cr"]
            < TIE_MARGIN
        ],
        "adaptation_beats_both_controls": adaptation_beats_both_controls,
        "preregistered_rule_outcome": {
            "decision": preregistered_decision,
            "reason": preregistered_reason,
            "preserved_for_audit": True,
        },
        "posthoc_standardized_comparisons": {
            "method": (
                "independent binomial SE with n=1200 per arm; descriptive "
                "single-evaluator-seed evidence, not a multi-seed test"
            ),
            "effect_threshold_se": POSTHOC_EFFECT_SE_THRESHOLD,
            "comparisons": comparisons,
        },
        "decision": decision,
        "decision_reason": reason,
        "engineering_action": (
            "STOP_SPEED0P7_ADAPTATION_AND_DEFINE_THE_SPEED_RATIO_CONTRACT_"
            "BEFORE_REBUILDING_SA3_TO_SA4"
        ),
        "accepted_parent": False,
        "training_started": False,
        "sa6_started": False,
    }
