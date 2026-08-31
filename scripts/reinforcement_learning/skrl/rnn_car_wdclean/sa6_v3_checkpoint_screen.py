"""Frozen two-phase checkpoint screen for the completed SA6-v3 lineage.

Phase A ranks the parent c50 and continuation c100-c350 on three exact SA6
training profiles plus four pure-family 4S2D regressions. Phase B evaluates
only the Phase-A top two on native, narrow, and the three retained P060
low-density profiles. This module is pure and never launches Isaac Sim or
training.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

PILOT_RUN_NAME = "sa6_v3_from_sa5_c600_ne1024_s42_p50_r1"
CONTINUATION_RUN_NAME = "sa6_v3_cont300_from_it50_ne1024_s42_p300_r1"
PILOT_RUN_DIR = REPO / "logs/rnn_car" / PILOT_RUN_NAME
CONTINUATION_RUN_DIR = REPO / "logs/rnn_car" / CONTINUATION_RUN_NAME

STAGE = 6
SEED = 818
NUM_ENVS = 64
PHASE_A_STEPS = 5000
PHASE_B_STEPS = 2500
MIN_EPISODES = 1000
DELAY_STEPS = 1
DELAY_MS = 200
ACTUATOR_PROFILE = "sa1_delay_only"
LIDAR_NOISE_MODE = "full"
LIDAR_DISTRACTOR_ELIGIBILITY = "valid_return_only"
SPEED_RATE = 0.7
SPEED_RATE_OBS = "ego"
DEPLOYMENT_SPEED_SCALE = 1.0
TIE_MARGIN = 0.005

PHASE_A_CELL_SCHEMA = "sa6_v3_checkpoint_screen_phase_a_cell/v1"
PHASE_B_CELL_SCHEMA = "sa6_v3_checkpoint_screen_phase_b_cell/v1"

CANDIDATES = (
    {
        "name": "c50",
        "conceptual_iteration": 50,
        "run_name": PILOT_RUN_NAME,
        "filename": "checkpoint_6400.pt",
        "sha256": "096a953132a4ec9a4208aa331448e83189ede6b35f40d77d5552d4e94909d65b",
    },
    {
        "name": "c100",
        "conceptual_iteration": 100,
        "run_name": CONTINUATION_RUN_NAME,
        "filename": "checkpoint_6400.pt",
        "sha256": "67c7e1131198740aad06261fe7fab86de1b2f4fc65146854200baaa2914986b2",
    },
    {
        "name": "c150",
        "conceptual_iteration": 150,
        "run_name": CONTINUATION_RUN_NAME,
        "filename": "checkpoint_12800.pt",
        "sha256": "4caab19a74a887fab81e6af0ed2d442cb7b1b97cf18509fb9fecaec8713570d8",
    },
    {
        "name": "c200",
        "conceptual_iteration": 200,
        "run_name": CONTINUATION_RUN_NAME,
        "filename": "checkpoint_19200.pt",
        "sha256": "6e027122e212dda79522f48afc4cd13204cd6d1b641a2ec682d07afcc01009dc",
    },
    {
        "name": "c250",
        "conceptual_iteration": 250,
        "run_name": CONTINUATION_RUN_NAME,
        "filename": "checkpoint_25600.pt",
        "sha256": "4044d6dff443b3d4a2a36dfa067a99acd0e60b167919caa8f9b89cf213de0803",
    },
    {
        "name": "c300",
        "conceptual_iteration": 300,
        "run_name": CONTINUATION_RUN_NAME,
        "filename": "checkpoint_32000.pt",
        "sha256": "f20c262442bb948ce2c259945ad9fb69d3bb2ded918d45af7df205f5fe624711",
    },
    {
        "name": "c350",
        "conceptual_iteration": 350,
        "run_name": CONTINUATION_RUN_NAME,
        "filename": "checkpoint_38400.pt",
        "sha256": "f8667eb91ba6f437bb455825c396c1ff0be74dc788f9ff390bcb8ae23624ed64",
    },
)


def _load_training_profiles() -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """Read the six profiles from the exact SA6 source config."""
    import sys

    skrl_root = HERE.parent
    if str(skrl_root) not in sys.path:
        sys.path.insert(0, str(skrl_root))
    from rnn_car_modular.configs.e2e_sa6_v3_from_sa5_c600_p50 import (
        SA6_SPEED_DENSITY_MIX,
    )

    if len(SA6_SPEED_DENSITY_MIX) != 6:
        raise RuntimeError("SA6 source config no longer contains six profiles")
    rows = []
    for index, (counts, speed, weight) in enumerate(SA6_SPEED_DENSITY_MIX):
        static_count, dynamic_count = (int(value) for value in counts)
        speed_range = tuple(float(value) for value in speed)
        speed_label = "p060" if speed_range == (0.50, 0.70) else "sa6"
        rows.append(
            {
                "label": f"{speed_label}_{static_count}s{dynamic_count}d",
                "density": f"{static_count}S{dynamic_count}D",
                "static_obstacles": static_count,
                "dynamic_obstacles": dynamic_count,
                "speed_range_m_s": speed_range,
                "training_weight": float(weight),
                "training_index": index,
            }
        )
    retention = tuple(rows[:3])
    target = tuple(rows[3:])
    expected_retention = ("p060_0s1d", "p060_1s1d", "p060_2s1d")
    expected_target = ("sa6_3s2d", "sa6_4s2d", "sa6_4s3d")
    if tuple(row["label"] for row in retention) != expected_retention:
        raise RuntimeError("SA6 low-density profile identity drifted")
    if tuple(row["label"] for row in target) != expected_target:
        raise RuntimeError("SA6 target profile identity drifted")
    return retention, target


RETENTION_PROFILES, TARGET_PROFILES = _load_training_profiles()
CORRIDOR_FAMILIES = ("lateral", "longitudinal", "random_2d", "mixed")
_FAMILY_REFERENCE_PROFILE = next(
    row for row in TARGET_PROFILES if row["label"] == "sa6_4s2d"
)
PROFILE_SPECS = tuple(
    {
        **profile,
        "label": f"profile_{profile['label']}",
        "source_profile": profile["label"],
        "cell_kind": "sa6_target_profile",
        "motion_mode": "env_stratified",
        "installation": "single-entry production count-mix",
    }
    for profile in TARGET_PROFILES
)
FAMILY_SPECS = tuple(
    {
        **_FAMILY_REFERENCE_PROFILE,
        "label": f"family_{family}_4s2d",
        "source_profile": _FAMILY_REFERENCE_PROFILE["label"],
        "cell_kind": "corridor_family_regression",
        "motion_mode": family,
        "installation": "legacy fixed-count 4S2D gate path",
    }
    for family in CORRIDOR_FAMILIES
)
PHASE_A_SPECS = PROFILE_SPECS + FAMILY_SPECS
PHASE_B_SCENARIOS = (
    "nav_native",
    "narrow_range",
    *(f"retention_{row['label']}" for row in RETENTION_PROFILES),
)


def _load_thresholds() -> tuple[dict, dict, dict]:
    import sys

    skrl_root = HERE.parent
    if str(skrl_root) not in sys.path:
        sys.path.insert(0, str(skrl_root))
    from rnn_car_wdclean.sa3_sa8_acceptance_contract import (
        CORRIDOR_THRESHOLDS,
        NARROW_THRESHOLDS,
        NATIVE_THRESHOLDS,
    )

    return (
        CORRIDOR_THRESHOLDS.to_dict(),
        NATIVE_THRESHOLDS[STAGE].to_dict(),
        NARROW_THRESHOLDS.to_dict(),
    )


CORRIDOR_THRESHOLDS, NATIVE_THRESHOLDS, NARROW_THRESHOLDS = _load_thresholds()


def candidate_by_name(name: str) -> dict:
    for candidate in CANDIDATES:
        if candidate["name"] == name:
            return dict(candidate)
    raise ValueError(f"unknown SA6-v3 candidate {name!r}")


def target_profile_by_label(label: str) -> dict:
    for profile in TARGET_PROFILES:
        if profile["label"] == label:
            return dict(profile)
    raise ValueError(f"unknown SA6 target profile {label!r}")


def phase_a_spec_by_label(label: str) -> dict:
    for spec in PHASE_A_SPECS:
        if spec["label"] == label:
            return dict(spec)
    raise ValueError(f"unknown SA6 Phase-A cell {label!r}")


def retention_profile_by_scenario(scenario: str) -> dict:
    prefix = "retention_"
    if not scenario.startswith(prefix):
        raise ValueError(f"not a low-density retention scenario: {scenario!r}")
    label = scenario.removeprefix(prefix)
    for profile in RETENTION_PROFILES:
        if profile["label"] == label:
            return dict(profile)
    raise ValueError(f"unknown SA6 retention profile {label!r}")


def checkpoint_path(candidate: dict | str) -> Path:
    data = candidate_by_name(candidate) if isinstance(candidate, str) else candidate
    run_dir = PILOT_RUN_DIR if data["run_name"] == PILOT_RUN_NAME else CONTINUATION_RUN_DIR
    return run_dir / str(data["filename"])


def phase_a_cell_label(checkpoint_name: str, fixed_cell: str) -> str:
    candidate_by_name(checkpoint_name)
    phase_a_spec_by_label(fixed_cell)
    return f"{checkpoint_name}_{fixed_cell}"


def phase_b_cell_label(checkpoint_name: str, scenario: str) -> str:
    candidate_by_name(checkpoint_name)
    if scenario not in PHASE_B_SCENARIOS:
        raise ValueError(f"unknown Phase-B scenario {scenario!r}")
    return f"{checkpoint_name}_{scenario}"


def phase_a_cells() -> tuple[dict, ...]:
    return tuple(
        {
            "cell": phase_a_cell_label(candidate["name"], spec["label"]),
            "checkpoint_name": candidate["name"],
            "fixed_cell": spec["label"],
            "cell_kind": spec["cell_kind"],
        }
        for candidate in CANDIDATES
        for spec in PHASE_A_SPECS
    )


def phase_a_steps(fixed_cell: str) -> int:
    phase_a_spec_by_label(fixed_cell)
    return PHASE_A_STEPS


def phase_b_steps(scenario: str) -> int:
    if scenario not in PHASE_B_SCENARIOS:
        raise ValueError(f"unknown Phase-B scenario {scenario!r}")
    return PHASE_B_STEPS


def _stage_scene() -> dict:
    import sys

    skrl_root = HERE.parent
    if str(skrl_root) not in sys.path:
        sys.path.insert(0, str(skrl_root))
    from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
        make_sim2real_curriculum_config,
    )
    from rnn_car_wdclean.sa3_sa8_acceptance_contract import stage_scene_contract

    values = stage_scene_contract(STAGE)
    values["room_half_extent_m"] = float(values["arena_size_m"]) / 2.0
    values["corridor_wall_span_m"] = float(values["arena_size_m"])
    values["corridor_boundary_wall_width_m"] = 1.0
    values["corridor_expected_boundary_overlap_m"] = 0.5
    values["corridor_sealed_to_boundary"] = True
    cfg = make_sim2real_curriculum_config(
        STAGE, checkpoint="__SA6_SCREEN_SCENE_PROBE_NOT_LOADED__.pt"
    )
    values["narrow_segment_length_m"] = float(cfg.narrow_passage_segment_length)
    return json.loads(json.dumps(values))


def _json_profile(profile: dict) -> dict:
    result = dict(profile)
    result["speed_range_m_s"] = list(result["speed_range_m_s"])
    return result


def screen_protocol() -> dict:
    payload = {
        "schema": "sa6_v3_checkpoint_screen_protocol/v3",
        "purpose": (
            "compare parent c50 and continuation c100-c350 under one fixed "
            "SA6 evaluator, then run retention only for the Phase-A top two"
        ),
        "lineage": {
            "pilot_run": PILOT_RUN_NAME,
            "continuation_run": CONTINUATION_RUN_NAME,
        },
        "candidates": [dict(candidate) for candidate in CANDIDATES],
        "phase_a": {
            "fixed_cells": [_json_profile(row) for row in PHASE_A_SPECS],
            "profile_cells": [row["label"] for row in PROFILE_SPECS],
            "family_cells": [row["label"] for row in FAMILY_SPECS],
            "family_reference_profile": "sa6_4s2d",
            "cells": len(CANDIDATES) * len(PHASE_A_SPECS),
            "ranking": {
                "eligibility_partition": "all seven hard-gate passes rank before failures",
                "primary": "minimum worst (1-SR) across seven Phase-A cells",
                "secondary": "minimum worst CR across seven Phase-A cells",
                "tertiary": "minimum mean (1-SR), then minimum mean CR",
                "last_tiebreak": "prefer later checkpoint only after exact score ties",
                "descriptive_tie_margin": TIE_MARGIN,
            },
            "promotion": "exactly two structurally valid candidates",
        },
        "phase_b": {
            "scenarios": list(PHASE_B_SCENARIOS),
            "low_density_profiles": [
                _json_profile(row) for row in RETENTION_PROFILES
            ],
            "low_density_motion_family": "lateral",
            "cells": 2 * len(PHASE_B_SCENARIOS),
        },
        "fixed_evaluation": {
            "stage": STAGE,
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "phase_a_steps": PHASE_A_STEPS,
            "phase_b_steps": PHASE_B_STEPS,
            "phase_a_step_budget_by_cell": {
                spec["label"]: phase_a_steps(spec["label"])
                for spec in PHASE_A_SPECS
            },
            "phase_b_step_budget_by_scenario": {
                scenario: phase_b_steps(scenario)
                for scenario in PHASE_B_SCENARIOS
            },
            "r1_correction": {
                "predecessor_protocol_sha256": (
                    "9e949ba1c1f115d9683bc1ea5d62e11d55b11a349d5243c62816ad8d0ca778e2"
                ),
                "predecessor_status": "INCOMPLETE_NO_VERDICT",
                "reason": (
                    "c50 family_mixed_4s2d completed 858 episodes in 2500 "
                    "steps; every mixed checkpoint now receives the same "
                    "5000-step budget"
                ),
            },
            "r2_correction": {
                "predecessor_protocol_sha256": (
                    "224b56ef21bdcd2dca2e3dbc48bc359b6f15e4212fab24b5360ececdbd2fb644"
                ),
                "predecessor_status": "INCOMPLETE_NO_VERDICT",
                "failed_cell": "c150_profile_sa6_4s2d",
                "completed_episodes": 912,
                "minimum_completed_episodes": MIN_EPISODES,
                "reason": (
                    "c150 profile_sa6_4s2d completed 912 episodes in 2500 "
                    "steps; all seven Phase-A cell types now receive the same "
                    "5000-step budget for every checkpoint"
                ),
            },
            "minimum_completed_episodes": MIN_EPISODES,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "corridor_installation": {
                "profile_cells": (
                    "single-entry production count-mix with env_stratified motion"
                ),
                "family_cells": "legacy fixed-count 4S2D pure-family gate path",
                "reason": (
                    "pure lateral supports at most two dynamic obstacles, so 4S3D "
                    "cannot be crossed with every pure family"
                ),
            },
            "count_mix_runtime_audit": {
                "authoritative_density_check": (
                    "actual static_obstacles_per_env and dynamic_obstacles_per_env"
                ),
                "legacy_fixed_slot_limit": 2,
                "production_dynamic_slot_capacity": 5,
                "limitation": (
                    "legacy obstacle_mix_pass, movement_pass, motion_mode_pass, and "
                    "penetration_expected_slot_frames use the fixed two-slot CLI "
                    "count. Exact production-profile cells therefore validate the "
                    "installed scheduler counts and active-slot accounting directly, "
                    "while reporting rather than gating the production path's "
                    "dynamic-static penetration audit. Pure-family 4S2D cells retain "
                    "the strict zero-penetration gate"
                ),
            },
            "scene": _stage_scene(),
        },
        "thresholds": {
            "corridor": dict(CORRIDOR_THRESHOLDS),
            "native_stage6": dict(NATIVE_THRESHOLDS),
            "narrow": dict(NARROW_THRESHOLDS),
        },
        "evidence_boundary": (
            "single training seed and fixed evaluator seed developmental screen; "
            "it ranks checkpoints but accepts no parent and is not a formal "
            "multi-seed/multi-delay SA6 graduation matrix"
        ),
        "authorization_limit": (
            "starts neither training nor SA7 and never accepts a parent"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def thresholds_for(scenario: str) -> dict:
    if scenario == "corridor" or scenario.startswith("retention_"):
        return dict(CORRIDOR_THRESHOLDS)
    if scenario == "nav_native":
        return dict(NATIVE_THRESHOLDS)
    if scenario == "narrow_range":
        return dict(NARROW_THRESHOLDS)
    raise ValueError(f"unknown metric scenario {scenario!r}")


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def evaluate_metrics(scenario: str, metrics: dict) -> dict:
    thresholds = thresholds_for(scenario)
    checks = {
        "episodes": int(metrics.get("n", 0)) >= int(thresholds["episodes_min"]),
        "sr": _finite(metrics.get("sr"), "sr") >= float(thresholds["sr_min"]),
        "cr": _finite(metrics.get("cr"), "cr") <= float(thresholds["cr_max"]),
        "to": _finite(metrics.get("to"), "to") <= float(thresholds["to_max"]),
    }
    if scenario == "narrow_range":
        checks["crossing"] = _finite(
            metrics.get("crossing_rate"), "crossing_rate"
        ) >= float(thresholds["crossing_min"])
        checks["direct_crossing"] = _finite(
            metrics.get("direct_crossing_rate"), "direct_crossing_rate"
        ) >= float(thresholds["direct_crossing_min"])
    return {"checks": checks, "pass": all(checks.values())}


def _validate_common(payload: dict, candidate: dict) -> None:
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("cell protocol hash mismatch")
    if payload.get("checkpoint_name") != candidate["name"]:
        raise ValueError("cell checkpoint name mismatch")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    if not payload.get("cell_valid"):
        raise ValueError("cell is not structurally valid")
    if int(payload.get("stage", -1)) != STAGE:
        raise ValueError("cell stage mismatch")
    if int(payload.get("seed", -1)) != SEED:
        raise ValueError("cell seed mismatch")
    if int(payload.get("delay_steps", -1)) != DELAY_STEPS:
        raise ValueError("cell delay mismatch")
    if _finite(payload.get("speed_rate"), "speed_rate") != SPEED_RATE:
        raise ValueError("cell speed-rate mismatch")


def validate_phase_a_cell(payload: dict) -> None:
    if payload.get("schema") != PHASE_A_CELL_SCHEMA:
        raise ValueError("unexpected Phase-A cell schema")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    spec = phase_a_spec_by_label(str(payload.get("fixed_cell")))
    _validate_common(payload, candidate)
    if payload.get("cell") != phase_a_cell_label(candidate["name"], spec["label"]):
        raise ValueError("Phase-A cell label mismatch")
    if payload.get("cell_kind") != spec["cell_kind"]:
        raise ValueError("Phase-A cell kind mismatch")
    metrics = payload.get("metrics") or {}
    gate = evaluate_metrics("corridor", metrics)
    if payload.get("gate") != gate:
        raise ValueError("Phase-A gate payload mismatch")
    runtime = payload.get("runtime") or {}
    if int(runtime.get("static_obstacles", -1)) != spec["static_obstacles"]:
        raise ValueError("runtime static density mismatch")
    if int(runtime.get("dynamic_obstacles", -1)) != spec["dynamic_obstacles"]:
        raise ValueError("runtime dynamic density mismatch")
    if list(runtime.get("pedestrian_speed_range_m_s") or []) != list(
        spec["speed_range_m_s"]
    ):
        raise ValueError("runtime pedestrian speed mismatch")
    if runtime.get("motion_mode") != spec["motion_mode"]:
        raise ValueError("runtime motion family mismatch")
    if runtime.get("installation") != spec["installation"]:
        raise ValueError("runtime corridor installation mismatch")


def rank_phase_a(payloads: Iterable[dict]) -> dict:
    items = list(payloads)
    expected_labels = {row["cell"] for row in phase_a_cells()}
    if len(items) != len(expected_labels):
        raise ValueError(
            f"Phase A needs exactly {len(expected_labels)} cells, got {len(items)}"
        )
    by_label: dict[str, dict] = {}
    for payload in items:
        validate_phase_a_cell(payload)
        label = str(payload["cell"])
        if label in by_label:
            raise ValueError(f"duplicate Phase-A cell {label}")
        by_label[label] = payload
    if set(by_label) != expected_labels:
        raise ValueError("Phase A is missing a frozen cell")

    summaries = []
    for candidate in CANDIDATES:
        rows = [
            by_label[phase_a_cell_label(candidate["name"], spec["label"])]
            for spec in PHASE_A_SPECS
        ]
        failures = [1.0 - float(row["metrics"]["sr"]) for row in rows]
        collisions = [float(row["metrics"]["cr"]) for row in rows]
        worst_index = max(range(len(rows)), key=lambda index: failures[index])
        summaries.append(
            {
                "checkpoint_name": candidate["name"],
                "conceptual_iteration": candidate["conceptual_iteration"],
                "checkpoint_sha256": candidate["sha256"],
                "all_target_hard_pass": all(bool(row["gate"]["pass"]) for row in rows),
                "worst_failure_rate": max(failures),
                "worst_cr": max(collisions),
                "mean_failure_rate": sum(failures) / len(failures),
                "mean_cr": sum(collisions) / len(collisions),
                "worst_cell": rows[worst_index]["cell"],
                "cells": {
                    row["cell"]: {
                        "fixed_cell": row["fixed_cell"],
                        "cell_kind": row["cell_kind"],
                        "n": int(row["metrics"]["n"]),
                        "sr": float(row["metrics"]["sr"]),
                        "cr": float(row["metrics"]["cr"]),
                        "to": float(row["metrics"]["to"]),
                        "failure_rate": 1.0 - float(row["metrics"]["sr"]),
                        "gate_pass": bool(row["gate"]["pass"]),
                    }
                    for row in rows
                },
            }
        )
    ranked_rows = sorted(
        summaries,
        key=lambda row: (
            not bool(row["all_target_hard_pass"]),
            float(row["worst_failure_rate"]),
            float(row["worst_cr"]),
            float(row["mean_failure_rate"]),
            float(row["mean_cr"]),
            -int(row["conceptual_iteration"]),
        ),
    )
    best = float(ranked_rows[0]["worst_failure_rate"])
    return {
        "schema": "sa6_v3_checkpoint_screen_phase_a_summary/v1",
        "ranked": [row["checkpoint_name"] for row in ranked_rows],
        "top_two": [row["checkpoint_name"] for row in ranked_rows[:2]],
        "leader_tie_group": [
            row["checkpoint_name"]
            for row in ranked_rows
            if float(row["worst_failure_rate"]) - best < TIE_MARGIN
        ],
        "tie_margin": TIE_MARGIN,
        "summaries": ranked_rows,
        "summaries_by_checkpoint": {
            row["checkpoint_name"]: row for row in ranked_rows
        },
    }


def validate_phase_b_cell(payload: dict, expected_checkpoint: str) -> None:
    if payload.get("schema") != PHASE_B_CELL_SCHEMA:
        raise ValueError("unexpected Phase-B cell schema")
    candidate = candidate_by_name(expected_checkpoint)
    _validate_common(payload, candidate)
    scenario = str(payload.get("scenario"))
    if scenario not in PHASE_B_SCENARIOS:
        raise ValueError("unknown Phase-B scenario")
    if payload.get("cell") != phase_b_cell_label(candidate["name"], scenario):
        raise ValueError("Phase-B cell label mismatch")
    metrics = payload.get("metrics") or {}
    if payload.get("gate") != evaluate_metrics(scenario, metrics):
        raise ValueError("Phase-B gate payload mismatch")


def final_verdict(
    phase_a_payloads: Iterable[dict], phase_b_payloads: Iterable[dict]
) -> dict:
    phase_a = rank_phase_a(phase_a_payloads)
    top_two = phase_a["top_two"]
    items = list(phase_b_payloads)
    expected = len(top_two) * len(PHASE_B_SCENARIOS)
    if len(items) != expected:
        raise ValueError(f"Phase B needs exactly {expected} cells, got {len(items)}")
    by_label = {}
    for payload in items:
        name = str(payload.get("checkpoint_name"))
        if name not in top_two:
            raise ValueError("Phase B contains an unselected checkpoint")
        validate_phase_b_cell(payload, name)
        label = str(payload["cell"])
        if label in by_label:
            raise ValueError(f"duplicate Phase-B cell {label}")
        by_label[label] = payload
    expected_labels = {
        phase_b_cell_label(name, scenario)
        for name in top_two
        for scenario in PHASE_B_SCENARIOS
    }
    if set(by_label) != expected_labels:
        raise ValueError("Phase B is missing a frozen cell")

    phase_a_by_name = phase_a["summaries_by_checkpoint"]
    candidates = []
    for name in top_two:
        cells = {
            scenario: by_label[phase_b_cell_label(name, scenario)]
            for scenario in PHASE_B_SCENARIOS
        }
        retention_pass = all(bool(row["gate"]["pass"]) for row in cells.values())
        target_pass = bool(phase_a_by_name[name]["all_target_hard_pass"])
        candidates.append(
            {
                "checkpoint_name": name,
                "target_hard_pass": target_pass,
                "retention_hard_pass": retention_pass,
                "all_hard_pass": target_pass and retention_pass,
                "cells": {
                    scenario: row["metrics"] for scenario, row in cells.items()
                },
            }
        )
    eligible = [row["checkpoint_name"] for row in candidates if row["all_hard_pass"]]
    recommendation = next(
        (name for name in phase_a["ranked"] if name in eligible), None
    )
    return {
        "schema": "sa6_v3_checkpoint_screen_verdict/v1",
        "phase_a": phase_a,
        "phase_b_candidates": candidates,
        "screen_eligible_checkpoints": eligible,
        "recommended_for_human_consideration": recommendation,
        "accepted_parent": False,
        "training_started": False,
        "sa7_started": False,
    }
