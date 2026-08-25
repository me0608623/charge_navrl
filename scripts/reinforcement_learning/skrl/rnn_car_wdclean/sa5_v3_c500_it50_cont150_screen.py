"""Frozen two-phase screen for the c500-it50 to it150 continuation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import sa5_v3_c500_parent_control_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

RUN_NAME = "sa5_v3_c500_it50_cont150_ne1024_s42_p100_r1"
RUN_DIR = REPO / "logs/rnn_car" / RUN_NAME
SCREEN_ROOT = (
    REPO
    / "logs/gates/sa5_v3_c500_it50_cont150_screen/screen_20260823_r1"
)
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"

STAGE = base.STAGE
SEED = base.SEED
NUM_ENVS = base.NUM_ENVS
MIN_EPISODES = base.MIN_EPISODES
DELAY_STEPS = base.DELAY_STEPS
DELAY_MS = base.DELAY_MS
ACTUATOR_PROFILE = base.ACTUATOR_PROFILE
LIDAR_NOISE_MODE = base.LIDAR_NOISE_MODE
LIDAR_DISTRACTOR_ELIGIBILITY = base.LIDAR_DISTRACTOR_ELIGIBILITY
SPEED_RATE = base.SPEED_RATE
SPEED_RATE_OBS = base.SPEED_RATE_OBS
DEPLOYMENT_SPEED_SCALE = base.DEPLOYMENT_SPEED_SCALE
P035 = base.P035
P060 = base.P060

TARGET_SCENARIO = base.TARGET_SCENARIO
CORRIDOR_SCENARIOS = (TARGET_SCENARIO,)
LOW_DENSITY_SCENARIOS = base.LOW_DENSITY_SCENARIOS
PHASE_A_SCENARIOS = (TARGET_SCENARIO, "nav_native")
PHASE_B_SCENARIOS = ("narrow_range", *LOW_DENSITY_SCENARIOS)
RETENTION_SCENARIOS = ("nav_native", *PHASE_B_SCENARIOS)
ALL_SCENARIOS = (*PHASE_A_SCENARIOS, *PHASE_B_SCENARIOS)
STEPS_BY_SCENARIO = dict(base.STEPS_BY_SCENARIO)

CANDIDATE_SPECS = (
    {
        "name": "c50",
        "conceptual_iteration": 50,
        "path": (
            "logs/rnn_car/"
            "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1/"
            "checkpoint_6400.pt"
        ),
        "expected_sha256": (
            "4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c"
        ),
    },
    {
        "name": "c75",
        "conceptual_iteration": 75,
        "path": f"logs/rnn_car/{RUN_NAME}/checkpoint_3200.pt",
        "expected_sha256": None,
    },
    {
        "name": "c100",
        "conceptual_iteration": 100,
        "path": f"logs/rnn_car/{RUN_NAME}/checkpoint_6400.pt",
        "expected_sha256": None,
    },
    {
        "name": "c125",
        "conceptual_iteration": 125,
        "path": f"logs/rnn_car/{RUN_NAME}/checkpoint_9600.pt",
        "expected_sha256": None,
    },
    {
        "name": "c150",
        "conceptual_iteration": 150,
        "path": f"logs/rnn_car/{RUN_NAME}/checkpoint_12800.pt",
        "expected_sha256": None,
    },
)

TARGET_THRESHOLDS = dict(base.TARGET_THRESHOLDS)
NATIVE_THRESHOLDS = dict(base.NATIVE_THRESHOLDS)
NARROW_THRESHOLDS = dict(base.NARROW_THRESHOLDS)
P060_THRESHOLDS = dict(base.P060_THRESHOLDS)
TARGET_MIN_CR_IMPROVEMENT = base.TARGET_MIN_CR_IMPROVEMENT
TARGET_MIN_Z = base.TARGET_MIN_Z
RETENTION_MAX_DEGRADATION = base.RETENTION_MAX_DEGRADATION
NATIVE_CR_GATE = NATIVE_THRESHOLDS["cr_max"]
CELL_SCHEMA = "sa5_v3_c500_it50_cont150_screen_cell/v1"

AUTHORIZATION = (
    REPO
    / "docs/freeze/sa5_v3_c500_it50_cont150_authorization_20260823.json"
)
AUTHORIZATION_SHA256 = (
    "837227255310974afa7fd855c6f81c75119a632b57ece32b032c5a6473781709"
)
HISTORICAL_C50_SUMMARY = (
    REPO
    / "logs/gates/sa5_v3_c500_parent_control_screen/"
    "screen_20260823_r1/SUMMARY.json"
)
HISTORICAL_C50_SUMMARY_SHA256 = (
    "35417943f31191496b56a24b5d0ba939d3e40ca754ed597d6643ced40f8a79e7"
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(spec: dict | str) -> Path:
    if isinstance(spec, str):
        candidate = next(
            (row for row in CANDIDATE_SPECS if row["name"] == spec), None
        )
        if candidate is None:
            raise ValueError(f"unknown continuation candidate {spec!r}")
        spec = candidate
    return REPO / str(spec["path"])


def _stage_scene() -> dict:
    return base._stage_scene()


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c500_it50_cont150_screen_protocol/v1",
        "run_name": RUN_NAME,
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
            "decision": (
                "HUMAN_AUTHORIZED_BOUNDED_CONTINUATION_C500_IT50_TO_IT150"
            ),
        },
        "lineage": {
            "sa4": "SA4_NOT_GRADUATED",
            "source": "PROVISIONAL_SA5_C500_IT50_CONTINUATION_SOURCE",
            "sa6": "HOLD_NOT_AUTHORIZED",
        },
        "historical_c50_retention_baseline": {
            "path": str(HISTORICAL_C50_SUMMARY.relative_to(REPO)),
            "sha256": HISTORICAL_C50_SUMMARY_SHA256,
        },
        "stage": STAGE,
        "candidates": [dict(row) for row in CANDIDATE_SPECS],
        "phases": {
            "phase_a": {
                "scenarios": list(PHASE_A_SCENARIOS),
                "cells": len(CANDIDATE_SPECS) * len(PHASE_A_SCENARIOS),
            },
            "phase_b": {
                "selection_count": 2,
                "scenarios": list(PHASE_B_SCENARIOS),
                "cells": 2 * len(PHASE_B_SCENARIOS),
            },
        },
        "fixed_evaluation": {
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps_by_scenario": dict(STEPS_BY_SCENARIO),
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "scene": _stage_scene(),
            "target": "4S2D P035 mixed",
            "p060": "0S1D/1S1D P060 lateral",
        },
        "ranking": {
            "formula": (
                "minimize max(target_CR/c50_target_CR, native_CR/0.10), "
                "then mean normalized ratio, target CR, native CR, then "
                "prefer later conceptual iteration only on an exact tie"
            ),
            "top_k": 2,
        },
        "acceptance": {
            "target_min_cr_improvement": TARGET_MIN_CR_IMPROVEMENT,
            "target_min_pooled_se_z": TARGET_MIN_Z,
            "native_cr_max": NATIVE_CR_GATE,
            "retention_max_cr_increase": RETENTION_MAX_DEGRADATION,
            "retention_max_sr_decrease": RETENTION_MAX_DEGRADATION,
            "retention_max_to_increase": RETENTION_MAX_DEGRADATION,
            "all_absolute_retention_gates_required": True,
            "target_absolute_gate_is_descriptive": True,
        },
        "decision": {
            "on_pass": (
                "accept selected c75-c150 checkpoint as provisional SA5 "
                "parent; keep SA6 on hold"
            ),
            "on_fail": (
                "hold SA5 after bounded continuation; no extra iterations "
                "without new authorization"
            ),
            "screen_starts_training": False,
            "screen_starts_sa6": False,
        },
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "screen; it cannot rewrite SA4 graduation or authorize SA6"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def load_checkpoint_manifest() -> dict:
    if not CHECKPOINT_MANIFEST.is_file():
        raise FileNotFoundError(CHECKPOINT_MANIFEST)
    manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("schema")
        != "sa5_v3_c500_it50_cont150_checkpoint_manifest/v1"
        or manifest.get("protocol_sha256") != screen_protocol()["sha256"]
    ):
        raise ValueError("unexpected continuation checkpoint manifest")
    names = [row["name"] for row in CANDIDATE_SPECS]
    if list((manifest.get("checkpoints") or {}).keys()) != names:
        raise ValueError("continuation checkpoint manifest order mismatch")
    return manifest


def candidate_by_name(name: str) -> dict:
    spec = next((row for row in CANDIDATE_SPECS if row["name"] == name), None)
    if spec is None:
        raise ValueError(f"unknown continuation candidate {name!r}")
    observed = (load_checkpoint_manifest().get("checkpoints") or {}).get(name)
    if not observed:
        raise ValueError(f"continuation manifest lacks {name}")
    path = checkpoint_path(spec).resolve()
    if Path(observed.get("path", "")).resolve() != path:
        raise ValueError(f"continuation checkpoint path mismatch for {name}")
    result = dict(spec)
    result["sha256"] = str(observed["sha256"])
    return result


def thresholds_for(scenario: str) -> dict:
    if scenario == TARGET_SCENARIO:
        return dict(TARGET_THRESHOLDS)
    if scenario == "nav_native":
        return dict(NATIVE_THRESHOLDS)
    if scenario == "narrow_range":
        return dict(NARROW_THRESHOLDS)
    if scenario in LOW_DENSITY_SCENARIOS:
        return dict(P060_THRESHOLDS)
    raise ValueError(f"unknown continuation scenario {scenario!r}")


def _finite(value, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def evaluate_metrics(scenario: str, metrics: dict) -> dict:
    thresholds = thresholds_for(scenario)
    checks = {
        "episodes": int(metrics.get("n", 0)) >= thresholds["episodes_min"],
        "sr": _finite(metrics.get("sr"), "sr") >= thresholds["sr_min"],
        "cr": _finite(metrics.get("cr"), "cr") <= thresholds["cr_max"],
        "to": _finite(metrics.get("to"), "to") <= thresholds["to_max"],
    }
    if scenario == "narrow_range":
        checks["crossing"] = (
            _finite(metrics.get("crossing_rate"), "crossing_rate")
            >= thresholds["crossing_min"]
        )
        checks["direct_crossing"] = (
            _finite(metrics.get("direct_crossing_rate"), "direct_crossing_rate")
            >= thresholds["direct_crossing_min"]
        )
    return {
        "thresholds": thresholds,
        "checks": checks,
        "threshold_pass": all(checks.values()),
    }


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != CELL_SCHEMA or not payload.get("cell_valid"):
        raise ValueError("invalid continuation cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("continuation protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = str(payload.get("scenario"))
    if scenario not in ALL_SCENARIOS:
        raise ValueError("unknown continuation cell scenario")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("continuation checkpoint hash mismatch")
    for key, expected in {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS_BY_SCENARIO[scenario],
    }.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"continuation {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("continuation LiDAR mode mismatch")
    if payload.get("lidar_distractor_eligibility") != LIDAR_DISTRACTOR_ELIGIBILITY:
        raise ValueError("continuation LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(runtime.get("deployment_speed_scale"), "deployment_speed_scale")
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("continuation speed-rate runtime mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("continuation cell has too few episodes")
    verdict = evaluate_metrics(scenario, metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("continuation threshold verdict mismatch")


def _relative_retention(parent: dict, candidate: dict) -> dict:
    deltas = {
        "cr_increase": float(candidate["cr"]) - float(parent["cr"]),
        "sr_decrease": float(parent["sr"]) - float(candidate["sr"]),
        "to_increase": float(candidate["to"]) - float(parent["to"]),
    }
    checks = {
        key: value <= RETENTION_MAX_DEGRADATION + 1e-12
        for key, value in deltas.items()
    }
    return {"deltas": deltas, "checks": checks, "pass": all(checks.values())}


def _target_improvement(parent: dict, candidate: dict) -> dict:
    p0, p1 = float(parent["cr"]), float(candidate["cr"])
    n0, n1 = int(parent["n"]), int(candidate["n"])
    se = math.sqrt(p0 * (1.0 - p0) / n0 + p1 * (1.0 - p1) / n1)
    improvement = p0 - p1
    z = improvement / se if se > 0.0 else (math.inf if improvement > 0 else 0.0)
    return {
        "parent_cr": p0,
        "candidate_cr": p1,
        "cr_improvement": improvement,
        "pooled_se": se,
        "z": z,
        "minimum_improvement_pass": improvement >= TARGET_MIN_CR_IMPROVEMENT,
        "minimum_z_pass": z >= TARGET_MIN_Z,
        "pass": improvement >= TARGET_MIN_CR_IMPROVEMENT and z >= TARGET_MIN_Z,
    }


def _matrix(
    payloads: list[dict], scenarios: tuple[str, ...], names: list[str]
) -> dict[str, dict[str, dict]]:
    expected = len(names) * len(scenarios)
    if len(payloads) != expected:
        raise ValueError(f"screen phase needs exactly {expected} cells")
    result: dict[str, dict[str, dict]] = {}
    for payload in payloads:
        validate_cell(payload)
        name = str(payload["checkpoint_name"])
        scenario = str(payload["scenario"])
        if name not in names or scenario not in scenarios:
            raise ValueError("screen phase contains an unexpected cell")
        if scenario in result.setdefault(name, {}):
            raise ValueError(f"duplicate screen cell {name}/{scenario}")
        result[name][scenario] = payload
    if list(result) != names:
        raise ValueError("screen phase candidate order mismatch")
    if any(set(cells) != set(scenarios) for cells in result.values()):
        raise ValueError("screen phase scenario matrix mismatch")
    return result


def rank_phase_a(payloads: list[dict]) -> dict:
    names = [row["name"] for row in CANDIDATE_SPECS]
    cells = _matrix(payloads, PHASE_A_SCENARIOS, names)
    baseline_target = cells["c50"][TARGET_SCENARIO]["metrics"]
    baseline_native = cells["c50"]["nav_native"]["metrics"]
    baseline_target_cr = float(baseline_target["cr"])
    rows = []
    for spec in CANDIDATE_SPECS:
        name = spec["name"]
        target_cell = cells[name][TARGET_SCENARIO]
        native_cell = cells[name]["nav_native"]
        target = target_cell["metrics"]
        native = native_cell["metrics"]
        target_ratio = float(target["cr"]) / baseline_target_cr
        native_ratio = float(native["cr"]) / NATIVE_CR_GATE
        target_gain = _target_improvement(baseline_target, target)
        native_relative = _relative_retention(baseline_native, native)
        rows.append(
            {
                "checkpoint_name": name,
                "conceptual_iteration": spec["conceptual_iteration"],
                "target_metrics": target,
                "native_metrics": native,
                "target_improvement": target_gain,
                "native_absolute_pass": bool(native_cell["threshold_pass"]),
                "native_relative": native_relative,
                "normalized": {
                    "target_vs_c50": target_ratio,
                    "native_vs_gate": native_ratio,
                    "worst": max(target_ratio, native_ratio),
                    "mean": (target_ratio + native_ratio) / 2.0,
                },
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            row["normalized"]["worst"],
            row["normalized"]["mean"],
            float(row["target_metrics"]["cr"]),
            float(row["native_metrics"]["cr"]),
            -int(row["conceptual_iteration"]),
        ),
    )
    return {
        "schema": "sa5_v3_c500_it50_cont150_phase_a_summary/v1",
        "status": "COMPLETE_VALID_PHASE_A",
        "protocol_sha256": screen_protocol()["sha256"],
        "baseline_checkpoint": "c50",
        "rows": rows,
        "ranking": [row["checkpoint_name"] for row in ranked],
        "selected_top_two": [row["checkpoint_name"] for row in ranked[:2]],
    }


def _historical_c50_retention() -> dict[str, dict]:
    if sha256_of(HISTORICAL_C50_SUMMARY) != HISTORICAL_C50_SUMMARY_SHA256:
        raise RuntimeError("historical c50 summary hash mismatch")
    summary = json.loads(HISTORICAL_C50_SUMMARY.read_text(encoding="utf-8"))
    row = next(
        item
        for item in summary["candidate_results"]
        if item["checkpoint_name"] == "control_it50"
    )
    return {
        scenario: row["retention"][scenario]["metrics"]
        for scenario in PHASE_B_SCENARIOS
    }


def final_verdict(phase_a_payloads: list[dict], phase_b_payloads: list[dict]) -> dict:
    phase_a = rank_phase_a(phase_a_payloads)
    selected_names = list(phase_a["selected_top_two"])
    phase_b_cells = _matrix(
        phase_b_payloads, PHASE_B_SCENARIOS, selected_names
    )
    phase_a_rows = {row["checkpoint_name"]: row for row in phase_a["rows"]}
    historical = _historical_c50_retention()

    results = []
    for name in selected_names:
        phase_a_row = phase_a_rows[name]
        retention = {
            "nav_native": {
                "metrics": phase_a_row["native_metrics"],
                "absolute_pass": phase_a_row["native_absolute_pass"],
                "relative": phase_a_row["native_relative"],
                "pass": phase_a_row["native_absolute_pass"]
                and phase_a_row["native_relative"]["pass"],
                "baseline": "fresh_c50_phase_a",
            }
        }
        for scenario in PHASE_B_SCENARIOS:
            cell = phase_b_cells[name][scenario]
            relative = _relative_retention(
                historical[scenario], cell["metrics"]
            )
            retention[scenario] = {
                "metrics": cell["metrics"],
                "absolute_pass": bool(cell["threshold_pass"]),
                "relative": relative,
                "pass": bool(cell["threshold_pass"]) and relative["pass"],
                "baseline": "frozen_historical_c50",
            }
        retention_pass = all(row["pass"] for row in retention.values())
        eligible = phase_a_row["target_improvement"]["pass"] and retention_pass
        results.append(
            {
                **phase_a_row,
                "retention": retention,
                "retention_pass": retention_pass,
                "acceptance_eligible": eligible,
            }
        )

    eligible = [row for row in results if row["acceptance_eligible"]]
    eligible.sort(
        key=lambda row: (
            row["normalized"]["worst"],
            row["normalized"]["mean"],
            float(row["target_metrics"]["cr"]),
            float(row["native_metrics"]["cr"]),
            -int(row["conceptual_iteration"]),
        )
    )
    selected = eligible[0]["checkpoint_name"] if eligible else None
    return {
        "schema": "sa5_v3_c500_it50_cont150_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_BOUNDED_CONTINUATION_SCREEN",
        "protocol_sha256": screen_protocol()["sha256"],
        "phase_a": phase_a,
        "phase_b_results": results,
        "selected_provisional_sa5_parent": selected,
        "acceptance_authorized": selected is not None,
        "next_action": (
            "ACCEPT_SELECTED_PROVISIONAL_SA5_PARENT_HOLD_SA6"
            if selected is not None
            else "HOLD_SA5_AFTER_BOUNDED_CONTINUATION_NO_EXTRA_ITERATIONS"
        ),
        "sa4_graduated": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["evidence_boundary"],
    }
