"""Frozen 12-cell comparison of conceptual SA5 B3 c500/c550/c600."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sa5_v3_c50_stage3_b3_cont25_screen as source


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCREEN_ROOT = (
    REPO
    / "logs/gates/sa5_v3_c50_stage3_b3_c500_c600_screen/screen_20260826_r1"
)
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"
CHECKPOINT_LOCK = (
    REPO
    / "docs/freeze/sa5_v3_c50_stage3_b3_c500_c600_screen_checkpoints_v1.json"
)
AUTHORIZATION = (
    REPO
    / "docs/freeze/sa5_v3_c50_stage3_b3_c500_c600_screen_authorization_20260826.json"
)

STAGE = source.STAGE
SEED = source.SEED
NUM_ENVS = source.NUM_ENVS
MIN_EPISODES = source.MIN_EPISODES
DELAY_STEPS = source.DELAY_STEPS
DELAY_MS = source.DELAY_MS
ACTUATOR_PROFILE = source.ACTUATOR_PROFILE
LIDAR_NOISE_MODE = source.LIDAR_NOISE_MODE
LIDAR_DISTRACTOR_ELIGIBILITY = source.LIDAR_DISTRACTOR_ELIGIBILITY
SPEED_RATE = source.SPEED_RATE
SPEED_RATE_OBS = source.SPEED_RATE_OBS
DEPLOYMENT_SPEED_SCALE = source.DEPLOYMENT_SPEED_SCALE
P060 = source.P060
P080 = source.P080
P060_SCENARIOS = source.P060_SCENARIOS
P080_SCENARIOS = source.P080_SCENARIOS
ALL_SCENARIOS = source.ALL_SCENARIOS
LOW_DENSITY_SCENARIOS = ALL_SCENARIOS
CORRIDOR_SCENARIOS: tuple[str, ...] = ()
TARGET_SCENARIO = "__c500_c600_comparison_has_no_target_cell__"
STEPS_BY_SCENARIO = dict(source.STEPS_BY_SCENARIO)
THRESHOLDS = dict(source.THRESHOLDS)
MAX_DEGRADATION = source.MAX_DEGRADATION
TIE_BAND = source.TIE_BAND
CELL_SCHEMA = "sa5_v3_c50_stage3_b3_c500_c600_screen_cell/v1"

sha256_of = source.sha256_of
speed_range_for = source.speed_range_for
density_for = source.density_for
thresholds_for = source.thresholds_for
evaluate_metrics = source.evaluate_metrics
_stage_scene = source._stage_scene
_finite = source._finite


def _load_checkpoint_lock() -> dict:
    payload = json.loads(CHECKPOINT_LOCK.read_text(encoding="utf-8"))
    if payload.get("schema") != "sa5_v3_c50_stage3_b3_c500_c600_checkpoint_lock/v1":
        raise RuntimeError("invalid c500-c600 checkpoint lock schema")
    return payload


def _candidate_specs() -> tuple[dict, ...]:
    locked = (_load_checkpoint_lock().get("checkpoints") or {})
    rows = []
    for name, conceptual, embedded_iteration, embedded_steps in (
        ("c500", 500, 399, 51_200),
        ("c550", 550, 449, 57_600),
        ("c600", 600, 499, 64_000),
    ):
        entry = locked.get(name) or {}
        if (
            int(entry.get("conceptual_iteration", -1)) != conceptual
            or int(entry.get("embedded_iteration", -1)) != embedded_iteration
            or int(entry.get("embedded_total_steps", -1)) != embedded_steps
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise RuntimeError(f"checkpoint lock mismatch for {name}")
        rows.append(
            {
                "name": name,
                "conceptual_iteration": conceptual,
                "path": entry["path"],
                "embedded_iteration": embedded_iteration,
                "embedded_total_steps": embedded_steps,
                "expected_sha256": entry["sha256"],
            }
        )
    return tuple(rows)


CANDIDATE_SPECS = _candidate_specs()
BASELINE = CANDIDATE_SPECS[0]


def candidate_specs() -> tuple[dict, ...]:
    return tuple(dict(row) for row in CANDIDATE_SPECS)


def candidate_by_name(name: str) -> dict:
    for spec in CANDIDATE_SPECS:
        if spec["name"] == name:
            result = dict(spec)
            result["sha256"] = spec["expected_sha256"]
            return result
    raise ValueError(f"unknown c500-c600 candidate {name!r}")


def checkpoint_path(spec: dict | str = BASELINE) -> Path:
    if isinstance(spec, str):
        spec = candidate_by_name(spec)
    return REPO / str(spec["path"])


def screen_protocol() -> dict:
    specs = candidate_specs()
    payload = {
        "schema": "sa5_v3_c50_stage3_b3_c500_c600_screen_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": sha256_of(AUTHORIZATION),
        },
        "checkpoint_lock": {
            "path": str(CHECKPOINT_LOCK.relative_to(REPO)),
            "sha256": sha256_of(CHECKPOINT_LOCK),
        },
        "question": (
            "which of conceptual c500, c550, and c600 best satisfies the "
            "unchanged P060 capability and P080 retention contract"
        ),
        "matrix": {
            "candidates": [row["name"] for row in specs],
            "conceptual_iterations": [row["conceptual_iteration"] for row in specs],
            "scenarios": list(ALL_SCENARIOS),
            "cell_count": len(specs) * len(ALL_SCENARIOS),
        },
        "fixed_evaluation": {
            "stage": STAGE,
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps_by_scenario": dict(STEPS_BY_SCENARIO),
            "minimum_episodes": MIN_EPISODES,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "motion_mode": "lateral",
            "p060_speed_range_m_s": list(P060),
            "p080_speed_range_m_s": list(P080),
            "densities": ["0S1D", "1S1D"],
            "scene": _stage_scene(),
        },
        "acceptance": {
            "p060_absolute_thresholds": dict(THRESHOLDS),
            "p080_retention_reference": BASELINE["name"],
            "p080_maximum_sr_cr_to_degradation": MAX_DEGRADATION,
            "ranking_primary": "minimum worst P060 CR",
            "ranking_secondary": "minimum mean P060 CR",
            "tie_band": TIE_BAND,
            "tie_break": "earlier conceptual checkpoint",
        },
        "decision": {
            "if_none_pass": "NO_CANDIDATE_PASSES_FIXED_P060_P080_SCREEN",
            "if_pass": "SA5_CANDIDATE_REQUIRES_NATIVE_NARROW_AND_PHASE_B",
            "starts_training": False,
            "starts_phase_b": False,
            "selects_parent": False,
            "graduates_sa5": False,
            "starts_sa6": False,
        },
        "interpretation_limit": (
            "single training seed and single evaluator seed development screen; "
            "c500 is only the preregistered P080 retention reference; a passing "
            "checkpoint remains an SA5 candidate"
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != CELL_SCHEMA or not payload.get("cell_valid"):
        raise ValueError("invalid c500-c600 cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("c500-c600 protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = str(payload.get("scenario"))
    expected_speed = speed_range_for(scenario)
    expected_density = density_for(scenario)
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("c500-c600 checkpoint hash mismatch")
    for key, expected in {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS_BY_SCENARIO[scenario],
    }.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"c500-c600 {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("c500-c600 LiDAR mode mismatch")
    if payload.get("lidar_distractor_eligibility") != LIDAR_DISTRACTOR_ELIGIBILITY:
        raise ValueError("c500-c600 LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(runtime.get("deployment_speed_scale"), "deployment_speed_scale")
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("c500-c600 speed-rate runtime mismatch")
    report = payload.get("corridor_report") or {}
    report_speed = tuple(
        float(value)
        for value in report.get("requested_dynamic_speed_range_m_s", ())
    )
    report_density = (
        int(report.get("static_obstacles_per_env", -1)),
        int(report.get("dynamic_obstacles_per_env", -1)),
    )
    if report_speed != expected_speed or report_density != expected_density:
        raise ValueError("c500-c600 speed/density runtime mismatch")
    if report.get("dynamic_motion_mode") != "lateral":
        raise ValueError("c500-c600 motion-mode mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("c500-c600 cell has too few episodes")
    verdict = evaluate_metrics(scenario, metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("c500-c600 threshold verdict mismatch")


def _row(payload: dict) -> dict:
    metrics = payload["metrics"]
    return {
        "n": int(metrics["n"]),
        "sr": _finite(metrics["sr"], "sr"),
        "cr": _finite(metrics["cr"], "cr"),
        "to": _finite(metrics["to"], "to"),
        "absolute_pass": bool(payload["threshold_pass"]),
    }


def _retention(baseline: dict, candidate: dict) -> dict:
    deltas = {
        "sr_degradation": baseline["sr"] - candidate["sr"],
        "cr_degradation": candidate["cr"] - baseline["cr"],
        "to_degradation": candidate["to"] - baseline["to"],
    }
    checks = {
        key: value <= MAX_DEGRADATION + 1e-12 for key, value in deltas.items()
    }
    return {"deltas": deltas, "checks": checks, "pass": all(checks.values())}


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(CANDIDATE_SPECS) * len(ALL_SCENARIOS)
    if len(payloads) != expected:
        raise ValueError(f"c500-c600 screen needs exactly {expected} cells")
    by_name: dict[str, dict[str, dict]] = {}
    for payload in payloads:
        validate_cell(payload)
        name = str(payload["checkpoint_name"])
        scenario = str(payload["scenario"])
        if scenario in by_name.setdefault(name, {}):
            raise ValueError(f"duplicate c500-c600 cell {name}/{scenario}")
        by_name[name][scenario] = payload
    expected_names = [row["name"] for row in CANDIDATE_SPECS]
    if list(by_name) != expected_names:
        raise ValueError("c500-c600 candidate order mismatch")
    if any(set(rows) != set(ALL_SCENARIOS) for rows in by_name.values()):
        raise ValueError("c500-c600 scenario matrix mismatch")

    baseline_rows = {
        scenario: _row(by_name[BASELINE["name"]][scenario])
        for scenario in ALL_SCENARIOS
    }
    results = []
    passing = []
    for spec in CANDIDATE_SPECS:
        rows = {
            scenario: _row(by_name[spec["name"]][scenario])
            for scenario in ALL_SCENARIOS
        }
        p060_pass = all(rows[name]["absolute_pass"] for name in P060_SCENARIOS)
        retention = {
            name: _retention(baseline_rows[name], rows[name])
            for name in P080_SCENARIOS
        }
        p080_pass = all(row["pass"] for row in retention.values())
        worst_p060_cr = max(rows[name]["cr"] for name in P060_SCENARIOS)
        mean_p060_cr = sum(rows[name]["cr"] for name in P060_SCENARIOS) / 2.0
        candidate_pass = p060_pass and p080_pass
        result = {
            "checkpoint_name": spec["name"],
            "conceptual_iteration": spec["conceptual_iteration"],
            "rows": rows,
            "p060_absolute_pass": p060_pass,
            "p080_retention": retention,
            "p080_retention_pass": p080_pass,
            "worst_p060_cr": worst_p060_cr,
            "mean_p060_cr": mean_p060_cr,
            "candidate_pass": candidate_pass,
        }
        results.append(result)
        if candidate_pass:
            passing.append(result)

    selected = None
    tie_group: list[str] = []
    if passing:
        best_worst = min(row["worst_p060_cr"] for row in passing)
        contenders = [
            row
            for row in passing
            if row["worst_p060_cr"] <= best_worst + TIE_BAND + 1e-12
        ]
        contenders.sort(
            key=lambda row: (row["mean_p060_cr"], row["conceptual_iteration"])
        )
        selected = contenders[0]["checkpoint_name"]
        tie_group = [row["checkpoint_name"] for row in contenders]

    return {
        "schema": "sa5_v3_c50_stage3_b3_c500_c600_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_TRAINING_SEED_12_CELL_SCREEN",
        "protocol_sha256": screen_protocol()["sha256"],
        "p080_retention_reference": BASELINE["name"],
        "candidate_results": results,
        "passing_candidates": [row["checkpoint_name"] for row in passing],
        "tie_group": tie_group,
        "selected_extension_checkpoint": selected,
        "sa5_candidate": selected,
        "phase_b_authorized": False,
        "parent_selected": False,
        "graduation_authorized": False,
        "sa6_started": False,
        "next_action": (
            "SA5_CANDIDATE_REQUIRES_NATIVE_NARROW_AND_PHASE_B"
            if selected is not None
            else "NO_CANDIDATE_PASSES_FIXED_P060_P080_SCREEN"
        ),
        "interpretation_limit": screen_protocol()["interpretation_limit"],
    }
