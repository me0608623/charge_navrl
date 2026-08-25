"""Frozen 24-cell screen for the bounded B3 it25-to-it50 continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sa5_v3_c50_stage3_phase_a_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCREEN_ROOT = REPO / "logs/gates/sa5_v3_c50_stage3_b3_cont25_screen/screen_20260824_r2"
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"
CHECKPOINT_LOCK = (
    REPO / "docs/freeze/sa5_v3_c50_stage3_b3_cont25_screen_checkpoints_v1.json"
)
AUTHORIZATION = (
    REPO / "docs/freeze/sa5_v3_c50_stage3_b3_cont25_authorization_20260824.json"
)
AUTHORIZATION_SHA256 = (
    "664ecb32476ccffdf6bbd337e27c39f72222ebe2aa35ec43d27065005c052f13"
)

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
P060 = base.P060
P080 = base.P080
P060_SCENARIOS = base.P060_SCENARIOS
P080_SCENARIOS = base.P080_SCENARIOS
ALL_SCENARIOS = base.ALL_SCENARIOS
LOW_DENSITY_SCENARIOS = ALL_SCENARIOS
CORRIDOR_SCENARIOS: tuple[str, ...] = ()
TARGET_SCENARIO = "__bounded_continuation_has_no_target_cell__"
STEPS_BY_SCENARIO = dict(base.STEPS_BY_SCENARIO)
THRESHOLDS = dict(base.THRESHOLDS)
MAX_DEGRADATION = 0.02
TIE_BAND = 0.005
CELL_SCHEMA = "sa5_v3_c50_stage3_b3_cont25_screen_cell/v1"

BASELINE = {
    "name": "b3_it25_baseline",
    "conceptual_iteration": 25,
    "path": (
        "logs/rnn_car/sa5_v3_c50_stage3_b3_p060aligned_"
        "ne1024_s42_p25_r1/checkpoint_3200.pt"
    ),
    "embedded_iteration": 24,
    "embedded_total_steps": 3200,
    "expected_sha256": (
        "5645f84625649a6050e862eb17b8e81e2e02225973fd371f29e663785104f2b8"
    ),
}
_CONT_RUN = (
    "logs/rnn_car/sa5_v3_c50_stage3_b3_cont25_from_it25_"
    "ne1024_s42_p25_r1"
)
_CONTINUATIONS = (
    ("it30", 30, "checkpoint_640.pt", 4, 640),
    ("it35", 35, "checkpoint_1280.pt", 9, 1280),
    ("it40", 40, "checkpoint_1920.pt", 14, 1920),
    ("it45", 45, "checkpoint_2560.pt", 19, 2560),
    ("it50", 50, "checkpoint_3200.pt", 24, 3200),
)


def _load_checkpoint_lock() -> dict:
    if not CHECKPOINT_LOCK.is_file():
        raise FileNotFoundError(
            "continuation checkpoint lock does not exist; training must finish first: "
            f"{CHECKPOINT_LOCK}"
        )
    payload = json.loads(CHECKPOINT_LOCK.read_text(encoding="utf-8"))
    if payload.get("schema") != "sa5_v3_c50_stage3_b3_cont25_checkpoint_lock/v1":
        raise RuntimeError("invalid B3 continuation checkpoint lock schema")
    return payload


def _candidate_specs() -> tuple[dict, ...]:
    lock = _load_checkpoint_lock()
    locked = lock.get("checkpoints") or {}
    rows = [dict(BASELINE)]
    for name, conceptual, filename, embedded_iteration, embedded_steps in _CONTINUATIONS:
        entry = locked.get(name) or {}
        expected_path = f"{_CONT_RUN}/{filename}"
        if (
            entry.get("path") != expected_path
            or int(entry.get("conceptual_iteration", -1)) != conceptual
            or int(entry.get("embedded_iteration", -1)) != embedded_iteration
            or int(entry.get("embedded_total_steps", -1)) != embedded_steps
            or not isinstance(entry.get("sha256"), str)
        ):
            raise RuntimeError(f"checkpoint lock mismatch for {name}")
        rows.append(
            {
                "name": name,
                "conceptual_iteration": conceptual,
                "path": expected_path,
                "embedded_iteration": embedded_iteration,
                "embedded_total_steps": embedded_steps,
                "expected_sha256": entry["sha256"],
            }
        )
    return tuple(rows)


def candidate_specs() -> tuple[dict, ...]:
    return _candidate_specs()


def candidate_by_name(name: str) -> dict:
    for spec in _candidate_specs():
        if spec["name"] == name:
            result = dict(spec)
            result["sha256"] = spec["expected_sha256"]
            return result
    raise ValueError(f"unknown B3 continuation candidate {name!r}")


def checkpoint_path(spec: dict | str = BASELINE) -> Path:
    if isinstance(spec, str):
        spec = candidate_by_name(spec)
    return REPO / str(spec["path"])


speed_range_for = base.speed_range_for
density_for = base.density_for
thresholds_for = base.thresholds_for
evaluate_metrics = base.evaluate_metrics
sha256_of = base.sha256_of
_stage_scene = base._stage_scene
_finite = base._finite


def screen_protocol() -> dict:
    specs = _candidate_specs()
    lock_hash = sha256_of(CHECKPOINT_LOCK)
    payload = {
        "schema": "sa5_v3_c50_stage3_b3_cont25_screen_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
        },
        "checkpoint_lock": {
            "path": str(CHECKPOINT_LOCK.relative_to(REPO)),
            "sha256": lock_hash,
        },
        "question": (
            "does exact B3 continuation produce a checkpoint that passes both "
            "P060 low-density gates without more than 2 pp P080 regression"
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
            "scene": _stage_scene(),
        },
        "acceptance": {
            "p060_absolute_thresholds": dict(THRESHOLDS),
            "p080_maximum_sr_cr_to_degradation": MAX_DEGRADATION,
            "baseline": BASELINE["name"],
            "tie_band": TIE_BAND,
        },
        "decision": {
            "if_none_pass": (
                "STOP_ITERATION_STACKING_AND_DESIGN_0S1D_TO_1S1D_"
                "DENSITY_CURRICULUM"
            ),
            "if_pass": (
                "SA5_CANDIDATE_ONLY_REQUIRES_SEPARATE_NATIVE_NARROW_"
                "AND_PHASE_B"
            ),
            "starts_training": False,
            "starts_phase_b": False,
            "selects_parent": False,
            "graduates_sa5": False,
            "starts_sa6": False,
        },
        "interpretation_limit": (
            "single training seed and single evaluator seed development screen; "
            "a passing checkpoint is only an SA5 candidate"
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != CELL_SCHEMA or not payload.get("cell_valid"):
        raise ValueError("invalid B3 continuation cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("B3 continuation protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = str(payload.get("scenario"))
    expected_speed = speed_range_for(scenario)
    expected_density = density_for(scenario)
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("B3 continuation checkpoint hash mismatch")
    for key, expected in {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS_BY_SCENARIO[scenario],
    }.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"B3 continuation {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("B3 continuation LiDAR mode mismatch")
    if payload.get("lidar_distractor_eligibility") != LIDAR_DISTRACTOR_ELIGIBILITY:
        raise ValueError("B3 continuation LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(runtime.get("deployment_speed_scale"), "deployment_speed_scale")
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("B3 continuation speed-rate runtime mismatch")
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
        raise ValueError("B3 continuation speed/density runtime mismatch")
    if report.get("dynamic_motion_mode") != "lateral":
        raise ValueError("B3 continuation motion-mode mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("B3 continuation cell has too few episodes")
    verdict = evaluate_metrics(scenario, metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("B3 continuation threshold verdict mismatch")


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
    specs = _candidate_specs()
    expected = len(specs) * len(ALL_SCENARIOS)
    if len(payloads) != expected:
        raise ValueError(f"B3 continuation screen needs exactly {expected} cells")
    by_name: dict[str, dict[str, dict]] = {}
    for payload in payloads:
        validate_cell(payload)
        name = str(payload["checkpoint_name"])
        scenario = str(payload["scenario"])
        if scenario in by_name.setdefault(name, {}):
            raise ValueError(f"duplicate B3 continuation cell {name}/{scenario}")
        by_name[name][scenario] = payload
    expected_names = [row["name"] for row in specs]
    if list(by_name) != expected_names:
        raise ValueError("B3 continuation candidate order mismatch")
    if any(set(rows) != set(ALL_SCENARIOS) for rows in by_name.values()):
        raise ValueError("B3 continuation scenario matrix mismatch")

    baseline_rows = {
        scenario: _row(by_name[BASELINE["name"]][scenario])
        for scenario in ALL_SCENARIOS
    }
    results = []
    passing = []
    for spec in specs:
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
            key=lambda row: (
                row["mean_p060_cr"],
                row["conceptual_iteration"],
            )
        )
        selected = contenders[0]["checkpoint_name"]
        tie_group = [row["checkpoint_name"] for row in contenders]

    return {
        "schema": "sa5_v3_c50_stage3_b3_cont25_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_TRAINING_SEED_24_CELL_SCREEN",
        "protocol_sha256": screen_protocol()["sha256"],
        "baseline": BASELINE["name"],
        "candidate_results": results,
        "passing_candidates": [
            row["checkpoint_name"] for row in passing
        ],
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
            else "STOP_ITERATION_STACKING_AND_DESIGN_0S1D_TO_1S1D_DENSITY_CURRICULUM"
        ),
        "interpretation_limit": screen_protocol()["interpretation_limit"],
    }
