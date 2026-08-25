"""Frozen parent/it25/it50 screen for the identical c500 SA5-v3 control."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import sa5_v3_provisional_pilot_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

RUN_NAME = "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1"
RUN_DIR = REPO / "logs/rnn_car" / RUN_NAME
SCREEN_ROOT = (
    REPO
    / "logs/gates/sa5_v3_c500_parent_control_screen/screen_20260823_r1"
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
CORRIDOR_SCENARIOS = base.CORRIDOR_SCENARIOS
LOW_DENSITY_SCENARIOS = base.LOW_DENSITY_SCENARIOS
RETENTION_SCENARIOS = base.RETENTION_SCENARIOS
ALL_SCENARIOS = base.ALL_SCENARIOS
STEPS_BY_SCENARIO = dict(base.STEPS_BY_SCENARIO)

CANDIDATE_SPECS = (
    {
        "name": "parent_c500",
        "conceptual_iteration": 0,
        "path": (
            "logs/rnn_car/sa4_v3_cont300_from_c300_ne1024_s42_p300_r1/"
            "checkpoint_25600.pt"
        ),
        "expected_sha256": (
            "e27677349eba8f37e2e6937f7d92400a1550e58d89d4430e556932697e9cb9f4"
        ),
    },
    {
        "name": "control_it25",
        "conceptual_iteration": 25,
        "path": f"logs/rnn_car/{RUN_NAME}/checkpoint_3200.pt",
        "expected_sha256": None,
    },
    {
        "name": "control_it50",
        "conceptual_iteration": 50,
        "path": f"logs/rnn_car/{RUN_NAME}/checkpoint_6400.pt",
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
CELL_SCHEMA = "sa5_v3_c500_parent_control_screen_cell/v1"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(spec: dict | str) -> Path:
    if isinstance(spec, str):
        candidate = next(
            (row for row in CANDIDATE_SPECS if row["name"] == spec), None
        )
        if candidate is None:
            raise ValueError(f"unknown c500-screen candidate {spec!r}")
        spec = candidate
    return REPO / str(spec["path"])


def _stage_scene() -> dict:
    return base._stage_scene()


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c500_parent_control_screen_protocol/v1",
        "run_name": RUN_NAME,
        "lineage": {
            "sa4": "SA4_NOT_GRADUATED",
            "failed_provisional_parent": "PROVISIONAL_SA5_PARENT_C550",
            "control_parent": "PROVISIONAL_SA5_CONTROL_PARENT_C500",
        },
        "trigger": {
            "summary": (
                "logs/gates/sa5_v3_provisional_pilot_screen/"
                "screen_20260822_r1/SUMMARY.json"
            ),
            "summary_sha256": (
                "4cd82cc3912b7ff70a053ca9203594899d5871e3f72fa65cbbecd6065069f744"
            ),
            "next_action": "RUN_IDENTICAL_C500_PARENT_CONTROL_P50",
        },
        "stage": STAGE,
        "candidates": [dict(row) for row in CANDIDATE_SPECS],
        "scenarios": {
            "sa5_target": {
                "name": TARGET_SCENARIO,
                "density": "4S2D",
                "speed_range_m_s": list(P035),
                "motion_mode": "mixed",
            },
            "retention": [
                "nav_native",
                "narrow_range",
                "corridor_low_0s1d P060 lateral",
                "corridor_low_1s1d P060 lateral",
            ],
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
            "target_speed_range_m_s": list(P035),
            "p060_speed_range_m_s": list(P060),
        },
        "thresholds": {
            "target_descriptive_absolute": dict(TARGET_THRESHOLDS),
            "native": dict(NATIVE_THRESHOLDS),
            "narrow": dict(NARROW_THRESHOLDS),
            "p060": dict(P060_THRESHOLDS),
        },
        "extension_rule": {
            "target_min_cr_improvement": TARGET_MIN_CR_IMPROVEMENT,
            "target_min_pooled_se_z": TARGET_MIN_Z,
            "retention_max_cr_increase": RETENTION_MAX_DEGRADATION,
            "retention_max_sr_decrease": RETENTION_MAX_DEGRADATION,
            "retention_max_to_increase": RETENTION_MAX_DEGRADATION,
            "retention_absolute_gates_required": True,
            "target_absolute_gate_is_descriptive": True,
            "selection": (
                "eligible candidate with lowest target CR, then lowest mean "
                "retention CR, then later iteration"
            ),
        },
        "decision": {
            "on_pass": (
                "authorize extension of selected c500 lineage to total it300"
            ),
            "on_fail": (
                "hold SA5; do not change reward or add iterations without a "
                "new authorization"
            ),
            "screen_starts_training": False,
            "screen_starts_sa6": False,
        },
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "fallback screen; it controls bounded continuation but cannot "
            "rewrite SA4 graduation or authorize SA6"
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
        != "sa5_v3_c500_parent_control_checkpoint_manifest/v1"
        or manifest.get("protocol_sha256") != screen_protocol()["sha256"]
    ):
        raise ValueError("unexpected c500 control checkpoint manifest")
    names = [row["name"] for row in CANDIDATE_SPECS]
    if list((manifest.get("checkpoints") or {}).keys()) != names:
        raise ValueError("c500 control checkpoint manifest order mismatch")
    return manifest


def candidate_by_name(name: str) -> dict:
    spec = next((row for row in CANDIDATE_SPECS if row["name"] == name), None)
    if spec is None:
        raise ValueError(f"unknown c500-screen candidate {name!r}")
    observed = (load_checkpoint_manifest().get("checkpoints") or {}).get(name)
    if not observed:
        raise ValueError(f"c500 checkpoint manifest lacks {name}")
    path = checkpoint_path(spec).resolve()
    if Path(observed.get("path", "")).resolve() != path:
        raise ValueError(f"c500 checkpoint path mismatch for {name}")
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
    raise ValueError(f"unknown c500-screen scenario {scenario!r}")


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
        raise ValueError("invalid c500-screen cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("c500-screen protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = str(payload.get("scenario"))
    if scenario not in ALL_SCENARIOS:
        raise ValueError("unknown c500-screen cell scenario")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("c500-screen checkpoint hash mismatch")
    for key, expected in {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS_BY_SCENARIO[scenario],
    }.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"c500-screen {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("c500-screen LiDAR mode mismatch")
    if payload.get("lidar_distractor_eligibility") != LIDAR_DISTRACTOR_ELIGIBILITY:
        raise ValueError("c500-screen LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(runtime.get("deployment_speed_scale"), "deployment_speed_scale")
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("c500-screen speed-rate runtime mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("c500-screen cell has too few episodes")
    verdict = evaluate_metrics(scenario, metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("c500-screen threshold verdict mismatch")


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


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(CANDIDATE_SPECS) * len(ALL_SCENARIOS)
    if len(payloads) != expected:
        raise ValueError(f"c500 screen needs exactly {expected} cells")
    by_name: dict[str, dict[str, dict]] = {}
    for payload in payloads:
        validate_cell(payload)
        name = str(payload["checkpoint_name"])
        scenario = str(payload["scenario"])
        if scenario in by_name.setdefault(name, {}):
            raise ValueError(f"duplicate c500-screen cell {name}/{scenario}")
        by_name[name][scenario] = payload
    expected_names = [row["name"] for row in CANDIDATE_SPECS]
    if list(by_name) != expected_names:
        raise ValueError("c500-screen candidate order mismatch")
    if any(set(cells) != set(ALL_SCENARIOS) for cells in by_name.values()):
        raise ValueError("c500-screen scenario matrix mismatch")

    parent_cells = by_name["parent_c500"]
    candidate_results = []
    for spec in CANDIDATE_SPECS[1:]:
        name = spec["name"]
        cells = by_name[name]
        target = _target_improvement(
            parent_cells[TARGET_SCENARIO]["metrics"],
            cells[TARGET_SCENARIO]["metrics"],
        )
        retention = {}
        for scenario in RETENTION_SCENARIOS:
            relative = _relative_retention(
                parent_cells[scenario]["metrics"], cells[scenario]["metrics"]
            )
            retention[scenario] = {
                "absolute_pass": bool(cells[scenario]["threshold_pass"]),
                "relative": relative,
                "pass": bool(cells[scenario]["threshold_pass"])
                and relative["pass"],
                "metrics": cells[scenario]["metrics"],
            }
        retention_pass = all(row["pass"] for row in retention.values())
        candidate_results.append(
            {
                "checkpoint_name": name,
                "conceptual_iteration": spec["conceptual_iteration"],
                "target_metrics": cells[TARGET_SCENARIO]["metrics"],
                "target_absolute_pass_descriptive": bool(
                    cells[TARGET_SCENARIO]["threshold_pass"]
                ),
                "target_improvement": target,
                "retention": retention,
                "retention_pass": retention_pass,
                "extension_eligible": target["pass"] and retention_pass,
            }
        )
    eligible = [row for row in candidate_results if row["extension_eligible"]]
    eligible.sort(
        key=lambda row: (
            float(row["target_metrics"]["cr"]),
            sum(float(item["metrics"]["cr"]) for item in row["retention"].values())
            / len(row["retention"]),
            -int(row["conceptual_iteration"]),
        )
    )
    selected = eligible[0]["checkpoint_name"] if eligible else None
    return {
        "schema": "sa5_v3_c500_parent_control_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_CONTROL_SCREEN",
        "protocol_sha256": screen_protocol()["sha256"],
        "parent": {
            "checkpoint_name": "parent_c500",
            "target_metrics": parent_cells[TARGET_SCENARIO]["metrics"],
            "retention_metrics": {
                scenario: parent_cells[scenario]["metrics"]
                for scenario in RETENTION_SCENARIOS
            },
        },
        "candidate_results": candidate_results,
        "selected_extension_checkpoint": selected,
        "extension_authorized": selected is not None,
        "next_action": (
            "EXTEND_SELECTED_C500_LINEAGE_TO_TOTAL_IT300"
            if selected is not None
            else "HOLD_SA5_REDESIGN_NO_REWARD_CHANGE_NO_EXTRA_ITERATIONS"
        ),
        "sa4_graduated": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["evidence_boundary"],
    }
