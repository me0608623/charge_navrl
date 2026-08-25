"""Frozen 8-cell direction decomposition: c50 anchor vs pilot it50 at 4S2D.

固定高密度走廊（4 靜態 + 2 行人），只變動行人運動型態與 checkpoint，
用來回答「究竟是哪一類行人運動沒有學會」。不訓練。

⚠️ 讀法（預先登記）：主指標為 **TO 與 SR**，CR 為輔。
   pilot screen 的 pooled target CR 變化僅 1.1 SE（雜訊內），
   而 TO 變化是 6.6 SE；只讀 CR 會得到「哪裡都沒變」的假結論。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import sa5_v3_provisional_pilot_screen as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

SCREEN_ROOT = REPO / "logs/gates/sa5_v3_c50_it50_direction_ab/screen_20260823_r1"
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"

AUTHORIZATION = (
    REPO / "docs/freeze/sa5_v3_c50_it50_direction_ab_authorization_20260823.json"
)
AUTHORIZATION_SHA256 = (
    "5dc349cf58f0168608715364e9261634ad050e9b94b53000e161fc8f88df3308"
)

# ---------------------------------------------------------------- 固定評測條件
# 與 difficulty-frontier / frontier pilot screen 逐項相同，確保三份結果可直接互比。
STAGE = 5
SEED = 818
NUM_ENVS = 64
STEPS = 2500
MIN_EPISODES = 1000
DELAY_STEPS = 1
DELAY_MS = 200
ACTUATOR_PROFILE = "sa1_delay_only"
LIDAR_NOISE_MODE = "full"
LIDAR_DISTRACTOR_ELIGIBILITY = "valid_return_only"
SPEED_RATE = 0.7
SPEED_RATE_OBS = "ego"
DEPLOYMENT_SPEED_SCALE = 1.0
DYNAMIC_SPEED_RANGE = (0.25, 0.45)

STATIC_COUNT = 4
DYNAMIC_COUNT = 2
MOTION_MODES = ("lateral", "longitudinal", "random_2d", "mixed")

ANCHOR = {
    "name": "anchor_c50",
    "conceptual_iteration": 50,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1/"
        "checkpoint_6400.pt"
    ),
    "expected_sha256": (
        "4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c"
    ),
    "status": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
}
PILOT_IT50 = {
    "name": "pilot_it50",
    "conceptual_iteration": 50,
    "path": (
        "logs/rnn_car/"
        "sa5_v3_c50_frontier_curriculum_phase2_ne1024_s42_p25_r1/"
        "checkpoint_3200.pt"
    ),
    "expected_sha256": (
        "16da26108ab8a61e2b7d440d90830df6ad69e91a97e31c56eaab7ecc92396ce3"
    ),
    "status": "REJECTED_PILOT_NOT_FORMAL_PARENT",
}
CANDIDATE_SPECS = (ANCHOR, PILOT_IT50)
_BY_NAME = {spec["name"]: spec for spec in CANDIDATE_SPECS}

# anchor 在 difficulty-frontier 已量過的同條件值 —— 用於決定論複驗（報告，不 fail-closed）
ANCHOR_REFERENCE = {
    "source": "logs/gates/sa5_v3_c50_difficulty_frontier/screen_20260823_r1",
    "cells": {
        "lateral": {"n": 1712, "cr": 0.6022196261682243},
        "longitudinal": {"n": 1830, "cr": 0.8147540983606557},
        "random_2d": {"n": 1333, "cr": 0.7366841710427607},
        "mixed": {"n": 1421, "cr": 0.5601688951442646},
    },
}

# 行為指標 —— 讀法規則要求每格必報（pilot screen 已證實這三項才是變化的直接證據）
BEHAVIOR_KEYS = (
    "stop_command_fraction",
    "linear_speed_abs_mean_mps",
    "reverse_command_fraction",
)

# 顯著性門檻：主指標 TO / SR，CR 為輔
MIN_ABS_Z = 2.0
MIN_ABS_DELTA = 0.005


def _scenario_name(mode: str) -> str:
    return f"s{STATIC_COUNT}_d{DYNAMIC_COUNT}_{mode}"


SCENARIO_SPECS = tuple(
    {
        "name": _scenario_name(mode),
        "static_obstacles": STATIC_COUNT,
        "dynamic_obstacles": DYNAMIC_COUNT,
        "motion_mode": mode,
    }
    for mode in MOTION_MODES
)
ALL_SCENARIOS = tuple(row["name"] for row in SCENARIO_SPECS)
CORRIDOR_SCENARIOS = ALL_SCENARIOS
LOW_DENSITY_SCENARIOS: tuple[str, ...] = ()
STEPS_BY_SCENARIO = {name: STEPS for name in ALL_SCENARIOS}

THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.10,
    "to_max": 0.05,
}
CELL_SCHEMA = "sa5_v3_c50_it50_direction_ab_cell/v1"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(spec: dict | str = ANCHOR) -> Path:
    if isinstance(spec, str):
        spec = _BY_NAME.get(spec)
        if spec is None:
            raise ValueError("unknown direction-AB checkpoint")
    return REPO / str(spec["path"])


def candidate_by_name(name: str) -> dict:
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError(f"unknown direction-AB checkpoint {name!r}")
    result = dict(spec)
    result["sha256"] = spec["expected_sha256"]
    return result


def scenario_by_name(name: str) -> dict:
    result = next((row for row in SCENARIO_SPECS if row["name"] == name), None)
    if result is None:
        raise ValueError(f"unknown direction-AB scenario {name!r}")
    return dict(result)


def _stage_scene() -> dict:
    values = base._stage_scene()
    values["corridor_speed_range_m_s"] = list(DYNAMIC_SPEED_RANGE)
    return values


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c50_it50_direction_ab_protocol/v1",
        "authorization": {
            "path": str(AUTHORIZATION.relative_to(REPO)),
            "sha256": AUTHORIZATION_SHA256,
        },
        "question": (
            "which pedestrian motion mode was not learned, holding the "
            "high-density corridor fixed at 4S2D"
        ),
        "lineage": {
            "sa4": "SA4_NOT_GRADUATED",
            "anchor": dict(ANCHOR),
            "pilot_it25": "REJECTED",
            "pilot_it50": "REJECTED",
            "sa6": "HOLD_NOT_AUTHORIZED",
        },
        "matrix": {
            "static_obstacles": STATIC_COUNT,
            "dynamic_obstacles": DYNAMIC_COUNT,
            "motion_modes": list(MOTION_MODES),
            "checkpoints": [spec["name"] for spec in CANDIDATE_SPECS],
            "cells": [dict(row) for row in SCENARIO_SPECS],
            "cell_count": len(SCENARIO_SPECS) * len(CANDIDATE_SPECS),
        },
        "fixed_evaluation": {
            "stage": STAGE,
            "seed": SEED,
            "num_envs": NUM_ENVS,
            "steps": STEPS,
            "minimum_episodes": MIN_EPISODES,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": DELAY_MS,
            "actuator_profile": ACTUATOR_PROFILE,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "lidar_noise_mode": LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": LIDAR_DISTRACTOR_ELIGIBILITY,
            "dynamic_speed_range_m_s": list(DYNAMIC_SPEED_RANGE),
            "scene": _stage_scene(),
        },
        "thresholds": dict(THRESHOLDS),
        "reading_rules": {
            "primary_metrics": ["to", "sr"],
            "secondary_metrics": ["cr"],
            "rationale": (
                "frontier pilot screen pooled target CR moved only 1.1 SE "
                "(within noise) while TO moved 6.6 SE; reading CR alone "
                "produces a false 'nothing changed' conclusion"
            ),
            "min_abs_z": MIN_ABS_Z,
            "min_abs_delta": MIN_ABS_DELTA,
            "required_behavior_metrics": list(BEHAVIOR_KEYS),
        },
        "anchor_replication": {
            "reference": dict(ANCHOR_REFERENCE),
            "handling": (
                "reported, not fail-closed; a mismatch means the evaluator is "
                "non-deterministic and must be investigated before the new "
                "anchor values replace the frontier values"
            ),
        },
        "decision_rules": {
            "direction_specific_degradation": (
                "at least one motion mode degrades significantly on a primary "
                "metric while at least one other mode does not"
            ),
            "no_direction_improved": (
                "no motion mode improves significantly on SR"
            ),
            "on_direction_specific": "RAISE_EXPOSURE_FOR_DEGRADED_DIRECTIONS_ONLY",
            "on_no_improvement": (
                "RESTART_FROM_C50_STAGED_1D_TO_2D_THEN_1S_2S_4S_"
                "NO_SIMULTANEOUS_TWO_AXIS_INCREASE"
            ),
        },
        "screen_starts_training": False,
        "screen_starts_sa6": False,
        "screen_authorizes_graduation": False,
        "interpretation_limit": (
            "single evaluator seed evaluation-only direction decomposition; "
            "it cannot graduate SA5, formalize any parent, or authorize SA6"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def thresholds_for(_: str) -> dict:
    return dict(THRESHOLDS)


def _finite(value: object, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def evaluate_metrics(_: str, metrics: dict) -> dict:
    checks = {
        "episodes": int(metrics.get("n", 0)) >= MIN_EPISODES,
        "sr": _finite(metrics.get("sr"), "sr") >= THRESHOLDS["sr_min"],
        "cr": _finite(metrics.get("cr"), "cr") <= THRESHOLDS["cr_max"],
        "to": _finite(metrics.get("to"), "to") <= THRESHOLDS["to_max"],
    }
    return {
        "thresholds": dict(THRESHOLDS),
        "checks": checks,
        "threshold_pass": all(checks.values()),
    }


def validate_cell(payload: dict) -> None:
    if payload.get("schema") != CELL_SCHEMA or not payload.get("cell_valid"):
        raise ValueError("invalid direction-AB cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("direction-AB protocol hash mismatch")
    name = str(payload.get("checkpoint_name"))
    spec = _BY_NAME.get(name)
    if spec is None:
        raise ValueError("direction-AB checkpoint name mismatch")
    if payload.get("checkpoint_sha256") != spec["expected_sha256"]:
        raise ValueError(f"direction-AB {name} checkpoint hash mismatch")
    scenario = scenario_by_name(str(payload.get("scenario")))
    expected_scalars = {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS,
    }
    for key, expected in expected_scalars.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"direction-AB {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("direction-AB LiDAR mode mismatch")
    if payload.get("lidar_distractor_eligibility") != LIDAR_DISTRACTOR_ELIGIBILITY:
        raise ValueError("direction-AB LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(runtime.get("deployment_speed_scale"), "deployment_speed_scale")
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("direction-AB speed runtime mismatch")
    contract = payload.get("scene_contract") or {}
    if contract.get("fixed_density") != f"{STATIC_COUNT}S{DYNAMIC_COUNT}D":
        raise ValueError("direction-AB density mismatch")
    if contract.get("motion_mode") != scenario["motion_mode"]:
        raise ValueError("direction-AB motion mode mismatch")
    report = payload.get("corridor_report") or {}
    if (
        int(report.get("configured_static_obstacles", -1)) != STATIC_COUNT
        or int(report.get("configured_dynamic_obstacles", -1)) != DYNAMIC_COUNT
    ):
        raise ValueError("direction-AB corridor count ledger mismatch")
    observed_speed = tuple(
        float(value)
        for value in report.get("requested_dynamic_speed_range_m_s", ())
    )
    if observed_speed != DYNAMIC_SPEED_RANGE:
        raise ValueError("direction-AB dynamic speed mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("direction-AB cell has too few episodes")
    if int(report.get("static_layout_episodes_total", -1)) != int(metrics["n"]):
        raise ValueError("direction-AB static layout ledger mismatch")
    layouts = report.get("static_layout_outcomes") or {}
    if set(layouts) != {"405", "410"}:
        raise ValueError("4S cell must contain layouts 405 and 410")
    if any(int(row.get("episodes", 0)) < 100 for row in layouts.values()):
        raise ValueError("4S layout subgroup has too few episodes")
    verdict = evaluate_metrics(str(payload["scenario"]), metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("direction-AB threshold verdict mismatch")


def _se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _compare(label: str, before: dict, after: dict) -> dict:
    p0, p1 = before[label], after[label]
    delta = p1 - p0
    se = math.hypot(_se(p0, before["n"]), _se(p1, after["n"]))
    z = delta / se if se > 0 else 0.0
    return {
        "metric": label,
        "anchor": p0,
        "it50": p1,
        "delta": delta,
        "se_diff": se,
        "z": z,
        "significant": abs(z) >= MIN_ABS_Z and abs(delta) >= MIN_ABS_DELTA,
    }


def _behavior(payload: dict) -> dict:
    report = payload.get("corridor_report") or {}
    out = {}
    for key in BEHAVIOR_KEYS:
        value = report.get(key)
        if value is None:
            value = report.get("pre_deployment_" + key)
        out[key] = None if value is None else float(value)
    return out


def _row(payload: dict) -> dict:
    validate_cell(payload)
    scenario = scenario_by_name(payload["scenario"])
    metrics = payload["metrics"]
    report = payload["corridor_report"]
    return {
        "checkpoint_name": payload["checkpoint_name"],
        "motion_mode": scenario["motion_mode"],
        "scenario": scenario["name"],
        "n": int(metrics["n"]),
        "sr": _finite(metrics["sr"], "sr"),
        "cr": _finite(metrics["cr"], "cr"),
        "to": _finite(metrics["to"], "to"),
        "gate_pass": bool(payload["threshold_pass"]),
        "wall_cr": _finite(report["wall_collision_rate"], "wall_cr"),
        "obstacle_cr": _finite(report["obstacle_collision_rate"], "obstacle_cr"),
        "behavior": _behavior(payload),
        "layouts": report.get("static_layout_outcomes") or {},
    }


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(SCENARIO_SPECS) * len(CANDIDATE_SPECS)
    if len(payloads) != expected:
        raise ValueError(f"expected exactly {expected} direction-AB cells")
    rows = [_row(payload) for payload in payloads]
    by_key = {(row["checkpoint_name"], row["motion_mode"]): row for row in rows}
    if len(by_key) != expected:
        raise ValueError("direction-AB cells are duplicated or missing")

    # ---- anchor 決定論複驗（報告，不 fail-closed）
    replication = []
    for mode, ref in ANCHOR_REFERENCE["cells"].items():
        row = by_key[(ANCHOR["name"], mode)]
        replication.append(
            {
                "motion_mode": mode,
                "reference_cr": ref["cr"],
                "observed_cr": row["cr"],
                "reference_n": ref["n"],
                "observed_n": row["n"],
                "identical": row["n"] == ref["n"]
                and abs(row["cr"] - ref["cr"]) < 1e-12,
            }
        )
    replication_exact = all(item["identical"] for item in replication)

    # ---- 逐方向 A/B（主指標 TO 與 SR）
    directions = []
    for mode in MOTION_MODES:
        a = by_key[(ANCHOR["name"], mode)]
        b = by_key[(PILOT_IT50["name"], mode)]
        to_cmp = _compare("to", a, b)
        sr_cmp = _compare("sr", a, b)
        cr_cmp = _compare("cr", a, b)
        degraded = (to_cmp["significant"] and to_cmp["delta"] > 0) or (
            sr_cmp["significant"] and sr_cmp["delta"] < 0
        )
        improved = sr_cmp["significant"] and sr_cmp["delta"] > 0
        directions.append(
            {
                "motion_mode": mode,
                "anchor": {k: a[k] for k in ("n", "sr", "cr", "to", "behavior")},
                "it50": {k: b[k] for k in ("n", "sr", "cr", "to", "behavior")},
                "primary": {"to": to_cmp, "sr": sr_cmp},
                "secondary": {"cr": cr_cmp},
                "degraded": degraded,
                "improved": improved,
            }
        )

    degraded_modes = [d["motion_mode"] for d in directions if d["degraded"]]
    improved_modes = [d["motion_mode"] for d in directions if d["improved"]]
    direction_specific = bool(degraded_modes) and len(degraded_modes) < len(MOTION_MODES)

    if not improved_modes:
        next_action = (
            "RESTART_FROM_C50_STAGED_1D_TO_2D_THEN_1S_2S_4S_"
            "NO_SIMULTANEOUS_TWO_AXIS_INCREASE"
        )
    elif direction_specific:
        next_action = "RAISE_EXPOSURE_FOR_DEGRADED_DIRECTIONS_ONLY"
    else:
        next_action = "HOLD_FOR_MANUAL_DIRECTION_INTERPRETATION"

    worst = max(directions, key=lambda d: d["it50"]["cr"])
    worst_to = max(directions, key=lambda d: d["it50"]["to"])

    return {
        "schema": "sa5_v3_c50_it50_direction_ab_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_EVALUATION_ONLY_SCREEN",
        "anchor_status": ANCHOR["status"],
        "pilot_status": PILOT_IT50["status"],
        "rows": sorted(
            rows,
            key=lambda row: (
                row["checkpoint_name"],
                MOTION_MODES.index(row["motion_mode"]),
            ),
        ),
        "directions": directions,
        "degraded_motion_modes": degraded_modes,
        "improved_motion_modes": improved_modes,
        "direction_specific_degradation": direction_specific,
        "no_direction_improved": not improved_modes,
        "worst_it50_cr_mode": worst["motion_mode"],
        "worst_it50_to_mode": worst_to["motion_mode"],
        "anchor_replication": replication,
        "anchor_replication_exact": replication_exact,
        "next_action": next_action,
        "selected_extension_checkpoint": None,
        "selected_development_checkpoint": None,
        "training_started": False,
        "sa6_started": False,
        "graduation_authorized": False,
        "interpretation_limit": screen_protocol()["interpretation_limit"],
    }
