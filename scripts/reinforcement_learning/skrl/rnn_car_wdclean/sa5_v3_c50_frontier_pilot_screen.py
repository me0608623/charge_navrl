"""Frozen acceptance contract for the c50 difficulty-frontier pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

ANCHOR_RUN = "sa5_v3_c500_parent_control_from_sa4v3_ne1024_s42_p50_r1"
PHASE1_RUN = "sa5_v3_c50_frontier_curriculum_phase1_ne1024_s42_p25_r1"
PHASE2_RUN = "sa5_v3_c50_frontier_curriculum_phase2_ne1024_s42_p25_r1"
SCREEN_ROOT = (
    REPO
    / "logs/gates/sa5_v3_c50_frontier_pilot_screen/screen_20260823_r1"
)
CHECKPOINT_MANIFEST = SCREEN_ROOT / "CHECKPOINT_MANIFEST.json"

STAGE = 5
SEED = 818
NUM_ENVS = 64
MIN_EPISODES = 1000
DELAY_STEPS = 1
DELAY_MS = 200
ACTUATOR_PROFILE = "sa1_delay_only"
LIDAR_NOISE_MODE = "full"
LIDAR_DISTRACTOR_ELIGIBILITY = "valid_return_only"
SPEED_RATE = 0.7
SPEED_RATE_OBS = "ego"
DEPLOYMENT_SPEED_SCALE = 1.0
P035 = (0.25, 0.45)
P060 = (0.50, 0.70)

TARGET_SCENARIO = "corridor_mixed"
CORRIDOR_SCENARIOS = (TARGET_SCENARIO,)
LOW_DENSITY_SCENARIOS = (
    "corridor_low_0s1d",
    "corridor_low_1s1d",
)
RETENTION_SCENARIOS = (
    "nav_native",
    "narrow_range",
    *LOW_DENSITY_SCENARIOS,
)
ALL_SCENARIOS = (TARGET_SCENARIO, *RETENTION_SCENARIOS)
STEPS_BY_SCENARIO = {
    TARGET_SCENARIO: 2500,
    "nav_native": 1200,
    "narrow_range": 1200,
    "corridor_low_0s1d": 2500,
    "corridor_low_1s1d": 2500,
}

CANDIDATE_SPECS = (
    {
        "name": "anchor_c50",
        "conceptual_iteration": 0,
        "path": f"logs/rnn_car/{ANCHOR_RUN}/checkpoint_6400.pt",
        "expected_sha256": (
            "4bc1744bb134688179ad2dfdc858bec245194d205ce61570f94bd2a0d726a99c"
        ),
    },
    {
        "name": "pilot_it25",
        "conceptual_iteration": 25,
        "path": f"logs/rnn_car/{PHASE1_RUN}/checkpoint_3200.pt",
        "expected_sha256": (
            "c120ef3c892030019af377f2932b1ddf6c8253071b7c98513c3af7f28cd7cb9f"
        ),
    },
    {
        "name": "pilot_it50",
        "conceptual_iteration": 50,
        "path": f"logs/rnn_car/{PHASE2_RUN}/checkpoint_3200.pt",
        "expected_sha256": None,
    },
)

TARGET_THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.10,
    "to_max": 0.05,
}
NATIVE_THRESHOLDS = dict(TARGET_THRESHOLDS)
NARROW_THRESHOLDS = {
    "episodes_min": MIN_EPISODES,
    "sr_min": 0.90,
    "cr_max": 0.05,
    "to_max": 0.05,
    "crossing_min": 0.95,
    "direct_crossing_min": 0.95,
}
P060_THRESHOLDS = dict(TARGET_THRESHOLDS)

TARGET_MIN_CR_IMPROVEMENT = 0.005
TARGET_MIN_SR_IMPROVEMENT = 0.005
TARGET_MIN_Z = 2.0
RETENTION_MAX_DEGRADATION = 0.02
CELL_SCHEMA = "sa5_v3_c50_frontier_pilot_screen_cell/v1"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(spec: dict | str) -> Path:
    if isinstance(spec, str):
        spec = next(
            (row for row in CANDIDATE_SPECS if row["name"] == spec), None
        )
        if spec is None:
            raise ValueError(f"unknown frontier-pilot candidate {spec!r}")
    return REPO / str(spec["path"])


def _stage_scene() -> dict:
    import sys

    skrl_root = HERE.parent
    if str(skrl_root) not in sys.path:
        sys.path.insert(0, str(skrl_root))
    from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
        make_sim2real_curriculum_config,
    )
    from sa3_sa8_acceptance_contract import stage_scene_contract

    values = stage_scene_contract(STAGE)
    values["room_half_extent_m"] = float(values["arena_size_m"]) / 2.0
    values["corridor_wall_span_m"] = float(values["arena_size_m"])
    values["corridor_boundary_wall_width_m"] = 1.0
    values["corridor_expected_boundary_overlap_m"] = 0.5
    values["corridor_sealed_to_boundary"] = True
    cfg = make_sim2real_curriculum_config(
        STAGE, checkpoint="__SA5_V3_FRONTIER_SCREEN_SCENE_PROBE__.pt"
    )
    values["narrow_segment_length_m"] = float(
        cfg.narrow_passage_segment_length
    )
    return json.loads(json.dumps(values))


def screen_protocol() -> dict:
    payload = {
        "schema": "sa5_v3_c50_frontier_pilot_screen_protocol/v1",
        "authorization": (
            "docs/freeze/"
            "sa5_v3_c50_frontier_curriculum_p50_authorization_20260823.json"
        ),
        "lineage": {
            "sa5": "HOLD_NOT_GRADUATED",
            "anchor": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
            "sa6": "HOLD_NOT_AUTHORIZED",
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
        "pilot_acceptance": {
            "target_min_cr_improvement": TARGET_MIN_CR_IMPROVEMENT,
            "target_min_sr_improvement": TARGET_MIN_SR_IMPROVEMENT,
            "target_min_pooled_se_z": TARGET_MIN_Z,
            "target_to_max": TARGET_THRESHOLDS["to_max"],
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
            "screen_starts_training": False,
            "screen_starts_sa6": False,
            "screen_authorizes_graduation": False,
            "on_pass": "await human decision; no automatic extension",
            "on_fail": "hold lineage as diagnostic evidence",
        },
        "evidence_boundary": (
            "single training seed and single evaluator seed developmental "
            "screen; it cannot graduate SA5, formalize c50 as a parent, or "
            "authorize SA6"
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
        != "sa5_v3_c50_frontier_pilot_checkpoint_manifest/v1"
        or manifest.get("protocol_sha256") != screen_protocol()["sha256"]
    ):
        raise ValueError("unexpected frontier-pilot checkpoint manifest")
    names = [row["name"] for row in CANDIDATE_SPECS]
    if list((manifest.get("checkpoints") or {}).keys()) != names:
        raise ValueError("frontier-pilot checkpoint manifest order mismatch")
    return manifest


def candidate_by_name(name: str) -> dict:
    spec = next((row for row in CANDIDATE_SPECS if row["name"] == name), None)
    if spec is None:
        raise ValueError(f"unknown frontier-pilot candidate {name!r}")
    observed = (load_checkpoint_manifest().get("checkpoints") or {}).get(name)
    if not observed:
        raise ValueError(f"frontier-pilot manifest lacks {name}")
    path = checkpoint_path(spec).resolve()
    if Path(observed.get("path", "")).resolve() != path:
        raise ValueError(f"frontier-pilot checkpoint path mismatch for {name}")
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
    raise ValueError(f"unknown frontier-pilot scenario {scenario!r}")


def _finite(value: object, label: str) -> float:
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
        raise ValueError("invalid frontier-pilot cell schema or validity")
    if payload.get("protocol_sha256") != screen_protocol()["sha256"]:
        raise ValueError("frontier-pilot protocol hash mismatch")
    candidate = candidate_by_name(str(payload.get("checkpoint_name")))
    scenario = str(payload.get("scenario"))
    if scenario not in ALL_SCENARIOS:
        raise ValueError("unknown frontier-pilot cell scenario")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("frontier-pilot checkpoint hash mismatch")
    for key, expected in {
        "geometry_stage": STAGE,
        "seed": SEED,
        "delay_steps": DELAY_STEPS,
        "num_envs": NUM_ENVS,
        "steps": STEPS_BY_SCENARIO[scenario],
    }.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"frontier-pilot {key} mismatch")
    if payload.get("lidar_noise_mode") != LIDAR_NOISE_MODE:
        raise ValueError("frontier-pilot LiDAR mode mismatch")
    if payload.get("lidar_distractor_eligibility") != (
        LIDAR_DISTRACTOR_ELIGIBILITY
    ):
        raise ValueError("frontier-pilot LiDAR eligibility mismatch")
    runtime = payload.get("speed_rate_runtime") or {}
    if (
        _finite(runtime.get("speed_rate"), "speed_rate") != SPEED_RATE
        or runtime.get("speed_rate_obs") != SPEED_RATE_OBS
        or _finite(
            runtime.get("deployment_speed_scale"), "deployment_speed_scale"
        )
        != DEPLOYMENT_SPEED_SCALE
    ):
        raise ValueError("frontier-pilot speed-rate runtime mismatch")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("frontier-pilot cell has too few episodes")
    verdict = evaluate_metrics(scenario, metrics)
    if bool(payload.get("threshold_pass")) != verdict["threshold_pass"]:
        raise ValueError("frontier-pilot threshold verdict mismatch")


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
    parent_cr = float(parent["cr"])
    candidate_cr = float(candidate["cr"])
    parent_sr = float(parent["sr"])
    candidate_sr = float(candidate["sr"])
    candidate_to = float(candidate["to"])
    n0, n1 = int(parent["n"]), int(candidate["n"])
    se = math.sqrt(
        parent_cr * (1.0 - parent_cr) / n0
        + candidate_cr * (1.0 - candidate_cr) / n1
    )
    cr_improvement = parent_cr - candidate_cr
    sr_improvement = candidate_sr - parent_sr
    z = cr_improvement / se if se > 0.0 else (
        math.inf if cr_improvement > 0.0 else 0.0
    )
    checks = {
        "minimum_cr_improvement": (
            cr_improvement >= TARGET_MIN_CR_IMPROVEMENT
        ),
        "minimum_sr_improvement": (
            sr_improvement >= TARGET_MIN_SR_IMPROVEMENT
        ),
        "minimum_z": z >= TARGET_MIN_Z,
        "timeout": candidate_to <= TARGET_THRESHOLDS["to_max"],
    }
    return {
        "parent_cr": parent_cr,
        "candidate_cr": candidate_cr,
        "cr_improvement": cr_improvement,
        "parent_sr": parent_sr,
        "candidate_sr": candidate_sr,
        "sr_improvement": sr_improvement,
        "candidate_to": candidate_to,
        "pooled_se": se,
        "z": z,
        "checks": checks,
        "pass": all(checks.values()),
    }


def final_verdict(payloads: list[dict]) -> dict:
    expected = len(CANDIDATE_SPECS) * len(ALL_SCENARIOS)
    if len(payloads) != expected:
        raise ValueError(f"frontier pilot screen needs exactly {expected} cells")
    by_name: dict[str, dict[str, dict]] = {}
    for payload in payloads:
        validate_cell(payload)
        name = str(payload["checkpoint_name"])
        scenario = str(payload["scenario"])
        if scenario in by_name.setdefault(name, {}):
            raise ValueError(f"duplicate frontier-pilot cell {name}/{scenario}")
        by_name[name][scenario] = payload
    expected_names = [row["name"] for row in CANDIDATE_SPECS]
    if list(by_name) != expected_names:
        raise ValueError("frontier-pilot candidate order mismatch")
    if any(set(cells) != set(ALL_SCENARIOS) for cells in by_name.values()):
        raise ValueError("frontier-pilot scenario matrix mismatch")

    parent_cells = by_name["anchor_c50"]
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
        target_absolute_pass = bool(
            cells[TARGET_SCENARIO]["threshold_pass"]
        )
        candidate_results.append(
            {
                "checkpoint_name": name,
                "conceptual_iteration": spec["conceptual_iteration"],
                "target_metrics": cells[TARGET_SCENARIO]["metrics"],
                "target_absolute_pass_descriptive": target_absolute_pass,
                "target_improvement": target,
                "retention": retention,
                "retention_pass": retention_pass,
                "pilot_acceptance_pass": target["pass"] and retention_pass,
            }
        )

    eligible = [
        row for row in candidate_results if row["pilot_acceptance_pass"]
    ]
    eligible.sort(
        key=lambda row: (
            float(row["target_metrics"]["cr"]),
            sum(
                float(item["metrics"]["cr"])
                for item in row["retention"].values()
            )
            / len(row["retention"]),
            -int(row["conceptual_iteration"]),
        )
    )
    selected = eligible[0]["checkpoint_name"] if eligible else None
    selected_row = eligible[0] if eligible else None
    target_absolute_pass = bool(
        selected_row and selected_row["target_absolute_pass_descriptive"]
    )
    if selected is None:
        next_action = "HOLD_LINEAGE_DIAGNOSTIC_ONLY"
    elif target_absolute_pass:
        next_action = "AWAIT_HUMAN_DECISION_ABSOLUTE_GATE_OBSERVED"
    else:
        next_action = "AWAIT_HUMAN_DECISION_IMPROVED_NOT_GRADUATED"
    return {
        "schema": "sa5_v3_c50_frontier_pilot_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_DEVELOPMENT_SCREEN",
        "protocol_sha256": screen_protocol()["sha256"],
        "anchor": {
            "checkpoint_name": "anchor_c50",
            "status": "UNGRADUATED_DIAGNOSTIC_ANCHOR_NOT_FORMAL_PARENT",
            "target_metrics": parent_cells[TARGET_SCENARIO]["metrics"],
            "retention_metrics": {
                scenario: parent_cells[scenario]["metrics"]
                for scenario in RETENTION_SCENARIOS
            },
        },
        "candidate_results": candidate_results,
        "selected_development_checkpoint": selected,
        # Compatibility with the shared queue's final status line. Extension
        # remains forbidden even when a development checkpoint is selected.
        "selected_extension_checkpoint": None,
        "pilot_acceptance_observed": selected is not None,
        "selected_target_absolute_pass": target_absolute_pass,
        "graduation_authorized": False,
        "extension_authorized": False,
        "next_action": next_action,
        "sa5_graduated": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["evidence_boundary"],
    }
