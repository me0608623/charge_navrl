"""Frozen no-training pedestrian-speed screen for the SA5-R2 c250 policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import sa5_r2_checkpoint_screen as parent_screen


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

CHECKPOINT_NAME = "c250"
CHECKPOINT = parent_screen.candidate_by_name(CHECKPOINT_NAME)
SCENARIO = "corridor_lateral"
SPEED_RATE = 1.0
SPEED_RATE_OBS = "ego"
DEPLOYMENT_SPEED_SCALE = 1.0
STEPS = 3000
MIN_EPISODES = 1000
DYNAMIC_FEASIBILITY_LEAD_S = 1.0
CR_REFERENCE_EPISODES = 1000
CR_INCREASE_SE_THRESHOLD = 2.0

PEDESTRIAN_SPEED_ARMS = (
    {"label": "P035", "range_m_s": (0.25, 0.45), "midpoint_m_s": 0.35},
    {"label": "P060", "range_m_s": (0.50, 0.70), "midpoint_m_s": 0.60},
    {"label": "P080", "range_m_s": (0.70, 0.90), "midpoint_m_s": 0.80},
    {"label": "P100", "range_m_s": (0.90, 1.10), "midpoint_m_s": 1.00},
)


def arm_by_label(label: str) -> dict:
    for arm in PEDESTRIAN_SPEED_ARMS:
        if arm["label"] == label:
            return dict(arm)
    raise ValueError(f"unknown pedestrian-speed arm {label!r}")


def checkpoint_path() -> Path:
    return parent_screen.checkpoint_path(CHECKPOINT)


def screen_protocol() -> dict:
    parent = parent_screen.screen_protocol()
    payload = {
        "schema": "sa5_c250_pedestrian_speed_screen_protocol/v1",
        "purpose": (
            "measure how the frozen c250 policy responds when pedestrian speed "
            "approaches or exceeds robot speed, before rebuilding SA3-to-SA4"
        ),
        "checkpoint": {
            **CHECKPOINT,
            "path": str(checkpoint_path().resolve()),
        },
        "arms": [dict(arm) for arm in PEDESTRIAN_SPEED_ARMS],
        "fixed_evaluation": {
            "stage": parent_screen.STAGE,
            "scenario": SCENARIO,
            "corridor_density": "4S2D",
            "corridor_motion_mode": "lateral",
            "corridor_geometry": "sealed 10m interaction zone in 15m arena",
            "seed": parent_screen.SEED,
            "num_envs": parent_screen.NUM_ENVS,
            "steps": STEPS,
            "minimum_completed_episodes": MIN_EPISODES,
            "speed_rate": SPEED_RATE,
            "speed_rate_obs": SPEED_RATE_OBS,
            "deployment_speed_scale": DEPLOYMENT_SPEED_SCALE,
            "actuator_delay_steps": parent_screen.DELAY_STEPS,
            "actuator_delay_ms": parent_screen.DELAY_MS,
            "actuator_profile": parent_screen.ACTUATOR_PROFILE,
            "lidar_noise_mode": parent_screen.LIDAR_NOISE_MODE,
            "lidar_distractor_eligibility": (
                parent_screen.LIDAR_DISTRACTOR_ELIGIBILITY
            ),
            "scene": parent["fixed_evaluation"]["scene"],
        },
        "only_independent_variable": "long_corridor_dynamic_speed_range",
        "outcomes": {
            "primary": ["SR", "CR", "TO"],
            "collision_breakdown": ["obstacle_CR", "wall_CR"],
            "impact_severity": (
                "radial closing speed at dynamic collision, p50/p90; this is "
                "the collision-normal component, not full 2D relative speed"
            ),
            "behavior": [
                "commanded_stop_fraction",
                "actual_body_stop_fraction",
                "actual_stop_lead_p50_s",
            ],
            "dynamic_feasibility": {
                "source": "D5 baseline-only shadow; policy action unchanged",
                "lead_window_s": DYNAMIC_FEASIBILITY_LEAD_S,
                "use": "dynamic-only feasible action count",
                "forbidden": (
                    "joint dynamic-plus-static feasibility is not used for the "
                    "mechanism classification"
                ),
            },
        },
        "interpretation_rule": {
            "policy_limited": (
                "CR rises by at least 2 conservative binomial SE and more than "
                "half of collision events still have a dynamic-feasible action "
                "within the final 1.0 s"
            ),
            "feasibility_limited": (
                "CR rises by at least 2 conservative binomial SE and fewer than "
                "half retain a dynamic-feasible action within the final 1.0 s"
            ),
            "mixed": "exactly half, or outcome increase is not established",
        },
        "forbidden": [
            "training",
            "reward changes",
            "geometry changes",
            "speed_rate changes",
            "checkpoint changes",
            "SA6 launch",
        ],
        "evidence_boundary": (
            "single checkpoint and evaluator seed diagnostic; determines the "
            "candidate pedestrian-speed curriculum contract but does not prove "
            "learnability, accept a parent, start training, or authorize SA6"
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


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p90": None}
    ordered = sorted(float(value) for value in values)

    def nearest(fraction: float) -> float:
        index = int(round(fraction * (len(ordered) - 1)))
        return ordered[index]

    return {"p50": nearest(0.50), "p90": nearest(0.90)}


def collision_radial_closing_summary(d3_payload: dict) -> dict:
    if d3_payload.get("schema") != "sa4_d3_yield_timing/v2":
        raise ValueError("unexpected D3 timing schema")
    if d3_payload.get("mode") != "baseline":
        raise ValueError("pedestrian-speed screen requires baseline D3 mode")
    if (d3_payload.get("self_check") or {}).get("reconciliation_ok") is not True:
        raise ValueError("D3 timing recorder failed reconciliation")
    events = [
        event
        for event in d3_payload.get("events", [])
        if event.get("event_type") == "dynamic_collision"
    ]
    values: list[float] = []
    for event in events:
        records = event.get("records") or []
        slot = int(event.get("tracked_obstacle_slot", -1))
        if not records or slot < 0:
            continue
        closings = records[-1].get(
            "dynamic_obstacle_relative_closing_speeds_mps"
        ) or []
        if slot >= len(closings):
            continue
        value = float(closings[slot])
        if math.isfinite(value):
            values.append(max(value, 0.0))
    return {
        "definition": (
            "positive radial closing component at the collision transition; "
            "not full 2D relative velocity magnitude"
        ),
        "events": len(events),
        "observed": len(values),
        "coverage": len(values) / max(len(events), 1),
        **_percentiles(values),
    }


def dynamic_feasibility_summary(d5_payload: dict) -> dict:
    if d5_payload.get("schema") != "sa4_d5_feasibility_frontier/v1":
        raise ValueError("unexpected D5 frontier schema")
    if (d5_payload.get("self_check") or {}).get("reconciliation_ok") is not True:
        raise ValueError("D5 feasibility recorder failed reconciliation")
    events = [
        event
        for event in d5_payload.get("events", [])
        if event.get("event_type") == "dynamic_collision"
    ]
    retained = 0
    observed = 0
    for event in events:
        frontier = event.get("frontier") or {}
        lead = frontier.get("closest_dynamic_feasible_s_before_event")
        if lead is None:
            continue
        observed += 1
        retained += float(lead) <= DYNAMIC_FEASIBILITY_LEAD_S
    frontier_summary = (d5_payload.get("frontier_summary") or {}).get(
        "dynamic_collision"
    ) or {}
    return {
        "events": len(events),
        "closest_dynamic_feasible_observed": observed,
        "dynamic_feasible_within_1s_events": retained,
        "dynamic_feasible_within_1s_fraction": retained / max(len(events), 1),
        "event_no_dynamic_feasible_fraction": _finite(
            frontier_summary.get("event_no_dynamic_feasible_fraction"),
            "event no-dynamic-feasible fraction",
        ),
        "persistent_dynamic_collapse_observed_fraction": _finite(
            frontier_summary.get(
                "persistent_dynamic_collapse_observed_fraction"
            ),
            "persistent dynamic collapse fraction",
        ),
        "persistent_dynamic_collapse_lead_s": frontier_summary.get(
            "persistent_dynamic_collapse_lead_s"
        ),
    }


def validate_cell(cell: dict) -> None:
    frozen = screen_protocol()
    if cell.get("schema") != "sa5_c250_pedestrian_speed_screen_cell/v1":
        raise ValueError("unexpected pedestrian-speed cell schema")
    if cell.get("protocol_sha256") != frozen["sha256"]:
        raise ValueError("cell protocol hash mismatch")
    if cell.get("checkpoint_sha256") != CHECKPOINT["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    arm = arm_by_label(str(cell.get("arm")))
    if list(cell.get("pedestrian_speed_range_m_s") or []) != list(
        arm["range_m_s"]
    ):
        raise ValueError("cell pedestrian-speed range mismatch")
    if _finite(cell.get("speed_rate"), "speed rate") != SPEED_RATE:
        raise ValueError("cell changed the fixed robot speed rate")
    metrics = cell.get("metrics") or {}
    if int(metrics.get("n", 0)) < MIN_EPISODES:
        raise ValueError("cell has fewer than the frozen minimum episodes")
    for key in ("sr", "cr", "to"):
        _finite(metrics.get(key), key)
    report = cell.get("corridor_report") or {}
    if list(report.get("requested_dynamic_speed_range_m_s") or []) != list(
        arm["range_m_s"]
    ):
        raise ValueError("runtime pedestrian-speed range mismatch")
    severity = cell.get("impact_severity") or {}
    if int(severity.get("events", -1)) != int(severity.get("observed", -2)):
        raise ValueError("impact severity lacks full collision coverage")
    feasibility = cell.get("dynamic_feasibility") or {}
    if int(feasibility.get("events", -1)) != int(severity.get("events", -2)):
        raise ValueError("D3/D5 collision event counts do not reconcile")


def summarize_cells(cells: list[dict]) -> dict:
    if len(cells) != len(PEDESTRIAN_SPEED_ARMS):
        raise ValueError("pedestrian-speed screen requires exactly four cells")
    by_arm = {}
    for cell in cells:
        validate_cell(cell)
        label = cell["arm"]
        if label in by_arm:
            raise ValueError(f"duplicate pedestrian-speed arm {label}")
        by_arm[label] = cell
    expected = {arm["label"] for arm in PEDESTRIAN_SPEED_ARMS}
    if set(by_arm) != expected:
        raise ValueError("pedestrian-speed screen is missing a frozen arm")

    baseline = by_arm["P035"]["metrics"]
    rows = []
    for arm in PEDESTRIAN_SPEED_ARMS:
        cell = by_arm[arm["label"]]
        metrics = cell["metrics"]
        report = cell["corridor_report"]
        severity = cell["impact_severity"]
        feasibility = cell["dynamic_feasibility"]
        reaction = cell["reaction_timing"]["dynamic_collision"]
        delta_cr = float(metrics["cr"]) - float(baseline["cr"])
        cr_se = math.sqrt(
            float(metrics["cr"])
            * (1.0 - float(metrics["cr"]))
            / CR_REFERENCE_EPISODES
            + float(baseline["cr"])
            * (1.0 - float(baseline["cr"]))
            / CR_REFERENCE_EPISODES
        )
        rows.append(
            {
                "arm": arm["label"],
                "pedestrian_speed_range_m_s": list(arm["range_m_s"]),
                "pedestrian_speed_midpoint_m_s": arm["midpoint_m_s"],
                "n": int(metrics["n"]),
                "sr": float(metrics["sr"]),
                "cr": float(metrics["cr"]),
                "to": float(metrics["to"]),
                "delta_cr_vs_p035": delta_cr,
                "delta_cr_se_vs_p035": (
                    0.0 if arm["label"] == "P035" else delta_cr / cr_se
                ),
                "obstacle_cr": float(report["obstacle_collision_rate"]),
                "wall_cr": float(report["wall_collision_rate"]),
                "commanded_stop_fraction": float(
                    report["stop_command_fraction"]
                ),
                "actual_body_stop_fraction": float(
                    report["actual_body_stop_fraction"]
                ),
                "actual_decel_observed_fraction": float(
                    reaction["first_actual_decel"]["observed_fraction"]
                ),
                "actual_decel_lead_p50_s": reaction["first_actual_decel"][
                    "lead_s"
                ]["p50"],
                "actual_stop_observed_fraction": float(
                    reaction["first_actual_stop"]["observed_fraction"]
                ),
                "actual_stop_lead_p50_s": reaction["first_actual_stop"][
                    "lead_s"
                ]["p50"],
                "impact_radial_closing_p50_mps": severity["p50"],
                "impact_radial_closing_p90_mps": severity["p90"],
                "dynamic_feasible_within_1s_fraction": feasibility[
                    "dynamic_feasible_within_1s_fraction"
                ],
                "event_no_dynamic_feasible_fraction": feasibility[
                    "event_no_dynamic_feasible_fraction"
                ],
                "persistent_dynamic_collapse_observed_fraction": feasibility[
                    "persistent_dynamic_collapse_observed_fraction"
                ],
                "cell": cell["cell"],
            }
        )

    rising_rows = [
        row
        for row in rows[1:]
        if row["delta_cr_se_vs_p035"] >= CR_INCREASE_SE_THRESHOLD
    ]
    if not rising_rows:
        mechanism = "NO_OBSERVED_CR_INCREASE"
    else:
        fractions = [
            row["dynamic_feasible_within_1s_fraction"]
            for row in rising_rows
        ]
        if all(value > 0.5 for value in fractions):
            mechanism = "POLICY_LIMITED_MAJORITY_DYNAMICALLY_FEASIBLE"
        elif all(value < 0.5 for value in fractions):
            mechanism = (
                "D5_MODEL_FEASIBILITY_LIMITED_MAJORITY_WITHOUT_"
                "FINAL_1S_DYNAMIC_SOLUTION"
            )
        else:
            mechanism = "MIXED_DYNAMIC_FEASIBILITY"

    return {
        "schema": "sa5_c250_pedestrian_speed_screen_summary/v1",
        "status": "COMPLETE_VALID_SINGLE_SEED_DIAGNOSTIC",
        "protocol_sha256": screen_protocol()["sha256"],
        "checkpoint": CHECKPOINT,
        "rows": rows,
        "mechanism_classification": mechanism,
        "accepted_parent": False,
        "training_started": False,
        "sa6_started": False,
        "interpretation_limit": screen_protocol()["evidence_boundary"],
    }
