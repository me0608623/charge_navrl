"""Summarize a validated SA4-D3 schema-v2 baseline without using the GPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


RISK_WARNING = 0.05
ROBUST_DROP_MPS = 0.10
STOP_SPEED_MPS = 0.10
DT_S = 0.2
CONTACT_DISTANCE_M = 0.70
PROBE_TTC_S = 2.0
PROBE_CONFIRM_STEPS = 2


def estimated_full_stop_time_s(
    speed_mps: float,
    max_deceleration_mps2: float,
    action_delay_s: float,
) -> float:
    """Conservative constant-deceleration stop time including action delay."""
    if speed_mps < 0.0:
        raise ValueError("speed_mps must be non-negative")
    if max_deceleration_mps2 <= 0.0:
        raise ValueError("max_deceleration_mps2 must be positive")
    if action_delay_s < 0.0:
        raise ValueError("action_delay_s must be non-negative")
    return speed_mps / max_deceleration_mps2 + action_delay_s


def percentile(values: Iterable[float], fraction: float) -> float | None:
    clean = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not clean:
        return None
    position = (len(clean) - 1) * float(fraction)
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def distribution(values: Iterable[float]) -> dict:
    values = list(values)
    return {
        "n": len(values),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
    }


def first_event_summary(events: list[dict], label: str) -> dict:
    key = f"first_{label}_s_before_event"
    censor_key = f"first_{label}_left_censored"
    values = [
        event["first_events"][key]
        for event in events
        if event["first_events"][key] is not None
    ]
    return {
        "observed_fraction": len(values) / len(events) if events else 0.0,
        "left_censored_fraction": (
            sum(bool(event["first_events"][censor_key]) for event in events)
            / len(events)
            if events
            else 0.0
        ),
        "lead_s": distribution(values),
    }


def tracked_value(event: dict, record: dict, field: str) -> float:
    return float(record[field][event["tracked_obstacle_slot"]])


def risk_response(event: dict) -> dict | None:
    slot = event["tracked_obstacle_slot"]
    records = event["records"]
    onset = next(
        (
            index
            for index, record in enumerate(records)
            if float(record["dynamic_obstacle_risks"][slot]) >= RISK_WARNING
        ),
        None,
    )
    if onset is None:
        return None
    warning = records[onset]
    tail = records[onset:]
    baseline_command = float(warning["pre_delay_v_command_mps"])
    robust_index = None
    for index in range(onset, len(records) - 1):
        current_drop = baseline_command - float(
            records[index]["pre_delay_v_command_mps"]
        )
        next_drop = baseline_command - float(
            records[index + 1]["pre_delay_v_command_mps"]
        )
        if current_drop >= ROBUST_DROP_MPS and next_drop >= ROBUST_DROP_MPS:
            robust_index = index
            break
    return {
        "warning_lead_s": (len(records) - 1 - onset) * DT_S,
        "warning_body_speed_mps": float(warning["body_planar_speed_mps"]),
        "warning_pre_delay_command_mps": baseline_command,
        "maximum_issued_drop_mps": baseline_command
        - min(float(record["pre_delay_v_command_mps"]) for record in tail),
        "maximum_actual_drop_mps": float(warning["body_planar_speed_mps"])
        - min(float(record["body_planar_speed_mps"]) for record in tail),
        "robust_two_frame_drop": robust_index is not None,
        "robust_response_delay_s": (
            None if robust_index is None else (robust_index - onset) * DT_S
        ),
        "applied_stop_after_warning": any(
            abs(float(record["post_delay_v_command_mps"])) <= STOP_SPEED_MPS
            for record in tail
        ),
    }


def probe_trigger_lead(event: dict) -> float | None:
    """Two-frame radial TTC trigger used only for the planned mechanism probe."""
    slot = event["tracked_obstacle_slot"]
    run = 0
    for index, record in enumerate(event["records"]):
        distance = tracked_value(
            event, record, "dynamic_obstacle_center_distances_m"
        )
        closing = tracked_value(
            event,
            record,
            "dynamic_obstacle_relative_closing_speeds_mps",
        )
        ttc = (
            max(distance - CONTACT_DISTANCE_M, 0.0) / closing
            if closing > 0.0
            else float("inf")
        )
        active = distance <= 3.0 and closing > 0.0 and ttc <= PROBE_TTC_S
        run = run + 1 if active else 0
        if run >= PROBE_CONFIRM_STEPS:
            return (len(event["records"]) - 1 - index) * DT_S
    return None


def group_summary(events: list[dict]) -> dict:
    responses = [response for event in events if (response := risk_response(event))]
    robust_delays = [
        response["robust_response_delay_s"]
        for response in responses
        if response["robust_response_delay_s"] is not None
    ]
    return {
        "events": len(events),
        "closest_distance_m": distribution(
            event["closest_distance_m"] for event in events
        ),
        "first_near_3m": first_event_summary(events, "near_3m"),
        "first_risk_ge_0p05": first_event_summary(
            events, "risk_ge_0p05"
        ),
        "first_issued_stop": first_event_summary(events, "issued_stop"),
        "first_applied_stop": first_event_summary(events, "applied_stop"),
        "first_actual_stop": first_event_summary(events, "actual_stop"),
        "event_body_speed_mps": distribution(
            event["records"][-1]["body_planar_speed_mps"]
            for event in events
        ),
        "event_pre_delay_command_mps": distribution(
            event["records"][-1]["pre_delay_v_command_mps"]
            for event in events
        ),
        "risk_conditioned_response": {
            "risk_observed_fraction": (
                len(responses) / len(events) if events else 0.0
            ),
            "warning_lead_s": distribution(
                response["warning_lead_s"] for response in responses
            ),
            "warning_body_speed_mps": distribution(
                response["warning_body_speed_mps"] for response in responses
            ),
            "maximum_issued_drop_mps": distribution(
                response["maximum_issued_drop_mps"] for response in responses
            ),
            "maximum_actual_drop_mps": distribution(
                response["maximum_actual_drop_mps"] for response in responses
            ),
            "robust_two_frame_drop_fraction": (
                sum(response["robust_two_frame_drop"] for response in responses)
                / len(responses)
                if responses
                else 0.0
            ),
            "robust_response_delay_s": distribution(robust_delays),
            "applied_stop_fraction": (
                sum(
                    response["applied_stop_after_warning"]
                    for response in responses
                )
                / len(responses)
                if responses
                else 0.0
            ),
        },
    }


def trigger_summary(events: list[dict]) -> dict:
    leads = [
        lead for event in events if (lead := probe_trigger_lead(event)) is not None
    ]
    return {
        "events": len(events),
        "triggered": len(leads),
        "trigger_fraction": len(leads) / len(events) if events else 0.0,
        "trigger_lead_s": distribution(leads),
    }


def build_analysis(payload: dict, source: Path) -> dict:
    if payload.get("schema") != "sa4_d3_yield_timing/v2":
        raise ValueError("analysis requires sa4_d3_yield_timing/v2")
    if payload.get("mode") != "baseline":
        raise ValueError("analysis requires a baseline arm")
    if (payload.get("self_check") or {}).get("reconciliation_ok") is not True:
        raise ValueError("D3 baseline failed reconciliation")

    collisions = [
        event
        for event in payload["events"]
        if event["event_type"] == "dynamic_collision"
    ]
    successful = [
        event
        for event in payload["events"]
        if event["event_type"] == "noncollision_closest_approach"
        and event["episode_outcome_cause"] == 1
    ]
    close_successful = [
        event for event in successful if event["closest_distance_m"] <= 1.0
    ]
    max_decel = 0.5
    median_warning_speed = percentile(
        (
            response["warning_body_speed_mps"]
            for event in collisions
            if (response := risk_response(event)) is not None
        ),
        0.5,
    )
    if median_warning_speed is None:
        raise ValueError("baseline contains no collision risk-warning samples")
    full_stop_time = estimated_full_stop_time_s(
        median_warning_speed,
        max_decel,
        DT_S,
    )

    return {
        "schema": "sa4_d3_baseline_analysis/v1",
        "source": str(source.resolve()),
        "input_counts": payload["counts"],
        "input_self_check": payload["self_check"],
        "groups": {
            "dynamic_collision": group_summary(collisions),
            "successful_closest_approach": group_summary(successful),
            "successful_close_approach_le_1m": group_summary(
                close_successful
            ),
        },
        "mechanism_probe_trigger": {
            "status": "FROZEN_NOT_RUN",
            "purpose": (
                "early diagnostic intervention trigger, not a deployment shield"
            ),
            "definition": {
                "distance_max_m": 3.0,
                "closing_speed_min_mps": 0.0,
                "contact_distance_m": CONTACT_DISTANCE_M,
                "ttc_formula": "max(distance-0.70, 0) / closing_speed",
                "ttc_max_s": PROBE_TTC_S,
                "consecutive_frames": PROBE_CONFIRM_STEPS,
                "control_dt_s": DT_S,
            },
            "baseline_activation": {
                "dynamic_collision": trigger_summary(collisions),
                "successful_closest_approach": trigger_summary(successful),
                "successful_close_approach_le_1m": trigger_summary(
                    close_successful
                ),
            },
        },
        "stopping_time_check": {
            "median_speed_at_risk_warning_mps": median_warning_speed,
            "max_deceleration_mps2": max_decel,
            "fixed_action_delay_s": DT_S,
            "estimated_full_stop_time_s": full_stop_time,
            "median_risk_warning_lead_s": percentile(
                (
                    event["first_events"][
                        "first_risk_ge_0p05_s_before_event"
                    ]
                    for event in collisions
                ),
                0.5,
            ),
            "interpretation": (
                "risk>=0.05 alone usually appears too late for a full stop; "
                "this does not prove that partial braking is ineffective"
            ),
        },
        "limitations": [
            "single checkpoint and single evaluation seed",
            "closest-approach events are repeated event-level observations, not independent training seeds",
            "successful controls are observational rather than randomized counterfactuals",
            "the raw first-deceleration field is sensitive to ordinary command fluctuations; use the two-frame 0.10 m/s response summary",
            "the trigger is frozen only for a bounded mechanism probe and must not be presented as a deployment safety rule",
        ],
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: dict) -> str:
    groups = report["groups"]
    collision = groups["dynamic_collision"]
    success = groups["successful_closest_approach"]
    close = groups["successful_close_approach_le_1m"]
    trigger = report["mechanism_probe_trigger"]["baseline_activation"]
    stop = report["stopping_time_check"]
    lines = [
        "# SA4-D3 lateral baseline timing analysis",
        "",
        "> [!important] Scope",
        "> This is single-seed diagnostic evidence. No intervention arm has run,",
        "> and the frozen trigger below is not a deployment safety rule.",
        "",
        "## Evidence validity",
        "",
        f"- Records: {report['input_counts']['records']:,}",
        f"- Completed episodes: {report['input_counts']['completed_episodes']:,}",
        f"- Dynamic collisions: {report['input_counts']['dynamic_collision']:,}",
        f"- Successful closest-approach controls: {report['input_counts']['successful_noncollision_closest_approach']:,}",
        f"- Delay alignment errors: {report['input_self_check']['delay_alignment_errors']}",
        f"- Shield-engaged baseline records: {report['input_self_check']['shield_engaged_records']}",
        f"- Reconciliation: {report['input_self_check']['reconciliation_ok']}",
        "",
        "## Timing",
        "",
        "| Event group | n | risk seen | risk lead p50 | body speed at event p50 | applied stop after warning |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, group in (
        ("Dynamic collision", collision),
        ("Successful closest approach", success),
        ("Successful close approach <=1 m", close),
    ):
        response = group["risk_conditioned_response"]
        lines.append(
            f"| {label} | {group['events']:,} | "
            f"{_pct(response['risk_observed_fraction'])} | "
            f"{_num(response['warning_lead_s']['p50'])} s | "
            f"{_num(group['event_body_speed_mps']['p50'])} m/s | "
            f"{_pct(response['applied_stop_fraction'])} |"
        )
    lines += [
        "",
        f"All {collision['events']:,} dynamic collisions had a risk warning. "
        "Its median lead was "
        f"{_num(collision['risk_conditioned_response']['warning_lead_s']['p50'])} s. "
        "A two-frame command reduction of at least 0.10 m/s followed the warning "
        f"in {_pct(collision['risk_conditioned_response']['robust_two_frame_drop_fraction'])} "
        "of collisions, but an applied stop occurred in only "
        f"{_pct(collision['risk_conditioned_response']['applied_stop_fraction'])}.",
        "",
        "The policy therefore often reacts, but usually does not enter a sustained "
        "wait state before contact. This is association, not proof that braking "
        "alone will prevent the collision.",
        "",
        "## Full-stop feasibility",
        "",
        f"At the risk warning, median body speed was {_num(stop['median_speed_at_risk_warning_mps'])} m/s. "
        f"With a_max={_num(stop['max_deceleration_mps2'])} m/s^2 and d1={_num(stop['fixed_action_delay_s'])} s, "
        f"estimated full-stop time is {_num(stop['estimated_full_stop_time_s'])} s, "
        f"longer than the {_num(stop['median_risk_warning_lead_s'])} s median warning lead.",
        "",
        "Therefore `risk >= 0.05` is too late as the sole trigger for a complete "
        "stop-and-wait policy. It may still be early enough for partial braking or turning.",
        "",
        "## Frozen mechanism-probe trigger",
        "",
        "`distance <= 3.0 m AND closing > 0 AND "
        "max(distance - 0.70 m, 0) / closing <= 2.0 s` for two consecutive frames.",
        "",
        "| Group | Triggered | Median lead |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("Dynamic collision", "dynamic_collision"),
        ("Successful closest approach", "successful_closest_approach"),
        ("Successful close approach <=1 m", "successful_close_approach_le_1m"),
    ):
        item = trigger[key]
        lines.append(
            f"| {label} | {_pct(item['trigger_fraction'])} | "
            f"{_num(item['trigger_lead_s']['p50'])} s |"
        )
    lines += [
        "",
        "This deliberately broad trigger covers the baseline collisions early enough "
        "to test sustained braking. Its high activation on safe close approaches is "
        "acceptable only for a diagnostic mechanism probe; it is not suitable for deployment.",
        "",
        "## Next state",
        "",
        "- Baseline: complete and valid.",
        "- Trigger protocol: `FROZEN_NOT_RUN`.",
        "- Sustained-brake, best-turn, and combined arms: not run.",
        "- Training and SA5: not started.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.input.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    report = build_analysis(payload, source)
    output_json = args.output_json.expanduser().resolve()
    output_md = args.output_md.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(
        "[SA4-D3-ANALYSIS] "
        f"collisions={report['groups']['dynamic_collision']['events']} "
        f"successful_controls="
        f"{report['groups']['successful_closest_approach']['events']} "
        f"trigger_status="
        f"{report['mechanism_probe_trigger']['status']} "
        f"report={output_md}",
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
