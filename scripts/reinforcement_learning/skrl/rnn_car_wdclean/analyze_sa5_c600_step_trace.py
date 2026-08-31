"""Event-level analysis for accepted SA5 c600 policy traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def _distribution(values: Iterable[float]) -> dict[str, int | float | None]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        return {"samples": 0, "p50": None, "p95": None, "maximum": None}
    return {
        "samples": int(data.size),
        "p50": float(np.quantile(data, 0.50)),
        "p95": float(np.quantile(data, 0.95)),
        "maximum": float(data.max()),
    }


def _continuous(data: dict[str, np.ndarray], t: int, env: int) -> bool:
    return (
        t > 0
        and int(data["episode_step"][t, env])
        == int(data["episode_step"][t - 1, env]) + 1
    )


def _nearest_moving_velocity(
    data: dict[str, np.ndarray],
    *,
    start: int,
    stop: int,
    env: int,
    slot: int,
    reverse: bool,
) -> np.ndarray | None:
    indices = range(start, stop, -1) if reverse else range(start, stop)
    for index in indices:
        if index < 0 or index >= data["dynamic_velocities_mps"].shape[0]:
            continue
        if index > 0 and not _continuous(data, index, env):
            break
        velocity = data["dynamic_velocities_mps"][index, env, slot]
        if np.linalg.norm(velocity) >= 0.05:
            return velocity
    return None


def _first_onset(
    data: dict[str, np.ndarray],
    command: np.ndarray,
    *,
    event_step: int,
    env: int,
    start: int,
    end: int,
    kind: str,
) -> int | None:
    for index in range(max(1, start), end):
        if not _continuous(data, index, env):
            continue
        previous = command[index - 1, env]
        current = command[index, env]
        if kind == "stop":
            onset = abs(float(previous[0])) > 0.10 and abs(float(current[0])) <= 0.10
        elif kind == "turn":
            onset = abs(float(previous[1])) < 0.30 and abs(float(current[1])) >= 0.30
        else:
            raise ValueError(f"unknown onset kind {kind!r}")
        if onset:
            return index
    return None


def _onset_indices(
    data: dict[str, np.ndarray],
    command: np.ndarray,
    *,
    env: int,
    start: int,
    end: int,
    kind: str,
) -> list[int]:
    indices: list[int] = []
    for index in range(max(1, start), end):
        if not _continuous(data, index, env):
            continue
        previous = command[index - 1, env]
        current = command[index, env]
        if kind == "stop":
            onset = abs(float(previous[0])) > 0.10 and abs(float(current[0])) <= 0.10
        elif kind == "turn":
            onset = abs(float(previous[1])) < 0.30 and abs(float(current[1])) >= 0.30
        else:
            raise ValueError(f"unknown onset kind {kind!r}")
        if onset:
            indices.append(index)
    return indices


def _episode_outcome(
    data: dict[str, np.ndarray], t: int, env: int
) -> int | None:
    for future in range(t, data["done"].shape[0]):
        if future > t and not _continuous(data, future, env):
            return None
        if bool(data["done"][future, env]):
            return int(data["termination_cause"][future, env])
    return None


def _outcomes(rows: list[dict]) -> dict[str, int | float | None]:
    causes = [row["episode_outcome_cause"] for row in rows]
    known = [cause for cause in causes if cause is not None]
    return {
        "events": len(rows),
        "known_outcomes": len(known),
        "success_rate": None if not known else known.count(1) / len(known),
        "wall_collision_rate": None if not known else known.count(2) / len(known),
        "obstacle_collision_rate": None if not known else known.count(3) / len(known),
        "timeout_rate": None if not known else known.count(4) / len(known),
    }


def _event_locked_profile(
    data: dict[str, np.ndarray], events: list[dict], *, dt_s: float
) -> list[dict]:
    issued = data["pre_delay_command_mps_rad_s"]
    applied = data["post_delay_command_mps_rad_s"]
    profile: list[dict] = []
    for offset in range(-15, 16):
        issued_rows = []
        applied_rows = []
        for event in events:
            index = int(event["rollout_step"]) + offset
            env = int(event["env_id"])
            if index < 0 or index >= issued.shape[0]:
                continue
            expected_episode_step = int(
                data["episode_step"][int(event["rollout_step"]), env]
            ) + offset
            if expected_episode_step < 0:
                continue
            if int(data["episode_step"][index, env]) != expected_episode_step:
                continue
            issued_rows.append(issued[index, env])
            applied_rows.append(applied[index, env])
        if not issued_rows:
            continue
        issued_array = np.asarray(issued_rows)
        applied_array = np.asarray(applied_rows)
        profile.append(
            {
                "relative_s": round(offset * dt_s, 3),
                "samples": len(issued_rows),
                "issued_speed_abs_mean_mps": float(np.abs(issued_array[:, 0]).mean()),
                "applied_speed_abs_mean_mps": float(np.abs(applied_array[:, 0]).mean()),
                "applied_stop_fraction": float((np.abs(applied_array[:, 0]) <= 0.10).mean()),
                "issued_angular_abs_mean_rad_s": float(np.abs(issued_array[:, 1]).mean()),
                "applied_angular_abs_mean_rad_s": float(np.abs(applied_array[:, 1]).mean()),
                "applied_strong_turn_fraction": float((np.abs(applied_array[:, 1]) >= 0.30).mean()),
            }
        )
    return profile


def analyze_lateral_trace(
    data: dict[str, np.ndarray], *, dt_s: float = 0.2
) -> dict:
    """Measure which side the robot actually occupies after a crossing.

    The fixed corridor local x-axis is lateral. A pedestrian crossing the
    episode-start robot x coordinate vacates the side containing its previous
    x position. Robot departure side is based on actual displacement, not the
    sign of one angular command.
    """
    positions = data["dynamic_positions_m"]
    velocities = data["dynamic_velocities_mps"]
    valid = data["dynamic_valid"] & data["patrol_active"]
    robot = data["robot_xy_m"]
    applied = data["post_delay_command_mps_rad_s"]
    steps, envs, slots, _ = positions.shape
    episode_origin_x = np.zeros(envs, dtype=np.float32)
    events: list[dict] = []
    ambiguous_frames = 0

    for t in range(steps):
        reset = data["episode_step"][t] == 0
        episode_origin_x[reset] = robot[t, reset, 0]
        if t == 0:
            continue
        for env in range(envs):
            if not _continuous(data, t, env):
                continue
            previous = positions[t - 1, env, :, 0] - episode_origin_x[env]
            current = positions[t, env, :, 0] - episode_origin_x[env]
            crossing = (
                valid[t - 1, env]
                & valid[t, env]
                & (np.abs(previous) >= 0.02)
                & (np.abs(velocities[t, env, :, 0]) >= 0.05)
                & (previous * current <= 0.0)
            )
            slots_now = np.flatnonzero(crossing)
            if slots_now.size != 1:
                ambiguous_frames += int(slots_now.size > 1)
                continue
            slot = int(slots_now[0])
            vacated_side = int(np.sign(previous[slot]))
            if vacated_side == 0:
                continue
            end = min(steps, t + 16)
            departure_t = None
            selected_side = 0
            origin = float(robot[t, env, 0])
            side_switches = 0
            stop_go_transitions = 0
            previous_side = 0
            previous_moving = bool(float(applied[t, env, 0]) >= 0.15)
            for future in range(t, end):
                if future > t and not _continuous(data, future, env):
                    break
                displacement = float(robot[future, env, 0]) - origin
                moving = float(applied[future, env, 0]) >= 0.15
                side = 1 if displacement >= 0.15 else (-1 if displacement <= -0.15 else 0)
                if side and previous_side and side != previous_side:
                    side_switches += 1
                if side:
                    previous_side = side
                if moving and not previous_moving:
                    stop_go_transitions += 1
                previous_moving = moving
                if (
                    departure_t is None
                    and moving
                    and abs(displacement) >= 0.15
                ):
                    departure_t = future
                    selected_side = 1 if displacement > 0.0 else -1
            events.append(
                {
                    "rollout_step": t,
                    "env_id": env,
                    "obstacle_slot": slot,
                    "vacated_side": vacated_side,
                    "departure_step": departure_t,
                    "departure_latency_s": (
                        None if departure_t is None else (departure_t - t) * dt_s
                    ),
                    "selected_side": selected_side,
                    "robot_obstacle_distance_m": float(
                        np.linalg.norm(
                            positions[t, env, slot] - robot[t, env]
                        )
                    ),
                    "episode_key": (
                        f"{env}:{t - int(data['episode_step'][t, env])}"
                    ),
                    "episode_outcome_cause": _episode_outcome(data, t, env),
                    "match": bool(
                        departure_t is not None and selected_side == vacated_side
                    ),
                    "side_switches_within_3s": side_switches,
                    "stop_go_transitions_within_3s": stop_go_transitions,
                }
            )

    departures = [row for row in events if row["departure_step"] is not None]
    matches = sum(int(row["match"]) for row in departures)
    first_departure_by_episode: dict[str, dict] = {}
    for row in departures:
        first_departure_by_episode.setdefault(row["episode_key"], row)
    independent = list(first_departure_by_episode.values())
    independent_matches = [row for row in independent if row["match"]]
    independent_mismatches = [row for row in independent if not row["match"]]
    near_events = [
        row for row in events if row["robot_obstacle_distance_m"] <= 3.0
    ]
    near_departures = [
        row for row in near_events if row["departure_step"] is not None
    ]
    near_first_by_episode: dict[str, dict] = {}
    for row in near_departures:
        near_first_by_episode.setdefault(row["episode_key"], row)
    near_independent = list(near_first_by_episode.values())
    near_matches = [row for row in near_independent if row["match"]]
    near_mismatches = [row for row in near_independent if not row["match"]]
    return {
        "schema": "sa5_c600_lateral_vacated_side_trace/v3",
        "crossing_events": len(events),
        "ambiguous_multi_crossing_frames": ambiguous_frames,
        "departure_events": len(departures),
        "vacated_side_matches": matches,
        "vacated_side_match_fraction": (
            None if not departures else matches / len(departures)
        ),
        "departure_latency_s": _distribution(
            row["departure_latency_s"] for row in departures
        ),
        "side_switches_within_3s": _distribution(
            row["side_switches_within_3s"] for row in events
        ),
        "stop_go_transitions_within_3s": _distribution(
            row["stop_go_transitions_within_3s"] for row in events
        ),
        "first_departure_per_episode": {
            "episodes": len(independent),
            "matches": len(independent_matches),
            "match_fraction": (
                None if not independent else len(independent_matches) / len(independent)
            ),
            "matched_outcomes": _outcomes(independent_matches),
            "mismatched_outcomes": _outcomes(independent_mismatches),
        },
        "near_interaction_le_3m": {
            "crossing_events": len(near_events),
            "departure_events": len(near_departures),
            "event_match_fraction": (
                None
                if not near_departures
                else sum(row["match"] for row in near_departures)
                / len(near_departures)
            ),
            "departure_latency_s": _distribution(
                row["departure_latency_s"] for row in near_departures
            ),
            "first_departure_per_episode": {
                "episodes": len(near_independent),
                "matches": len(near_matches),
                "match_fraction": (
                    None
                    if not near_independent
                    else len(near_matches) / len(near_independent)
                ),
                "matched_outcomes": _outcomes(near_matches),
                "mismatched_outcomes": _outcomes(near_mismatches),
            },
        },
        "events": events,
    }


def analyze_random2d_trace(
    data: dict[str, np.ndarray], *, dt_s: float = 0.2
) -> dict:
    """Measure policy timing around patrol waypoint switches and reversals."""
    wp = data["patrol_waypoint_index"]
    active = data["patrol_active"]
    velocity = data["dynamic_velocities_mps"]
    issued = data["pre_delay_command_mps_rad_s"]
    applied = data["post_delay_command_mps_rad_s"]
    done = data["done"]
    cause = data["termination_cause"]
    steps, envs, slots = wp.shape
    events: list[dict] = []

    for t in range(1, steps):
        for env in range(envs):
            if not _continuous(data, t, env):
                continue
            for slot in range(slots):
                if not (active[t - 1, env, slot] and active[t, env, slot]):
                    continue
                if int(wp[t, env, slot]) == int(wp[t - 1, env, slot]):
                    continue
                # Patrol may pause for 0--5 steps at a waypoint. Look around
                # that pause instead of requiring both adjacent frames to move.
                old_v = _nearest_moving_velocity(
                    data,
                    start=t - 1,
                    stop=max(-1, t - 7),
                    env=env,
                    slot=slot,
                    reverse=True,
                )
                new_v = _nearest_moving_velocity(
                    data,
                    start=t,
                    stop=min(steps, t + 7),
                    env=env,
                    slot=slot,
                    reverse=False,
                )
                reversal = bool(
                    old_v is not None
                    and new_v is not None
                    and float(np.dot(old_v, new_v)) < 0.0
                )
                start = max(0, t - 15)
                end = min(steps, t + 16)
                onset_groups = {
                    "issued_stop": _onset_indices(
                        data, issued, env=env, start=start, end=end, kind="stop"
                    ),
                    "applied_stop": _onset_indices(
                        data, applied, env=env, start=start, end=end, kind="stop"
                    ),
                    "issued_turn": _onset_indices(
                        data, issued, env=env, start=start, end=end, kind="turn"
                    ),
                    "applied_turn": _onset_indices(
                        data, applied, env=env, start=start, end=end, kind="turn"
                    ),
                }
                timing = {}
                for label, indices in onset_groups.items():
                    before = [index for index in indices if index < t]
                    after = [index for index in indices if index >= t]
                    timing[f"last_{label}_before_s"] = (
                        None if not before else (max(before) - t) * dt_s
                    )
                    timing[f"first_{label}_after_s"] = (
                        None if not after else (min(after) - t) * dt_s
                    )
                collision_t = None
                for future in range(t, end):
                    if future > t and not _continuous(data, future, env):
                        break
                    if bool(done[future, env]) and int(cause[future, env]) == 3:
                        collision_t = future
                        break
                events.append(
                    {
                        "rollout_step": t,
                        "env_id": env,
                        "obstacle_slot": slot,
                        "direction_reversal": reversal,
                        "robot_obstacle_distance_m": float(
                            np.linalg.norm(
                                data["dynamic_positions_m"][t, env, slot]
                                - data["robot_xy_m"][t, env]
                            )
                        ),
                        **timing,
                        "applied_stopped_at_reversal": bool(
                            abs(float(applied[t, env, 0])) <= 0.10
                        ),
                        "applied_strong_turn_at_reversal": bool(
                            abs(float(applied[t, env, 1])) >= 0.30
                        ),
                        "episode_outcome_cause": _episode_outcome(data, t, env),
                        "obstacle_collision_within_3s": collision_t is not None,
                        "collision_latency_s": (
                            None if collision_t is None else (collision_t - t) * dt_s
                        ),
                    }
                )
    reversals = [row for row in events if row["direction_reversal"]]
    near_reversals = [
        row for row in reversals if row["robot_obstacle_distance_m"] <= 3.0
    ]

    def _relative_distribution(
        key: str, rows: list[dict] | None = None
    ) -> dict[str, int | float | None]:
        selected = reversals if rows is None else rows
        return _distribution(
            row[key] for row in selected if row[key] is not None
        )

    def _random_subset(rows: list[dict]) -> dict:
        return {
            "events": len(rows),
            "issued_reaction_within_1s_before_fraction": (
                None
                if not rows
                else sum(
                    any(
                        row[key] is not None and row[key] >= -1.0
                        for key in (
                            "last_issued_stop_before_s",
                            "last_issued_turn_before_s",
                        )
                    )
                    for row in rows
                )
                / len(rows)
            ),
            "stopped_at_reversal_fraction": (
                None
                if not rows
                else sum(row["applied_stopped_at_reversal"] for row in rows)
                / len(rows)
            ),
            "strong_turn_at_reversal_fraction": (
                None
                if not rows
                else sum(row["applied_strong_turn_at_reversal"] for row in rows)
                / len(rows)
            ),
            "first_applied_stop_after_s": _relative_distribution(
                "first_applied_stop_after_s", rows
            ),
            "first_applied_turn_after_s": _relative_distribution(
                "first_applied_turn_after_s", rows
            ),
            "collision_within_3s_fraction": (
                None
                if not rows
                else sum(row["obstacle_collision_within_3s"] for row in rows)
                / len(rows)
            ),
            "event_locked_profile": _event_locked_profile(
                data, rows, dt_s=dt_s
            ),
        }

    return {
        "schema": "sa5_c600_random2d_reversal_trace/v3",
        "waypoint_switch_events": len(events),
        "direction_reversal_events": len(reversals),
        "stop_latency_s": _relative_distribution(
            "first_applied_stop_after_s"
        ),
        "turn_latency_s": _relative_distribution(
            "first_applied_turn_after_s"
        ),
        "last_issued_stop_before_s": _relative_distribution(
            "last_issued_stop_before_s"
        ),
        "last_applied_stop_before_s": _relative_distribution(
            "last_applied_stop_before_s"
        ),
        "last_issued_turn_before_s": _relative_distribution(
            "last_issued_turn_before_s"
        ),
        "last_applied_turn_before_s": _relative_distribution(
            "last_applied_turn_before_s"
        ),
        "issued_reaction_within_1s_before_fraction": (
            None
            if not reversals
            else sum(
                any(
                    row[key] is not None and row[key] >= -1.0
                    for key in (
                        "last_issued_stop_before_s",
                        "last_issued_turn_before_s",
                    )
                )
                for row in reversals
            )
            / len(reversals)
        ),
        "stopped_at_reversal_fraction": (
            None if not reversals else sum(row["applied_stopped_at_reversal"] for row in reversals) / len(reversals)
        ),
        "strong_turn_at_reversal_fraction": (
            None if not reversals else sum(row["applied_strong_turn_at_reversal"] for row in reversals) / len(reversals)
        ),
        "event_locked_profile": _event_locked_profile(
            data, reversals, dt_s=dt_s
        ),
        "near_interaction_le_3m": _random_subset(near_reversals),
        "collision_within_3s_fraction": (
            None
            if not reversals
            else sum(row["obstacle_collision_within_3s"] for row in reversals)
            / len(reversals)
        ),
        "events": events,
    }


def load_trace(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "metadata_json"}
        metadata = json.loads(str(archive["metadata_json"].item()))
    return arrays, metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--mode", choices=("lateral", "random_2d"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    arrays, metadata = load_trace(args.trace)
    report = (
        analyze_lateral_trace(arrays)
        if args.mode == "lateral"
        else analyze_random2d_trace(arrays)
    )
    report["trace_metadata"] = metadata
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
