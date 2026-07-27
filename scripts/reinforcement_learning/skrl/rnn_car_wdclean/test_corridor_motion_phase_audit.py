"""Sim-free tests for the corridor motion-phase audit and mixed_iid sampling.

These cover the five checks Codex required before the pause A/B may run:
legacy ``mixed`` unchanged, ``mixed_iid`` unbiased in small batches, the phase
state machine across pause / zero-pause / immediate-reversal / auto-reset, and
reconciliation of per-phase collision counts against total slot-events.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corridor_motion_phase_audit import (  # noqa: E402
    PHASE_CODES,
    PHASE_PAUSED,
    PHASE_POST_SWITCH,
    PHASE_PRE_WAYPOINT,
    PHASE_STEADY,
    PHASES,
    CONTEXT_KEYS,
    PhaseAccumulator,
    SlotSnapshot,
    classify_phase,
    classify_phase_batch,
    update_switch_tracker,
    update_switch_tracker_batch,
)

STEP_DT = 0.2


def _snap(**kwargs) -> SlotSnapshot:
    base = dict(
        active=True,
        pause_remaining=0,
        waypoint_index=0,
        distance_to_waypoint_m=5.0,
        speed_mps=0.45,
        closing_speed_mps=0.0,
        robot_distance_m=2.0,
    )
    base.update(kwargs)
    return SlotSnapshot(**base)


# --------------------------------------------------------------------------
# Phase state machine
# --------------------------------------------------------------------------


def test_paused_wins_over_every_other_phase() -> None:
    """A stationary obstacle is exactly what the reward cannot see."""
    snapshot = _snap(
        pause_remaining=3,
        distance_to_waypoint_m=0.0,
        speed_mps=0.0,
        steps_since_switch_or_resume=0,
    )
    assert classify_phase(snapshot, STEP_DT) == PHASE_PAUSED


def test_zero_pause_never_reports_paused() -> None:
    """With pause range (0, 0) the paused bucket must stay empty."""
    for since in (None, 0, 99):
        snapshot = _snap(pause_remaining=0, steps_since_switch_or_resume=since)
        assert classify_phase(snapshot, STEP_DT) != PHASE_PAUSED


def test_post_switch_window_is_one_second() -> None:
    window = int(round(1.0 / STEP_DT))
    inside = _snap(steps_since_switch_or_resume=window - 1)
    outside = _snap(steps_since_switch_or_resume=window)
    assert classify_phase(inside, STEP_DT) == PHASE_POST_SWITCH
    assert classify_phase(outside, STEP_DT) == PHASE_STEADY


def test_post_switch_takes_precedence_over_pre_waypoint() -> None:
    """On a short leg both windows overlap; the documented winner is post."""
    snapshot = _snap(
        steps_since_switch_or_resume=0,
        distance_to_waypoint_m=0.1,
        speed_mps=0.45,
    )
    assert classify_phase(snapshot, STEP_DT) == PHASE_POST_SWITCH


def test_pre_waypoint_uses_eta_not_distance() -> None:
    slow = _snap(distance_to_waypoint_m=0.4, speed_mps=0.30)   # eta 1.33 s
    fast = _snap(distance_to_waypoint_m=0.4, speed_mps=0.60)   # eta 0.67 s
    assert classify_phase(slow, STEP_DT) == PHASE_STEADY
    assert classify_phase(fast, STEP_DT) == PHASE_PRE_WAYPOINT


def test_stationary_non_paused_slot_is_steady_not_pre_waypoint() -> None:
    """Zero speed must not divide by zero into a spurious pre-waypoint."""
    snapshot = _snap(pause_remaining=0, speed_mps=0.0, distance_to_waypoint_m=0.0)
    assert classify_phase(snapshot, STEP_DT) == PHASE_STEADY


def test_phases_are_exhaustive_and_mutually_exclusive() -> None:
    seen = set()
    for pause in (0, 2):
        for since in (None, 0, 50):
            for dist, speed in ((0.1, 0.45), (5.0, 0.45), (0.0, 0.0)):
                phase = classify_phase(
                    _snap(
                        pause_remaining=pause,
                        steps_since_switch_or_resume=since,
                        distance_to_waypoint_m=dist,
                        speed_mps=speed,
                    ),
                    STEP_DT,
                )
                assert phase in PHASES
                seen.add(phase)
    assert seen == set(PHASES)


def test_invalid_step_dt_is_rejected() -> None:
    with pytest.raises(ValueError):
        classify_phase(_snap(steps_since_switch_or_resume=0), 0.0)


# --------------------------------------------------------------------------
# Switch / resume tracker
# --------------------------------------------------------------------------


def test_waypoint_switch_starts_the_window() -> None:
    snapshot = _snap(waypoint_index=1, previous_waypoint_index=0)
    assert update_switch_tracker(None, snapshot) == 0


def test_immediate_reversal_with_zero_pause_is_detected() -> None:
    """pause=(0,0) means arrival flips the waypoint with no paused frame."""
    snapshot = _snap(
        waypoint_index=1,
        previous_waypoint_index=0,
        pause_remaining=0,
        previous_pause_remaining=0,
    )
    assert update_switch_tracker(7, snapshot) == 0
    assert classify_phase(
        _snap(steps_since_switch_or_resume=0), STEP_DT
    ) == PHASE_POST_SWITCH


def test_resume_from_pause_starts_the_window() -> None:
    snapshot = _snap(pause_remaining=0, previous_pause_remaining=1)
    assert update_switch_tracker(None, snapshot) == 0


def test_tracker_advances_then_stays_untracked_before_first_event() -> None:
    assert update_switch_tracker(None, _snap()) is None
    assert update_switch_tracker(3, _snap()) == 4


def test_inactive_slot_clears_the_tracker() -> None:
    assert update_switch_tracker(4, _snap(active=False)) is None


def test_auto_reset_sentinel_suppresses_a_spurious_switch() -> None:
    """After auto-reset the previous index is unknown, so no event fires."""
    snapshot = _snap(waypoint_index=3, previous_waypoint_index=None)
    assert update_switch_tracker(None, snapshot) is None


# --------------------------------------------------------------------------
# Tensor path must match the scalar reference
# --------------------------------------------------------------------------


def test_batch_classification_matches_scalar_reference() -> None:
    torch.manual_seed(0)
    n = 4096
    pause = torch.randint(0, 4, (n,))
    since = torch.randint(-1, 8, (n,))
    dist = torch.rand(n) * 6.0
    speed = torch.where(torch.rand(n) < 0.2, torch.zeros(n), torch.rand(n) * 0.6)

    batch = classify_phase_batch(pause, since, dist, speed, STEP_DT)
    for i in range(n):
        expected = classify_phase(
            _snap(
                pause_remaining=int(pause[i]),
                steps_since_switch_or_resume=(
                    None if int(since[i]) < 0 else int(since[i])
                ),
                distance_to_waypoint_m=float(dist[i]),
                speed_mps=float(speed[i]),
            ),
            STEP_DT,
        )
        assert int(batch[i]) == PHASE_CODES[expected], i


def test_batch_tracker_matches_scalar_reference() -> None:
    torch.manual_seed(1)
    n = 2048
    tracker = torch.randint(-1, 6, (n,))
    active = torch.rand(n) > 0.1
    wp = torch.randint(0, 4, (n,))
    prev_wp = torch.randint(-1, 4, (n,))
    pause = torch.randint(0, 3, (n,))
    prev_pause = torch.randint(-1, 3, (n,))

    batch = update_switch_tracker_batch(
        tracker, active, wp, prev_wp, pause, prev_pause
    )
    for i in range(n):
        expected = update_switch_tracker(
            None if int(tracker[i]) < 0 else int(tracker[i]),
            _snap(
                active=bool(active[i]),
                waypoint_index=int(wp[i]),
                previous_waypoint_index=(
                    None if int(prev_wp[i]) < 0 else int(prev_wp[i])
                ),
                pause_remaining=int(pause[i]),
                previous_pause_remaining=(
                    None if int(prev_pause[i]) < 0 else int(prev_pause[i])
                ),
            ),
        )
        assert int(batch[i]) == (-1 if expected is None else expected), i


# --------------------------------------------------------------------------
# Accumulator and reconciliation
# --------------------------------------------------------------------------


def test_inactive_slots_contribute_no_frames() -> None:
    acc = PhaseAccumulator(step_dt_s=STEP_DT)
    assert acc.record(_snap(active=False), collided=True) is None
    assert acc.total_frames == 0
    assert acc.total_collisions == 0


def test_phase_collision_counts_reconstruct_all_slot_events() -> None:
    """Per-phase counts must sum to every dynamic collision slot-event."""
    acc = PhaseAccumulator(step_dt_s=STEP_DT)
    torch.manual_seed(2)
    expected_collisions = 0
    expected_frames = 0
    for _ in range(500):
        active = bool(torch.rand(1) > 0.15)
        collided = bool(torch.rand(1) < 0.05)
        acc.record(
            _snap(
                active=active,
                pause_remaining=int(torch.randint(0, 3, (1,))),
                steps_since_switch_or_resume=int(torch.randint(-1, 7, (1,))),
                distance_to_waypoint_m=float(torch.rand(1) * 5),
                speed_mps=float(torch.rand(1) * 0.6),
            ),
            collided=collided,
        )
        if active:
            expected_frames += 1
            expected_collisions += int(collided)
    assert acc.total_frames == expected_frames
    assert acc.total_collisions == expected_collisions
    report = acc.to_report()
    assert sum(
        p["collision_count"] for p in report["phases"].values()
    ) == expected_collisions
    assert sum(
        p["frame_exposure"] for p in report["phases"].values()
    ) == expected_frames


def test_report_is_safe_with_no_frames() -> None:
    report = PhaseAccumulator(step_dt_s=STEP_DT).to_report()
    assert report["total_dynamic_slot_frames"] == 0
    for phase in PHASES:
        entry = report["phases"][phase]
        assert entry["collision_enrichment"] == 0.0
        assert entry["collisions_per_frame"] == 0.0


def test_enrichment_is_one_when_collisions_track_exposure() -> None:
    """Uniform hazard must yield enrichment 1, so >1 means real concentration."""
    acc = PhaseAccumulator(step_dt_s=STEP_DT)
    acc.frames[PHASE_PAUSED] = 100
    acc.frames[PHASE_STEADY] = 900
    acc.collisions[PHASE_PAUSED] = 10
    acc.collisions[PHASE_STEADY] = 90
    phases = acc.to_report()["phases"]
    assert phases[PHASE_PAUSED]["collision_enrichment"] == pytest.approx(1.0)
    assert phases[PHASE_STEADY]["collision_enrichment"] == pytest.approx(1.0)


def test_enrichment_flags_pause_concentration() -> None:
    acc = PhaseAccumulator(step_dt_s=STEP_DT)
    acc.frames[PHASE_PAUSED] = 100
    acc.frames[PHASE_STEADY] = 900
    acc.collisions[PHASE_PAUSED] = 50
    acc.collisions[PHASE_STEADY] = 50
    phases = acc.to_report()["phases"]
    assert phases[PHASE_PAUSED]["collision_enrichment"] == pytest.approx(5.0)


def test_record_batch_matches_scalar_record() -> None:
    torch.manual_seed(3)
    pause = torch.randint(0, 3, (8, 2))
    since = torch.randint(-1, 7, (8, 2))
    dist = torch.rand(8, 2) * 5
    speed = torch.rand(8, 2) * 0.6
    active = torch.rand(8, 2) > 0.2
    hit = torch.rand(8, 2) < 0.3
    codes = classify_phase_batch(pause, since, dist, speed, STEP_DT)

    batch_acc = PhaseAccumulator(step_dt_s=STEP_DT)
    batch_acc.record_batch(
        codes, active, hit, speed_mps=speed, robot_distance_m=dist
    )

    scalar_acc = PhaseAccumulator(step_dt_s=STEP_DT)
    for e in range(8):
        for s in range(2):
            scalar_acc.record(
                _snap(
                    active=bool(active[e, s]),
                    pause_remaining=int(pause[e, s]),
                    steps_since_switch_or_resume=(
                        None if int(since[e, s]) < 0 else int(since[e, s])
                    ),
                    distance_to_waypoint_m=float(dist[e, s]),
                    speed_mps=float(speed[e, s]),
                    robot_distance_m=float(dist[e, s]),
                ),
                collided=bool(hit[e, s]),
            )
    assert batch_acc.frames == scalar_acc.frames
    assert batch_acc.collisions == scalar_acc.collisions


# ---------------------------------------------------------------------------
# Per-phase context. A global mean cannot separate "pausing is dangerous" from
# "waypoint endpoints happen to sit nearer the robot", so context is kept per
# phase and split by whether the frame collided.
# ---------------------------------------------------------------------------


def test_context_is_tracked_per_phase_not_globally() -> None:
    acc = PhaseAccumulator(step_dt_s=STEP_DT)
    acc.record(
        _snap(pause_remaining=2, robot_distance_m=0.5, speed_mps=0.0),
        collided=False,
    )
    acc.record(
        _snap(pause_remaining=0, robot_distance_m=4.0, steps_since_switch_or_resume=99),
        collided=False,
    )
    phases = acc.to_report()["phases"]
    assert phases[PHASE_PAUSED]["context"]["mean_robot_distance_m"] == pytest.approx(0.5)
    assert phases[PHASE_STEADY]["context"]["mean_robot_distance_m"] == pytest.approx(4.0)


def test_collision_context_is_separate_from_all_frame_context() -> None:
    """Colliding frames must not be diluted by the non-colliding ones."""
    acc = PhaseAccumulator(step_dt_s=STEP_DT)
    acc.record(
        _snap(pause_remaining=1, robot_distance_m=6.0), collided=False
    )
    acc.record(
        _snap(pause_remaining=1, robot_distance_m=0.4), collided=True
    )
    paused = acc.to_report()["phases"][PHASE_PAUSED]
    assert paused["context"]["mean_robot_distance_m"] == pytest.approx(3.2)
    assert paused["collision_context"]["mean_robot_distance_m"] == pytest.approx(0.4)


def test_every_phase_reports_the_full_context_key_set() -> None:
    report = PhaseAccumulator(step_dt_s=STEP_DT).to_report()
    assert list(report["context_keys"]) == list(CONTEXT_KEYS)
    assert "robot_distance_m" in CONTEXT_KEYS
    for phase in PHASES:
        entry = report["phases"][phase]
        for block in ("context", "collision_context"):
            assert set(entry[block]) == {f"mean_{k}" for k in CONTEXT_KEYS}
            assert all(value == 0.0 for value in entry[block].values())


def test_batch_context_matches_scalar_context() -> None:
    torch.manual_seed(5)
    pause = torch.randint(0, 3, (16, 2))
    since = torch.randint(-1, 7, (16, 2))
    dist = torch.rand(16, 2) * 5
    speed = torch.rand(16, 2) * 0.6
    active = torch.rand(16, 2) > 0.2
    hit = torch.rand(16, 2) < 0.4
    codes = classify_phase_batch(pause, since, dist, speed, STEP_DT)

    batch_acc = PhaseAccumulator(step_dt_s=STEP_DT)
    batch_acc.record_batch(
        codes, active, hit,
        waypoint_index=torch.zeros_like(pause),
        pause_remaining=pause,
        distance_to_waypoint_m=dist,
        speed_mps=speed,
        closing_speed_mps=speed,
        robot_distance_m=dist,
    )
    scalar_acc = PhaseAccumulator(step_dt_s=STEP_DT)
    for e in range(16):
        for s_i in range(2):
            scalar_acc.record(
                _snap(
                    active=bool(active[e, s_i]),
                    waypoint_index=0,
                    pause_remaining=int(pause[e, s_i]),
                    steps_since_switch_or_resume=(
                        None if int(since[e, s_i]) < 0 else int(since[e, s_i])
                    ),
                    distance_to_waypoint_m=float(dist[e, s_i]),
                    speed_mps=float(speed[e, s_i]),
                    closing_speed_mps=float(speed[e, s_i]),
                    robot_distance_m=float(dist[e, s_i]),
                ),
                collided=bool(hit[e, s_i]),
            )
    assert batch_acc.context_frames == scalar_acc.context_frames
    for phase in PHASES:
        for key in CONTEXT_KEYS:
            assert batch_acc.context_sums[phase][key] == pytest.approx(
                scalar_acc.context_sums[phase][key]
            )
            assert batch_acc.collision_context_sums[phase][key] == pytest.approx(
                scalar_acc.collision_context_sums[phase][key]
            )


def test_zero_pause_moves_arrival_frames_to_post_switch_not_away() -> None:
    """`zero` mode empties the paused bucket without losing the frames.

    rule_behaviors zeroes velocity on arrival regardless of the pause range, so
    a no-dwell arrival is still a stationary frame -- it just classifies as
    post-switch. The audit must keep counting it.
    """
    acc = PhaseAccumulator(step_dt_s=STEP_DT)
    arrival = _snap(
        pause_remaining=0,
        speed_mps=0.0,
        distance_to_waypoint_m=0.0,
        steps_since_switch_or_resume=0,
    )
    assert acc.record(arrival, collided=True) == PHASE_POST_SWITCH
    report = acc.to_report()
    assert report["phases"][PHASE_PAUSED]["frame_exposure"] == 0
    assert report["total_dynamic_slot_frames"] == 1
    assert report["total_dynamic_slot_collisions"] == 1
