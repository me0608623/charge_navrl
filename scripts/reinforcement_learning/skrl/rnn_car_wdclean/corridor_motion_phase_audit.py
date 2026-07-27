"""Eval-only motion-phase attribution for deployment-corridor dynamic obstacles.

Why this exists
---------------
The corridor patrol obstacles hard-zero their velocity on waypoint arrival
(``rule_behaviors.py``: ``sched.velocities[r_env, r_obs] = 0.0``) and then
reverse. The future-occupancy reward only treats an obstacle as a threat when
``\\|v\\| >= future_occupancy_move_threshold_mps`` (0.1 m/s by default) and
extrapolates it at constant velocity, so a paused obstacle contributes exactly
zero risk and a reversal is mispredicted right up to the switch.

``random_2d`` patrols a ~1.5 m leg against ``longitudinal``'s ~6.8 m, so it
arrives, pauses and reverses several times more often. That makes the
mechanism a *candidate* driver of its gate failure -- frequency alone is not
proof. This module measures the distribution directly: it splits every dynamic
obstacle frame into mutually exclusive phases and reports how collisions
concentrate across them.

What this can and cannot establish
----------------------------------
This audit, and the pause A/B it feeds, run a **fixed policy**. They can show
whether stationary/reversing frames carry disproportionate collision risk *for
the policy as it currently is*. They **cannot** by themselves show that the
future-occupancy reward caused the training outcome -- that would need a
training-side intervention, which is out of scope here.

Two further limits on the ``zero`` pause mode specifically:

* ``rule_behaviors.py`` zeroes the obstacle velocity on waypoint arrival
  *unconditionally*, outside the pause branch. With ``pause_steps_range``
  set to ``(0, 0)`` there is still one stationary frame per arrival, so
  ``zero`` removes the extra **dwell**, not every stationary-blind frame.
* With no dwell, arrival frames carry ``pause_remaining == 0`` and are
  therefore classified as ``post_switch_or_resume_1s`` rather than ``paused``.
  A ``zero``-mode run legitimately reports an empty ``paused`` bucket; the
  stationary arrival frames have moved into the post-switch bucket, they have
  not disappeared.

Design constraints
------------------
* **Snapshot before ``env.step()``.** ``ManagerBasedRLEnv`` auto-resets done
  envs inside ``step``, which rewrites waypoint index, pause counter and
  positions. Classifying afterwards would attribute a collision to the *next*
  episode's phase.
* **Never touch collision determination.** The auditor consumes a per-slot hit
  mask that the termination function records under an eval-only flag; it does
  not decide what a collision is.
* **Reconcilable.** Per-phase collision counts sum to the total dynamic
  slot-events, so the audit JSON can be checked against the main corridor JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field


PHASE_PAUSED = "paused"
PHASE_POST_SWITCH = "post_switch_or_resume_1s"
PHASE_PRE_WAYPOINT = "pre_waypoint_1s"
PHASE_STEADY = "steady"

#: Mutually exclusive, exhaustive, and ordered by precedence.
PHASES = (PHASE_PAUSED, PHASE_POST_SWITCH, PHASE_PRE_WAYPOINT, PHASE_STEADY)

#: The audit window either side of a waypoint event, in seconds.
PHASE_WINDOW_S = 1.0


@dataclass(frozen=True)
class SlotSnapshot:
    """State of one dynamic obstacle slot captured before ``env.step()``.

    Attributes are plain floats/ints so the state machine is testable without
    a simulator or torch.
    """

    active: bool
    pause_remaining: int
    waypoint_index: int
    distance_to_waypoint_m: float
    speed_mps: float
    closing_speed_mps: float
    #: Distance from the obstacle to the robot. Without it, a phase's
    #: enrichment could simply reflect that waypoint endpoints sit nearer the
    #: robot's lane rather than anything about the motion phase itself.
    robot_distance_m: float = 0.0
    #: ``waypoint_index`` from the previous captured frame, or ``None`` on the
    #: first frame of an episode. A change means the obstacle just switched
    #: target, i.e. it arrived and (after any pause) reversed.
    previous_waypoint_index: int | None = None
    #: ``pause_remaining`` from the previous captured frame, or ``None``.
    previous_pause_remaining: int | None = None
    #: Steps since the most recent switch/resume, or ``None`` if there has not
    #: been one yet this episode.
    steps_since_switch_or_resume: int | None = None


def classify_phase(snapshot: SlotSnapshot, step_dt_s: float) -> str:
    """Return the mutually exclusive phase for one slot-frame.

    Precedence is deliberate and documented because the windows overlap:

    1. ``paused`` -- the obstacle is stationary *now*. This is the state the
       reward is blind to, so it wins outright.
    2. ``post_switch_or_resume_1s`` -- within one second of a waypoint switch
       or of resuming from a pause. Constant-velocity extrapolation was
       pointing the wrong way immediately before this.
    3. ``pre_waypoint_1s`` -- within one second of *reaching* the next
       waypoint, i.e. about to pause/reverse. Extrapolation is about to become
       wrong.
    4. ``steady`` -- everything else.

    A frame can satisfy both (2) and (3) on a short leg; post-switch wins
    because the mispredicted history is already in the policy's observation.
    """
    if snapshot.pause_remaining > 0:
        return PHASE_PAUSED

    window_steps = _window_steps(step_dt_s)
    since = snapshot.steps_since_switch_or_resume
    if since is not None and since < window_steps:
        return PHASE_POST_SWITCH

    if snapshot.speed_mps > 0.0:
        eta_s = snapshot.distance_to_waypoint_m / snapshot.speed_mps
        if eta_s <= PHASE_WINDOW_S:
            return PHASE_PRE_WAYPOINT

    return PHASE_STEADY


def _window_steps(step_dt_s: float) -> int:
    """Number of steps covering :data:`PHASE_WINDOW_S`, at least one."""
    if step_dt_s <= 0.0:
        raise ValueError(f"step_dt_s must be positive, got {step_dt_s}")
    return max(1, int(round(PHASE_WINDOW_S / step_dt_s)))


def update_switch_tracker(
    previous: int | None,
    snapshot: SlotSnapshot,
) -> int | None:
    """Advance ``steps_since_switch_or_resume`` for one slot.

    Returns ``0`` on the frame a switch or resume is detected, ``previous + 1``
    while a tracked window runs, and ``None`` when nothing has happened yet.

    A *resume* is ``pause_remaining`` reaching zero after having been positive;
    a *switch* is ``waypoint_index`` changing. Both are the same event class
    for the reward: the extrapolated velocity just became stale.
    """
    if not snapshot.active:
        return None

    switched = (
        snapshot.previous_waypoint_index is not None
        and snapshot.waypoint_index != snapshot.previous_waypoint_index
    )
    resumed = (
        snapshot.previous_pause_remaining is not None
        and snapshot.previous_pause_remaining > 0
        and snapshot.pause_remaining == 0
    )
    if switched or resumed:
        return 0
    if previous is None:
        return None
    return previous + 1


CONTEXT_KEYS = (
    "waypoint_index",
    "pause_remaining",
    "distance_to_waypoint_m",
    "speed_mps",
    "closing_speed_mps",
    "robot_distance_m",
)


#: Integer codes matching :data:`PHASES` order, for the tensor path.
PHASE_CODES = {phase: index for index, phase in enumerate(PHASES)}


def classify_phase_batch(
    pause_remaining,
    steps_since_switch_or_resume,
    distance_to_waypoint_m,
    speed_mps,
    step_dt_s: float,
):
    """Vectorised :func:`classify_phase` over ``[E, S]`` tensors.

    ``steps_since_switch_or_resume`` uses a negative sentinel for "no switch
    seen yet this episode", which is the tensor equivalent of ``None``.

    Returns an integer tensor of :data:`PHASE_CODES` values. The scalar
    :func:`classify_phase` remains the reference implementation; the two are
    checked against each other elementwise in the test suite so the fast path
    cannot drift from the documented precedence.
    """
    import torch

    window_steps = _window_steps(step_dt_s)
    phase = torch.full_like(pause_remaining, PHASE_CODES[PHASE_STEADY])

    moving = speed_mps > 0.0
    eta = torch.where(
        moving,
        distance_to_waypoint_m / torch.where(
            moving, speed_mps, torch.ones_like(speed_mps)
        ),
        torch.full_like(speed_mps, float("inf")),
    )
    phase = torch.where(
        eta <= PHASE_WINDOW_S,
        torch.full_like(phase, PHASE_CODES[PHASE_PRE_WAYPOINT]),
        phase,
    )
    tracked = steps_since_switch_or_resume >= 0
    phase = torch.where(
        tracked & (steps_since_switch_or_resume < window_steps),
        torch.full_like(phase, PHASE_CODES[PHASE_POST_SWITCH]),
        phase,
    )
    phase = torch.where(
        pause_remaining > 0,
        torch.full_like(phase, PHASE_CODES[PHASE_PAUSED]),
        phase,
    )
    return phase


def update_switch_tracker_batch(
    tracker,
    active,
    waypoint_index,
    previous_waypoint_index,
    pause_remaining,
    previous_pause_remaining,
):
    """Vectorised :func:`update_switch_tracker`; ``-1`` is the ``None`` sentinel.

    ``previous_*`` carry ``-1`` when there is no previous frame (episode start
    or just after an auto-reset), which suppresses a spurious switch event.
    """
    import torch

    switched = (previous_waypoint_index >= 0) & (
        waypoint_index != previous_waypoint_index
    )
    resumed = (previous_pause_remaining > 0) & (pause_remaining == 0)
    event = switched | resumed

    advanced = torch.where(
        tracker >= 0, tracker + 1, torch.full_like(tracker, -1)
    )
    updated = torch.where(event, torch.zeros_like(tracker), advanced)
    return torch.where(active, updated, torch.full_like(tracker, -1))


@dataclass
class PhaseAccumulator:
    """Frame exposure and collision counts per phase.

    ``collision_fraction`` is the share of *collisions* landing in a phase;
    ``collision_enrichment`` is that share divided by the phase's share of
    *frames*. Enrichment above one means collisions concentrate there beyond
    what mere exposure explains -- which is the quantity that distinguishes
    "random_2d pauses more often" from "pausing is what makes it crash".
    """

    step_dt_s: float
    frames: dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in PHASES}
    )
    collisions: dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in PHASES}
    )
    #: Extra per-frame context, kept **per phase** and split by whether the
    #: frame collided. A global mean cannot answer "are paused collisions
    #: concentrated at a particular position or closing speed", which is
    #: exactly what distinguishes a real phase effect from a geometric
    #: coincidence.
    context_sums: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            phase: {key: 0.0 for key in CONTEXT_KEYS} for phase in PHASES
        }
    )
    collision_context_sums: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            phase: {key: 0.0 for key in CONTEXT_KEYS} for phase in PHASES
        }
    )
    context_frames: dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in PHASES}
    )

    def record(
        self,
        snapshot: SlotSnapshot,
        *,
        collided: bool,
    ) -> str | None:
        """Record one slot-frame. Returns its phase, or ``None`` if inactive."""
        if not snapshot.active:
            return None
        phase = classify_phase(snapshot, self.step_dt_s)
        self.frames[phase] += 1
        if collided:
            self.collisions[phase] += 1
        values = {
            "waypoint_index": float(snapshot.waypoint_index),
            "pause_remaining": float(snapshot.pause_remaining),
            "distance_to_waypoint_m": float(snapshot.distance_to_waypoint_m),
            "speed_mps": float(snapshot.speed_mps),
            "closing_speed_mps": float(snapshot.closing_speed_mps),
            "robot_distance_m": float(snapshot.robot_distance_m),
        }
        for key, value in values.items():
            self.context_sums[phase][key] += value
            if collided:
                self.collision_context_sums[phase][key] += value
        self.context_frames[phase] += 1
        return phase

    def record_batch(
        self,
        phase_codes,
        active,
        collided,
        *,
        waypoint_index=None,
        pause_remaining=None,
        distance_to_waypoint_m=None,
        speed_mps=None,
        closing_speed_mps=None,
        robot_distance_m=None,
    ) -> None:
        """Accumulate one step of ``[E, S]`` tensors from the play loop."""
        import torch

        active_bool = active.bool()
        n_active = int(active_bool.sum().item())
        if n_active == 0:
            return
        collided_active = collided.bool() & active_bool
        for phase in PHASES:
            code = PHASE_CODES[phase]
            in_phase = active_bool & (phase_codes == code)
            self.frames[phase] += int(in_phase.sum().item())
            self.collisions[phase] += int(
                (in_phase & collided_active).sum().item()
            )
        context = {
            "waypoint_index": waypoint_index,
            "pause_remaining": pause_remaining,
            "distance_to_waypoint_m": distance_to_waypoint_m,
            "speed_mps": speed_mps,
            "closing_speed_mps": closing_speed_mps,
            "robot_distance_m": robot_distance_m,
        }
        for phase in PHASES:
            in_phase = active_bool & (phase_codes == PHASE_CODES[phase])
            if not bool(in_phase.any()):
                continue
            hit_in_phase = in_phase & collided_active
            self.context_frames[phase] += int(in_phase.sum().item())
            for key, tensor in context.items():
                if tensor is None:
                    continue
                values = tensor.to(torch.float64)
                self.context_sums[phase][key] += float(
                    values[in_phase].sum().item()
                )
                if bool(hit_in_phase.any()):
                    self.collision_context_sums[phase][key] += float(
                        values[hit_in_phase].sum().item()
                    )

    @property
    def total_frames(self) -> int:
        return sum(self.frames.values())

    @property
    def total_collisions(self) -> int:
        return sum(self.collisions.values())

    def to_report(self) -> dict:
        """Serializable summary, safe to emit with zero frames recorded."""
        total_frames = self.total_frames
        total_collisions = self.total_collisions
        phases = {}
        for phase in PHASES:
            frames = self.frames[phase]
            collisions = self.collisions[phase]
            frame_share = frames / total_frames if total_frames else 0.0
            collision_fraction = (
                collisions / total_collisions if total_collisions else 0.0
            )
            context_frames = self.context_frames[phase]
            phases[phase] = {
                "frame_exposure": frames,
                "frame_exposure_fraction": frame_share,
                "collision_count": collisions,
                "collision_fraction": collision_fraction,
                "collision_enrichment": (
                    collision_fraction / frame_share if frame_share > 0.0 else 0.0
                ),
                "collisions_per_frame": (
                    collisions / frames if frames else 0.0
                ),
                # Per-phase context. Compare `context` (all frames in the
                # phase) against `collision_context` (only the frames that
                # collided): if enrichment were merely geometric -- waypoint
                # endpoints sitting nearer the robot's lane -- the two would
                # agree, and robot_distance_m would be the tell.
                "context": {
                    f"mean_{key}": (
                        self.context_sums[phase][key] / context_frames
                        if context_frames else 0.0
                    )
                    for key in CONTEXT_KEYS
                },
                "collision_context": {
                    f"mean_{key}": (
                        self.collision_context_sums[phase][key] / collisions
                        if collisions else 0.0
                    )
                    for key in CONTEXT_KEYS
                },
            }
        return {
            "step_dt_s": self.step_dt_s,
            "phase_window_s": PHASE_WINDOW_S,
            "total_dynamic_slot_frames": total_frames,
            "total_dynamic_slot_collisions": total_collisions,
            "context_keys": list(CONTEXT_KEYS),
            "phases": phases,
        }
