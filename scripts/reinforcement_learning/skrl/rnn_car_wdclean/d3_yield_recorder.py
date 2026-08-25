"""SA4-D3 timing recorder and frozen mechanism-probe shield arms.

The recorder and future intervention arms share one hook so their timelines are
directly comparable. The intervention protocol was frozen only after the
transparent baseline timing evidence had been analysed.

The hook belongs before ``env.step``::

    policy indices -> optional shield -> process_actions
                   -> decode -> delay -> scale -> lag -> simulator

Changing ``processed_actions`` would bypass the actuator queue and invalidate a
fixed-delay comparison. This module never starts a process or a training run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import torch


DT_S = 0.2
# Twenty-five lead steps plus the event frame span exactly 5.0 seconds.
MAX_LEAD_STEPS = 25
WINDOW_CAPACITY = MAX_LEAD_STEPS + 1
NEAR_DISTANCE_M = 3.0
NEAR_EXIT_DISTANCE_M = 3.25
RISK_WARNING = 0.05
DECEL_DELTA_MPS = 0.02
STOP_SPEED_MPS = 0.10
DELAY_TOLERANCE = 1.0e-6
CONTACT_DISTANCE_M = 0.70
PROBE_TTC_S = 2.0
PROBE_CONFIRM_STEPS = 2
PROBE_RELEASE_STEPS = 2
TURN_SIDE_EPSILON_M = 0.05
ACTION_CENTER_INDEX = 9
ACTION_RIGHT_INDEX = 0
ACTION_LEFT_INDEX = 18
PROTOCOL_SCHEMA = "sa4_d3_intervention_protocol/v1"


class ShieldMode(str, Enum):
    """Frozen 2x2 mechanism-probe arms."""

    BASELINE = "baseline"
    SUSTAINED_BRAKE = "sustained_brake"
    BEST_TURN = "best_turn"
    COMBINED = "combined"


ENABLED_MODES = frozenset(ShieldMode)
# D4 reuses the frozen D3 timing recorder without changing the historical D3
# intervention enum or protocol hash. These modes supply their own selector
# and protocol, while the recorder treats them as non-baseline interventions.
EXTERNAL_RECORDER_MODES = frozenset(
    {"geometry_feasible", "geometry_feasible_argmin"}
)
CORRIDOR_MOTION_MODES = frozenset(
    {"lateral", "longitudinal", "random_2d", "mixed"}
)


def validate_audit_motion_scope(
    motion_mode: str,
    shield_mode: ShieldMode | str,
    *,
    feasibility_shadow: bool,
) -> None:
    """Keep historical D3 interventions lateral-only.

    D5 only borrows the identity recorder for transition and event accounting.
    Its baseline tensor contract is motion-agnostic, so the D5 shadow may use
    every registered corridor motion family without enabling a D3/D4 action
    intervention outside the original lateral protocol.
    """

    motion = str(motion_mode)
    mode = shield_mode.value if isinstance(shield_mode, ShieldMode) else str(
        shield_mode
    )
    if motion not in CORRIDOR_MOTION_MODES:
        raise ValueError(f"unsupported corridor motion mode {motion!r}")
    if motion == "lateral":
        return
    if mode == ShieldMode.BASELINE.value and bool(feasibility_shadow):
        return
    raise ValueError(
        "non-lateral D3 recording is allowed only for the identity baseline "
        "used by D5 feasibility shadow; intervention protocols remain "
        "lateral-only"
    )


def shield_protocol() -> dict:
    """Return the pre-registered trigger, hold and action-factor contract."""
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "scope": "diagnostic mechanism probe; not a deployment safety rule",
        "trigger": {
            "distance_max_m": NEAR_DISTANCE_M,
            "closing_speed_min_mps": 0.0,
            "contact_distance_m": CONTACT_DISTANCE_M,
            "ttc_formula": "max(distance-0.70, 0) / closing_speed",
            "ttc_max_s": PROBE_TTC_S,
            "consecutive_frames": PROBE_CONFIRM_STEPS,
            "control_dt_s": DT_S,
        },
        "hold": {
            "tracked_obstacle_latched_at_trigger": True,
            "release_when": (
                "tracked obstacle distance >= 3.25m OR radial closing <= 0"
            ),
            "release_consecutive_frames": PROBE_RELEASE_STEPS,
            "episode_reset_clears_latch": True,
            "maximum_hold_steps": None,
        },
        "factors": {
            "baseline": "policy linear and angular indices unchanged",
            "sustained_brake": (
                "replace only linear index with the action whose decoded next "
                "velocity is closest to zero; preserve policy angular index"
            ),
            "best_turn": (
                "preserve policy linear index; latch maximum turn away from "
                "the tracked obstacle (near center, turn behind its lateral motion)"
            ),
            "combined": "apply both sustained_brake and best_turn factors",
        },
        "actions": {
            "num_bins": 19,
            "center_index": ACTION_CENTER_INDEX,
            "maximum_right_index": ACTION_RIGHT_INDEX,
            "maximum_left_index": ACTION_LEFT_INDEX,
            "turn_side_epsilon_m": TURN_SIDE_EPSILON_M,
        },
        "actuator_order": (
            "policy indices -> shield indices -> decode -> d1 queue -> applied"
        ),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


class BaselineShield:
    """Identity shield: return the exact object supplied by the policy."""

    mode = ShieldMode.BASELINE

    def __init__(self) -> None:
        self._calls = 0
        self._environment_frames = 0

    def __call__(self, action_indices, context):
        del context
        self._calls += 1
        self._environment_frames += int(action_indices.shape[0])
        return action_indices

    def reset(self, done_mask) -> None:
        del done_mask

    def report(self) -> dict:
        return {
            "mode": self.mode.value,
            "calls": self._calls,
            "environment_frames": self._environment_frames,
            "trigger_activations": 0,
            "release_events": 0,
            "active_environment_frames": 0,
            "override_environment_frames": 0,
            "active_at_end": 0,
            "protocol_sha256": shield_protocol()["sha256"],
        }


class InterventionShield:
    """Stateful early-yield intervention applied to policy action indices."""

    def __init__(self, mode: ShieldMode) -> None:
        if mode is ShieldMode.BASELINE:
            raise ValueError("InterventionShield requires an intervention mode")
        self.mode = mode
        self._active: torch.Tensor | None = None
        self._confirm: torch.Tensor | None = None
        self._release: torch.Tensor | None = None
        self._tracked_slot: torch.Tensor | None = None
        self._turn_index: torch.Tensor | None = None
        self._calls = 0
        self._environment_frames = 0
        self._trigger_activations = 0
        self._release_events = 0
        self._active_environment_frames = 0
        self._override_environment_frames = 0

    def _ensure_state(self, action_indices: torch.Tensor) -> None:
        if action_indices.ndim != 2 or action_indices.shape[1] != 2:
            raise ValueError("D3 shield actions must have shape [env, 2]")
        count = int(action_indices.shape[0])
        device = action_indices.device
        if self._active is not None:
            if self._active.shape != (count,) or self._active.device != device:
                raise RuntimeError("D3 shield environment shape/device changed")
            return
        self._active = torch.zeros(count, dtype=torch.bool, device=device)
        self._confirm = torch.zeros(count, dtype=torch.long, device=device)
        self._release = torch.zeros(count, dtype=torch.long, device=device)
        self._tracked_slot = torch.zeros(count, dtype=torch.long, device=device)
        self._turn_index = torch.full(
            (count,), ACTION_CENTER_INDEX, dtype=torch.long, device=device
        )

    @staticmethod
    def _require_context_tensor(
        context: dict,
        key: str,
        *,
        shape: tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        value = context.get(key)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"D3 shield context {key!r} must be a tensor")
        if tuple(value.shape) != shape:
            raise ValueError(
                f"D3 shield context {key!r} has shape {tuple(value.shape)}, "
                f"expected {shape}"
            )
        if value.device != device:
            raise ValueError(f"D3 shield context {key!r} is on the wrong device")
        if not bool(torch.isfinite(value.float()).all()):
            raise ValueError(f"D3 shield context {key!r} contains non-finite data")
        return value

    def __call__(self, action_indices, context):
        self._ensure_state(action_indices)
        assert self._active is not None
        assert self._confirm is not None
        assert self._release is not None
        assert self._tracked_slot is not None
        assert self._turn_index is not None

        envs = int(action_indices.shape[0])
        device = action_indices.device
        distances = context.get("obstacle_distances_m")
        closings = context.get("relative_closing_speeds_mps")
        if not isinstance(distances, torch.Tensor) or distances.ndim != 2:
            raise ValueError(
                "D3 shield context 'obstacle_distances_m' must have shape [env, slot]"
            )
        slots = int(distances.shape[1])
        expected_slots = (envs, slots)
        distances = self._require_context_tensor(
            context,
            "obstacle_distances_m",
            shape=expected_slots,
            device=device,
        ).float()
        closings = self._require_context_tensor(
            context,
            "relative_closing_speeds_mps",
            shape=expected_slots,
            device=device,
        ).float()
        body_y = self._require_context_tensor(
            context,
            "obstacle_body_y_m",
            shape=expected_slots,
            device=device,
        ).float()
        body_vy = self._require_context_tensor(
            context,
            "obstacle_body_vy_mps",
            shape=expected_slots,
            device=device,
        ).float()
        brake_indices = self._require_context_tensor(
            context,
            "brake_action_indices",
            shape=(envs,),
            device=device,
        ).round().long()
        if bool(((brake_indices < 0) | (brake_indices > 18)).any()):
            raise ValueError("D3 brake action index is outside [0,18]")

        env_index = torch.arange(envs, device=device)
        tracked_distance = distances[env_index, self._tracked_slot]
        tracked_closing = closings[env_index, self._tracked_slot]
        resolved = (
            (tracked_distance >= NEAR_EXIT_DISTANCE_M)
            | (tracked_closing <= 0.0)
        )
        self._release = torch.where(
            self._active & resolved,
            self._release + 1,
            torch.zeros_like(self._release),
        )
        release_now = self._active & (self._release >= PROBE_RELEASE_STEPS)
        self._release_events += int(release_now.sum().item())
        self._active[release_now] = False
        self._release[release_now] = 0
        self._tracked_slot[release_now] = 0
        self._turn_index[release_now] = ACTION_CENTER_INDEX

        ttc = torch.where(
            closings > 0.0,
            (distances - CONTACT_DISTANCE_M).clamp(min=0.0)
            / closings.clamp(min=1.0e-6),
            torch.full_like(distances, float("inf")),
        )
        qualifies = (
            (distances <= NEAR_DISTANCE_M)
            & (closings > 0.0)
            & (ttc <= PROBE_TTC_S)
        )
        any_qualifies = qualifies.any(dim=1)
        inactive = ~self._active
        self._confirm = torch.where(
            inactive & any_qualifies,
            (self._confirm + 1).clamp(max=PROBE_CONFIRM_STEPS),
            torch.zeros_like(self._confirm),
        )
        trigger_now = inactive & (self._confirm >= PROBE_CONFIRM_STEPS)
        candidate_ttc = torch.where(
            qualifies, ttc, torch.full_like(ttc, float("inf"))
        )
        selected_slot = candidate_ttc.argmin(dim=1)
        selected_y = body_y[env_index, selected_slot]
        selected_vy = body_vy[env_index, selected_slot]
        near_center = selected_y.abs() <= TURN_SIDE_EPSILON_M
        turn_right = torch.where(
            near_center,
            selected_vy >= 0.0,
            selected_y > 0.0,
        )
        selected_turn = torch.where(
            turn_right,
            torch.full_like(selected_slot, ACTION_RIGHT_INDEX),
            torch.full_like(selected_slot, ACTION_LEFT_INDEX),
        )
        self._tracked_slot[trigger_now] = selected_slot[trigger_now]
        self._turn_index[trigger_now] = selected_turn[trigger_now]
        self._active[trigger_now] = True
        self._confirm[trigger_now] = 0
        self._trigger_activations += int(trigger_now.sum().item())

        effective = action_indices.clone()
        active = self._active
        if self.mode in (ShieldMode.SUSTAINED_BRAKE, ShieldMode.COMBINED):
            effective[active, 0] = brake_indices[active].to(effective.dtype)
        if self.mode in (ShieldMode.BEST_TURN, ShieldMode.COMBINED):
            effective[active, 1] = self._turn_index[active].to(effective.dtype)

        changed = (
            effective.round().long() != action_indices.round().long()
        ).any(dim=1)
        self._calls += 1
        self._environment_frames += envs
        self._active_environment_frames += int(active.sum().item())
        self._override_environment_frames += int(changed.sum().item())
        return effective

    def reset(self, done_mask) -> None:
        if self._active is None:
            return
        if not isinstance(done_mask, torch.Tensor):
            raise ValueError("D3 shield reset mask must be a tensor")
        done = done_mask.reshape(-1).to(
            device=self._active.device, dtype=torch.bool
        )
        if done.shape != self._active.shape:
            raise ValueError("D3 shield reset mask has the wrong shape")
        self._active[done] = False
        self._confirm[done] = 0
        self._release[done] = 0
        self._tracked_slot[done] = 0
        self._turn_index[done] = ACTION_CENTER_INDEX

    def report(self) -> dict:
        return {
            "mode": self.mode.value,
            "calls": self._calls,
            "environment_frames": self._environment_frames,
            "trigger_activations": self._trigger_activations,
            "release_events": self._release_events,
            "active_environment_frames": self._active_environment_frames,
            "override_environment_frames": self._override_environment_frames,
            "active_at_end": (
                0 if self._active is None else int(self._active.sum().item())
            ),
            "protocol_sha256": shield_protocol()["sha256"],
        }


def build_shield(mode: ShieldMode | str):
    """Build one pre-registered mechanism-probe arm."""
    mode = ShieldMode(mode)
    if mode is ShieldMode.BASELINE:
        return BaselineShield()
    return InterventionShield(mode)


@dataclass(frozen=True)
class StepRecord:
    """One pre-state/action/transition tuple for one environment.

    ``body_*`` and obstacle fields describe the state in which the policy chose
    the action. ``pre_delay_*`` and ``post_delay_*`` describe the command that
    was decoded and applied during the resulting transition. Terminal flags are
    the outcome of that same transition.
    """

    step: int
    env_id: int
    episode_id: int
    policy_action_indices: tuple[int, int]
    effective_action_indices: tuple[int, int]
    shield_engaged: bool
    pre_delay_v_command_mps: float
    pre_delay_w_command_rad_s: float
    post_delay_v_command_mps: float
    post_delay_w_command_rad_s: float
    body_forward_speed_mps: float
    body_planar_speed_mps: float
    obstacle_center_distance_m: float
    relative_closing_speed_mps: float
    risk: float
    risk_active: bool
    dynamic_obstacle_center_distances_m: tuple[float, ...]
    dynamic_obstacle_relative_closing_speeds_mps: tuple[float, ...]
    dynamic_obstacle_risks: tuple[float, ...]
    dynamic_obstacle_risk_active: tuple[bool, ...]
    dynamic_collision: bool
    done: bool
    termination_cause: int = 0

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["policy_action_indices"] = list(self.policy_action_indices)
        payload["effective_action_indices"] = list(
            self.effective_action_indices
        )
        for key in (
            "dynamic_obstacle_center_distances_m",
            "dynamic_obstacle_relative_closing_speeds_mps",
            "dynamic_obstacle_risks",
            "dynamic_obstacle_risk_active",
        ):
            payload[key] = list(payload[key])
        return payload


def _first_index(records: list[StepRecord], predicate) -> int | None:
    for index, record in enumerate(records):
        if predicate(record):
            return index
    return None


def _first_drop(
    records: list[StepRecord], attribute: str
) -> int | None:
    previous = None
    for index, record in enumerate(records):
        value = abs(float(getattr(record, attribute)))
        if previous is not None and value < previous - DECEL_DELTA_MPS:
            return index
        previous = value
    return None


def _steps_before(index: int | None, last: int) -> int | None:
    return None if index is None else last - index


def _seconds(steps: int | None) -> float | None:
    return None if steps is None else round(float(steps) * DT_S, 3)


def first_events(
    window: Iterable[StepRecord], *, obstacle_slot: int | None = None
) -> dict:
    """Return first warning/deceleration/stop times before the event frame.

    A trigger already true in the oldest row of a full window is left-censored:
    the true onset happened at least five seconds before the event. The report
    marks that explicitly instead of presenting the window edge as an exact
    onset time.
    """
    records = list(window)
    if not records:
        raise ValueError("window is empty")
    last = len(records) - 1

    def _distance(row: StepRecord) -> float:
        if obstacle_slot is None:
            return row.obstacle_center_distance_m
        return row.dynamic_obstacle_center_distances_m[obstacle_slot]

    def _risk(row: StepRecord) -> float:
        if obstacle_slot is None:
            return row.risk
        return row.dynamic_obstacle_risks[obstacle_slot]

    indices = {
        "near_3m": _first_index(
            records,
            lambda row: _distance(row) <= NEAR_DISTANCE_M,
        ),
        "risk_ge_0p05": _first_index(
            records, lambda row: _risk(row) >= RISK_WARNING
        ),
        "issued_decel": _first_drop(
            records, "pre_delay_v_command_mps"
        ),
        "applied_decel": _first_drop(
            records, "post_delay_v_command_mps"
        ),
        "actual_decel": _first_drop(records, "body_planar_speed_mps"),
        "issued_stop": _first_index(
            records,
            lambda row: abs(row.pre_delay_v_command_mps) <= STOP_SPEED_MPS,
        ),
        "applied_stop": _first_index(
            records,
            lambda row: abs(row.post_delay_v_command_mps) <= STOP_SPEED_MPS,
        ),
        "actual_stop": _first_index(
            records, lambda row: row.body_planar_speed_mps <= STOP_SPEED_MPS
        ),
    }
    report: dict[str, int | float | None] = {
        "window_records": len(records),
        "tracked_obstacle_slot": obstacle_slot,
        "max_available_lead_steps": last,
        "max_available_lead_s": round(last * DT_S, 3),
    }
    for label, index in indices.items():
        steps = _steps_before(index, last)
        report[f"first_{label}_steps_before_event"] = steps
        report[f"first_{label}_s_before_event"] = _seconds(steps)
        report[f"first_{label}_left_censored"] = bool(
            index == 0 and len(records) == WINDOW_CAPACITY
        )
    return report


@dataclass
class WindowBuffer:
    """Per-environment ring buffer retaining 25 lead steps plus event frame."""

    capacity: int = WINDOW_CAPACITY
    records: list[StepRecord] = field(default_factory=list)

    def append(self, record: StepRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity :]

    def clear(self) -> None:
        self.records.clear()

    def window(self) -> list[StepRecord]:
        return list(self.records)


class D3YieldRecorder:
    """Collect collision and non-collision near-encounter timing windows."""

    def __init__(
        self,
        num_envs: int,
        *,
        mode: ShieldMode | str = ShieldMode.BASELINE,
        expected_delay_steps: int = 1,
        num_dynamic_obstacles: int = 2,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if expected_delay_steps != 1:
            raise ValueError("SA4-D3 is defined only for fixed d1")
        if num_dynamic_obstacles < 1:
            raise ValueError("num_dynamic_obstacles must be positive")
        self.num_envs = int(num_envs)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        mode_value = mode.value if isinstance(mode, ShieldMode) else str(mode)
        if mode_value in {item.value for item in ShieldMode}:
            self.mode: ShieldMode | str = ShieldMode(mode_value)
        elif mode_value in EXTERNAL_RECORDER_MODES:
            self.mode = mode_value
        else:
            raise ValueError(f"unsupported D3 recorder mode: {mode_value}")
        self.mode_value = mode_value
        self._is_baseline = mode_value == ShieldMode.BASELINE.value
        self.expected_delay_steps = int(expected_delay_steps)
        self._buffers = [WindowBuffer() for _ in range(self.num_envs)]
        self._episode_ids = [0 for _ in range(self.num_envs)]
        self._encounter_phase = [
            ["idle" for _ in range(self.num_dynamic_obstacles)]
            for _ in range(self.num_envs)
        ]
        self._encounter_min_distance = [
            [float("inf") for _ in range(self.num_dynamic_obstacles)]
            for _ in range(self.num_envs)
        ]
        self._previous: list[StepRecord | None] = [
            None for _ in range(self.num_envs)
        ]
        self.events: list[dict] = []
        self._episode_event_indices = [
            [] for _ in range(self.num_envs)
        ]
        self.total_records = 0
        self.shield_engaged_records = 0
        self.dynamic_collision_records = 0
        self.done_records = 0
        self.termination_cause_counts: dict[int, int] = {}
        self.delay_alignment_samples = 0
        self.delay_alignment_errors = 0
        self.delay_alignment_max_abs_error = 0.0

    def episode_id(self, env_id: int) -> int:
        """Return the recorder-owned episode id for one environment."""
        if not 0 <= int(env_id) < self.num_envs:
            raise ValueError(f"invalid env_id {env_id}")
        return self._episode_ids[int(env_id)]

    def _check_record(self, record: StepRecord) -> None:
        if not 0 <= record.env_id < self.num_envs:
            raise ValueError(f"invalid env_id {record.env_id}")
        if record.episode_id != self._episode_ids[record.env_id]:
            raise ValueError(
                f"env {record.env_id}: expected episode "
                f"{self._episode_ids[record.env_id]}, got {record.episode_id}"
            )
        action_changed = (
            record.policy_action_indices != record.effective_action_indices
        )
        if bool(record.shield_engaged) != action_changed:
            raise RuntimeError(
                "shield_engaged does not match the policy/effective action delta"
            )
        if self._is_baseline and action_changed:
            raise RuntimeError("baseline changed the policy action")
        numeric = (
            record.pre_delay_v_command_mps,
            record.pre_delay_w_command_rad_s,
            record.post_delay_v_command_mps,
            record.post_delay_w_command_rad_s,
            record.body_forward_speed_mps,
            record.body_planar_speed_mps,
            record.obstacle_center_distance_m,
            record.relative_closing_speed_mps,
            record.risk,
            *record.dynamic_obstacle_center_distances_m,
            *record.dynamic_obstacle_relative_closing_speeds_mps,
            *record.dynamic_obstacle_risks,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("D3 record contains a non-finite numeric value")
        slot_lengths = {
            len(record.dynamic_obstacle_center_distances_m),
            len(record.dynamic_obstacle_relative_closing_speeds_mps),
            len(record.dynamic_obstacle_risks),
            len(record.dynamic_obstacle_risk_active),
        }
        if slot_lengths != {self.num_dynamic_obstacles}:
            raise ValueError(
                "D3 per-obstacle arrays must all match "
                f"num_dynamic_obstacles={self.num_dynamic_obstacles}, got "
                f"lengths={sorted(slot_lengths)}"
            )

    def _check_delay_alignment(self, record: StepRecord) -> None:
        previous = self._previous[record.env_id]
        if previous is None or previous.episode_id != record.episode_id:
            return
        errors = (
            abs(
                record.post_delay_v_command_mps
                - previous.pre_delay_v_command_mps
            ),
            abs(
                record.post_delay_w_command_rad_s
                - previous.pre_delay_w_command_rad_s
            ),
        )
        self.delay_alignment_samples += 1
        maximum = max(errors)
        self.delay_alignment_max_abs_error = max(
            self.delay_alignment_max_abs_error, maximum
        )
        if maximum > DELAY_TOLERANCE:
            self.delay_alignment_errors += 1

    def _emit(
        self,
        event_type: str,
        env_id: int,
        *,
        obstacle_slot: int,
        closest_distance_m: float,
    ) -> None:
        window = self._buffers[env_id].window()
        if not window:
            raise RuntimeError("cannot emit an empty D3 event window")
        if len({row.episode_id for row in window}) != 1:
            raise RuntimeError("D3 event window crossed an episode boundary")
        self.events.append(
            {
                "event_type": event_type,
                "env_id": env_id,
                "episode_id": window[-1].episode_id,
                "terminal_cause": window[-1].termination_cause,
                "episode_outcome_cause": None,
                "tracked_obstacle_slot": int(obstacle_slot),
                "closest_distance_m": float(closest_distance_m),
                "first_events": first_events(
                    window, obstacle_slot=obstacle_slot
                ),
                "records": [row.as_dict() for row in window],
            }
        )
        self._episode_event_indices[env_id].append(len(self.events) - 1)

    def _reset_encounters(self, env_id: int) -> None:
        for slot in range(self.num_dynamic_obstacles):
            self._encounter_phase[env_id][slot] = "idle"
            self._encounter_min_distance[env_id][slot] = float("inf")

    def _record_noncollision_encounters(self, record: StepRecord) -> None:
        env_id = record.env_id
        previous = self._previous[env_id]
        for slot in range(self.num_dynamic_obstacles):
            distance = record.dynamic_obstacle_center_distances_m[slot]
            closing = record.dynamic_obstacle_relative_closing_speeds_mps[slot]
            phase = self._encounter_phase[env_id][slot]
            if phase == "idle":
                if distance <= NEAR_DISTANCE_M and closing > 0.0:
                    self._encounter_phase[env_id][slot] = "approaching"
                    self._encounter_min_distance[env_id][slot] = distance
                continue
            if phase == "approaching":
                self._encounter_min_distance[env_id][slot] = min(
                    self._encounter_min_distance[env_id][slot], distance
                )
                previous_closing = (
                    previous.dynamic_obstacle_relative_closing_speeds_mps[slot]
                    if previous is not None
                    else closing
                )
                if previous_closing > 0.0 and closing <= 0.0:
                    self._emit(
                        "noncollision_closest_approach",
                        env_id,
                        obstacle_slot=slot,
                        closest_distance_m=self._encounter_min_distance[
                            env_id
                        ][slot],
                    )
                    self._encounter_phase[env_id][slot] = "resolved"
                continue
            if phase == "resolved" and distance >= NEAR_EXIT_DISTANCE_M:
                self._encounter_phase[env_id][slot] = "idle"
                self._encounter_min_distance[env_id][slot] = float("inf")

    def record_batch(self, records: Iterable[StepRecord]) -> None:
        """Record exactly one transition per environment."""
        rows = list(records)
        if len(rows) != self.num_envs:
            raise ValueError(
                f"expected {self.num_envs} records, received {len(rows)}"
            )
        if {row.env_id for row in rows} != set(range(self.num_envs)):
            raise ValueError("record batch must contain every env exactly once")

        for record in sorted(rows, key=lambda row: row.env_id):
            self._check_record(record)
            self._check_delay_alignment(record)
            env_id = record.env_id
            self.total_records += 1
            self.shield_engaged_records += int(record.shield_engaged)
            self.dynamic_collision_records += int(record.dynamic_collision)
            self.done_records += int(record.done)
            if record.done:
                cause = int(record.termination_cause)
                self.termination_cause_counts[cause] = (
                    self.termination_cause_counts.get(cause, 0) + 1
                )
            self._buffers[env_id].append(record)

            if record.dynamic_collision:
                slot = min(
                    range(self.num_dynamic_obstacles),
                    key=lambda index: (
                        record.dynamic_obstacle_center_distances_m[index]
                    ),
                )
                self._emit(
                    "dynamic_collision",
                    env_id,
                    obstacle_slot=slot,
                    closest_distance_m=(
                        record.dynamic_obstacle_center_distances_m[slot]
                    ),
                )
            elif not record.done:
                self._record_noncollision_encounters(record)

            if record.done:
                for event_index in self._episode_event_indices[env_id]:
                    self.events[event_index]["episode_outcome_cause"] = int(
                        record.termination_cause
                    )
                self._episode_event_indices[env_id].clear()
                self._buffers[env_id].clear()
                self._reset_encounters(env_id)
                self._previous[env_id] = None
                self._episode_ids[env_id] += 1
            else:
                self._previous[env_id] = record

    def report(self, *, metadata: dict) -> dict:
        event_counts = {
            label: sum(
                1 for event in self.events if event["event_type"] == label
            )
            for label in (
                "dynamic_collision",
                "noncollision_closest_approach",
            )
        }
        event_counts["successful_noncollision_closest_approach"] = sum(
            1
            for event in self.events
            if event["event_type"] == "noncollision_closest_approach"
            and event["episode_outcome_cause"] == 1
        )
        event_counts["unresolved_noncollision_closest_approach"] = sum(
            1
            for event in self.events
            if event["event_type"] == "noncollision_closest_approach"
            and event["episode_outcome_cause"] is None
        )
        collision_reconciles = (
            event_counts["dynamic_collision"]
            == self.dynamic_collision_records
        )
        delay_ok = (
            self.delay_alignment_samples > 0
            and self.delay_alignment_errors == 0
        )
        baseline_identity_ok = (
            self.shield_engaged_records == 0
            if self._is_baseline
            else None
        )
        intervention_observed = (
            self.shield_engaged_records > 0
            if not self._is_baseline
            else None
        )
        action_contract_ok = (
            bool(baseline_identity_ok)
            if self._is_baseline
            else bool(intervention_observed)
        )
        expected_records = metadata.get("expected_records")
        record_coverage_ok = (
            expected_records is None
            or int(expected_records) == self.total_records
        )
        expected_episodes = metadata.get("completed_episodes")
        episode_reconciliation_ok = (
            expected_episodes is None
            or int(expected_episodes) == self.done_records
        )
        return {
            "schema": "sa4_d3_yield_timing/v2",
            "mode": self.mode_value,
            "dt_s": DT_S,
            "max_lead_steps": MAX_LEAD_STEPS,
            "max_lead_s": MAX_LEAD_STEPS * DT_S,
            "window_capacity": WINDOW_CAPACITY,
            "definitions": {
                "near_distance_m": NEAR_DISTANCE_M,
                "near_exit_distance_m": NEAR_EXIT_DISTANCE_M,
                "noncollision_control_event": (
                    "per-obstacle first transition from positive radial "
                    "closing speed to non-positive while inside a tracked "
                    "3m encounter; rearmed only after that obstacle exits 3.25m"
                ),
                "risk_warning": RISK_WARNING,
                "decel_delta_mps": DECEL_DELTA_MPS,
                "stop_speed_mps": STOP_SPEED_MPS,
                "distance_semantics": "robot-to-obstacle center distance",
                "closing_semantics": (
                    "positive radial closing speed from relative robot/obstacle velocity"
                ),
            },
            "metadata": metadata,
            "counts": {
                "records": self.total_records,
                "completed_episodes": self.done_records,
                "events": len(self.events),
                "termination_causes": {
                    str(key): value
                    for key, value in sorted(
                        self.termination_cause_counts.items()
                    )
                },
                **event_counts,
            },
            "self_check": {
                "baseline_action_identity_ok": baseline_identity_ok,
                "intervention_action_observed": intervention_observed,
                "action_contract_ok": action_contract_ok,
                "shield_engaged_records": self.shield_engaged_records,
                "record_coverage_ok": record_coverage_ok,
                "episode_reconciliation_ok": episode_reconciliation_ok,
                "delay_alignment_samples": self.delay_alignment_samples,
                "delay_alignment_errors": self.delay_alignment_errors,
                "delay_alignment_max_abs_error": (
                    self.delay_alignment_max_abs_error
                ),
                "delay_alignment_ok": delay_ok,
                "dynamic_collision_reconciliation_ok": collision_reconciles,
                "reconciliation_ok": (
                    action_contract_ok
                    and record_coverage_ok
                    and episode_reconciliation_ok
                    and delay_ok
                    and collision_reconciles
                ),
            },
            "events": self.events,
        }

    def write(self, path: str | Path, *, metadata: dict) -> dict:
        report = self.report(metadata=metadata)
        output = Path(path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report
