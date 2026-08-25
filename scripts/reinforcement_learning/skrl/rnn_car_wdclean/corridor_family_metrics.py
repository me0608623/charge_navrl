"""Training-time accounting for deployment-corridor motion families.

The environment owns the family label. Metrics must snapshot that label before
``env.step()`` because Isaac Lab resets completed environments inside the step.
This module is monitoring-only: it does not alter rewards, actions, sampling, or
optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch

from .corridor_eval_metrics import (
    EXTREME_TURN_RAD_S,
    HIGH_TURN_RAD_S,
    LOW_SPEED_MPS,
    REVERSE_SPEED_MPS,
    STOP_SPEED_MPS,
)


MOTION_LATERAL = 0
MOTION_LONGITUDINAL = 1
MOTION_RANDOM_2D = 2

CORRIDOR_FAMILY_NAMES: tuple[str, ...] = (
    "lateral",
    "longitudinal",
    "random_2d",
    "mixed",
    "no_dynamic",
    "unready",
)
_FAMILY_ID = {
    name: index for index, name in enumerate(CORRIDOR_FAMILY_NAMES)
}
NON_CORRIDOR = -1

_COUNT_KEYS = (
    "episodes",
    "goal",
    "collision",
    "timeout",
    "other",
    "wall_collision",
    "obstacle_collision",
    "static_obstacle_collision",
    "dynamic_obstacle_collision",
    "wall_obstacle_overlap",
    "obstacle_subtype_overlap",
    "obstacle_subtype_unattributed",
)


@dataclass(frozen=True)
class CorridorFamilySnapshot:
    """Per-env family identity for the transition about to be executed."""

    family_id: torch.Tensor
    is_reset: torch.Tensor
    profile_id: torch.Tensor | None = None


def _vector(
    value,
    *,
    name: str,
    num_envs: int,
    device,
    dtype=None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.shape != (num_envs,):
        raise RuntimeError(
            f"{name} must have shape ({num_envs},), got {tuple(tensor.shape)}"
        )
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def snapshot_corridor_families(
    env_unwrapped,
    num_envs: int,
    device,
) -> CorridorFamilySnapshot:
    """Read the authoritative per-env corridor family before ``env.step``.

    Pure rows map to lateral, longitudinal, or random_2d. Rows containing more
    than one active family remain explicit as ``mixed``. A corridor with zero
    dynamic obstacles is ``no_dynamic``; an active corridor whose deferred
    obstacle installation is not ready is ``unready``.
    """

    family_id = torch.full(
        (num_envs,),
        NON_CORRIDOR,
        dtype=torch.long,
        device=device,
    )
    profile_id = torch.full_like(family_id, NON_CORRIDOR)
    active_raw = getattr(env_unwrapped, "_long_corridor_active", None)
    if active_raw is None:
        return CorridorFamilySnapshot(
            family_id=family_id,
            is_reset=torch.zeros(num_envs, dtype=torch.bool, device=device),
            profile_id=profile_id,
        )
    active = _vector(
        active_raw,
        name="_long_corridor_active",
        num_envs=num_envs,
        device=device,
        dtype=torch.bool,
    )
    if not bool(active.any()):
        reset_raw = getattr(env_unwrapped, "episode_length_buf", None)
        is_reset = (
            _vector(
                reset_raw,
                name="episode_length_buf",
                num_envs=num_envs,
                device=device,
            )
            == 0
            if reset_raw is not None
            else torch.zeros(num_envs, dtype=torch.bool, device=device)
        )
        return CorridorFamilySnapshot(
            family_id=family_id,
            is_reset=is_reset,
            profile_id=profile_id,
        )

    ready_raw = getattr(
        env_unwrapped, "_long_corridor_obstacles_ready", None
    )
    motion_raw = getattr(
        env_unwrapped, "_long_corridor_dynamic_motion_type", None
    )
    reset_raw = getattr(env_unwrapped, "episode_length_buf", None)
    profile_raw = getattr(
        env_unwrapped, "_long_corridor_speed_density_profile", None
    )
    if ready_raw is None or motion_raw is None or reset_raw is None:
        missing = [
            name
            for name, value in (
                ("_long_corridor_obstacles_ready", ready_raw),
                ("_long_corridor_dynamic_motion_type", motion_raw),
                ("episode_length_buf", reset_raw),
            )
            if value is None
        ]
        raise RuntimeError(
            "active corridor is missing family accounting state: "
            + ", ".join(missing)
        )

    ready = _vector(
        ready_raw,
        name="_long_corridor_obstacles_ready",
        num_envs=num_envs,
        device=device,
        dtype=torch.bool,
    )
    if profile_raw is not None:
        profile_id = _vector(
            profile_raw,
            name="_long_corridor_speed_density_profile",
            num_envs=num_envs,
            device=device,
            dtype=torch.long,
        ).clone()
        profile_id[~active] = NON_CORRIDOR
    motion = torch.as_tensor(motion_raw, device=device).long()
    if motion.ndim != 2 or motion.shape[0] != num_envs:
        raise RuntimeError(
            "_long_corridor_dynamic_motion_type must have shape "
            f"({num_envs}, slots), got {tuple(motion.shape)}"
        )
    unknown = active[:, None] & ((motion < -1) | (motion > MOTION_RANDOM_2D))
    if bool(unknown.any()):
        values = sorted(set(motion[unknown].detach().cpu().tolist()))
        raise RuntimeError(f"unknown corridor motion family ids: {values}")

    is_reset = (
        _vector(
            reset_raw,
            name="episode_length_buf",
            num_envs=num_envs,
            device=device,
        )
        == 0
    )
    unready = active & ~ready
    family_id[unready] = _FAMILY_ID["unready"]

    eligible = active & ready
    slot_active = motion >= 0
    dynamic_count = slot_active.sum(dim=1)
    no_dynamic = eligible & (dynamic_count == 0)
    family_id[no_dynamic] = _FAMILY_ID["no_dynamic"]

    if motion.shape[1] > 0:
        sentinel_hi = torch.full_like(motion, MOTION_RANDOM_2D + 1)
        sentinel_lo = torch.full_like(motion, -1)
        row_min = torch.where(
            slot_active, motion, sentinel_hi
        ).min(dim=1).values
        row_max = torch.where(
            slot_active, motion, sentinel_lo
        ).max(dim=1).values
        pure = eligible & (dynamic_count > 0) & (row_min == row_max)
        family_id[pure] = row_min[pure]
        mixed = eligible & (dynamic_count > 0) & ~pure
        family_id[mixed] = _FAMILY_ID["mixed"]

    if bool((active & (family_id < 0)).any()):
        raise RuntimeError("active corridor envs were not fully classified")
    return CorridorFamilySnapshot(
        family_id=family_id,
        is_reset=is_reset,
        profile_id=profile_id,
    )


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_")


class CorridorFamilyMetrics:
    """Device-side counters for per-family training diagnostics."""

    def __init__(self, num_envs: int, device):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.num_families = len(CORRIDOR_FAMILY_NAMES)
        self._episode_family_id = torch.full(
            (self.num_envs,),
            NON_CORRIDOR,
            dtype=torch.long,
            device=self.device,
        )
        self.reset_iteration()

    def reset_iteration(self) -> None:
        """Clear iteration counters while preserving open episode identity."""

        shape = (self.num_families,)
        self._active_steps = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self._reset_count = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self._counts = {
            key: torch.zeros(shape, dtype=torch.long, device=self.device)
            for key in _COUNT_KEYS
        }
        self._action_frames = torch.zeros(
            shape, dtype=torch.long, device=self.device
        )
        self._action_sums = {
            key: torch.zeros(shape, dtype=torch.float64, device=self.device)
            for key in (
                "linear_speed",
                "linear_speed_abs",
                "angular_abs",
                "high_turn",
                "extreme_turn",
                "stop",
                "reverse",
                "low_speed_high_turn",
            )
        }
        self._signal_sums: dict[str, torch.Tensor] = {}
        self._signal_counts: dict[str, torch.Tensor] = {}
        self._signal_nonfinite_counts: dict[str, torch.Tensor] = {}
        self._env_steps_total = 0
        self._late_attach_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._unready_upgrade_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._nonfinite_action_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._bad_subtype_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._bad_outcome_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )

    def _ids(self, mask: torch.Tensor, family_id: torch.Tensor) -> torch.Tensor:
        return family_id[mask].long()

    def _bincount(
        self,
        mask: torch.Tensor,
        family_id: torch.Tensor,
    ) -> torch.Tensor:
        return torch.bincount(
            self._ids(mask, family_id),
            minlength=self.num_families,
        )

    def _add_signal(
        self,
        key: str,
        values: torch.Tensor,
        valid: torch.Tensor,
        family_id: torch.Tensor,
    ) -> None:
        flat = values.detach().reshape(-1)
        if flat.shape != (self.num_envs,):
            return
        if flat.dtype == torch.bool:
            return
        selected_ids = self._ids(valid, family_id)
        selected = flat[valid].to(dtype=torch.float64)
        if selected.numel() == 0:
            return
        safe = _safe_key(key)
        if not safe:
            return
        finite = torch.isfinite(selected)
        bucket = self._signal_sums.setdefault(
            safe,
            torch.zeros(
                self.num_families,
                dtype=torch.float64,
                device=self.device,
            ),
        )
        counts = self._signal_counts.setdefault(
            safe,
            torch.zeros(
                self.num_families,
                dtype=torch.long,
                device=self.device,
            ),
        )
        nonfinite_counts = self._signal_nonfinite_counts.setdefault(
            safe,
            torch.zeros(
                self.num_families,
                dtype=torch.long,
                device=self.device,
            ),
        )
        bucket.index_add_(0, selected_ids[finite], selected[finite])
        counts += torch.bincount(
            selected_ids[finite], minlength=self.num_families
        )
        nonfinite_counts += torch.bincount(
            selected_ids[~finite], minlength=self.num_families
        )

    def step(
        self,
        *,
        snapshot: CorridorFamilySnapshot,
        corridor_scene_mask: torch.Tensor,
        reward: torch.Tensor,
        reward_breakdown: dict[str, torch.Tensor] | None,
        charge_actions: dict[str, torch.Tensor] | None,
        done: torch.Tensor,
        terminated_flat: torch.Tensor,
        truncated_flat: torch.Tensor,
    ) -> None:
        """Accumulate one transition without changing the training path."""

        family_id = _vector(
            snapshot.family_id,
            name="corridor family_id",
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        is_reset = _vector(
            snapshot.is_reset,
            name="corridor is_reset",
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        corridor_scene_mask = _vector(
            corridor_scene_mask,
            name="corridor_scene_mask",
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        classified = family_id >= 0
        current = self._episode_family_id
        established = current >= 0
        unready_id = _FAMILY_ID["unready"]
        unready_upgrade = (
            established
            & classified
            & (current == unready_id)
            & (family_id != unready_id)
        )
        changed = (
            established
            & classified
            & (current != family_id)
            & ~unready_upgrade
        )
        disappeared = established & ~classified
        reset_mid_episode = established & is_reset
        bad_family_id = (family_id < NON_CORRIDOR) | (
            family_id >= self.num_families
        )
        mismatch = classified ^ corridor_scene_mask
        lifecycle_error = changed | disappeared | reset_mid_episode
        if bool((bad_family_id | mismatch | lifecycle_error).any()):
            if bool(bad_family_id.any()):
                raise RuntimeError("corridor family id is out of range")
            if bool(mismatch.any()):
                raise RuntimeError(
                    "corridor scene/family masks disagree for "
                    f"{int(mismatch.sum().item())} envs"
                )
            raise RuntimeError(
                "corridor family changed or disappeared before episode end"
            )
        self._episode_family_id[unready_upgrade] = family_id[
            unready_upgrade
        ]
        self._unready_upgrade_count += unready_upgrade.sum()

        attach = ~established & classified
        late_attach = attach & ~is_reset
        self._late_attach_count += late_attach.sum()
        self._episode_family_id[attach] = family_id[attach]
        assigned_reset = attach & is_reset
        self._reset_count += self._bincount(assigned_reset, family_id)

        self._env_steps_total += self.num_envs
        self._active_steps += self._bincount(classified, family_id)
        self._add_signal("total_reward", reward, classified, family_id)
        if reward_breakdown is None:
            raise RuntimeError(
                "corridor family metrics require reward_breakdown"
            )
        for key, value in reward_breakdown.items():
            if isinstance(value, torch.Tensor):
                self._add_signal(key, value, classified, family_id)

        if charge_actions is None:
            raise RuntimeError(
                "corridor family metrics require charge action diagnostics"
            )
        linear = _vector(
            charge_actions["v_x"],
            name="charge_actions.v_x",
            num_envs=self.num_envs,
            device=self.device,
        ).float()
        angular_abs = _vector(
            charge_actions["omega"],
            name="charge_actions.omega",
            num_envs=self.num_envs,
            device=self.device,
        ).float().abs()
        selected_linear = linear[classified]
        selected_angular_abs = angular_abs[classified]
        finite_actions = torch.isfinite(
            selected_linear
        ) & torch.isfinite(selected_angular_abs)
        self._nonfinite_action_count += (~finite_actions).sum()
        selected_linear = torch.nan_to_num(
            selected_linear, nan=0.0, posinf=0.0, neginf=0.0
        )
        selected_angular_abs = torch.nan_to_num(
            selected_angular_abs, nan=0.0, posinf=0.0, neginf=0.0
        )
        ids = self._ids(classified, family_id)
        self._action_frames += self._bincount(classified, family_id)
        action_values = {
            "linear_speed": selected_linear,
            "linear_speed_abs": selected_linear.abs(),
            "angular_abs": selected_angular_abs,
            "high_turn": (
                selected_angular_abs > HIGH_TURN_RAD_S
            ).float(),
            "extreme_turn": (
                selected_angular_abs > EXTREME_TURN_RAD_S
            ).float(),
            "stop": (
                selected_linear.abs() < STOP_SPEED_MPS
            ).float(),
            "reverse": (
                selected_linear < REVERSE_SPEED_MPS
            ).float(),
            "low_speed_high_turn": (
                (selected_linear.abs() < LOW_SPEED_MPS)
                & (selected_angular_abs > HIGH_TURN_RAD_S)
            ).float(),
        }
        for key, values in action_values.items():
            self._action_sums[key].index_add_(
                0, ids, values.to(dtype=torch.float64)
            )

        done = _vector(
            done,
            name="done",
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        terminated_flat = _vector(
            terminated_flat,
            name="terminated_flat",
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        truncated_flat = _vector(
            truncated_flat,
            name="truncated_flat",
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.bool,
        )
        corridor_done = done & classified

        def event(key: str) -> torch.Tensor:
            value = reward_breakdown.get(key)
            if value is None:
                return torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            return _vector(
                value,
                name=f"reward_breakdown.{key}",
                num_envs=self.num_envs,
                device=self.device,
                dtype=torch.bool,
            )

        goal = event("goal_reached")
        wall = event("wall_collision")
        obstacle = event("obs_collision")
        static_obstacle = event("static_obs_collision")
        dynamic_obstacle = event("dynamic_obs_collision")
        bad_subtype = corridor_done & (
            (static_obstacle | dynamic_obstacle) & ~obstacle
        )
        self._bad_subtype_count += bad_subtype.sum()
        collided = wall | obstacle
        other = event("other_death")
        timed_out = truncated_flat & ~terminated_flat
        hits = (
            goal.to(torch.int8)
            + collided.to(torch.int8)
            + other.to(torch.int8)
            + timed_out.to(torch.int8)
        )
        self._bad_outcome_count += (corridor_done & (hits != 1)).sum()

        self._counts["episodes"] += self._bincount(
            corridor_done, family_id
        )
        for key, mask in (
            ("goal", goal),
            ("collision", collided),
            ("timeout", timed_out),
            ("other", other),
            ("wall_collision", wall),
            ("obstacle_collision", obstacle),
            ("static_obstacle_collision", static_obstacle),
            ("dynamic_obstacle_collision", dynamic_obstacle),
            ("wall_obstacle_overlap", wall & obstacle),
            (
                "obstacle_subtype_overlap",
                static_obstacle & dynamic_obstacle,
            ),
            (
                "obstacle_subtype_unattributed",
                obstacle & ~(static_obstacle | dynamic_obstacle),
            ),
        ):
            selected = corridor_done & mask
            self._counts[key] += self._bincount(selected, family_id)

        self._episode_family_id[done] = NON_CORRIDOR

    def collect(
        self,
        *,
        expected_corridor_episodes: int,
    ) -> dict[str, float]:
        """Emit metrics and enforce aggregate/family reconciliation."""

        expected = int(expected_corridor_episodes)
        nonfinite_actions = int(self._nonfinite_action_count.item())
        if nonfinite_actions:
            raise RuntimeError(
                "corridor family action diagnostics contain "
                f"{nonfinite_actions} non-finite frames"
            )
        bad_subtypes = int(self._bad_subtype_count.item())
        if bad_subtypes:
            raise RuntimeError(
                "corridor obstacle subtype occurred without obs_collision "
                f"for {bad_subtypes} envs"
            )
        bad_outcomes = int(self._bad_outcome_count.item())
        if bad_outcomes:
            raise RuntimeError(
                "corridor family outcomes require exactly one class; "
                f"{bad_outcomes} envs violated the contract"
            )
        total_reward_nonfinite = int(
            self._signal_nonfinite_counts.get(
                "total_reward",
                torch.zeros((), dtype=torch.long, device=self.device),
            ).sum().item()
        )
        if total_reward_nonfinite:
            raise RuntimeError(
                "corridor family total_reward contains "
                f"{total_reward_nonfinite} non-finite values"
            )
        episodes_total = int(self._counts["episodes"].sum().item())
        if episodes_total != expected:
            raise RuntimeError(
                "corridor family episodes do not reconcile with scene/corridor: "
                f"{episodes_total} != {expected}"
            )
        classified_outcomes = sum(
            int(self._counts[key].sum().item())
            for key in ("goal", "collision", "timeout", "other")
        )
        if classified_outcomes != episodes_total:
            raise RuntimeError(
                "corridor family outcomes do not reconcile with episodes: "
                f"{classified_outcomes} != {episodes_total}"
            )
        for index, name in enumerate(CORRIDOR_FAMILY_NAMES):
            collisions = int(self._counts["collision"][index].item())
            wall = int(self._counts["wall_collision"][index].item())
            obstacle = int(
                self._counts["obstacle_collision"][index].item()
            )
            wall_obstacle_overlap = int(
                self._counts["wall_obstacle_overlap"][index].item()
            )
            if collisions != wall + obstacle - wall_obstacle_overlap:
                raise RuntimeError(
                    f"corridor family {name} collision breakdown does not "
                    "reconcile"
                )
            static = int(
                self._counts["static_obstacle_collision"][index].item()
            )
            dynamic = int(
                self._counts["dynamic_obstacle_collision"][index].item()
            )
            subtype_overlap = int(
                self._counts["obstacle_subtype_overlap"][index].item()
            )
            unattributed = int(
                self._counts[
                    "obstacle_subtype_unattributed"
                ][index].item()
            )
            if obstacle != static + dynamic - subtype_overlap + unattributed:
                raise RuntimeError(
                    f"corridor family {name} obstacle subtype breakdown "
                    "does not reconcile"
                )

        metrics: dict[str, float] = {}
        active_total = int(self._active_steps.sum().item())
        reset_total = int(self._reset_count.sum().item())
        metrics["corridor_family/accounting/env_steps_total"] = float(
            self._env_steps_total
        )
        metrics["corridor_family/accounting/active_steps_total"] = float(
            active_total
        )
        metrics["corridor_family/accounting/resets_total"] = float(reset_total)
        metrics["corridor_family/accounting/episodes_total"] = float(
            episodes_total
        )
        metrics[
            "corridor_family/accounting/expected_corridor_episodes"
        ] = float(expected)
        metrics["corridor_family/accounting/late_attach_count"] = float(
            self._late_attach_count.item()
        )
        metrics["corridor_family/accounting/unready_upgrade_count"] = float(
            self._unready_upgrade_count.item()
        )
        metrics[
            "corridor_family/accounting/nonfinite_action_count"
        ] = float(nonfinite_actions)
        metrics["corridor_family/accounting/open_episode_count"] = float(
            (self._episode_family_id >= 0).sum().item()
        )
        metrics["corridor_family/accounting/active_fraction_all"] = (
            active_total / self._env_steps_total
            if self._env_steps_total
            else 0.0
        )
        metrics["corridor_family/accounting/active_step_share_sum"] = (
            1.0 if active_total else 0.0
        )
        metrics["corridor_family/accounting/reset_share_sum"] = (
            1.0 if reset_total else 0.0
        )
        metrics[
            "corridor_family/accounting/completed_episode_share_sum"
        ] = 1.0 if episodes_total else 0.0
        metrics["corridor_family/accounting/reconciliation_ok"] = 1.0

        for index, name in enumerate(CORRIDOR_FAMILY_NAMES):
            prefix = f"corridor_family/{name}"
            active_steps = int(self._active_steps[index].item())
            resets = int(self._reset_count[index].item())
            episodes = int(self._counts["episodes"][index].item())
            if active_steps == 0 and resets == 0 and episodes == 0:
                continue

            metrics[f"{prefix}/active_steps"] = float(active_steps)
            metrics[f"{prefix}/active_step_share"] = (
                active_steps / active_total if active_total else 0.0
            )
            metrics[f"{prefix}/active_step_fraction_all"] = (
                active_steps / self._env_steps_total
                if self._env_steps_total
                else 0.0
            )
            metrics[f"{prefix}/reset_count"] = float(resets)
            metrics[f"{prefix}/reset_share"] = (
                resets / reset_total if reset_total else 0.0
            )
            metrics[f"{prefix}/episodes"] = float(episodes)
            metrics[f"{prefix}/completed_episode_share"] = (
                episodes / episodes_total if episodes_total else 0.0
            )
            if episodes:
                for key, output in (
                    ("goal", "sr"),
                    ("collision", "cr"),
                    ("timeout", "timeout"),
                    ("other", "other_death"),
                    ("wall_collision", "wall_cr"),
                    ("obstacle_collision", "obstacle_cr"),
                    (
                        "static_obstacle_collision",
                        "static_obstacle_cr",
                    ),
                    (
                        "dynamic_obstacle_collision",
                        "dynamic_obstacle_cr",
                    ),
                    (
                        "wall_obstacle_overlap",
                        "wall_obstacle_overlap_cr",
                    ),
                    (
                        "obstacle_subtype_overlap",
                        "obstacle_subtype_overlap_cr",
                    ),
                    (
                        "obstacle_subtype_unattributed",
                        "obstacle_subtype_unattributed_cr",
                    ),
                ):
                    metrics[f"{prefix}/{output}"] = (
                        int(self._counts[key][index].item()) / episodes
                    )

            action_frames = int(self._action_frames[index].item())
            if action_frames:
                metrics[f"{prefix}/action_frames"] = float(action_frames)
                metrics[f"{prefix}/action_coverage"] = (
                    action_frames / active_steps if active_steps else 0.0
                )
                for key, output in (
                    ("linear_speed", "linear_speed_mean_mps"),
                    ("linear_speed_abs", "linear_speed_abs_mean_mps"),
                    ("angular_abs", "angular_abs_mean_rad_s"),
                    ("high_turn", "high_turn_fraction"),
                    ("extreme_turn", "extreme_turn_fraction"),
                    ("stop", "stop_command_fraction"),
                    ("reverse", "reverse_command_fraction"),
                    (
                        "low_speed_high_turn",
                        "low_speed_high_turn_fraction",
                    ),
                ):
                    metrics[f"{prefix}/{output}"] = (
                        float(self._action_sums[key][index].item())
                        / action_frames
                    )

            if active_steps:
                for key, sums in self._signal_sums.items():
                    finite_count = int(
                        self._signal_counts[key][index].item()
                    )
                    nonfinite_count = int(
                        self._signal_nonfinite_counts[key][index].item()
                    )
                    observed = finite_count + nonfinite_count
                    if observed == 0:
                        continue
                    signal_prefix = f"{prefix}/signal/{key}"
                    metrics[f"{signal_prefix}_coverage"] = (
                        finite_count / active_steps
                    )
                    metrics[f"{signal_prefix}_nonfinite_fraction"] = (
                        nonfinite_count / observed
                    )
                    if finite_count:
                        metrics[f"{signal_prefix}_mean_per_step"] = (
                            float(sums[index].item()) / finite_count
                        )

        return metrics
