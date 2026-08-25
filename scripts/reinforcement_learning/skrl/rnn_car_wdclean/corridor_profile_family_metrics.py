"""Training-only accounting by corridor density/speed profile and motion family.

The selected profile and family are snapshotted before ``env.step()`` and held
as episode identity. This module only reads diagnostics and increments counters;
it cannot alter actions, rewards, observations, sampling, or optimizer state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .corridor_family_metrics import (
    CORRIDOR_FAMILY_NAMES,
    NON_CORRIDOR,
    CorridorFamilySnapshot,
    _vector,
)


_OUTCOMES = ("episodes", "goal", "collision", "timeout", "other")


@dataclass(frozen=True)
class CorridorProfileSpec:
    """Stable metric label for one configured density/speed profile."""

    profile_id: int
    static_obstacles: int
    dynamic_obstacles: int
    speed_min: float
    speed_max: float
    weight: float
    label: str


def _speed_band_label(speed_min: float, speed_max: float) -> str:
    midpoint = 0.5 * (float(speed_min) + float(speed_max))
    return f"p{round(midpoint * 100):03d}"


def normalize_profile_specs(
    speed_density_mix: Iterable[tuple] | None,
) -> tuple[CorridorProfileSpec, ...]:
    """Validate config rows and assign deterministic human-readable labels."""

    if speed_density_mix is None:
        return ()
    specs: list[CorridorProfileSpec] = []
    labels: set[str] = set()
    for profile_id, row in enumerate(tuple(speed_density_mix)):
        if len(row) != 3:
            raise ValueError(
                "corridor speed-density profile must be "
                "((static, dynamic), (speed_min, speed_max), weight)"
            )
        counts, speed_range, weight = row
        if len(counts) != 2 or len(speed_range) != 2:
            raise ValueError("invalid corridor speed-density profile shape")
        static_obstacles, dynamic_obstacles = (int(v) for v in counts)
        speed_min, speed_max = (float(v) for v in speed_range)
        weight = float(weight)
        if (
            static_obstacles < 0
            or dynamic_obstacles < 0
            or not speed_min < speed_max
            or not weight > 0.0
        ):
            raise ValueError("invalid corridor speed-density profile values")
        label = (
            f"{_speed_band_label(speed_min, speed_max)}_"
            f"{static_obstacles}s{dynamic_obstacles}d"
        )
        if label in labels:
            raise ValueError(f"duplicate corridor profile metric label: {label}")
        labels.add(label)
        specs.append(
            CorridorProfileSpec(
                profile_id=profile_id,
                static_obstacles=static_obstacles,
                dynamic_obstacles=dynamic_obstacles,
                speed_min=speed_min,
                speed_max=speed_max,
                weight=weight,
                label=label,
            )
        )
    if specs and abs(sum(spec.weight for spec in specs) - 1.0) > 1e-8:
        raise ValueError("corridor speed-density profile weights must sum to 1")
    return tuple(specs)


class CorridorProfileFamilyMetrics:
    """Exact counters over configured profile x observed motion-family buckets."""

    def __init__(self, num_envs: int, device, speed_density_mix: Iterable[tuple]):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.specs = normalize_profile_specs(speed_density_mix)
        if not self.specs:
            raise ValueError("profile-family metrics require a non-empty profile mix")
        self.num_profiles = len(self.specs)
        self.num_families = len(CORRIDOR_FAMILY_NAMES)
        self._episode_profile_id = torch.full(
            (self.num_envs,), NON_CORRIDOR, dtype=torch.long, device=self.device
        )
        self._episode_family_id = torch.full_like(
            self._episode_profile_id, NON_CORRIDOR
        )
        self.reset_iteration()

    def reset_iteration(self) -> None:
        """Clear rollout counters while preserving episodes crossing rollouts."""

        shape = (self.num_profiles, self.num_families)
        self._active_steps = torch.zeros(shape, dtype=torch.long, device=self.device)
        self._reset_count = torch.zeros_like(self._active_steps)
        self._counts = {
            key: torch.zeros_like(self._active_steps) for key in _OUTCOMES
        }
        self._env_steps_total = 0
        self._late_attach_count = torch.zeros((), dtype=torch.long, device=self.device)
        self._unready_upgrade_count = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._bad_outcome_count = torch.zeros((), dtype=torch.long, device=self.device)

    def _flat_bucket(
        self, profile_id: torch.Tensor, family_id: torch.Tensor
    ) -> torch.Tensor:
        return profile_id.long() * self.num_families + family_id.long()

    def _bincount(
        self,
        mask: torch.Tensor,
        profile_id: torch.Tensor,
        family_id: torch.Tensor,
    ) -> torch.Tensor:
        ids = self._flat_bucket(profile_id[mask], family_id[mask])
        return torch.bincount(
            ids, minlength=self.num_profiles * self.num_families
        ).reshape(self.num_profiles, self.num_families)

    def step(
        self,
        *,
        snapshot: CorridorFamilySnapshot,
        corridor_scene_mask: torch.Tensor,
        reward_breakdown: dict[str, torch.Tensor] | None,
        done: torch.Tensor,
        terminated_flat: torch.Tensor,
        truncated_flat: torch.Tensor,
    ) -> None:
        """Account one transition using only pre-step identity and post-step outcome."""

        if snapshot.profile_id is None:
            raise RuntimeError("corridor profile snapshot is missing profile_id")
        family_id = _vector(
            snapshot.family_id,
            name="corridor family_id",
            num_envs=self.num_envs,
            device=self.device,
            dtype=torch.long,
        )
        profile_id = _vector(
            snapshot.profile_id,
            name="corridor profile_id",
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
        unknown_profile = classified & (
            (profile_id < 0) | (profile_id >= self.num_profiles)
        )
        mismatch = classified ^ corridor_scene_mask
        if bool(unknown_profile.any()):
            values = sorted(set(profile_id[unknown_profile].detach().cpu().tolist()))
            raise RuntimeError(
                "active corridor has unknown speed-density profile ids: "
                f"{values}"
            )
        if bool(mismatch.any()):
            raise RuntimeError(
                "corridor scene/profile-family masks disagree for "
                f"{int(mismatch.sum().item())} envs"
            )

        established = self._episode_profile_id >= 0
        unready_id = CORRIDOR_FAMILY_NAMES.index("unready")
        unready_upgrade = (
            established
            & classified
            & (self._episode_family_id == unready_id)
            & (family_id != unready_id)
            & (self._episode_profile_id == profile_id)
        )
        changed = established & classified & (
            (self._episode_profile_id != profile_id)
            | (
                (self._episode_family_id != family_id)
                & ~unready_upgrade
            )
        )
        disappeared = established & ~classified
        reset_mid_episode = established & is_reset
        if bool((changed | disappeared | reset_mid_episode).any()):
            raise RuntimeError(
                "corridor profile or family changed/disappeared before episode end"
            )
        self._episode_family_id[unready_upgrade] = family_id[unready_upgrade]
        self._unready_upgrade_count += unready_upgrade.sum()

        attach = ~established & classified
        self._late_attach_count += (attach & ~is_reset).sum()
        self._episode_profile_id[attach] = profile_id[attach]
        self._episode_family_id[attach] = family_id[attach]
        assigned_reset = attach & is_reset
        self._reset_count += self._bincount(
            assigned_reset, profile_id, family_id
        )

        self._env_steps_total += self.num_envs
        self._active_steps += self._bincount(
            classified, profile_id, family_id
        )
        if reward_breakdown is None:
            raise RuntimeError("corridor profile-family metrics require reward_breakdown")

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
        collision = event("wall_collision") | event("obs_collision")
        other = event("other_death")
        timeout = truncated_flat & ~terminated_flat
        outcome_hits = (
            goal.to(torch.int8)
            + collision.to(torch.int8)
            + timeout.to(torch.int8)
            + other.to(torch.int8)
        )
        self._bad_outcome_count += (corridor_done & (outcome_hits != 1)).sum()
        self._counts["episodes"] += self._bincount(
            corridor_done, profile_id, family_id
        )
        for key, mask in (
            ("goal", goal),
            ("collision", collision),
            ("timeout", timeout),
            ("other", other),
        ):
            self._counts[key] += self._bincount(
                corridor_done & mask, profile_id, family_id
            )

        self._episode_profile_id[done] = NON_CORRIDOR
        self._episode_family_id[done] = NON_CORRIDOR

    def collect(
        self,
        *,
        expected_corridor_episodes: int,
        expected_active_steps: int,
        expected_resets: int,
    ) -> dict[str, float]:
        """Emit profile and profile-family metrics after strict reconciliation."""

        bad_outcomes = int(self._bad_outcome_count.item())
        if bad_outcomes:
            raise RuntimeError(
                "corridor profile-family outcomes require exactly one class; "
                f"{bad_outcomes} envs violated the contract"
            )
        active_total = int(self._active_steps.sum().item())
        reset_total = int(self._reset_count.sum().item())
        episode_total = int(self._counts["episodes"].sum().item())
        expected = (
            int(expected_active_steps),
            int(expected_resets),
            int(expected_corridor_episodes),
        )
        actual = (active_total, reset_total, episode_total)
        if actual != expected:
            raise RuntimeError(
                "corridor profile-family accounting does not reconcile with "
                f"corridor family totals: {actual} != {expected}"
            )
        outcome_total = sum(
            int(self._counts[key].sum().item())
            for key in ("goal", "collision", "timeout", "other")
        )
        if outcome_total != episode_total:
            raise RuntimeError(
                "corridor profile-family outcomes do not reconcile with episodes"
            )

        metrics: dict[str, float] = {
            "corridor_profile_family/accounting/env_steps_total": float(
                self._env_steps_total
            ),
            "corridor_profile_family/accounting/active_steps_total": float(
                active_total
            ),
            "corridor_profile_family/accounting/resets_total": float(reset_total),
            "corridor_profile_family/accounting/episodes_total": float(
                episode_total
            ),
            "corridor_profile_family/accounting/late_attach_count": float(
                self._late_attach_count.item()
            ),
            "corridor_profile_family/accounting/unready_upgrade_count": float(
                self._unready_upgrade_count.item()
            ),
            "corridor_profile_family/accounting/open_episode_count": float(
                (self._episode_profile_id >= 0).sum().item()
            ),
            "corridor_profile_family/accounting/active_step_share_sum": (
                1.0 if active_total else 0.0
            ),
            "corridor_profile_family/accounting/reset_share_sum": (
                1.0 if reset_total else 0.0
            ),
            "corridor_profile_family/accounting/completed_episode_share_sum": (
                1.0 if episode_total else 0.0
            ),
            "corridor_profile_family/accounting/reconciliation_ok": 1.0,
        }

        def emit(prefix: str, active: int, resets: int, episodes: int, index) -> None:
            metrics[f"{prefix}/active_steps"] = float(active)
            metrics[f"{prefix}/active_step_share"] = (
                active / active_total if active_total else 0.0
            )
            metrics[f"{prefix}/reset_count"] = float(resets)
            metrics[f"{prefix}/reset_share"] = (
                resets / reset_total if reset_total else 0.0
            )
            metrics[f"{prefix}/episodes"] = float(episodes)
            metrics[f"{prefix}/completed_episode_share"] = (
                episodes / episode_total if episode_total else 0.0
            )
            if episodes:
                for key, output in (
                    ("goal", "sr"),
                    ("collision", "cr"),
                    ("timeout", "timeout"),
                    ("other", "other_death"),
                ):
                    metrics[f"{prefix}/{output}"] = (
                        int(self._counts[key][index].sum().item()) / episodes
                    )

        for profile_index, spec in enumerate(self.specs):
            active = int(self._active_steps[profile_index].sum().item())
            resets = int(self._reset_count[profile_index].sum().item())
            episodes = int(self._counts["episodes"][profile_index].sum().item())
            profile_prefix = f"corridor_profile/{spec.label}"
            emit(profile_prefix, active, resets, episodes, profile_index)
            metrics[f"{profile_prefix}/configured_weight"] = spec.weight
            metrics[f"{profile_prefix}/static_obstacles"] = float(
                spec.static_obstacles
            )
            metrics[f"{profile_prefix}/dynamic_obstacles"] = float(
                spec.dynamic_obstacles
            )
            metrics[f"{profile_prefix}/speed_min_mps"] = spec.speed_min
            metrics[f"{profile_prefix}/speed_max_mps"] = spec.speed_max

            for family_index, family_name in enumerate(CORRIDOR_FAMILY_NAMES):
                active = int(self._active_steps[profile_index, family_index].item())
                resets = int(self._reset_count[profile_index, family_index].item())
                episodes = int(
                    self._counts["episodes"][profile_index, family_index].item()
                )
                if active == 0 and resets == 0 and episodes == 0:
                    continue
                emit(
                    f"corridor_profile_family/{spec.label}/{family_name}",
                    active,
                    resets,
                    episodes,
                    (profile_index, family_index),
                )
        return metrics
