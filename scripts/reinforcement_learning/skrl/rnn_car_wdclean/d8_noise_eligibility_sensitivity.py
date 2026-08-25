"""Paired shadow audit for the SA4 mixed-pixel eligibility rule.

The policy and simulator keep using the currently realized LiDAR observation.
For the same ray samples and random draws, this audit also evaluates a minimal
counterfactual in which a mixed-pixel replacement is eligible only when the
ray had a finite physical return and that return survived dropout.  The two
72-bin sweeps are passed through the same frozen D4 geometry model on the same
active frames.  No action is modified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import torch

from rnn_car_wdclean.d4_geometry_selector import (
    GeometrySelectorSpec,
    geometry_argmin_selector_protocol,
    geometry_feasible_action_grid,
)
from rnn_car_wdclean.d7_lidar_residual_origin import (
    ResidualOriginSpec,
    _validate_trace_contract,
)


MODE = "mixed_pixel_valid_return_shadow"
PROTOCOL_SCHEMA = "sa4_d8_noise_eligibility_sensitivity/v1"
REPORT_SCHEMA = "sa4_d8_noise_eligibility_report/v1"
COUNTERFACTUAL_SCHEMA = "valid_return_only_mixed_pixel/v1"


@dataclass(frozen=True)
class NoiseEligibilitySpec:
    """Frozen interpretation rule for the paired sensitivity audit."""

    major_rescue_fraction_of_current_no_feasible: float = 0.25
    material_net_recovery_fraction_of_active: float = 0.05
    material_net_recovery_min_frames: int = 100


def _validate_spec(spec: NoiseEligibilitySpec) -> None:
    if not 0.0 < spec.major_rescue_fraction_of_current_no_feasible <= 1.0:
        raise ValueError("major rescue fraction must be in (0,1]")
    if not 0.0 < spec.material_net_recovery_fraction_of_active <= 1.0:
        raise ValueError("material net recovery fraction must be in (0,1]")
    if spec.material_net_recovery_min_frames < 1:
        raise ValueError("material net recovery minimum frames must be positive")


def noise_eligibility_protocol(
    spec: NoiseEligibilitySpec = NoiseEligibilitySpec(),
    noise_spec: ResidualOriginSpec = ResidualOriginSpec(),
    geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
) -> dict:
    """Return the content-addressed D8 preregistration contract."""

    _validate_spec(spec)
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "mode": MODE,
        "scope": (
            "single-checkpoint, single-evaluator-seed baseline-only paired "
            "shadow; policy action and realized policy observation are unchanged"
        ),
        "current_noise": (
            "the realized full-noise sweep: every one of 5760 raw ray slots is "
            "eligible for the Bernoulli mixed-pixel replacement"
        ),
        "counterfactual_noise": (
            "reuse the exact realized masks and replacement values, but restore "
            "a distractor to its pre-distractor value unless raw_valid AND not "
            "hole; no random number is redrawn"
        ),
        "counterfactual_schema": COUNTERFACTUAL_SCHEMA,
        "comparison_scope": (
            "the exact frozen D4 two-frame trigger/release active frames; both "
            "sweeps use their realized winner-ray angles and the same dynamic state"
        ),
        "geometry_protocol_sha256": geometry_argmin_selector_protocol(
            geometry_spec
        )["sha256"],
        "descriptive_rules": {
            "major_fake_obstacle_contributor": (
                "corrected_only_feasible / current_no_feasible >= "
                f"{spec.major_rescue_fraction_of_current_no_feasible:.3f}, "
                "net recovery / active >= "
                f"{spec.material_net_recovery_fraction_of_active:.3f}, and "
                f"net recovery >= {spec.material_net_recovery_min_frames} frames"
            ),
            "inferential_claim": False,
        },
        "interpretation_limits": [
            "this is not a validated VLP-16 noise model; it is an eligibility sensitivity",
            "the U(0.2,2.0m) replacement distribution is deliberately unchanged",
            "the 0.2515% estimate is not reinterpreted as a calibrated per-ray probability",
            "the counterfactual is not fed to the policy, so no SR/CR causal effect is estimated",
            "feasibility is the frozen D4 model, not proof of physical avoidability",
            "single checkpoint and evaluator seed are diagnostic evidence only",
        ],
        "spec": asdict(spec),
        "noise_trace_spec": asdict(noise_spec),
        "geometry_spec": asdict(geometry_spec),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def _tensor(mapping: dict, key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"D8 field {key!r} must be a tensor")
    return value


class NoiseEligibilitySensitivity:
    """Track the frozen D4 active state and compare two LiDAR shadows."""

    mode = MODE

    def __init__(
        self,
        *,
        spec: NoiseEligibilitySpec = NoiseEligibilitySpec(),
        noise_spec: ResidualOriginSpec = ResidualOriginSpec(),
        geometry_spec: GeometrySelectorSpec = GeometrySelectorSpec(),
    ) -> None:
        _validate_spec(spec)
        self.spec = spec
        self.noise_spec = noise_spec
        self.geometry_spec = geometry_spec
        self._active: torch.Tensor | None = None
        self._confirm: torch.Tensor | None = None
        self._release: torch.Tensor | None = None
        self._tracked_slot: torch.Tensor | None = None
        self._calls = 0
        self._transitions = 0
        self._environment_frames = 0
        self._active_frames = 0
        self._trigger_activations = 0
        self._release_events = 0
        self._trace_matches = 0
        self._action_identity_checks = 0
        self._action_identity_errors = 0
        self._monotonicity_errors = 0
        self._dynamic_grid_mismatch_frames = 0
        self._raw_ray_slots = 0
        self._realized_distractor_rays = 0
        self._eligible_distractor_rays = 0
        self._removed_distractor_rays = 0
        self._frames_with_removed_distractor = 0
        self._current_winner_removed_bins = 0
        self._changed_bins = 0
        self._frames_with_changed_bin = 0
        self._current_feasible = 0
        self._corrected_feasible = 0
        self._both_feasible = 0
        self._current_only = 0
        self._corrected_only = 0
        self._neither = 0
        self._current_static_feasible = 0
        self._corrected_static_feasible = 0
        self._current_feasible_actions = 0
        self._corrected_feasible_actions = 0

    def _ensure_state(self, envs: int, device: torch.device) -> None:
        if self._active is not None:
            if self._active.shape != (envs,) or self._active.device != device:
                raise RuntimeError("D8 environment shape/device changed")
            return
        self._active = torch.zeros(envs, dtype=torch.bool, device=device)
        self._confirm = torch.zeros(envs, dtype=torch.long, device=device)
        self._release = torch.zeros(envs, dtype=torch.long, device=device)
        self._tracked_slot = torch.zeros(envs, dtype=torch.long, device=device)

    def _update_active(
        self, distances: torch.Tensor, closings: torch.Tensor
    ) -> torch.Tensor:
        """Mirror the frozen D4 two-frame trigger/release state machine."""

        envs, slots = distances.shape
        self._ensure_state(envs, distances.device)
        assert self._active is not None
        assert self._confirm is not None
        assert self._release is not None
        assert self._tracked_slot is not None
        env_index = torch.arange(envs, device=distances.device)
        tracked_distance = distances[env_index, self._tracked_slot]
        tracked_closing = closings[env_index, self._tracked_slot]
        resolved = (
            (tracked_distance >= float(self.geometry_spec.release_distance_m))
            | (tracked_closing <= 0.0)
        )
        self._release = torch.where(
            self._active & resolved,
            self._release + 1,
            torch.zeros_like(self._release),
        )
        release_now = self._active & (
            self._release >= int(self.geometry_spec.release_confirm_steps)
        )
        self._release_events += int(release_now.sum().item())
        self._active[release_now] = False
        self._release[release_now] = 0
        self._tracked_slot[release_now] = 0

        ttc = torch.where(
            closings > 0.0,
            (distances - float(self.geometry_spec.contact_distance_m)).clamp(min=0.0)
            / closings.clamp(min=1.0e-6),
            torch.full_like(distances, float("inf")),
        )
        qualifies = (
            (distances <= float(self.geometry_spec.trigger_distance_m))
            & (closings > 0.0)
            & (ttc <= float(self.geometry_spec.trigger_ttc_s))
        )
        inactive = ~self._active
        self._confirm = torch.where(
            inactive & qualifies.any(dim=1),
            (self._confirm + 1).clamp(
                max=int(self.geometry_spec.trigger_confirm_steps)
            ),
            torch.zeros_like(self._confirm),
        )
        trigger_now = inactive & (
            self._confirm >= int(self.geometry_spec.trigger_confirm_steps)
        )
        candidate_ttc = torch.where(
            qualifies, ttc, torch.full_like(ttc, float("inf"))
        )
        selected_slot = candidate_ttc.argmin(dim=1).clamp(0, slots - 1)
        self._tracked_slot[trigger_now] = selected_slot[trigger_now]
        self._active[trigger_now] = True
        self._confirm[trigger_now] = 0
        self._trigger_activations += int(trigger_now.sum().item())
        return self._active.clone()

    @torch.no_grad()
    def observe(
        self,
        policy_actions: torch.Tensor,
        *,
        policy_lidar: torch.Tensor,
        trace: dict,
        trace_match_count: int,
        context: dict,
    ) -> dict:
        if policy_actions.ndim != 2 or policy_actions.shape[1] != 2:
            raise ValueError("D8 policy actions must have shape [E,2]")
        before = policy_actions.detach().clone()
        envs = int(policy_actions.shape[0])
        _validate_trace_contract(trace, self.noise_spec)
        if int(trace_match_count) < 1:
            raise RuntimeError("D8 has no exact policy LiDAR trace match")
        current = _tensor(trace, "sweep_normalized")
        corrected = _tensor(trace, "valid_return_only_sweep_normalized")
        if not torch.equal(current, policy_lidar):
            raise RuntimeError("D8 current sweep is not bitwise policy LiDAR")
        if corrected.shape != current.shape:
            raise ValueError("D8 corrected sweep shape mismatch")
        if bool((corrected + 1.0e-7 < current).any()):
            self._monotonicity_errors += int(
                (corrected + 1.0e-7 < current).any(dim=1).sum().item()
            )
            raise RuntimeError("D8 correction introduced a nearer LiDAR return")

        realized = _tensor(trace, "realized_distractor_rays").long()
        eligible = _tensor(trace, "valid_return_only_eligible_distractor_rays").long()
        removed = _tensor(trace, "valid_return_only_removed_distractor_rays").long()
        current_winner_removed = _tensor(
            trace, "current_winner_removed_by_valid_return_only"
        ).bool()
        if realized.shape != (envs,) or eligible.shape != (envs,) or removed.shape != (envs,):
            raise ValueError("D8 per-environment distractor ledger shape mismatch")
        if not torch.equal(realized, eligible + removed):
            raise RuntimeError("D8 distractor eligibility ledger does not reconcile")
        changed = corrected != current
        self._raw_ray_slots += envs * int(self.noise_spec.expected_num_rays)
        self._realized_distractor_rays += int(realized.sum().item())
        self._eligible_distractor_rays += int(eligible.sum().item())
        self._removed_distractor_rays += int(removed.sum().item())
        self._frames_with_removed_distractor += int((removed > 0).sum().item())
        self._current_winner_removed_bins += int(current_winner_removed.sum().item())
        self._changed_bins += int(changed.sum().item())
        self._frames_with_changed_bin += int(changed.any(dim=1).sum().item())

        distances = _tensor(context, "obstacle_distances_m").float()
        closings = _tensor(context, "relative_closing_speeds_mps").float()
        if distances.ndim != 2 or closings.shape != distances.shape:
            raise ValueError("D8 distance/closing context must have shape [E,N]")
        if distances.shape[0] != envs:
            raise ValueError("D8 context environment count mismatch")
        active = self._update_active(distances, closings)
        active_count = int(active.sum().item())

        if active_count:
            keys = (
                "current_velocity_mps",
                "current_omega_rad_s",
                "pending_command",
                "dynamic_positions_body_m",
                "dynamic_velocities_body_mps",
                "dynamic_radii_m",
                "dynamic_valid",
            )
            values = {key: _tensor(context, key) for key in keys}
            common = {
                "policy_actions": policy_actions[active],
                "current_velocity_mps": values["current_velocity_mps"][active],
                "current_omega_rad_s": values["current_omega_rad_s"][active],
                "pending_command": values["pending_command"][active],
                "dynamic_positions_body_m": values["dynamic_positions_body_m"][active],
                "dynamic_velocities_body_mps": values["dynamic_velocities_body_mps"][active],
                "dynamic_radii_m": values["dynamic_radii_m"][active],
                "dynamic_valid": values["dynamic_valid"][active].bool(),
                "num_bins": int(context["num_bins"]),
                "max_linear_velocity": float(context["max_linear_velocity"]),
                "reverse_velocity_scale": float(context["reverse_velocity_scale"]),
                "max_linear_accel": float(context["max_linear_accel"]),
                "max_angular_velocity": float(context["max_angular_velocity"]),
                "max_angular_accel": float(context["max_angular_accel"]),
                "spec": self.geometry_spec,
            }
            current_grid = geometry_feasible_action_grid(
                **common,
                policy_lidar_clearance=current[active],
                lidar_beam_angles_rad=_tensor(
                    trace, "winner_actual_angle_rad"
                )[active],
            )
            corrected_grid = geometry_feasible_action_grid(
                **common,
                policy_lidar_clearance=corrected[active],
                lidar_beam_angles_rad=_tensor(
                    trace, "valid_return_only_winner_actual_angle_rad"
                )[active],
            )
            if not torch.equal(
                current_grid["dynamic_clearance_grid_m"],
                corrected_grid["dynamic_clearance_grid_m"],
            ):
                self._dynamic_grid_mismatch_frames += active_count
                raise RuntimeError("D8 dynamic grid changed between LiDAR shadows")
            current_any = current_grid["any_feasible"].bool()
            corrected_any = corrected_grid["any_feasible"].bool()
            both = current_any & corrected_any
            current_only = current_any & ~corrected_any
            corrected_only = ~current_any & corrected_any
            neither = ~current_any & ~corrected_any
            self._current_feasible += int(current_any.sum().item())
            self._corrected_feasible += int(corrected_any.sum().item())
            self._both_feasible += int(both.sum().item())
            self._current_only += int(current_only.sum().item())
            self._corrected_only += int(corrected_only.sum().item())
            self._neither += int(neither.sum().item())
            current_static = (
                current_grid["static_clearance_grid_m"]
                >= float(self.geometry_spec.static_surface_clearance_m)
            ).flatten(1).any(dim=1)
            corrected_static = (
                corrected_grid["static_clearance_grid_m"]
                >= float(self.geometry_spec.static_surface_clearance_m)
            ).flatten(1).any(dim=1)
            self._current_static_feasible += int(current_static.sum().item())
            self._corrected_static_feasible += int(corrected_static.sum().item())
            self._current_feasible_actions += int(
                current_grid["feasible_grid"].sum().item()
            )
            self._corrected_feasible_actions += int(
                corrected_grid["feasible_grid"].sum().item()
            )

        if not torch.equal(policy_actions, before):
            self._action_identity_errors += envs
            raise RuntimeError("D8 shadow modified policy actions")
        self._action_identity_checks += envs
        self._trace_matches += int(trace_match_count)
        self._calls += 1
        self._environment_frames += envs
        self._active_frames += active_count
        return {"active": active}

    def record_transition(self, done: torch.Tensor) -> None:
        self._transitions += 1
        if self._active is None:
            return
        done_mask = done.reshape(-1).to(
            device=self._active.device, dtype=torch.bool
        )
        if done_mask.shape != self._active.shape:
            raise ValueError("D8 done mask shape mismatch")
        assert self._confirm is not None
        assert self._release is not None
        assert self._tracked_slot is not None
        self._active[done_mask] = False
        self._confirm[done_mask] = 0
        self._release[done_mask] = 0
        self._tracked_slot[done_mask] = 0

    def report(self, *, metadata: dict) -> dict:
        active = self._active_frames
        current_no = active - self._current_feasible
        net = self._corrected_only - self._current_only
        rescue_fraction = self._corrected_only / current_no if current_no else 0.0
        net_fraction = net / active if active else 0.0
        major = (
            rescue_fraction
            >= float(self.spec.major_rescue_fraction_of_current_no_feasible)
            and net_fraction
            >= float(self.spec.material_net_recovery_fraction_of_active)
            and net >= int(self.spec.material_net_recovery_min_frames)
        )
        expected_frames = int(metadata.get("expected_records", -1))
        paired_ok = (
            self._both_feasible
            + self._current_only
            + self._corrected_only
            + self._neither
            == active
            and self._current_feasible == self._both_feasible + self._current_only
            and self._corrected_feasible
            == self._both_feasible + self._corrected_only
        )
        report = {
            "schema": REPORT_SCHEMA,
            "mode": MODE,
            "protocol": noise_eligibility_protocol(
                self.spec, self.noise_spec, self.geometry_spec
            ),
            "metadata": metadata,
            "runtime": {
                "calls": self._calls,
                "transitions": self._transitions,
                "environment_frames": self._environment_frames,
                "active_environment_frames": active,
                "trigger_activations": self._trigger_activations,
                "release_events": self._release_events,
                "trace_match_candidates": self._trace_matches,
                "action_identity_checks": self._action_identity_checks,
                "action_identity_errors": self._action_identity_errors,
                "monotonicity_errors": self._monotonicity_errors,
                "dynamic_grid_mismatch_frames": self._dynamic_grid_mismatch_frames,
            },
            "noise_ledger": {
                "raw_ray_slots": self._raw_ray_slots,
                "realized_distractor_rays": self._realized_distractor_rays,
                "eligible_distractor_rays": self._eligible_distractor_rays,
                "removed_no_return_or_dropped_distractor_rays": self._removed_distractor_rays,
                "frames_with_removed_distractor": self._frames_with_removed_distractor,
                "current_winner_bins_removed": self._current_winner_removed_bins,
                "changed_72_bins": self._changed_bins,
                "frames_with_changed_72_bin": self._frames_with_changed_bin,
            },
            "paired_feasibility": {
                "active_frames": active,
                "current_feasible_frames": self._current_feasible,
                "corrected_feasible_frames": self._corrected_feasible,
                "both_feasible_frames": self._both_feasible,
                "current_only_feasible_frames": self._current_only,
                "corrected_only_feasible_frames": self._corrected_only,
                "both_no_feasible_frames": self._neither,
                "current_no_feasible_frames": current_no,
                "current_no_feasible_fraction": current_no / active if active else 0.0,
                "corrected_no_feasible_fraction": (
                    (active - self._corrected_feasible) / active if active else 0.0
                ),
                "rescued_current_no_feasible_fraction": rescue_fraction,
                "net_feasible_recovery_frames": net,
                "net_feasible_recovery_fraction_of_active": net_fraction,
                "current_static_feasible_frames": self._current_static_feasible,
                "corrected_static_feasible_frames": self._corrected_static_feasible,
                "current_feasible_action_count": self._current_feasible_actions,
                "corrected_feasible_action_count": self._corrected_feasible_actions,
                "major_fake_obstacle_contributor": major,
                "descriptive_rule": asdict(self.spec),
            },
            "self_check": {
                "baseline_action_identity_ok": self._action_identity_errors == 0,
                "record_coverage_ok": expected_frames == self._environment_frames,
                "transition_coverage_ok": self._calls == self._transitions,
                "exact_policy_trace_match_ok": self._trace_matches >= self._calls,
                "counterfactual_range_monotonic_ok": self._monotonicity_errors == 0,
                "dynamic_grid_identity_ok": self._dynamic_grid_mismatch_frames == 0,
                "noise_ledger_ok": self._realized_distractor_rays
                == self._eligible_distractor_rays + self._removed_distractor_rays,
                "paired_feasibility_ledger_ok": paired_ok,
            },
        }
        report["self_check"]["reconciliation_ok"] = all(
            bool(value) for value in report["self_check"].values()
        )
        return report

    def write(self, path: str | Path, *, metadata: dict) -> dict:
        report = self.report(metadata=metadata)
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        return report


def build_noise_eligibility_sensitivity() -> NoiseEligibilitySensitivity:
    return NoiseEligibilitySensitivity()
