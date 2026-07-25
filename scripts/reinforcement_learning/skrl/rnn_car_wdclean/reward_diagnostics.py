"""Scene-conditioned reward diagnostics used by the training monitor."""

from __future__ import annotations

import torch


_LONG_CORRIDOR_STEP_KEYS = (
    "future_occupancy",
    "future_occupancy_active",
    "future_occupancy_risk",
    "progress_reward",
    "wall_collision",
    "static_obs_collision",
    "dynamic_obs_collision",
)


def add_long_corridor_reward_diagnostics(
    reward_breakdown: dict[str, torch.Tensor],
    scene_mask: torch.Tensor | None,
) -> bool:
    """Add scalar diagnostics for transitions that began in corridor replay.

    ``scene_mask`` must be captured before ``env.step``. Isaac Lab resets done
    environments inside ``env.step``, so reading the live scene mask afterward
    can associate a terminal reward with the next episode's scene type.
    """

    if scene_mask is None:
        return False
    scene_mask = scene_mask.reshape(-1).bool()
    if not bool(scene_mask.any()):
        return False

    def _value(key: str) -> torch.Tensor | None:
        value = reward_breakdown.get(key)
        if value is None:
            return None
        value = value.detach().reshape(-1)
        if value.shape != scene_mask.shape:
            raise ValueError(
                f"reward diagnostic {key!r} has shape {tuple(value.shape)}, "
                f"expected {tuple(scene_mask.shape)}"
            )
        return value

    for key in _LONG_CORRIDOR_STEP_KEYS:
        value = _value(key)
        if value is not None:
            reward_breakdown[f"{key}_long_corridor"] = (
                value[scene_mask].float().mean()
            )

    active = _value("future_occupancy_active")
    risk = _value("future_occupancy_risk")
    min_distance = _value("future_occupancy_min_distance_m")
    if active is not None:
        active_mask = scene_mask & active.bool()
        if bool(active_mask.any()):
            if risk is not None:
                reward_breakdown[
                    "future_occupancy_risk_active_long_corridor"
                ] = risk[active_mask].float().mean()
            if min_distance is not None:
                reward_breakdown[
                    "future_occupancy_min_distance_m_active_long_corridor"
                ] = min_distance[active_mask].float().mean()

    dynamic_collision = _value("dynamic_obs_collision")
    if dynamic_collision is not None:
        collision_mask = scene_mask & dynamic_collision.bool()
        if bool(collision_mask.any()):
            if risk is not None:
                reward_breakdown[
                    "future_occupancy_risk_on_dynamic_collision_long_corridor"
                ] = risk[collision_mask].float().mean()
            if active is not None:
                reward_breakdown[
                    "future_occupancy_active_on_dynamic_collision_long_corridor"
                ] = active[collision_mask].float().mean()
            if min_distance is not None:
                reward_breakdown[
                    "future_occupancy_min_distance_m_on_dynamic_collision_long_corridor"
                ] = min_distance[collision_mask].float().mean()

    return True


def decode_discrete_drive_action_grid(
    current_velocity: torch.Tensor,
    current_omega: torch.Tensor,
    *,
    num_bins: int,
    dt: float,
    max_linear_velocity: float,
    reverse_velocity_scale: float,
    max_linear_accel: float,
    max_angular_velocity: float,
    max_angular_accel: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode every linear/angular action pair without mutating the env."""

    if num_bins < 3 or num_bins % 2 == 0:
        raise ValueError("num_bins must be odd and at least 3")
    current_velocity = current_velocity.reshape(-1).float()
    current_omega = current_omega.reshape(-1).float()
    if current_velocity.shape != current_omega.shape:
        raise ValueError("current velocity and omega shapes must match")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    center = num_bins // 2
    ratios = (
        torch.arange(
            num_bins,
            device=current_velocity.device,
            dtype=current_velocity.dtype,
        )
        - center
    ) / float(center)

    accel_max = torch.minimum(
        torch.full_like(current_velocity, float(max_linear_accel)),
        (float(max_linear_velocity) - current_velocity) / float(dt),
    )
    reverse_limit = float(max_linear_velocity) * float(
        reverse_velocity_scale
    )
    accel_min = torch.maximum(
        torch.full_like(current_velocity, -float(max_linear_accel)),
        (-reverse_limit - current_velocity) / float(dt),
    )
    linear_accel = torch.where(
        ratios[None, :] >= 0.0,
        ratios[None, :] * accel_max[:, None],
        -ratios[None, :] * accel_min[:, None],
    )
    linear_accel = torch.maximum(
        torch.minimum(linear_accel, accel_max[:, None]),
        accel_min[:, None],
    )
    next_velocity = (
        current_velocity[:, None] + linear_accel * float(dt)
    ).clamp(-reverse_limit, float(max_linear_velocity))

    target_omega = ratios[None, :] * float(max_angular_velocity)
    max_delta_omega = float(max_angular_accel) * float(dt)
    next_omega = current_omega[:, None] + (
        target_omega - current_omega[:, None]
    ).clamp(-max_delta_omega, max_delta_omega)
    next_omega = next_omega.clamp(
        -float(max_angular_velocity),
        float(max_angular_velocity),
    )

    linear_grid = next_velocity[:, :, None].expand(
        -1, num_bins, num_bins
    )
    angular_grid = next_omega[:, None, :].expand(
        -1, num_bins, num_bins
    )
    return linear_grid, angular_grid


def future_occupancy_risk_grid(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    obstacle_positions_body_m: torch.Tensor,
    obstacle_velocities_body_mps: torch.Tensor,
    *,
    horizon_s: float = 1.5,
    samples: int = 8,
    safe_distance_m: float = 1.0,
    near_distance_m: float = 3.0,
    move_threshold_mps: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate the existing future-occupancy risk for an action grid."""

    if linear_velocity.shape != angular_velocity.shape:
        raise ValueError("linear and angular action grids must match")
    if linear_velocity.ndim != 3:
        raise ValueError("action grids must have shape [E,L,A]")
    if (
        obstacle_positions_body_m.shape
        != obstacle_velocities_body_mps.shape
        or obstacle_positions_body_m.ndim != 3
        or obstacle_positions_body_m.shape[-1] != 2
    ):
        raise ValueError("obstacle states must have shape [E,N,2]")
    if obstacle_positions_body_m.shape[0] != linear_velocity.shape[0]:
        raise ValueError("action and obstacle environment counts must match")
    if samples < 1:
        raise ValueError("samples must be positive")

    dtype = linear_velocity.dtype
    device = linear_velocity.device
    times = torch.linspace(
        float(horizon_s) / int(samples),
        float(horizon_s),
        int(samples),
        dtype=dtype,
        device=device,
    )
    omega_safe = torch.where(
        angular_velocity.abs() < 1e-4,
        torch.ones_like(angular_velocity),
        angular_velocity,
    )
    angle = angular_velocity[..., None] * times
    radius = linear_velocity / omega_safe
    robot_x = radius[..., None] * torch.sin(angle)
    robot_y = radius[..., None] * (1.0 - torch.cos(angle))
    straight = angular_velocity.abs() < 1e-4
    robot_x = torch.where(
        straight[..., None],
        linear_velocity[..., None] * times,
        robot_x,
    )
    robot_y = torch.where(
        straight[..., None],
        torch.zeros_like(robot_y),
        robot_y,
    )
    robot_path = torch.stack([robot_x, robot_y], dim=-1)

    obstacle_path = (
        obstacle_positions_body_m[:, :, None, :]
        + obstacle_velocities_body_mps[:, :, None, :]
        * times[None, None, :, None]
    )
    separation = (
        robot_path[:, :, :, None, :, :]
        - obstacle_path[:, None, None, :, :, :]
    ).norm(dim=-1)
    min_separation = separation.min(dim=-1).values
    moving = (
        obstacle_velocities_body_mps.norm(dim=-1)
        >= float(move_threshold_mps)
    )
    near = (
        obstacle_positions_body_m.norm(dim=-1)
        <= float(near_distance_m)
    )
    valid = moving & near
    valid_separation = torch.where(
        valid[:, None, None, :],
        min_separation,
        torch.full_like(min_separation, float("inf")),
    )
    min_distance = valid_separation.min(dim=-1).values
    per_obstacle_risk = (
        (float(safe_distance_m) - valid_separation)
        / float(safe_distance_m)
    ).clamp(0.0, 1.0).square()
    risk = per_obstacle_risk.max(dim=-1).values
    active = valid.any(dim=-1)
    min_distance = torch.where(
        active[:, None, None],
        min_distance,
        torch.zeros_like(min_distance),
    )
    return risk, min_distance, active


def future_occupancy_action_counterfactuals(
    current_velocity: torch.Tensor,
    current_omega: torch.Tensor,
    selected_actions: torch.Tensor,
    obstacle_positions_body_m: torch.Tensor,
    obstacle_velocities_body_mps: torch.Tensor,
    *,
    num_bins: int,
    dt: float,
    max_linear_velocity: float,
    reverse_velocity_scale: float,
    max_linear_accel: float,
    max_angular_velocity: float,
    max_angular_accel: float,
    horizon_s: float = 1.5,
    samples: int = 8,
    safe_distance_m: float = 1.0,
    near_distance_m: float = 3.0,
    move_threshold_mps: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compare selected, turn-only, brake-only and globally best risks."""

    linear_grid, angular_grid = decode_discrete_drive_action_grid(
        current_velocity,
        current_omega,
        num_bins=num_bins,
        dt=dt,
        max_linear_velocity=max_linear_velocity,
        reverse_velocity_scale=reverse_velocity_scale,
        max_linear_accel=max_linear_accel,
        max_angular_velocity=max_angular_velocity,
        max_angular_accel=max_angular_accel,
    )
    risk, min_distance, active = future_occupancy_risk_grid(
        linear_grid,
        angular_grid,
        obstacle_positions_body_m,
        obstacle_velocities_body_mps,
        horizon_s=horizon_s,
        samples=samples,
        safe_distance_m=safe_distance_m,
        near_distance_m=near_distance_m,
        move_threshold_mps=move_threshold_mps,
    )
    actions = selected_actions.round().long()
    linear_index = actions[:, 0].clamp(0, num_bins - 1)
    angular_index = actions[:, 1].clamp(0, num_bins - 1)
    env_index = torch.arange(
        actions.shape[0], device=actions.device
    )
    selected_risk = risk[env_index, linear_index, angular_index]
    selected_min_distance = min_distance[
        env_index, linear_index, angular_index
    ]
    best_any_risk = risk.flatten(1).min(dim=-1).values
    best_turn_risk = risk[env_index, linear_index, :].min(dim=-1).values
    best_brake_risk = risk[env_index, :, angular_index].min(dim=-1).values
    return {
        "selected_risk": selected_risk,
        "selected_min_distance_m": selected_min_distance,
        "best_any_risk": best_any_risk,
        "best_turn_risk": best_turn_risk,
        "best_brake_risk": best_brake_risk,
        "active": active,
    }


class FutureOccupancyLeadTimeAudit:
    """Aggregate counterfactual opportunities before dynamic collisions."""

    def __init__(
        self,
        num_envs: int,
        *,
        max_lead_steps: int = 8,
        device: torch.device | str = "cpu",
        safe_risk_threshold: float = 0.01,
        meaningful_risk_reduction: float = 0.20,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if max_lead_steps < 1:
            raise ValueError("max_lead_steps must be positive")
        self.num_envs = int(num_envs)
        self.max_lead_steps = int(max_lead_steps)
        self.safe_risk_threshold = float(safe_risk_threshold)
        self.meaningful_risk_reduction = float(
            meaningful_risk_reduction
        )
        self.device = torch.device(device)
        shape = (self.max_lead_steps, self.num_envs)
        self._present = torch.zeros(
            shape, dtype=torch.bool, device=self.device
        )
        self._audited = torch.zeros_like(self._present)
        self._selected = torch.zeros(
            shape, dtype=torch.float32, device=self.device
        )
        self._best_any = torch.zeros_like(self._selected)
        self._best_turn = torch.zeros_like(self._selected)
        self._best_brake = torch.zeros_like(self._selected)
        self._cursor = 0

        stat_shape = (self.max_lead_steps,)
        self._eligible = torch.zeros(
            stat_shape, dtype=torch.float32, device=self.device
        )
        self._audited_count = torch.zeros_like(self._eligible)
        self._selected_sum = torch.zeros_like(self._eligible)
        self._best_any_sum = torch.zeros_like(self._eligible)
        self._best_turn_sum = torch.zeros_like(self._eligible)
        self._best_brake_sum = torch.zeros_like(self._eligible)
        self._safe_any = torch.zeros_like(self._eligible)
        self._safe_turn = torch.zeros_like(self._eligible)
        self._safe_brake = torch.zeros_like(self._eligible)
        self._meaningful_any = torch.zeros_like(self._eligible)
        self._meaningful_turn = torch.zeros_like(self._eligible)
        self._meaningful_brake = torch.zeros_like(self._eligible)

    @torch.no_grad()
    def record_step(
        self,
        *,
        selected_risk: torch.Tensor,
        best_any_risk: torch.Tensor,
        best_turn_risk: torch.Tensor,
        best_brake_risk: torch.Tensor,
        audited_mask: torch.Tensor,
        dynamic_collision: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Record this state and score prior states when a collision occurs."""

        tensors = (
            selected_risk,
            best_any_risk,
            best_turn_risk,
            best_brake_risk,
            audited_mask,
            dynamic_collision,
            done,
        )
        if any(tensor.reshape(-1).shape[0] != self.num_envs for tensor in tensors):
            raise ValueError("all lead-audit inputs must contain num_envs values")
        selected = selected_risk.reshape(-1).to(
            device=self.device, dtype=torch.float32
        )
        best_any = best_any_risk.reshape(-1).to(
            device=self.device, dtype=torch.float32
        )
        best_turn = best_turn_risk.reshape(-1).to(
            device=self.device, dtype=torch.float32
        )
        best_brake = best_brake_risk.reshape(-1).to(
            device=self.device, dtype=torch.float32
        )
        audited = audited_mask.reshape(-1).to(
            device=self.device, dtype=torch.bool
        )
        collision = dynamic_collision.reshape(-1).to(
            device=self.device, dtype=torch.bool
        )
        done_mask = done.reshape(-1).to(
            device=self.device, dtype=torch.bool
        )

        for lead_index in range(self.max_lead_steps):
            lead_steps = lead_index + 1
            slot = (self._cursor - lead_steps) % self.max_lead_steps
            eligible = collision & self._present[slot]
            self._eligible[lead_index] += eligible.float().sum()
            valid = eligible & self._audited[slot]
            self._audited_count[lead_index] += valid.float().sum()
            if not bool(valid.any()):
                continue
            prior_selected = self._selected[slot, valid]
            self._selected_sum[lead_index] += prior_selected.sum()
            for label in ("any", "turn", "brake"):
                prior_best = getattr(self, f"_best_{label}")[slot, valid]
                getattr(self, f"_best_{label}_sum")[
                    lead_index
                ] += prior_best.sum()
                getattr(self, f"_safe_{label}")[
                    lead_index
                ] += (
                    prior_best <= self.safe_risk_threshold
                ).float().sum()
                getattr(self, f"_meaningful_{label}")[
                    lead_index
                ] += (
                    prior_selected - prior_best
                    >= self.meaningful_risk_reduction
                ).float().sum()

        # Reusing a ring slot must erase data from max_lead_steps ago.
        self._present[self._cursor] = False
        self._audited[self._cursor] = False
        self._selected[self._cursor] = 0.0
        self._best_any[self._cursor] = 0.0
        self._best_turn[self._cursor] = 0.0
        self._best_brake[self._cursor] = 0.0

        # Terminal transitions reset inside env.step. Their pre-step state is
        # not useful to the next episode, and all older history must be cleared.
        if bool(done_mask.any()):
            self._present[:, done_mask] = False
            self._audited[:, done_mask] = False
            self._selected[:, done_mask] = 0.0
            self._best_any[:, done_mask] = 0.0
            self._best_turn[:, done_mask] = 0.0
            self._best_brake[:, done_mask] = 0.0

        keep = ~done_mask
        self._present[self._cursor, keep] = True
        self._audited[self._cursor, keep] = audited[keep]
        self._selected[self._cursor, keep] = selected[keep]
        self._best_any[self._cursor, keep] = best_any[keep]
        self._best_turn[self._cursor, keep] = best_turn[keep]
        self._best_brake[self._cursor, keep] = best_brake[keep]
        self._cursor = (self._cursor + 1) % self.max_lead_steps

    def report(self, *, step_dt: float) -> list[dict[str, float | int]]:
        if step_dt <= 0.0:
            raise ValueError("step_dt must be positive")

        def _float(tensor: torch.Tensor, index: int) -> float:
            return float(tensor[index].detach().cpu().item())

        rows: list[dict[str, float | int]] = []
        for index in range(self.max_lead_steps):
            eligible = int(round(_float(self._eligible, index)))
            audited = int(round(_float(self._audited_count, index)))
            audited_denom = float(max(audited, 1))
            eligible_denom = float(max(eligible, 1))
            selected_mean = _float(self._selected_sum, index) / audited_denom
            row: dict[str, float | int] = {
                "lead_steps": index + 1,
                "lead_seconds": (index + 1) * float(step_dt),
                "eligible_collision_events": eligible,
                "audited_risky_frames": audited,
                "audited_fraction": audited / eligible_denom,
                "selected_risk_mean": selected_mean,
            }
            for label in ("any", "turn", "brake"):
                best_mean = (
                    _float(getattr(self, f"_best_{label}_sum"), index)
                    / audited_denom
                )
                row[f"best_{label}_risk_mean"] = best_mean
                row[f"{label}_risk_reduction_mean"] = (
                    selected_mean - best_mean
                )
                row[f"safe_{label}_fraction_of_audited"] = (
                    _float(getattr(self, f"_safe_{label}"), index)
                    / audited_denom
                )
                row[f"safe_{label}_fraction_of_collisions"] = (
                    _float(getattr(self, f"_safe_{label}"), index)
                    / eligible_denom
                )
                row[f"meaningful_{label}_fraction_of_audited"] = (
                    _float(
                        getattr(self, f"_meaningful_{label}"), index
                    )
                    / audited_denom
                )
            rows.append(row)
        return rows
