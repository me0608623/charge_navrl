"""Fixed, deployment-oriented reward for the end-to-end PPO baseline."""

from __future__ import annotations

import math

import torch


def _flat_bool(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(-1).bool() if x.dim() > 1 else x.bool()


def _termination_masks(
    env_unwrapped: object,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Read terminal causes once and return mutually useful masks."""
    terminated_flat = _flat_bool(terminated)
    truncated_flat = _flat_bool(truncated)
    n = terminated_flat.shape[0]
    device = terminated_flat.device

    goal = torch.zeros(n, dtype=torch.bool, device=device)
    wall = torch.zeros_like(goal)
    obstacle = torch.zeros_like(goal)
    other = torch.zeros_like(goal)

    try:
        tm = env_unwrapped.termination_manager
        for name in tm._term_names:
            value = tm.get_term(name)
            if value is None:
                continue
            value = value.reshape(-1).bool()
            if "goal_reached" in name or "reaching_goal" in name:
                goal |= value
            elif "wall_collision" in name:
                wall |= value
            elif "obstacle_collision" in name or "collision" in name:
                obstacle |= value
            elif "tipped" in name or "explosion" in name or "flying" in name:
                other |= value
    except (AttributeError, RuntimeError):
        obstacle = terminated_flat & ~truncated_flat

    # A success wins if the simulator reports two terminal causes on one step.
    wall &= ~goal
    obstacle &= ~goal & ~wall
    other &= ~goal & ~wall & ~obstacle
    unexplained = terminated_flat & ~goal & ~wall & ~obstacle & ~other
    obstacle |= unexplained

    static_obstacle = getattr(
        env_unwrapped,
        "_obs_collision_static_mask",
        torch.zeros_like(goal),
    ).reshape(-1).bool()
    dynamic_obstacle = getattr(
        env_unwrapped,
        "_obs_collision_dynamic_mask",
        torch.zeros_like(goal),
    ).reshape(-1).bool()

    return {
        "goal_reached": goal,
        "wall_collision": wall,
        "obs_collision": obstacle,
        "other_death": other,
        "static_obs_collision": static_obstacle & obstacle,
        "dynamic_obs_collision": dynamic_obstacle & obstacle,
        "timeout": truncated_flat & ~terminated_flat,
        "done": terminated_flat | truncated_flat,
    }


class CleanProgressReward:
    """One fixed reward function shared by every curriculum stage.

    Per transition:

      r = (d_t - d_t+1) - 0.01
          - 0.01 |delta a_v| - 0.005 |delta a_w|
          + 10 [goal] - 15 [collision] - 5 [timeout]

    Action deltas are normalized by the 18-index action range. Progress is
    suppressed on terminal transitions because Isaac Lab has already reset
    ``next_obs`` for those environments.
    """

    name = "clean_progress"

    progress_scale = 1.0
    step_penalty = -0.01
    linear_smoothness = 0.01
    angular_smoothness = 0.005
    goal_reward_value = 10.0
    collision_penalty_value = -15.0
    timeout_penalty_value = -5.0

    def __init__(
        self,
        anti_spin_weight: float = 0.0,
        anti_spin_hazard_distance: float = 1.5,
        anti_spin_omega_threshold: float = 0.8,
        anti_spin_progress_threshold: float = 0.02,
        anti_spin_grace_steps: int = 5,
        anti_spin_ramp_steps: int = 5,
        anti_spin_yaw_grace_deg: float = 180.0,
        anti_spin_yaw_ramp_deg: float = 180.0,
        anti_spin_dt: float = 0.2,
        future_occupancy_weight: float = 0.0,
        future_occupancy_horizon_s: float = 1.5,
        future_occupancy_samples: int = 8,
        future_occupancy_safe_distance_m: float = 1.0,
        future_occupancy_near_distance_m: float = 3.0,
        future_occupancy_move_threshold_mps: float = 0.1,
    ) -> None:
        self.anti_spin_weight = max(0.0, float(anti_spin_weight))
        self.anti_spin_hazard_distance = float(anti_spin_hazard_distance)
        self.anti_spin_omega_threshold = float(anti_spin_omega_threshold)
        self.anti_spin_progress_threshold = float(anti_spin_progress_threshold)
        self.anti_spin_grace_steps = max(0, int(anti_spin_grace_steps))
        self.anti_spin_ramp_steps = max(1, int(anti_spin_ramp_steps))
        self.anti_spin_yaw_grace = math.radians(max(0.0, float(anti_spin_yaw_grace_deg)))
        self.anti_spin_yaw_ramp = math.radians(max(1e-3, float(anti_spin_yaw_ramp_deg)))
        self.anti_spin_dt = max(1e-6, float(anti_spin_dt))
        self.future_occupancy_weight = max(0.0, float(future_occupancy_weight))
        self.future_occupancy_horizon_s = max(1e-3, float(future_occupancy_horizon_s))
        self.future_occupancy_samples = max(1, int(future_occupancy_samples))
        self.future_occupancy_safe_distance_m = max(1e-3, float(future_occupancy_safe_distance_m))
        self.future_occupancy_near_distance_m = max(1e-3, float(future_occupancy_near_distance_m))
        self.future_occupancy_move_threshold_mps = max(0.0, float(future_occupancy_move_threshold_mps))
        self._anti_spin_run: torch.Tensor | None = None
        self._anti_spin_sign: torch.Tensor | None = None
        self._anti_spin_yaw: torch.Tensor | None = None

    def update_params(self, curriculum_info: dict) -> None:
        # Intentionally fixed across SA1..SA8.
        return None

    def compute(
        self,
        env_unwrapped: object,
        actions: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        context: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if context is None or "obs" not in context or "next_obs" not in context:
            raise ValueError("clean_progress requires context['obs'] and context['next_obs']")

        obs = context["obs"]
        next_obs = context["next_obs"]
        prev_actions = context.get("prev_actions")
        masks = _termination_masks(env_unwrapped, terminated, truncated)
        n = actions.shape[0]
        device = actions.device

        distance_before = obs[:, 4:6].norm(dim=-1)
        distance_after = next_obs[:, 4:6].norm(dim=-1)
        progress_reward = self.progress_scale * (distance_before - distance_after)
        progress_reward = torch.where(masks["done"], torch.zeros_like(progress_reward), progress_reward)

        anti_spin_reward = torch.zeros(n, dtype=obs.dtype, device=device)
        anti_spin_active = torch.zeros(n, dtype=torch.bool, device=device)
        anti_spin_run = torch.zeros(n, dtype=obs.dtype, device=device)
        anti_spin_yaw_deg = torch.zeros(n, dtype=obs.dtype, device=device)
        if self.anti_spin_weight > 0.0:
            required = ("near_obs_dist_m", "omega_actual_rad_s")
            missing = [key for key in required if key not in context]
            if missing:
                raise ValueError(f"clean_progress anti-spin requires context keys: {missing}")

            if self._anti_spin_run is None or self._anti_spin_run.shape != (n,) or self._anti_spin_run.device != device:
                self._anti_spin_run = torch.zeros(n, dtype=torch.long, device=device)
                self._anti_spin_sign = torch.zeros(n, dtype=torch.long, device=device)
                self._anti_spin_yaw = torch.zeros(n, dtype=obs.dtype, device=device)
            near_hazard = context["near_obs_dist_m"] < self.anti_spin_hazard_distance
            omega = context["omega_actual_rad_s"]
            high_omega = omega.abs() > self.anti_spin_omega_threshold
            low_progress = progress_reward <= self.anti_spin_progress_threshold
            anti_spin_active = near_hazard & high_omega & low_progress & ~masks["done"]
            omega_sign = torch.sign(omega).to(torch.long)
            same_sign = anti_spin_active & (omega_sign == self._anti_spin_sign)
            self._anti_spin_run = torch.where(
                anti_spin_active,
                self._anti_spin_run + 1,
                torch.zeros_like(self._anti_spin_run),
            )
            step_yaw = omega.abs() * self.anti_spin_dt
            self._anti_spin_yaw = torch.where(
                anti_spin_active,
                torch.where(same_sign, self._anti_spin_yaw + step_yaw, step_yaw),
                torch.zeros_like(self._anti_spin_yaw),
            )
            self._anti_spin_sign = torch.where(
                anti_spin_active, omega_sign, torch.zeros_like(self._anti_spin_sign)
            )
            excess = (self._anti_spin_run - self.anti_spin_grace_steps).clamp_min(0).to(obs.dtype)
            time_ramp = (excess / float(self.anti_spin_ramp_steps)).clamp(0.0, 1.0)
            yaw_ramp = (
                (self._anti_spin_yaw - self.anti_spin_yaw_grace) / self.anti_spin_yaw_ramp
            ).clamp(0.0, 1.0)
            ramp = torch.minimum(time_ramp, yaw_ramp)
            anti_spin_reward = -self.anti_spin_weight * ramp
            anti_spin_run = self._anti_spin_run.to(obs.dtype)
            anti_spin_yaw_deg = torch.rad2deg(self._anti_spin_yaw)
            self._anti_spin_run = torch.where(
                masks["done"], torch.zeros_like(self._anti_spin_run), self._anti_spin_run
            )
            self._anti_spin_sign = torch.where(
                masks["done"], torch.zeros_like(self._anti_spin_sign), self._anti_spin_sign
            )
            self._anti_spin_yaw = torch.where(
                masks["done"], torch.zeros_like(self._anti_spin_yaw), self._anti_spin_yaw
            )

        future_occupancy_reward = torch.zeros(n, dtype=obs.dtype, device=device)
        future_occupancy_active = torch.zeros(n, dtype=obs.dtype, device=device)
        future_occupancy_risk = torch.zeros(n, dtype=obs.dtype, device=device)
        future_occupancy_min_distance = torch.full((n,), float("inf"), dtype=obs.dtype, device=device)
        if self.future_occupancy_weight > 0.0:
            required = (
                "dynamic_obstacle_positions_body_m",
                "dynamic_obstacle_velocities_body_mps",
                "v_forward_actual_m",
                "omega_actual_rad_s",
            )
            missing = [key for key in required if key not in context]
            if missing:
                raise ValueError(f"clean_progress future occupancy requires context keys: {missing}")
            obstacle_pos = context["dynamic_obstacle_positions_body_m"]
            obstacle_vel = context["dynamic_obstacle_velocities_body_mps"]
            if obstacle_pos.shape != obstacle_vel.shape or obstacle_pos.ndim != 3 or obstacle_pos.shape[-1] != 2:
                raise ValueError(
                    "future occupancy expects obstacle positions/velocities shaped [E, N, 2], "
                    f"got {tuple(obstacle_pos.shape)} and {tuple(obstacle_vel.shape)}"
                )

            times = torch.linspace(
                self.future_occupancy_horizon_s / self.future_occupancy_samples,
                self.future_occupancy_horizon_s,
                self.future_occupancy_samples,
                dtype=obs.dtype,
                device=device,
            )
            v_forward = context["v_forward_actual_m"].to(obs.dtype)
            omega = context["omega_actual_rad_s"].to(obs.dtype)
            omega_safe = torch.where(omega.abs() < 1e-4, torch.ones_like(omega), omega)
            angle = omega[:, None] * times[None, :]
            radius = v_forward / omega_safe
            robot_x = radius[:, None] * torch.sin(angle)
            robot_y = radius[:, None] * (1.0 - torch.cos(angle))
            straight = omega.abs() < 1e-4
            robot_x = torch.where(straight[:, None], v_forward[:, None] * times[None, :], robot_x)
            robot_y = torch.where(straight[:, None], torch.zeros_like(robot_y), robot_y)
            robot_path = torch.stack([robot_x, robot_y], dim=-1)  # [E,T,2]

            obstacle_path = obstacle_pos[:, :, None, :] + obstacle_vel[:, :, None, :] * times[None, None, :, None]
            separation = (robot_path[:, None, :, :] - obstacle_path).norm(dim=-1)  # [E,N,T]
            min_separation = separation.min(dim=-1).values
            moving = obstacle_vel.norm(dim=-1) >= self.future_occupancy_move_threshold_mps
            near = obstacle_pos.norm(dim=-1) <= self.future_occupancy_near_distance_m
            valid = moving & near
            valid_separation = torch.where(valid, min_separation, torch.full_like(min_separation, float("inf")))
            future_occupancy_min_distance = valid_separation.min(dim=-1).values
            future_occupancy_active = valid.any(dim=-1).to(obs.dtype)
            per_obstacle_risk = (
                (self.future_occupancy_safe_distance_m - valid_separation)
                / self.future_occupancy_safe_distance_m
            ).clamp(0.0, 1.0).square()
            future_occupancy_risk = per_obstacle_risk.max(dim=-1).values
            future_occupancy_reward = -self.future_occupancy_weight * future_occupancy_risk
            future_occupancy_reward = torch.where(
                masks["done"], torch.zeros_like(future_occupancy_reward), future_occupancy_reward
            )
            future_occupancy_min_distance = torch.where(
                future_occupancy_active.bool(),
                future_occupancy_min_distance,
                torch.zeros_like(future_occupancy_min_distance),
            )

        time_reward = torch.full((n,), self.step_penalty, dtype=obs.dtype, device=device)
        smoothness_reward = torch.zeros(n, dtype=obs.dtype, device=device)
        if prev_actions is not None:
            delta = (actions.float() - prev_actions.float()).abs() / 18.0
            smoothness_reward = -(
                self.linear_smoothness * delta[:, 0]
                + self.angular_smoothness * delta[:, 1]
            )

        goal_reward = masks["goal_reached"].to(obs.dtype) * self.goal_reward_value
        collision = masks["wall_collision"] | masks["obs_collision"] | masks["other_death"]
        collision_reward = collision.to(obs.dtype) * self.collision_penalty_value
        timeout_reward = masks["timeout"].to(obs.dtype) * self.timeout_penalty_value
        total = (
            progress_reward
            + time_reward
            + smoothness_reward
            + goal_reward
            + collision_reward
            + timeout_reward
            + anti_spin_reward
            + future_occupancy_reward
        )

        wall_hit_reward = masks["wall_collision"].to(obs.dtype) * self.collision_penalty_value
        obs_hit_reward = (
            masks["obs_collision"] | masks["other_death"]
        ).to(obs.dtype) * self.collision_penalty_value
        breakdown = {
            "goal_reward": goal_reward,
            "wall_hit_reward": wall_hit_reward,
            "obs_hit_reward": obs_hit_reward,
            "timeout_reward": timeout_reward,
            "floor_reward": torch.zeros_like(total),
            "action_reward": time_reward + smoothness_reward,
            "progress_reward": progress_reward,
            "smoothness_reward": smoothness_reward,
            "anti_spin": anti_spin_reward,
            "anti_spin_active": anti_spin_active.to(obs.dtype),
            "anti_spin_run_steps": anti_spin_run,
            "anti_spin_same_sign_yaw_deg": anti_spin_yaw_deg,
            "future_occupancy": future_occupancy_reward,
            "future_occupancy_active": future_occupancy_active,
            "future_occupancy_risk": future_occupancy_risk,
            "future_occupancy_min_distance_m": future_occupancy_min_distance,
            **{key: value for key, value in masks.items() if key not in ("done", "timeout")},
        }
        return total, breakdown
