"""Auditable closed-loop metrics for the privileged corridor teacher."""

from __future__ import annotations

import math

import torch


class TeacherClosedLoopMetrics:
    """Track selected-path clearance and outcomes after entering goal near-field."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        *,
        near_goal_threshold_m: float = 3.0,
        near_hard_clearance_threshold_m: float = 0.11,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if (
            not math.isfinite(near_goal_threshold_m)
            or near_goal_threshold_m <= 0.0
        ):
            raise ValueError("near_goal_threshold_m must be finite and positive")
        self.num_envs = int(num_envs)
        self.near_goal_threshold_m = float(near_goal_threshold_m)
        if (
            not math.isfinite(near_hard_clearance_threshold_m)
            or near_hard_clearance_threshold_m <= 0.0
        ):
            raise ValueError(
                "near_hard_clearance_threshold_m must be finite and positive"
            )
        self.near_hard_clearance_threshold_m = float(
            near_hard_clearance_threshold_m
        )
        self._entered_near_goal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._all_clearance: list[torch.Tensor] = []
        self._near_clearance: list[torch.Tensor] = []
        self._frames = 0
        self._near_goal_frames = 0
        self._entered_outcomes = {
            "success": 0,
            "wall_collision": 0,
            "obstacle_collision": 0,
            "timeout": 0,
            "other": 0,
        }

    def record_step(
        self,
        *,
        goal_distance_m: torch.Tensor,
        selected_clearance_m: torch.Tensor,
        feasible: torch.Tensor,
    ) -> None:
        """Record pre-action state and the teacher-selected predicted path."""

        expected = (self.num_envs,)
        if goal_distance_m.shape != expected:
            raise ValueError("goal_distance_m must have shape [E]")
        if selected_clearance_m.shape != expected:
            raise ValueError("selected_clearance_m must have shape [E]")
        if feasible.shape != expected:
            raise ValueError("feasible must have shape [E]")

        near_goal = goal_distance_m <= self.near_goal_threshold_m
        self._entered_near_goal |= near_goal
        self._frames += self.num_envs
        self._near_goal_frames += int(near_goal.sum().item())

        valid = feasible.bool() & torch.isfinite(selected_clearance_m)
        if bool(valid.any()):
            self._all_clearance.append(selected_clearance_m[valid].detach())
        near_valid = valid & near_goal
        if bool(near_valid.any()):
            self._near_clearance.append(
                selected_clearance_m[near_valid].detach()
            )

    def finish_episodes(
        self,
        done_ids: torch.Tensor,
        cause: torch.Tensor,
    ) -> None:
        """Account terminal causes for episodes that entered D_goal <= 3 m."""

        if done_ids.ndim != 1:
            raise ValueError("done_ids must have shape [D]")
        if cause.shape != (self.num_envs,):
            raise ValueError("cause must have shape [E]")
        if done_ids.numel() == 0:
            return

        entered_ids = done_ids[self._entered_near_goal[done_ids]]
        if entered_ids.numel() > 0:
            entered_causes = cause[entered_ids]
            cause_map = {
                1: "success",
                2: "wall_collision",
                3: "obstacle_collision",
                4: "timeout",
            }
            accounted = 0
            for code, name in cause_map.items():
                count = int((entered_causes == code).sum().item())
                self._entered_outcomes[name] += count
                accounted += count
            self._entered_outcomes["other"] += (
                int(entered_ids.numel()) - accounted
            )
        self._entered_near_goal[done_ids] = False

    @staticmethod
    def _clearance_summary(
        values: list[torch.Tensor],
        *,
        near_hard_threshold_m: float,
    ) -> dict[str, int | float | None]:
        if not values:
            return {
                "samples": 0,
                "minimum_m": None,
                "p05_m": None,
                "median_m": None,
                "mean_m": None,
                "near_hard_threshold_m": near_hard_threshold_m,
                "near_hard_fraction": None,
            }
        merged = torch.cat(values).float()
        return {
            "samples": int(merged.numel()),
            "minimum_m": float(merged.min().item()),
            "p05_m": float(torch.quantile(merged, 0.05).item()),
            "median_m": float(torch.quantile(merged, 0.50).item()),
            "mean_m": float(merged.mean().item()),
            "near_hard_threshold_m": near_hard_threshold_m,
            "near_hard_fraction": float(
                (merged < near_hard_threshold_m).float().mean().item()
            ),
        }

    def to_report(self) -> dict[str, object]:
        entered = sum(self._entered_outcomes.values())
        denom = max(entered, 1)
        collisions = (
            self._entered_outcomes["wall_collision"]
            + self._entered_outcomes["obstacle_collision"]
        )
        return {
            "clearance_semantics": (
                "minimum predicted OBB-to-obstacle surface clearance along "
                "the selected 2 s path, after robot buffer"
            ),
            "selected_predicted_clearance": self._clearance_summary(
                self._all_clearance,
                near_hard_threshold_m=(
                    self.near_hard_clearance_threshold_m
                ),
            ),
            "near_goal_selected_predicted_clearance": self._clearance_summary(
                self._near_clearance,
                near_hard_threshold_m=(
                    self.near_hard_clearance_threshold_m
                ),
            ),
            "near_goal_episode_outcomes": {
                "threshold_m": self.near_goal_threshold_m,
                "entry_semantics": "pre-action goal distance <= threshold",
                "teacher_frames": self._frames,
                "near_goal_teacher_frames": self._near_goal_frames,
                "near_goal_frame_fraction": (
                    self._near_goal_frames / max(self._frames, 1)
                ),
                "entered_episodes": entered,
                **self._entered_outcomes,
                "completion_rate": (
                    self._entered_outcomes["success"] / denom
                ),
                "collision_rate": collisions / denom,
                "timeout_rate": self._entered_outcomes["timeout"] / denom,
            },
        }
