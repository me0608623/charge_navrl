"""Direct-crossing metrics for the randomised production narrow replay.

`narrow_gap_eval.NarrowGapController` owns the *fixed* Gate5 scene: it
places the walls itself and assumes barrier x=0, gap on the axis, and
west->east travel. Production narrow replay
(`events/narrow_passage_bridge.py`) randomises the gap center in
y in [-1,+1], the barrier in x in [-0.5,+0.5] and the travel direction, so
two of Gate5's four "direct" tests stop meaning what they say:

* pre-cross lateral excursion as |y| charges an offset gap for driving
  straight down its own axis;
* a crossing/backtrack test written as "x increases" scores every
  east->west episode as a permanent backtrack.

This module keeps the same four direct-crossing criteria but measures them
in each env's own frame: lateral excursion against that env's gap center,
and crossing/backtrack along that env's own travel direction. Scene
placement stays with the replay event — this only observes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NarrowReplayThresholds:
    """Same direct-crossing bar as Gate5, expressed geometry-agnostically."""

    path_length_ratio_max: float = 1.35
    first_cross_time_s_max: float = 10.0
    max_pre_cross_lateral_m: float = 1.0
    backtrack_distance_m_max: float = 0.5


class NarrowReplayMetrics:
    """Accumulate per-env crossing statistics across randomised episodes."""

    def __init__(
        self,
        *,
        num_envs: int,
        device,
        control_dt_s: float,
        thresholds: NarrowReplayThresholds = NarrowReplayThresholds(),
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = device
        self.control_dt_s = float(control_dt_s)
        self.thresholds = thresholds

        z = lambda: torch.zeros(self.num_envs, device=device)  # noqa: E731
        self._active = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
        self._barrier_x = z()
        self._gap_y = z()
        self._direction = torch.ones(self.num_envs, device=device)
        self._goal_xy = torch.zeros(self.num_envs, 2, device=device)
        self._start_xy = torch.zeros(self.num_envs, 2, device=device)
        self._previous_xy = torch.zeros(self.num_envs, 2, device=device)
        self._has_previous = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )
        self._path_length_m = z()
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=device
        )
        self._first_cross_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=device
        )
        self._max_pre_cross_lateral_m = z()
        self._backtrack_distance_m = z()
        self._crossed = torch.zeros(
            self.num_envs, dtype=torch.bool, device=device
        )

        self.completed_episodes = 0
        self.crossed_episodes = 0
        self.direct_crossed_episodes = 0
        self._path_ratio_samples: list[np.ndarray] = []
        self._first_cross_time_samples: list[np.ndarray] = []
        self._lateral_samples: list[np.ndarray] = []
        self._backtrack_samples: list[np.ndarray] = []

    def begin_episodes(
        self,
        env_ids: torch.Tensor,
        *,
        barrier_x_m: torch.Tensor,
        gap_center_y_m: torch.Tensor,
        goal_xy_m: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        """Latch the freshly installed geometry for the given envs."""

        if env_ids.numel() == 0:
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        self._active[ids] = active.to(device=self.device, dtype=torch.bool)[ids]
        self._barrier_x[ids] = barrier_x_m.to(self.device)[ids].float()
        self._gap_y[ids] = gap_center_y_m.to(self.device)[ids].float()
        self._goal_xy[ids] = goal_xy_m.to(self.device)[ids].float()

        direction = torch.sign(self._goal_xy[ids, 0] - self._barrier_x[ids])
        self._direction[ids] = torch.where(
            direction == 0.0, torch.ones_like(direction), direction
        )

        self._has_previous[ids] = False
        self._path_length_m[ids] = 0.0
        self._elapsed_steps[ids] = 0
        self._first_cross_step[ids] = -1
        self._max_pre_cross_lateral_m[ids] = 0.0
        self._backtrack_distance_m[ids] = 0.0
        self._crossed[ids] = False

    def observe(
        self, robot_xy_m: torch.Tensor, done: torch.Tensor | None = None
    ) -> None:
        """Fold one control step of motion into the per-env accumulators.

        `done` envs are skipped for this step: the environment auto-resets
        inside `step()`, so their post-step pose is already the next
        episode's spawn and folding it in would charge a teleport to the
        finished episode.
        """

        pos = robot_xy_m.to(self.device).float()
        live = self._active
        if done is not None:
            live = live & ~done.to(device=self.device, dtype=torch.bool).reshape(-1)
        if not bool(live.any()):
            return

        first = live & ~self._has_previous
        if bool(first.any()):
            self._start_xy[first] = pos[first]
            self._previous_xy[first] = pos[first]
            self._has_previous |= first

        step = torch.linalg.vector_norm(pos - self._previous_xy, dim=1)
        self._path_length_m += torch.where(live, step, torch.zeros_like(step))
        self._elapsed_steps += live.long()

        # 沿各 env 自己的行進方向量測「朝牆的進度」，鏡像場景才不會被誤判倒退。
        signed_now = self._direction * (pos[:, 0] - self._barrier_x)
        signed_prev = self._direction * (self._previous_xy[:, 0] - self._barrier_x)

        not_crossed = live & ~self._crossed
        # 側偏對「該 env 自己的缺口中心」量，缺口偏移時直行不該被記成繞行。
        lateral = (pos[:, 1] - self._gap_y).abs()
        self._max_pre_cross_lateral_m = torch.where(
            not_crossed,
            torch.maximum(self._max_pre_cross_lateral_m, lateral),
            self._max_pre_cross_lateral_m,
        )
        self._backtrack_distance_m += torch.where(
            not_crossed,
            (signed_prev - signed_now).clamp_min(0.0),
            torch.zeros_like(signed_now),
        )

        crossed_now = not_crossed & (signed_prev < 0.0) & (signed_now >= 0.0)
        self._first_cross_step[crossed_now] = self._elapsed_steps[crossed_now]
        self._crossed |= crossed_now
        self._previous_xy[live] = pos[live]

    def finish_episodes(self, env_ids: torch.Tensor) -> None:
        """Score the finished episodes and reset their accumulators."""

        if env_ids.numel() == 0:
            return
        ids = env_ids.to(device=self.device, dtype=torch.long)
        ids = ids[self._active[ids]]
        if ids.numel() == 0:
            return

        self.completed_episodes += int(ids.numel())
        crossed = self._crossed[ids]
        self.crossed_episodes += int(crossed.sum().item())
        self._active[ids] = False

        crossed_ids = ids[crossed]
        if crossed_ids.numel() == 0:
            return

        straight = torch.linalg.vector_norm(
            self._goal_xy[crossed_ids] - self._start_xy[crossed_ids], dim=1
        ).clamp_min(1e-6)
        path_ratio = self._path_length_m[crossed_ids] / straight
        first_cross_time_s = (
            self._first_cross_step[crossed_ids].float() * self.control_dt_s
        )
        lateral = self._max_pre_cross_lateral_m[crossed_ids]
        backtrack = self._backtrack_distance_m[crossed_ids]

        t = self.thresholds
        direct = (
            (path_ratio <= t.path_length_ratio_max)
            & (first_cross_time_s <= t.first_cross_time_s_max)
            & (lateral <= t.max_pre_cross_lateral_m)
            & (backtrack <= t.backtrack_distance_m_max)
        )
        self.direct_crossed_episodes += int(direct.sum().item())
        self._path_ratio_samples.append(path_ratio.detach().cpu().numpy())
        self._first_cross_time_samples.append(
            first_cross_time_s.detach().cpu().numpy()
        )
        self._lateral_samples.append(lateral.detach().cpu().numpy())
        self._backtrack_samples.append(backtrack.detach().cpu().numpy())

    def summary(self) -> dict[str, float]:
        def pct(samples: list[np.ndarray]) -> tuple[float, float, float]:
            values = (
                np.concatenate(samples)
                if samples
                else np.empty(0, dtype=np.float32)
            )
            if not values.size:
                return float("nan"), float("nan"), float("nan")
            return tuple(float(x) for x in np.percentile(values, [50, 90, 95]))

        n = self.completed_episodes
        out: dict[str, float] = {
            "episodes": n,
            "crossed": self.crossed_episodes,
            "direct_crossed": self.direct_crossed_episodes,
            "crossing_rate": self.crossed_episodes / n if n else float("nan"),
            "direct_crossing_rate": (
                self.direct_crossed_episodes / n if n else float("nan")
            ),
        }
        for name, samples in (
            ("path_length_ratio", self._path_ratio_samples),
            ("first_cross_time_s", self._first_cross_time_samples),
            ("max_pre_cross_lateral_m", self._lateral_samples),
            ("backtrack_distance_m", self._backtrack_samples),
        ):
            p50, p90, p95 = pct(samples)
            out[f"{name}_p50"] = p50
            out[f"{name}_p90"] = p90
            out[f"{name}_p95"] = p95
        return out

    def summary_line(self) -> str:
        s = self.summary()
        parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in s.items()]
        return "[NARROW-REPLAY-METRICS] " + " ".join(parts)
