"""Deterministic configurable wall-gap policy evaluation scene.

Two wall segments span the arena from the north/south boundary to a centered
opening. The robot starts west of the barrier and the goal is east, so
reaching the goal requires traversing the opening rather than driving around it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class NarrowGapSpec:
    gap_width: float = 0.85
    arena_half_extent: float = 5.0
    boundary_wall_width: float = 1.0
    wall_width: float = 1.0
    barrier_x: float = 0.0
    start_x: float = -3.0
    goal_x: float = 3.0
    yaw_limit_deg: float = 2.52
    control_dt_s: float = 0.2
    direct_path_length_ratio_max: float = 1.35
    direct_first_cross_time_s_max: float = 10.0
    direct_max_pre_cross_abs_y_m: float = 1.0
    direct_backtrack_distance_m_max: float = 0.5

    @property
    def boundary_inner_y(self) -> float:
        return self.arena_half_extent - 0.5 * self.boundary_wall_width

    @property
    def segment_length(self) -> float:
        return self.boundary_inner_y - 0.5 * self.gap_width

    @property
    def segment_center_y(self) -> float:
        return 0.5 * (self.boundary_inner_y + 0.5 * self.gap_width)

    def side_opening_m(self, actual_arena_size: float) -> float:
        """牆端到實際外牆內緣的空隙；>0 表示機器人可繞過中央缺口。

        牆的頂端恆等於 `self.boundary_inner_y`（由 segment 幾何保證），所以
        側口就是「實際場地的內緣」減去「spec 假設場地的內緣」。
        """
        return max(0.0, 0.5 * float(actual_arena_size) - self.arena_half_extent)


# 歷史上所有 Gate5 都在 10 m 場地執行，spec 的 arena_half_extent 預設也是 5.0，
# 兩者剛好重合 —— 這正是「--arena_size 沒有傳進 spec」這個錯配長期未被發現的原因。
LEGACY_ARENA_HALF_EXTENT = 5.0


def spec_for_arena(
    arena_size: float | None,
    *,
    gap_width: float,
    sealed: bool,
    **kwargs,
) -> NarrowGapSpec:
    """依場地尺寸建立 spec；`sealed` 決定兩種語意完全不同的閘。

    * ``sealed=True``（**Gate5a**）：牆隨場地延伸到外牆，只留中央窄口。
      這才是「純測直穿能力」的閘。
    * ``sealed=False``（**Gate5b**）：牆長固定按 10 m 場地算，不隨場地變，
      側口隨場地線性增加。這是「房間尺度捷徑」壓測 —— 也是歷史 Gate5 的行為。

    2026-07-27 實測（D0, gap 1.2 m）：legacy 在 10 m 側口 0 → direct 0%、撞牆 89.4%；
    12 m 側口 1.0 m → direct 0.42%（幾乎全繞牆端，橫偏中位 5.02 m）；
    14 m 側口 2.0 m → direct 100%。政策在中央通道餘裕 ~0.4 m、側口僅 ~0.2 m 的情況下
    仍選側口，故非路徑最佳化，而是外牆距離造成的 LiDAR 情境捷徑。
    """
    if sealed and arena_size is None:
        # sealed 的整個意義就是「牆隨場地延伸到外牆」。少了場地尺寸就退回
        # 寫死的 5.0，那正是 2026-07-27 錯配的成因 —— 寧可炸掉也不要靜默算錯。
        raise ValueError(
            "sealed narrow-gap mode requires an explicit arena size; "
            "falling back to the hard-coded 5.0 half extent is exactly the "
            "2026-07-27 mismatch that made every Gate5 measure dead-corner "
            "behaviour instead of direct crossing"
        )
    arena_size = (
        2.0 * LEGACY_ARENA_HALF_EXTENT if arena_size is None else float(arena_size)
    )
    if arena_size <= 0.0:
        raise ValueError("arena_size must be positive")
    half = 0.5 * arena_size if sealed else LEGACY_ARENA_HALF_EXTENT
    spec = NarrowGapSpec(
        gap_width=gap_width,
        arena_half_extent=half,
        **kwargs,
    )
    if spec.segment_length <= 0.0:
        raise ValueError(
            f"gap {gap_width} m leaves no wall inside a {arena_size} m arena"
        )
    return spec


def configure_narrow_gap_env(env_cfg, scene_final: dict, cli_args, spec=None) -> None:
    """Remove random geometry and reserve two exact barrier-wall segments."""
    spec = spec or NarrowGapSpec()
    scene_final.update({
        "num_goals": 1,
        "num_static": 0,
        "num_dynamic": 0,
        "walls_min": 2,
        "walls_max": 2,
        "goal_dist_min": abs(spec.goal_x - spec.start_x),
        "goal_dist_max": abs(spec.goal_x - spec.start_x),
    })
    cli_args.no_goal_movement = True

    goal_cfg = env_cfg.commands.goal_command
    goal_cfg.num_goals = 1
    goal_cfg.num_obstacles = 0
    # Bootstrap only. The controller installs the exact target after env creation.
    goal_cfg.ranges.distance = (1.5, 3.0)
    goal_cfg.ranges.angle = (-math.pi, math.pi)

    for event_name in ("randomize_obstacles", "randomize_obstacles_startup"):
        event = getattr(env_cfg.events, event_name, None)
        if event is not None:
            event.params.update({
                "empty_ratio": 1.0,
                "static_ratio": 1.0,
                "dynamic_ratio": 0.0,
                "num_obstacles_static": 0,
                "num_obstacles_dynamic": 0,
            })

    wall_event = getattr(env_cfg.events, "randomize_wall_positions", None)
    if wall_event is not None:
        wall_event.params.update({"min_walls": 0, "max_walls": 0})

    reset_event = getattr(env_cfg.events, "reset_base", None)
    if reset_event is not None:
        pose_range = reset_event.params.setdefault("pose_range", {})
        pose_range.update({"x": (spec.start_x, spec.start_x), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})
        velocity_range = reset_event.params.setdefault("velocity_range", {})
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)

    for slot in NarrowGapController.WALL_SLOTS:
        wall_cfg = getattr(env_cfg.scene, f"wall_internal_{slot}", None)
        if wall_cfg is not None:
            # Asset local x is the long axis; the controller rotates it by 90 deg.
            wall_cfg.spawn.size = (spec.segment_length, spec.wall_width, 3.0)


class NarrowGapController:
    """Own exact scene placement and collect throat yaw/crossing statistics."""

    WALL_SLOTS = (1, 7)

    def __init__(self, raw_env, spec=None) -> None:
        self.env = raw_env
        self.device = raw_env.device
        self.num_envs = raw_env.num_envs
        self.spec = spec or NarrowGapSpec()
        self._previous_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._previous_xy[:, 0] = self.spec.start_x
        self._crossed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._path_length_m = torch.zeros(self.num_envs, device=self.device)
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._first_cross_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._max_pre_cross_abs_y_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self._backtrack_distance_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.completed_episodes = 0
        self.crossed_episodes = 0
        self.direct_crossed_episodes = 0
        self._yaw_samples: list[np.ndarray] = []
        self._path_length_ratio_samples: list[np.ndarray] = []
        self._first_cross_time_s_samples: list[np.ndarray] = []
        self._max_pre_cross_abs_y_m_samples: list[np.ndarray] = []
        self._backtrack_distance_m_samples: list[np.ndarray] = []

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        self._place_robot_goal(env_ids)
        self._place_walls(env_ids)
        self._previous_xy[env_ids] = 0.0
        self._previous_xy[env_ids, 0] = self.spec.start_x
        self._crossed[env_ids] = False
        self._path_length_m[env_ids] = 0.0
        self._elapsed_steps[env_ids] = 0
        self._first_cross_step[env_ids] = -1
        self._max_pre_cross_abs_y_m[env_ids] = 0.0
        self._backtrack_distance_m[env_ids] = 0.0

    def observe(self, done: torch.Tensor | None = None) -> None:
        robot = self.env.scene["robot"]
        pos = robot.data.root_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        quat = robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2),
        )
        if done is None:
            active = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        else:
            active = ~done.to(device=self.device, dtype=torch.bool).reshape(-1)
        previous = self._previous_xy
        not_crossed = active & ~self._crossed
        step_distance = torch.linalg.vector_norm(pos - previous, dim=1)
        self._path_length_m += torch.where(
            active, step_distance, torch.zeros_like(step_distance)
        )
        self._elapsed_steps += active.long()
        self._max_pre_cross_abs_y_m = torch.where(
            not_crossed,
            torch.maximum(self._max_pre_cross_abs_y_m, pos[:, 1].abs()),
            self._max_pre_cross_abs_y_m,
        )
        self._backtrack_distance_m += torch.where(
            not_crossed,
            (previous[:, 0] - pos[:, 0]).clamp_min(0.0),
            torch.zeros_like(pos[:, 0]),
        )
        # Vehicle overlaps the barrier slab longitudinally. These are the frames
        # where alignment determines whether the configured opening is traversable.
        throat = active & (
            (pos[:, 0] - self.spec.barrier_x).abs()
            <= (0.5 * self.spec.wall_width + 0.35)
        )
        if throat.any():
            yaw_abs_deg = torch.rad2deg(torch.atan2(torch.sin(yaw), torch.cos(yaw)).abs())
            self._yaw_samples.append(yaw_abs_deg[throat].detach().cpu().numpy())

        crossed_now = (
            not_crossed
            & (previous[:, 0] < self.spec.barrier_x)
            & (pos[:, 0] >= self.spec.barrier_x)
        )
        self._first_cross_step[crossed_now] = self._elapsed_steps[crossed_now]
        self._crossed |= crossed_now
        self._previous_xy[active] = pos[active]

    def finish_episodes(self, env_ids: torch.Tensor) -> None:
        self.completed_episodes += int(env_ids.numel())
        crossed = self._crossed[env_ids]
        self.crossed_episodes += int(crossed.sum().item())
        crossed_ids = env_ids[crossed]
        if crossed_ids.numel() == 0:
            return

        straight_distance = abs(self.spec.goal_x - self.spec.start_x)
        path_ratio = self._path_length_m[crossed_ids] / max(
            straight_distance, 1e-6
        )
        first_cross_time_s = (
            self._first_cross_step[crossed_ids].float()
            * float(self.spec.control_dt_s)
        )
        max_pre_cross_y = self._max_pre_cross_abs_y_m[crossed_ids]
        backtrack = self._backtrack_distance_m[crossed_ids]
        direct = (
            (path_ratio <= self.spec.direct_path_length_ratio_max)
            & (
                first_cross_time_s
                <= self.spec.direct_first_cross_time_s_max
            )
            & (
                max_pre_cross_y
                <= self.spec.direct_max_pre_cross_abs_y_m
            )
            & (
                backtrack
                <= self.spec.direct_backtrack_distance_m_max
            )
        )
        self.direct_crossed_episodes += int(direct.sum().item())
        self._path_length_ratio_samples.append(
            path_ratio.detach().cpu().numpy()
        )
        self._first_cross_time_s_samples.append(
            first_cross_time_s.detach().cpu().numpy()
        )
        self._max_pre_cross_abs_y_m_samples.append(
            max_pre_cross_y.detach().cpu().numpy()
        )
        self._backtrack_distance_m_samples.append(
            backtrack.detach().cpu().numpy()
        )

    def summary_line(self) -> str:
        yaw = np.concatenate(self._yaw_samples) if self._yaw_samples else np.empty(0, dtype=np.float32)
        if yaw.size:
            p50, p90, p95 = np.percentile(yaw, [50, 90, 95])
            within = float((yaw <= self.spec.yaw_limit_deg).mean())
        else:
            p50 = p90 = p95 = float("nan")
            within = float("nan")
        crossing_rate = self.crossed_episodes / self.completed_episodes if self.completed_episodes else float("nan")
        direct_crossing_rate = (
            self.direct_crossed_episodes / self.completed_episodes
            if self.completed_episodes
            else float("nan")
        )

        def _percentiles(samples: list[np.ndarray]) -> tuple[float, float, float]:
            values = (
                np.concatenate(samples)
                if samples
                else np.empty(0, dtype=np.float32)
            )
            if not values.size:
                return float("nan"), float("nan"), float("nan")
            return tuple(float(x) for x in np.percentile(values, [50, 90, 95]))

        path_p50, path_p90, path_p95 = _percentiles(
            self._path_length_ratio_samples
        )
        time_p50, time_p90, time_p95 = _percentiles(
            self._first_cross_time_s_samples
        )
        lateral_p50, lateral_p90, lateral_p95 = _percentiles(
            self._max_pre_cross_abs_y_m_samples
        )
        backtrack_p50, backtrack_p90, backtrack_p95 = _percentiles(
            self._backtrack_distance_m_samples
        )
        return (
            f"[NARROW-GAP-METRICS] episodes={self.completed_episodes} "
            f"crossed={self.crossed_episodes} crossing_rate={crossing_rate:.6f} "
            f"direct_crossed={self.direct_crossed_episodes} "
            f"direct_crossing_rate={direct_crossing_rate:.6f} "
            f"path_length_ratio_p50={path_p50:.4f} "
            f"path_length_ratio_p90={path_p90:.4f} "
            f"path_length_ratio_p95={path_p95:.4f} "
            f"first_cross_time_s_p50={time_p50:.4f} "
            f"first_cross_time_s_p90={time_p90:.4f} "
            f"first_cross_time_s_p95={time_p95:.4f} "
            f"max_pre_cross_abs_y_m_p50={lateral_p50:.4f} "
            f"max_pre_cross_abs_y_m_p90={lateral_p90:.4f} "
            f"max_pre_cross_abs_y_m_p95={lateral_p95:.4f} "
            f"backtrack_distance_m_p50={backtrack_p50:.4f} "
            f"backtrack_distance_m_p90={backtrack_p90:.4f} "
            f"backtrack_distance_m_p95={backtrack_p95:.4f} "
            f"direct_path_length_ratio_max="
            f"{self.spec.direct_path_length_ratio_max:.4f} "
            f"direct_first_cross_time_s_max="
            f"{self.spec.direct_first_cross_time_s_max:.4f} "
            f"direct_max_pre_cross_abs_y_m="
            f"{self.spec.direct_max_pre_cross_abs_y_m:.4f} "
            f"direct_backtrack_distance_m_max="
            f"{self.spec.direct_backtrack_distance_m_max:.4f} "
            f"yaw_frames={yaw.size} yaw_abs_p50_deg={p50:.4f} "
            f"yaw_abs_p90_deg={p90:.4f} yaw_abs_p95_deg={p95:.4f} "
            f"yaw_within_{self.spec.yaw_limit_deg:.2f}deg={within:.6f}"
        )

    def _place_robot_goal(self, env_ids: torch.Tensor) -> None:
        origins = self.env.scene.env_origins[env_ids]
        robot = self.env.scene["robot"]
        pose = robot.data.default_root_state[env_ids, :7].clone()
        pose[:, :2] = origins[:, :2]
        pose[:, 0] += self.spec.start_x
        pose[:, 3:7] = 0.0
        pose[:, 3] = 1.0
        robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)

        goal = torch.zeros(len(env_ids), 3, device=self.device)
        goal[:, 0] = origins[:, 0] + self.spec.goal_x
        goal[:, 1] = origins[:, 1]
        goal_term = self.env.command_manager.get_term("goal_command")
        if hasattr(goal_term, "goal_pos_w"):
            goal_term.goal_pos_w[env_ids] = goal
        if hasattr(goal_term, "all_goals_pos_w"):
            goal_term.all_goals_pos_w[env_ids] = goal[:, None, :]
        if hasattr(self.env, "_local_goal_world") and self.env._local_goal_world is not None:
            self.env._local_goal_world[env_ids, :2] = goal[:, :2]
        if hasattr(goal_term, "_update_goal_markers"):
            goal_term._update_goal_markers()

    def _place_walls(self, env_ids: torch.Tensor) -> None:
        origins = self.env.scene.env_origins[env_ids]
        centers_y = (self.spec.segment_center_y, -self.spec.segment_center_y)
        half_yaw = math.pi / 4.0
        wall_quat = torch.tensor(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            device=self.device,
        )

        for slot in range(8):
            try:
                wall = self.env.scene[f"wall_internal_{slot}"]
            except KeyError:
                continue
            pose = torch.zeros(len(env_ids), 7, device=self.device)
            pose[:, :2] = origins[:, :2]
            pose[:, 2] = -10.0
            pose[:, 3] = 1.0
            if slot in self.WALL_SLOTS:
                idx = self.WALL_SLOTS.index(slot)
                pose[:, 0] += self.spec.barrier_x
                pose[:, 1] += centers_y[idx]
                pose[:, 2] = 1.5
                pose[:, 3:7] = wall_quat
            wall.write_root_pose_to_sim(pose, env_ids=env_ids)

        if hasattr(self.env, "_maze_wall_mask"):
            self.env._maze_wall_mask[env_ids] = False
            for idx, slot in enumerate(self.WALL_SLOTS):
                self.env._maze_wall_mask[env_ids, slot] = True
                self.env._maze_wall_centers[env_ids, slot, 0] = self.spec.barrier_x
                self.env._maze_wall_centers[env_ids, slot, 1] = centers_y[idx]
                self.env._maze_wall_sizes[env_ids, slot, 0] = self.spec.wall_width
                self.env._maze_wall_sizes[env_ids, slot, 1] = self.spec.segment_length
