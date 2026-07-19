"""Episode-start reachability audit for generated navigation scenes.

This module is diagnostic only. It rasterizes the local arena after inflating
walls and obstacles by the robot footprint, then checks start-to-goal
connectivity. Dynamic obstacles are reported separately because their initial
position does not imply that the episode is permanently unsolvable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage


CAUSE_NAMES = {
    0: "running",
    1: "goal",
    2: "wall_collision",
    3: "obstacle_collision",
    4: "timeout",
    5: "other",
}


@dataclass(frozen=True)
class SolvabilitySpec:
    half_extent: float
    resolution: float = 0.15
    robot_radius: float = 0.35
    safety_buffer: float = 0.10

    @property
    def footprint_radius(self) -> float:
        return self.robot_radius + self.safety_buffer


def _grid(spec: SolvabilitySpec) -> tuple[np.ndarray, np.ndarray]:
    coords = np.arange(
        -spec.half_extent + 0.5 * spec.resolution,
        spec.half_extent,
        spec.resolution,
        dtype=np.float32,
    )
    return np.meshgrid(coords, coords, indexing="xy")


def rasterize_occupancy(
    spec: SolvabilitySpec,
    wall_centers: np.ndarray,
    wall_sizes: np.ndarray,
    wall_mask: np.ndarray,
    obstacle_centers: np.ndarray | None = None,
    obstacle_radii: np.ndarray | None = None,
    obstacle_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterize inflated AABB walls and circular obstacles."""
    xx, yy = _grid(spec)
    occupied = np.zeros(xx.shape, dtype=bool)
    inflate = spec.footprint_radius

    for center, size, active in zip(wall_centers, wall_sizes, wall_mask, strict=True):
        if not active:
            continue
        occupied |= (
            (np.abs(xx - center[0]) <= 0.5 * size[0] + inflate)
            & (np.abs(yy - center[1]) <= 0.5 * size[1] + inflate)
        )

    if obstacle_centers is not None:
        if obstacle_radii is None or obstacle_mask is None:
            raise ValueError("obstacle radii and mask are required with obstacle centers")
        for center, radius, active in zip(
            obstacle_centers, obstacle_radii, obstacle_mask, strict=True
        ):
            if not active:
                continue
            total_radius = float(radius) + inflate
            occupied |= (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= total_radius**2
    return occupied


def _point_to_cell(point: np.ndarray, spec: SolvabilitySpec, shape: tuple[int, int]) -> tuple[int, int] | None:
    col = int(np.floor((float(point[0]) + spec.half_extent) / spec.resolution))
    row = int(np.floor((float(point[1]) + spec.half_extent) / spec.resolution))
    if row < 0 or col < 0 or row >= shape[0] or col >= shape[1]:
        return None
    return row, col


def _nearest_free_cell(
    point: np.ndarray,
    occupied: np.ndarray,
    spec: SolvabilitySpec,
) -> tuple[int, int] | None:
    """Map an exact free endpoint to a nearby free cell without grid-edge false positives."""
    cell = _point_to_cell(point, spec, occupied.shape)
    if cell is None:
        return None
    if not occupied[cell]:
        return cell
    row, col = cell
    candidates = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = row + dr, col + dc
            if 0 <= rr < occupied.shape[0] and 0 <= cc < occupied.shape[1] and not occupied[rr, cc]:
                x = -spec.half_extent + (cc + 0.5) * spec.resolution
                y = -spec.half_extent + (rr + 0.5) * spec.resolution
                candidates.append(((x - point[0]) ** 2 + (y - point[1]) ** 2, rr, cc))
    if not candidates:
        return None
    _, row, col = min(candidates)
    return row, col


def _endpoint_clearance(
    point: np.ndarray,
    wall_centers: np.ndarray,
    wall_sizes: np.ndarray,
    wall_mask: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_radii: np.ndarray,
    obstacle_mask: np.ndarray,
) -> float:
    clearances = []
    for center, size, active in zip(wall_centers, wall_sizes, wall_mask, strict=True):
        if not active:
            continue
        outside = np.maximum(np.abs(point - center) - 0.5 * size, 0.0)
        clearances.append(float(np.linalg.norm(outside)))
    for center, radius, active in zip(
        obstacle_centers, obstacle_radii, obstacle_mask, strict=True
    ):
        if active:
            clearances.append(float(np.linalg.norm(point - center) - radius))
    return min(clearances, default=float("inf"))


def analyze_occupancy(
    occupied: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    spec: SolvabilitySpec,
    start_exact_free: bool | None = None,
    goal_exact_free: bool | None = None,
) -> dict[str, bool | float]:
    """Return endpoint validity, free-space fraction, and 4-connected reachability."""
    start_cell = _nearest_free_cell(start, occupied, spec)
    goal_cell = _nearest_free_cell(goal, occupied, spec)
    start_free = start_cell is not None if start_exact_free is None else bool(start_exact_free)
    goal_free = goal_cell is not None if goal_exact_free is None else bool(goal_exact_free)
    reachable = False
    if start_free and goal_free:
        labels, _ = ndimage.label(~occupied, structure=ndimage.generate_binary_structure(2, 1))
        reachable = labels[start_cell] != 0 and labels[start_cell] == labels[goal_cell]
    return {
        "start_free": bool(start_free),
        "goal_free": bool(goal_free),
        "reachable": bool(reachable),
        "free_fraction": float((~occupied).mean()),
    }


def analyze_scene(
    spec: SolvabilitySpec,
    start: np.ndarray,
    goal: np.ndarray,
    wall_centers: np.ndarray,
    wall_sizes: np.ndarray,
    wall_mask: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_radii: np.ndarray,
    static_mask: np.ndarray,
    active_mask: np.ndarray,
) -> dict[str, dict[str, bool | float]]:
    """Analyze walls-only, permanent-static, and instantaneous-all occupancy."""
    result = {}
    for tier, obs_mask in (
        ("walls", np.zeros_like(active_mask, dtype=bool)),
        ("static", active_mask & static_mask),
        ("all_initial", active_mask),
    ):
        start_clearance = _endpoint_clearance(
            start,
            wall_centers,
            wall_sizes,
            wall_mask,
            obstacle_centers,
            obstacle_radii,
            obs_mask,
        )
        goal_clearance = _endpoint_clearance(
            goal,
            wall_centers,
            wall_sizes,
            wall_mask,
            obstacle_centers,
            obstacle_radii,
            obs_mask,
        )
        occupied = rasterize_occupancy(
            spec,
            wall_centers,
            wall_sizes,
            wall_mask,
            obstacle_centers,
            obstacle_radii,
            obs_mask,
        )
        result[tier] = analyze_occupancy(
            occupied,
            start,
            goal,
            spec,
            start_exact_free=start_clearance >= spec.footprint_radius,
            goal_exact_free=goal_clearance >= spec.footprint_radius,
        )
        result[tier]["start_clearance"] = start_clearance
        result[tier]["goal_clearance"] = goal_clearance
    return result


def summarize_records(records: Iterable[dict]) -> dict:
    records = list(records)
    completed = [record for record in records if record.get("cause") is not None]
    summary: dict[str, object] = {
        "episodes_captured": len(records),
        "episodes_completed": len(completed),
    }
    for tier in ("walls", "static", "all_initial"):
        tier_summary: dict[str, object] = {}
        for subset_name, subset in (
            ("all", completed),
            ("reachable", [r for r in completed if r[tier]["reachable"]]),
            ("unreachable", [r for r in completed if not r[tier]["reachable"]]),
        ):
            n = len(subset)
            causes = {name: 0 for name in CAUSE_NAMES.values()}
            for record in subset:
                causes[CAUSE_NAMES.get(int(record["cause"]), "other")] += 1
            tier_summary[subset_name] = {
                "episodes": n,
                "fraction": n / len(completed) if completed else 0.0,
                "cause_fraction": {
                    name: count / n if n else 0.0 for name, count in causes.items()
                },
            }
        values = [r[tier]["free_fraction"] for r in completed]
        tier_summary["free_fraction_mean"] = float(np.mean(values)) if values else 0.0
        tier_summary["free_fraction_min"] = float(np.min(values)) if values else 0.0
        summary[tier] = tier_summary
    return summary


class SceneSolvabilityAudit:
    """Capture generated scene geometry at each episode start and correlate outcomes."""

    def __init__(self, raw_env, scheduler, output_path: str, spec: SolvabilitySpec):
        self.env = raw_env
        self.scheduler = scheduler
        self.output_path = Path(output_path)
        self.spec = spec
        self.records: list[dict] = []
        self.current: list[dict | None] = [None] * raw_env.num_envs
        self.capture_count = 0

    def _snapshot_tensors(self):
        import torch

        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.events.state import (
            get_obstacle_sizes,
        )
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.wall_layout import (
            get_combined_wall_data,
        )

        origins = self.env.scene.env_origins[:, :2]
        robot = self.env.scene["robot"].data.root_pos_w[:, :2] - origins
        goal = self.env.command_manager.get_term("goal_command").goal_pos_w[:, :2] - origins
        wall_centers, wall_sizes, wall_mask = get_combined_wall_data(self.env)

        obstacle_positions = []
        obstacle_visible = []
        for slot in range(getattr(self.scheduler, "max_obstacles", 0)):
            entity = self.env.scene[f"obstacle_{slot}"]
            obstacle_positions.append(entity.data.root_pos_w[:, :2] - origins)
            obstacle_visible.append(entity.data.root_pos_w[:, 2] > 0.0)
        if obstacle_positions:
            obs_pos = torch.stack(obstacle_positions, dim=1)
            visible = torch.stack(obstacle_visible, dim=1)
        else:
            obs_pos = torch.empty(self.env.num_envs, 0, 2, device=self.env.device)
            visible = torch.empty(self.env.num_envs, 0, dtype=torch.bool, device=self.env.device)

        if self.scheduler is not None:
            behavior = self.scheduler.behavior_type
            active = (behavior != 0) & visible
            static = behavior == 1
        else:
            active = visible
            static = visible

        if hasattr(self.env, "_obstacle_phys_radii"):
            radii = self.env._obstacle_phys_radii[:, : obs_pos.shape[1]]
        else:
            sizes = get_obstacle_sizes() or []
            base = [0.5 * float(sizes[i]) if i < len(sizes) else 0.35 for i in range(obs_pos.shape[1])]
            radii = torch.tensor(base, device=self.env.device).unsqueeze(0).expand(self.env.num_envs, -1)

        return tuple(
            tensor.detach().cpu().numpy()
            for tensor in (robot, goal, wall_centers, wall_sizes, wall_mask, obs_pos, radii, static, active)
        )

    def capture(self, env_ids=None) -> None:
        arrays = self._snapshot_tensors()
        if env_ids is None:
            ids = range(self.env.num_envs)
        else:
            ids = env_ids.detach().cpu().tolist()
        robot, goal, wall_c, wall_s, wall_m, obs_pos, radii, static, active = arrays
        for env_id in ids:
            tiers = analyze_scene(
                self.spec,
                robot[env_id],
                goal[env_id],
                wall_c[env_id],
                wall_s[env_id],
                wall_m[env_id],
                obs_pos[env_id],
                radii[env_id],
                static[env_id],
                active[env_id],
            )
            record = {
                "capture_id": self.capture_count,
                "env_id": int(env_id),
                "start": robot[env_id].tolist(),
                "goal": goal[env_id].tolist(),
                "active_walls": int(wall_m[env_id].sum()),
                "active_obstacles": int(active[env_id].sum()),
                "static_obstacles": int((active[env_id] & static[env_id]).sum()),
                **tiers,
                "cause": None,
            }
            self.capture_count += 1
            self.records.append(record)
            self.current[env_id] = record

    def finish(self, env_ids, cause) -> None:
        ids = env_ids.detach().cpu().tolist()
        causes = cause[env_ids].detach().cpu().tolist()
        for env_id, cause_id in zip(ids, causes, strict=True):
            record = self.current[env_id]
            if record is not None:
                record["cause"] = int(cause_id)
            self.current[env_id] = None

    def write_report(self) -> None:
        payload = {
            "spec": {
                "half_extent": self.spec.half_extent,
                "resolution": self.spec.resolution,
                "robot_radius": self.spec.robot_radius,
                "safety_buffer": self.spec.safety_buffer,
            },
            "summary": summarize_records(self.records),
            "episodes": self.records,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary = payload["summary"]
        print(f"[SOLVABILITY] report: {self.output_path}")
        for tier in ("walls", "static", "all_initial"):
            data = summary[tier]
            unreachable = data["unreachable"]
            print(
                f"[SOLVABILITY] {tier}: unreachable="
                f"{unreachable['episodes']}/{data['all']['episodes']} "
                f"({unreachable['fraction']:.1%}), free_mean={data['free_fraction_mean']:.1%}, "
                f"free_min={data['free_fraction_min']:.1%}"
            )
