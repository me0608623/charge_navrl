"""Offline probe: does the C_goal normalisation denominator cause the
1.5-3 m clearance danger band in the privileged corridor teacher?

Pure torch, no Isaac, no GPU.

Stage 1 reproduces the frozen 2026-07-29 sweep
(``logs/probes/b2_clearance_sweep.json``) so the harness is validated before any
counterfactual is trusted.

Stage 2 changes exactly one line of the cost -- the goal normalisation
denominator -- reusing the identical geometry helpers from the production
teacher, so the only behavioural difference is that denominator.

Baseline production rule (privileged_corridor_teacher.py:372-374):

    denom = max(||g - p_0||, 1.0)
    C_goal = ||g - p_H|| / denom
    => marginal value of 1 m of progress = goal_weight / max(D, 1)

That makes forward progress *more* valuable the closer the goal is, which is the
mechanism suspected of crushing clearance in the 1.5-3 m band.

Counterfactual: denom = constant R, so the marginal value of progress is
distance-independent (goal_weight / R at every D).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_SKRL = "/home/aa/IsaacLab/scripts/reinforcement_learning/skrl"
if _SKRL not in sys.path:
    sys.path.insert(0, _SKRL)

from rnn_car_wdclean.privileged_corridor_teacher import (  # noqa: E402
    CorridorTeacherSpec,
    _obb_circle_clearance,
    _obb_wall_collision,
    _unicycle_paths,
)
from rnn_car_wdclean.reward_diagnostics import (  # noqa: E402
    decode_discrete_drive_action_grid,
)

FROZEN = Path("/home/aa/IsaacLab/logs/probes/b2_clearance_sweep.json")

# Scene per Obsidian 方法論/teacher設計驗收.md §4.3.1
ROBOT_X, ROBOT_Y = 0.5, 0.0
ROBOT_V = 1.0
ROBOT_OMEGA = 0.0
CORRIDOR_HALF_WIDTH_M = 2.0     # free width 4.0 m, inner faces at y = +/-2.0
CORRIDOR_LENGTH_M = 10.0
WALL_THICKNESS_M = 0.5
OBSTACLE_RADIUS_M = 0.3

DTYPE = torch.float32


def build_scene(
    goal_distances: list[float],
    offsets: list[float],
    *,
    obstacle_x_m: float,
    samples: int,
) -> dict[str, torch.Tensor]:
    """One batch element per (goal distance, obstacle lateral offset) pair."""

    combos = [(d, o) for d in goal_distances for o in offsets]
    n = len(combos)
    goal_d = torch.tensor([c[0] for c in combos], dtype=DTYPE)
    off = torch.tensor([c[1] for c in combos], dtype=DTYPE)

    robot_xy = torch.stack(
        [torch.full((n,), ROBOT_X, dtype=DTYPE),
         torch.full((n,), ROBOT_Y, dtype=DTYPE)],
        dim=-1,
    )
    goal_xy = torch.stack([robot_xy[:, 0] + goal_d, torch.zeros(n, dtype=DTYPE)], dim=-1)

    # Static obstacle: identical position at every horizon sample.
    obstacle_xy = torch.stack(
        [torch.full((n,), obstacle_x_m, dtype=DTYPE), off], dim=-1
    )
    obstacle_paths = obstacle_xy[:, None, None, :].expand(n, 1, samples, 2).contiguous()
    obstacle_radii = torch.full((n, 1), OBSTACLE_RADIUS_M, dtype=DTYPE)
    obstacle_valid = torch.ones(n, 1, dtype=torch.bool)

    half_t = WALL_THICKNESS_M * 0.5
    centre_x = CORRIDOR_LENGTH_M * 0.5
    wall_centers = torch.tensor(
        [
            [centre_x, CORRIDOR_HALF_WIDTH_M + half_t],
            [centre_x, -(CORRIDOR_HALF_WIDTH_M + half_t)],
        ],
        dtype=DTYPE,
    )[None].expand(n, 2, 2).contiguous()
    wall_sizes = torch.tensor(
        [[CORRIDOR_LENGTH_M, WALL_THICKNESS_M]] * 2, dtype=DTYPE
    )[None].expand(n, 2, 2).contiguous()
    wall_valid = torch.ones(n, 2, dtype=torch.bool)

    return {
        "combos": combos,
        "current_velocity": torch.full((n,), ROBOT_V, dtype=DTYPE),
        "current_omega": torch.full((n,), ROBOT_OMEGA, dtype=DTYPE),
        "robot_xy_m": robot_xy,
        "robot_yaw_rad": torch.zeros(n, dtype=DTYPE),
        "goal_xy_m": goal_xy,
        "obstacle_paths_m": obstacle_paths,
        "obstacle_radii_m": obstacle_radii,
        "obstacle_valid": obstacle_valid,
        "wall_centers_m": wall_centers,
        "wall_sizes_m": wall_sizes,
        "wall_valid": wall_valid,
    }


def rank(
    scene: dict[str, torch.Tensor],
    drive: dict,
    spec: CorridorTeacherSpec,
    *,
    goal_denominator: str | float,
) -> dict[str, torch.Tensor]:
    """Mirror corridor_teacher_action_grid, with a swappable goal denominator.

    ``goal_denominator="production"`` reproduces max(||g-p0||, 1.0) exactly.
    A float R uses that constant instead, making the marginal value of forward
    progress independent of goal distance.
    """

    linear, angular = decode_discrete_drive_action_grid(
        scene["current_velocity"],
        scene["current_omega"],
        num_bins=drive["num_bins"],
        dt=drive["dt"],
        max_linear_velocity=drive["max_linear_velocity"],
        reverse_velocity_scale=drive["reverse_velocity_scale"],
        max_linear_accel=drive["max_linear_accel"],
        max_angular_velocity=drive["max_angular_velocity"],
        max_angular_accel=drive["max_angular_accel"],
    )
    path, yaw = _unicycle_paths(
        linear, angular, scene["robot_xy_m"], scene["robot_yaw_rad"],
        horizon_s=spec.horizon_s, samples=spec.samples,
    )
    obstacle_clearance = _obb_circle_clearance(
        path, yaw, scene["obstacle_paths_m"], scene["obstacle_radii_m"],
        scene["obstacle_valid"], spec,
    )
    min_obstacle_clearance = obstacle_clearance.amin(dim=(-1, -2))
    obstacle_collision = min_obstacle_clearance < spec.hard_obstacle_clearance_m
    wall_collision = _obb_wall_collision(
        path, yaw, scene["wall_centers_m"], scene["wall_sizes_m"],
        scene["wall_valid"], spec,
    ).any(dim=-1)
    feasible = ~obstacle_collision & ~wall_collision
    if not spec.allow_reverse:
        feasible &= linear >= -1e-4

    endpoint = path[..., -1, :]
    endpoint_yaw = yaw[..., -1]
    goal_delta = scene["goal_xy_m"][:, None, None, :] - endpoint
    goal_distance = goal_delta.norm(dim=-1)

    if goal_denominator == "production":
        denom = (
            scene["goal_xy_m"] - scene["robot_xy_m"]
        ).norm(dim=-1).clamp_min(1.0)
    else:
        denom = torch.full(
            (scene["goal_xy_m"].shape[0],), float(goal_denominator), dtype=DTYPE
        )
    goal_distance_cost = goal_distance / denom[:, None, None]

    goal_heading = torch.atan2(goal_delta[..., 1], goal_delta[..., 0])
    heading_error = torch.atan2(
        torch.sin(goal_heading - endpoint_yaw),
        torch.cos(goal_heading - endpoint_yaw),
    ).abs() / torch.pi
    clearance_cost = (
        (spec.obstacle_clearance_margin_m - min_obstacle_clearance)
        / spec.obstacle_clearance_margin_m
    ).clamp(0.0, 1.0)
    smoothness = (
        (linear - scene["current_velocity"][:, None, None]).abs()
        / max(float(drive["max_linear_velocity"]), 1e-6)
        + (angular - scene["current_omega"][:, None, None]).abs()
        / max(float(drive["max_angular_velocity"]), 1e-6)
    )
    stall = (linear.abs() < spec.stall_speed_mps).float()
    cost = (
        spec.obstacle_clearance_weight * clearance_cost
        + spec.goal_distance_weight * goal_distance_cost
        + spec.goal_heading_weight * heading_error
        + spec.action_smoothness_weight * smoothness
        + spec.stall_weight * stall
    )
    ranked = torch.where(feasible, cost, torch.full_like(cost, float("inf")))
    flat = ranked.flatten(1).argmin(dim=-1)
    any_feasible = feasible.flatten(1).any(dim=-1)

    flat_clear = min_obstacle_clearance.flatten(1)
    chosen_clearance = flat_clear.gather(1, flat[:, None]).squeeze(1)
    # Best clearance among feasible actions only.
    masked = torch.where(
        feasible.flatten(1), flat_clear, torch.full_like(flat_clear, -float("inf"))
    )
    best_clearance = masked.amax(dim=-1)

    # Progress guard: a planner that buys clearance by crawling is not a fix.
    chosen_linear = linear.flatten(1).gather(1, flat[:, None]).squeeze(1)
    chosen_angular = angular.flatten(1).gather(1, flat[:, None]).squeeze(1)
    initial_goal_distance = (
        scene["goal_xy_m"] - scene["robot_xy_m"]
    ).norm(dim=-1)
    chosen_end_goal_distance = (
        goal_distance.flatten(1).gather(1, flat[:, None]).squeeze(1)
    )
    # Metres of goal distance removed over the 2 s horizon.
    chosen_progress = initial_goal_distance - chosen_end_goal_distance
    return {
        "any_feasible": any_feasible,
        "chosen_clearance": chosen_clearance,
        "best_clearance": best_clearance,
        "feasible_count": feasible.flatten(1).sum(dim=-1),
        "chosen_linear_mps": chosen_linear,
        "chosen_angular_rps": chosen_angular,
        "chosen_progress_m": chosen_progress,
    }


def to_grid(values: torch.Tensor, n_goals: int, n_off: int) -> list[list[float]]:
    return values.reshape(n_goals, n_off).tolist()


def main() -> int:
    frozen = json.loads(FROZEN.read_text())
    drive = frozen["drive"]
    goals = frozen["goal_distances_m"]
    offsets = frozen["obstacle_offsets_m"]
    spec = CorridorTeacherSpec()

    assert abs(spec.hard_obstacle_clearance_m - frozen["hard_limit_m"]) < 1e-9
    assert abs(spec.obstacle_clearance_margin_m - frozen["margin_m"]) < 1e-9

    ref_chosen = torch.tensor(frozen["grid"]["chosen_clearance_m"], dtype=DTYPE)
    ref_best = torch.tensor(frozen["grid"]["best_available_clearance_m"], dtype=DTYPE)

    # --- Stage 1: calibrate obstacle longitudinal placement, then reproduce ---
    print("=" * 74)
    print("STAGE 1  reproduce frozen 2026-07-29 sweep")
    print("=" * 74)
    best_candidate, best_err = None, None
    for obstacle_x in (2.0, 2.5):
        scene = build_scene(goals, offsets, obstacle_x_m=obstacle_x, samples=spec.samples)
        out = rank(scene, drive, spec, goal_denominator="production")
        chosen = torch.tensor(to_grid(out["chosen_clearance"], len(goals), len(offsets)))
        err = (chosen - ref_chosen).abs().max().item()
        print(f"  obstacle_x = {obstacle_x:.2f} m  ->  max |chosen - frozen| = {err:.6f}")
        if best_err is None or err < best_err:
            best_candidate, best_err = obstacle_x, err

    obstacle_x = best_candidate
    scene = build_scene(goals, offsets, obstacle_x_m=obstacle_x, samples=spec.samples)
    base = rank(scene, drive, spec, goal_denominator="production")
    chosen = torch.tensor(to_grid(base["chosen_clearance"], len(goals), len(offsets)))
    best = torch.tensor(to_grid(base["best_clearance"], len(goals), len(offsets)))
    chosen_err = (chosen - ref_chosen).abs().max().item()
    best_err_v = (best - ref_best).abs().max().item()

    print(f"\n  selected obstacle_x = {obstacle_x:.2f} m")
    print(f"  max |chosen - frozen| = {chosen_err:.8f}")
    print(f"  max |best   - frozen| = {best_err_v:.8f}")
    reproduced = chosen_err < 1e-4 and best_err_v < 1e-4
    print(f"  REPRODUCED = {reproduced}")
    if not reproduced:
        print("\n  !! harness does not reproduce the frozen sweep.")
        print("     Counterfactual arms below are NOT trustworthy. Stopping.")
        return 1

    # --- Stage 2: counterfactual denominators ---
    print()
    print("=" * 74)
    print("STAGE 2  counterfactual goal-normalisation denominator")
    print("=" * 74)
    col = offsets.index(0.3)
    arms = [("production  max(D,1)", "production"), ("R=2.0", 2.0),
            ("R=4.0", 4.0), ("R=10.0", 10.0)]
    results = {}
    print(f"\nchosen clearance (m) at obstacle offset {offsets[col]} m")
    print("  the frozen danger band is goal distance 1.5-3.0 m\n")
    header = "  D (m) | " + " | ".join(f"{name:>18}" for name, _ in arms)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for arm_name, denom in arms:
        out = rank(scene, drive, spec, goal_denominator=denom)
        results[arm_name] = {
            "chosen": to_grid(out["chosen_clearance"], len(goals), len(offsets)),
            "best": to_grid(out["best_clearance"], len(goals), len(offsets)),
            "feasible_count": to_grid(
                out["feasible_count"].to(DTYPE), len(goals), len(offsets)
            ),
            "any_feasible": to_grid(
                out["any_feasible"].to(DTYPE), len(goals), len(offsets)
            ),
            "chosen_linear_mps": to_grid(
                out["chosen_linear_mps"], len(goals), len(offsets)
            ),
            "chosen_angular_rps": to_grid(
                out["chosen_angular_rps"], len(goals), len(offsets)
            ),
            "chosen_progress_m": to_grid(
                out["chosen_progress_m"], len(goals), len(offsets)
            ),
        }
    for gi, d in enumerate(goals):
        cells = " | ".join(
            f"{results[name]['chosen'][gi][col]:>18.4f}" for name, _ in arms
        )
        band = " <-- band" if 1.5 <= d <= 3.0 else ""
        print(f"  {d:>5.2f} | {cells}{band}")

    avail = results["production  max(D,1)"]["best"]
    print(f"\n  best available clearance at this offset: {avail[0][col]:.4f} m (constant)")

    print("\n  surrendered margin = best available - chosen, danger band only")
    for arm_name, _ in arms:
        band = [
            avail[gi][col] - results[arm_name]["chosen"][gi][col]
            for gi, d in enumerate(goals) if 1.5 <= d <= 3.0
        ]
        print(f"    {arm_name:>20}: max {max(band):.4f} m")

    print("\n  whole-grid minimum chosen clearance (all D, all offsets)")
    for arm_name, _ in arms:
        flat = [v for row in results[arm_name]["chosen"] for v in row]
        n_at_limit = sum(1 for v in flat if v < spec.hard_obstacle_clearance_m + 0.01)
        print(
            f"    {arm_name:>20}: min {min(flat):.4f} m, "
            f"cells within 1 cm of hard limit: {n_at_limit}/{len(flat)}"
        )

    # --- Progress guard -----------------------------------------------------
    print()
    print("=" * 74)
    print("PROGRESS GUARD  did the safer arm just slow down?")
    print("=" * 74)
    print(f"\nchosen linear velocity (m/s) at obstacle offset {offsets[col]} m")
    print(f"  robot enters at v = {ROBOT_V} m/s; reachable next-step band is "
          f"v +/- a_max*dt = +/-{drive['max_linear_accel']*drive['dt']:.2f} m/s\n")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for gi, d in enumerate(goals):
        cells = " | ".join(
            f"{results[name]['chosen_linear_mps'][gi][col]:>18.4f}" for name, _ in arms
        )
        band = " <-- band" if 1.5 <= d <= 3.0 else ""
        print(f"  {d:>5.2f} | {cells}{band}")

    print(f"\ngoal distance removed over the {spec.horizon_s} s horizon (m), "
          f"offset {offsets[col]} m\n")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for gi, d in enumerate(goals):
        cells = " | ".join(
            f"{results[name]['chosen_progress_m'][gi][col]:>18.4f}" for name, _ in arms
        )
        band = " <-- band" if 1.5 <= d <= 3.0 else ""
        print(f"  {d:>5.2f} | {cells}{band}")

    print("\n  whole-grid summary")
    for arm_name, _ in arms:
        lin = [v for row in results[arm_name]["chosen_linear_mps"] for v in row]
        prog = [v for row in results[arm_name]["chosen_progress_m"] for v in row]
        n_stall = sum(1 for v in lin if abs(v) < spec.stall_speed_mps)
        n_backward = sum(1 for v in prog if v < 0.0)
        print(
            f"    {arm_name:>20}: mean v {sum(lin)/len(lin):.4f} m/s, "
            f"min v {min(lin):.4f}, stall cells {n_stall}/{len(lin)}, "
            f"negative-progress cells {n_backward}/{len(prog)}"
        )

    out_path = Path(
        "/home/aa/IsaacLab/logs/probes/teacher_goal_denominator_probe.json"
    )
    out_path.write_text(json.dumps({
        "provenance": {
            "purpose": "offline sensitivity of teacher clearance to the C_goal "
                       "normalisation denominator",
            "reproduces": str(FROZEN),
            "reproduction_max_abs_error": {
                "chosen_clearance_m": chosen_err,
                "best_available_clearance_m": best_err_v,
            },
            "calibrated_obstacle_x_m": obstacle_x,
            "scene": {
                "robot_xy_m": [ROBOT_X, ROBOT_Y],
                "robot_v_mps": ROBOT_V,
                "robot_omega_rps": ROBOT_OMEGA,
                "corridor_free_half_width_m": CORRIDOR_HALF_WIDTH_M,
                "corridor_length_m": CORRIDOR_LENGTH_M,
                "wall_thickness_m": WALL_THICKNESS_M,
                "obstacle_radius_m": OBSTACLE_RADIUS_M,
                "obstacle_static": True,
            },
            "drive": drive,
            "limits": {
                "hard_obstacle_clearance_m": spec.hard_obstacle_clearance_m,
                "obstacle_clearance_margin_m": spec.obstacle_clearance_margin_m,
                "goal_distance_weight": spec.goal_distance_weight,
                "obstacle_clearance_weight": spec.obstacle_clearance_weight,
            },
            "caveats": [
                "single static obstacle, single initial speed, synthetic scene",
                "measures cost-function tendency, not a realised rollout distribution",
                "reverse_velocity_scale not swept",
                "no dynamic obstacle pause branch",
            ],
        },
        "goal_distances_m": goals,
        "obstacle_offsets_m": offsets,
        "arms": results,
    }, indent=2))
    print(f"\n  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
