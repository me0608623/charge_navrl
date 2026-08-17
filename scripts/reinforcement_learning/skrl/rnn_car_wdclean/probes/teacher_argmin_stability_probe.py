"""Offline probe for known gap #4: is the teacher's argmin label CE-friendly?

Pure torch, no Isaac, no GPU. Calls the PRODUCTION
``corridor_teacher_action_grid`` so what is measured is exactly what training
labels would have been.

Three questions:

C. ACTION-GRID DEGENERACY
   How many of the 361 bins are physically distinct (v, omega) commands? If many
   bins decode to the same command, "exact bin agreement" is not a measure of
   behavioural agreement, and cross-entropy on the bin index penalises the
   student for choosing a physically identical action.

A. NEAR-TIE AMONG *PHYSICALLY DISTINCT* ALTERNATIVES
   Ignoring duplicates, does a genuinely different command cost almost the same
   as the argmin? If so the label is fragile to small state changes.

B. CONTINUITY ALONG A TRAVERSE
   Step the robot down the corridor and watch the label. Infeasible states are
   excluded because they emit the (9,9) placeholder and never enter supervision.

Soft-label relevance: distillation puts ``neighbor_mass=0.20`` on bins +/-1
only, so jumps of >=2 bins are outside what soft labels can smooth.
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
    corridor_teacher_action_grid,
)
from rnn_car_wdclean.reward_diagnostics import (  # noqa: E402
    decode_discrete_drive_action_grid,
)

FROZEN = Path("/home/aa/IsaacLab/logs/probes/b2_clearance_sweep.json")
OUT = Path("/home/aa/IsaacLab/logs/probes/teacher_argmin_stability_probe.json")

ROBOT_Y = 0.0
ROBOT_V = 1.0
ROBOT_OMEGA = 0.0
CORRIDOR_HALF_WIDTH_M = 2.0
CORRIDOR_LENGTH_M = 10.0
WALL_THICKNESS_M = 0.5
OBSTACLE_RADIUS_M = 0.3
OBSTACLE_X_M = 2.5          # calibrated to exactly reproduce b2_clearance_sweep
SOFT_LABEL_REACH_BINS = 1
# Two commands count as physically distinct only beyond these tolerances.
V_TOL_MPS = 0.01
W_TOL_RPS = 0.01

DTYPE = torch.float32


def make_batch(robot_x, robot_y, goal_xy, obstacle_xy, *, samples):
    n = robot_x.numel()
    obstacle_paths = obstacle_xy[:, None, None, :].expand(n, 1, samples, 2).contiguous()
    half_t = WALL_THICKNESS_M * 0.5
    cx = CORRIDOR_LENGTH_M * 0.5
    wall_centers = torch.tensor(
        [[cx, CORRIDOR_HALF_WIDTH_M + half_t], [cx, -(CORRIDOR_HALF_WIDTH_M + half_t)]],
        dtype=DTYPE,
    )[None].expand(n, 2, 2).contiguous()
    wall_sizes = torch.tensor(
        [[CORRIDOR_LENGTH_M, WALL_THICKNESS_M]] * 2, dtype=DTYPE
    )[None].expand(n, 2, 2).contiguous()
    return {
        "current_velocity": torch.full((n,), ROBOT_V, dtype=DTYPE),
        "current_omega": torch.full((n,), ROBOT_OMEGA, dtype=DTYPE),
        "robot_xy_m": torch.stack([robot_x, robot_y], dim=-1),
        "robot_yaw_rad": torch.zeros(n, dtype=DTYPE),
        "goal_xy_m": goal_xy,
        "obstacle_paths_m": obstacle_paths,
        "obstacle_radii_m": torch.full((n, 1), OBSTACLE_RADIUS_M, dtype=DTYPE),
        "obstacle_valid": torch.ones(n, 1, dtype=torch.bool),
        "wall_centers_m": wall_centers,
        "wall_sizes_m": wall_sizes,
        "wall_valid": torch.ones(n, 2, dtype=torch.bool),
    }


def run_teacher(batch, drive, spec):
    return corridor_teacher_action_grid(
        num_bins=drive["num_bins"], dt=drive["dt"],
        max_linear_velocity=drive["max_linear_velocity"],
        reverse_velocity_scale=drive["reverse_velocity_scale"],
        max_linear_accel=drive["max_linear_accel"],
        max_angular_velocity=drive["max_angular_velocity"],
        max_angular_accel=drive["max_angular_accel"],
        spec=spec, **batch,
    )


def probe_c(drive) -> dict:
    print("=" * 78)
    print("PROBE C   action-grid degeneracy: how many bins are really distinct?")
    print("=" * 78)
    nb = drive["num_bins"]
    total = nb * nb
    print(f"\n  grid is {nb} x {nb} = {total} bins")
    print(f"  slew limit on omega = alpha_max * dt = "
          f"{drive['max_angular_accel'] * drive['dt']:.2f} rad/s, "
          f"but omega_max = {drive['max_angular_velocity']:.2f} rad/s")
    print("\n  entry v | distinct v | distinct w | distinct (v,w) | duplicated")
    print("  --------+------------+------------+----------------+-----------")
    rows = []
    for v_in in (0.0, 0.2, 0.5, 0.8, 0.9, 1.0):
        lin, ang = decode_discrete_drive_action_grid(
            torch.tensor([v_in], dtype=DTYPE), torch.tensor([ROBOT_OMEGA], dtype=DTYPE),
            num_bins=nb, dt=drive["dt"],
            max_linear_velocity=drive["max_linear_velocity"],
            reverse_velocity_scale=drive["reverse_velocity_scale"],
            max_linear_accel=drive["max_linear_accel"],
            max_angular_velocity=drive["max_angular_velocity"],
            max_angular_accel=drive["max_angular_accel"],
        )
        n_v = len({round(x, 6) for x in lin[0, :, 0].tolist()})
        n_w = len({round(x, 6) for x in ang[0, 0, :].tolist()})
        pairs = {
            (round(lin[0, i, j].item(), 6), round(ang[0, i, j].item(), 6))
            for i in range(nb) for j in range(nb)
        }
        dup = 1.0 - len(pairs) / total
        print(f"  {v_in:>7.2f} | {n_v:>10d} | {n_w:>10d} | {len(pairs):>14d} | {dup:>10.1%}")
        rows.append({"entry_v_mps": v_in, "distinct_linear": n_v,
                     "distinct_angular": n_w, "distinct_pairs": len(pairs),
                     "duplicated_fraction": dup})
    print("\n  at cruise the linear axis saturates because positive acceleration is")
    print("  clamped by (v_max - v)/dt, and the angular axis saturates because the")
    print("  slew limit is smaller than omega_max.")
    return {"per_entry_speed": rows, "grid_size": total}


def probe_a(drive, spec, goals, offsets) -> dict:
    print()
    print("=" * 78)
    print("PROBE A   near-tie among PHYSICALLY DISTINCT alternatives")
    print("=" * 78)
    combos = [(d, o) for d in goals for o in offsets]
    n = len(combos)
    gd = torch.tensor([c[0] for c in combos], dtype=DTYPE)
    off = torch.tensor([c[1] for c in combos], dtype=DTYPE)
    rx = torch.full((n,), 0.5, dtype=DTYPE)
    batch = make_batch(
        rx, torch.full((n,), ROBOT_Y, dtype=DTYPE),
        torch.stack([rx + gd, torch.zeros(n, dtype=DTYPE)], dim=-1),
        torch.stack([torch.full((n,), OBSTACLE_X_M, dtype=DTYPE), off], dim=-1),
        samples=spec.samples,
    )
    res = run_teacher(batch, drive, spec)
    feas_state = res["any_feasible"]
    cost = res["cost_grid"].flatten(1)
    act = res["actions"]
    lin = res["linear_velocity_grid"].flatten(1)
    ang = res["angular_velocity_grid"].flatten(1)
    nb = drive["num_bins"]
    row = torch.arange(n)
    flat_best = (act[:, 0] * nb + act[:, 1]).clamp(0, nb * nb - 1)
    best_cost = cost.gather(1, flat_best[:, None]).squeeze(1)
    best_v = lin.gather(1, flat_best[:, None])
    best_w = ang.gather(1, flat_best[:, None])

    finite = torch.isfinite(cost)
    # physically identical => same command within tolerance
    same_cmd = ((lin - best_v).abs() <= V_TOL_MPS) & ((ang - best_w).abs() <= W_TOL_RPS)
    n_equiv = (same_cmd & finite).sum(dim=-1)

    keep = feas_state
    print(f"\n  states: {n} total, {int(keep.sum())} with a feasible action")
    print(f"  feasible bins that are physically IDENTICAL to the chosen one:")
    eq = n_equiv[keep].float()
    print(f"    median {eq.median():.1f}, mean {eq.mean():.1f}, max {int(eq.max())}")
    print("    -> cross-entropy on the bin index penalises the student for")
    print("       choosing any of these physically equivalent bins.")

    distinct = finite & ~same_cmd
    print("\n  extra cost to pick a physically DIFFERENT command that is also")
    print(f"  at least K bins away in the grid  (tol: dv>{V_TOL_MPS}, dw>{W_TOL_RPS})\n")
    print("   K | median dJ | p10 dJ  | frac dJ<0.01 | frac dJ<0.05 | n states")
    print("  ---+-----------+---------+--------------+--------------+---------")
    idx = torch.arange(nb)
    dl = (idx[None, :, None] - act[:, 0][:, None, None]).abs()
    da = (idx[None, None, :] - act[:, 1][:, None, None]).abs()
    cheb = torch.maximum(dl, da).flatten(1)
    rows = []
    for k in (1, 2, 3, 5):
        cand = distinct & (cheb >= k) & keep[:, None]
        masked = torch.where(cand, cost, torch.full_like(cost, float("inf")))
        gap = masked.amin(dim=-1) - best_cost
        ok = torch.isfinite(gap) & keep
        g = gap[ok]
        if g.numel() == 0:
            print(f"  {k:>2} |        -- |      -- |           -- |           -- | 0")
            continue
        print(f"  {k:>2} | {g.median():>9.4f} | {g.quantile(0.10):>7.4f} | "
              f"{(g < 0.01).float().mean():>12.1%} | "
              f"{(g < 0.05).float().mean():>12.1%} | {g.numel():>8d}")
        rows.append({"k_bins": k, "median_delta_cost": float(g.median()),
                     "p10_delta_cost": float(g.quantile(0.10)),
                     "frac_below_0p01": float((g < 0.01).float().mean()),
                     "frac_below_0p05": float((g < 0.05).float().mean()),
                     "n_states": int(g.numel())})
    print(f"\n  reference: clearance term max contribution "
          f"{spec.obstacle_clearance_weight * 0.75:.3f}, stall term {spec.stall_weight:.3f}")
    return {"per_k": rows,
            "equivalent_bins_median": float(eq.median()),
            "equivalent_bins_max": int(eq.max()),
            "n_states_feasible": int(keep.sum()), "n_states": n}


def probe_b(drive, spec) -> dict:
    print()
    print("=" * 78)
    print("PROBE B   label continuity along a traverse (feasible states only)")
    print("=" * 78)
    nb = drive["num_bins"]
    step = ROBOT_V * drive["dt"]
    out = {}
    for name, dx in (("one control step (0.20 m)", step), ("fine scan (0.02 m)", 0.02)):
        xs = torch.arange(0.5, 8.0 + 1e-9, dx, dtype=DTYPE)
        n = xs.numel()
        batch = make_batch(
            xs, torch.full((n,), ROBOT_Y, dtype=DTYPE),
            torch.stack([torch.full((n,), 9.0, dtype=DTYPE),
                         torch.zeros(n, dtype=DTYPE)], dim=-1),
            torch.stack([torch.full((n,), OBSTACLE_X_M, dtype=DTYPE),
                         torch.full((n,), 0.3, dtype=DTYPE)], dim=-1),
            samples=spec.samples,
        )
        res = run_teacher(batch, drive, spec)
        feas = res["any_feasible"]
        act = res["actions"]
        row = torch.arange(n)
        w = res["angular_velocity_grid"][row, act[:, 0], act[:, 1]]
        v = res["linear_velocity_grid"][row, act[:, 0], act[:, 1]]

        # Only transitions where BOTH endpoints are feasible are real label pairs.
        pair_ok = feas[1:] & feas[:-1]
        d_cheb = torch.maximum((act[1:, 1] - act[:-1, 1]).abs(),
                               (act[1:, 0] - act[:-1, 0]).abs())[pair_ok]
        d_w = (w[1:] - w[:-1]).abs()[pair_ok]
        d_v = (v[1:] - v[:-1]).abs()[pair_ok]

        print(f"\n  {name}: {n} states, feasible {int(feas.sum())}/{n}, "
              f"usable transitions {int(pair_ok.sum())}")
        if d_cheb.numel() == 0:
            print("    no usable transitions")
            continue
        print(f"    label jump in bins: max {int(d_cheb.max())}, "
              f"mean {d_cheb.float().mean():.2f}")
        for thr in (1, 2, 3, 5):
            print(f"      >= {thr} bins: {(d_cheb >= thr).float().mean():>6.1%}")
        print(f"    beyond soft-label reach (+/-{SOFT_LABEL_REACH_BINS} bin): "
              f"{(d_cheb > SOFT_LABEL_REACH_BINS).float().mean():.1%}")
        print(f"    commanded turn-rate change: max {d_w.max():.4f} rad/s, "
              f"mean {d_w.mean():.4f}")
        print(f"    commanded speed change:     max {d_v.max():.4f} m/s, "
              f"mean {d_v.mean():.4f}")
        out[name] = {
            "n_states": n, "step_m": float(dx),
            "n_feasible": int(feas.sum()), "n_usable_transitions": int(pair_ok.sum()),
            "max_jump_bins": int(d_cheb.max()),
            "mean_jump_bins": float(d_cheb.float().mean()),
            "frac_ge_2_bins": float((d_cheb >= 2).float().mean()),
            "frac_ge_3_bins": float((d_cheb >= 3).float().mean()),
            "frac_beyond_soft_reach": float(
                (d_cheb > SOFT_LABEL_REACH_BINS).float().mean()),
            "max_delta_omega_rps": float(d_w.max()),
            "max_delta_v_mps": float(d_v.max()),
            "feasible_mask": feas.tolist(),
            "angular_bin_trace": act[:, 1].tolist(),
            "linear_bin_trace": act[:, 0].tolist(),
            "robot_x_m": xs.tolist(),
        }
        if dx == step:
            print("\n    trace, F=feasible label / . = infeasible placeholder (9,9)")
            marks = "".join("F" if f else "." for f in feas.tolist())
            print(f"      {marks}")
            print("    angular bin (only F positions are real labels):")
            tr = act[:, 1].tolist()
            for i in range(0, len(tr), 12):
                print(f"      x={xs[i].item():>4.1f}  " +
                      " ".join(f"{b:>3d}" for b in tr[i:i + 12]))
    return out


def main() -> int:
    frozen = json.loads(FROZEN.read_text())
    drive = frozen["drive"]
    spec = CorridorTeacherSpec()
    c = probe_c(drive)
    a = probe_a(drive, spec, frozen["goal_distances_m"], frozen["obstacle_offsets_m"])
    b = probe_b(drive, spec)
    OUT.write_text(json.dumps({
        "provenance": {
            "purpose": "known gap #4: is the teacher argmin label CE-friendly",
            "uses_production_function": "corridor_teacher_action_grid",
            "obstacle_x_m": OBSTACLE_X_M,
            "obstacle_x_calibration":
                "exactly reproduces logs/probes/b2_clearance_sweep.json",
            "drive": drive,
            "soft_label_reach_bins": SOFT_LABEL_REACH_BINS,
            "distinctness_tolerance": {"v_mps": V_TOL_MPS, "omega_rps": W_TOL_RPS},
            "caveats": [
                "synthetic single static obstacle, fixed entry v and omega=0",
                "traverse teleports the robot; not a closed-loop rollout",
                "state-space smoothness is necessary but not sufficient for "
                "observation-space learnability",
                "centre bin 9 = zero acceleration, not stop; infeasible states "
                "emit the (9,9) placeholder and are excluded",
                "degeneracy depends on entry (v, omega); only omega=0 was swept",
            ],
        },
        "probe_c_grid_degeneracy": c,
        "probe_a_near_tie_distinct": a,
        "probe_b_continuity": b,
    }, indent=2))
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
