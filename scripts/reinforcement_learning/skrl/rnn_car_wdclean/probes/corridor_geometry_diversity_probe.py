"""How much does the long-corridor static geometry actually vary per reset?

Motivated by a GUI observation: obstacles appeared to sit in the same places on
every reset. Reading the sampler suggested a fixed lattice plus a small jitter,
but that is a code reading, not a measurement. This probe measures it.

Pure torch, no Isaac, no GPU. Calls the PRODUCTION sampler
``sample_conflict_free_layout`` (including its rejection/redraw loop), so what
is measured is the realised reset distribution, not an idealised one.

Reported:
  1. Per-slot static position statistics -> is there a fixed lattice?
  2. Mirror fraction -> how many discrete skeletons exist?
  3. Distinct-layout count after clustering at the jitter scale.
  4. Occupancy coverage: what fraction of the corridor can a static centre reach?
  5. Passable-gap topology: does the jitter ever change which side is passable?
     This is the decision-relevant one. A jitter that never flips passability
     means every episode poses the same routing problem.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

_EVENTS = (
    "/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/"
    "locomotion/velocity/config/charge_skrl/mdp/events"
)
if _EVENTS not in sys.path:
    sys.path.insert(0, _EVENTS)

from long_corridor_replay_geometry import (  # noqa: E402
    LongCorridorSpec,
    sample_conflict_free_layout,
)

OUT = Path("/home/aa/IsaacLab/logs/probes/corridor_geometry_diversity_probe.json")
N_DRAWS = 20000
# Robot footprint that must fit through a gap (OBB half-width + buffer).
ROBOT_HALF_WIDTH_M = 0.30
ROBOT_BUFFER_M = 0.10
REQUIRED_HALF_CLEARANCE_M = ROBOT_HALF_WIDTH_M + ROBOT_BUFFER_M


def analyse(free_width: float, mode: str, n_static: int, n_dynamic: int) -> dict:
    spec = LongCorridorSpec(free_width=free_width)
    torch.manual_seed(0)
    static, dynamic, waypoints, motion_types, _ = sample_conflict_free_layout(
        N_DRAWS, spec, torch.device("cpu"), mode,
        static_obstacles=n_static, dynamic_obstacles=n_dynamic,
    )
    s = static[:, :n_static]          # [N, n_static, 2]
    d = dynamic[:, :n_dynamic]

    inner = spec.inner_half_width
    r = spec.obstacle_radius

    print("=" * 78)
    print(f"free_width={free_width} m  mode={mode}  {n_static}S+{n_dynamic}D  "
          f"N={N_DRAWS}")
    print(f"  inner half width = +/-{inner:.3f} m, obstacle radius = {r:.2f} m")
    print("=" * 78)

    print("\n-- static slots: is there a fixed lattice? --")
    print("  slot |      x range      |      y range      | uniq|x| | uniq y")
    print("  -----+-------------------+-------------------+---------+-------")
    slots = []
    for i in range(n_static):
        x, y = s[:, i, 0], s[:, i, 1]
        # cluster at 0.25 m: coarser than the 0.10 m jitter, finer than lattice
        ux = len({round(v, 2) for v in (x.abs() / 0.25).round().mul(0.25).tolist()})
        uy = len({round(v, 2) for v in (y / 0.25).round().mul(0.25).tolist()})
        print(f"  {i:>4} | {x.min():+7.3f} .. {x.max():+7.3f} | "
              f"{y.min():+7.3f} .. {y.max():+7.3f} | {ux:>7} | {uy:>6}")
        slots.append({"slot": i,
                      "x_min": float(x.min()), "x_max": float(x.max()),
                      "y_min": float(y.min()), "y_max": float(y.max()),
                      "abs_x_clusters": ux, "y_clusters": uy})

    # Mirror = sign pattern of the x column across slots.
    sign_key = ["".join("+" if v > 0 else "-" for v in row)
                for row in s[:, :, 0].sign().tolist()]
    patterns = Counter(sign_key)
    print(f"\n-- discrete skeletons (sign pattern of static x) --")
    for pat, c in patterns.most_common():
        print(f"  {pat}  {c:>6}  ({c / N_DRAWS:6.2%})")
    print(f"  distinct sign patterns: {len(patterns)}")

    # Distinct layouts if we cluster at the jitter scale.
    jit = spec.static_xy_jitter
    grid = (s / max(jit, 1e-6)).round().to(torch.int32)
    layout_keys = {tuple(row.flatten().tolist()) for row in grid}
    print(f"\n-- distinct layouts clustered at the jitter scale "
          f"({jit:.2f} m) --")
    print(f"  {len(layout_keys)} distinct out of {N_DRAWS} draws")

    # Coverage: area a static centre can reach vs corridor free area.
    corridor_area = (2 * inner) * spec.length
    reach = 0.0
    for i in range(n_static):
        x, y = s[:, i, 0], s[:, i, 1]
        # two x lobes (mirror), so measure each sign separately then sum
        for sgn in (-1.0, 1.0):
            m = (x.sign() == sgn)
            if not bool(m.any()):
                continue
            reach += float((x[m].max() - x[m].min()) * (y[m].max() - y[m].min()))
    print(f"\n-- occupancy coverage --")
    print(f"  corridor free area           = {corridor_area:8.2f} m^2")
    print(f"  area static centres can hit  = {reach:8.2f} m^2  "
          f"({reach / corridor_area:.2%})")

    # Passable-gap topology per static row.
    print("\n-- passable-gap topology (the decision-relevant test) --")
    print(f"  a gap is passable if half-width >= {REQUIRED_HALF_CLEARANCE_M:.2f} m "
          f"(robot {ROBOT_HALF_WIDTH_M} + buffer {ROBOT_BUFFER_M})")
    print("\n  slot | left gap (m)      | right gap (m)     | left ok | right ok | flips")
    print("  -----+-------------------+-------------------+---------+----------+------")
    topo = []
    for i in range(n_static):
        x = s[:, i, 0]
        left = (x - r) - (-inner)      # wall(-inner) .. obstacle left edge
        right = inner - (x + r)        # obstacle right edge .. wall(+inner)
        lok = left >= 2 * REQUIRED_HALF_CLEARANCE_M
        rok = right >= 2 * REQUIRED_HALF_CLEARANCE_M
        # does passability ever differ within one mirror side?
        flips = 0
        for sgn in (-1.0, 1.0):
            m = x.sign() == sgn
            if bool(m.any()):
                if len({bool(v) for v in lok[m].tolist()}) > 1: flips += 1
                if len({bool(v) for v in rok[m].tolist()}) > 1: flips += 1
        print(f"  {i:>4} | {left.min():6.3f} .. {left.max():6.3f}   | "
              f"{right.min():6.3f} .. {right.max():6.3f}   | "
              f"{lok.float().mean():>6.1%}  | {rok.float().mean():>7.1%}  | {flips:>5}")
        topo.append({"slot": i,
                     "left_gap_min": float(left.min()), "left_gap_max": float(left.max()),
                     "right_gap_min": float(right.min()), "right_gap_max": float(right.max()),
                     "left_passable_fraction": float(lok.float().mean()),
                     "right_passable_fraction": float(rok.float().mean()),
                     "passability_flips_within_mirror": flips})

    print("\n-- dynamic obstacles --")
    for i in range(n_dynamic):
        x, y = d[:, i, 0], d[:, i, 1]
        uy = len({round(v, 2) for v in (y.abs() / 0.25).round().mul(0.25).tolist()})
        print(f"  slot {i}: x {x.min():+7.3f}..{x.max():+7.3f}  "
              f"y {y.min():+7.3f}..{y.max():+7.3f}  uniq|y| clusters {uy}")
    mt = Counter(motion_types.flatten().tolist())
    print(f"  motion type counts: {dict(mt)}")

    return {
        "free_width_m": free_width, "mode": mode,
        "n_static": n_static, "n_dynamic": n_dynamic, "n_draws": N_DRAWS,
        "spec": {"static_x": spec.static_x, "static_y": list(spec.static_y),
                 "dynamic_y": list(spec.dynamic_y),
                 "static_xy_jitter": spec.static_xy_jitter,
                 "obstacle_radius": spec.obstacle_radius,
                 "inner_half_width": inner, "length": spec.length},
        "static_slots": slots,
        "sign_patterns": dict(patterns),
        "distinct_sign_patterns": len(patterns),
        "distinct_layouts_at_jitter_scale": len(layout_keys),
        "corridor_free_area_m2": corridor_area,
        "static_reachable_area_m2": reach,
        "coverage_fraction": reach / corridor_area,
        "gap_topology": topo,
    }


def main() -> int:
    # (free_width, static, dynamic). The first row is the density SA4 stage-4
    # actually evaluates -- confirmed from a realised screen cell
    # (configured_static_obstacles=3, free_width 4.4 m). 4S rows are kept only
    # as a contrast; an earlier version of this probe reported 4S numbers as if
    # they described SA4, which was wrong.
    cells = (
        (4.4, 3, 2),   # <-- SA4 stage-4 as evaluated
        (4.0, 3, 2),
        (4.4, 4, 2),
        (4.0, 4, 2),
    )
    results = {}
    for fw, ns, nd in cells:
        for mode in ("lateral", "longitudinal"):
            key = f"fw{fw}_{ns}S{nd}D_{mode}"
            results[key] = analyse(fw, mode, ns, nd)
            print()
    OUT.write_text(json.dumps({
        "provenance": {
            "purpose": "measure realised per-reset diversity of long-corridor "
                       "static geometry",
            "uses_production_sampler": "sample_conflict_free_layout "
                                       "(including rejection/redraw loop)",
            "robot_required_half_clearance_m": REQUIRED_HALF_CLEARANCE_M,
            "caveats": [
                "SA4 stage-4 evaluates 3S+2D at free_width 4.4 m (confirmed "
                "from a realised screen cell); the 4S rows are contrast only",
                "measures the sampler, not a realised training run; scene "
                "selection/curriculum shares are not modelled here",
                "goal placement, dynamic speeds, patrol waypoints and pause "
                "steps vary independently and are only partly summarised",
                "gap topology uses a straight-line half-width test, not the "
                "full OBB swept-path feasibility used by the teacher",
            ],
        },
        "cells": results,
    }, indent=2))
    print(f"  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
