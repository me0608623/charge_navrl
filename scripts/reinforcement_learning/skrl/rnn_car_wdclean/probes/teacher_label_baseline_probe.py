"""Is the 53.7% learnability figure meaningful? Compare it to trivial baselines.

The stored probe datasets contain the teacher label and the policy action, but
``policy_input`` is already whitened by a RunningStandardScaler, so the physical
(v, omega) cannot be recovered and the decoded-command agreement cannot be
recomputed from this artefact. What CAN be computed exactly, with no inversion,
is the majority-class baseline for every metric the probe reported.

A classifier metric without its majority baseline is uninterpretable: if 73% of
labels are one class, "94.5% within one bin" may be worse than a constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PROBE_DIR = Path("/home/aa/IsaacLab/logs/probes")
OUT = PROBE_DIR / "teacher_label_baseline_probe.json"
DATASETS = {
    "c12_teacher_v4_corridor_2s1d": PROBE_DIR / "c12_teacher_v4_corridor_2s1d_dataset.npz",
    "dagger_round1_s515": PROBE_DIR / "dagger_round1_s515_dataset.npz",
}
# Reported in Obsidian 09_走廊攻堅_特權蒸餾_SA6轉場.md
REPORTED = {
    "joint_exact": 0.537,
    "angular_within_1": 0.945,
    "linear_direction": 0.844,
    "angular_direction": 0.778,
}
CENTRE = 9
NUM_BINS = 19
# Angular bins that collapse onto the slew boundary when omega_cur = 0
# (target |omega| > alpha_max*dt = 0.6 while omega_max = 1.2).
ANGULAR_DEGENERATE_LOW = list(range(0, 5))
ANGULAR_DEGENERATE_HIGH = list(range(14, 19))


def direction(bins: np.ndarray) -> np.ndarray:
    """-1 / 0 / +1 relative to the centre bin."""
    return np.sign(bins.astype(np.int16) - CENTRE)


def majority_rate(labels: np.ndarray) -> tuple[float, int]:
    vals, counts = np.unique(labels, return_counts=True)
    i = int(counts.argmax())
    return float(counts[i] / labels.size), int(vals[i])


def within_one_rate_for_constant(bins: np.ndarray, c: int) -> float:
    return float((np.abs(bins.astype(np.int16) - c) <= 1).mean())


def analyse(name: str, path: Path) -> dict:
    d = np.load(path)
    ta = d["teacher_action"].astype(np.int16)
    pa = d["policy_action"].astype(np.int16)
    feas = d["feasible"].astype(bool)
    n = ta.shape[0]

    print("=" * 78)
    print(f"{name}   n={n}, feasible={int(feas.sum())}")
    print("=" * 78)

    # Only feasible frames ever become supervision targets.
    ta_f, pa_f = ta[feas], pa[feas]
    nf = ta_f.shape[0]

    out: dict = {"n_total": n, "n_feasible": nf}

    print("\n-- teacher label concentration (feasible frames) --")
    for ax, axis_name in ((0, "linear"), (1, "angular")):
        rate, mode = majority_rate(ta_f[:, ax])
        w1 = within_one_rate_for_constant(ta_f[:, ax], CENTRE)
        w1_mode = within_one_rate_for_constant(ta_f[:, ax], mode)
        print(f"  {axis_name:>8}: modal bin {mode:>2} covers {rate:>6.1%}; "
              f"within +/-1 of centre {w1:>6.1%}; within +/-1 of modal {w1_mode:>6.1%}")
        out[f"{axis_name}_modal_bin"] = mode
        out[f"{axis_name}_modal_rate"] = rate
        out[f"{axis_name}_within1_of_centre"] = w1
        out[f"{axis_name}_within1_of_modal"] = w1_mode

    pairs, counts = np.unique(ta_f, axis=0, return_counts=True)
    j = int(counts.argmax())
    joint_base = float(counts[j] / nf)
    print(f"\n  modal joint label {tuple(int(v) for v in pairs[j])} covers "
          f"{joint_base:.1%}  <-- baseline for 'joint exact'")
    out["joint_modal_pair"] = [int(v) for v in pairs[j]]
    out["joint_modal_rate"] = joint_base

    print("\n-- direction-class baselines --")
    for ax, axis_name in ((0, "linear"), (1, "angular")):
        rate, mode = majority_rate(direction(ta_f[:, ax]))
        print(f"  {axis_name:>8} direction: modal class {mode:+d} covers {rate:.1%}")
        out[f"{axis_name}_direction_baseline"] = rate

    low = float(np.isin(ta_f[:, 1], ANGULAR_DEGENERATE_LOW).mean())
    high = float(np.isin(ta_f[:, 1], ANGULAR_DEGENERATE_HIGH).mean())
    print("\n-- angular slew-degenerate blocks (at omega_cur = 0) --")
    print(f"  label in bins 0-4   (all decode to -0.6 rad/s): {low:.2%}")
    print(f"  label in bins 14-18 (all decode to +0.6 rad/s): {high:.2%}")
    print(f"  combined {low + high:.2%} of labels sit inside a degenerate block")
    out["angular_degenerate_low_mass"] = low
    out["angular_degenerate_high_mass"] = high

    print("\n-- stored policy action vs teacher label (feasible frames) --")
    ex_l = float((pa_f[:, 0] == ta_f[:, 0]).mean())
    ex_a = float((pa_f[:, 1] == ta_f[:, 1]).mean())
    ex_j = float(((pa_f == ta_f).all(axis=1)).mean())
    w1_a = float((np.abs(pa_f[:, 1] - ta_f[:, 1]) <= 1).mean())
    dir_l = float((direction(pa_f[:, 0]) == direction(ta_f[:, 0])).mean())
    dir_a = float((direction(pa_f[:, 1]) == direction(ta_f[:, 1])).mean())
    print(f"  exact linear {ex_l:.1%}, exact angular {ex_a:.1%}, joint exact {ex_j:.1%}")
    print(f"  angular within +/-1 {w1_a:.1%}")
    print(f"  linear direction {dir_l:.1%}, angular direction {dir_a:.1%}")
    out["policy_vs_teacher"] = {
        "exact_linear": ex_l, "exact_angular": ex_a, "joint_exact": ex_j,
        "angular_within_1": w1_a,
        "linear_direction": dir_l, "angular_direction": dir_a,
    }
    return out


def verdict(main: dict) -> None:
    print()
    print("=" * 78)
    print("VERDICT   reported probe numbers vs trivial baselines")
    print("=" * 78)
    rows = [
        ("joint exact", REPORTED["joint_exact"], main["joint_modal_rate"],
         f"always predict {tuple(main['joint_modal_pair'])}"),
        ("angular within +/-1", REPORTED["angular_within_1"],
         main["angular_within1_of_modal"],
         f"always predict angular {main['angular_modal_bin']}"),
        ("linear direction", REPORTED["linear_direction"],
         main["linear_direction_baseline"], "always predict modal direction"),
        ("angular direction", REPORTED["angular_direction"],
         main["angular_direction_baseline"], "always predict modal direction"),
    ]
    print("\n  metric                | reported | baseline | margin  | baseline rule")
    print("  ----------------------+----------+----------+---------+---------------")
    for label, rep, base, rule in rows:
        margin = rep - base
        flag = "   <-- BELOW BASELINE" if margin < 0 else ""
        print(f"  {label:<21} | {rep:>7.1%} | {base:>7.1%} | "
              f"{margin:>+6.1%} | {rule}{flag}")
    print("\n  A metric at or below its majority baseline is no evidence that the")
    print("  student can infer the teacher's action. Only positive margin informs.")


def main() -> int:
    results = {}
    for name, path in DATASETS.items():
        if not path.is_file():
            print(f"missing: {path}")
            continue
        results[name] = analyse(name, path)
        print()
    if "c12_teacher_v4_corridor_2s1d" in results:
        verdict(results["c12_teacher_v4_corridor_2s1d"])
    OUT.write_text(json.dumps({
        "provenance": {
            "purpose": "majority-class baselines for the corridor teacher "
                       "learnability probe metrics",
            "reported_source": "Obsidian isaaclab_v4/09_走廊攻堅_特權蒸餾_SA6轉場.md",
            "reported_values": REPORTED,
            "blocked": "decoded (v, omega) agreement cannot be recomputed from "
                       "these artefacts: policy_input is already whitened by a "
                       "RunningStandardScaler, so physical current v and omega are "
                       "unrecoverable. Recomputation needs a fresh dump storing raw "
                       "current_velocity and current_omega per sample.",
            "caveats": [
                "baselines are computed on the stored dataset; the probe reported "
                "held-out numbers, so the split differs and this is indicative, "
                "not a like-for-like re-evaluation",
                "angular degenerate blocks listed for omega_cur = 0; which bins "
                "collapse moves with omega_cur, the count stays about 11/19",
                "stored policy_action is the collection-time policy, not the probe "
                "classifier",
            ],
        },
        "datasets": results,
    }, indent=2))
    print(f"\n  wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
