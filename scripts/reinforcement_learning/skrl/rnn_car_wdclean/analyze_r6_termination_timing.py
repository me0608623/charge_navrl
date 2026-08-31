"""Zero-GPU: when the dominant binder patterns appear relative to termination.

Pure post-hoc analysis of an already-collected v3 stateful-teacher-diagnostic
npz. No rollout, no simulation code path touched, no FSM/threshold/scene
change. For each completed episode this walks backward from the terminal
frame up to a fixed window (clipped at episode start) and, per relative
offset, reports the fraction of episodes whose frame at that offset exhibits
a target binder pattern -- split by terminal cause. A flat curve across the
whole window means the pattern is already fully present well before
termination (a standing condition); a curve that ramps up only near offset 0
means it develops acutely as termination approaches (a late consequence).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAUSE_NAMES = {1: "goal", 2: "wall", 3: "obstacle", 4: "timeout"}
SIDES = ("left", "right")
DIAGNOSTIC_SCHEMA = "stateful_teacher_step_diagnostic/v3"

# Must match rnn_car_wdclean.stateful_corridor_teacher.FUNNEL_MASK_NAMES.
FUNNEL_MASK_NAMES = (
    "static_obstacle",
    "dynamic_obstacle",
    "wall",
    "launch",
    "progress",
    "heading",
    "side_signal",
)
FUNNEL_MASK_BITS = {name: 1 << i for i, name in enumerate(FUNNEL_MASK_NAMES)}
_LAUNCH_PROGRESS_BITMASK = FUNNEL_MASK_BITS["launch"] | FUNNEL_MASK_BITS["progress"]


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError(
            f"expected schema {DIAGNOSTIC_SCHEMA}, got {metadata.get('schema')!r}"
        )
    return arrays


def dynamic_obstacle_recovers_a_blocked_side(arrays: dict, row: int, env: int) -> bool:
    for side in SIDES:
        blocked = arrays[f"n_final_{side}"][row, env] == 0
        recovers = arrays[f"n_without_dynamic_obstacle_{side}"][row, env] > 0
        if blocked and recovers:
            return True
    return False


def static_and_progress_both_recover_a_blocked_side(
    arrays: dict, row: int, env: int
) -> bool:
    for side in SIDES:
        blocked = arrays[f"n_final_{side}"][row, env] == 0
        static_recovers = arrays[f"n_without_static_obstacle_{side}"][row, env] > 0
        progress_recovers = arrays[f"n_without_progress_{side}"][row, env] > 0
        if blocked and static_recovers and progress_recovers:
            return True
    return False


def launch_and_progress_is_the_best_multi_filter_pair_for_a_blocked_side(
    arrays: dict, row: int, env: int
) -> bool:
    """True iff a side is blocked, no single gate alone recovers it (true
    multi_filter, not unique/ambiguous), and the best-recovering two-gate
    relaxation for that side is exactly {launch, progress}."""
    for side in SIDES:
        blocked = arrays[f"n_final_{side}"][row, env] == 0
        if not blocked:
            continue
        num_singles = sum(
            int(arrays[f"n_without_{name}_{side}"][row, env] > 0)
            for name in FUNNEL_MASK_NAMES
        )
        if num_singles != 0:
            continue
        best_count = arrays[f"pairwise_best_count_{side}"][row, env]
        bitmask = arrays[f"pairwise_best_bitmask_{side}"][row, env]
        if best_count > 0 and bitmask == _LAUNCH_PROGRESS_BITMASK:
            return True
    return False


def episode_windows(
    episode_end: np.ndarray, cause_name: str, window_steps: int
) -> list[tuple[int, int, int]]:
    """Return (terminal_row, env, usable_length) for episodes of one cause.

    ``usable_length`` is the window actually available before hitting the
    episode's own start, i.e. ``min(window_steps, episode_steps)``.
    """
    windows = []
    for terminal, env, cause, length in episode_end:
        if CAUSE_NAMES.get(int(cause)) != cause_name:
            continue
        windows.append((int(terminal), int(env), min(window_steps, int(length))))
    return windows


def timing_curve(
    arrays: dict,
    windows: list[tuple[int, int, int]],
    metric_fn,
    window_steps: int,
) -> dict[int, tuple[int, int]]:
    """curve[k] = (n_hit, n_total) at relative offset k (0=terminal, negative=earlier)."""
    curve = {k: [0, 0] for k in range(0, -window_steps, -1)}
    for terminal, env, usable in windows:
        for k in range(0, -usable, -1):
            row = terminal + k
            hit = metric_fn(arrays, row, env)
            curve[k][0] += int(hit)
            curve[k][1] += 1
    return {k: (h, t) for k, (h, t) in curve.items()}


def build_report(
    archives: dict[str, dict],
    *,
    window_steps: int = 25,
    dt_s: float = 0.2,
) -> dict:
    metrics = {
        "dynamic_obstacle_recovers": dynamic_obstacle_recovers_a_blocked_side,
        "static_and_progress_both_recover": (
            static_and_progress_both_recover_a_blocked_side
        ),
        "launch_and_progress_best_multi_filter_pair": (
            launch_and_progress_is_the_best_multi_filter_pair_for_a_blocked_side
        ),
    }
    causes = {}
    for cause_name in ("obstacle", "timeout", "goal"):
        curves_by_metric = {}
        n_episodes = 0
        for metric_name, metric_fn in metrics.items():
            curve = {k: [0, 0] for k in range(0, -window_steps, -1)}
            episodes_seen = 0
            for arrays in archives.values():
                windows = episode_windows(
                    arrays["episode_end"], cause_name, window_steps
                )
                episodes_seen += len(windows)
                for terminal, env, usable in windows:
                    for k in range(0, -usable, -1):
                        row = terminal + k
                        hit = metric_fn(arrays, row, env)
                        curve[k][0] += int(hit)
                        curve[k][1] += 1
            n_episodes = episodes_seen
            curves_by_metric[metric_name] = [
                {
                    "seconds_before_end": round(-k * dt_s, 2),
                    "n_hit": curve[k][0],
                    "n_total": curve[k][1],
                    "fraction": (curve[k][0] / curve[k][1]) if curve[k][1] else None,
                }
                for k in sorted(curve, reverse=True)
            ]
        causes[cause_name] = {
            "n_episodes": n_episodes,
            "curves": curves_by_metric,
        }
    return {
        "schema": "r6_termination_timing/v1",
        "window_steps": window_steps,
        "dt_s": dt_s,
        "metric_definitions": {
            "dynamic_obstacle_recovers": (
                "at least one side is blocked (n_final==0) and relaxing "
                "dynamic_obstacle alone would recover it"
            ),
            "static_and_progress_both_recover": (
                "at least one side is blocked and relaxing static_obstacle "
                "alone would recover it AND relaxing progress alone would "
                "also recover it (the ambiguous static+progress pair). "
                "This is static_obstacle+progress, not launch+progress -- "
                "but that scoping is limited to the unique/ambiguous "
                "classification (no single gate, or two-or-more single "
                "gates, individually recover the side). Within the separate "
                "multi_filter classification (no single gate recovers it,"
                " but a specific two-gate relaxation does), "
                "launch+progress is in fact the dominant pair for timeout-"
                "terminal frames -- see "
                "launch_and_progress_best_multi_filter_pair below"
            ),
            "launch_and_progress_best_multi_filter_pair": (
                "at least one side is blocked, no single gate alone "
                "recovers it (true multi_filter, disjoint from the "
                "ambiguous static+progress pair above), and the best-"
                "recovering two-gate relaxation for that side is exactly "
                "{launch, progress}"
            ),
        },
        "causes": causes,
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# r6 termination timing (zero-GPU, record-only)",
        "",
        "> Post-hoc analysis of the r6 npz only. No rollout, no code path "
        "touched, no FSM/threshold/scene change.",
        "",
        (
            f"Window: last {report['window_steps']} steps "
            f"({report['window_steps'] * report['dt_s']:.1f}s at "
            f"dt={report['dt_s']}s), clipped at episode start."
        ),
        "",
    ]
    for cause_name, cause in report["causes"].items():
        lines.append(f"## {cause_name}-terminal episodes (n={cause['n_episodes']})")
        lines.append("")
        for metric_name, curve in cause["curves"].items():
            lines.append(f"### {metric_name}")
            lines.append("")
            lines.append("| seconds before end | fraction |")
            lines.append("|---:|---:|")
            for point in curve:
                frac = point["fraction"]
                frac_text = f"{frac:.2%}" if frac is not None else "n/a"
                marker = " (terminal frame)" if point["seconds_before_end"] == 0 else ""
                lines.append(
                    f"| -{point['seconds_before_end']:.1f}s{marker} | {frac_text} |"
                )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, nargs="+")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--window-steps", type=int, default=25)
    args = parser.parse_args(argv)

    archives = {str(p): load_archive(p) for p in args.npz}
    report = build_report(archives, window_steps=args.window_steps)
    rendered_json = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(rendered_json, encoding="utf-8")
    else:
        print(rendered_json)
    if args.markdown_output:
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
