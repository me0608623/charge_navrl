"""Analyze v3 feasibility-mask-decomposition stateful-teacher diagnostics.

This is the second-layer analyzer for the r6 candidate-funnel decomposition:
obstacle collision split into static/dynamic, and the kinematic filter split
into launch/progress/heading/side-signal. It refuses to read the r3-r5
schemas (v1/v2), which do not carry these fields, instead of silently
misinterpreting them.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


SCENARIOS = ("lateral_lateral", "mixed")
CAUSE_NAMES = {1: "goal", 2: "wall", 3: "obstacle", 4: "timeout"}
STATE_NAMES = {0: "WAIT", 1: "COMMIT_SIDE", 2: "PASS"}
SIDES = ("left", "right")

DIAGNOSTIC_SCHEMA = "stateful_teacher_step_diagnostic/v3"
REPORT_SCHEMA = "feasibility_mask_decomposition/v1"

# Must match rnn_car_wdclean.stateful_corridor_teacher.FUNNEL_MASK_NAMES.
# Duplicated (not imported) so this analyzer only depends on the recorded
# NPZ contract, not on importing simulation code.
FUNNEL_MASK_NAMES = (
    "static_obstacle",
    "dynamic_obstacle",
    "wall",
    "launch",
    "progress",
    "heading",
    "side_signal",
)

_CORE_ARRAYS = {
    "lateral_m",
    "longitudinal_m",
    "state",
    "committed_side",
    "committed_valid",
    "left_valid",
    "right_valid",
    "used_wait",
    "emergency_brake",
    "interaction_active",
    "applied_v_mps",
    "applied_omega_rps",
    "episode_step",
    "pred_err_1step_m",
    "pred_err_5step_m",
    "robot_yaw_rad",
    "nearest_dyn_slot",
    "nearest_dyn_bearing_rad",
    "nearest_dyn_distance_m",
}
_FUNNEL_SCALAR_ARRAYS = {
    "n_raw_left",
    "n_raw_right",
    "n_final_left",
    "n_final_right",
    "reachable_linear_mps",
    "reachable_progress_m",
    "reachable_lateral_m",
    "launch_threshold_mps",
    "progress_threshold_m",
    "side_threshold_m",
    "pairwise_recovering_count_left",
    "pairwise_recovering_count_right",
    "pairwise_best_count_left",
    "pairwise_best_count_right",
    "pairwise_best_bitmask_left",
    "pairwise_best_bitmask_right",
}
_FUNNEL_WITHOUT_ARRAYS = {
    f"n_without_{name}_{side}" for name in FUNNEL_MASK_NAMES for side in SIDES
}
REQUIRED_ARRAYS = (
    _CORE_ARRAYS
    | _FUNNEL_SCALAR_ARRAYS
    | _FUNNEL_WITHOUT_ARRAYS
    | {"episode_end", "metadata_json"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        files = set(archive.files)
        missing = REQUIRED_ARRAYS - files
        extra = files - REQUIRED_ARRAYS
        if missing or extra:
            raise ValueError(
                f"archive does not match {DIAGNOSTIC_SCHEMA}: "
                f"missing={sorted(missing)} extra={sorted(extra)}. "
                "This analyzer only reads v3 feasibility-mask-decomposition "
                "archives; it refuses to guess field meanings for v1/v2."
            )
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(archive["metadata_json"]))
    schema = metadata.get("schema")
    if schema != DIAGNOSTIC_SCHEMA:
        raise ValueError(
            f"expected schema {DIAGNOSTIC_SCHEMA} (v3); refusing to silently "
            f"reinterpret a differently-versioned archive as v3, got "
            f"{schema!r}"
        )
    return arrays, metadata


def classify_candidates_cleared(
    n_final: np.ndarray,
    n_without: dict[str, np.ndarray],
    pairwise_recovering_count: np.ndarray,
) -> np.ndarray:
    """Classify why a side's passage candidates are (or are not) empty.

    Returns an int8 code array shaped like ``n_final``:
      0 = has_candidates    (n_final != 0)
      1 = unique binder     (exactly one of the 7 leave-one-outs recovers it)
      2 = ambiguous binders (two or more leave-one-outs each recover it)
      3 = multi_filter      (no single leave-one-out recovers it, but some
                              pair does -- see the pairwise_* fields)
      4 = no_recovery_with_up_to_two_relaxations (neither a single filter
                              nor any of the 21 pairs recovers it; three or
                              more gates may jointly bind -- this code does
                              NOT claim the grid or scene has no solution,
                              only that relaxing one or two gates at a time
                              was tried and found insufficient)
    """
    cleared = n_final == 0
    single_recovers = np.stack(
        [n_without[name] > 0 for name in FUNNEL_MASK_NAMES], axis=0
    )
    num_single_recoverers = single_recovers.sum(axis=0)
    code = np.zeros(np.asarray(n_final).shape, dtype=np.int8)
    code[cleared & (num_single_recoverers == 1)] = 1
    code[cleared & (num_single_recoverers >= 2)] = 2
    code[
        cleared
        & (num_single_recoverers == 0)
        & (pairwise_recovering_count > 0)
    ] = 3
    code[
        cleared
        & (num_single_recoverers == 0)
        & (pairwise_recovering_count == 0)
    ] = 4
    return code


def _binder_bucket(
    arrays: dict[str, np.ndarray],
    side: str,
    subset: np.ndarray,
    denom_name: str,
) -> dict:
    n_final = arrays[f"n_final_{side}"]
    n_without = {
        name: arrays[f"n_without_{name}_{side}"] for name in FUNNEL_MASK_NAMES
    }
    pairwise = arrays[f"pairwise_recovering_count_{side}"]
    code = classify_candidates_cleared(n_final, n_without, pairwise)
    code_in_subset = code[subset]
    denom = int(subset.sum())
    fraction_key = f"fraction_of_{denom_name}"

    def _rate(count: int) -> float:
        return (count / denom) if denom else 0.0

    unique_by_mask = {}
    single_recovers = np.stack(
        [n_without[name] > 0 for name in FUNNEL_MASK_NAMES], axis=0
    )
    for i, name in enumerate(FUNNEL_MASK_NAMES):
        unique_mask = subset & (code == 1) & single_recovers[i]
        unique_by_mask[name] = int(unique_mask.sum())

    involved_mask_counts = {}
    for i, name in enumerate(FUNNEL_MASK_NAMES):
        involved_mask = subset & (code == 2) & single_recovers[i]
        involved_mask_counts[name] = int(involved_mask.sum())

    has_candidates_count = int((code_in_subset == 0).sum())
    unique_count = int((code_in_subset == 1).sum())
    ambiguous_count = int((code_in_subset == 2).sum())
    multi_filter_count = int((code_in_subset == 3).sum())
    no_recovery_with_up_to_two_relaxations_count = int(
        (code_in_subset == 4).sum()
    )

    return {
        "denominator": denom,
        "denominator_name": denom_name,
        "has_candidates": {
            "count": has_candidates_count,
            fraction_key: _rate(has_candidates_count),
        },
        "unique": {
            "count": unique_count,
            fraction_key: _rate(unique_count),
            "by_mask": unique_by_mask,
        },
        "ambiguous": {
            "count": ambiguous_count,
            fraction_key: _rate(ambiguous_count),
            "involved_mask_counts": involved_mask_counts,
        },
        "multi_filter": {
            "count": multi_filter_count,
            fraction_key: _rate(multi_filter_count),
        },
        "no_recovery_with_up_to_two_relaxations": {
            "count": no_recovery_with_up_to_two_relaxations_count,
            fraction_key: _rate(no_recovery_with_up_to_two_relaxations_count),
        },
    }


def _denominator_masks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    shape = arrays["state"].shape
    all_frames = np.ones(shape, dtype=bool)
    interaction_active = arrays["interaction_active"].astype(bool)
    committed = arrays["committed_side"] != 0
    committed_invalid = committed & ~arrays["committed_valid"]
    both_sides_blocked = (
        committed_invalid & ~arrays["left_valid"] & ~arrays["right_valid"]
    )
    masks = {
        "all_frames": all_frames,
        "interaction_active_frames": interaction_active,
        "committed_invalid_frames": committed_invalid,
        "both_sides_blocked_frames": both_sides_blocked,
    }
    ends = arrays["episode_end"]
    terminal_step = ends[:, 0].astype(np.int64)
    env_id = ends[:, 1].astype(np.int64)
    cause = ends[:, 2].astype(np.int64)
    for code, name in CAUSE_NAMES.items():
        terminal_mask = np.zeros(shape, dtype=bool)
        rows = terminal_step[cause == code]
        cols = env_id[cause == code]
        terminal_mask[rows, cols] = True
        masks[f"terminal_{name}_frames"] = terminal_mask
    return masks


def analyze_cell(path: Path) -> dict:
    arrays, metadata = _load_archive(path)
    denom_masks = _denominator_masks(arrays)

    denominators = {name: int(mask.sum()) for name, mask in denom_masks.items()}
    ends = arrays["episode_end"]
    cause = ends[:, 2].astype(np.int64)
    terminal_frames_by_cause = {
        name: int((cause == code).sum()) for code, name in CAUSE_NAMES.items()
    }

    binder_breakdown = {}
    for side in SIDES:
        per_denom = {}
        for denom_name, mask in denom_masks.items():
            per_denom[denom_name] = _binder_bucket(
                arrays, side, mask, denom_name
            )
        binder_breakdown[side] = per_denom

    return {
        "source_npz": str(path.resolve()),
        "metadata": metadata,
        "denominators": {
            "all_frames": denominators["all_frames"],
            "interaction_active_frames": denominators[
                "interaction_active_frames"
            ],
            "committed_invalid_frames": denominators[
                "committed_invalid_frames"
            ],
            "both_sides_blocked_frames": denominators[
                "both_sides_blocked_frames"
            ],
            "terminal_frames_by_cause": terminal_frames_by_cause,
            "completed_episodes": int(ends.shape[0]),
        },
        "binder_breakdown": binder_breakdown,
    }


def analyze_suite(suite_dir: Path, reference_dir: Path) -> dict:
    manifest_path = suite_dir / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE":
        raise ValueError("suite is not complete valid diagnostic evidence")
    before = suite_dir / "source_fingerprint_before.json"
    after = suite_dir / "source_fingerprint_after.json"
    if before.read_bytes() != after.read_bytes():
        raise ValueError("source fingerprint drifted during the r6 suite")

    equivalence = {}
    cells = {}
    for scenario in SCENARIOS:
        compared = {}
        for suffix in ("teacher.json", "corridor.json", "cell.json"):
            current = suite_dir / f"{scenario}_{suffix}"
            reference = reference_dir / f"{scenario}_{suffix}"
            current_hash = _sha256(current)
            reference_hash = _sha256(reference)
            if current_hash != reference_hash:
                raise ValueError(
                    "r5/r6 behavior-equivalence failed for "
                    f"{scenario}_{suffix}"
                )
            compared[suffix] = {
                "sha256": current_hash,
                "byte_identical": True,
            }
        equivalence[scenario] = compared
        cells[scenario] = analyze_cell(
            suite_dir / f"{scenario}_stateful_diag.npz"
        )
        expected_episodes = int(
            manifest["decision"]["cells"][scenario]["episodes"]
        )
        observed_episodes = cells[scenario]["denominators"][
            "completed_episodes"
        ]
        if observed_episodes != expected_episodes:
            raise ValueError(
                f"{scenario} NPZ episodes {observed_episodes} != "
                f"manifest episodes {expected_episodes}"
            )

    return {
        "schema": REPORT_SCHEMA,
        "suite_dir": str(suite_dir.resolve()),
        "reference_suite_dir": str(reference_dir.resolve()),
        "status": manifest["status"],
        "protocol_sha256": manifest["protocol_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "teacher_pass": bool(manifest["decision"]["teacher_pass"]),
        "training_authorized": bool(manifest["decision"]["training_authorized"]),
        "distillation_authorized": bool(
            manifest["decision"]["distillation_authorized"]
        ),
        "sa6_authorized": bool(manifest["decision"]["sa6_authorized"]),
        "source_fingerprint_stable": True,
        "r5_r6_behavior_equivalence": equivalence,
        "cells": cells,
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: dict) -> str:
    lines = [
        "# Feasibility-mask decomposition (r6)",
        "",
        "> [!important] Scope",
        "> Record-only diagnostic. r6 is behavior-equivalent to r5 -- same",
        "> teacher actions, same protocol SHA, same aggregate outputs -- with",
        "> obstacle collision split into static/dynamic and the kinematic",
        "> filter split into launch/progress/heading/side-signal.",
        "",
        "## Validity",
        "",
        f"- Status: `{report['status']}`",
        f"- Protocol SHA-256: `{report['protocol_sha256']}`",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256']}`",
    ]
    equivalence = report.get("r5_r6_behavior_equivalence", {})
    mismatches = [
        f"{scenario}_{suffix}"
        for scenario, files in equivalence.items()
        for suffix, entry in files.items()
        if not entry.get("byte_identical", True)
    ]
    if not equivalence:
        lines.append("- r5/r6 behavior equivalence: not evaluated")
    elif not mismatches:
        lines.append(
            "- r5/r6 behavior equivalence: all teacher/corridor/cell JSON "
            "byte-identical"
        )
    else:
        lines.append(
            "- r5/r6 behavior equivalence: **NOT byte-identical** -- "
            + ", ".join(mismatches)
        )
        note = report.get("r5_r6_equivalence_note")
        if note:
            lines.append(f"  - {note}")
    fingerprint_stable = report.get(
        "source_fingerprint_stable_within_r6_run",
        report.get("source_fingerprint_stable"),
    )
    lines.append(
        f"- Source fingerprint stable across the r6 suite: {fingerprint_stable}"
    )
    incident = report.get("run_incident")
    if incident:
        lines.append(f"- Run incident: {incident}")
    lines.append("")
    for scenario in sorted(report["cells"]):
        cell = report["cells"][scenario]
        denom = cell["denominators"]
        lines.extend(
            [
                f"## {scenario}",
                "",
                (
                    f"Denominators -- all frames: {denom['all_frames']:,}; "
                    f"interaction-active: {denom['interaction_active_frames']:,}; "
                    f"committed-invalid: {denom['committed_invalid_frames']:,}; "
                    f"both-sides-blocked: {denom['both_sides_blocked_frames']:,}; "
                    f"completed episodes: {denom['completed_episodes']:,}"
                ),
                (
                    "Terminal frames by cause: " + ", ".join(
                        f"{name}={count:,}"
                        for name, count in denom["terminal_frames_by_cause"].items()
                    )
                ),
                "",
            ]
        )
        for side in SIDES:
            lines.append(f"### {side}, both-sides-blocked frames")
            lines.append("")
            lines.append(
                "| bucket | count | fraction | dominant mask |"
            )
            lines.append("|---|---:|---:|---|")
            bucket = cell["binder_breakdown"][side]["both_sides_blocked_frames"]
            for name in (
                "unique",
                "ambiguous",
                "multi_filter",
                "no_recovery_with_up_to_two_relaxations",
            ):
                entry = bucket[name]
                fraction_key = next(
                    k for k in entry if k.startswith("fraction_of_")
                )
                dominant = "-"
                if name == "unique" and entry["by_mask"]:
                    dominant = max(entry["by_mask"].items(), key=lambda kv: kv[1])[0]
                elif name == "ambiguous" and entry["involved_mask_counts"]:
                    dominant = max(
                        entry["involved_mask_counts"].items(),
                        key=lambda kv: kv[1],
                    )[0]
                lines.append(
                    f"| {name} | {entry['count']:,} | "
                    f"{_pct(entry[fraction_key])} | {dominant} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Decision",
            "",
            f"- Teacher pass: **{report['teacher_pass']}**",
            f"- Training authorized: **{report['training_authorized']}**",
            f"- Distillation authorized: **{report['distillation_authorized']}**",
            f"- SA6 authorized: **{report['sa6_authorized']}**",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir", type=Path)
    parser.add_argument("reference_suite_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args(argv)

    report = analyze_suite(args.suite_dir, args.reference_suite_dir)
    rendered_json = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.write_text(rendered_json, encoding="utf-8")
    else:
        print(rendered_json)
    if args.markdown_output:
        args.markdown_output.write_text(
            render_markdown(report), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
