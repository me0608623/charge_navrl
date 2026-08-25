"""Validate and summarize stateful-teacher per-step diagnostic archives."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


SCENARIOS = ("lateral_lateral", "mixed")
CAUSE_NAMES = {1: "goal", 2: "wall", 3: "obstacle", 4: "timeout"}
STATE_NAMES = {0: "WAIT", 1: "COMMIT_SIDE", 2: "PASS"}
DIAGNOSTIC_SCHEMA = "stateful_teacher_step_diagnostic/v1"
REPORT_SCHEMA = "stateful_teacher_step_diagnostic_analysis/v1"
REQUIRED_ARRAYS = {
    "applied_omega_rps",
    "applied_v_mps",
    "committed_side",
    "committed_valid",
    "emergency_brake",
    "episode_end",
    "episode_step",
    "interaction_active",
    "lateral_m",
    "left_valid",
    "longitudinal_m",
    "metadata_json",
    "pred_err_1step_m",
    "pred_err_5step_m",
    "right_valid",
    "state",
    "used_wait",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: np.ndarray | list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
        }
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _load_archive(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        extra = set(archive.files) - REQUIRED_ARRAYS
        if missing or extra:
            raise ValueError(
                f"unexpected diagnostic keys: missing={sorted(missing)} "
                f"extra={sorted(extra)}"
            )
        arrays = {
            key: np.asarray(archive[key])
            for key in archive.files
            if key != "metadata_json"
        }
        metadata = json.loads(str(archive["metadata_json"]))
    if metadata.get("schema") != DIAGNOSTIC_SCHEMA:
        raise ValueError(
            f"expected {DIAGNOSTIC_SCHEMA}, got {metadata.get('schema')!r}"
        )
    return arrays, metadata


def _validate_archive(arrays: dict[str, np.ndarray]) -> dict:
    state = arrays["state"]
    if state.ndim != 2:
        raise ValueError("state must have [rollout_step, env] layout")
    steps, envs = state.shape
    for key in (
        "applied_omega_rps",
        "applied_v_mps",
        "committed_side",
        "committed_valid",
        "emergency_brake",
        "episode_step",
        "interaction_active",
        "lateral_m",
        "left_valid",
        "longitudinal_m",
        "right_valid",
        "used_wait",
    ):
        if arrays[key].shape != (steps, envs):
            raise ValueError(f"{key} shape does not match state")
    for key in ("pred_err_1step_m", "pred_err_5step_m"):
        if arrays[key].ndim != 3 or arrays[key].shape[:2] != (steps, envs):
            raise ValueError(f"{key} must have [rollout_step, env, slot] layout")

    ends = arrays["episode_end"]
    if ends.ndim != 2 or ends.shape[1] != 4:
        raise ValueError("episode_end must have shape [completed_episode, 4]")
    if ends.size == 0:
        raise ValueError("diagnostic contains no completed episodes")
    terminal_step = ends[:, 0].astype(np.int64)
    env_id = ends[:, 1].astype(np.int64)
    cause = ends[:, 2].astype(np.int64)
    episode_steps = ends[:, 3].astype(np.int64)
    if (
        (terminal_step < 0).any()
        or (terminal_step >= steps).any()
        or (env_id < 0).any()
        or (env_id >= envs).any()
        or (episode_steps <= 0).any()
        or not set(np.unique(cause)).issubset(CAUSE_NAMES)
    ):
        raise ValueError("episode_end contains an out-of-range value")
    recorded_step = arrays["episode_step"][terminal_step, env_id]
    if not np.array_equal(recorded_step.astype(np.int64), episode_steps - 1):
        raise ValueError("episode_end is not aligned to the terminal action row")

    checked_sequences = 0
    next_reset_rows = 0
    for terminal, env, _, length in ends:
        start = int(terminal) - (int(length) - 1)
        if start < 0:
            raise ValueError("episode starts before the diagnostic archive")
        sequence = arrays["episode_step"][start : int(terminal) + 1, int(env)]
        expected = np.arange(int(length), dtype=sequence.dtype)
        if not np.array_equal(sequence, expected):
            raise ValueError("episode_step is not contiguous within an episode")
        checked_sequences += 1
        if int(terminal) + 1 < steps:
            if arrays["episode_step"][int(terminal) + 1, int(env)] != 0:
                raise ValueError("episode_step did not reset after termination")
            next_reset_rows += 1
    return {
        "rollout_steps": int(steps),
        "envs": int(envs),
        "completed_episodes": int(len(ends)),
        "episode_sequences_checked": checked_sequences,
        "next_reset_rows_checked": next_reset_rows,
        "terminal_alignment_ok": True,
    }


def _failure_masks(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    side = arrays["committed_side"]
    committed = side != 0
    opposite_valid = np.where(
        side > 0,
        arrays["right_valid"],
        np.where(side < 0, arrays["left_valid"], False),
    )
    committed_invalid = committed & ~arrays["committed_valid"]
    no_switch = committed_invalid & opposite_valid
    both_blocked = (
        committed_invalid & ~arrays["left_valid"] & ~arrays["right_valid"]
    )
    if not np.array_equal(committed_invalid, no_switch | both_blocked):
        raise ValueError("committed-invalid frames do not reconcile")
    if not np.all(arrays["used_wait"][committed_invalid]):
        raise ValueError("committed-invalid frame did not use the wait fallback")
    return {
        "committed": committed,
        "committed_invalid": committed_invalid,
        "no_switch_other_open": no_switch,
        "both_sides_blocked": both_blocked,
    }


def _episode_outcomes(
    arrays: dict[str, np.ndarray], masks: dict[str, np.ndarray]
) -> dict:
    result = {}
    for cause_code, cause_name in CAUSE_NAMES.items():
        ends = arrays["episode_end"][arrays["episode_end"][:, 2] == cause_code]
        lateral = []
        longitudinal = []
        terminal_speed = []
        categories: Counter[str] = Counter()
        terminal_categories: Counter[str] = Counter()
        final_states: Counter[str] = Counter()
        episodes_with_no_switch = 0
        episodes_with_both_blocked = 0
        for terminal, env, _, length in ends:
            terminal = int(terminal)
            env = int(env)
            start = terminal - (int(length) - 1)
            no_switch = masks["no_switch_other_open"][start : terminal + 1, env]
            both_blocked = masks["both_sides_blocked"][start : terminal + 1, env]
            has_no_switch = bool(no_switch.any())
            has_both_blocked = bool(both_blocked.any())
            episodes_with_no_switch += int(has_no_switch)
            episodes_with_both_blocked += int(has_both_blocked)
            categories[
                ("N" if has_no_switch else "-")
                + ("B" if has_both_blocked else "-")
            ] += 1
            if masks["no_switch_other_open"][terminal, env]:
                terminal_categories["no_switch_other_open"] += 1
            elif masks["both_sides_blocked"][terminal, env]:
                terminal_categories["both_sides_blocked"] += 1
            else:
                terminal_categories["other"] += 1
            state_code = int(arrays["state"][terminal, env])
            final_states[STATE_NAMES.get(state_code, f"UNKNOWN_{state_code}")] += 1
            lateral.append(float(arrays["lateral_m"][terminal, env]))
            longitudinal.append(float(arrays["longitudinal_m"][terminal, env]))
            terminal_speed.append(float(arrays["applied_v_mps"][terminal, env]))

        lateral_array = np.asarray(lateral, dtype=np.float64)
        count = int(len(ends))
        result[cause_name] = {
            "episodes": count,
            "final_state_counts": dict(final_states),
            "episode_failure_classes": dict(categories),
            "episodes_with_no_switch_other_open": episodes_with_no_switch,
            "episodes_with_both_sides_blocked": episodes_with_both_blocked,
            "terminal_failure_classes": dict(terminal_categories),
            "terminal_lateral_m": _distribution(lateral_array),
            "terminal_abs_lateral_m": _distribution(np.abs(lateral_array)),
            "terminal_longitudinal_m": _distribution(longitudinal),
            "terminal_applied_v_mps": _distribution(terminal_speed),
            "terminal_near_center_abs_x_lt_0p5": {
                "count": int((np.abs(lateral_array) < 0.5).sum()),
                "fraction": (
                    float((np.abs(lateral_array) < 0.5).mean()) if count else 0.0
                ),
            },
            "terminal_near_wall_abs_x_gt_1p5": {
                "count": int((np.abs(lateral_array) > 1.5).sum()),
                "fraction": (
                    float((np.abs(lateral_array) > 1.5).mean()) if count else 0.0
                ),
            },
        }
    return result


def _prediction_summary(arrays: dict[str, np.ndarray]) -> dict:
    obstacle_ends = arrays["episode_end"][arrays["episode_end"][:, 2] == 3]
    result = {}
    for lag_steps, key in ((1, "pred_err_1step_m"), (5, "pred_err_5step_m")):
        values = arrays[key]
        terminal_values = []
        for terminal, env, _, _ in obstacle_ends:
            row = values[int(terminal), int(env)]
            terminal_values.extend(row[np.isfinite(row)].tolist())
        result[key] = {
            "lag_steps": lag_steps,
            "lag_s": 0.2 * lag_steps,
            "all_dynamic_slot_samples_m": _distribution(
                values[np.isfinite(values)]
            ),
            "obstacle_collision_terminal_dynamic_slot_samples_m": _distribution(
                terminal_values
            ),
        }
    return result


def analyze_cell(path: Path) -> dict:
    arrays, metadata = _load_archive(path)
    validation = _validate_archive(arrays)
    masks = _failure_masks(arrays)
    total_frames = int(arrays["state"].size)
    invalid_frames = int(masks["committed_invalid"].sum())
    no_switch_frames = int(masks["no_switch_other_open"].sum())
    both_blocked_frames = int(masks["both_sides_blocked"].sum())
    state_breakdown = {}
    for code, name in STATE_NAMES.items():
        selected = arrays["state"] == code
        state_breakdown[name] = {
            "frames": int(selected.sum()),
            "committed_invalid": int(
                (selected & masks["committed_invalid"]).sum()
            ),
            "no_switch_other_open": int(
                (selected & masks["no_switch_other_open"]).sum()
            ),
            "both_sides_blocked": int(
                (selected & masks["both_sides_blocked"]).sum()
            ),
        }
    return {
        "source_npz": str(path.resolve()),
        "metadata": metadata,
        "validation": validation,
        "frame_accounting": {
            "total_frames": total_frames,
            "committed_frames": int(masks["committed"].sum()),
            "committed_invalid_frames": invalid_frames,
            "committed_invalid_fraction_of_all_frames": invalid_frames
            / total_frames,
            "no_switch_other_open_frames": no_switch_frames,
            "no_switch_fraction_of_committed_invalid": no_switch_frames
            / invalid_frames,
            "both_sides_blocked_frames": both_blocked_frames,
            "both_sides_blocked_fraction_of_committed_invalid": (
                both_blocked_frames / invalid_frames
            ),
            "state_breakdown": state_breakdown,
            "reconciliation_ok": True,
        },
        "episode_outcomes": _episode_outcomes(arrays, masks),
        "prediction_error": _prediction_summary(arrays),
    }


def analyze_suite(suite_dir: Path, reference_dir: Path) -> dict:
    manifest_path = suite_dir / "suite_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE":
        raise ValueError("suite is not complete valid diagnostic evidence")
    before = suite_dir / "source_fingerprint_before.json"
    after = suite_dir / "source_fingerprint_after.json"
    if before.read_bytes() != after.read_bytes():
        raise ValueError("source fingerprint drifted during the suite")

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
                    f"behavior-equivalence failed for {scenario}_{suffix}"
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
        observed_episodes = cells[scenario]["validation"][
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
        "training_authorized": bool(
            manifest["decision"]["training_authorized"]
        ),
        "distillation_authorized": bool(
            manifest["decision"]["distillation_authorized"]
        ),
        "sa6_authorized": bool(manifest["decision"]["sa6_authorized"]),
        "source_fingerprint_stable": True,
        "r3_r4_behavior_equivalence": equivalence,
        "cells": cells,
        "interpretation": {
            "primary_observation": (
                "Most committed-invalid frames have no left or right passage "
                "candidate under the frozen teacher model."
            ),
            "secondary_observation": (
                "The opposite side remains open in a smaller but non-zero "
                "share of committed-invalid frames, so no-switch deadlock exists "
                "but is not the dominant recorded mode."
            ),
            "prediction_observation": (
                "Pedestrian point-prediction residuals are not elevated at "
                "obstacle-collision terminal rows relative to all slot samples."
            ),
            "next_step": (
                "Add record-only feasibility-mask decomposition for passage "
                "kinematics, obstacle collision, and wall collision before "
                "changing margins, FSM rules, training, or distillation."
            ),
        },
        "limitations": [
            "single checkpoint and single evaluator seed",
            (
                "both_sides_blocked means no candidate under the frozen 19x19 "
                "teacher model, not physical inevitability"
            ),
            (
                "prediction errors are dynamic-slot samples and are not matched "
                "to the obstacle that caused contact"
            ),
            (
                "r4 preserves the r3 teacher FAIL verdict and authorizes no "
                "training, distillation, or SA6"
            ),
        ],
    }


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_markdown(report: dict) -> str:
    lines = [
        "# Stateful teacher per-step diagnostic",
        "",
        "> [!important] Scope",
        (
            "> r4 is record-only diagnostic evidence. Its aggregate outputs are "
            "byte-identical"
        ),
        (
            "> to r3, whose teacher verdict remains FAIL. No training, "
            "distillation, or SA6 is authorized."
        ),
        "",
        "## Validity",
        "",
        f"- Status: `{report['status']}`",
        f"- Protocol SHA-256: `{report['protocol_sha256']}`",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        "- r3/r4 behavior equivalence: all teacher/corridor/cell JSON files byte-identical",
        "- Source fingerprint stable: true",
        "- Episode-end alignment and reset reconciliation: true",
        "",
        "## Main mechanism",
        "",
        "| scenario | committed-invalid frames | other side open | both sides blocked |",
        "|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        frame = report["cells"][scenario]["frame_accounting"]
        lines.append(
            f"| {scenario} | {frame['committed_invalid_frames']:,} "
            f"({_pct(frame['committed_invalid_fraction_of_all_frames'])} of all) | "
            f"{frame['no_switch_other_open_frames']:,} "
            f"({_pct(frame['no_switch_fraction_of_committed_invalid'])}) | "
            f"{frame['both_sides_blocked_frames']:,} "
            f"({_pct(frame['both_sides_blocked_fraction_of_committed_invalid'])}) |"
        )
    lines.extend(
        [
            "",
            "The dominant recorded mode is loss of both passage candidates, not the",
            "no-side-switch rule. This is a statement about the frozen teacher model,",
            "not proof that the physical scene is unavoidable.",
            "",
            "## Timeout location",
            "",
            (
                "| scenario | timeout n | median abs(x) | abs(x) < 0.5 m | "
                "median y | median applied v |"
            ),
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        timeout = report["cells"][scenario]["episode_outcomes"]["timeout"]
        lines.append(
            f"| {scenario} | {timeout['episodes']:,} | "
            f"{timeout['terminal_abs_lateral_m']['p50']:.3f} m | "
            f"{_pct(timeout['terminal_near_center_abs_x_lt_0p5']['fraction'])} | "
            f"{timeout['terminal_longitudinal_m']['p50']:.3f} m | "
            f"{timeout['terminal_applied_v_mps']['p50']:.3f} m/s |"
        )
    lines.extend(
        [
            "",
            "Timeouts usually occur near the corridor center, with the robot stopped",
            "well before the goal; they are not predominantly wall-edge traps.",
            "",
            "## Collision and timeout linkage",
            "",
            (
                "| scenario | obstacle episodes with both blocked | terminal "
                "both blocked | timeout episodes with both blocked | terminal "
                "both blocked |"
            ),
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        outcomes = report["cells"][scenario]["episode_outcomes"]
        obstacle = outcomes["obstacle"]
        timeout = outcomes["timeout"]
        lines.append(
            f"| {scenario} | {obstacle['episodes_with_both_sides_blocked']:,}/"
            f"{obstacle['episodes']:,} | "
            f"{obstacle['terminal_failure_classes'].get('both_sides_blocked', 0):,}/"
            f"{obstacle['episodes']:,} | "
            f"{timeout['episodes_with_both_sides_blocked']:,}/"
            f"{timeout['episodes']:,} | "
            f"{timeout['terminal_failure_classes'].get('both_sides_blocked', 0):,}/"
            f"{timeout['episodes']:,} |"
        )
    lines.extend(
        [
            "",
            "## Pedestrian prediction residual",
            "",
            "| scenario | lag | all-slot p95 | obstacle-terminal p95 |",
            "|---|---:|---:|---:|",
        ]
    )
    for scenario in SCENARIOS:
        prediction = report["cells"][scenario]["prediction_error"]
        for key in ("pred_err_1step_m", "pred_err_5step_m"):
            item = prediction[key]
            lines.append(
                f"| {scenario} | {item['lag_s']:.1f} s | "
                f"{item['all_dynamic_slot_samples_m']['p95']:.3f} m | "
                f"{item['obstacle_collision_terminal_dynamic_slot_samples_m']['p95']:.3f} m |"
            )
    lines.extend(
        [
            "",
            "No collision-linked residual increase is observed. Because the archive does",
            "not identify which obstacle caused contact, this rules out only a large general",
            "prediction-error explanation.",
            "",
            "## Decision",
            "",
            "- Teacher remains FAIL.",
            "- Keep training, distillation, and SA6 on hold.",
            "- Next: add record-only feasibility-mask decomposition (passage kinematics,",
            "  obstacle collision, wall collision), then rerun the same frozen two cells.",
            "- Do not change clearance, horizon, FSM switching, or reward before that split.",
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
    rendered_json = json.dumps(
        report, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered_json, encoding="utf-8")
    else:
        print(rendered_json, end="")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
