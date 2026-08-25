"""Render the completed SA5-R2 screen bundle without changing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(summary: dict) -> str:
    verdict = summary["verdict"]
    phase_a = verdict["phase_a"]
    phase_a_by_name = {
        row["checkpoint_name"]: row for row in phase_a["summaries"]
    }
    phase_b_by_name = {
        row["checkpoint_name"]: row
        for row in verdict["phase_b_candidates"]
    }
    raw_phase_b = {
        (row["checkpoint_name"], row["scenario"]): row
        for row in summary["phase_b_base_cells"]
    }

    lines = [
        "# SA5-R2 fixed sealed checkpoint screen",
        "",
        f"Status: `{summary['status']}`",
        f"Source fingerprint stable: `{summary['source_fingerprint_stable']}`",
        "",
        "## Phase A corridor-family ranking",
        "",
        "| rank | checkpoint | lateral CR | longitudinal CR | random2d CR | mixed CR | worst CR | mean CR | hard gate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for rank, name in enumerate(phase_a["ranked"], start=1):
        row = phase_a_by_name[name]
        cr = row["family_cr"]
        lines.append(
            f"| {rank} | {name} | {cr['corridor_lateral']:.2%} | "
            f"{cr['corridor_longitudinal']:.2%} | "
            f"{cr['corridor_random2d']:.2%} | "
            f"{cr['corridor_mixed']:.2%} | "
            f"{row['worst_family_cr']:.2%} | {row['mean_family_cr']:.2%} | "
            f"{'PASS' if row['all_corridor_hard_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Top two: `{', '.join(phase_a['top_two'])}`",
            "",
            "## Phase B retention",
            "",
            "| checkpoint | native SR | native CR | narrow SR | narrow CR | crossing | direct | retention gates |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name in phase_a["top_two"]:
        row = phase_b_by_name[name]
        native = row["native"]
        narrow = row["narrow"]
        lines.append(
            f"| {name} | {native['sr']:.2%} | {native['cr']:.2%} | "
            f"{narrow['sr']:.2%} | {narrow['cr']:.2%} | "
            f"{narrow['crossing_rate']:.2%} | "
            f"{narrow['direct_crossing_rate']:.2%} | "
            f"{'PASS' if row['retention_hard_pass'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "## Low-density sealed corridor",
            "",
            "| checkpoint | density | SR | CR | TO | mean |v| | stop fraction | gate |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for name in phase_a["top_two"]:
        for scenario in ("corridor_low_0s1d", "corridor_low_1s1d"):
            raw = raw_phase_b[(name, scenario)]
            metrics = raw["metrics"]
            report = raw.get("corridor_report") or {}
            speed = report.get("linear_speed_abs_mean_mps")
            stop = report.get("stop_command_fraction")
            speed_text = "N/A" if speed is None else f"{float(speed):.4f} m/s"
            stop_text = "N/A" if stop is None else f"{float(stop):.2%}"
            gate = (
                float(metrics["sr"]) >= 0.90
                and float(metrics["cr"]) <= 0.10
                and float(metrics["to"]) <= 0.05
            )
            lines.append(
                f"| {name} | {scenario.removeprefix('corridor_low_').upper()} | "
                f"{metrics['sr']:.2%} | {metrics['cr']:.2%} | "
                f"{metrics['to']:.2%} | {speed_text} | {stop_text} | "
                f"{'PASS' if gate else 'FAIL'} |"
            )

    replacements = verdict.get("sample_size_replacements", {})
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"Screen-eligible checkpoints: `{verdict['screen_eligible_checkpoints']}`",
            "Recommended for human consideration: "
            f"`{verdict['recommended_for_human_consideration']}`",
            "",
            "No parent was accepted; no training or SA6 was started.",
            "",
            "## Evidence notes",
            "",
            f"- Phase-A sample replacements: `{len(replacements.get('phase_a', []))}`.",
            f"- Phase-B sample replacements: `{len(replacements.get('phase_b', []))}`.",
            "- Repair trigger used completed episode count only; base and repair samples were not pooled.",
            "- Single training seed and single evaluator seed: checkpoint screen, not formal graduation.",
            "- Low-density speed and stop fraction are descriptive, not post-hoc gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    summary_path = args.summary.expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not str(summary.get("status", "")).startswith("COMPLETE_VALID"):
        raise RuntimeError("refusing to render a non-complete summary")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else summary_path.with_suffix(".md")
    )
    output.write_text(render(summary), encoding="utf-8")
    recovery = output.parent / "REPORT_RENDER_RECOVERY.json"
    recovery.write_text(
        json.dumps(
            {
                "schema": "sa5_r2_checkpoint_screen_report_recovery/v1",
                "status": "REPORT_RENDER_RECOVERED_AFTER_COMPLETE_MEASUREMENT",
                "summary": str(summary_path),
                "summary_sha256": sha256_of(summary_path),
                "output": str(output),
                "measurement_payload_modified": False,
                "reason": (
                    "original renderer read low-density descriptive fields from "
                    "metrics instead of corridor_report"
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"rendered {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

