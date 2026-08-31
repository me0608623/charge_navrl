"""Capture exact c600 K8 inputs and run held-out interaction probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rnn_car_wdclean.analyze_sa5_c600_k8_observability import analyze_suite
from rnn_car_wdclean.analyze_sa5_c600_step_trace import load_trace
from rnn_car_wdclean import run_sa5_c600_step_trace as base_runner
from rnn_car_wdclean import sa5_c600_k8_observability_protocol as protocol
from rnn_car_wdclean import sa5_c600_step_trace_protocol as base_protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SCENARIOS = base_protocol.SCENARIOS


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    # Recompute the complete concrete source set so the suite remains
    # fail-closed instead of trusting a previously rendered fingerprint.
    concrete = {
        "base_runner": Path(base_runner.__file__).resolve(),
        "base_protocol": Path(base_protocol.__file__).resolve(),
        "play": base_runner.PLAY,
        "action": base_runner.ACTION,
        "scheduler": base_runner.SCHEDULER,
        "d3": base_runner.D3,
        "d5": base_runner.D5,
        "trace": base_runner.TRACE,
        "base_analyzer": base_runner.ANALYZER,
        "runner": Path(__file__).resolve(),
        "protocol": Path(protocol.__file__).resolve(),
        "analyzer": HERE / "analyze_sa5_c600_k8_observability.py",
    }
    return {name: sha256(path) for name, path in concrete.items()}


def _selected(report: dict, feature: str) -> dict:
    return report["feature_sets"][feature]["selected_by_validation"]


def render_summary(report: dict) -> str:
    lines = [
        "# SA5 c600 K8 interaction observability",
        "",
        "> Information diagnostic only. No policy training, override, teacher, distillation, or SA6 launch.",
        "",
        "| target | feature | selected model | test balanced acc | positive recall | negative recall |",
        "|---|---|---|---:|---:|---:|",
    ]
    for task_name in ("lateral_vacated_side", "random2d_reversal_1s_ahead"):
        task = report[task_name]
        for feature in (
            "policy_current_input",
            "policy_k8_input",
            "policy_representation",
        ):
            selected = _selected(task, feature)
            metrics = selected["test"]
            lines.append(
                f"| {task_name} | {feature} | {selected['model']} | "
                f"{metrics['balanced_accuracy']:.3f} | "
                f"{metrics['positive_recall']:.3f} | "
                f"{metrics['negative_recall']:.3f} |"
            )
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- vacated-side policy representation observable: `{decision['vacated_side']['policy_representation_observable']}`",
            f"- 1 s reversal policy representation observable: `{decision['reversal_1s_ahead']['policy_representation_observable']}`",
            "- reversal observability is explicitly non-blocking for SA6.",
            "- this probe does not itself authorize or block SA6.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="logs/gates/sa5_c600_k8_observability/screen_20260826_r1",
    )
    parser.add_argument("--steps", type=int, default=base_protocol.ROLLOUT_STEPS)
    parser.add_argument("--num-envs", type=int, default=base_protocol.NUM_ENVS)
    args = parser.parse_args()
    output_arg = Path(args.output)
    output = output_arg if output_arg.is_absolute() else REPO / output_arg
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = source_fingerprint()
    document = protocol.frozen_protocol()
    if sha256(base_protocol.CHECKPOINT) != base_protocol.CHECKPOINT_SHA256:
        raise RuntimeError("c600 checkpoint SHA-256 drifted")
    (output / "PROTOCOL.json").write_text(
        json.dumps(
            {
                "status": "FROZEN_BEFORE_GPU",
                "protocol": document,
                "source_fingerprint": before,
                "gpu_before": base_runner.gpu_snapshot(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    try:
        cells = {
            scenario: base_runner.run_cell(
                output,
                scenario,
                steps=args.steps,
                envs=args.num_envs,
                include_k8_input=True,
            )
            for scenario in SCENARIOS
        }
        lateral, lateral_metadata = load_trace(
            cells["lateral"]["paths"]["trace"]
        )
        random2d, random2d_metadata = load_trace(
            cells["random_2d"]["paths"]["trace"]
        )
        for name, metadata in (
            ("lateral", lateral_metadata),
            ("random_2d", random2d_metadata),
        ):
            if metadata.get("schema") != "policy_step_trace/v2-k8-observability":
                raise RuntimeError(f"{name} did not produce a K8 trace")
            if metadata.get("policy_inputs", {}).get("dimensions") != {
                "policy_current_input": 83,
                "policy_k8_input": 587,
                "policy_representation": 179,
            }:
                raise RuntimeError(f"{name} K8 input dimensions drifted")
        report = analyze_suite(lateral, random2d)
        (output / "observability_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        (output / "SUMMARY.md").write_text(
            render_summary(report), encoding="utf-8"
        )
        after = source_fingerprint()
        if before != after:
            raise RuntimeError("source fingerprint drifted during observability suite")
        if sha256(base_protocol.CHECKPOINT) != base_protocol.CHECKPOINT_SHA256:
            raise RuntimeError("checkpoint drifted during observability suite")
        manifest = {
            "schema": "sa5_c600_k8_observability_suite/v1",
            "status": "COMPLETE_VALID_INFORMATION_DIAGNOSTIC",
            "protocol_sha256": document["sha256"],
            "checkpoint_sha256": base_protocol.CHECKPOINT_SHA256,
            "source_fingerprint_stable": True,
            "policy_actions_replaced": False,
            "training_authorized": False,
            "sa6_authorized_by_this_probe": False,
            "sa6_blocked_by_this_probe": False,
            "cells": cells,
            "decision": report["decision"],
            "artifacts": {
                "report": str(output / "observability_report.json"),
                "summary": str(output / "SUMMARY.md"),
            },
            "gpu_after": base_runner.gpu_snapshot(),
        }
        (output / "suite_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    except Exception as error:
        (output / "INCOMPLETE_NO_VERDICT.json").write_text(
            json.dumps(
                {
                    "status": "INCOMPLETE_NO_VERDICT",
                    "error": repr(error),
                    "completed_outputs_are_preserved": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    print(f"[SA5-C600-K8-OBSERVABILITY] complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
