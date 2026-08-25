"""Run the frozen four-cell SA5 low-density speed-ratio screen."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
RUNNER = HERE / "run_sa5_low_density_speed_ratio_cell.py"
FREEZE = REPO / "docs/freeze/sa5_low_density_speed_ratio_screen_v1.json"
sys.path.insert(0, str(HERE))

import sa5_low_density_speed_ratio_screen as protocol  # noqa: E402
import sa5_r2_checkpoint_screen_queue as resource_guard  # noqa: E402


SOURCE_PATHS = (
    HERE / "sa5_low_density_speed_ratio_screen.py",
    RUNNER,
    Path(__file__).resolve(),
    FREEZE,
    HERE / "sa5_pedestrian_speed_screen.py",
    HERE / "run_sa5_pedestrian_speed_screen_cell.py",
    HERE / "sa5_r2_checkpoint_screen.py",
    HERE / "run_sa5_r2_checkpoint_screen_cell.py",
    HERE / "sa5_r2_speed_scale_screen.py",
    HERE / "run_sa5_r2_speed_scale_screen_cell.py",
    HERE / "d3_yield_recorder.py",
    HERE / "d5_feasibility_shadow.py",
    HERE / "d4_geometry_selector.py",
    HERE / "fixed_actuator_eval.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events/long_corridor_replay.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/events/long_corridor_replay_geometry.py",
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> dict:
    missing = [str(path) for path in SOURCE_PATHS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"fingerprinted sources missing: {missing}")
    files = {
        str(path.relative_to(REPO)): sha256_of(path) for path in SOURCE_PATHS
    }
    checkpoint = protocol.checkpoint_path().resolve()
    files[str(checkpoint.relative_to(REPO))] = sha256_of(checkpoint)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "sa5_low_density_speed_ratio_fingerprint/v1",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5 c250 low-density P100 x speed-rate screen",
        "",
        f"Status: `{summary['status']}`",
        f"Protocol: `{summary['protocol_sha256']}`",
        f"Curriculum boundary: `{summary['curriculum_boundary']}`",
        "",
        "| density | speed_rate | n | SR | CR | TO | obstacle CR | wall CR | body stop | impact radial p50/p90 | dynamic feasible within 1s | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        p50 = row["impact_radial_closing_p50_mps"]
        p90 = row["impact_radial_closing_p90_mps"]
        severity = "-" if p50 is None else f"{p50:.3f}/{p90:.3f} m/s"
        lines.append(
            f"| {row['density']} | {row['speed_rate']:.1f} | {row['n']} | "
            f"{row['sr']:.2%} | {row['cr']:.2%} | {row['to']:.2%} | "
            f"{row['obstacle_cr']:.2%} | {row['wall_cr']:.2%} | "
            f"{row['actual_body_stop_fraction']:.2%} | {severity} | "
            f"{row['dynamic_feasible_within_1s_fraction']:.2%} | "
            f"{'PASS' if row['gate']['pass'] else 'FAIL'} |"
        )
    lines.extend(["", "## Within-density speed-rate comparisons", ""])
    for density, comparison in summary["within_density_comparisons"].items():
        lines.append(
            f"- {density}: delta CR(s070-s100) "
            f"{comparison['delta_cr_s070_minus_s100']:+.2%}, "
            f"{comparison['standardized_delta']:+.2f} SE, "
            f"`{comparison['classification']}`"
        )
    lines.extend(
        [
            "",
            "D5 feasibility is model-bounded to the frozen action grid, finite horizon, and final one-second window; it is not physical inevitability.",
            "",
            "## Evidence boundary",
            "",
            summary["interpretation_limit"],
            "",
            "No parent was accepted; no training or SA6 was started.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    frozen = protocol.screen_protocol()
    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)
    if json.loads(FREEZE.read_text(encoding="utf-8")) != frozen:
        raise RuntimeError("frozen protocol file differs from runtime protocol")
    checkpoint = protocol.checkpoint_path().resolve()
    if sha256_of(checkpoint) != protocol.CHECKPOINT["sha256"]:
        raise RuntimeError("checkpoint hash mismatch")
    snapshot = resource_guard.resource_snapshot(allow_shared_gpu=False)
    if not snapshot["ready"]:
        raise RuntimeError(f"GPU resources are not ready: {snapshot}")

    (output / "PREREGISTRATION.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
    )
    before = source_fingerprint()
    (output / "SOURCE_FINGERPRINT_BEFORE.json").write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )

    completed = []
    try:
        for spec in protocol.cells():
            label = spec["cell"]
            cell_dir = output / label
            command = [
                str(PYTHON),
                str(RUNNER),
                "--output-dir",
                str(cell_dir),
                "--density",
                spec["label"],
                "--speed-rate",
                f"{spec['speed_rate']:g}",
                "--expect-checkpoint-sha256",
                protocol.CHECKPOINT["sha256"],
                "--expect-protocol-sha256",
                frozen["sha256"],
            ]
            print(
                f"[LOW-DENSITY-SPEED-QUEUE] START {label} P100",
                flush=True,
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            result = subprocess.run(command, cwd=REPO, env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"{label} runner exited {result.returncode}")
            matches = list(cell_dir.glob("*_cell.json"))
            if len(matches) != 1:
                raise RuntimeError(f"{label} produced {len(matches)} cell JSON files")
            cell = json.loads(matches[0].read_text(encoding="utf-8"))
            protocol.validate_cell(cell)
            completed.append(cell)
            if source_fingerprint() != before:
                raise RuntimeError(f"source fingerprint drift after {label}")
            print(
                f"[LOW-DENSITY-SPEED-QUEUE] DONE {label} "
                f"n={cell['metrics']['n']} CR={cell['metrics']['cr']:.4f}",
                flush=True,
            )

        after = source_fingerprint()
        (output / "SOURCE_FINGERPRINT_AFTER.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        if before != after:
            raise RuntimeError("source fingerprint drifted during the screen")
        summary = protocol.summarize_cells(completed)
        summary.update(
            {
                "completed_at": datetime.now().astimezone().isoformat(),
                "source_fingerprint_stable": True,
                "source_fingerprint_sha256": before["sha256"],
            }
        )
        (output / "SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output / "SUMMARY.md").write_text(
            render_markdown(summary), encoding="utf-8"
        )
        print(
            "[LOW-DENSITY-SPEED-QUEUE] COMPLETE "
            f"boundary={summary['curriculum_boundary']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        incomplete = {
            "schema": "sa5_low_density_speed_ratio_incomplete/v1",
            "status": "INCOMPLETE_FAIL_CLOSED",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_cells": [cell.get("cell") for cell in completed],
            "timestamp": datetime.now().astimezone().isoformat(),
            "accepted_parent": False,
            "training_started": False,
            "sa6_started": False,
        }
        (output / "INCOMPLETE.json").write_text(
            json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
