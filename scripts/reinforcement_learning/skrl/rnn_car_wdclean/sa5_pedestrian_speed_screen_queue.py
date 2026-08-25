"""Run the frozen four-arm SA5 c250 pedestrian-speed diagnostic screen."""

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
RUNNER = HERE / "run_sa5_pedestrian_speed_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_pedestrian_speed_screen_v1.json"
sys.path.insert(0, str(HERE))

import sa5_pedestrian_speed_screen as protocol  # noqa: E402
import sa5_r2_checkpoint_screen_queue as resource_guard  # noqa: E402


SOURCE_PATHS = (
    HERE / "sa5_pedestrian_speed_screen.py",
    RUNNER,
    Path(__file__).resolve(),
    FREEZE,
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
        "schema": "sa5_pedestrian_speed_screen_fingerprint/v1",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5 c250 pedestrian-speed screen",
        "",
        f"Status: `{summary['status']}`",
        f"Protocol: `{summary['protocol_sha256']}`",
        f"Mechanism: `{summary['mechanism_classification']}`",
        "",
        "| arm | pedestrian speed | n | SR | CR | delta CR | delta SE | obstacle CR | wall CR | impact radial p50/p90 | body stop | stop observed in collision window | first-stop lead p50 | dynamic feasible within 1s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        low, high = row["pedestrian_speed_range_m_s"]
        p50 = row["impact_radial_closing_p50_mps"]
        p90 = row["impact_radial_closing_p90_mps"]
        severity = (
            "-"
            if p50 is None or p90 is None
            else f"{p50:.3f}/{p90:.3f} m/s"
        )
        lines.append(
            f"| {row['arm']} | {low:.2f}-{high:.2f} m/s | {row['n']} | "
            f"{row['sr']:.2%} | {row['cr']:.2%} | "
            f"{row['delta_cr_vs_p035']:+.2%} | "
            f"{row['delta_cr_se_vs_p035']:+.2f} | "
            f"{row['obstacle_cr']:.2%} | {row['wall_cr']:.2%} | "
            f"{severity} | {row['actual_body_stop_fraction']:.2%} | "
            f"{row['actual_stop_observed_fraction']:.2%} | "
            f"{row['actual_stop_lead_p50_s']:.2f}s | "
            f"{row['dynamic_feasible_within_1s_fraction']:.2%} |"
        )
    lines.extend(
        [
            "",
            "Impact severity is the positive radial closing component at the collision transition, not full 2D relative speed.",
            "Dynamic feasibility is D5 dynamic-only shadow evidence; joint static-plus-dynamic feasibility is intentionally excluded from the classification.",
            "D5-model feasibility is not physical inevitability: it is limited to the frozen 19x19 action grid, finite horizon, and the final one-second window. Earlier sustained waiting remains untested.",
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

    cells = []
    try:
        for arm in protocol.PEDESTRIAN_SPEED_ARMS:
            label = arm["label"]
            cell_dir = output / label.lower()
            command = [
                str(PYTHON),
                str(RUNNER),
                "--output-dir",
                str(cell_dir),
                "--arm",
                label,
                "--expect-checkpoint-sha256",
                protocol.CHECKPOINT["sha256"],
                "--expect-protocol-sha256",
                frozen["sha256"],
            ]
            print(
                f"[PED-SPEED-QUEUE] START {label} "
                f"range={arm['range_m_s']}",
                flush=True,
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            result = subprocess.run(command, cwd=REPO, env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"{label} runner exited {result.returncode}")
            matches = list(cell_dir.glob("*_cell.json"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"{label} produced {len(matches)} cell JSON files"
                )
            cell = json.loads(matches[0].read_text(encoding="utf-8"))
            protocol.validate_cell(cell)
            cells.append(cell)
            if source_fingerprint() != before:
                raise RuntimeError(f"source fingerprint drift after {label}")
            print(
                f"[PED-SPEED-QUEUE] DONE {label} n={cell['metrics']['n']} "
                f"CR={cell['metrics']['cr']:.4f}",
                flush=True,
            )

        after = source_fingerprint()
        (output / "SOURCE_FINGERPRINT_AFTER.json").write_text(
            json.dumps(after, indent=2, sort_keys=True), encoding="utf-8"
        )
        if before != after:
            raise RuntimeError("source fingerprint drifted during the screen")
        summary = protocol.summarize_cells(cells)
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
            "[PED-SPEED-QUEUE] COMPLETE "
            f"mechanism={summary['mechanism_classification']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        incomplete = {
            "schema": "sa5_pedestrian_speed_screen_incomplete/v1",
            "status": "INCOMPLETE_FAIL_CLOSED",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_cells": [cell.get("arm") for cell in cells],
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
