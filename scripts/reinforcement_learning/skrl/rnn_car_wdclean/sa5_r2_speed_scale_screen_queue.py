"""Run the frozen four-arm SA5-R2 c250 vehicle speed-rate screen."""

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
RUNNER = HERE / "run_sa5_r2_speed_scale_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_r2_c250_speed_scale_screen_v2.json"
sys.path.insert(0, str(HERE))

import sa5_r2_speed_scale_screen as protocol  # noqa: E402


SOURCE_PATHS = (
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/sa5_r2_speed_scale_screen.py",
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa5_r2_speed_scale_screen_cell.py",
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/sa5_r2_speed_scale_screen_queue.py",
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/run_sa5_r2_checkpoint_screen_cell.py",
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/sa5_r2_checkpoint_screen.py",
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/fixed_actuator_eval.py",
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/d3_yield_recorder.py",
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/analyze_sa4_d3_baseline.py",
    "scripts/reinforcement_learning/skrl/rnn_car_wdclean/corridor_eval_metrics.py",
    "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/actions/discrete_differential_drive.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/long_corridor_replay.py",
    "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/mdp/events/long_corridor_replay_geometry.py",
    "rover_rl/src/rover_rl_inference/rover_rl_inference/policy_node.py",
    "rover_rl/install/rover_rl_inference/lib/python3.12/site-packages/rover_rl_inference/policy_node.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> dict:
    files = {}
    for relative in SOURCE_PATHS:
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = _sha256(path)
    checkpoint = Path(protocol.screen_protocol()["checkpoint"]["path"])
    files[str(checkpoint)] = _sha256(checkpoint)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"))
    return {
        "schema": "sa5_r2_c250_speed_scale_source_fingerprint/v2",
        "files": files,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _render_markdown(summary: dict) -> str:
    lines = [
        "# SA5-R2 c250 vehicle speed-rate screen",
        "",
        f"Status: `{summary['status']}`",
        f"Protocol: `{summary['protocol_sha256']}`",
        "",
        "c250 remains a relative-best diagnostic checkpoint, not a graduated parent.",
        "",
        "| rate | n | SR | CR | TO | obstacle CR | wall CR | cmd |v| | cmd stop | body |v| | body stop | actual decel observed | decel lead p50 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lead = row["actual_decel_lead_p50_s"]
        lead_text = "-" if lead is None else f"{lead:.2f}s"
        lines.append(
            f"| {row['scale']:.1f} | {row['n']} | {row['sr']:.2%} | "
            f"{row['cr']:.2%} | {row['to']:.2%} | "
            f"{row['obstacle_cr']:.2%} | {row['wall_cr']:.2%} | "
            f"{row['commanded_mean_abs_speed_mps']:.4f} m/s | "
            f"{row['commanded_stop_fraction']:.2%} | "
            f"{row['actual_body_mean_abs_speed_mps']:.4f} m/s | "
            f"{row['actual_body_stop_fraction']:.2%} | "
            f"{row['actual_decel_observed_fraction']:.2%} | {lead_text} |"
        )
    target = summary["scale_0p7_vs_1p0"]
    lines.extend(
        [
            "",
            "## 0.7 versus 1.0",
            "",
            f"- Delta SR: {target['delta_sr_vs_1p0']:+.2%}",
            f"- Delta CR: {target['delta_cr_vs_1p0']:+.2%}",
            f"- Delta TO: {target['delta_to_vs_1p0']:+.2%}",
            f"- Lowest observed CR scale: `{summary['lowest_observed_cr_scale']}`",
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
    (output / "PREREGISTRATION.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True), encoding="utf-8"
    )
    before = source_fingerprint()
    (output / "SOURCE_FINGERPRINT_BEFORE.json").write_text(
        json.dumps(before, indent=2, sort_keys=True), encoding="utf-8"
    )

    cells = []
    try:
        for scale in protocol.SPEED_SCALES:
            label = protocol.scale_label(scale)
            cell_dir = output / label
            command = [
                str(PYTHON),
                str(RUNNER),
                "--output-dir",
                str(cell_dir),
                "--speed-rate",
                f"{scale:g}",
                "--expect-checkpoint-sha256",
                protocol.CHECKPOINT["sha256"],
                "--expect-protocol-sha256",
                frozen["sha256"],
            ]
            print(
                f"[SA5-R2-SPEED-QUEUE] START {label} scale={scale:g}",
                flush=True,
            )
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            result = subprocess.run(
                command, cwd=REPO, env=env, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"{label} runner exited {result.returncode}"
                )
            matches = list(cell_dir.glob("*_cell.json"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"{label} produced {len(matches)} cell JSON files"
                )
            cell = json.loads(matches[0].read_text(encoding="utf-8"))
            protocol.validate_cell(cell)
            cells.append(cell)
            print(
                f"[SA5-R2-SPEED-QUEUE] DONE {label} "
                f"n={cell['metrics']['n']} CR={cell['metrics']['cr']:.4f}",
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
            _render_markdown(summary), encoding="utf-8"
        )
        print(
            "[SA5-R2-SPEED-QUEUE] COMPLETE "
            f"lowest_cr_scale={summary['lowest_observed_cr_scale']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        incomplete = {
            "schema": "sa5_r2_c250_speed_scale_incomplete/v2",
            "status": "INCOMPLETE_FAIL_CLOSED",
            "error": f"{type(exc).__name__}: {exc}",
            "completed_cells": [cell.get("scale_label") for cell in cells],
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        (output / "INCOMPLETE.json").write_text(
            json.dumps(incomplete, indent=2, sort_keys=True), encoding="utf-8"
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
