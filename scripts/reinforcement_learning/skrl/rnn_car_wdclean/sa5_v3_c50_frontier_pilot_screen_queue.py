"""Run the frozen 15-cell c50 difficulty-frontier pilot screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_v3_c50_frontier_pilot_screen as protocol  # noqa: E402
import sa5_v3_provisional_pilot_screen_queue as base_queue  # noqa: E402


RUNNER = HERE / "run_sa5_v3_c50_frontier_pilot_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_v3_c50_frontier_pilot_screen_v1.json"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    REPO
    / "docs/freeze/"
    "sa5_v3_c50_frontier_curriculum_p50_authorization_20260823.json",
    REPO
    / "docs/freeze/sa5_v3_c50_frontier_curriculum_phase1_complete_v1.json",
    REPO
    / "docs/freeze/sa5_v3_c50_frontier_curriculum_phase2_complete_v1.json",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_frontier_curriculum_phase1_p25.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c50_frontier_curriculum_phase2_p25.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    HERE / "run_sa5_r2_checkpoint_screen_cell.py",
    HERE / "run_sa3_v3_retention_screen_cell.py",
    HERE / "run_sa5_joint_retention_gates.py",
    HERE / "validate_gates.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    HERE / "sa3_sa8_acceptance_contract.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_stage_curriculum_v2.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/observations/obs_functions.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/long_corridor_replay.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/long_corridor_replay_geometry.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/corridor_density.py",
)


def build_checkpoint_manifest() -> dict:
    checkpoints = {}
    expected_runtime = {
        "anchor_c50": (49, 6_400, "diagnostic_anchor"),
        "pilot_it25": (24, 3_200, "curriculum_phase_1"),
        "pilot_it50": (24, 3_200, "curriculum_phase_2"),
    }
    for spec in protocol.CANDIDATE_SPECS:
        name = spec["name"]
        path = protocol.checkpoint_path(spec).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = protocol.sha256_of(path)
        expected_hash = spec.get("expected_sha256")
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError(f"{name} checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_iteration, expected_steps, phase = expected_runtime[name]
        if (
            int(payload.get("iteration", -1)) != expected_iteration
            or int(payload.get("total_steps", -1)) != expected_steps
        ):
            raise RuntimeError(f"{name} checkpoint iteration ledger mismatch")
        optimizer = payload.get("charge_opt_rl") or {}
        if not optimizer.get("state") or not optimizer.get("param_groups"):
            raise RuntimeError(f"{name} lacks resumable RL optimizer state")
        checkpoints[name] = {
            "path": str(path),
            "sha256": digest,
            "embedded_iteration": expected_iteration,
            "embedded_total_steps": expected_steps,
            "conceptual_iteration": spec["conceptual_iteration"],
            "phase": phase,
            "rl_optimizer_state_entries": len(optimizer["state"]),
        }
    return {
        "schema": "sa5_v3_c50_frontier_pilot_checkpoint_manifest/v1",
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoints": checkpoints,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5-v3 c50 difficulty-frontier pilot screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| checkpoint | target SR | target CR | target TO | z | retention | acceptance |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    anchor = summary["anchor"]
    metrics = anchor["target_metrics"]
    lines.append(
        f"| anchor_c50 | {metrics['sr']:.2%} | {metrics['cr']:.2%} | "
        f"{metrics['to']:.2%} | - | diagnostic baseline | - |"
    )
    for row in summary["candidate_results"]:
        metrics = row["target_metrics"]
        lines.append(
            f"| {row['checkpoint_name']} | {metrics['sr']:.2%} | "
            f"{metrics['cr']:.2%} | {metrics['to']:.2%} | "
            f"{row['target_improvement']['z']:.2f} | "
            f"{'PASS' if row['retention_pass'] else 'FAIL'} | "
            f"{'PASS' if row['pilot_acceptance_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "Selected development checkpoint: "
            f"`{summary['selected_development_checkpoint']}`",
            f"Next action: `{summary['next_action']}`",
            "",
            "This single-seed screen authorizes neither graduation, extension, nor SA6.",
        ]
    )
    return "\n".join(lines) + "\n"


base_queue.protocol = protocol
base_queue.RUNNER = RUNNER
base_queue.FREEZE = FREEZE
base_queue.SOURCE_PATHS = SOURCE_PATHS
base_queue.build_checkpoint_manifest = build_checkpoint_manifest
base_queue.render_markdown = render_markdown


def main(argv: list[str] | None = None) -> int:
    return base_queue.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
