"""Run the frozen 15-cell c500 SA5-v3 parent-control screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_v3_c500_parent_control_screen as protocol  # noqa: E402
import sa5_v3_provisional_pilot_screen_queue as implementation  # noqa: E402


RUNNER = HERE / "run_sa5_v3_c500_parent_control_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_v3_c500_parent_control_screen_v1.json"

implementation.RUNNER = RUNNER
implementation.FREEZE = FREEZE
implementation.protocol = protocol
implementation.SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    Path(implementation.__file__).resolve(),
    RUNNER,
    FREEZE,
    REPO
    / "docs/freeze/sa5_v3_c500_parent_control_authorization_20260823.json",
    REPO
    / "logs/gates/sa5_v3_provisional_pilot_screen/"
    "screen_20260822_r1/SUMMARY.json",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_c500_parent_control_from_sa4v3_p50.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "e2e_sa5_v3_provisional_from_sa4v3_c550_p50.py",
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
        "parent_c500": (199, 25_600),
        "control_it25": (24, 3_200),
        "control_it50": (49, 6_400),
    }
    for spec in protocol.CANDIDATE_SPECS:
        name = spec["name"]
        path = protocol.checkpoint_path(spec).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = implementation.sha256_of(path)
        expected_hash = spec.get("expected_sha256")
        if expected_hash is not None and digest != expected_hash:
            raise RuntimeError(f"{name} checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_iteration, expected_steps = expected_runtime[name]
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
            "iteration": expected_iteration,
            "total_steps": expected_steps,
            "rl_optimizer_state_entries": len(optimizer["state"]),
        }
    return {
        "schema": "sa5_v3_c500_parent_control_checkpoint_manifest/v1",
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "run_name": protocol.RUN_NAME,
        "checkpoints": checkpoints,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5-v3 c500 parent-control screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| checkpoint | target SR | target CR | target z | retention | extension |",
        "|---|---:|---:|---:|---|---|",
    ]
    parent = summary["parent"]
    lines.append(
        f"| parent_c500 | {parent['target_metrics']['sr']:.2%} | "
        f"{parent['target_metrics']['cr']:.2%} | - | baseline | - |"
    )
    for row in summary["candidate_results"]:
        lines.append(
            f"| {row['checkpoint_name']} | {row['target_metrics']['sr']:.2%} | "
            f"{row['target_metrics']['cr']:.2%} | "
            f"{row['target_improvement']['z']:.2f} | "
            f"{'PASS' if row['retention_pass'] else 'FAIL'} | "
            f"{'PASS' if row['extension_eligible'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"Selected extension checkpoint: `{summary['selected_extension_checkpoint']}`",
            f"Next action: `{summary['next_action']}`",
            "",
            "SA4 remains not graduated. This screen starts neither training nor SA6.",
        ]
    )
    return "\n".join(lines) + "\n"


implementation.build_checkpoint_manifest = build_checkpoint_manifest
implementation.render_markdown = render_markdown


def main(argv: list[str] | None = None) -> int:
    return implementation.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
