"""Run the frozen 8-cell c50 versus weighted-it25 4S2D screen."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_v3_c50_random2d_weighted_screen as protocol  # noqa: E402
import sa5_v3_provisional_pilot_screen_queue as base_queue  # noqa: E402


RUNNER = HERE / "run_sa5_v3_c50_random2d_weighted_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_v3_c50_random2d_weighted_screen_v1.json"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    protocol.AUTHORIZATION,
    REPO / "docs/freeze/sa5_v3_c50_random2d_weighted_p25_completion_20260824.json",
    HERE / "sa5_v3_c50_it50_direction_ab.py",
    HERE / "sa5_v3_provisional_pilot_screen.py",
    Path(base_queue.__file__).resolve(),
    HERE / "run_sa5_r2_checkpoint_screen_cell.py",
    HERE / "run_sa3_v3_retention_screen_cell.py",
    HERE / "run_sa5_joint_retention_gates.py",
    HERE / "validate_gates.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    HERE / "sa3_sa8_acceptance_contract.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_stage_curriculum_v2.py",
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

_EXPECTED_RUNTIME = {
    "anchor_c50": (49, 6_400, "diagnostic_anchor"),
    "weighted_it25": (24, 3_200, "random_2d_weighted_pilot"),
}


def build_checkpoint_manifest() -> dict:
    checkpoints = {}
    for spec in protocol.CANDIDATE_SPECS:
        name = spec["name"]
        path = protocol.checkpoint_path(spec).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = protocol.sha256_of(path)
        if digest != spec["expected_sha256"]:
            raise RuntimeError(f"{name} checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected_iteration, expected_steps, phase = _EXPECTED_RUNTIME[name]
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
            "status": spec["status"],
            "rl_optimizer_state_entries": len(optimizer["state"]),
        }
    return {
        "schema": "sa5_v3_c50_random2d_weighted_screen_checkpoint_manifest/v1",
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoints": checkpoints,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5-v3 c50 vs weighted it25 - fixed 4S2D screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| motion | arm | n | SR | CR | TO | stop | speed | reverse |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: float | None) -> str:
        return "-" if value is None else f"{value:.4f}"

    for row in summary["rows"]:
        behavior = row["behavior"]
        lines.append(
            f"| {row['motion_mode']} | {row['checkpoint_name']} | {row['n']} | "
            f"{row['sr']:.2%} | {row['cr']:.2%} | {row['to']:.2%} | "
            f"{fmt(behavior['stop_command_fraction'])} | "
            f"{fmt(behavior['linear_speed_abs_mean_mps'])} | "
            f"{fmt(behavior['reverse_command_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Weighted it25 minus anchor",
            "",
            "| motion | dSR | z(SR) | dCR | z(CR) | dTO | z(TO) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["comparisons"]:
        lines.append(
            f"| {row['motion_mode']} | {row['sr']['delta']:+.2%} | "
            f"{row['sr']['z']:+.2f} | {row['cr']['delta']:+.2%} | "
            f"{row['cr']['z']:+.2f} | {row['to']['delta']:+.2%} | "
            f"{row['to']['z']:+.2f} |"
        )
    lines.extend(
        [
            "",
            f"Target pass: `{summary['target_pass']}`",
            f"Direction retention pass: `{summary['retention_pass']}`",
            f"Acceptance observed: `{summary['pilot_acceptance_observed']}`",
            f"Conservative shift (descriptive): `{summary['conservative_shift_descriptive']}`",
            f"Anchor replication exact: `{summary['anchor_replication_exact']}`",
            f"Next action: `{summary['next_action']}`",
            "",
            "This screen starts no training and authorizes neither extension nor SA6.",
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
