"""Run the frozen 4-cell matched-control screen and three-arm analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_v3_c50_equalweight_control_screen as protocol  # noqa: E402
import sa5_v3_provisional_pilot_screen_queue as base_queue  # noqa: E402


RUNNER = HERE / "run_sa5_v3_c50_equalweight_control_screen_cell.py"
FREEZE = REPO / "docs/freeze/sa5_v3_c50_equalweight_control_screen_v1.json"
SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    protocol.AUTHORIZATION,
    protocol.CONTROL_COMPLETION,
    protocol.FROZEN_PRIOR_SUMMARY,
    protocol.FROZEN_PRIOR_PROTOCOL,
    HERE / "run_sa5_v3_c50_random2d_weighted_screen_cell.py",
    HERE / "sa5_v3_c50_random2d_weighted_screen.py",
    HERE / "sa5_v3_provisional_pilot_screen_queue.py",
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


def build_checkpoint_manifest() -> dict:
    spec = protocol.CONTROL_IT25
    path = protocol.checkpoint_path(spec).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = protocol.sha256_of(path)
    if digest != spec["expected_sha256"]:
        raise RuntimeError("equal-weight control checkpoint hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        int(payload.get("iteration", -1)) != 24
        or int(payload.get("total_steps", -1)) != 3_200
    ):
        raise RuntimeError("equal-weight control checkpoint iteration ledger mismatch")
    optimizer = payload.get("charge_opt_rl") or {}
    if not optimizer.get("state") or not optimizer.get("param_groups"):
        raise RuntimeError("equal-weight control lacks resumable RL optimizer state")
    return {
        "schema": "sa5_v3_c50_equalweight_control_screen_checkpoint_manifest/v1",
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoints": {
            spec["name"]: {
                "path": str(path),
                "sha256": digest,
                "embedded_iteration": 24,
                "embedded_total_steps": 3_200,
                "conceptual_iteration": 25,
                "phase": "matched_equalweight_control",
                "status": spec["status"],
                "rl_optimizer_state_entries": len(optimizer["state"]),
            }
        },
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5-v3 c50 matched three-arm fixed screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| motion | anchor CR | control CR | weighted CR | weighted-control dCR | z |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["comparisons"]:
        effect = row["weighted_minus_control"]["cr"]
        lines.append(
            f"| {row['motion_mode']} | {row['anchor']['cr']:.2%} | "
            f"{row['equalweight_control_it25']['cr']:.2%} | "
            f"{row['weighted_it25']['cr']:.2%} | {effect['delta']:+.2%} | "
            f"{effect['z']:+.2f} |"
        )
    lines.extend(
        [
            "",
            f"Weight-specific target pass: `{summary['weight_specific_target_pass']}`",
            f"Weight-specific retention pass: `{summary['weight_specific_retention_pass']}`",
            f"Weight intervention accepted: `{summary['weight_intervention_accepted']}`",
            f"Next action: `{summary['next_action']}`",
            "",
            "Single evaluator seed diagnostic only. No extension, graduation, training, or SA6 is authorized.",
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
