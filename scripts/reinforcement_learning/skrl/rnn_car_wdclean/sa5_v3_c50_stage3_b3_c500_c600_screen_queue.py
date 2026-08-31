"""Run the frozen 12-cell SA5 B3 c500/c550/c600 comparison."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import sa5_v3_c50_stage3_b3_c500_c600_screen as protocol  # noqa: E402
import sa5_v3_provisional_pilot_screen_queue as base_queue  # noqa: E402


RUNNER = HERE / "run_sa5_v3_c50_stage3_b3_c500_c600_screen_cell.py"
FREEZE = (
    REPO / "docs/freeze/sa5_v3_c50_stage3_b3_c500_c600_screen_v1.json"
)
SOURCE_PATHS = (
    Path(__file__).resolve(),
    Path(protocol.__file__).resolve(),
    RUNNER,
    FREEZE,
    protocol.CHECKPOINT_LOCK,
    protocol.AUTHORIZATION,
    HERE / "sa5_v3_c50_stage3_b3_cont25_screen.py",
    HERE / "sa5_v3_c50_stage3_phase_a_screen.py",
    HERE / "run_sa5_v3_provisional_pilot_screen_cell.py",
    Path(base_queue.__file__).resolve(),
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
    "e2e_sa5_v3_c50_stage3_b3_cont500_from_it100.py",
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


def verify_frozen_protocol() -> dict:
    if not FREEZE.is_file():
        raise FileNotFoundError(FREEZE)
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    current = protocol.screen_protocol()
    if (
        frozen.get("schema")
        != "sa5_v3_c50_stage3_b3_c500_c600_screen_freeze/v1"
        or frozen.get("protocol_sha256") != current["sha256"]
        or frozen.get("authorization_sha256")
        != protocol.sha256_of(protocol.AUTHORIZATION)
        or frozen.get("checkpoint_lock_sha256")
        != protocol.sha256_of(protocol.CHECKPOINT_LOCK)
        or int(frozen.get("cell_count", -1)) != 12
        or bool(frozen.get("starts_training"))
        or bool(frozen.get("starts_phase_b"))
        or bool(frozen.get("selects_parent"))
        or bool(frozen.get("graduates_sa5"))
        or bool(frozen.get("starts_sa6"))
    ):
        raise RuntimeError("c500-c600 screen differs from frozen protocol")
    return current


def build_checkpoint_manifest() -> dict:
    checkpoints = {}
    for spec in protocol.CANDIDATE_SPECS:
        path = protocol.checkpoint_path(spec).resolve()
        digest = protocol.sha256_of(path)
        if digest != spec["expected_sha256"]:
            raise RuntimeError(f"{spec['name']} checkpoint hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            int(payload.get("iteration", -1)) != spec["embedded_iteration"]
            or int(payload.get("total_steps", -1))
            != spec["embedded_total_steps"]
        ):
            raise RuntimeError(f"{spec['name']} checkpoint ledger mismatch")
        optimizer = payload.get("charge_opt_rl") or {}
        if not optimizer.get("state") or not optimizer.get("param_groups"):
            raise RuntimeError(f"{spec['name']} lacks resumable RL optimizer")
        checkpoints[spec["name"]] = {
            "path": str(path),
            "sha256": digest,
            "embedded_iteration": spec["embedded_iteration"],
            "embedded_total_steps": spec["embedded_total_steps"],
            "conceptual_iteration": spec["conceptual_iteration"],
            "rl_optimizer_state_entries": len(optimizer["state"]),
        }
    return {
        "schema": "sa5_v3_c50_stage3_b3_c500_c600_checkpoint_manifest/v1",
        "protocol_sha256": protocol.screen_protocol()["sha256"],
        "checkpoints": checkpoints,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# SA5 B3 c500/c550/c600 fixed 12-cell screen",
        "",
        f"Status: `{summary['status']}`",
        "",
        "| checkpoint | P060 0S1D SR/CR/TO | P060 1S1D SR/CR/TO | worst CR | P060 | P080 retention | candidate |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in summary["candidate_results"]:
        rows = row["rows"]
        p0 = rows["p060_0s1d"]
        p1 = rows["p060_1s1d"]
        lines.append(
            f"| {row['checkpoint_name']} | "
            f"{p0['sr']:.2%}/{p0['cr']:.2%}/{p0['to']:.2%} | "
            f"{p1['sr']:.2%}/{p1['cr']:.2%}/{p1['to']:.2%} | "
            f"{row['worst_p060_cr']:.2%} | "
            f"{'PASS' if row['p060_absolute_pass'] else 'FAIL'} | "
            f"{'PASS' if row['p080_retention_pass'] else 'FAIL'} | "
            f"{'PASS' if row['candidate_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"P080 reference: `{summary['p080_retention_reference']}`",
            f"Selected candidate: `{summary['sa5_candidate']}`",
            f"Tie group: `{summary['tie_group']}`",
            f"Next action: `{summary['next_action']}`",
            "",
            "No training, Phase B, parent selection, SA5 graduation, or SA6 start is automatic.",
        ]
    )
    return "\n".join(lines) + "\n"


base_queue.protocol = protocol
base_queue.RUNNER = RUNNER
base_queue.FREEZE = FREEZE
base_queue.SOURCE_PATHS = SOURCE_PATHS
base_queue.verify_frozen_protocol = verify_frozen_protocol
base_queue.build_checkpoint_manifest = build_checkpoint_manifest
base_queue.render_markdown = render_markdown


def main(argv: list[str] | None = None) -> int:
    return base_queue.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
