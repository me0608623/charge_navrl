"""Run the frozen 21-cell c300-c600 SA4-v3 Gate comparison."""

from __future__ import annotations

from pathlib import Path

import sa4_v3_checkpoint_screen_queue as implementation
import sa4_v3_c300_c600_gate as protocol


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNNER = HERE / "run_sa4_v3_c300_c600_gate_cell.py"
FREEZE = REPO / "docs/freeze/sa4_v3_c300_c600_gate_v1.json"

implementation.protocol = protocol
implementation.RUNNER = RUNNER
implementation.FREEZE = FREEZE
implementation.SOURCE_PATHS = (
    HERE / "sa4_v3_c300_c600_gate.py",
    RUNNER,
    Path(__file__).resolve(),
    FREEZE,
    HERE / "sa4_v3_checkpoint_screen.py",
    HERE / "run_sa4_v3_checkpoint_screen_cell.py",
    HERE / "run_sa3_gate_bc.py",
    HERE / "fixed_actuator_eval.py",
    HERE / "run_sa4_d9_noise_closed_loop_ab.py",
    HERE / "run_sa5_joint_retention_gates.py",
    protocol.SOURCE_GATE_FREEZE,
    protocol.CONTINUATION_AUTHORIZATION,
    REPO / "docs/freeze/sim2real_speed_density_curriculum_v3_20260821.json",
    REPO
    / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs/"
    "sim2real_speed_density_curriculum_v3.py",
    REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
    REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/actions/discrete_differential_drive.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/corridor_density.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/long_corridor_replay.py",
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/"
    "config/charge_skrl/mdp/events/long_corridor_replay_geometry.py",
)


def main(argv: list[str] | None = None) -> int:
    return implementation.main(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

