"""Export the frozen training-side 19x19 differential-drive contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_SKRL_DIR = _SCRIPT_DIR.parent
if str(_SKRL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKRL_DIR))

from rnn_car_wdclean.reward_diagnostics import (
    decode_discrete_drive_action_grid,
)


REPO = Path(__file__).resolve().parents[4]
ACTION_TERM = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py"
)
DECODER_HELPER = Path(__file__).resolve().parent / "reward_diagnostics.py"
DEFAULT_OUTPUT = REPO / "docs/freeze/sa1_action_contract_v1.json"

PARAMS = {
    "num_bins": 19,
    "control_dt_s": 0.2,
    "max_linear_velocity_mps": 1.0,
    "reverse_velocity_scale": 0.2,
    "max_linear_accel_mps2": 0.5,
    "max_angular_velocity_rad_s": 1.2,
    "max_angular_accel_rad_s2": 3.0,
    "action_history_accel_normalizer_mps2": 0.5,
    "action_history_omega_normalizer_rad_s": 1.2,
    "action_history_clip_abs": 2.0,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decode_grid(
    current_velocity: float,
    current_omega: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return decode_discrete_drive_action_grid(
        torch.tensor([current_velocity]),
        torch.tensor([current_omega]),
        num_bins=PARAMS["num_bins"],
        dt=PARAMS["control_dt_s"],
        max_linear_velocity=PARAMS["max_linear_velocity_mps"],
        reverse_velocity_scale=PARAMS["reverse_velocity_scale"],
        max_linear_accel=PARAMS["max_linear_accel_mps2"],
        max_angular_velocity=PARAMS["max_angular_velocity_rad_s"],
        max_angular_accel=PARAMS["max_angular_accel_rad_s2"],
    )


def _round(value: float) -> float:
    return round(float(value), 6)


def _decode_one(
    current_velocity: float,
    current_omega: float,
    linear_index: int,
    angular_index: int,
) -> dict:
    linear, angular = _decode_grid(current_velocity, current_omega)
    next_velocity = float(linear[0, linear_index, angular_index])
    next_omega = float(angular[0, linear_index, angular_index])
    return {
        "linear_index": linear_index,
        "angular_index": angular_index,
        "current_velocity_mps": _round(current_velocity),
        "current_omega_rad_s": _round(current_omega),
        "issued_linear_accel_mps2": _round(
            (next_velocity - current_velocity) / PARAMS["control_dt_s"]
        ),
        "issued_velocity_mps": _round(next_velocity),
        "issued_omega_rad_s": _round(next_omega),
    }


def _simulate_sequence(name: str, actions: list[tuple[int, int]]) -> dict:
    velocity = 0.0
    omega = 0.0
    steps = []
    history = [[0.0, 0.0], [0.0, 0.0]]
    for step, (linear_index, angular_index) in enumerate(actions):
        decoded = _decode_one(
            velocity, omega, linear_index, angular_index
        )
        history = [
            history[-1],
            [
                decoded["issued_linear_accel_mps2"],
                decoded["issued_omega_rad_s"],
            ],
        ]
        decoded["step"] = step
        decoded["history_after_step"] = history
        steps.append(decoded)
        velocity = decoded["issued_velocity_mps"]
        omega = decoded["issued_omega_rad_s"]
    return {"name": name, "steps": steps}


def build_contract() -> dict:
    zero_grid = [
        _decode_one(0.0, 0.0, linear_index, angular_index)
        for linear_index in range(PARAMS["num_bins"])
        for angular_index in range(PARAMS["num_bins"])
    ]
    sequences = [
        _simulate_sequence(
            "reverse_saturation",
            [(0, 9)] * 4,
        ),
        _simulate_sequence(
            "forward_saturation",
            [(18, 9)] * 12,
        ),
        _simulate_sequence(
            "positive_omega_slew",
            [(9, 18)] * 3,
        ),
        _simulate_sequence(
            "omega_sign_flip",
            [(9, 18)] * 2 + [(9, 0)] * 4,
        ),
    ]
    return {
        "schema": "isaaclab.sa1_action_contract.v1",
        "source": {
            "action_term": str(ACTION_TERM.relative_to(REPO)),
            "action_term_sha256": _sha256(ACTION_TERM),
            "decoder_helper": str(DECODER_HELPER.relative_to(REPO)),
            "decoder_helper_sha256": _sha256(DECODER_HELPER),
        },
        "parameters": PARAMS,
        "semantics": {
            "action": "[linear_acceleration_index, angular_velocity_index]",
            "issued_history": (
                "two previous [linear_accel_mps2, omega_rad_s] values after "
                "decode/slew and before actuator delay"
            ),
            "issued_history_normalized": (
                "[linear_accel/0.5, omega/1.2] controls from the same decode "
                "invocation that constructs queued [velocity, omega], "
                "newest to oldest, "
                "clipped to [-2,2]"
            ),
            "reverse_bound_location": "inside decoder",
            "actuator_delay_location": "after decode/slew/history",
        },
        "zero_state_all_361_actions": zero_grid,
        "stateful_sequences": sequences,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_contract(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
