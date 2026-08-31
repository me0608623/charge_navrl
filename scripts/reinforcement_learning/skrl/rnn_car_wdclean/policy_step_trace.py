"""Policy-only, delay-aware step recorder for corridor diagnostics.

The recorder is observational. It rejects any run where the effective action
differs from the policy action and reconciles the fixed d1 command queue before
the trace can be treated as valid evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "policy_step_trace/v1"
K8_SCHEMA = "policy_step_trace/v2-k8-observability"
DELAY_TOLERANCE = 1.0e-6


class PolicyStepTraceRecorder:
    """Accumulate one policy-state/action/transition tensor row per step."""

    def __init__(
        self,
        num_envs: int,
        num_dynamic_obstacles: int,
        *,
        expected_delay_steps: int,
        include_policy_inputs: bool = False,
    ) -> None:
        if num_envs <= 0 or num_dynamic_obstacles <= 0:
            raise ValueError("trace dimensions must be positive")
        if expected_delay_steps != 1:
            raise ValueError("policy step trace currently supports fixed d1 only")
        self.num_envs = int(num_envs)
        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        self.expected_delay_steps = int(expected_delay_steps)
        self.include_policy_inputs = bool(include_policy_inputs)
        self._policy_input_dims: dict[str, int] = {}
        self._rows: dict[str, list[np.ndarray]] = {}
        self._steps: list[int] = []
        self._previous_pre_delay: np.ndarray | None = None
        self._previous_episode_step: np.ndarray | None = None
        self._delay_samples = 0
        self._delay_errors = 0
        self._delay_max_error = 0.0
        self._done_count = 0
        self._bad_cause_rows = 0

    @staticmethod
    def _numpy(value: torch.Tensor, name: str) -> np.ndarray:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        array = value.detach().cpu().numpy()
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")
        return array

    def _shape(self, array: np.ndarray, expected: tuple[int, ...], name: str) -> None:
        if array.shape != expected:
            raise ValueError(
                f"{name} has shape {array.shape}, expected {expected}"
            )

    def record_tensor_batch(
        self,
        *,
        step: int,
        episode_step: torch.Tensor,
        policy_actions: torch.Tensor,
        effective_actions: torch.Tensor,
        pre_delay_command: torch.Tensor,
        post_delay_command: torch.Tensor,
        robot_xy_m: torch.Tensor,
        robot_yaw_rad: torch.Tensor,
        robot_velocity_body_mps: torch.Tensor,
        dynamic_positions_m: torch.Tensor,
        dynamic_velocities_mps: torch.Tensor,
        dynamic_valid: torch.Tensor,
        patrol_waypoint_index: torch.Tensor,
        patrol_pause_remaining: torch.Tensor,
        patrol_target_m: torch.Tensor,
        patrol_active: torch.Tensor,
        done: torch.Tensor,
        termination_cause: torch.Tensor,
        policy_current_input: torch.Tensor | None = None,
        policy_k8_input: torch.Tensor | None = None,
        policy_representation: torch.Tensor | None = None,
    ) -> None:
        """Record pre-state data and the command produced by its transition."""
        e, n = self.num_envs, self.num_dynamic_obstacles
        arrays = {
            "episode_step": self._numpy(episode_step, "episode_step"),
            "policy_action_indices": self._numpy(
                policy_actions.round().to(torch.int8), "policy_actions"
            ),
            "effective_action_indices": self._numpy(
                effective_actions.round().to(torch.int8), "effective_actions"
            ),
            "pre_delay_command_mps_rad_s": self._numpy(
                pre_delay_command.to(torch.float32), "pre_delay_command"
            ),
            "post_delay_command_mps_rad_s": self._numpy(
                post_delay_command.to(torch.float32), "post_delay_command"
            ),
            "robot_xy_m": self._numpy(robot_xy_m.to(torch.float32), "robot_xy_m"),
            "robot_yaw_rad": self._numpy(
                robot_yaw_rad.to(torch.float32), "robot_yaw_rad"
            ),
            "robot_velocity_body_mps": self._numpy(
                robot_velocity_body_mps.to(torch.float32),
                "robot_velocity_body_mps",
            ),
            "dynamic_positions_m": self._numpy(
                dynamic_positions_m.to(torch.float32), "dynamic_positions_m"
            ),
            "dynamic_velocities_mps": self._numpy(
                dynamic_velocities_mps.to(torch.float32),
                "dynamic_velocities_mps",
            ),
            "dynamic_valid": self._numpy(
                dynamic_valid.to(torch.bool), "dynamic_valid"
            ),
            "patrol_waypoint_index": self._numpy(
                patrol_waypoint_index.to(torch.int16), "patrol_waypoint_index"
            ),
            "patrol_pause_remaining": self._numpy(
                patrol_pause_remaining.to(torch.int16),
                "patrol_pause_remaining",
            ),
            "patrol_target_m": self._numpy(
                patrol_target_m.to(torch.float32), "patrol_target_m"
            ),
            "patrol_active": self._numpy(
                patrol_active.to(torch.bool), "patrol_active"
            ),
            "done": self._numpy(done.reshape(-1).to(torch.bool), "done"),
            "termination_cause": self._numpy(
                termination_cause.reshape(-1).to(torch.int8),
                "termination_cause",
            ),
        }
        policy_inputs = {
            "policy_current_input": policy_current_input,
            "policy_k8_input": policy_k8_input,
            "policy_representation": policy_representation,
        }
        if self.include_policy_inputs:
            missing = [name for name, value in policy_inputs.items() if value is None]
            if missing:
                raise ValueError(
                    "K8 observability trace is missing policy inputs: "
                    + ", ".join(missing)
                )
            for name, value in policy_inputs.items():
                assert value is not None
                array = self._numpy(value.to(torch.float16), name)
                if array.ndim != 2 or array.shape[0] != e:
                    raise ValueError(
                        f"{name} has shape {array.shape}, expected [{e}, D]"
                    )
                dimension = int(array.shape[1])
                previous = self._policy_input_dims.setdefault(name, dimension)
                if previous != dimension:
                    raise ValueError(
                        f"{name} dimension changed from {previous} to {dimension}"
                    )
                arrays[name] = array
        elif any(value is not None for value in policy_inputs.values()):
            raise ValueError(
                "policy inputs were supplied without include_policy_inputs=True"
            )
        expected_shapes = {
            "episode_step": (e,),
            "policy_action_indices": (e, 2),
            "effective_action_indices": (e, 2),
            "pre_delay_command_mps_rad_s": (e, 2),
            "post_delay_command_mps_rad_s": (e, 2),
            "robot_xy_m": (e, 2),
            "robot_yaw_rad": (e,),
            "robot_velocity_body_mps": (e, 2),
            "dynamic_positions_m": (e, n, 2),
            "dynamic_velocities_mps": (e, n, 2),
            "dynamic_valid": (e, n),
            "patrol_waypoint_index": (e, n),
            "patrol_pause_remaining": (e, n),
            "patrol_target_m": (e, n, 2),
            "patrol_active": (e, n),
            "done": (e,),
            "termination_cause": (e,),
        }
        for name, expected in expected_shapes.items():
            self._shape(arrays[name], expected, name)

        if not np.array_equal(
            arrays["policy_action_indices"], arrays["effective_action_indices"]
        ):
            raise RuntimeError("policy step trace modified policy action")

        episode = arrays["episode_step"].astype(np.int64, copy=False)
        pre = arrays["pre_delay_command_mps_rad_s"]
        post = arrays["post_delay_command_mps_rad_s"]
        if self._previous_pre_delay is not None:
            assert self._previous_episode_step is not None
            comparable = episode == self._previous_episode_step + 1
            if comparable.any():
                errors = np.abs(post[comparable] - self._previous_pre_delay[comparable])
                self._delay_samples += int(comparable.sum())
                maximum = float(errors.max(initial=0.0))
                self._delay_max_error = max(self._delay_max_error, maximum)
                self._delay_errors += int(
                    (errors > DELAY_TOLERANCE).any(axis=1).sum()
                )
        self._previous_pre_delay = pre.copy()
        self._previous_episode_step = episode.copy()

        done_array = arrays["done"]
        cause = arrays["termination_cause"]
        self._done_count += int(done_array.sum())
        self._bad_cause_rows += int(((done_array & (cause == 0)) | (~done_array & (cause != 0))).sum())

        self._steps.append(int(step))
        for name, array in arrays.items():
            self._rows.setdefault(name, []).append(array.copy())

    def _stacked(self) -> dict[str, np.ndarray]:
        if not self._steps:
            raise RuntimeError("policy step trace has no records")
        arrays = {
            name: np.stack(rows, axis=0) for name, rows in self._rows.items()
        }
        arrays["rollout_step"] = np.asarray(self._steps, dtype=np.int32)
        return arrays

    def write(self, output: str | Path, *, metadata: dict[str, Any]) -> dict:
        arrays = self._stacked()
        delay_ok = self._delay_samples > 0 and self._delay_errors == 0
        expected_records = len(self._steps) * self.num_envs
        policy_identity_ok = np.array_equal(
            arrays["policy_action_indices"], arrays["effective_action_indices"]
        )
        cause_ok = self._bad_cause_rows == 0
        report = {
            "schema": SCHEMA,
            "counts": {
                "steps": len(self._steps),
                "environment_records": expected_records,
                "completed_episodes": self._done_count,
            },
            "self_check": {
                "policy_identity_ok": bool(policy_identity_ok),
                "delay_alignment_samples": self._delay_samples,
                "delay_alignment_errors": self._delay_errors,
                "delay_alignment_max_abs_error": self._delay_max_error,
                "delay_alignment_ok": bool(delay_ok),
                "termination_cause_reconciliation_ok": bool(cause_ok),
                "reconciliation_ok": bool(
                    policy_identity_ok and delay_ok and cause_ok
                ),
            },
        }
        enriched_metadata = {
            **metadata,
            "schema": K8_SCHEMA if self.include_policy_inputs else SCHEMA,
            "array_layout": "[rollout_step, env, ...]",
            "dt_s": 0.2,
            "expected_delay_steps": self.expected_delay_steps,
            "teacher_replaced_policy_actions": False,
            "policy_inputs": {
                "included": self.include_policy_inputs,
                "storage_dtype": "float16" if self.include_policy_inputs else None,
                "dimensions": self._policy_input_dims,
                "current_input": "normalized current policy observation",
                "k8_input": (
                    "exact normalized extractor input: current observation plus "
                    "seven prior 72-bin LiDAR frames"
                ),
                "representation": (
                    "exact policy-head input before logits: current observation plus "
                    "K8 CNN embedding"
                ),
            },
            "timing": {
                "state": "sampled when policy selected the action",
                "pre_delay_command": "decoded command entering d1 queue",
                "post_delay_command": "command applied by resulting transition",
            },
            "self_check": report["self_check"],
        }
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            **arrays,
            metadata_json=np.asarray(json.dumps(enriched_metadata)),
        )
        return report
