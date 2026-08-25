"""Frozen decision rules for the teacher goal-denominator closed-loop A/B."""

from __future__ import annotations

import hashlib
import json
import math


ARMS = ("historical_r1", "candidate_r10")
FLOOR_BY_ARM = {"historical_r1": 1.0, "candidate_r10": 10.0}
SEEDS = (515, 616, 717)
NUM_ENVS = 64
ROLLOUT_STEPS = 700
MIN_EPISODES_PER_CELL = 500

SR_MIN = 0.90
CR_MAX = 0.10
TO_MAX = 0.05
NONINFERIOR_MARGIN = 0.005
NEAR_GOAL_COMPLETION_MIN = 0.90
NEAR_GOAL_COMPLETION_MARGIN = 0.02
NEAR_GOAL_P05_IMPROVEMENT_M = 0.05
MAX_SPEED_LOSS_MPS = 0.10
MAX_STOP_FRACTION_INCREASE = 0.10
MAX_PHASE_COLLISIONS_PER_FRAME_INCREASE = 0.005


def frozen_protocol_payload() -> dict[str, object]:
    return {
        "schema": "teacher_goal_denominator_ab_protocol/v1",
        "purpose": (
            "teacher-only closed-loop comparison of historical goal cost "
            "denominator R=1 m against candidate R=10 m"
        ),
        "arms": FLOOR_BY_ARM,
        "run_order": [
            [515, "historical_r1"],
            [515, "candidate_r10"],
            [616, "candidate_r10"],
            [616, "historical_r1"],
            [717, "historical_r1"],
            [717, "candidate_r10"],
        ],
        "fixed_cell": {
            "checkpoint_lineage": "historical SA5 c12 observation/fallback policy",
            "task": "Isaac-Navigation-Charge-VLP16-Curriculum-WD",
            "curriculum_version": "warp_drive_e2e_final20_v1",
            "stage": 1,
            "deterministic": True,
            "teacher_override": True,
            "num_envs": NUM_ENVS,
            "steps": ROLLOUT_STEPS,
            "corridor_free_width_m": 4.0,
            "corridor_length_m": 10.0,
            "static_obstacles": 2,
            "dynamic_obstacles": 1,
            "dynamic_motion_mode": "lateral",
            "dynamic_speed_range_m_s": [0.30, 0.60],
            "dynamic_pause_mode": "default",
            "dynamic_pause_steps_range": [0, 5],
            "actuator_dr": False,
            "near_goal_threshold_m": 3.0,
            "near_hard_clearance_threshold_m": 0.11,
        },
        "validity": {
            "minimum_episodes_per_cell": MIN_EPISODES_PER_CELL,
            "all_outputs_finite": True,
            "source_fingerprint_stable": True,
            "checkpoint_sha256_stable": True,
            "motion_phase_accounting_reconciles": True,
        },
        "candidate_pass_rules": {
            "each_candidate_cell_gate_pass": True,
            "pooled_sr_min": SR_MIN,
            "pooled_cr_max": CR_MAX,
            "pooled_to_max": TO_MAX,
            "pooled_sr_delta_min": -NONINFERIOR_MARGIN,
            "pooled_cr_delta_max": NONINFERIOR_MARGIN,
            "pooled_to_delta_max": NONINFERIOR_MARGIN,
            "near_goal_completion_min": NEAR_GOAL_COMPLETION_MIN,
            "near_goal_completion_delta_min": -NEAR_GOAL_COMPLETION_MARGIN,
            "near_goal_clearance_p05_improvement_m": (
                NEAR_GOAL_P05_IMPROVEMENT_M
            ),
            "clearance_improvement_required_seed_count": 2,
            "speed_abs_mean_delta_min_mps": -MAX_SPEED_LOSS_MPS,
            "stop_fraction_delta_max": MAX_STOP_FRACTION_INCREASE,
            "paused_and_post_switch_collisions_per_frame_delta_max": (
                MAX_PHASE_COLLISIONS_PER_FRAME_INCREASE
            ),
        },
        "decision_boundary": (
            "Passing authorizes creation of a new READY_NOT_RUN SA5 config. "
            "It does not authorize training, SA6, or modification of old artifacts."
        ),
    }


def frozen_protocol() -> dict[str, object]:
    payload = frozen_protocol_payload()
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **payload,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _finite_rate(value: object, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"invalid {name}={value!r}")
    return number


def extract_cell_metrics(teacher_report: dict) -> dict[str, object]:
    corridor = teacher_report["closed_loop_outcome"]
    clearance = teacher_report["near_goal_selected_predicted_clearance"]
    near = teacher_report["near_goal_episode_outcomes"]
    phase = corridor["motion_phase_audit"]
    if not phase or not phase.get("applicable"):
        raise ValueError("dynamic-pause phase audit is missing or inapplicable")
    phase_collision_sum = sum(
        int(bucket["collision_count"])
        for bucket in phase["phases"].values()
    )
    phase_exposure_sum = sum(
        int(bucket["frame_exposure"])
        for bucket in phase["phases"].values()
    )
    if phase_collision_sum != int(phase["total_dynamic_slot_collisions"]):
        raise ValueError("motion-phase collision accounting mismatch")
    if phase_exposure_sum != int(phase["total_dynamic_slot_frames"]):
        raise ValueError("motion-phase exposure accounting mismatch")

    episodes = int(corridor["episodes"])
    if episodes < MIN_EPISODES_PER_CELL:
        raise ValueError(
            f"too few episodes: {episodes} < {MIN_EPISODES_PER_CELL}"
        )
    sr = _finite_rate(corridor["success_rate"], "success_rate")
    cr = _finite_rate(corridor["collision_rate"], "collision_rate")
    to = _finite_rate(corridor["timeout_rate"], "timeout_rate")
    if abs(sr + cr + to - 1.0) > 1.0e-6:
        raise ValueError("SR/CR/TO do not reconcile")

    near_entered = int(near["entered_episodes"])
    if near_entered <= 0:
        raise ValueError("no episode entered the registered near-goal band")
    p05 = clearance["p05_m"]
    near_hard = clearance["near_hard_fraction"]
    if p05 is None or near_hard is None:
        raise ValueError("near-goal clearance distribution is empty")

    phase_rates = {}
    for name, bucket in phase["phases"].items():
        exposure = int(bucket["frame_exposure"])
        collisions = int(bucket["collision_count"])
        phase_rates[name] = {
            "exposure": exposure,
            "collisions": collisions,
            "collisions_per_frame": collisions / max(exposure, 1),
        }

    return {
        "episodes": episodes,
        "successes": int(round(sr * episodes)),
        "collisions": int(round(cr * episodes)),
        "timeouts": int(round(to * episodes)),
        "sr": sr,
        "cr": cr,
        "to": to,
        "wall_cr": _finite_rate(
            corridor["wall_collision_rate"], "wall_collision_rate"
        ),
        "obstacle_cr": _finite_rate(
            corridor["obstacle_collision_rate"],
            "obstacle_collision_rate",
        ),
        "gate_pass": bool(corridor["gate_pass"]),
        "action_frames": int(corridor["action_frames"]),
        "linear_speed_abs_mean_mps": float(
            corridor["linear_speed_abs_mean_mps"]
        ),
        "stop_command_fraction": _finite_rate(
            corridor["stop_command_fraction"], "stop_command_fraction"
        ),
        "near_goal_entered_episodes": near_entered,
        "near_goal_successes": int(near["success"]),
        "near_goal_collisions": int(near["wall_collision"])
        + int(near["obstacle_collision"]),
        "near_goal_timeouts": int(near["timeout"]),
        "near_goal_completion_rate": _finite_rate(
            near["completion_rate"], "near_goal_completion_rate"
        ),
        "near_goal_clearance_samples": int(clearance["samples"]),
        "near_goal_clearance_p05_m": float(p05),
        "near_goal_clearance_mean_m": float(clearance["mean_m"]),
        "near_goal_near_hard_fraction": _finite_rate(
            near_hard, "near_goal_near_hard_fraction"
        ),
        "phase": phase_rates,
    }


def _pool(cells: list[dict[str, object]]) -> dict[str, object]:
    episodes = sum(int(cell["episodes"]) for cell in cells)
    action_frames = sum(int(cell["action_frames"]) for cell in cells)
    near_episodes = sum(
        int(cell["near_goal_entered_episodes"]) for cell in cells
    )
    clearance_samples = sum(
        int(cell["near_goal_clearance_samples"]) for cell in cells
    )

    def weighted(key: str, weight_key: str, denominator: int) -> float:
        return sum(
            float(cell[key]) * int(cell[weight_key]) for cell in cells
        ) / max(denominator, 1)

    phase = {}
    for name in cells[0]["phase"]:
        exposure = sum(int(cell["phase"][name]["exposure"]) for cell in cells)
        collisions = sum(
            int(cell["phase"][name]["collisions"]) for cell in cells
        )
        phase[name] = {
            "exposure": exposure,
            "collisions": collisions,
            "collisions_per_frame": collisions / max(exposure, 1),
        }

    successes = sum(int(cell["successes"]) for cell in cells)
    collisions = sum(int(cell["collisions"]) for cell in cells)
    timeouts = sum(int(cell["timeouts"]) for cell in cells)
    return {
        "episodes": episodes,
        "sr": successes / episodes,
        "cr": collisions / episodes,
        "to": timeouts / episodes,
        "wall_cr": weighted("wall_cr", "episodes", episodes),
        "obstacle_cr": weighted("obstacle_cr", "episodes", episodes),
        "action_frames": action_frames,
        "linear_speed_abs_mean_mps": weighted(
            "linear_speed_abs_mean_mps", "action_frames", action_frames
        ),
        "stop_command_fraction": weighted(
            "stop_command_fraction", "action_frames", action_frames
        ),
        "near_goal_entered_episodes": near_episodes,
        "near_goal_completion_rate": (
            sum(int(cell["near_goal_successes"]) for cell in cells)
            / max(near_episodes, 1)
        ),
        "near_goal_collision_rate": (
            sum(int(cell["near_goal_collisions"]) for cell in cells)
            / max(near_episodes, 1)
        ),
        "near_goal_timeout_rate": (
            sum(int(cell["near_goal_timeouts"]) for cell in cells)
            / max(near_episodes, 1)
        ),
        "near_goal_clearance_samples": clearance_samples,
        "near_goal_clearance_mean_m": weighted(
            "near_goal_clearance_mean_m",
            "near_goal_clearance_samples",
            clearance_samples,
        ),
        "near_goal_near_hard_fraction": weighted(
            "near_goal_near_hard_fraction",
            "near_goal_clearance_samples",
            clearance_samples,
        ),
        "phase": phase,
    }


def compare(cells_by_arm: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    if set(cells_by_arm) != set(ARMS):
        raise ValueError("comparison requires exactly the frozen two arms")
    for arm in ARMS:
        if len(cells_by_arm[arm]) != len(SEEDS):
            raise ValueError(f"arm {arm} must contain {len(SEEDS)} cells")

    pooled = {arm: _pool(cells_by_arm[arm]) for arm in ARMS}
    a = pooled["historical_r1"]
    b = pooled["candidate_r10"]
    delta = {
        key: float(b[key]) - float(a[key])
        for key in (
            "sr",
            "cr",
            "to",
            "wall_cr",
            "obstacle_cr",
            "linear_speed_abs_mean_mps",
            "stop_command_fraction",
            "near_goal_completion_rate",
            "near_goal_collision_rate",
            "near_goal_timeout_rate",
            "near_goal_clearance_mean_m",
            "near_goal_near_hard_fraction",
        )
    }
    per_seed = []
    clearance_improved = 0
    near_hard_not_worse = 0
    for index, seed in enumerate(SEEDS):
        seed_a = cells_by_arm["historical_r1"][index]
        seed_b = cells_by_arm["candidate_r10"][index]
        p05_delta = (
            float(seed_b["near_goal_clearance_p05_m"])
            - float(seed_a["near_goal_clearance_p05_m"])
        )
        hard_delta = (
            float(seed_b["near_goal_near_hard_fraction"])
            - float(seed_a["near_goal_near_hard_fraction"])
        )
        if p05_delta >= NEAR_GOAL_P05_IMPROVEMENT_M:
            clearance_improved += 1
        if hard_delta <= 0.0:
            near_hard_not_worse += 1
        per_seed.append(
            {
                "seed": seed,
                "near_goal_clearance_p05_delta_m": p05_delta,
                "near_goal_near_hard_fraction_delta": hard_delta,
                "sr_delta": float(seed_b["sr"]) - float(seed_a["sr"]),
                "cr_delta": float(seed_b["cr"]) - float(seed_a["cr"]),
                "to_delta": float(seed_b["to"]) - float(seed_a["to"]),
            }
        )

    phase_noninferior = {}
    for phase_name in ("paused", "post_switch_or_resume_1s"):
        phase_delta = (
            float(b["phase"][phase_name]["collisions_per_frame"])
            - float(a["phase"][phase_name]["collisions_per_frame"])
        )
        phase_noninferior[phase_name] = {
            "delta": phase_delta,
            "pass": phase_delta
            <= MAX_PHASE_COLLISIONS_PER_FRAME_INCREASE,
        }

    checks = {
        "all_candidate_cells_gate_pass": all(
            bool(cell["gate_pass"])
            for cell in cells_by_arm["candidate_r10"]
        ),
        "pooled_absolute_gate": (
            float(b["sr"]) >= SR_MIN
            and float(b["cr"]) <= CR_MAX
            and float(b["to"]) <= TO_MAX
        ),
        "pooled_outcome_noninferior": (
            delta["sr"] >= -NONINFERIOR_MARGIN
            and delta["cr"] <= NONINFERIOR_MARGIN
            and delta["to"] <= NONINFERIOR_MARGIN
        ),
        "near_goal_completion": (
            float(b["near_goal_completion_rate"])
            >= NEAR_GOAL_COMPLETION_MIN
            and delta["near_goal_completion_rate"]
            >= -NEAR_GOAL_COMPLETION_MARGIN
        ),
        "near_goal_clearance": (
            clearance_improved >= 2 and near_hard_not_worse >= 2
        ),
        "speed_and_stop_noninferior": (
            delta["linear_speed_abs_mean_mps"] >= -MAX_SPEED_LOSS_MPS
            and delta["stop_command_fraction"]
            <= MAX_STOP_FRACTION_INCREASE
        ),
        "dynamic_pause_noninferior": all(
            item["pass"] for item in phase_noninferior.values()
        ),
    }
    return {
        "pooled": pooled,
        "delta_candidate_minus_historical": delta,
        "per_seed": per_seed,
        "phase_noninferiority": phase_noninferior,
        "checks": checks,
        "candidate_teacher_pass": all(checks.values()),
    }
