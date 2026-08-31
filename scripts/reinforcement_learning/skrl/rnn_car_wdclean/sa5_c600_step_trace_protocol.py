"""Frozen protocol for accepted-c600 policy-only event traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
CHECKPOINT = (
    REPO
    / "logs/rnn_car/sa5_v3_c50_stage3_b3_cont500_from_it100_ne1024_s42_p500_r1"
    / "checkpoint_64000.pt"
)
CHECKPOINT_SHA256 = "c1a24684eb787a0865b007a2a712b744f0f74c39f812c62b9b18d4ed893c271a"
SCENARIOS = ("lateral", "random_2d")
SEED = 818
NUM_ENVS = 64
ROLLOUT_STEPS = 4000


def payload() -> dict[str, object]:
    return {
        "schema": "sa5_c600_policy_step_trace_protocol/v1",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_status": (
            "human-accepted c600 development parent; prior machine Gate waiver "
            "remains disclosed and is not rewritten by this diagnostic"
        ),
        "purpose": {
            "lateral": (
                "measure actual robot departure side and d1 command timing after "
                "an unambiguous pedestrian centerline crossing"
            ),
            "random_2d": (
                "measure issued/applied stop-turn timing around patrol waypoint "
                "switches and direction reversals"
            ),
        },
        "fixed_cell": {
            "stage": 5,
            "seed": SEED,
            "deterministic": True,
            "num_envs": NUM_ENVS,
            "steps": ROLLOUT_STEPS,
            "static_obstacles": 2,
            "dynamic_obstacles": 2,
            "corridor_free_width_m": 4.2,
            "dynamic_speed_range_m_s": [0.25, 0.45],
            "pause_steps": [0, 5],
            "speed_rate": 0.7,
            "vlp16_noise_mode": "full",
            "lidar_distractor_eligibility": "valid_return_only",
            "actuator_delay_steps": [1, 1],
            "actuator_velocity_scale": [1.0, 1.0],
            "actuator_motor_lag": 1.0,
        },
        "policy_contract": {
            "training": False,
            "teacher": False,
            "shield_mode": "baseline_identity",
            "action_override": False,
            "d1_reconciliation_required": True,
        },
        "event_windows": {
            "lateral_post_crossing_s": 3.0,
            "random_2d_pre_switch_s": 3.0,
            "random_2d_post_switch_s": 3.0,
        },
        "decision_boundary": (
            "diagnostic evidence only; no gradient update, no c600 replacement, "
            "no SA6 authorization"
        ),
    }


def frozen_protocol() -> dict[str, object]:
    document = payload()
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **document,
        "sha256": hashlib.sha256(canonical).hexdigest(),
    }

