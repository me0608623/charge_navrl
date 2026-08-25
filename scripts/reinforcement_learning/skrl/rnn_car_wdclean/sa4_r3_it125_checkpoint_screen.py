"""Frozen three-scenario screen for the conceptual SA4-R3 it125 checkpoint."""

from __future__ import annotations

import hashlib
import json


PROTOCOL_SCHEMA = "sa4_r3_it125_checkpoint_screen/v1"
RUN_NAME = "sa4_r3_cont25_from_it100_ne1024_s42_p25_r1"
CHECKPOINT_NAME = "sa4_r3_it125"
CHECKPOINT_FILENAME = "checkpoint_3200.pt"
CHECKPOINT_SHA256 = (
    "57f43d07255971ad607e0bf3234dd7d46984c9c71cc9550e2e9461fbc95ab71d"
)
CONCEPTUAL_ITERATION = 125
SCENARIOS = ("corridor_lateral", "corridor_longitudinal", "nav_native")
STEPS_BY_SCENARIO = {
    "corridor_lateral": 2500,
    "corridor_longitudinal": 4000,
    "nav_native": 1200,
}
SEED = 818
DELAY_STEPS = 1
ELIGIBILITY = "valid_return_only"

# These hashes were recorded when it125 finished training. The queue refuses to
# evaluate if policy-input or actuator semantics drift before the screen.
TRAINING_SOURCE_HASHES = {
    "config": "5e06a7a256e2d270ad4180927b0b8cf0761d2b7c4dcc58ebe4dfbbceffacc947",
    "trainer": "6ac0e362c0effd9493d3a36237c752e780ec95f356196c28944fe88147b07165",
    "env_overrides": "82ed5f133f99614de70a212ef186b24122fff6d0cd28d5847e08d3423f74a39c",
    "lidar_observation": "27f976bdb0a51100f9c749aaf4010a995fadfa3ce2319d4b10b5d2586ae87e7f",
    "actuator_action": "551baa1dc38d8be37751e530f964a1223876ed3b9ad0f50b7b4788e0bdc11b63",
}
EVALUATION_SOURCE_HASHES = {
    "stage4_runner": "4dab0e5d7f823badf3f2744d1c98ba9c7eb0b9f9a3f0ef3f5959688ba83b69ac",
    "stage_runner_base": "443ac5a1fdd662646d4ee41cc4f2bbfb37e604e168f2bb23b34bb31794d65920",
    "play_launcher": "cb8ec519417fcbbeda9f65e34374383cd8f50dac75e3530936a4759d60bb5984",
    "gate_parser": "9c1e3a354e3bf08596d10a7f813d76f1642bf6351773063c9682ab88337d85e5",
    "fixed_actuator": "0374e5a4b2b357e5ff8fb37e46c82cae58789c2ea3183e9b9413059f6f970b07",
    "play": "5aec929b60dfe6b094a0634fe2e53589ebaa916e3b033985ab8e3aeb6d281417",
    "env_overrides": "82ed5f133f99614de70a212ef186b24122fff6d0cd28d5847e08d3423f74a39c",
    "lidar_observation": "27f976bdb0a51100f9c749aaf4010a995fadfa3ce2319d4b10b5d2586ae87e7f",
    "actuator_action": "551baa1dc38d8be37751e530f964a1223876ed3b9ad0f50b7b4788e0bdc11b63",
    "acceptance_contract": "9ee71fe3213dcad332444fc04a98a40907cf86c4a370fada0bf32205c0631c5e",
}


def screen_protocol() -> dict:
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "purpose": (
            "apply the frozen SA4 three-scenario absolute screen to the exact "
            "optimizer continuation checkpoint at conceptual iteration 125"
        ),
        "checkpoint": {
            "run_name": RUN_NAME,
            "name": CHECKPOINT_NAME,
            "filename": CHECKPOINT_FILENAME,
            "conceptual_iteration": CONCEPTUAL_ITERATION,
            "sha256": CHECKPOINT_SHA256,
        },
        "fixed_evaluation": {
            "geometry_stage": 4,
            "evaluator_seed": SEED,
            "actuator_delay_steps": DELAY_STEPS,
            "actuator_delay_ms": 200,
            "actuator_profile": "sa1_delay_only",
            "lidar_distractor_eligibility": ELIGIBILITY,
            "num_envs": 64,
            "steps_by_scenario": STEPS_BY_SCENARIO,
            "minimum_completed_episodes_per_scenario": 1000,
        },
        "threshold_source": "sa3_sa8_acceptance_contract.py",
        "absolute_thresholds": {
            "corridor_lateral": {"sr_min": 0.90, "cr_max": 0.10, "to_max": 0.05},
            "corridor_longitudinal": {
                "sr_min": 0.90,
                "cr_max": 0.10,
                "to_max": 0.05,
            },
            "nav_native": {"sr_min": 0.92, "cr_max": 0.07, "to_max": 0.035},
        },
        "candidate_rule": (
            "lateral, longitudinal, and nav_native must each independently "
            "pass their absolute SR/CR/TO thresholds"
        ),
        "validity": [
            "all three cells produce cell JSON",
            "checkpoint hash equals the preregistered it125 hash",
            "training-time policy-input and actuator source hashes still match",
            "evaluation source fingerprints remain stable across all cells",
            "runtime eligibility is valid_return_only for policy and critic",
            "fixed d1 actuator and stage-4 geometry markers match",
            "console and exact episode ledgers reconcile",
            "no NaN, OOM, traceback, action override, training, or SA5 launch",
        ],
        "evidence_boundary": (
            "this is a single training seed and fixed evaluator seed screen; "
            "play_rnn_car.py has additive default-off diagnostics since the it100 "
            "screen, so historical it100 deltas are descriptive, while the it125 "
            "verdict uses the frozen absolute thresholds"
        ),
        "authorization_limit": (
            "a pass permits human consideration of formal SA4 verification; the "
            "screen itself accepts no parent and starts neither training nor SA5"
        ),
        "training_source_hashes": TRAINING_SOURCE_HASHES,
        "evaluation_source_hashes": EVALUATION_SOURCE_HASHES,
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def evaluate_candidate(payloads: list[dict]) -> dict:
    by_scenario = {payload["scenario"]: payload for payload in payloads}
    if set(by_scenario) != set(SCENARIOS):
        raise ValueError(
            f"expected exactly {sorted(SCENARIOS)}, got {sorted(by_scenario)}"
        )
    for payload in payloads:
        if payload["checkpoint_name"] != CHECKPOINT_NAME:
            raise ValueError(
                f"unexpected checkpoint name {payload['checkpoint_name']!r}"
            )

    cells = {
        scenario: {
            "n": int(by_scenario[scenario]["metrics"]["n"]),
            "sr": float(by_scenario[scenario]["metrics"]["sr"]),
            "cr": float(by_scenario[scenario]["metrics"]["cr"]),
            "to": float(by_scenario[scenario]["metrics"]["to"]),
            "hard_pass": bool(by_scenario[scenario]["threshold_pass"]),
        }
        for scenario in SCENARIOS
    }
    all_three_pass = all(cell["hard_pass"] for cell in cells.values())
    lateral_cr = cells["corridor_lateral"]["cr"]
    longitudinal_cr = cells["corridor_longitudinal"]["cr"]
    return {
        "schema": "sa4_r3_it125_checkpoint_screen_verdict/v1",
        "checkpoint_name": CHECKPOINT_NAME,
        "conceptual_iteration": CONCEPTUAL_ITERATION,
        "cells": cells,
        "worst_corridor_cr": max(lateral_cr, longitudinal_cr),
        "mean_corridor_cr": 0.5 * (lateral_cr + longitudinal_cr),
        "all_three_hard_pass": all_three_pass,
        "screen_pass": all_three_pass,
        "selected_checkpoint": CHECKPOINT_NAME if all_three_pass else None,
        "formal_sa4_verification_consideration_eligible": all_three_pass,
        "accepted_parent": False,
        "additional_training_started": False,
        "sa5_started": False,
    }
