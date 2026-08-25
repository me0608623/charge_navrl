"""Frozen protocol and comparison logic for SA4-D9 closed-loop noise A/B."""

from __future__ import annotations

import hashlib
import json
import math


PROTOCOL_SCHEMA = "sa4_d9_noise_closed_loop_ab/v1"
CURRENT_ARM = "current_full"
CORRECTED_ARM = "valid_return_only"
ARMS = (CURRENT_ARM, CORRECTED_ARM)
ELIGIBILITY_BY_ARM = {
    CURRENT_ARM: "all_rays",
    CORRECTED_ARM: "valid_return_only",
}
CHECKPOINT_SHA256 = (
    "c6dbd94bcbd6cc17ceca3cf8d5abfb4d956c98ed173c0dd890b9fc45959a2197"
)
MATERIAL_DELTA = 0.005
TO_NONINFERIORITY = 0.005
PASS_SR = 0.90
PASS_CR = 0.10
PASS_TO = 0.05


def closed_loop_protocol() -> dict:
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "purpose": (
            "test whether the D8 valid-return-only mixed-pixel eligibility "
            "sensitivity improves realized closed-loop episode outcomes"
        ),
        "arms": {
            CURRENT_ARM: {
                "distractor_eligibility": "all_rays",
                "policy_observation": "current historical full-noise sweep",
            },
            CORRECTED_ARM: {
                "distractor_eligibility": "valid_return_only",
                "policy_observation": "corrected sweep is fed to the policy",
            },
        },
        "sole_behavioral_difference": "mixed-pixel distractor eligibility",
        "fixed_cell": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "stage": 4,
            "scenario": "corridor_lateral",
            "evaluator_seed": 818,
            "actuator_delay_steps": 1,
            "actuator_delay_ms": 200,
            "actuator_profile": "sa1_delay_only",
            "num_envs": 64,
            "rollout_steps": 2500,
            "vlp16_noise_mode": "full",
            "deterministic": True,
            "action_override": "none",
        },
        "random_draw_contract": (
            "both modes draw the same Bernoulli mask and U(0.2,2.0m) values; "
            "valid_return_only gates the realized mask after those draws"
        ),
        "primary_outcomes": [
            "success_rate",
            "collision_rate",
            "obstacle_collision_rate",
            "wall_collision_rate",
            "timeout_rate",
        ],
        "descriptive_rules": {
            "directional_improvement": (
                "B SR > A SR, B CR < A CR, and B TO <= A TO + 0.005"
            ),
            "material_improvement": (
                "directional improvement and both SR gain and CR reduction >= 0.005"
            ),
            "graduation_thresholds_reported_only": {
                "sr_min": PASS_SR,
                "cr_max": PASS_CR,
                "to_max": PASS_TO,
            },
            "inferential_claim": False,
        },
        "validity": [
            "both fresh-process arms complete",
            "runtime eligibility banner matches the requested arm",
            "checkpoint and source fingerprints remain stable",
            "corridor and console episode ledgers reconcile",
            "no NaN, OOM, traceback, action override, training, or SA5 launch",
        ],
        "interpretation_limits": [
            "single checkpoint, corridor family, and evaluator seed",
            "closed-loop divergence prevents per-episode paired-state interpretation",
            "valid-return-only remains a sensitivity model, not a validated VLP-16 model",
            "this diagnostic cannot authorize an SA4 parent or SA5",
        ],
        "d8_parent_protocol_sha256": (
            "1fcce8d769fef404d01304b184eeef96a073cce7b4275849184a1a850e701185"
        ),
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def _validate_metrics(metrics: dict) -> None:
    if int(metrics.get("n", 0)) < 1:
        raise ValueError("arm has no completed episodes")
    for key in ("sr", "cr", "to", "obstacle_cr", "wall_cr"):
        value = float(metrics.get(key, float("nan")))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid {key}={value!r}")
    if abs(
        float(metrics["sr"]) + float(metrics["cr"]) + float(metrics["to"]) - 1.0
    ) > 1.0e-6:
        raise ValueError("SR/CR/TO do not reconcile to one")


def compare_closed_loop(arms: dict[str, dict]) -> dict:
    if set(arms) != set(ARMS):
        raise ValueError(f"expected exactly {ARMS}, got {sorted(arms)}")
    for metrics in arms.values():
        _validate_metrics(metrics)
    current = arms[CURRENT_ARM]
    corrected = arms[CORRECTED_ARM]
    deltas = {
        key: float(corrected[key]) - float(current[key])
        for key in ("sr", "cr", "to", "obstacle_cr", "wall_cr")
    }
    directional = (
        deltas["sr"] > 0.0
        and deltas["cr"] < 0.0
        and deltas["to"] <= TO_NONINFERIORITY
    )
    material = (
        directional
        and deltas["sr"] >= MATERIAL_DELTA
        and -deltas["cr"] >= MATERIAL_DELTA
    )
    corrected_threshold_pass = (
        float(corrected["sr"]) >= PASS_SR
        and float(corrected["cr"]) <= PASS_CR
        and float(corrected["to"]) <= PASS_TO
    )
    return {
        "schema": "sa4_d9_noise_closed_loop_comparison/v1",
        "arms": arms,
        "delta_b_minus_a": deltas,
        "directional_closed_loop_improvement_observed": directional,
        "material_closed_loop_improvement_observed": material,
        "corrected_arm_sa4_threshold_pass": corrected_threshold_pass,
        "material_delta_threshold": MATERIAL_DELTA,
        "timeout_noninferiority_tolerance": TO_NONINFERIORITY,
        "inferential_claim": False,
        "parent_or_sa5_authorized": False,
    }
