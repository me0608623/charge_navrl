"""Frozen cross-seed decision rule for the SA4-D9 closed-loop A/B."""

from __future__ import annotations

import hashlib
import json

import d9_noise_closed_loop_ab as d9


PROTOCOL_SCHEMA = "sa4_d9_seed_replication/v1"
NEW_SEEDS = (515, 616)
ALL_SEEDS = (515, 616, 818)
MAJORITY_COUNT = 2
WALL_CR_NONINFERIORITY = 0.005
ORIGINAL_SEED818_MANIFEST_SHA256 = (
    "f5c6370715a9a306929f3cd942c97e06fe5188aa6d1299b325902de0fac42127"
)


def replication_protocol() -> dict:
    protocol = {
        "schema": PROTOCOL_SCHEMA,
        "purpose": (
            "replicate the SA4-D9 current_full versus valid_return_only "
            "closed-loop comparison on two evaluator seeds fixed before rollout"
        ),
        "parent_d9_protocol_sha256": d9.closed_loop_protocol()["sha256"],
        "original_seed818_manifest_sha256": ORIGINAL_SEED818_MANIFEST_SHA256,
        "new_evaluator_seeds": list(NEW_SEEDS),
        "all_evaluator_seeds": list(ALL_SEEDS),
        "fixed_cell_except_seed": {
            "checkpoint_sha256": d9.CHECKPOINT_SHA256,
            "stage": 4,
            "scenario": "corridor_lateral",
            "actuator_delay_steps": 1,
            "actuator_delay_ms": 200,
            "actuator_profile": "sa1_delay_only",
            "num_envs": 64,
            "rollout_steps": 2500,
            "vlp16_noise_mode": "full",
            "deterministic": True,
            "action_override": "none",
        },
        "arms": {
            arm: {"distractor_eligibility": eligibility}
            for arm, eligibility in d9.ELIGIBILITY_BY_ARM.items()
        },
        "seed_level_rule": {
            "cr_delta_b_minus_a": "< 0",
            "wall_cr_delta_b_minus_a": f"<= {WALL_CR_NONINFERIORITY}",
            "timeout_delta_b_minus_a": f"<= {d9.TO_NONINFERIORITY}",
        },
        "pilot_authorization_rule": {
            "qualifying_seeds_required": MAJORITY_COUNT,
            "seeds_total": len(ALL_SEEDS),
            "plain_language": (
                "at least two of the three fixed seeds show lower total CR, "
                "wall CR increase no greater than 0.5 percentage points, and "
                "timeout increase no greater than 0.5 percentage points"
            ),
        },
        "validity": [
            "all four new arm cells complete with at least 1000 episodes",
            "the frozen seed818 manifest hash and checkpoint hash match",
            "runtime eligibility banners match the requested arms",
            "source fingerprints remain stable before, between, and after cells",
            "corridor and console episode ledgers reconcile",
            "no NaN, OOM, traceback, action override, training, or SA5 launch",
        ],
        "interpretation_limits": [
            "three evaluator seeds are replication evidence, not training-seed replication",
            "closed-loop arms are same-seed but not per-episode paired counterfactuals",
            "valid_return_only remains a sensitivity model, not a validated VLP-16 model",
            "passing authorizes only a short SA4-R3 pilot from the clean SA3 c100 parent",
        ],
    }
    canonical = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return protocol


def seed_qualifies(comparison: dict) -> bool:
    delta = comparison["delta_b_minus_a"]
    return (
        float(delta["cr"]) < 0.0
        and float(delta["wall_cr"]) <= WALL_CR_NONINFERIORITY
        and float(delta["to"]) <= d9.TO_NONINFERIORITY
    )


def compare_replications(comparisons_by_seed: dict[int, dict]) -> dict:
    normalized = {int(seed): value for seed, value in comparisons_by_seed.items()}
    if set(normalized) != set(ALL_SEEDS):
        raise ValueError(
            f"expected fixed seeds {ALL_SEEDS}, got {sorted(normalized)}"
        )
    per_seed = {
        str(seed): {
            "comparison": normalized[seed],
            "qualifies_for_majority": seed_qualifies(normalized[seed]),
        }
        for seed in ALL_SEEDS
    }
    qualifying = sum(
        int(item["qualifies_for_majority"]) for item in per_seed.values()
    )
    pilot_authorized = qualifying >= MAJORITY_COUNT
    return {
        "schema": "sa4_d9_seed_replication_comparison/v1",
        "per_seed": per_seed,
        "qualifying_seed_count": qualifying,
        "seed_count": len(ALL_SEEDS),
        "majority_required": MAJORITY_COUNT,
        "wall_cr_noninferiority_tolerance": WALL_CR_NONINFERIORITY,
        "sa4_r3_short_pilot_authorized": pilot_authorized,
        "sa5_authorized": False,
        "inferential_claim": False,
    }

