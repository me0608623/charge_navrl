"""Deterministic sample-size repair for the frozen SA5-R2 screen.

The base screen fixed rollout steps before execution, but a few slow policies
completed fewer than the separately frozen minimum of 1,000 episodes. This
addendum changes no scene, checkpoint, metric, threshold, or ranking rule. It
only replaces an underfilled cell with a longer from-scratch rollout selected
from its episode count. Base and repair samples are never pooled.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import sa5_r2_checkpoint_screen as base


TARGET_EPISODES = 1200
STEP_QUANTUM = 500
MAX_REPAIR_STEPS = 7500


def _hash_payload(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def repair_addendum() -> dict:
    payload = {
        "schema": "sa5_r2_checkpoint_screen_sample_repair/v1",
        "base_protocol_sha256": base.screen_protocol()["sha256"],
        "trigger": {
            "metric": "completed episodes n",
            "condition": f"n < {base.MIN_EPISODES}",
            "outcome_metrics_consulted": False,
        },
        "replacement_rule": {
            "target_episodes": TARGET_EPISODES,
            "step_quantum": STEP_QUANTUM,
            "max_repair_steps": MAX_REPAIR_STEPS,
            "formula": (
                "ceil((base_steps * target_episodes / base_n) / "
                "step_quantum) * step_quantum"
            ),
            "minimum": "base_steps + one step_quantum",
            "from_scratch_same_seed": True,
            "replace_short_cell": True,
            "pool_base_and_repair": False,
        },
        "unchanged": [
            "checkpoint identity",
            "scene geometry and obstacle density",
            "motion family",
            "evaluator seed",
            "VLP-16 noise and valid-return-only eligibility",
            "200 ms actuator delay",
            "metrics, thresholds, and ranking",
        ],
        "phase_scope": "any Phase-A or Phase-B cell below the frozen minimum",
        "evidence_boundary": (
            "post-Phase-A sample-size addendum; trigger and duration use n only, "
            "not SR/CR/TO; repaired cells replace rather than pool with base cells"
        ),
    }
    payload = json.loads(json.dumps(payload, sort_keys=True))
    payload["sha256"] = _hash_payload(payload)
    return payload


def repair_steps(base_steps: int, base_n: int) -> int | None:
    if base_steps <= 0:
        raise ValueError("base_steps must be positive")
    if base_n <= 0:
        raise ValueError("base_n must be positive")
    if base_n >= base.MIN_EPISODES:
        return None
    estimated = base_steps * TARGET_EPISODES / base_n
    rounded = int(math.ceil(estimated / STEP_QUANTUM) * STEP_QUANTUM)
    steps = max(base_steps + STEP_QUANTUM, rounded)
    if steps > MAX_REPAIR_STEPS:
        raise ValueError(
            f"repair would require {steps} steps, above {MAX_REPAIR_STEPS}"
        )
    return steps


def execution_protocol(scenario: str, steps: int) -> dict:
    if scenario not in base.ALL_SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    payload = copy.deepcopy(base.screen_protocol())
    payload.pop("sha256")
    payload["fixed_evaluation"]["steps_by_scenario"][scenario] = int(steps)
    payload["sha256"] = _hash_payload(payload)
    return payload


def cell_key(payload: dict) -> tuple[str, str]:
    return str(payload.get("checkpoint_name")), str(payload.get("scenario"))


def _validate_identity(payload: dict, candidate: dict, scenario: str) -> None:
    if payload.get("checkpoint_name") != candidate["name"]:
        raise ValueError("cell checkpoint name mismatch")
    if payload.get("checkpoint_sha256") != candidate["sha256"]:
        raise ValueError("cell checkpoint hash mismatch")
    if payload.get("scenario") != scenario:
        raise ValueError("cell scenario mismatch")
    if not payload.get("cell_valid"):
        raise ValueError("cell is not structurally valid")
    metrics = payload.get("metrics") or {}
    for name in ("n", "sr", "cr", "to"):
        if name not in metrics or not math.isfinite(float(metrics[name])):
            raise ValueError(f"missing or non-finite metric {name}")


def validate_base_cell(payload: dict, candidate: dict, scenario: str) -> None:
    _validate_identity(payload, candidate, scenario)
    if payload.get("protocol_sha256") != base.screen_protocol()["sha256"]:
        raise ValueError("base cell protocol hash mismatch")
    expected_steps = base.STEPS_BY_SCENARIO[scenario]
    if int(payload.get("steps", -1)) != expected_steps:
        raise ValueError("base cell step count mismatch")
    if payload.get("sample_size_repair") is not None:
        raise ValueError("base cell unexpectedly marked as a repair")


def validate_repair_cell(
    payload: dict,
    candidate: dict,
    scenario: str,
    *,
    base_n: int,
) -> None:
    _validate_identity(payload, candidate, scenario)
    expected_steps = repair_steps(base.STEPS_BY_SCENARIO[scenario], base_n)
    if expected_steps is None:
        raise ValueError("repair supplied for a cell that already met minimum n")
    if int(payload.get("steps", -1)) != expected_steps:
        raise ValueError("repair cell step count mismatch")
    expected_execution = execution_protocol(scenario, expected_steps)["sha256"]
    if payload.get("protocol_sha256") != expected_execution:
        raise ValueError("repair execution protocol hash mismatch")
    repair = payload.get("sample_size_repair") or {}
    expected_addendum = repair_addendum()["sha256"]
    if repair.get("addendum_sha256") != expected_addendum:
        raise ValueError("repair addendum hash mismatch")
    if int(repair.get("base_n", -1)) != int(base_n):
        raise ValueError("repair base_n mismatch")
    if bool(repair.get("pooled_with_base", True)):
        raise ValueError("base and repair samples must not be pooled")
    if int((payload.get("metrics") or {}).get("n", 0)) < base.MIN_EPISODES:
        raise ValueError("repair cell still has fewer than minimum episodes")


def choose_effective_cells(
    base_payloads: Iterable[dict],
    repair_payloads: Iterable[dict],
    *,
    checkpoint_names: Iterable[str],
    scenarios: Iterable[str],
) -> tuple[list[dict], list[dict]]:
    names = tuple(checkpoint_names)
    scenario_names = tuple(scenarios)
    expected = {(name, scenario) for name in names for scenario in scenario_names}
    base_by_key = {cell_key(payload): payload for payload in base_payloads}
    repair_by_key = {cell_key(payload): payload for payload in repair_payloads}
    if set(base_by_key) != expected:
        raise ValueError("base cell matrix is incomplete or contains extras")
    if not set(repair_by_key).issubset(expected):
        raise ValueError("repair matrix contains unexpected cells")

    effective = []
    replacement_log = []
    for name in names:
        candidate = base.candidate_by_name(name)
        for scenario in scenario_names:
            key = (name, scenario)
            original = base_by_key[key]
            validate_base_cell(original, candidate, scenario)
            base_n = int(original["metrics"]["n"])
            needed_steps = repair_steps(
                base.STEPS_BY_SCENARIO[scenario], base_n
            )
            replacement = repair_by_key.get(key)
            if needed_steps is None:
                if replacement is not None:
                    raise ValueError("unexpected repair for sufficient base cell")
                effective.append(original)
                continue
            if replacement is None:
                raise ValueError(f"missing required repair for {name} {scenario}")
            validate_repair_cell(
                replacement, candidate, scenario, base_n=base_n
            )
            effective.append(replacement)
            replacement_log.append(
                {
                    "checkpoint_name": name,
                    "scenario": scenario,
                    "base_n": base_n,
                    "base_steps": int(original["steps"]),
                    "repair_n": int(replacement["metrics"]["n"]),
                    "repair_steps": int(replacement["steps"]),
                    "pooled": False,
                }
            )
    return effective, replacement_log


def _normalized_for_base_validator(payloads: Iterable[dict]) -> list[dict]:
    protocol_sha = base.screen_protocol()["sha256"]
    normalized = []
    for payload in payloads:
        item = copy.deepcopy(payload)
        item["protocol_sha256"] = protocol_sha
        normalized.append(item)
    return normalized


def rank_phase_a(
    base_payloads: Iterable[dict], repair_payloads: Iterable[dict]
) -> dict:
    effective, replacements = choose_effective_cells(
        base_payloads,
        repair_payloads,
        checkpoint_names=[item["name"] for item in base.CANDIDATES],
        scenarios=base.CORRIDOR_SCENARIOS,
    )
    result = base.rank_phase_a(_normalized_for_base_validator(effective))
    result["sample_size_replacements"] = replacements
    result["sample_size_addendum_sha256"] = repair_addendum()["sha256"]
    return result


def final_verdict(
    phase_a_base: Iterable[dict],
    phase_a_repairs: Iterable[dict],
    phase_b_base: Iterable[dict],
    phase_b_repairs: Iterable[dict],
) -> dict:
    phase_a_result = rank_phase_a(phase_a_base, phase_a_repairs)
    top_two = phase_a_result["top_two"]
    effective_a, replacements_a = choose_effective_cells(
        phase_a_base,
        phase_a_repairs,
        checkpoint_names=[item["name"] for item in base.CANDIDATES],
        scenarios=base.CORRIDOR_SCENARIOS,
    )
    effective_b, replacements_b = choose_effective_cells(
        phase_b_base,
        phase_b_repairs,
        checkpoint_names=top_two,
        scenarios=base.RETENTION_SCENARIOS,
    )
    verdict = base.final_verdict(
        _normalized_for_base_validator(effective_a),
        _normalized_for_base_validator(effective_b),
    )
    verdict["phase_a"]["sample_size_replacements"] = replacements_a
    verdict["sample_size_replacements"] = {
        "phase_a": replacements_a,
        "phase_b": replacements_b,
    }
    verdict["sample_size_addendum_sha256"] = repair_addendum()["sha256"]
    return verdict

