"""Frozen protocol for c600 K8 interaction-label observability."""

from __future__ import annotations

import hashlib
import json

from rnn_car_wdclean import sa5_c600_step_trace_protocol as base
from rnn_car_wdclean.analyze_sa5_c600_k8_observability import (
    OBSERVABLE_BALANCED_ACCURACY,
    OBSERVABLE_CLASS_RECALL,
    SPLIT_MODULUS,
)


def payload() -> dict[str, object]:
    return {
        "schema": "sa5_c600_k8_observability_protocol/v1",
        "checkpoint": str(base.CHECKPOINT),
        "checkpoint_sha256": base.CHECKPOINT_SHA256,
        "fixed_cells": base.payload()["fixed_cell"],
        "scenarios": list(base.SCENARIOS),
        "policy_contract": {
            **base.payload()["policy_contract"],
            "saved_inputs": {
                "current": "normalized current 83D observation",
                "k8": "exact 587D K8 extractor input",
                "representation": "exact 179D policy-head input",
            },
            "storage_dtype": "float16",
        },
        "labels": {
            "vacated_side": (
                "first <=3m pedestrian centerline crossing per episode; "
                "target is the side occupied immediately before crossing"
            ),
            "reversal_1s_ahead": (
                "actual patrol velocity resumes in the opposite direction "
                "exactly five 0.2s steps after the sample"
            ),
            "labels_are_privileged_offline_only": True,
        },
        "split": (
            f"whole environment IDs; test env_id % {SPLIT_MODULUS}=0, "
            f"validation=1, remaining train"
        ),
        "models": ["linear", "mlp_32"],
        "selection": "model selected on validation balanced accuracy; report once on test",
        "observable_thresholds": {
            "test_balanced_accuracy_min": OBSERVABLE_BALANCED_ACCURACY,
            "test_positive_recall_min": OBSERVABLE_CLASS_RECALL,
            "test_negative_recall_min": OBSERVABLE_CLASS_RECALL,
        },
        "decision_boundary": {
            "purpose": "information availability only, not policy use or causality",
            "reversal_failure_blocks_sa6": False,
            "probe_alone_authorizes_sa6": False,
            "training": False,
            "teacher": False,
            "distillation": False,
        },
    }


def frozen_protocol() -> dict[str, object]:
    document = payload()
    canonical = json.dumps(
        document, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {**document, "sha256": hashlib.sha256(canonical).hexdigest()}


__all__ = ["frozen_protocol", "payload"]
