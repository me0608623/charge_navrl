"""Held-out probes for c600 K8 interaction observability.

These probes test information availability. They do not train or modify the
navigation policy, and a successful probe does not prove that the policy uses
the decoded label.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from rnn_car_wdclean.analyze_sa5_c600_step_trace import (
    analyze_lateral_trace,
    analyze_random2d_trace,
)


FEATURE_KEYS = (
    "policy_current_input",
    "policy_k8_input",
    "policy_representation",
)
OBSERVABLE_BALANCED_ACCURACY = 0.70
OBSERVABLE_CLASS_RECALL = 0.65
SPLIT_MODULUS = 5
MAX_SAMPLES_PER_CLASS = 3000


@dataclass(frozen=True)
class ProbeDataset:
    features: dict[str, np.ndarray]
    targets: np.ndarray
    env_ids: np.ndarray
    metadata: dict[str, object]


def _continuous(data: dict[str, np.ndarray], t: int, env: int) -> bool:
    return (
        t > 0
        and int(data["episode_step"][t, env])
        == int(data["episode_step"][t - 1, env]) + 1
    )


def _binary_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    probabilities = logits.sigmoid()
    predictions = logits >= 0
    truth = targets.bool()
    tp = int((predictions & truth).sum())
    tn = int((~predictions & ~truth).sum())
    fp = int((predictions & ~truth).sum())
    fn = int((~predictions & truth).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    scores = probabilities.detach().cpu().numpy().astype(np.float64)
    labels = truth.detach().cpu().numpy().astype(bool)
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    positives = int(sorted_labels.sum())
    negatives = int((~sorted_labels).sum())
    if positives and negatives:
        tpr = np.concatenate([[0.0], np.cumsum(sorted_labels) / positives, [1.0]])
        fpr = np.concatenate([[0.0], np.cumsum(~sorted_labels) / negatives, [1.0]])
        roc_auc = float(np.trapz(tpr, fpr))
        precision_curve = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
        average_precision = float(precision_curve[sorted_labels].mean())
    else:
        roc_auc = float("nan")
        average_precision = float("nan")
    return {
        "samples": int(targets.numel()),
        "positive_fraction": float(truth.float().mean()),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "positive_recall": recall,
        "negative_recall": specificity,
        "precision": tp / max(tp + fp, 1),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _split_masks(env_ids: np.ndarray) -> dict[str, np.ndarray]:
    remainder = np.remainder(env_ids, SPLIT_MODULUS)
    return {
        "train": (remainder != 0) & (remainder != 1),
        "validation": remainder == 1,
        "test": remainder == 0,
    }


def _validate_dataset(dataset: ProbeDataset) -> None:
    rows = int(dataset.targets.shape[0])
    if rows < 200:
        raise ValueError(f"observability dataset is too small: {rows}")
    if dataset.targets.shape != (rows,) or dataset.env_ids.shape != (rows,):
        raise ValueError("target/env-id shapes do not reconcile")
    for name in FEATURE_KEYS:
        values = dataset.features.get(name)
        if values is None or values.ndim != 2 or values.shape[0] != rows:
            raise ValueError(f"invalid or missing feature set {name}")
        if not np.isfinite(values).all():
            raise ValueError(f"feature set {name} contains non-finite values")
    masks = _split_masks(dataset.env_ids)
    for split, mask in masks.items():
        labels = dataset.targets[mask]
        if labels.size < 20 or np.unique(labels).size != 2:
            raise ValueError(f"{split} split lacks both classes")


def _balanced_cap(dataset: ProbeDataset, *, seed: int = 42) -> ProbeDataset:
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for label in (0, 1):
        indices = np.flatnonzero(dataset.targets == label)
        if indices.size > MAX_SAMPLES_PER_CLASS:
            indices = np.sort(
                rng.choice(indices, size=MAX_SAMPLES_PER_CLASS, replace=False)
            )
        selected.append(indices)
    rows = np.sort(np.concatenate(selected))
    return ProbeDataset(
        features={name: values[rows] for name, values in dataset.features.items()},
        targets=dataset.targets[rows],
        env_ids=dataset.env_ids[rows],
        metadata={**dataset.metadata, "balanced_cap_rows": int(rows.size)},
    )


def _fit_one(
    inputs: np.ndarray,
    targets: np.ndarray,
    env_ids: np.ndarray,
    *,
    hidden_dim: int,
    seed: int,
    max_epochs: int = 80,
    patience: int = 12,
) -> dict[str, object]:
    torch.manual_seed(seed)
    masks_np = _split_masks(env_ids)
    masks = {name: torch.from_numpy(mask) for name, mask in masks_np.items()}
    x = torch.from_numpy(inputs.astype(np.float32, copy=False))
    y = torch.from_numpy(targets.astype(np.float32, copy=False))
    train_x = x[masks["train"]]
    mean = train_x.mean(dim=0)
    scale = train_x.std(dim=0, unbiased=False).clamp(min=1.0e-3)
    x = ((x - mean) / scale).clamp(-8.0, 8.0)
    if hidden_dim <= 0:
        model: nn.Module = nn.Linear(x.shape[1], 1)
    else:
        model = nn.Sequential(
            nn.Linear(x.shape[1], hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    train_targets = y[masks["train"]]
    positives = train_targets.sum().clamp(min=1.0)
    negatives = train_targets.numel() - positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=(negatives / positives).reshape(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x[masks["train"]]).squeeze(-1), train_targets)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = _binary_metrics(
                model(x[masks["validation"]]).squeeze(-1),
                y[masks["validation"]],
            )
        score = float(validation["balanced_accuracy"])
        if score > best_validation + 1.0e-4:
            best_validation = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_metrics = _binary_metrics(
            model(x[masks["validation"]]).squeeze(-1), y[masks["validation"]]
        )
        test_metrics = _binary_metrics(
            model(x[masks["test"]]).squeeze(-1), y[masks["test"]]
        )
    return {
        "model": "linear" if hidden_dim <= 0 else f"mlp_{hidden_dim}",
        "input_dim": int(x.shape[1]),
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
    }


def _fit_dataset(dataset: ProbeDataset) -> dict[str, object]:
    dataset = _balanced_cap(dataset)
    _validate_dataset(dataset)
    masks = _split_masks(dataset.env_ids)
    report: dict[str, object] = {
        "samples": int(dataset.targets.size),
        "positive_fraction": float(dataset.targets.mean()),
        "split": {
            name: {
                "samples": int(mask.sum()),
                "positive_fraction": float(dataset.targets[mask].mean()),
                "env_ids": sorted(set(dataset.env_ids[mask].tolist())),
            }
            for name, mask in masks.items()
        },
        "metadata": dataset.metadata,
        "feature_sets": {},
    }
    feature_reports: dict[str, object] = {}
    for feature_index, name in enumerate(FEATURE_KEYS):
        candidates = [
            _fit_one(
                dataset.features[name],
                dataset.targets,
                dataset.env_ids,
                hidden_dim=hidden,
                seed=42 + feature_index * 10 + model_index,
            )
            for model_index, hidden in enumerate((0, 32))
        ]
        selected = max(
            candidates,
            key=lambda row: float(row["validation"]["balanced_accuracy"]),
        )
        feature_reports[name] = {
            "candidates": candidates,
            "selected_by_validation": selected,
        }
    report["feature_sets"] = feature_reports
    return report


def _feature_rows(
    data: dict[str, np.ndarray], rows: list[tuple[int, int]]
) -> dict[str, np.ndarray]:
    time = np.asarray([row[0] for row in rows], dtype=np.int64)
    env = np.asarray([row[1] for row in rows], dtype=np.int64)
    return {
        name: data[name][time, env].astype(np.float32)
        for name in FEATURE_KEYS
    }


def build_lateral_dataset(data: dict[str, np.ndarray]) -> ProbeDataset:
    analysis = analyze_lateral_trace(data)
    selected: dict[str, dict] = {}
    for event in analysis["events"]:
        if float(event["robot_obstacle_distance_m"]) > 3.0:
            continue
        t = int(event["rollout_step"])
        env = int(event["env_id"])
        if int(data["episode_step"][t, env]) < 7:
            continue
        selected.setdefault(str(event["episode_key"]), event)
    events = list(selected.values())
    rows = [(int(event["rollout_step"]), int(event["env_id"])) for event in events]
    return ProbeDataset(
        features=_feature_rows(data, rows),
        targets=np.asarray(
            [int(event["vacated_side"]) > 0 for event in events], dtype=np.int8
        ),
        env_ids=np.asarray([event["env_id"] for event in events], dtype=np.int16),
        metadata={
            "task": "just_vacated_side_at_first_near_crossing_per_episode",
            "positive_label": "+local_x side was just vacated",
            "distance_limit_m": 3.0,
            "events_before_filter": int(analysis["crossing_events"]),
            "independent_episode_events": len(events),
        },
    )


def _actual_reversal_onsets(data: dict[str, np.ndarray]) -> list[dict[str, int | float]]:
    analysis = analyze_random2d_trace(data)
    velocity = data["dynamic_velocities_mps"]
    onsets: list[dict[str, int | float]] = []
    for event in analysis["events"]:
        if not bool(event["direction_reversal"]):
            continue
        switch = int(event["rollout_step"])
        env = int(event["env_id"])
        slot = int(event["obstacle_slot"])
        old_velocity = None
        for t in range(switch - 1, max(-1, switch - 8), -1):
            if t < 0 or (t < switch - 1 and not _continuous(data, t + 1, env)):
                break
            candidate = velocity[t, env, slot]
            if np.linalg.norm(candidate) >= 0.05:
                old_velocity = candidate
                break
        if old_velocity is None:
            continue
        onset = None
        for t in range(switch, min(velocity.shape[0], switch + 8)):
            if t > switch and not _continuous(data, t, env):
                break
            candidate = velocity[t, env, slot]
            if (
                np.linalg.norm(candidate) >= 0.05
                and float(np.dot(old_velocity, candidate)) < 0.0
            ):
                onset = t
                break
        if onset is None:
            continue
        sample = onset - 5
        if sample < 0 or int(data["episode_step"][sample, env]) < 7:
            continue
        if int(data["episode_step"][onset, env]) != int(data["episode_step"][sample, env]) + 5:
            continue
        distance = float(
            np.linalg.norm(
                data["dynamic_positions_m"][sample, env, slot]
                - data["robot_xy_m"][sample, env]
            )
        )
        onsets.append(
            {
                "onset": onset,
                "sample": sample,
                "env": env,
                "slot": slot,
                "distance_m": distance,
            }
        )
    return onsets


def build_reversal_dataset(data: dict[str, np.ndarray]) -> ProbeDataset:
    onsets = _actual_reversal_onsets(data)
    positive_by_key: dict[tuple[int, int], dict[str, int | float]] = {}
    all_onsets_by_env: dict[int, list[int]] = {}
    for row in onsets:
        env = int(row["env"])
        all_onsets_by_env.setdefault(env, []).append(int(row["onset"]))
        if float(row["distance_m"]) <= 3.0:
            positive_by_key.setdefault((int(row["sample"]), env), row)
    positives = list(positive_by_key.values())
    candidates: dict[int, list[tuple[int, float]]] = {}
    positions = data["dynamic_positions_m"]
    robot = data["robot_xy_m"]
    active = data["dynamic_valid"] & data["patrol_active"]
    steps, envs = data["episode_step"].shape
    for env in range(envs):
        onset_times = np.asarray(all_onsets_by_env.get(env, []), dtype=np.int64)
        for t in range(7, steps - 5, 2):
            if int(data["episode_step"][t + 5, env]) != int(data["episode_step"][t, env]) + 5:
                continue
            valid = active[t, env]
            if not bool(valid.any()):
                continue
            distance = np.linalg.norm(positions[t, env] - robot[t, env], axis=1)
            near = valid & (distance <= 3.0)
            if not bool(near.any()):
                continue
            if onset_times.size and bool((np.abs(onset_times - t) <= 5).any()):
                continue
            candidates.setdefault(env, []).append((t, float(distance[near].min())))
    rng = np.random.default_rng(42)
    negatives: list[dict[str, int | float]] = []
    positives_by_env: dict[int, list[dict[str, int | float]]] = {}
    for row in positives:
        positives_by_env.setdefault(int(row["env"]), []).append(row)
    for env, positive_rows in positives_by_env.items():
        available = candidates.get(env, [])
        count = min(len(positive_rows), len(available))
        if count <= 0:
            continue
        chosen = rng.choice(len(available), size=count, replace=False)
        negatives.extend(
            {
                "sample": int(available[index][0]),
                "env": env,
                "distance_m": float(available[index][1]),
            }
            for index in chosen
        )
    rows = [
        (int(row["sample"]), int(row["env"]))
        for row in positives + negatives
    ]
    targets = np.concatenate(
        [
            np.ones(len(positives), dtype=np.int8),
            np.zeros(len(negatives), dtype=np.int8),
        ]
    )
    return ProbeDataset(
        features=_feature_rows(data, rows),
        targets=targets,
        env_ids=np.asarray([row[1] for row in rows], dtype=np.int16),
        metadata={
            "task": "near_pedestrian_actual_direction_reversal_exactly_1s_ahead",
            "positive_definition": (
                "actual patrol velocity resumes opposite to its pre-waypoint "
                "direction exactly five simulator steps later"
            ),
            "negative_definition": (
                "near active pedestrian with no actual reversal onset within "
                "plus or minus one second"
            ),
            "distance_limit_m": 3.0,
            "positive_samples": len(positives),
            "negative_samples": len(negatives),
            "positive_distance_p50_m": (
                None
                if not positives
                else float(np.median([row["distance_m"] for row in positives]))
            ),
            "negative_distance_p50_m": (
                None
                if not negatives
                else float(np.median([row["distance_m"] for row in negatives]))
            ),
        },
    )


def _selected_test(report: dict[str, object], feature: str) -> dict[str, object]:
    return report["feature_sets"][feature]["selected_by_validation"]["test"]


def _observable(metrics: dict[str, object]) -> bool:
    return bool(
        float(metrics["balanced_accuracy"]) >= OBSERVABLE_BALANCED_ACCURACY
        and float(metrics["positive_recall"]) >= OBSERVABLE_CLASS_RECALL
        and float(metrics["negative_recall"]) >= OBSERVABLE_CLASS_RECALL
    )


def decision(lateral: dict[str, object], reversal: dict[str, object]) -> dict[str, object]:
    lateral_current = _selected_test(lateral, "policy_current_input")
    lateral_k8 = _selected_test(lateral, "policy_k8_input")
    lateral_repr = _selected_test(lateral, "policy_representation")
    reversal_current = _selected_test(reversal, "policy_current_input")
    reversal_k8 = _selected_test(reversal, "policy_k8_input")
    reversal_repr = _selected_test(reversal, "policy_representation")
    return {
        "thresholds_frozen_before_gpu": {
            "balanced_accuracy_min": OBSERVABLE_BALANCED_ACCURACY,
            "positive_recall_min": OBSERVABLE_CLASS_RECALL,
            "negative_recall_min": OBSERVABLE_CLASS_RECALL,
        },
        "vacated_side": {
            "raw_k8_observable": _observable(lateral_k8),
            "policy_representation_observable": _observable(lateral_repr),
            "k8_minus_current_balanced_accuracy": float(
                lateral_k8["balanced_accuracy"]
            ) - float(lateral_current["balanced_accuracy"]),
        },
        "reversal_1s_ahead": {
            "raw_k8_observable": _observable(reversal_k8),
            "policy_representation_observable": _observable(reversal_repr),
            "k8_minus_current_balanced_accuracy": float(
                reversal_k8["balanced_accuracy"]
            ) - float(reversal_current["balanced_accuracy"]),
            "blocking_for_sa6": False,
        },
        "scope": (
            "information-availability diagnostic only; neither outcome alone "
            "authorizes or blocks SA6, and reversal failure is explicitly non-blocking"
        ),
        "training_authorized": False,
        "sa6_authorized_by_this_probe": False,
        "sa6_blocked_by_this_probe": False,
    }


def analyze_suite(
    lateral_data: dict[str, np.ndarray], random2d_data: dict[str, np.ndarray]
) -> dict[str, object]:
    lateral_dataset = build_lateral_dataset(lateral_data)
    reversal_dataset = build_reversal_dataset(random2d_data)
    lateral_report = _fit_dataset(lateral_dataset)
    reversal_report = _fit_dataset(reversal_dataset)
    return {
        "schema": "sa5_c600_k8_interaction_observability/v1",
        "lateral_vacated_side": lateral_report,
        "random2d_reversal_1s_ahead": reversal_report,
        "decision": decision(lateral_report, reversal_report),
    }


__all__ = [
    "FEATURE_KEYS",
    "OBSERVABLE_BALANCED_ACCURACY",
    "OBSERVABLE_CLASS_RECALL",
    "ProbeDataset",
    "_binary_metrics",
    "_split_masks",
    "analyze_suite",
    "build_lateral_dataset",
    "build_reversal_dataset",
    "decision",
]
