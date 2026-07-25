"""Measure whether policy inputs identify deployment-corridor scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    predictions = logits >= 0
    targets = targets.bool()
    tp = int((predictions & targets).sum())
    tn = int((~predictions & ~targets).sum())
    fp = int((predictions & ~targets).sum())
    fn = int((~predictions & targets).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _fit(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
) -> tuple[nn.Module, dict[str, float]]:
    input_dim = inputs.shape[1]
    model: nn.Module
    if hidden_dim <= 0:
        model = nn.Linear(input_dim, 1)
    else:
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
    positives = targets[train_mask].sum().clamp(min=1)
    negatives = train_mask.sum() - positives
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=(negatives / positives).reshape(1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_inputs = inputs[train_mask]
    train_targets = targets[train_mask].float()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_inputs).squeeze(-1)
        loss = criterion(logits, train_targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        test_logits = model(inputs[test_mask]).squeeze(-1)
        result = _metrics(test_logits, targets[test_mask])
        result["loss"] = float(
            nn.functional.binary_cross_entropy_with_logits(
                test_logits, targets[test_mask].float()
            )
        )
    return model, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--heldout-modulus", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(42)
    payload = torch.load(
        args.dataset.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    inputs = payload["inputs"].float()
    labels = payload["labels"].long()
    env_ids = payload["env_ids"].long()
    policy_obs_dim = int(payload["metadata"]["policy_obs_dim"])
    if inputs.ndim != 2 or labels.ndim != 1:
        raise ValueError("scene probe dataset has invalid tensor shapes")
    if inputs.shape[0] != labels.numel() or labels.numel() != env_ids.numel():
        raise ValueError("scene probe dataset row counts do not match")
    if args.heldout_modulus < 2:
        raise ValueError("heldout modulus must be at least 2")

    test_mask = env_ids.remainder(args.heldout_modulus) == 0
    train_mask = ~test_mask
    targets = labels == 3
    if not bool(targets[train_mask].any()) or not bool(targets[test_mask].any()):
        raise ValueError("corridor positives are missing from a split")

    feature_sets = {
        "current_83d": inputs[:, :policy_obs_dim],
        "full_k8": inputs,
    }
    results: dict[str, dict[str, float]] = {}
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    for feature_name, feature_inputs in feature_sets.items():
        for model_name, hidden_dim in (("linear", 0), ("mlp", 64)):
            key = f"{feature_name}/{model_name}"
            model, metrics = _fit(
                feature_inputs,
                targets,
                train_mask,
                test_mask,
                hidden_dim=hidden_dim,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
            )
            results[key] = metrics
            torch.save(
                model.state_dict(),
                output.with_name(f"{output.stem}_{feature_name}_{model_name}.pt"),
            )

    best_key = max(
        results,
        key=lambda key: results[key]["balanced_accuracy"],
    )
    report = {
        "dataset": str(args.dataset.expanduser().resolve()),
        "samples": int(labels.numel()),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "corridor_fraction": float(targets.float().mean()),
        "split": f"held-out env_id % {args.heldout_modulus} == 0",
        "results": results,
        "best": best_key,
        "pass": bool(
            results[best_key]["recall"] >= 0.90
            and results[best_key]["specificity"] >= 0.90
        ),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
