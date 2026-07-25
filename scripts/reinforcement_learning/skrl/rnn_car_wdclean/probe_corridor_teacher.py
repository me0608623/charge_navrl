#!/usr/bin/env python3
"""Held-out probe for predicting privileged corridor-teacher actions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ActionProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_bins: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.linear_head = nn.Linear(hidden_dim, num_bins)
        self.angular_head = nn.Linear(hidden_dim, num_bins)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(inputs)
        return self.linear_head(features), self.angular_head(features)


def _class_weights(labels: torch.Tensor, num_bins: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_bins).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()
    return weights.clamp(max=5.0)


def _balanced_accuracy(
    truth: torch.Tensor,
    prediction: torch.Tensor,
    classes: tuple[int, ...],
) -> float:
    recalls = []
    for class_id in classes:
        mask = truth == class_id
        if bool(mask.any()):
            recalls.append(
                float((prediction[mask] == truth[mask]).float().mean())
            )
    return float(sum(recalls) / max(len(recalls), 1))


def _intent(index: torch.Tensor, center: int) -> torch.Tensor:
    return torch.where(
        index < center,
        torch.zeros_like(index),
        torch.where(
            index > center,
            torch.full_like(index, 2),
            torch.ones_like(index),
        ),
    )


def _evaluate(
    model: ActionProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    center: int,
) -> dict[str, float | int | list[float]]:
    model.eval()
    with torch.no_grad():
        linear_logits, angular_logits = model(features)
        linear_pred = linear_logits.argmax(dim=-1)
        angular_pred = angular_logits.argmax(dim=-1)
    linear_true = labels[:, 0]
    angular_true = labels[:, 1]
    linear_intent_true = _intent(linear_true, center)
    angular_intent_true = _intent(angular_true, center)
    linear_intent_pred = _intent(linear_pred, center)
    angular_intent_pred = _intent(angular_pred, center)

    def _majority_accuracy(values: torch.Tensor) -> float:
        counts = torch.bincount(values)
        return float(counts.max() / max(values.numel(), 1))

    return {
        "frames": int(labels.shape[0]),
        "linear_exact_accuracy": float(
            (linear_pred == linear_true).float().mean()
        ),
        "angular_exact_accuracy": float(
            (angular_pred == angular_true).float().mean()
        ),
        "joint_exact_accuracy": float(
            (
                (linear_pred == linear_true)
                & (angular_pred == angular_true)
            ).float().mean()
        ),
        "linear_within_1_bin_accuracy": float(
            ((linear_pred - linear_true).abs() <= 1).float().mean()
        ),
        "angular_within_1_bin_accuracy": float(
            ((angular_pred - angular_true).abs() <= 1).float().mean()
        ),
        "linear_intent_accuracy": float(
            (linear_intent_pred == linear_intent_true).float().mean()
        ),
        "angular_direction_accuracy": float(
            (angular_intent_pred == angular_intent_true).float().mean()
        ),
        "linear_intent_balanced_accuracy": _balanced_accuracy(
            linear_intent_true, linear_intent_pred, (0, 1, 2)
        ),
        "angular_direction_balanced_accuracy": _balanced_accuracy(
            angular_intent_true, angular_intent_pred, (0, 1, 2)
        ),
        "linear_exact_majority_baseline": _majority_accuracy(linear_true),
        "angular_exact_majority_baseline": _majority_accuracy(angular_true),
        "linear_intent_majority_baseline": _majority_accuracy(
            linear_intent_true
        ),
        "angular_direction_majority_baseline": _majority_accuracy(
            angular_intent_true
        ),
        "linear_intent_distribution": (
            torch.bincount(linear_intent_true, minlength=3).float()
            / max(linear_intent_true.numel(), 1)
        ).tolist(),
        "angular_direction_distribution": (
            torch.bincount(angular_intent_true, minlength=3).float()
            / max(angular_intent_true.numel(), 1)
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = np.load(args.npz)
    features_np = data["policy_input"].astype(np.float32)
    labels_np = data["teacher_action"].astype(np.int64)
    feasible_np = data["feasible"].astype(bool)
    env_ids_np = data["env_id"].astype(np.int64)
    features_np = features_np[feasible_np]
    labels_np = labels_np[feasible_np]
    env_ids_np = env_ids_np[feasible_np]

    unique_envs = np.unique(env_ids_np)
    if unique_envs.size < 4:
        raise ValueError("probe requires at least four distinct env IDs")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_envs)
    test_count = max(1, int(round(unique_envs.size * 0.25)))
    test_envs = unique_envs[:test_count]
    test_mask_np = np.isin(env_ids_np, test_envs)
    train_mask_np = ~test_mask_np
    if not train_mask_np.any() or not test_mask_np.any():
        raise RuntimeError("empty train or held-out split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features = torch.from_numpy(features_np[train_mask_np])
    test_features = torch.from_numpy(features_np[test_mask_np])
    train_labels = torch.from_numpy(labels_np[train_mask_np]).long()
    test_labels = torch.from_numpy(labels_np[test_mask_np]).long()
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-4)
    train_features = ((train_features - mean) / std).clamp(-8.0, 8.0)
    test_features = ((test_features - mean) / std).clamp(-8.0, 8.0)

    num_bins = int(max(labels_np.max() + 1, 19))
    model = ActionProbe(
        train_features.shape[1], args.hidden_dim, num_bins
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    linear_weight = _class_weights(train_labels[:, 0], num_bins).to(device)
    angular_weight = _class_weights(train_labels[:, 1], num_bins).to(device)
    linear_loss = nn.CrossEntropyLoss(weight=linear_weight)
    angular_loss = nn.CrossEntropyLoss(weight=angular_weight)
    loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    best_loss = float("inf")
    best_state = None
    for _ in range(args.epochs):
        model.train()
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            linear_logits, angular_logits = model(batch_features)
            loss = linear_loss(
                linear_logits, batch_labels[:, 0]
            ) + angular_loss(angular_logits, batch_labels[:, 1])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_linear, val_angular = model(test_features.to(device))
            val_loss = (
                nn.functional.cross_entropy(
                    val_linear, test_labels[:, 0].to(device)
                )
                + nn.functional.cross_entropy(
                    val_angular, test_labels[:, 1].to(device)
                )
            ).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("probe did not produce a model")
    model.load_state_dict(best_state)
    report = {
        "dataset": str(Path(args.npz).resolve()),
        "input_dim": int(train_features.shape[1]),
        "num_bins": num_bins,
        "train_env_ids": sorted(
            int(value) for value in np.unique(env_ids_np[train_mask_np])
        ),
        "held_out_env_ids": sorted(int(value) for value in test_envs),
        "train_frames": int(train_features.shape[0]),
        "best_held_out_loss": best_loss,
        "held_out": _evaluate(
            model,
            test_features.to(device),
            test_labels.to(device),
            center=num_bins // 2,
        ),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
