#!/usr/bin/env python3
"""Fit a deployable corridor residual adapter to privileged teacher actions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from modular_rnn_models import CorridorResidualAdapter, PolicyHead


def _load_datasets(
    paths: list[str],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, object]],
]:
    feature_parts: list[torch.Tensor] = []
    action_parts: list[torch.Tensor] = []
    env_id_parts: list[torch.Tensor] = []
    dataset_id_parts: list[torch.Tensor] = []
    summaries: list[dict[str, object]] = []
    env_offset = 0

    for dataset_id, raw_path in enumerate(paths):
        path = Path(raw_path).expanduser().resolve()
        with np.load(path) as data:
            features = torch.from_numpy(
                data["policy_input"].astype(np.float32)
            )
            teacher_actions = torch.from_numpy(
                data["teacher_action"].astype(np.int64)
            ).long()
            feasible = torch.from_numpy(data["feasible"].astype(bool))
            raw_env_ids = data["env_id"].astype(np.int64)
            unique_env_ids, local_env_ids = np.unique(
                raw_env_ids,
                return_inverse=True,
            )
            env_ids = torch.from_numpy(
                local_env_ids.astype(np.int64) + env_offset
            ).long()
            metadata: dict[str, object] = {}
            if "metadata_json" in data.files:
                metadata = json.loads(
                    str(np.asarray(data["metadata_json"]).item())
                )

        if features.ndim != 2 or features.shape[1] <= 83:
            raise ValueError(
                f"{path}: dataset must contain full K8 policy features"
            )
        if teacher_actions.shape != (features.shape[0], 2):
            raise ValueError(
                f"{path}: teacher actions must have shape [N, 2]"
            )
        if feasible.shape != features.shape[:1]:
            raise ValueError(f"{path}: feasible must have shape [N]")
        if env_ids.shape != features.shape[:1]:
            raise ValueError(f"{path}: env_id must have shape [N]")

        feature_parts.append(features[feasible])
        action_parts.append(teacher_actions[feasible])
        env_id_parts.append(env_ids[feasible])
        dataset_id_parts.append(
            torch.full(
                (int(feasible.sum()),),
                dataset_id,
                dtype=torch.long,
            )
        )
        summaries.append(
            {
                "path": str(path),
                "trajectory_source": metadata.get(
                    "trajectory_source", "unknown"
                ),
                "teacher_replaced_policy_actions": metadata.get(
                    "teacher_replaced_policy_actions", None
                ),
                "frames": int(features.shape[0]),
                "feasible_frames": int(feasible.sum()),
                "environment_count": int(unique_env_ids.size),
            }
        )
        env_offset += int(unique_env_ids.size)

    return (
        torch.cat(feature_parts, dim=0),
        torch.cat(action_parts, dim=0),
        torch.cat(env_id_parts, dim=0),
        torch.cat(dataset_id_parts, dim=0),
        summaries,
    )


def _intent(indices: torch.Tensor, center: int) -> torch.Tensor:
    return torch.where(
        indices < center,
        torch.zeros_like(indices),
        torch.where(
            indices > center,
            torch.full_like(indices, 2),
            torch.ones_like(indices),
        ),
    )


def _balanced_accuracy(
    truth: torch.Tensor,
    prediction: torch.Tensor,
) -> float:
    recalls = []
    for class_id in torch.unique(truth):
        mask = truth == class_id
        recalls.append(
            float((prediction[mask] == truth[mask]).float().mean())
        )
    return float(sum(recalls) / max(len(recalls), 1))


def _action_indices(logits: torch.Tensor, num_bins: int) -> torch.Tensor:
    return torch.stack(
        (
            logits[:, :num_bins].argmax(dim=-1),
            logits[:, num_bins:].argmax(dim=-1),
        ),
        dim=-1,
    )


def _metrics(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    *,
    num_bins: int,
) -> dict[str, float]:
    center = num_bins // 2
    exact = prediction == teacher
    within_one = (prediction - teacher).abs() <= 1
    linear_intent = _intent(prediction[:, 0], center)
    teacher_linear_intent = _intent(teacher[:, 0], center)
    angular_intent = _intent(prediction[:, 1], center)
    teacher_angular_intent = _intent(teacher[:, 1], center)
    return {
        "linear_exact": float(exact[:, 0].float().mean()),
        "angular_exact": float(exact[:, 1].float().mean()),
        "joint_exact": float(exact.all(dim=-1).float().mean()),
        "linear_within_one": float(within_one[:, 0].float().mean()),
        "angular_within_one": float(within_one[:, 1].float().mean()),
        "joint_within_one": float(within_one.all(dim=-1).float().mean()),
        "linear_intent": float(
            (linear_intent == teacher_linear_intent).float().mean()
        ),
        "angular_direction": float(
            (angular_intent == teacher_angular_intent).float().mean()
        ),
        "linear_intent_balanced": _balanced_accuracy(
            teacher_linear_intent, linear_intent
        ),
        "angular_direction_balanced": _balanced_accuracy(
            teacher_angular_intent, angular_intent
        ),
        "brake_fraction": float((prediction[:, 0] < center).float().mean()),
        "coast_fraction": float((prediction[:, 0] == center).float().mean()),
        "strong_turn_fraction": float(
            ((prediction[:, 1] - center).abs() >= 6).float().mean()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Teacher-labeled NPZ; repeat this option to aggregate "
            "teacher- and student-trajectory datasets"
        ),
    )
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--gate_checkpoint", required=True)
    parser.add_argument("--output_checkpoint", required=True)
    parser.add_argument("--output_report", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--max_logit_delta", type=float, default=6.0)
    parser.add_argument("--held_out_fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 < args.held_out_fraction < 1.0:
        raise ValueError("held_out_fraction must be in (0, 1)")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    (
        features,
        teacher_actions,
        env_ids,
        dataset_ids,
        dataset_summaries,
    ) = _load_datasets(args.dataset)

    unique_envs = np.unique(env_ids.numpy())
    if unique_envs.size < 4:
        raise ValueError("at least four environment IDs are required")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_envs)
    held_out_count = max(
        1, int(round(unique_envs.size * args.held_out_fraction))
    )
    held_out_envs = unique_envs[:held_out_count]
    held_out_mask = torch.from_numpy(
        np.isin(env_ids.numpy(), held_out_envs)
    )
    train_mask = ~held_out_mask
    if not bool(train_mask.any()) or not bool(held_out_mask.any()):
        raise RuntimeError("empty training or held-out split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = dict(checkpoint.get("args", {}))
    num_bins = int(checkpoint_args.get("num_bins", 19))
    expected_logits = 2 * num_bins
    policy_head = PolicyHead(input_dim=features.shape[1]).to(device)
    policy_head.load_state_dict(checkpoint["policy_head"], strict=True)
    policy_head.eval()
    policy_head.requires_grad_(False)

    adapter = CorridorResidualAdapter(
        input_dim=83,
        residual_input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        gate_init_probability=0.01,
        max_logit_delta=args.max_logit_delta,
    ).to(device)
    gate_state = torch.load(
        args.gate_checkpoint,
        map_location=device,
        weights_only=True,
    )
    adapter.gate_net.load_state_dict(gate_state, strict=True)
    adapter.gate_net.requires_grad_(False)
    optimizer = torch.optim.Adam(
        adapter.residual_net.parameters(),
        lr=args.learning_rate,
    )

    train_features = features[train_mask]
    train_actions = teacher_actions[train_mask]
    held_out_features = features[held_out_mask].to(device)
    held_out_actions = teacher_actions[held_out_mask].to(device)
    loader = DataLoader(
        TensorDataset(train_features, train_actions),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        adapter.gate_net.eval()
        for batch_features, batch_actions in loader:
            batch_features = batch_features.to(device)
            batch_actions = batch_actions.to(device)
            with torch.no_grad():
                base_logits = policy_head(batch_features)
            residual, _, _ = adapter(
                batch_features[:, :83],
                batch_features,
            )
            logits = base_logits + residual
            if logits.shape[1] != expected_logits:
                raise RuntimeError("unexpected policy logit width")
            loss = (
                nn.functional.cross_entropy(
                    logits[:, :num_bins], batch_actions[:, 0]
                )
                + nn.functional.cross_entropy(
                    logits[:, num_bins:], batch_actions[:, 1]
                )
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        adapter.eval()
        with torch.no_grad():
            held_out_residual, _, _ = adapter(
                held_out_features[:, :83],
                held_out_features,
            )
            held_out_logits = (
                policy_head(held_out_features) + held_out_residual
            )
            held_out_loss = (
                nn.functional.cross_entropy(
                    held_out_logits[:, :num_bins],
                    held_out_actions[:, 0],
                )
                + nn.functional.cross_entropy(
                    held_out_logits[:, num_bins:],
                    held_out_actions[:, 1],
                )
            ).item()
        if held_out_loss < best_loss:
            best_loss = held_out_loss
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())

    if best_state is None:
        raise RuntimeError("adapter fitting produced no checkpoint")
    adapter.load_state_dict(best_state)
    adapter.eval()
    with torch.no_grad():
        residual, _, gate_probability = adapter(
            held_out_features[:, :83],
            held_out_features,
        )
        base_logits = policy_head(held_out_features)
        adapted_prediction = _action_indices(
            base_logits + residual, num_bins
        )
        base_prediction = _action_indices(base_logits, num_bins)

    report = {
        "datasets": dataset_summaries,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "gate_checkpoint": str(Path(args.gate_checkpoint).resolve()),
        "train_frames": int(train_mask.sum()),
        "held_out_frames": int(held_out_mask.sum()),
        "train_frames_by_dataset": {
            str(index): int(
                ((dataset_ids == index) & train_mask).sum()
            )
            for index in range(len(dataset_summaries))
        },
        "held_out_frames_by_dataset": {
            str(index): int(
                ((dataset_ids == index) & held_out_mask).sum()
            )
            for index in range(len(dataset_summaries))
        },
        "train_env_ids": sorted(
            int(value)
            for value in np.unique(env_ids[train_mask].numpy())
        ),
        "held_out_env_ids": sorted(int(value) for value in held_out_envs),
        "best_epoch": best_epoch,
        "best_held_out_cross_entropy": best_loss,
        "hidden_dim": args.hidden_dim,
        "max_logit_delta": args.max_logit_delta,
        "gate_probability_mean": float(gate_probability.mean()),
        "base": _metrics(
            base_prediction, held_out_actions, num_bins=num_bins
        ),
        "adapted": _metrics(
            adapted_prediction, held_out_actions, num_bins=num_bins
        ),
        "teacher": _metrics(
            held_out_actions, held_out_actions, num_bins=num_bins
        ),
    }

    output_checkpoint = copy.deepcopy(checkpoint)
    output_checkpoint["corridor_adapter"] = adapter.state_dict()
    output_checkpoint["args"] = checkpoint_args
    output_checkpoint["args"].update(
        {
            "corridor_adapter_enabled": True,
            "corridor_adapter_hidden_dim": args.hidden_dim,
            "corridor_adapter_gate_loss_weight": 0.05,
            "corridor_adapter_gate_init_probability": 0.01,
            "corridor_adapter_max_logit_delta": args.max_logit_delta,
            "corridor_adapter_freeze_base": True,
            "corridor_adapter_gate_checkpoint": str(
                Path(args.gate_checkpoint).resolve()
            ),
            "corridor_adapter_residual_features": "policy_features",
            "no_resume_optimizer": True,
        }
    )
    output_checkpoint["corridor_adapter_distillation"] = report

    checkpoint_path = Path(args.output_checkpoint).expanduser()
    report_path = Path(args.output_report).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, checkpoint_path)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    print(f"[SAVE] checkpoint={checkpoint_path}")
    print(f"[SAVE] report={report_path}")


if __name__ == "__main__":
    main()
