"""Offline probe for crossing-direction observability in the policy LiDAR stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


DISTANCE_BINS = ((0.0, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, float("inf")))


class Probe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = (
            nn.Linear(input_dim, 2)
            if hidden_dim == 0
            else nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _trajectory_split(env_id: np.ndarray, episode_id: np.ndarray, labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.stack([env_id, episode_id], axis=1)
    unique, inverse = np.unique(groups, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    train_group = np.zeros(len(unique), dtype=bool)
    for crossing in (-1, 1):
        for wall in (-1, 1):
            candidate = []
            for group_index in range(len(unique)):
                first = np.flatnonzero(inverse == group_index)[0]
                if labels[first, 0] == crossing and labels[first, 1] == wall:
                    candidate.append(group_index)
            rng.shuffle(candidate)
            n_train = max(1, int(round(0.8 * len(candidate))))
            if len(candidate) > 1:
                n_train = min(n_train, len(candidate) - 1)
            train_group[np.asarray(candidate[:n_train], dtype=np.int64)] = True
    train = train_group[inverse]
    return train, ~train


def _fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    hidden_dim: int,
    epochs: int,
    seed: int,
) -> tuple[Probe, np.ndarray, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-5] = 1.0
    train_x = torch.from_numpy((x_train - mean) / std)
    train_y = torch.from_numpy((y_train > 0).astype(np.float32))
    model = Probe(train_x.shape[1], hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), 2048):
            index = order[start : start + 2048]
            loss = nn.functional.binary_cross_entropy_with_logits(model(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = model(torch.from_numpy((x_test - mean) / std)).numpy()
    return model, mean, std, logits


def _accuracy(logits: np.ndarray, labels: np.ndarray) -> list[float]:
    prediction = np.where(logits >= 0.0, 1, -1)
    return [(prediction[:, index] == labels[:, index]).mean().item() for index in range(2)]


def _rebuild_stack(
    current: np.ndarray,
    env_id: np.ndarray,
    episode_id: np.ndarray,
    episode_step: np.ndarray,
    frame_stack: int,
) -> np.ndarray:
    """Reconstruct current-to-oldest K-frame inputs without crossing episode boundaries."""
    rebuilt = np.zeros((len(current), frame_stack * 72), dtype=np.float32)
    groups = np.stack([env_id, episode_id], axis=1)
    _, inverse = np.unique(groups, axis=0, return_inverse=True)
    for group_index in range(inverse.max(initial=-1) + 1):
        indices = np.flatnonzero(inverse == group_index)
        indices = indices[np.argsort(episode_step[indices])]
        for offset in range(frame_stack - 1, len(indices)):
            current_index = indices[offset]
            history_indices = indices[offset - np.arange(frame_stack)]
            rebuilt[current_index] = current[history_indices].reshape(-1)
    return rebuilt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-stack", type=int, default=None,
                        help="重建指定 K 幀；省略時使用 dump 內 policy 原始 stack")
    args = parser.parse_args()

    dump = np.load(args.input, allow_pickle=False)
    source_stack = dump["lidar_stack"].astype(np.float32)
    source_frame_stack = int(dump["frame_stack"])
    if source_stack.shape[1] != source_frame_stack * 72:
        raise ValueError(f"invalid stack shape {source_stack.shape} for K={source_frame_stack}")
    if not bool(dump["factorial_wall_side"]):
        raise ValueError("probe dump must decorrelate wall side from crossing direction")

    frame_stack = source_frame_stack if args.frame_stack is None else args.frame_stack
    if frame_stack < 1:
        raise ValueError("frame stack must be positive")
    if frame_stack == source_frame_stack:
        stack = source_stack
    elif frame_stack < source_frame_stack:
        stack = source_stack[:, : frame_stack * 72]
    else:
        stack = _rebuild_stack(
            source_stack[:, :72], dump["env_id"], dump["episode_id"], dump["episode_step"], frame_stack
        )

    labels = np.stack([dump["crossing_sign"], dump["wall_sign"]], axis=1).astype(np.int8)
    ready = dump["episode_step"] >= frame_stack - 1
    usable = dump["moving"].astype(bool) & ready
    stack = stack[usable]
    labels = labels[usable]
    distance = dump["obstacle_distance_m"][usable]
    pedestrian_body_xy = dump["pedestrian_body_xy"][usable]
    env_id = dump["env_id"][usable]
    episode_id = dump["episode_id"][usable]
    train_mask, test_mask = _trajectory_split(env_id, episode_id, labels, args.seed)
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise RuntimeError("trajectory split produced an empty partition")

    feature_sets = {"current": stack[:, :72], f"stack_{frame_stack}": stack}
    report: dict[str, object] = {
        "input": str(args.input),
        "samples": int(len(stack)),
        "train_samples": int(train_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "frame_stack": frame_stack,
        "source_frame_stack": source_frame_stack,
        "split": "80/20 grouped by (env_id, episode_id), stratified by crossing_sign x wall_sign",
        "models": {},
    }
    fitted: dict[str, tuple[Probe, np.ndarray, np.ndarray]] = {}
    for feature_name, features in feature_sets.items():
        for model_name, hidden_dim in (("linear", 0), ("mlp", 128)):
            key = f"{feature_name}_{model_name}"
            model, mean, std, logits = _fit_probe(
                features[train_mask], labels[train_mask], features[test_mask], hidden_dim, args.epochs, args.seed
            )
            crossing_acc, wall_acc = _accuracy(logits, labels[test_mask])
            report["models"][key] = {"crossing_accuracy": crossing_acc, "wall_accuracy": wall_acc}
            fitted[key] = (model, mean, std)

    stack_key = f"stack_{frame_stack}_mlp"
    model, mean, std = fitted[stack_key]
    test_stack = stack[test_mask].copy()
    reversed_stack = test_stack.reshape(-1, frame_stack, 72)[:, ::-1, :].reshape(-1, frame_stack * 72).copy()
    with torch.no_grad():
        reversed_logits = model(torch.from_numpy((reversed_stack - mean) / std)).numpy()
    reversed_crossing, reversed_wall = _accuracy(reversed_logits, labels[test_mask])
    report["temporal_order_reversal"] = {
        "crossing_accuracy": reversed_crossing,
        "wall_accuracy": reversed_wall,
    }

    test_logits = None
    with torch.no_grad():
        test_logits = model(torch.from_numpy((test_stack - mean) / std)).numpy()
    bins = []
    test_distance = distance[test_mask]
    test_labels = labels[test_mask]
    for lower, upper in DISTANCE_BINS:
        selected = (test_distance >= lower) & (test_distance < upper)
        crossing_acc = None if not selected.any() else _accuracy(test_logits[selected], test_labels[selected])[0]
        bins.append({"range_m": [lower, None if np.isinf(upper) else upper], "n": int(selected.sum()), "crossing_accuracy": crossing_acc})
    report["crossing_accuracy_by_distance"] = bins

    corridor = np.abs(pedestrian_body_xy[test_mask, 1]) <= 0.75
    corridor_acc = None if not corridor.any() else _accuracy(test_logits[corridor], test_labels[corridor])[0]
    report["shared_crossing_corridor"] = {"abs_pedestrian_body_y_max_m": 0.75, "n": int(corridor.sum()), "crossing_accuracy": corridor_acc}

    stack_acc = float(report["models"][stack_key]["crossing_accuracy"])
    current_acc = float(report["models"]["current_mlp"]["crossing_accuracy"])
    temporal_drop = stack_acc - reversed_crossing
    report["gate"] = {
        "threshold": 0.90,
        "stack_accuracy_pass": stack_acc >= 0.90,
        "current_frame_accuracy": current_acc,
        "temporal_order_drop": temporal_drop,
        "temporal_evidence_pass": temporal_drop >= 0.05,
        "pass": stack_acc >= 0.90 and temporal_drop >= 0.05,
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
