"""Unit tests for corridor observability probe metrics."""

import torch

from scene_observability_probe import _metrics


def test_binary_metrics_report_recall_and_specificity() -> None:
    logits = torch.tensor([2.0, -2.0, 1.0, -1.0])
    targets = torch.tensor([True, False, False, True])
    metrics = _metrics(logits, targets)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
