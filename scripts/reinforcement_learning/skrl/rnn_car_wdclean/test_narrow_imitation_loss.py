"""Hard-label imitation loss against the scripted direct-crossing teacher.

07-27 verdict (N1): the SA5 narrow teacher KL is removed because that
teacher detours; the replacement supervision is a cross-entropy against the
scripted teacher's *action indices*, applied only on narrow-replay frames:

    total = PPO loss + lambda * [CE(linear head, teacher linear)
                                 + CE(angular head, teacher angular)]

The existing `masked_two_head_retention_loss` consumes teacher *logits*
(soft targets from a network); the scripted teacher emits a single chosen
bin per head, so it needs its own hard-label path. These tests pin the
masking contract in particular: an all-false mask must still return a
gradient-carrying zero, and non-narrow frames must contribute nothing.
"""

import math
import unittest

import torch

from rnn_car_wdclean.narrow_imitation_loss import (
    NUM_ACTION_BINS,
    scripted_action_ce_loss,
)


def _logits(batch, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, 2 * NUM_ACTION_BINS, generator=g, requires_grad=True)


def _indices(pairs):
    return torch.tensor(pairs, dtype=torch.long)


class MaskingContractTest(unittest.TestCase):
    def test_empty_mask_returns_gradient_carrying_zero(self):
        """No narrow env this batch must not break backward()."""
        logits = _logits(4)
        out = scripted_action_ce_loss(
            logits, _indices([[0, 0]] * 4), torch.zeros(4, dtype=torch.bool)
        )
        self.assertEqual(float(out.loss), 0.0)
        self.assertEqual(out.active_count, 0)
        out.loss.backward()
        self.assertIsNotNone(logits.grad)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_only_masked_frames_receive_gradient(self):
        logits = _logits(4, seed=1)
        mask = torch.tensor([False, True, False, True])
        out = scripted_action_ce_loss(logits, _indices([[3, 5]] * 4), mask)
        out.loss.backward()
        grad = logits.grad
        self.assertTrue(bool((grad[~mask] == 0).all()))
        self.assertTrue(bool((grad[mask] != 0).any()))

    def test_loss_equals_mean_over_masked_frames_only(self):
        logits = _logits(6, seed=2)
        idx = _indices([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        mask = torch.tensor([True, False, True, False, False, True])
        out = scripted_action_ce_loss(logits, idx, mask)

        sel = logits[mask]
        sel_idx = idx[mask]
        expect_lin = torch.nn.functional.cross_entropy(
            sel[:, :NUM_ACTION_BINS], sel_idx[:, 0]
        )
        expect_ang = torch.nn.functional.cross_entropy(
            sel[:, NUM_ACTION_BINS:], sel_idx[:, 1]
        )
        torch.testing.assert_close(out.ce_linear, expect_lin)
        torch.testing.assert_close(out.ce_angular, expect_ang)
        torch.testing.assert_close(out.loss, expect_lin + expect_ang)
        self.assertEqual(out.active_count, 3)

    def test_masked_result_matches_running_only_the_masked_rows(self):
        """Padding a batch with non-narrow frames must not move the loss."""
        core = _logits(3, seed=3)
        pad = torch.zeros(2, 2 * NUM_ACTION_BINS, requires_grad=True)
        full = torch.cat([core, pad], dim=0)
        idx_core = _indices([[2, 3], [4, 5], [6, 7]])
        idx_full = torch.cat([idx_core, _indices([[0, 0], [0, 0]])], dim=0)

        a = scripted_action_ce_loss(
            core, idx_core, torch.ones(3, dtype=torch.bool)
        )
        b = scripted_action_ce_loss(
            full,
            idx_full,
            torch.tensor([True, True, True, False, False]),
        )
        torch.testing.assert_close(a.loss, b.loss)


class ValueTest(unittest.TestCase):
    def test_confident_agreement_drives_loss_to_zero(self):
        logits = torch.zeros(1, 2 * NUM_ACTION_BINS)
        logits[0, 7] = 60.0                       # linear head picks bin 7
        logits[0, NUM_ACTION_BINS + 11] = 60.0    # angular head picks bin 11
        out = scripted_action_ce_loss(
            logits, _indices([[7, 11]]), torch.ones(1, dtype=torch.bool)
        )
        self.assertLess(float(out.loss), 1e-6)

    def test_uniform_logits_give_two_log_bins(self):
        logits = torch.zeros(2, 2 * NUM_ACTION_BINS)
        out = scripted_action_ce_loss(
            logits, _indices([[0, 1], [2, 3]]), torch.ones(2, dtype=torch.bool)
        )
        self.assertAlmostEqual(
            float(out.loss), 2.0 * math.log(NUM_ACTION_BINS), places=5
        )

    def test_agreement_rates_are_reported(self):
        logits = torch.zeros(2, 2 * NUM_ACTION_BINS)
        logits[0, 4] = 10.0
        logits[0, NUM_ACTION_BINS + 4] = 10.0
        logits[1, 9] = 10.0
        logits[1, NUM_ACTION_BINS + 2] = 10.0
        out = scripted_action_ce_loss(
            logits,
            _indices([[4, 4], [9, 15]]),   # env1's angular choice disagrees
            torch.ones(2, dtype=torch.bool),
        )
        self.assertAlmostEqual(out.agreement_linear, 1.0, places=6)
        self.assertAlmostEqual(out.agreement_angular, 0.5, places=6)


class ValidationTest(unittest.TestCase):
    def test_rejects_wrong_logit_width(self):
        with self.assertRaises(ValueError):
            scripted_action_ce_loss(
                torch.zeros(2, 30),
                _indices([[0, 0], [0, 0]]),
                torch.ones(2, dtype=torch.bool),
            )

    def test_rejects_mismatched_batch(self):
        with self.assertRaises(ValueError):
            scripted_action_ce_loss(
                torch.zeros(3, 2 * NUM_ACTION_BINS),
                _indices([[0, 0], [0, 0]]),
                torch.ones(3, dtype=torch.bool),
            )
        with self.assertRaises(ValueError):
            scripted_action_ce_loss(
                torch.zeros(2, 2 * NUM_ACTION_BINS),
                _indices([[0, 0], [0, 0]]),
                torch.ones(3, dtype=torch.bool),
            )

    def test_rejects_out_of_range_teacher_index(self):
        with self.assertRaises(ValueError):
            scripted_action_ce_loss(
                torch.zeros(1, 2 * NUM_ACTION_BINS),
                _indices([[NUM_ACTION_BINS, 0]]),
                torch.ones(1, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
