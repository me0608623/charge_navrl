"""Tests for the diagnostic static-layout identity used to split corridor
outcomes by which mirror of the fixed lattice an episode drew.

Motivation: a GUI observation suggested obstacles sit in the same places on
every reset. Reading the sampler showed a fixed lattice plus a +/-0.10 m jitter
and a left/right mirror drawn at p=0.5. This identity records which mirror was
installed so realised SR/CR can be split by it.

The identity must:
  * separate the two mirrors of a 4-obstacle scene,
  * be invariant to the jitter (only the sign matters),
  * never let two different densities collide on one id,
  * disambiguate parked/inactive slots via an explicit active count,
  * and stay purely diagnostic.

Pure torch; no Isaac import needed because the helper lives in the geometry
module alongside the sampler.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_EVENTS = Path(
    "/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/"
    "locomotion/velocity/config/charge_skrl/mdp/events"
)
if str(_EVENTS) not in sys.path:
    sys.path.insert(0, str(_EVENTS))

from long_corridor_replay_geometry import (  # noqa: E402
    LongCorridorSpec,
    sample_conflict_free_layout,
    static_layout_id,
)


def _xy(x_signs, y=0.0, jitter=0.0):
    """Build ``[1, slots, 2]`` from a list of x signs."""
    xs = [s * 1.25 + jitter for s in x_signs]
    return torch.tensor([[[x, y] for x in xs]], dtype=torch.float32)


def test_two_mirrors_of_four_slots_differ():
    a = static_layout_id(_xy([-1, 1, -1, 1]))
    b = static_layout_id(_xy([1, -1, 1, -1]))
    assert int(a) != int(b)
    # active * 100 + bitmask; -+-+ sets bits 1 and 3 => 0b1010 = 10
    assert int(a) == 4 * 100 + 10
    assert int(b) == 4 * 100 + 5


def test_jitter_does_not_change_identity():
    base = int(static_layout_id(_xy([-1, 1, -1, 1])))
    for j in (-0.10, -0.05, 0.05, 0.10):
        assert int(static_layout_id(_xy([-1, 1, -1, 1], jitter=j))) == base


def test_densities_cannot_collide():
    """All-negative layouts of different densities must not share an id."""
    ids = {n: int(static_layout_id(_xy([-1] * n))) for n in (1, 2, 3, 4)}
    assert len(set(ids.values())) == 4, ids


def test_inactive_slots_disambiguated_by_active_count():
    """The mixed-density path passes every slot; the count must separate them."""
    two_active = torch.tensor(
        [[[1.25, 0.0], [-1.25, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    three_active = torch.tensor(
        [[[1.25, 0.0], [-1.25, 0.0], [-1.25, 0.0], [0.0, 0.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    assert int(static_layout_id(two_active, torch.tensor([2]))) != int(
        static_layout_id(three_active, torch.tensor([3]))
    )
    # Parked slots sit at x = 0 and therefore read as bit 0, not +x.
    assert int(static_layout_id(two_active)) == 5 * 100 + 1


def test_zero_slots_is_zero():
    empty = torch.zeros(3, 0, 2)
    assert torch.equal(static_layout_id(empty), torch.zeros(3, dtype=torch.long))


def test_batch_is_elementwise():
    batch = torch.cat([_xy([-1, 1, -1, 1]), _xy([1, -1, 1, -1])], dim=0)
    ids = static_layout_id(batch)
    assert ids.tolist() == [410, 405]


def test_rejects_bad_shape_and_count_mismatch():
    with pytest.raises(ValueError):
        static_layout_id(torch.zeros(2, 3))
    with pytest.raises(ValueError):
        static_layout_id(torch.zeros(2, 3, 2), torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError):
        static_layout_id(torch.zeros(1, 7, 2))


@pytest.mark.parametrize("mode", ["lateral", "longitudinal"])
def test_production_sampler_yields_exactly_two_layouts(mode):
    """Pins the measured claim this instrumentation exists to record.

    If a future change adds real static randomisation this test must be updated
    deliberately, which is the point: the split is only interpretable while the
    number of distinct skeletons is known.
    """
    spec = LongCorridorSpec(free_width=4.0)
    torch.manual_seed(0)
    static, _, _, _, _ = sample_conflict_free_layout(
        4000, spec, torch.device("cpu"), mode,
        static_obstacles=4, dynamic_obstacles=2,
    )
    ids = static_layout_id(static[:, :4])
    assert sorted({int(v) for v in ids.tolist()}) == [405, 410]
    # Both mirrors must actually appear near 50/50 or a split is meaningless.
    share = (ids == 405).float().mean().item()
    assert 0.45 < share < 0.55, share
