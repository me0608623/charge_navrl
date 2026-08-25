from __future__ import annotations

import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import d9_noise_closed_loop_ab as d9  # noqa: E402
import d9_seed_replication as replication  # noqa: E402


FREEZE = REPO / "docs/freeze/sa4_d9_seed_replication_v1.json"
RUNNER = HERE / "run_sa4_d9_seed_replication.py"


def _comparison(cr_delta: float, wall_delta: float, to_delta: float = 0.0) -> dict:
    return {
        "delta_b_minus_a": {
            "cr": cr_delta,
            "wall_cr": wall_delta,
            "to": to_delta,
        }
    }


def test_frozen_protocol_matches_runtime_protocol():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == replication.replication_protocol()


def test_protocol_freezes_preexisting_seeds_and_majority_rule():
    protocol = replication.replication_protocol()
    assert protocol["new_evaluator_seeds"] == [515, 616]
    assert protocol["all_evaluator_seeds"] == [515, 616, 818]
    assert protocol["pilot_authorization_rule"]["qualifying_seeds_required"] == 2
    assert protocol["seed_level_rule"]["wall_cr_delta_b_minus_a"] == "<= 0.005"


def test_seed_rule_requires_lower_cr_and_wall_noninferiority():
    assert replication.seed_qualifies(_comparison(-0.001, 0.005)) is True
    assert replication.seed_qualifies(_comparison(0.0, 0.0)) is False
    assert replication.seed_qualifies(_comparison(-0.01, 0.0051)) is False
    assert replication.seed_qualifies(_comparison(-0.01, 0.0, 0.0051)) is False


def test_two_of_three_authorizes_only_short_pilot():
    result = replication.compare_replications(
        {
            515: _comparison(-0.01, 0.001),
            616: _comparison(+0.01, 0.0),
            818: _comparison(-0.01, 0.005),
        }
    )
    assert result["qualifying_seed_count"] == 2
    assert result["sa4_r3_short_pilot_authorized"] is True
    assert result["sa5_authorized"] is False


def test_one_of_three_does_not_authorize_pilot():
    result = replication.compare_replications(
        {
            515: _comparison(-0.01, 0.006),
            616: _comparison(+0.01, 0.0),
            818: _comparison(-0.01, 0.004),
        }
    )
    assert result["qualifying_seed_count"] == 1
    assert result["sa4_r3_short_pilot_authorized"] is False


def test_runner_is_seed_parameterized_and_fail_closed():
    source = RUNNER.read_text(encoding="utf-8")
    assert "for seed in replication.NEW_SEEDS" in source
    assert "seed=seed" in source
    assert "baseline.SEED" not in source
    assert '"status": "INCOMPLETE_NO_VERDICT"' in source
    assert "train_rnn_car_wdclip" not in source
    assert "systemctl" not in source
    assert '"training_started": False' in source
    assert '"next_stage_started": False' in source
    assert d9.CURRENT_ARM in replication.replication_protocol()["arms"]

