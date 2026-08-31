from __future__ import annotations

from scripts.reinforcement_learning.skrl.rnn_car_wdclean import (
    analyze_sa5_b3_c500_c600_exposure as audit,
)


def _row(iteration: int, p060_0_active: int, p060_1_active: int):
    row = {
        "iteration": iteration,
        "corridor_profile_family/accounting/reconciliation_ok": 1.0,
        "corridor_profile_family/accounting/active_steps_total": float(
            p060_0_active + p060_1_active
        ),
        "corridor_profile_family/accounting/resets_total": 20.0,
        "corridor_profile_family/accounting/episodes_total": 20.0,
    }
    for profile, active in (
        ("p060_0s1d", p060_0_active),
        ("p060_1s1d", p060_1_active),
    ):
        prefix = f"corridor_profile/{profile}"
        row.update(
            {
                f"{prefix}/active_steps": float(active),
                f"{prefix}/reset_count": 10.0,
                f"{prefix}/episodes": 10.0,
                f"{prefix}/sr": 0.9,
                f"{prefix}/cr": 0.1,
                f"{prefix}/timeout": 0.0,
            }
        )
        family_prefix = f"corridor_profile_family/{profile}/lateral"
        row.update(
            {
                f"{family_prefix}/active_steps": float(active),
                f"{family_prefix}/reset_count": 10.0,
                f"{family_prefix}/episodes": 10.0,
                f"{family_prefix}/sr": 0.9,
                f"{family_prefix}/cr": 0.1,
                f"{family_prefix}/timeout": 0.0,
            }
        )
    for profile in ("p060_2s1d", "p035_3s2d", "p035_4s2d"):
        prefix = f"corridor_profile/{profile}"
        row.update(
            {
                f"{prefix}/active_steps": 0.0,
                f"{prefix}/reset_count": 0.0,
                f"{prefix}/episodes": 0.0,
                f"{prefix}/sr": 0.0,
                f"{prefix}/cr": 0.0,
                f"{prefix}/timeout": 0.0,
            }
        )
    return row


def _screen():
    def candidate(name, p060_pass, p080_pass):
        return {
            "checkpoint_name": name,
            "p060_absolute_pass": p060_pass,
            "p080_retention_pass": p080_pass,
        }

    return {
        "candidate_results": [
            candidate("c500", False, True),
            candidate("c550", True, False),
            candidate("c600", False, True),
        ]
    }


def test_aggregate_window_recomputes_shares_from_counts():
    rows = [_row(i, 40, 60) for i in range(1, 3)]
    result = audit.aggregate_window(rows, 1, 2)
    assert result["profiles"]["p060_0s1d"]["active_step_share"] == 0.4
    assert result["profiles"]["p060_1s1d"]["active_step_share"] == 0.6
    assert result["profiles"]["p060_0s1d"]["cr"] == 0.1


def test_decision_requires_zero_p080_stable_exposure_and_fixed_tradeoff():
    window = {
        "profiles": {
            name: {"active_step_share": 0.1, "reset_share": 0.1}
            for name in audit.LOW_DENSITY_PROFILES
        }
    }
    windows = {name: window for name in audit.WINDOWS}
    assert audit.decide_pilot(windows, _screen(), 0)["establish_pilot"] is True
    assert audit.decide_pilot(windows, _screen(), 1)["establish_pilot"] is False


def test_proposed_mix_preserves_p060_and_sums_to_one():
    current_p060 = [row for row in audit.CURRENT_MIX if row[1] == (0.50, 0.70)]
    proposed_p060 = [row for row in audit.PROPOSED_MIX if row[1] == (0.50, 0.70)]
    assert proposed_p060 == current_p060
    assert sum(row[2] for row in audit.PROPOSED_MIX) == 1.0
    assert [row for row in audit.PROPOSED_MIX if row[1] == (0.70, 0.90)] == [
        ((0, 1), (0.70, 0.90), 0.05),
        ((1, 1), (0.70, 0.90), 0.05),
    ]


def test_default_paths_resolve_inside_repo():
    assert audit.REPO.name == "IsaacLab"
    assert audit.DEFAULT_METRICS.is_file()
    assert audit.DEFAULT_SCREEN.is_file()
