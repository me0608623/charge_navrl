from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/reinforcement_learning/skrl/utils"))

import d9_noise_closed_loop_ab as d9  # noqa: E402
from charge_env_overrides import (  # noqa: E402
    _apply_lidar_distractor_eligibility_config,
)


OBS_FUNCTIONS = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py"
)
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
RUNNER = HERE / "run_sa4_d9_noise_closed_loop_ab.py"
FREEZE = REPO / "docs/freeze/sa4_d9_noise_closed_loop_ab_v1.json"


def _metrics(sr: float, cr: float, *, to: float = 0.0) -> dict:
    return {
        "n": 2000,
        "sr": sr,
        "cr": cr,
        "to": to,
        "obstacle_cr": max(0.0, cr - 0.005),
        "wall_cr": min(0.005, cr),
    }


def test_frozen_protocol_matches_runtime_protocol():
    assert json.loads(FREEZE.read_text(encoding="utf-8")) == d9.closed_loop_protocol()


def test_protocol_freezes_only_noise_eligibility_difference():
    protocol = d9.closed_loop_protocol()
    assert protocol["sole_behavioral_difference"] == "mixed-pixel distractor eligibility"
    assert protocol["arms"][d9.CURRENT_ARM]["distractor_eligibility"] == "all_rays"
    assert protocol["arms"][d9.CORRECTED_ARM]["distractor_eligibility"] == "valid_return_only"
    assert protocol["fixed_cell"]["actuator_delay_steps"] == 1
    assert protocol["fixed_cell"]["rollout_steps"] == 2500
    assert protocol["descriptive_rules"]["inferential_claim"] is False


def test_comparison_reports_directional_and_material_improvement():
    result = d9.compare_closed_loop(
        {
            d9.CURRENT_ARM: _metrics(0.83, 0.17),
            d9.CORRECTED_ARM: _metrics(0.86, 0.14),
        }
    )
    assert result["directional_closed_loop_improvement_observed"] is True
    assert result["material_closed_loop_improvement_observed"] is True
    assert result["delta_b_minus_a"]["cr"] == -0.03
    assert result["parent_or_sa5_authorized"] is False


def test_comparison_does_not_call_subthreshold_change_material():
    result = d9.compare_closed_loop(
        {
            d9.CURRENT_ARM: _metrics(0.830, 0.170),
            d9.CORRECTED_ARM: _metrics(0.834, 0.166),
        }
    )
    assert result["directional_closed_loop_improvement_observed"] is True
    assert result["material_closed_loop_improvement_observed"] is False


def test_env_override_sets_policy_and_critic_terms():
    policy_term = SimpleNamespace(params={})
    critic_term = SimpleNamespace(params={})
    env_cfg = SimpleNamespace(
        observations=SimpleNamespace(
            policy=SimpleNamespace(lidar_static=policy_term),
            critic=SimpleNamespace(lidar_static=critic_term),
        )
    )
    args = SimpleNamespace(lidar_distractor_eligibility="valid_return_only")
    _apply_lidar_distractor_eligibility_config(env_cfg, args)
    assert policy_term.params["distractor_eligibility"] == "valid_return_only"
    assert critic_term.params["distractor_eligibility"] == "valid_return_only"


def test_observation_draws_before_gating_and_defaults_to_historical_mode():
    source = OBS_FUNCTIONS.read_text(encoding="utf-8")
    signature = source[source.index("def wd_like_sweep_72(") : source.index(") -> torch.Tensor:", source.index("def wd_like_sweep_72("))]
    assert 'distractor_eligibility: str = "all_rays"' in signature
    block = source[source.index("# --- Mixed-pixel / distractor (ghost)") : source.index("if d7_trace_enabled:", source.index("# --- Mixed-pixel / distractor (ghost)"))]
    assert block.index("sampled_distractor_mask = torch.rand_like") < block.index("if distractor_eligibility ==")
    assert block.index("distractor_values = torch.rand_like") < block.index("if distractor_eligibility ==")
    assert "valid_return = valid_return & ~hole_mask" in block
    assert "distractor_mask = sampled_distractor_mask & valid_return" in block


def test_play_and_runner_are_explicit_and_fail_closed():
    play = PLAY.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert '"--lidar-distractor-eligibility"' in play
    assert 'default="all_rays"' in play
    assert '"--lidar-distractor-eligibility",\n                    eligibility' in runner
    assert "D9 closed-loop A/B refuses to overwrite evidence" in runner
    assert '"status": "INCOMPLETE_NO_VERDICT"' in runner
    assert '"training_started": False' in runner
    assert '"next_stage_started": False' in runner
    assert "train_rnn_car_wdclip" not in runner
    assert "systemctl" not in runner
