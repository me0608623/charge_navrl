from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from rnn_car_modular.experiment_config import (
    ExperimentConfig,
    apply_experiment_config,
)


SKRL_ROOT = Path(__file__).resolve().parents[1]
TRAINER = SKRL_ROOT / "train" / "train_rnn_car_wdclip.py"
PLAY = SKRL_ROOT / "play_eval" / "play_rnn_car.py"
METRICS = SKRL_ROOT / "rnn_car_wdclean" / "teacher_closed_loop_metrics.py"


def test_historical_experiment_default_is_r1():
    assert ExperimentConfig().corridor_teacher_goal_denominator_floor_m == 1.0


def test_experiment_config_can_opt_in_without_changing_cli_override_rules():
    args = Namespace(corridor_teacher_goal_denominator_floor_m=1.0)
    cfg = replace(
        ExperimentConfig(),
        corridor_teacher_goal_denominator_floor_m=10.0,
    )

    applied = apply_experiment_config(args, cfg, ["train.py"])
    assert "corridor_teacher_goal_denominator_floor_m" in applied
    assert args.corridor_teacher_goal_denominator_floor_m == 10.0

    args = Namespace(corridor_teacher_goal_denominator_floor_m=1.0)
    apply_experiment_config(
        args,
        cfg,
        ["train.py", "--corridor_teacher_goal_denominator_floor_m", "4"],
    )
    assert args.corridor_teacher_goal_denominator_floor_m == 1.0


def test_train_and_play_both_wire_the_parameter_into_teacher_spec():
    trainer = TRAINER.read_text(encoding="utf-8")
    play = PLAY.read_text(encoding="utf-8")
    field = "corridor_teacher_goal_denominator_floor_m"

    assert f'"--{field}"' in trainer
    assert f'"--{field}"' in play
    assert "goal_denominator_floor_m=(" in trainer
    assert "goal_denominator_floor_m=(" in play
    assert f'"{field}": _corridor_teacher_goal_denominator_floor_m' in trainer
    assert '"goal_denominator_floor_m": (' in play


def test_play_reports_clearance_near_goal_and_dynamic_pause_evidence():
    play = PLAY.read_text(encoding="utf-8")
    metrics = METRICS.read_text(encoding="utf-8")
    for marker in (
        "selected_predicted_clearance",
        "near_goal_episode_outcomes",
    ):
        assert marker in metrics
    for marker in (
        "motion_phase_audit",
        "dynamic_pause_steps_range",
    ):
        assert marker in play
