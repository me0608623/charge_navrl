"""Static launch and wiring contracts for the c600 step trace."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
RUNNER = HERE / "run_sa5_c600_step_trace.py"


def test_play_hook_is_policy_only_and_after_d1_queue():
    source = PLAY.read_text(encoding="utf-8")
    capture = source.index("_policy_step_trace_recorder.record_tensor_batch(")
    step = source.rindex("next_obs, reward, terminated, truncated, info = env.step(", 0, capture)
    post = source.index("last_post_delay_command", step)
    assert step < post < capture
    assert '"teacher_replaced_policy_actions": False' in source


def test_runner_is_fixed_policy_eval_and_never_trains():
    source = RUNNER.read_text(encoding="utf-8")
    for required in (
        'SCENARIOS = ("lateral", "random_2d")',
        '"--d3_shield_mode", "baseline"',
        '"--policy_step_trace_output"',
        '"--speed_rate", "0.7"',
        '"--actuator_delay_range", "1", "1"',
    ):
        assert required in source
    for forbidden in (
        "train_rnn_car_wdclip.py",
        "systemctl start",
        "sa6-",
    ):
        assert forbidden not in source


def test_k8_capture_is_opt_in_and_keeps_base_protocol_unchanged():
    source = RUNNER.read_text(encoding="utf-8")
    assert "include_k8_input: bool = False" in source
    assert 'command.append("--policy_step_trace_include_k8_input")' in source
