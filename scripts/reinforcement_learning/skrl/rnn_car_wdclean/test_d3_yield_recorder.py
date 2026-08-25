"""CPU-only contracts for the staged SA4-D3 recorder and shield hook."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import analyze_sa4_d3_baseline as analysis
import d3_yield_recorder as d3


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
ACTION_TERM = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/actions/discrete_differential_drive.py"
)
RUNNER = HERE / "run_sa4_d3_baseline.py"
INTERVENTION_RUNNER = HERE / "run_sa4_d3_intervention_suite.py"


def _record(
    step: int,
    *,
    env_id: int = 0,
    episode_id: int = 0,
    policy: tuple[int, int] = (9, 9),
    effective: tuple[int, int] | None = None,
    engaged: bool = False,
    pre: tuple[float, float] = (0.5, 0.0),
    post: tuple[float, float] = (0.5, 0.0),
    forward_speed: float = 0.5,
    planar_speed: float = 0.5,
    distance: float = 5.0,
    closing: float = 0.0,
    risk: float = 0.0,
    risk_active: bool = True,
    slot_distances: tuple[float, ...] | None = None,
    slot_closings: tuple[float, ...] | None = None,
    slot_risks: tuple[float, ...] | None = None,
    slot_risk_active: tuple[bool, ...] | None = None,
    dynamic_collision: bool = False,
    done: bool = False,
    cause: int = 0,
) -> d3.StepRecord:
    return d3.StepRecord(
        step=step,
        env_id=env_id,
        episode_id=episode_id,
        policy_action_indices=policy,
        effective_action_indices=policy if effective is None else effective,
        shield_engaged=engaged,
        pre_delay_v_command_mps=pre[0],
        pre_delay_w_command_rad_s=pre[1],
        post_delay_v_command_mps=post[0],
        post_delay_w_command_rad_s=post[1],
        body_forward_speed_mps=forward_speed,
        body_planar_speed_mps=planar_speed,
        obstacle_center_distance_m=distance,
        relative_closing_speed_mps=closing,
        risk=risk,
        risk_active=risk_active,
        dynamic_obstacle_center_distances_m=(
            (distance, distance + 1.0)
            if slot_distances is None
            else slot_distances
        ),
        dynamic_obstacle_relative_closing_speeds_mps=(
            (closing, 0.0) if slot_closings is None else slot_closings
        ),
        dynamic_obstacle_risks=(
            (risk, 0.0) if slot_risks is None else slot_risks
        ),
        dynamic_obstacle_risk_active=(
            (risk_active, False)
            if slot_risk_active is None
            else slot_risk_active
        ),
        dynamic_collision=dynamic_collision,
        done=done,
        termination_cause=cause,
    )


def test_baseline_shield_returns_bitwise_identical_tensor_object():
    shield = d3.build_shield(d3.ShieldMode.BASELINE)
    actions = torch.tensor([[0.0, 18.0], [9.0, 9.0]])
    effective = shield(actions, context={"risk": torch.ones(2)})
    assert effective is actions
    assert torch.equal(effective, actions)


def test_all_preregistered_arms_are_enabled_after_baseline_analysis():
    assert {mode.value for mode in d3.ShieldMode} == {
        "baseline",
        "sustained_brake",
        "best_turn",
        "combined",
    }
    assert d3.ENABLED_MODES == frozenset(d3.ShieldMode)


def _shield_context(
    *,
    distance: float = 2.5,
    closing: float = 1.0,
    body_y: float = 0.5,
    body_vy: float = 0.2,
    brake_index: int = 0,
) -> dict[str, torch.Tensor]:
    return {
        "obstacle_distances_m": torch.tensor([[distance]]),
        "relative_closing_speeds_mps": torch.tensor([[closing]]),
        "obstacle_body_y_m": torch.tensor([[body_y]]),
        "obstacle_body_vy_mps": torch.tensor([[body_vy]]),
        "brake_action_indices": torch.tensor([brake_index]),
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (d3.ShieldMode.SUSTAINED_BRAKE, [0.0, 12.0]),
        (d3.ShieldMode.BEST_TURN, [16.0, 0.0]),
        (d3.ShieldMode.COMBINED, [0.0, 0.0]),
    ],
)
def test_two_frame_trigger_applies_only_the_registered_action_factors(
    mode, expected
):
    shield = d3.build_shield(mode)
    policy = torch.tensor([[16.0, 12.0]])
    first = shield(policy, _shield_context())
    second = shield(policy, _shield_context())
    assert torch.equal(first, policy)
    assert second[0].tolist() == expected
    report = shield.report()
    assert report["trigger_activations"] == 1
    assert report["active_environment_frames"] == 1


def test_turn_direction_is_latched_until_the_encounter_resolves():
    shield = d3.build_shield(d3.ShieldMode.BEST_TURN)
    policy = torch.tensor([[16.0, 9.0]])
    shield(policy, _shield_context(body_y=0.5))
    triggered = shield(policy, _shield_context(body_y=0.5))
    crossed_center = shield(policy, _shield_context(body_y=-0.5))
    assert triggered[0, 1].item() == d3.ACTION_RIGHT_INDEX
    assert crossed_center[0, 1].item() == d3.ACTION_RIGHT_INDEX


def test_centered_obstacle_turns_behind_its_lateral_motion():
    right = d3.build_shield(d3.ShieldMode.BEST_TURN)
    left = d3.build_shield(d3.ShieldMode.BEST_TURN)
    policy = torch.tensor([[16.0, 9.0]])
    for _ in range(2):
        right_action = right(
            policy, _shield_context(body_y=0.0, body_vy=0.2)
        )
        left_action = left(
            policy, _shield_context(body_y=0.0, body_vy=-0.2)
        )
    assert right_action[0, 1].item() == d3.ACTION_RIGHT_INDEX
    assert left_action[0, 1].item() == d3.ACTION_LEFT_INDEX


def test_release_requires_two_resolved_frames_and_reset_clears_latch():
    shield = d3.build_shield(d3.ShieldMode.COMBINED)
    policy = torch.tensor([[16.0, 12.0]])
    shield(policy, _shield_context())
    shield(policy, _shield_context())
    first_resolved = shield(policy, _shield_context(closing=0.0))
    second_resolved = shield(policy, _shield_context(closing=0.0))
    assert first_resolved[0].tolist() == [0.0, 0.0]
    assert torch.equal(second_resolved, policy)
    assert shield.report()["release_events"] == 1

    shield(policy, _shield_context())
    shield(policy, _shield_context())
    shield.reset(torch.tensor([True]))
    after_reset = shield(policy, _shield_context(distance=5.0, closing=0.0))
    assert torch.equal(after_reset, policy)
    assert shield.report()["active_at_end"] == 0


def test_frozen_protocol_is_complete_and_content_addressed():
    protocol = d3.shield_protocol()
    assert protocol["schema"] == "sa4_d3_intervention_protocol/v1"
    assert protocol["trigger"]["ttc_max_s"] == 2.0
    assert protocol["trigger"]["consecutive_frames"] == 2
    assert protocol["hold"]["release_consecutive_frames"] == 2
    assert protocol["actions"]["center_index"] == 9
    assert len(protocol["sha256"]) == 64


def test_window_holds_25_lead_steps_plus_event_frame():
    assert d3.MAX_LEAD_STEPS == 25
    assert d3.WINDOW_CAPACITY == 26
    assert d3.MAX_LEAD_STEPS * d3.DT_S == pytest.approx(5.0)
    buffer = d3.WindowBuffer()
    for step in range(40):
        buffer.append(_record(step))
    rows = buffer.window()
    assert len(rows) == 26
    assert rows[0].step == 14
    assert rows[-1].step == 39


def test_first_events_keep_issued_applied_and_actual_timelines_separate():
    rows = [
        _record(0, pre=(0.8, 0.0), post=(0.7, 0.0), planar_speed=0.7, distance=4.0),
        _record(1, pre=(0.8, 0.0), post=(0.7, 0.0), planar_speed=0.7, distance=2.9),
        _record(2, pre=(0.5, 0.0), post=(0.7, 0.0), planar_speed=0.7, distance=2.2, risk=0.06),
        _record(3, pre=(0.4, 0.0), post=(0.5, 0.0), planar_speed=0.7, distance=1.5, risk=0.10),
        _record(4, pre=(0.3, 0.0), post=(0.4, 0.0), planar_speed=0.4, distance=0.7, risk=0.20),
    ]
    events = d3.first_events(rows)
    assert events["first_near_3m_s_before_event"] == pytest.approx(0.6)
    assert events["first_risk_ge_0p05_s_before_event"] == pytest.approx(0.4)
    assert events["first_issued_decel_s_before_event"] == pytest.approx(0.4)
    assert events["first_applied_decel_s_before_event"] == pytest.approx(0.2)
    assert events["first_actual_decel_s_before_event"] == pytest.approx(0.0)
    assert events["first_issued_stop_s_before_event"] is None


def test_full_window_marks_warning_onset_as_left_censored():
    rows = [
        _record(step, distance=2.5, risk=0.10)
        for step in range(d3.WINDOW_CAPACITY)
    ]
    events = d3.first_events(rows)
    assert events["first_near_3m_s_before_event"] == pytest.approx(5.0)
    assert events["first_near_3m_left_censored"] is True
    assert events["first_risk_ge_0p05_left_censored"] is True


def test_d1_applied_command_equals_previous_issued_command():
    recorder = d3.D3YieldRecorder(1, expected_delay_steps=1)
    rows = [
        _record(0, pre=(0.8, 0.1), post=(0.0, 0.0)),
        _record(1, pre=(0.6, 0.2), post=(0.8, 0.1)),
        _record(
            2,
            pre=(0.4, 0.3),
            post=(0.6, 0.2),
            distance=0.7,
            dynamic_collision=True,
            done=True,
            cause=3,
        ),
    ]
    for row in rows:
        recorder.record_batch([row])
    report = recorder.report(
        metadata={"expected_records": 3, "completed_episodes": 1}
    )
    check = report["self_check"]
    assert check["delay_alignment_samples"] == 2
    assert check["delay_alignment_errors"] == 0
    assert check["delay_alignment_ok"] is True
    assert check["reconciliation_ok"] is True
    assert report["counts"]["dynamic_collision"] == 1


def test_immediate_application_is_a_d1_reconciliation_failure():
    recorder = d3.D3YieldRecorder(1, expected_delay_steps=1)
    recorder.record_batch([_record(0, pre=(0.8, 0.1), post=(0.0, 0.0))])
    # If the shield writes after the queue, 0.6 appears immediately instead of
    # the previous step's 0.8 command.
    recorder.record_batch([_record(1, pre=(0.6, 0.2), post=(0.6, 0.2))])
    report = recorder.report(metadata={"expected_records": 2})
    check = report["self_check"]
    assert check["delay_alignment_errors"] == 1
    assert check["delay_alignment_ok"] is False
    assert check["reconciliation_ok"] is False


def test_baseline_record_rejects_any_effective_action_change():
    recorder = d3.D3YieldRecorder(1)
    with pytest.raises(RuntimeError, match="baseline changed"):
        recorder.record_batch(
            [_record(0, policy=(9, 9), effective=(0, 18), engaged=True)]
        )


def test_record_rejects_an_incorrect_shield_engaged_flag():
    recorder = d3.D3YieldRecorder(
        1, mode=d3.ShieldMode.SUSTAINED_BRAKE
    )
    with pytest.raises(RuntimeError, match="does not match"):
        recorder.record_batch(
            [_record(0, policy=(9, 9), effective=(0, 9), engaged=False)]
        )


def test_intervention_report_requires_and_accepts_an_observed_override():
    recorder = d3.D3YieldRecorder(
        1, mode=d3.ShieldMode.SUSTAINED_BRAKE
    )
    recorder.record_batch(
        [
            _record(
                0,
                policy=(16, 12),
                effective=(0, 12),
                engaged=True,
                pre=(0.8, 0.1),
                post=(0.0, 0.0),
            )
        ]
    )
    recorder.record_batch(
        [
            _record(
                1,
                policy=(16, 12),
                effective=(0, 12),
                engaged=True,
                pre=(0.6, 0.2),
                post=(0.8, 0.1),
            )
        ]
    )
    report = recorder.report(metadata={"expected_records": 2})
    check = report["self_check"]
    assert check["baseline_action_identity_ok"] is None
    assert check["intervention_action_observed"] is True
    assert check["action_contract_ok"] is True
    assert check["reconciliation_ok"] is True


def test_geometry_selector_can_reuse_recorder_without_expanding_d3_enum():
    assert "geometry_feasible" not in {mode.value for mode in d3.ShieldMode}
    recorder = d3.D3YieldRecorder(1, mode="geometry_feasible")
    recorder.record_batch(
        [
            _record(
                0,
                policy=(18, 9),
                effective=(18, 13),
                engaged=True,
                pre=(0.8, 0.1),
                post=(0.0, 0.0),
            )
        ]
    )
    recorder.record_batch(
        [
            _record(
                1,
                policy=(18, 9),
                effective=(18, 13),
                engaged=True,
                pre=(0.7, 0.2),
                post=(0.8, 0.1),
            )
        ]
    )

    report = recorder.report(metadata={"expected_records": 2})

    assert report["mode"] == "geometry_feasible"
    assert report["self_check"]["action_contract_ok"] is True
    assert report["self_check"]["reconciliation_ok"] is True


def test_recorder_rejects_unknown_external_mode():
    with pytest.raises(ValueError, match="unsupported D3 recorder mode"):
        d3.D3YieldRecorder(1, mode="not_registered")


def test_collision_and_noncollision_control_events_are_separate():
    recorder = d3.D3YieldRecorder(2)
    recorder.record_batch(
        [
            _record(0, env_id=0, distance=2.5, closing=0.3),
            _record(0, env_id=1, distance=2.5, closing=0.3),
        ]
    )
    recorder.record_batch(
        [
            _record(
                1,
                env_id=0,
                distance=0.7,
                dynamic_collision=True,
                done=True,
                cause=3,
            ),
            _record(1, env_id=1, distance=2.0, closing=-0.1),
        ]
    )
    kinds = [event["event_type"] for event in recorder.events]
    assert kinds == ["dynamic_collision", "noncollision_closest_approach"]
    assert recorder.episode_id(0) == 1
    assert recorder.episode_id(1) == 0


def test_done_clears_history_and_next_episode_cannot_cross_boundary():
    recorder = d3.D3YieldRecorder(1)
    recorder.record_batch(
        [_record(0, distance=2.0, closing=0.3, done=True, cause=1)]
    )
    recorder.record_batch(
        [_record(1, episode_id=1, distance=2.0, closing=0.3)]
    )
    recorder.record_batch(
        [_record(2, episode_id=1, distance=1.8, closing=-0.1)]
    )
    latest = recorder.events[-1]
    assert latest["event_type"] == "noncollision_closest_approach"
    assert {row["episode_id"] for row in latest["records"]} == {1}


def test_closest_approach_control_is_annotated_with_episode_success():
    recorder = d3.D3YieldRecorder(1)
    recorder.record_batch([_record(0, distance=2.5, closing=0.3)])
    recorder.record_batch([_record(1, distance=1.8, closing=-0.1)])
    assert recorder.events[0]["episode_outcome_cause"] is None
    recorder.record_batch([_record(2, done=True, cause=1)])
    assert recorder.events[0]["episode_outcome_cause"] == 1
    report = recorder.report(metadata={})
    assert report["counts"]["successful_noncollision_closest_approach"] == 1


def test_nonfinite_data_is_refused():
    recorder = d3.D3YieldRecorder(1)
    with pytest.raises(ValueError, match="non-finite"):
        recorder.record_batch([_record(0, distance=float("nan"))])


def test_report_fails_closed_on_record_or_episode_ledger_mismatch():
    recorder = d3.D3YieldRecorder(1)
    recorder.record_batch([_record(0)])
    report = recorder.report(
        metadata={"expected_records": 2, "completed_episodes": 1}
    )
    assert report["self_check"]["record_coverage_ok"] is False
    assert report["self_check"]["episode_reconciliation_ok"] is False
    assert report["self_check"]["reconciliation_ok"] is False


def test_action_term_snapshots_bracket_delay_and_survive_reset():
    source = ACTION_TERM.read_text(encoding="utf-8")
    pre = source.index("self._last_pre_delay_command[:] = target_vel")
    delay = source.index("target_vel = apply_actuator_dynamics(", pre)
    post = source.index("self._last_post_delay_command[:, 0]", delay)
    assert pre < delay < post
    reset_source = source[source.index("    def reset(") :]
    assert "_last_pre_delay_command" not in reset_source
    assert "_last_post_delay_command" not in reset_source


def test_play_shield_hook_precedes_env_step_and_checks_tensor_identity():
    source = PLAY.read_text(encoding="utf-8")
    hook = source.index("_d3_effective_actions = _d3_shield(")
    identity = source.index("_d3_effective_actions is not _d3_policy_object", hook)
    bitwise = source.index("torch.equal(", identity)
    step = source.index(
        "next_obs, reward, terminated, truncated, info = env.step(", bitwise
    )
    assert hook < identity < bitwise < step


def test_play_passes_per_obstacle_trigger_context_and_resets_after_step():
    source = PLAY.read_text(encoding="utf-8")
    hook = source.index("_d3_effective_actions = _d3_shield(")
    for key in (
        '"obstacle_distances_m"',
        '"relative_closing_speeds_mps"',
        '"obstacle_body_y_m"',
        '"obstacle_body_vy_mps"',
        '"brake_action_indices"',
    ):
        assert source.index(key, hook) < source.index("env.step(", hook)
    step = source.index("env.step(", hook)
    record = source.index("_d3_recorder.record_batch(_d3_records)", step)
    reset = source.index("_d3_shield.reset(done)", record)
    assert hook < step < record < reset


def test_runner_freezes_the_baseline_cell_and_never_starts_training():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'GEOMETRY_STAGE = 4' in source
    assert 'SCENARIO = "corridor_lateral"' in source
    assert "SEED = 818" in source
    assert "DELAY_STEPS = 1" in source
    assert 'ACTUATOR_PROFILE = "sa1_delay_only"' in source
    assert "NUM_ENVS = 64" in source
    assert "ROLLOUT_STEPS = 2500" in source
    assert '"--d3_shield_mode",\n        "baseline"' in source
    for forbidden in (
        "train_rnn_car_wdclip.py",
        "systemctl start",
        "sa5-sim",
    ):
        assert forbidden not in source


def test_intervention_suite_preregisters_all_arms_before_gpu_rollouts():
    source = INTERVENTION_RUNNER.read_text(encoding="utf-8")
    protocol_write = source.index("protocol_path.write_text(")
    rollout = source.index("arms = [", protocol_write)
    assert protocol_write < rollout
    assert "ShieldMode.SUSTAINED_BRAKE" in source
    assert "ShieldMode.BEST_TURN" in source
    assert "ShieldMode.COMBINED" in source
    assert "--modes" not in source
    assert "PASS_SR = 0.90" in source
    assert "PASS_CR = 0.10" in source
    assert "PASS_TO = 0.05" in source
    for forbidden in (
        "train_rnn_car_wdclip.py",
        "systemctl start",
        "sa5-sim",
    ):
        assert forbidden not in source


def test_recorder_module_starts_no_processes():
    tree = ast.parse(Path(d3.__file__).read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(name.name.split(".")[0] for name in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert "subprocess" not in imports


def _analysis_event(records: list[d3.StepRecord], *, slot: int = 0) -> dict:
    return {
        "tracked_obstacle_slot": slot,
        "records": [record.as_dict() for record in records],
    }


def test_probe_trigger_requires_two_consecutive_qualifying_frames():
    one_frame = _analysis_event(
        [
            _record(0, slot_distances=(2.6,), slot_closings=(1.0,)),
            _record(1, slot_distances=(3.2,), slot_closings=(1.0,)),
        ]
    )
    assert analysis.probe_trigger_lead(one_frame) is None

    confirmed = _analysis_event(
        [
            _record(0, slot_distances=(3.2,), slot_closings=(1.0,)),
            _record(1, slot_distances=(2.6,), slot_closings=(1.0,)),
            _record(2, slot_distances=(2.4,), slot_closings=(1.0,)),
            _record(3, slot_distances=(0.7,), slot_closings=(1.0,)),
        ]
    )
    assert analysis.probe_trigger_lead(confirmed) == pytest.approx(0.2)


def test_full_stop_estimate_includes_the_d1_delay():
    assert analysis.estimated_full_stop_time_s(0.9634, 0.5, 0.2) == pytest.approx(
        2.1268
    )
    with pytest.raises(ValueError, match="positive"):
        analysis.estimated_full_stop_time_s(0.5, 0.0, 0.2)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"schema": "sa4_d3_yield_timing/v1"}, "yield_timing/v2"),
        ({"mode": "sustained_brake"}, "baseline arm"),
        ({"self_check": {"reconciliation_ok": False}}, "reconciliation"),
    ],
)
def test_analysis_fails_closed_before_interpreting_invalid_evidence(
    override, message
):
    payload = {
        "schema": "sa4_d3_yield_timing/v2",
        "mode": "baseline",
        "self_check": {"reconciliation_ok": True},
    }
    payload.update(override)
    with pytest.raises(ValueError, match=message):
        analysis.build_analysis(payload, Path("invalid.json"))


def test_markdown_uses_the_input_collision_count_instead_of_a_literal():
    report = {
        "input_counts": {
            "records": 10,
            "completed_episodes": 1,
            "dynamic_collision": 7,
            "successful_noncollision_closest_approach": 1,
        },
        "input_self_check": {
            "delay_alignment_errors": 0,
            "shield_engaged_records": 0,
            "reconciliation_ok": True,
        },
        "groups": {
            key: {
                "events": events,
                "event_body_speed_mps": {"p50": 0.5},
                "risk_conditioned_response": {
                    "risk_observed_fraction": 1.0,
                    "warning_lead_s": {"p50": 1.0},
                    "robust_two_frame_drop_fraction": 0.5,
                    "applied_stop_fraction": 0.25,
                },
            }
            for key, events in (
                ("dynamic_collision", 7),
                ("successful_closest_approach", 1),
                ("successful_close_approach_le_1m", 1),
            )
        },
        "mechanism_probe_trigger": {
            "baseline_activation": {
                key: {
                    "trigger_fraction": 1.0,
                    "trigger_lead_s": {"p50": 1.0},
                }
                for key in (
                    "dynamic_collision",
                    "successful_closest_approach",
                    "successful_close_approach_le_1m",
                )
            }
        },
        "stopping_time_check": {
            "median_speed_at_risk_warning_mps": 0.5,
            "max_deceleration_mps2": 0.5,
            "fixed_action_delay_s": 0.2,
            "estimated_full_stop_time_s": 1.2,
            "median_risk_warning_lead_s": 1.0,
        },
    }
    markdown = analysis.render_markdown(report)
    assert "All 7 dynamic collisions" in markdown
    assert "All 312 dynamic collisions" not in markdown
