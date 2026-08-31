"""Static fail-closed contracts for the c600 K8 probe runner."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_sa5_c600_k8_observability.py"
PROTOCOL = HERE / "sa5_c600_k8_observability_protocol.py"


def test_runner_is_record_only_and_uses_exact_k8_flag():
    source = RUNNER.read_text(encoding="utf-8")
    for required in (
        "include_k8_input=True",
        '"policy_actions_replaced": False',
        '"training_authorized": False',
        '"sa6_blocked_by_this_probe": False',
    ):
        assert required in source
    for forbidden in ("train_rnn_car_wdclip.py", "systemctl start", "sa6-"):
        assert forbidden not in source


def test_protocol_freezes_nonblocking_reversal_interpretation():
    source = PROTOCOL.read_text(encoding="utf-8")
    assert '"reversal_failure_blocks_sa6": False' in source
    assert '"probe_alone_authorizes_sa6": False' in source
