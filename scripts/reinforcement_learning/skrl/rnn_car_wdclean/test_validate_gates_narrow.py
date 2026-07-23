import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = Path(__file__).with_name("validate_gates.py")
_SPEC = importlib.util.spec_from_file_location("validate_gates", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gates = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gates
_SPEC.loader.exec_module(gates)


def _write_log(path: Path, *, yaw_limit: float = 10.0) -> None:
    path.write_text(
        "\n".join(
            [
                "總回合數: 100",
                "成功率: 92/100 (92.0%)",
                "碰撞率 (總計): 4/100 (4.0%)",
                "超時率: 4/100 (4.0%)",
                (
                    "[NARROW-GAP-METRICS] episodes=100 crossed=98 crossing_rate=0.980000 "
                    "yaw_frames=500 yaw_abs_p50_deg=2.0 yaw_abs_p90_deg=7.0 "
                    f"yaw_abs_p95_deg=9.0 yaw_within_{yaw_limit:.2f}deg=0.960000"
                ),
            ]
        ),
        encoding="utf-8",
    )


def test_parse_and_check_deployment_narrow_gap(tmp_path: Path):
    log = tmp_path / "deploy.log"
    _write_log(log)

    metrics = gates.parse_narrow_gap(str(log))
    checks = gates.narrow_gap_checks(metrics)

    assert metrics["yaw_within_10.00deg"] == 0.96
    assert checks
    assert all(ok for ok, _ in checks.values())


def test_old_yaw_metric_cannot_satisfy_new_deployment_gate(tmp_path: Path):
    log = tmp_path / "legacy.log"
    _write_log(log, yaw_limit=2.52)

    metrics = gates.parse_narrow_gap(str(log))

    assert gates.narrow_gap_checks(metrics) == {}


def test_stress_width_is_not_the_deployment_width():
    assert gates.NARROW_DEPLOY_WIDTH_M == 1.2
    assert gates.NARROW_STRESS_WIDTH_M == 1.0
