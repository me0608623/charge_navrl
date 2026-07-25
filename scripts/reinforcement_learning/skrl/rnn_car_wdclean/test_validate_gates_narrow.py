import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = Path(__file__).with_name("validate_gates.py")
_SPEC = importlib.util.spec_from_file_location("validate_gates", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gates = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = gates
_SPEC.loader.exec_module(gates)


def _write_log(
    path: Path,
    *,
    yaw_limit: float = 10.0,
    yaw_p95: float = 9.0,
    yaw_within: float = 0.96,
    direct_crossing: float = 0.96,
) -> None:
    path.write_text(
        "\n".join(
            [
                "總回合數: 100",
                "成功率: 92/100 (92.0%)",
                "碰撞率 (總計): 4/100 (4.0%)",
                "超時率: 4/100 (4.0%)",
                (
                    "[NARROW-GAP-METRICS] episodes=100 crossed=98 crossing_rate=0.980000 "
                    f"direct_crossed={round(direct_crossing * 100)} "
                    f"direct_crossing_rate={direct_crossing:.6f} "
                    "path_length_ratio_p50=1.04 path_length_ratio_p90=1.18 "
                    "path_length_ratio_p95=1.24 first_cross_time_s_p50=4.8 "
                    "first_cross_time_s_p90=6.0 first_cross_time_s_p95=6.4 "
                    "max_pre_cross_abs_y_m_p50=0.1 "
                    "max_pre_cross_abs_y_m_p90=0.4 "
                    "max_pre_cross_abs_y_m_p95=0.6 "
                    "backtrack_distance_m_p50=0.0 "
                    "backtrack_distance_m_p90=0.1 "
                    "backtrack_distance_m_p95=0.2 "
                    "yaw_frames=500 yaw_abs_p50_deg=2.0 yaw_abs_p90_deg=7.0 "
                    f"yaw_abs_p95_deg={yaw_p95} "
                    f"yaw_within_{yaw_limit:.2f}deg={yaw_within:.6f}"
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


def test_yaw_distribution_does_not_gate_safe_passage(tmp_path: Path):
    log = tmp_path / "high_yaw.log"
    _write_log(log, yaw_p95=15.54, yaw_within=0.89)

    metrics = gates.parse_narrow_gap(str(log))

    assert all(ok for ok, _ in gates.narrow_gap_checks(metrics).values())


def test_missing_yaw_metrics_do_not_block_hard_checks():
    metrics = {
        "sr": 1.0,
        "cr": 0.0,
        "crossing_rate": 1.0,
        "direct_crossing_rate": 1.0,
    }

    assert all(ok for ok, _ in gates.narrow_gap_checks(metrics).values())


def test_indirect_crossing_fails_only_directness_gate(tmp_path: Path):
    log = tmp_path / "indirect.log"
    _write_log(log, direct_crossing=0.80)

    checks = gates.narrow_gap_checks(gates.parse_narrow_gap(str(log)))

    assert checks["到達SR"][0]
    assert checks["碰撞CR"][0]
    assert checks["穿越率"][0]
    assert not checks["直接穿越率"][0]


def test_old_log_without_directness_is_not_accepted():
    metrics = {"sr": 1.0, "cr": 0.0, "crossing_rate": 1.0}

    assert gates.narrow_gap_checks(metrics) == {}


def test_stress_width_is_not_the_deployment_width():
    assert gates.NARROW_DEPLOY_WIDTH_M == 1.2
    assert gates.NARROW_STRESS_WIDTH_M == 1.0
