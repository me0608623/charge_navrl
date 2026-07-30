"""CPU-only contract tests for the SA1 Nav20 runner."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sa1_nav20_suite as nav20  # noqa: E402


def _value_after(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


@pytest.mark.parametrize(
    ("mode", "static_obstacles", "blocking"),
    [
        ("clean", "0", True),
        ("native", "2", False),
    ],
)
def test_scene_args_freeze_nav20_contract(
    mode,
    static_obstacles,
    blocking,
):
    args = nav20.build_scene_args(
        mode,
        seed=515,
        jitter_path=Path("/tmp/jitter.json"),
        actuator_args=["--enable_actuator_dr"],
    )
    assert _value_after(args, "--stage") == "1"
    assert _value_after(args, "--arena_size") == "20.0"
    assert _value_after(args, "--num_static_obs") == static_obstacles
    assert _value_after(args, "--num_dynamic_obs") == "0"
    assert _value_after(args, "--num_walls") == "0"
    assert _value_after(args, "--goal_distance_min") == "2.0"
    assert _value_after(args, "--goal_distance_max") == "9.0"
    assert _value_after(args, "--episode_length_s") == "60.0"
    assert _value_after(args, "--obs_near_goal_count") == "0"
    assert _value_after(args, "--obstacle_behavior") == "static"
    assert _value_after(args, "--seed") == "515"
    assert "--jitter_eval" in args
    assert "--enable_actuator_dr" in args
    assert nav20.BLOCKING[mode] is blocking


def _runtime_text(static_obstacles: int) -> str:
    return "\n".join(
        [
            "[SIM2REAL][VLP16-ablation] mode=full sigma=ON",
            "[PLAY] 場景配置 (stage_parameter=True):",
            (
                f"  目標數=1  靜態障礙={static_obstacles}  "
                "動態障礙=0"
            ),
            "  牆壁=0~0  牆長=3.0m",
            (
                "  目標距離=2.0~9.0m  episode=60.0s  "
                "障礙速度=0.80"
            ),
            "[PLAY] Arena 縮放: 20×20m → 20×20m (ratio=1.00)",
            (
                "[PLAY] curriculum=warp_drive_e2e_final20_v1 stage=1 "
                "name=SA1_nav_bootstrap"
            ),
        ]
    )


@pytest.mark.parametrize(
    ("mode", "static_obstacles"),
    [("clean", 0), ("native", 2)],
)
def test_runtime_verification_returns_non_null_scene_contract(
    tmp_path,
    mode,
    static_obstacles,
):
    log = tmp_path / "play.log"
    log.write_text(_runtime_text(static_obstacles), encoding="utf-8")
    contract = nav20.verify_nav20_runtime(log, mode)
    assert contract["mode"] == mode
    assert contract["arena_size_m"] == 20.0
    assert contract["static_obstacles"] == static_obstacles
    assert contract["dynamic_obstacles"] == 0
    assert contract["internal_walls"] == 0
    assert contract["goal_movement"] is False


@pytest.mark.parametrize(
    ("mode", "text"),
    [
        ("clean", _runtime_text(2)),
        ("native", _runtime_text(0)),
        ("clean", _runtime_text(0).replace("20×20m → 20×20m", "20×20m → 15×15m")),
        ("clean", _runtime_text(0).replace("動態障礙=0", "動態障礙=1")),
        ("clean", _runtime_text(0).replace("mode=full", "mode=ideal")),
    ],
)
def test_runtime_verification_fails_closed(tmp_path, mode, text):
    log = tmp_path / "play.log"
    log.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError):
        nav20.verify_nav20_runtime(log, mode)


def test_clean_is_strict_and_native_is_advisory():
    clean = nav20.thresholds_for("clean")
    native = nav20.thresholds_for("native")
    assert clean == {"sr": 0.98, "cr": 0.015, "to": 0.005}
    assert native == {"sr": 0.90, "cr": 0.08, "to": 0.03}
    assert nav20.BLOCKING == {"clean": True, "native": False}


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        nav20.build_scene_args(
            "deployment",
            seed=515,
            jitter_path=Path("/tmp/jitter.json"),
            actuator_args=[],
        )
