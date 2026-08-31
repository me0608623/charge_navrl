import pytest

import sa4_it100_noise_distribution_sensitivity as protocol


def test_override_noise_mode_replaces_value_only():
    args = ["--stage", "4", "--vlp16_noise_mode", "full", "--seed", "818"]
    out = protocol.override_noise_mode(args, "ideal")
    assert out == ["--stage", "4", "--vlp16_noise_mode", "ideal", "--seed", "818"]


def test_override_noise_mode_does_not_mutate_input():
    args = ["--vlp16_noise_mode", "full"]
    protocol.override_noise_mode(args, "sigma")
    assert args == ["--vlp16_noise_mode", "full"]


def test_override_noise_mode_missing_flag_raises():
    with pytest.raises(ValueError):
        protocol.override_noise_mode(["--seed", "818"], "ideal")


_SYNTHETIC_LOG = (
    "[SIM2REAL][VLP16-ablation] mode={mode} sigma=ON bias=ON dropout=ON\n"
    "外牆 ±8.00m\n"
    "[PLAY] curriculum=warp_drive_e2e_final20_v1 stage=4 name=SA4_spatial_plan "
    "目標=2 障礙=8S+2D 牆壁=1~2\n"
)
_VALUES = {"room_half_extent_m": 8.0, "stage": 4}


def test_verify_common_runtime_for_arm_accepts_matching_marker(tmp_path):
    log_path = tmp_path / "cell.log"
    log_path.write_text(_SYNTHETIC_LOG.format(mode="ideal"), encoding="utf-8")
    result = protocol.verify_common_runtime_for_arm(log_path, _VALUES, "ideal")
    assert result["stage_built"] == 4


def test_verify_common_runtime_for_arm_rejects_wrong_arm_marker(tmp_path):
    log_path = tmp_path / "cell.log"
    # log says full, but the cell was supposed to be the ideal arm
    log_path.write_text(_SYNTHETIC_LOG.format(mode="full"), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ideal.*marker absent"):
        protocol.verify_common_runtime_for_arm(log_path, _VALUES, "ideal")


def test_protocol_constants_match_freeze_spec():
    assert protocol.SCENARIOS == (
        "corridor_lateral",
        "corridor_longitudinal",
        "nav_native",
    )
    assert protocol.SEEDS == (818, 515, 616)
    assert protocol.ARMS == ("ideal", "full", "sigma", "bias", "dropout")
    assert protocol.STEPS_BY_SCENARIO == {
        "corridor_lateral": 2500,
        "corridor_longitudinal": 4000,
        "nav_native": 1200,
    }
