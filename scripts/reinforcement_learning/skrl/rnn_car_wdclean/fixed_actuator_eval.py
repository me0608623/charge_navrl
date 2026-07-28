"""Shared fixed-delay actuator profiles for deployment gate runners."""

from __future__ import annotations

from pathlib import Path


CONTROL_DT_S = 0.2
VELOCITY_SCALE = (0.9, 1.1)
MOTOR_LAG_ALPHA = 0.3
VALID_DELAY_STEPS = (0, 1, 2)
ACTUATOR_PROFILES = ("bridge", "sa1_delay_only")

_PROFILE_SPECS = {
    # Historical W1 bridge bundle. Keep as the default so old commands and
    # reports remain byte-for-byte equivalent unless a profile is requested.
    "bridge": {
        "velocity_scale": VELOCITY_SCALE,
        "motor_lag_alpha": MOTOR_LAG_ALPHA,
        "randomized_components": ["velocity_scale_per_episode"],
    },
    # New SA1 lineage: only the measured command dead time is active.
    "sa1_delay_only": {
        "velocity_scale": (1.0, 1.0),
        "motor_lag_alpha": 1.0,
        "randomized_components": [],
    },
}


def _profile_spec(profile: str) -> dict:
    try:
        return _PROFILE_SPECS[profile]
    except KeyError as exc:
        raise ValueError(
            f"actuator profile must be one of {ACTUATOR_PROFILES}, "
            f"got {profile!r}"
        ) from exc


def fixed_actuator_cli_args(
    delay_steps: int | None,
    profile: str = "bridge",
) -> list[str]:
    """Return play CLI arguments for one fixed-delay actuator evaluation."""
    spec = _profile_spec(profile)
    if delay_steps is None:
        return []
    if isinstance(delay_steps, bool) or delay_steps not in VALID_DELAY_STEPS:
        raise ValueError(
            f"delay_steps must be one of {VALID_DELAY_STEPS}, got {delay_steps!r}"
        )
    return [
        "--enable_actuator_dr",
        "--actuator_delay_range",
        str(delay_steps),
        str(delay_steps),
        "--actuator_velocity_scale",
        str(spec["velocity_scale"][0]),
        str(spec["velocity_scale"][1]),
        "--actuator_motor_lag",
        str(spec["motor_lag_alpha"]),
    ]


def fixed_actuator_metadata(
    delay_steps: int | None,
    profile: str = "bridge",
) -> dict:
    """Return a self-describing JSON record for a gate report."""
    spec = _profile_spec(profile)
    if delay_steps is None:
        return {
            "enabled": False,
            "profile": profile,
            "delay_steps": None,
            "delay_ms": None,
            "velocity_scale_range": None,
            "motor_lag_alpha": None,
        }
    # Reuse validation instead of maintaining a second admissibility rule.
    fixed_actuator_cli_args(delay_steps, profile)
    return {
        "enabled": True,
        "profile": profile,
        "delay_steps": delay_steps,
        "delay_ms": round(delay_steps * CONTROL_DT_S * 1000),
        "velocity_scale_range": list(spec["velocity_scale"]),
        "motor_lag_alpha": spec["motor_lag_alpha"],
        "fixed_component": "delay_only",
        "randomized_components": list(spec["randomized_components"]),
        "pipeline": "decode->delay->scale->lag",
        "history": "issued_command_queue",
    }


def verify_fixed_actuator_runtime(
    log_path: Path,
    delay_steps: int | None,
    profile: str = "bridge",
) -> None:
    """Fail closed when a requested actuator bundle did not reach the simulator."""
    spec = _profile_spec(profile)
    if delay_steps is None:
        return
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    required = (
        "[SIM2REAL] Actuator DR:",
        f"delay=({delay_steps}, {delay_steps}) steps",
        f"vel_scale={spec['velocity_scale']}",
        f"motor_lag alpha={spec['motor_lag_alpha']}",
        "pipeline=decode->delay->scale->lag",
        "history=issued_command_queue",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(
            f"actuator gate wiring incomplete in {log_path}: missing {missing}"
        )
