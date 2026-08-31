"""Frozen protocol constants for the SA4 it100 noise-distribution-sensitivity
experiment (docs/freeze/sa4_it100_noise_distribution_sensitivity_v1.json).

Descriptive only (``inferential_claim: false`` in the freeze spec): cells here
are never given a pass/fail verdict against SA3/SA4 acceptance thresholds.
"""

from __future__ import annotations

from pathlib import Path


CHECKPOINT_NAME = "sa4_r3_it100"
CHECKPOINT_SHA256 = (
    "aeddf32b3e41d1920bfc839651f18e09d7384a8f6ab3a5ac6de3c11628520d7d"
)
SCENARIOS = ("corridor_lateral", "corridor_longitudinal", "nav_native")
STEPS_BY_SCENARIO = {
    "corridor_lateral": 2500,
    "corridor_longitudinal": 4000,
    "nav_native": 1200,
}
SEEDS = (818, 515, 616)
DELAY_STEPS = 1
ELIGIBILITY = "valid_return_only"
ARMS = ("ideal", "full", "sigma", "bias", "dropout")


def override_noise_mode(scene_args: list[str], mode: str) -> list[str]:
    """Return a copy of ``scene_args`` with ``--vlp16_noise_mode`` set to ``mode``.

    ``build_scene_args`` hardcodes ``full``; every arm in this experiment
    except ``treatment`` needs a different value, so the flag is patched in
    place after scene construction rather than re-deriving all the
    scenario-specific flags by hand.
    """
    args = list(scene_args)
    try:
        idx = args.index("--vlp16_noise_mode")
    except ValueError as exc:
        raise ValueError(
            "scene_args does not contain --vlp16_noise_mode"
        ) from exc
    args[idx + 1] = mode
    return args


def verify_common_runtime_for_arm(log_path: Path, values: dict, arm: str) -> dict:
    """Like ``run_sa3_gate_bc.verify_common_runtime``, but the VLP16 banner
    must match the requested ``arm`` instead of being hardcoded to ``full``.

    Every other gate/screen in this codebase only ever runs ``full``, so the
    shared helper bakes that in; this experiment sweeps the noise mode, so a
    parallel arm-aware check is needed rather than editing the shared one.
    Reuses the same regexes as the shared helper to stay consistent with it.
    """
    import run_sa3_gate_bc as shared

    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    marker = f"[SIM2REAL][VLP16-ablation] mode={arm}"
    if marker not in text:
        raise RuntimeError(f"VLP16 {arm!r}-mode marker absent in {log_path}")

    problems = []
    room = shared._ROOM_RE.search(text)
    if room is None:
        raise RuntimeError(f"no room geometry line in {log_path}")
    half = float(room.group(1))
    if abs(half - values["room_half_extent_m"]) > 1e-6:
        problems.append(f"room half extent {half} != {values['room_half_extent_m']}")

    banner = shared._STAGE_BANNER_RE.search(text)
    if banner is None:
        raise RuntimeError(
            f"no stage banner in {log_path}; the built stage cannot be confirmed"
        )
    curriculum, built_stage, stage_name, n_static, n_dynamic = banner.groups()
    if int(built_stage) != int(values["stage"]):
        problems.append(f"built stage {built_stage} != requested {values['stage']}")
    if problems:
        raise RuntimeError(
            f"runtime geometry contract failed for {log_path}: {problems}"
        )
    return {
        "curriculum": curriculum,
        "stage_built": int(built_stage),
        "stage_name": stage_name,
        "native_static_obstacles": int(n_static),
        "native_dynamic_obstacles": int(n_dynamic),
        "room_half_extent_m_built": half,
    }
