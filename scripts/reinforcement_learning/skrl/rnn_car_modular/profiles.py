"""Trainer profile resolver -- metadata first, behavior later.

Phase 0/1: profiles are resolved from CLI args and recorded in WandB / console,
but do NOT change training behavior. Only wd_sparse actually executes;
other reward profiles raise NotImplementedError if used beyond metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# Valid profile values (for validation)
VALID_REWARD_PROFILES = ("wd_sparse", "clean_progress", "navrl_dense_v8", "hybrid_progress", "ttc_risk")
VALID_ALGORITHM_PROFILES = ("a2c_wd", "ppo_clip")
VALID_AUX_PROFILES = ("wd_7d_geometry", "none", "future_collision_risk")


@dataclass(frozen=True)
class TrainerProfiles:
    reward_profile: str
    scene_profile: str
    algorithm_profile: str
    aux_profile: str
    encoder_profile: str


def resolve_profiles(args_cli: Any) -> TrainerProfiles:
    """Derive TrainerProfiles from parsed CLI args.

    Uses explicit --reward_profile etc. if provided; otherwise infers
    from existing flags to guarantee backward-compatible defaults.
    """
    # reward_profile
    reward_profile = getattr(args_cli, "reward_profile", "wd_sparse")

    # scene_profile
    explicit_scene = getattr(args_cli, "scene_profile", None)
    if explicit_scene is not None:
        scene_profile = explicit_scene
    else:
        scene_profile = getattr(args_cli, "curriculum_version", "warp_drive_single_agent_v1")

    # algorithm_profile
    explicit_algo = getattr(args_cli, "algorithm_profile", None)
    if explicit_algo is not None:
        algorithm_profile = explicit_algo
    else:
        use_a2c = getattr(args_cli, "use_a2c", True)
        algorithm_profile = "a2c_wd" if use_a2c else "ppo_clip"

    # aux_profile
    explicit_aux = getattr(args_cli, "aux_profile", None)
    if explicit_aux is not None:
        aux_profile = explicit_aux
    else:
        disable_aux = getattr(args_cli, "disable_aux_training", False)
        aux_profile = "none" if disable_aux else "wd_7d_geometry"

    # encoder_profile
    explicit_enc = getattr(args_cli, "encoder_profile", None)
    if explicit_enc is not None:
        encoder_profile = explicit_enc
    else:
        encoder_profile = getattr(args_cli, "charge_encoder_mode", "extractor_rnn")

    return TrainerProfiles(
        reward_profile=reward_profile,
        scene_profile=scene_profile,
        algorithm_profile=algorithm_profile,
        aux_profile=aux_profile,
        encoder_profile=encoder_profile,
    )


def validate_profiles(p: TrainerProfiles, args_cli: Any | None = None) -> None:
    """Validate profile consistency. Raises ValueError on conflicts."""
    if p.reward_profile not in VALID_REWARD_PROFILES:
        raise ValueError(
            f"Unknown reward_profile={p.reward_profile!r}. "
            f"Valid: {VALID_REWARD_PROFILES}"
        )

    if p.algorithm_profile not in VALID_ALGORITHM_PROFILES:
        raise ValueError(
            f"Unknown algorithm_profile={p.algorithm_profile!r}. "
            f"Valid: {VALID_ALGORITHM_PROFILES}"
        )

    if p.aux_profile not in VALID_AUX_PROFILES:
        raise ValueError(
            f"Unknown aux_profile={p.aux_profile!r}. "
            f"Valid: {VALID_AUX_PROFILES}"
        )

    # Cross-check: algorithm_profile vs --use_a2c
    if args_cli is not None:
        use_a2c = getattr(args_cli, "use_a2c", True)
        expected_algo = "a2c_wd" if use_a2c else "ppo_clip"
        if p.algorithm_profile != expected_algo:
            raise ValueError(
                f"algorithm_profile={p.algorithm_profile!r} conflicts with "
                f"--use_a2c={use_a2c} (expected {expected_algo!r}). "
                f"Either remove --algorithm_profile or match it with --use_a2c/--use_ppo."
            )

        # Cross-check: aux_profile vs --disable_aux_training
        disable_aux = getattr(args_cli, "disable_aux_training", False)
        if p.aux_profile == "none" and not disable_aux:
            raise ValueError(
                "aux_profile='none' but --disable_aux_training is not set. "
                "Either pass --disable_aux_training or use a different aux_profile."
            )
        if p.aux_profile != "none" and disable_aux:
            raise ValueError(
                f"aux_profile={p.aux_profile!r} but --disable_aux_training is set. "
                f"Either remove --disable_aux_training or set --aux_profile none."
            )

    # Only wd_sparse and navrl_dense_v8 are implemented
    implemented_rewards = ("wd_sparse", "clean_progress", "navrl_dense_v8")
    if p.reward_profile not in implemented_rewards:
        raise NotImplementedError(
            f"reward_profile={p.reward_profile!r} is registered but not yet implemented. "
            f"Available: {implemented_rewards}."
        )

    if p.aux_profile == "future_collision_risk":
        raise NotImplementedError(
            f"aux_profile='future_collision_risk' is registered but not yet implemented."
        )


def profiles_to_dict(p: TrainerProfiles) -> dict[str, str]:
    """Convert profiles to a flat dict for WandB config / run_metadata."""
    return asdict(p)
