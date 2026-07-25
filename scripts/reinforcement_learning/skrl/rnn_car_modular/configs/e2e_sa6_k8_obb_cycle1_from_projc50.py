"""Cycle-1 short corridor continuation from the projection-recovered c50.

Iterate project<->train (user decision 2026-07-24): the single +100-iter jump
drove the encoder past actor-projection recoverability (iter160 narrow crossing
stuck at 9.8% post-projection). Instead, start from the projection-recovered
c50 (narrow crossing 98.4%, 4S+2D corridor ~14.9%) and take a SHORT corridor
continuation so the encoder drifts only a little, keeping narrow projection-
recoverable. Save every 5 iters; the projection step then picks the checkpoint
whose corridor is closest to <10% while narrow is still recoverable, and a
joint gate follows. Frozen SA6 recipe otherwise (68/10/12/10 mix, reward, lr,
horizon, network all inherited from e2e_sa6_k8_obb).
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


_PROJECTED_C50 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_c50_narrow_fullactor_projection8_1u_s42/checkpoint_128.pt"
)


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_cycle1_from_projc50",
    description=(
        "Cycle-1 short corridor continuation from the projection-recovered c50 "
        "(narrow 98.4%, corridor ~14.9%). Twenty iterations, checkpoint every "
        "five, to keep the encoder drift small enough that narrow stays "
        "projection-recoverable."
    ),
    checkpoint=_PROJECTED_C50,
    no_resume_optimizer=False,
    timesteps=20 * _SA6.rollout_length,
    save_interval=5,
    tags=_SA6.tags
    + ("iterate_project_train_cycle1", "from_projected_c50", "short_corridor_nudge"),
    notes=(
        "Base = sa6_c50_narrow_fullactor_projection8_1u_s42/checkpoint_128 "
        "(projection-recovered narrow 98.4%, 4S+2D corridor ~14.9%). Resume the "
        "Adam state and run only twenty corridor iterations with the frozen SA6 "
        "recipe. Checkpoints at 5/10/15/20 iters. Next: run projection8 on the "
        "checkpoint whose corridor is nearest <10% with narrow still "
        "recoverable, then joint-gate for a JOINT PASS candidate."
    ),
)
