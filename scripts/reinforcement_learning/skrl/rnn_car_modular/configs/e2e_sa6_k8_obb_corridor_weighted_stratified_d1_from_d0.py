"""D1: weighted env-stratified corridor replay from the D0 probe.

D0 fixed the corridor train/eval distribution mismatch with a balanced 1:1:1
env-level stratification and took lateral 11.76% -> 7.41% and longitudinal
18.35% -> 2.23% (both PASS) while random_2d only moved 38.72% -> 33.79%.

D1 keeps every other setting identical and changes exactly one thing: the
per-family env weights inside the existing 10% corridor replay become
lateral 30% / longitudinal 10% / random_2d 60%, concentrating sampling on the
remaining bottleneck. lateral is deliberately held at 30% (not lowered further)
because it only has about 2.6 points of margin to the 10% gate; longitudinal
yields the most because it sits at 2.23% with roughly 7.8 points of margin.

Acceptance is decided by the four-mode three-seed suite plus Gate2, with a
five-iteration sentinel on seed 616 as early warning only.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_corridor_env_stratified_probe_from_c100 import (
    CONFIG as _D0,
)


_D0_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_corridor_env_stratified_probe_c100_s42/checkpoint_3840.pt"
)


CONFIG = replace(
    _D0,
    name="e2e_sa6_k8_obb_corridor_weighted_stratified_d1_from_d0",
    description=(
        "Thirty-iteration D1 probe from the D0 checkpoint. Only the corridor "
        "motion-family env weights change, to lateral 30% / longitudinal 10% / "
        "random_2d 60%; the corridor replay fraction stays at 10%."
    ),
    checkpoint=_D0_CHECKPOINT,
    no_resume_optimizer=False,
    timesteps=30 * _D0.rollout_length,
    save_interval=5,
    long_corridor_dynamic_motion_mode="env_stratified",
    long_corridor_dynamic_motion_weights=(0.30, 0.10, 0.60),
    tags=_D0.tags + ("weighted_stratified_corridor_motion", "d1_probe"),
    notes=(
        "Resume the D0 model and Adam state. Reward, network, PPO settings, "
        "horizon, LR and the 68/10/12/10 replay mix are unchanged, as is the "
        "10% corridor share; only the family weights inside that share move to "
        "30/10/60. Save every five iterations. Sentinel every five iterations "
        "on seed 616 for lateral, random_2d and mixed; full four-mode three-seed "
        "suite every ten iterations; Gate2 at iterations 10, 20 and 30. Stop "
        "rules: three-seed lateral CR above 10% rejects the checkpoint, and "
        "three-seed mixed CR above 21.15% (more than two points worse than the "
        "D0 aggregate of 19.15%) pauses for analysis. A sentinel mixed delta of "
        "at least +1.0 point versus the D0 seed-616 baseline of 18.79% triggers "
        "the full three-seed suite early but never kills the run by itself."
    ),
)
