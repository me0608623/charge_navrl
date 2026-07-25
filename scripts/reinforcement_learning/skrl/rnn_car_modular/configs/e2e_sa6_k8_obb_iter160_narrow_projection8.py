"""Frozen-encoder full-actor c20 narrow projection from corridor-improved iter160.

Staged-decoupling final step (2026-07-24): the long continuation drove the
4m x 10m 4S+2D corridor CR down to 6.1% (well below the <10% target) while
Gate2 held (CR 7.1%). The only remaining gap is Gate5 narrow, eroded to a
15.8% crossing by the corridor-heavy PPO. This run restores the narrow skill
by projecting the actor onto the frozen c20 narrow teacher WITHOUT any PPO or
value step, so the corridor ability that lives in the LiDAR CNN encoder is
preserved. Same eight-step full-actor recipe as the c50 projection8 diagnostic
(which raised Gate5 crossing 78.8 -> 98.4%), only the warm-start checkpoint
changes to the iter160 corridor-improved model.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_fullactor_projection8 import (
    CONFIG as _PROJ8,
)


_ITER160 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_cont300_resume_iter60_s42/checkpoint_12800.pt"
)


CONFIG = replace(
    _PROJ8,
    name="e2e_sa6_k8_obb_iter160_narrow_projection8",
    description=(
        "Eight-step full-actor c20 narrow projection from the iter160 "
        "corridor-improved policy (corridor CR 6.1%). No PPO/value step; only "
        "the actor is projected onto the frozen c20 narrow teacher to restore "
        "Gate5 while preserving the encoder-resident corridor ability."
    ),
    checkpoint=_ITER160,
    tags=_PROJ8.tags + ("from_iter160", "corridor6p1_base", "joint_pass_candidate"),
    notes=(
        "Base = sa6_k8_obb_long_cont300_resume_iter60_s42/checkpoint_12800 "
        "(abs iter160, corridor 4S+2D CR 6.1%, Gate2 CR 7.1%, Gate5 crossing "
        "15.8%). ppo_epochs=0, one rollout, eight full-batch SGD steps through "
        "the LiDAR CNN encoder and policy head, forward KL + argmax margin "
        "against frozen c20 on narrow frames. After projection, rerun the joint "
        "gate: Gate5 must recover to >=95% crossing / CR <=5% while the 4S+2D "
        "corridor stays <10% and Gate2 holds -> first JOINT PASS candidate."
    ),
)
