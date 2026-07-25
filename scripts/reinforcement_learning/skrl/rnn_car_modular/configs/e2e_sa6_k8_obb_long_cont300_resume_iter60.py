"""Resume the long SA6 continuation from iter 60 to push corridor CR below 10%.

Supervisor-driven resume (2026-07-24): long_cont300 stopped at iter 60. The
frozen recipe is unchanged (68/10/12/10 mix, c20 beta=0.30 narrow forward-KL,
reward/lr/horizon/network all inherited). Only the warm-start checkpoint and
the training budget change, so this continues the exact same run from its
iter-60 Adam state for +100 iterations (target ~iter 160). Narrow erosion is
expected and is recovered afterwards by the frozen c20 actor projection.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_long_cont300 import CONFIG as _LONG


_ITER60 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_cont300_c128_s42/checkpoint_7680.pt"
)


CONFIG = replace(
    _LONG,
    name="e2e_sa6_k8_obb_long_cont300_resume_iter60",
    description=(
        "Resume long_cont300 from iter 60 (checkpoint_7680) for +100 "
        "iterations to test whether corridor CR keeps descending below 10%."
    ),
    checkpoint=_ITER60,
    no_resume_optimizer=False,
    timesteps=100 * _LONG.rollout_length,
    save_interval=10,
    tags=_LONG.tags + ("resume_from_iter60", "supervisor_driven_continuation"),
    notes=(
        "Continue the fixed recipe from the iter-60 model and Adam state. "
        "Do not change reward, learning rate, horizon, network, or replay "
        "mix. Save every 10 iterations. Gate 4S+2D corridor per checkpoint; "
        "once corridor <= 10%, re-run the frozen-encoder c20 actor projection "
        "to restore Gate5 narrow and produce the merged JOINT candidate."
    ),
)
