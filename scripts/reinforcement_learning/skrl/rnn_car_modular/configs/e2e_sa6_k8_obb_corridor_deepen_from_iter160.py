"""Corridor-priority deepening from iter160 (user decision 2026-07-25).

The iterate-cycle probe proved the corridor<->narrow trade-off is Pareto-
fundamental at the encoder: pushing corridor below ~13-15% erodes narrow past
projection recoverability. The user chose the corridor-priority deliverable
(deploy the iter160 corridor policy; defer narrow to SA6-8 / a separate
mechanism). Narrow is therefore already sacrificed, so pure corridor
optimization has no downside. This continues corridor training from iter160
(checkpoint_12800, corridor CR 6.1%, Gate2 CR 7.1%) for +150 iterations to push
corridor CR lower and more robust. Frozen SA6 recipe otherwise (68/10/12/10
mix, reward, lr, horizon, network all inherited); the 12% narrow replay is now
inert but kept for stability/consistency.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_long_cont300_resume_iter60 import (
    CONFIG as _RESUME,
)


_ITER160 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_long_cont300_resume_iter60_s42/checkpoint_12800.pt"
)


CONFIG = replace(
    _RESUME,
    name="e2e_sa6_k8_obb_corridor_deepen_from_iter160",
    description=(
        "Corridor-priority deepening from iter160 (corridor CR 6.1%, Gate2 CR "
        "7.1%). +150 iterations with the frozen SA6 recipe to drive corridor CR "
        "lower and more robust; narrow is intentionally sacrificed per the "
        "user's corridor-priority decision."
    ),
    checkpoint=_ITER160,
    no_resume_optimizer=False,
    timesteps=150 * _RESUME.rollout_length,
    save_interval=10,
    tags=_RESUME.tags + ("corridor_priority_deliverable", "deepen_from_iter160"),
    notes=(
        "Warm-start iter160 checkpoint_12800 + Adam. Continue the fixed recipe "
        "for 150 iterations; save every 10. Goal: robust corridor-priority SA6 "
        "policy (corridor CR well below 10%, Gate2 held). Narrow gate is "
        "deferred and expected to stay failed. Gate corridor + Gate2 per late "
        "checkpoint; pick the lowest-corridor checkpoint that keeps Gate2."
    ),
)
