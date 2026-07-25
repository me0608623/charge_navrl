"""Projection8 recoverability test from cycle-1 iter20 (short-drift, low crossing).

Calibration decision (2026-07-25, overnight): cycle-1 (20 corridor iters from
projected-c50) dropped the 4S+2D corridor CR 14.9 -> 12.4% but cratered Gate5
narrow crossing 98.4 -> 0.5%. The open question is whether a SHORT-drift base
(only 20 iters off c50) stays projection-recoverable despite the low behavioral
crossing -- unlike iter160 (100+ iters of drift, projection failed 15.8 ->
9.8%). This run projects the iter20 checkpoint with the same eight-step
full-actor c20 recipe to answer that definitively. If narrow recovers to >=95%
while corridor holds, short cycles are viable; if not, the iterate-cycle
approach is infeasible at this operating point.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c50_narrow_fullactor_projection8 import (
    CONFIG as _PROJ8,
)


_CYCLE1_ITER20 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_cycle1_from_projc50_s42/checkpoint_2560.pt"
)


CONFIG = replace(
    _PROJ8,
    name="e2e_sa6_k8_obb_cycle1_iter20_projection8",
    description=(
        "Eight-step full-actor c20 narrow projection from cycle-1 iter20 "
        "(corridor CR 12.4%, narrow crossing 0.5%). Tests whether a short-drift "
        "base stays projection-recoverable despite low behavioral crossing."
    ),
    checkpoint=_CYCLE1_ITER20,
    tags=_PROJ8.tags + ("from_cycle1_iter20", "recoverability_test"),
    notes=(
        "Base = sa6_k8_obb_cycle1_from_projc50_s42/checkpoint_2560 (20 corridor "
        "iters off projected-c50; corridor 12.4%, narrow 0.5%). Same recipe as "
        "the iter160 projection (ppo_epochs=0, eight full-batch SGD steps, "
        "forward KL + argmax margin vs frozen c20, narrow frames only). Decisive "
        "for iterate-cycle feasibility: recover >=95% -> viable; fail -> the "
        "corridor<->narrow trade-off is Pareto-fundamental at the encoder."
    ),
)
