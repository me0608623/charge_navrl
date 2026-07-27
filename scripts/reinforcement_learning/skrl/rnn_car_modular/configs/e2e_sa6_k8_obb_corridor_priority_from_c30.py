"""Corridor-priority long continuation from the mixed probe's c30 checkpoint.

User decision (2026-07-25): the deployment corridor is the priority. The 30-iter
mixed probe moved the 4S+2D mixed-motion corridor CR from 48.85% to 30.05% while
narrow crossing collapsed (95.4% -> 13.1%). Narrow is therefore deliberately
sacrificed on this branch and will be handled separately (frozen narrow expert /
multi-expert routing), so this run optimizes corridor only.

Continues from c30 (the best corridor point) for 100 iterations with the same
frozen recipe and the mixed corridor motion family. Nothing else changes: the
68/10/12/10 replay mix, K8+future reward, network, LR and horizon are inherited.
Save every ten iterations so the corridor trend can be gated periodically; the
narrow gate is intentionally not part of this run's acceptance.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_mixed_corridor_probe_from_c19200 import (
    CONFIG as _PROBE,
)


_C30 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_mixed_corridor_probe_c19200_s42/checkpoint_3840.pt"
)


CONFIG = replace(
    _PROBE,
    name="e2e_sa6_k8_obb_corridor_priority_from_c30",
    description=(
        "Corridor-priority 100-iteration continuation from the mixed probe c30 "
        "checkpoint (mixed corridor CR 30.05%). Narrow is knowingly sacrificed; "
        "acceptance is corridor CR plus Gate2 only."
    ),
    checkpoint=_C30,
    no_resume_optimizer=False,
    timesteps=100 * _PROBE.rollout_length,
    save_interval=10,
    tags=_PROBE.tags + ("corridor_priority", "c30_warm_start", "narrow_sacrificed"),
    notes=(
        "Warm-start c30 model and Adam. Keep the mixed corridor motion family "
        "and every other frozen setting. Goal: drive the 4S+2D mixed corridor "
        "CR below 10% while Gate2 CR stays at or under 9%. Watch the lateral "
        "corridor CR as well: it regressed from 5.06% to 8.05% during the probe "
        "and only has about two points of margin left. Narrow (Gate5) is not an "
        "acceptance criterion on this branch."
    ),
)
