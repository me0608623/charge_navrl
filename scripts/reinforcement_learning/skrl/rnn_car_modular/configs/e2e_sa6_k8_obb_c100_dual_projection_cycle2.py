"""Second on-policy c20/c100 projection cycle after partial Gate5 recovery."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb_c100_dual_projection import (
    CONFIG as _CYCLE_ONE,
)


_CYCLE_ONE_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_c100_dual_projection_lambda0p75_1u_s42/checkpoint_128.pt"
)


CONFIG = replace(
    _CYCLE_ONE,
    name="e2e_sa6_k8_obb_c100_dual_projection_cycle2",
    description=(
        "Second on-policy projection cycle from the partially recovered c100 "
        "merge, retaining the original c20 and c100 teachers."
    ),
    checkpoint=_CYCLE_ONE_CHECKPOINT,
    teacher_retention_post_kl_epochs=4,
    tags=_CYCLE_ONE.tags
    + (
        "projection_cycle_2",
        "on_policy_dataset_refresh",
        "projection_steps_4",
    ),
    notes=(
        "Cycle one recovered Gate5 to SR 93.5%, CR 2.2%, and crossing 93.69%, "
        "but missed the 95% crossing gate. Recollect one rollout from that "
        "partially recovered policy so the projection sees states closer to "
        "and beyond the throat. Keep c20 as narrow teacher, the original "
        "unprojected c100 as non-narrow teacher, lambda 0.75, and no PPO. Use "
        "four updates to limit reverse forgetting."
    ),
)
