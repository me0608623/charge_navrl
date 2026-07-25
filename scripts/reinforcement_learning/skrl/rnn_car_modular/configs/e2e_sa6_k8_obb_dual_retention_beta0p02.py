"""One-update SA6 probe with independent narrow and SA5-general teachers."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


_C128 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_sa5replay10_recovery_c10_2u_s42/checkpoint_128.pt"
)
_SA5_TEACHER = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa5_e2e_k8_obb_corridor_teacher_distill_c12_stepnorm_s42/"
    "checkpoint_128.pt"
)


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_dual_retention_beta0p02",
    description=(
        "One production-size SA6 update from the best SA5-replay c128 "
        "candidate. Keep c20 beta=0.30 on narrow frames and add c12 "
        "beta=0.02 only on SA5-general replay frames."
    ),
    checkpoint=_C128,
    no_resume_optimizer=False,
    timesteps=128,
    save_interval=1,
    previous_stage_teacher_checkpoint=_SA5_TEACHER,
    previous_stage_teacher_retention_weight=0.02,
    tags=_SA6.tags + (
        "c128_warm_start",
        "dual_teacher_retention",
        "sa5_teacher_beta0p02",
        "one_update_probe",
    ),
    notes=(
        "Scene mix and reward are unchanged at 68/10/12/10. The c20 "
        "forward-KL beta=0.30 remains narrow-only. A separate c12 "
        "forward-KL beta=0.02 applies only to the disjoint SA5-general "
        "replay mask. Beta is scale-calibrated from the 64-env smoke: "
        "unweighted c12 KL was about 0.20 per frame, so beta=0.30 would "
        "overwhelm PPO. Evaluate fixed three-seed Gate2 before other gates."
    ),
)
