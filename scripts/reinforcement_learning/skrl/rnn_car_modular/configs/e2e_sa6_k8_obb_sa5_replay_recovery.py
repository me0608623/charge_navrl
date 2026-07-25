"""Two-update SA6 test of fixed SA5 general-scene replay."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


_C10 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_beta0p3_joint_replay_cont10_s42/checkpoint_1280.pt"
)


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_sa5_replay_recovery",
    description=(
        "Two uninterrupted production-size SA6 updates from c10 with 10% "
        "fixed SA5-general replay added to the accepted narrow/corridor mix."
    ),
    checkpoint=_C10,
    no_resume_optimizer=False,
    timesteps=256,
    save_interval=1,
    tags=_SA6.tags + (
        "c10_warm_start",
        "sa5_general_replay_recovery",
        "two_update_probe",
    ),
    notes=(
        "Resume the c10 Adam state. Scene mix is 68% native SA6, 10% SA5 "
        "general, 12% narrow, and 10% deployment corridor. Evaluate fixed "
        "three-seed SA5 Gate2 on checkpoint_128 and checkpoint_256 first. "
        "Run Gate5 and the 4S+2D corridor only for a Gate2-improving candidate."
    ),
)
