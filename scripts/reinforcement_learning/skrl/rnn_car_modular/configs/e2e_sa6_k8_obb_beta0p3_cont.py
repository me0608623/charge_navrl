"""Ten-update SA6 continuation from the calibrated beta=0.30 checkpoint."""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_k8_obb import CONFIG as _SA6


_CALIBRATED_C1 = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_c12_beta0p3_formal1update_ne1024_s42/checkpoint_128.pt"
)


CONFIG = replace(
    _SA6,
    name="e2e_sa6_k8_obb_beta0p3_cont",
    description=(
        "Ten uninterrupted SA6 updates from the calibrated beta=0.30 c1 "
        "checkpoint with fixed native/narrow/corridor replay."
    ),
    checkpoint=_CALIBRATED_C1,
    no_resume_optimizer=False,
    timesteps=1280,
    save_interval=2,
    tags=_SA6.tags + ("beta0p3_calibrated", "ten_update_continuation"),
    notes=(
        "Resume the accepted c1 Adam state. Save every two updates. Evaluate "
        "each checkpoint in fail-fast order: fixed 1.2m Gate5, three-seed "
        "Gate2, then the exact 4m x 10m 4S+2D corridor. Stop at the first "
        "joint improvement or any narrow-passage regression."
    ),
)
