"""Fresh SA1 sim-to-real mainline: command delay plus measured VLP-16 noise.

This is a robustness bundle, not a causal ablation. Its paired control is
``e2e_sa1_k8_obb_actuator_delay_only``. Both arms use the same fresh SA1
architecture, 78/12/10 native/narrow/corridor scene mix and actuator delay;
this arm additionally enables the fixed measured VLP-16 ``full`` preset
(sigma + per-ring bias + dropout).

Physics, external-force and observation-delay DR stay disabled. Calibrated
per-channel actuator lag can be added later as a separate locked factor after
vehicle step-response system identification.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa1_k8_obb_actuator_delay_only import (
    CONFIG as _DELAY_ONLY,
)


VLP16_NOISE_MODE = "full"


CONFIG = replace(
    _DELAY_ONLY,
    name="e2e_sa1_k8_obb_sim2real_v1",
    description=(
        "Fresh SA1 K8 OBB sim-to-real scene-mix training with U{0,1,2} "
        "decoded-command delay and the fixed measured VLP-16 full noise preset."
    ),
    lidar_no_noise=False,
    vlp16_noise_mode=VLP16_NOISE_MODE,
    # Keep event-level physics/force DR off. The measured observation noise
    # path and action-term actuator delay are applied independently.
    no_domain_randomization=True,
    tags=_DELAY_ONLY.tags + ("vlp16_measured_full", "combined_robustness_bundle"),
    notes=(
        "Primary new SA1 sim-to-real recipe. It trains on 78% native SA1, "
        "12% narrow-passage and 10% deployment-corridor resets. Relative to "
        "the delay-only scene-mix arm, the only behavioural change is "
        "measured VLP-16 full noise "
        "(8.672 mm fixed sigma, measured per-ring bias, 19.4859% holes and "
        "0.2515% mixed-pixel distractors). It deliberately excludes "
        "uncalibrated velocity scaling, motor lag, observation delay, physics "
        "DR and external pushes. Do not claim a causal delay or LiDAR effect "
        "from this combined arm alone; compare against the clean SA1 and "
        "delay-only controls. Do not launch a long run until the train/export/"
        "vehicle action contract is repaired and its parity gate passes."
    ),
)
