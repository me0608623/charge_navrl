"""Five-iteration continuation of the corrected actuator bridge.

This is a bounded continuation, not a new experiment arm. It resumes the
corrected c10 checkpoint with model and optimizer state and changes no
behavioural setting. The only differences from the bridge config are
checkpoint location, run metadata, budget, and save cadence.
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_deploy_bridge_actuator_delay_from_w1c10 import (
    CONFIG as _BRIDGE,
)


BRIDGE_C10_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "bridge_actdelay_w1c10_s42_r2_cmdqueue/checkpoint_1280.pt"
)
CONTINUATION_TIMESTEPS = 640
CONTINUATION_SAVE_INTERVAL = 1


CONFIG = replace(
    _BRIDGE,
    name="e2e_deploy_bridge_actuator_delay_continue_c10",
    description=(
        "Bounded five-iteration continuation from the corrected actuator "
        "bridge c10. Model and Adam state resume; actuator bundle, scene mix, "
        "reward, network, PPO, LR, observation layout, and seed are unchanged."
    ),
    checkpoint=BRIDGE_C10_CHECKPOINT,
    timesteps=CONTINUATION_TIMESTEPS,
    save_interval=CONTINUATION_SAVE_INTERVAL,
    tags=_BRIDGE.tags + ("c10_continuation",),
    notes=(
        "The corrected c10 retained true-clean Gate2 but reached only "
        "SR=88.5%, CR=11.3% at fixed 200 ms, short of the 90%/9% deployment "
        "gate. Continue for exactly five iterations with model + Adam state. "
        "Every iteration is saved so no additional training is needed to "
        "inspect c11-c15. No behavioural config field differs from c10."
    ),
)
