"""Prepared SA2 sim-to-real curriculum v2 continuation."""

from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    make_sim2real_curriculum_config,
    pending_parent_checkpoint,
)


PENDING_PARENT_CHECKPOINT = pending_parent_checkpoint(2)
PARENT_CHECKPOINT = PENDING_PARENT_CHECKPOINT
CONFIG = make_sim2real_curriculum_config(2, checkpoint=PARENT_CHECKPOINT)

