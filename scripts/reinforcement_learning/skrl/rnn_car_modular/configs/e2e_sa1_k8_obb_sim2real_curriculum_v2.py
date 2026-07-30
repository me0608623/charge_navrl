"""Canonical navigation-first SA1 sim-to-real curriculum v2."""

from rnn_car_modular.configs.sim2real_stage_curriculum_v2 import (
    make_sim2real_curriculum_config,
)


CONFIG = make_sim2real_curriculum_config(1, checkpoint=None)

