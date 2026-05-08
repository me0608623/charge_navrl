"""Experiment config registry -- lookup by name or file path."""

from __future__ import annotations

import os

from rnn_car_modular.experiment_config import ExperimentConfig, load_experiment_config_from_file

# Lazy imports to avoid circular dependency; configs import ExperimentConfig
from rnn_car_modular.configs.navrl_ground_a2c_aux import CONFIG as navrl_ground_a2c_aux
from rnn_car_modular.configs.wd_sa2_a2c_aux import CONFIG as wd_sa2_a2c_aux
from rnn_car_modular.configs.wd_sa2_a2c_aux_lowent import CONFIG as wd_sa2_a2c_aux_lowent
from rnn_car_modular.configs.wd_sa2_a2c_noaux import CONFIG as wd_sa2_a2c_noaux
from rnn_car_modular.configs.wd_sa2_ppo_noaux import CONFIG as wd_sa2_ppo_noaux


EXPERIMENT_CONFIGS: dict[str, ExperimentConfig] = {
    "navrl_ground_a2c_aux": navrl_ground_a2c_aux,
    "wd_sa2_a2c_aux": wd_sa2_a2c_aux,
    "wd_sa2_a2c_aux_lowent": wd_sa2_a2c_aux_lowent,
    "wd_sa2_a2c_noaux": wd_sa2_a2c_noaux,
    "wd_sa2_ppo_noaux": wd_sa2_ppo_noaux,
}


def get_experiment_config(name_or_path: str) -> ExperimentConfig:
    """Get an ExperimentConfig by registry name or Python file path.

    Args:
        name_or_path: Either a registered config name (e.g. "wd_sa2_a2c_aux")
                      or a path to a .py file containing CONFIG.

    Returns:
        The resolved ExperimentConfig.

    Raises:
        KeyError: If name not in registry and not a valid file path.
        FileNotFoundError: If path does not exist.
    """
    # Try registry first
    if name_or_path in EXPERIMENT_CONFIGS:
        return EXPERIMENT_CONFIGS[name_or_path]

    # Try as file path
    if os.path.isfile(name_or_path) and name_or_path.endswith(".py"):
        return load_experiment_config_from_file(name_or_path)

    available = ", ".join(sorted(EXPERIMENT_CONFIGS.keys()))
    raise KeyError(
        f"Unknown experiment config: {name_or_path!r}. "
        f"Available: [{available}]. Or provide a .py file path."
    )
