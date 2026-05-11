"""Experiment config registry -- lookup by name or file path.

支援 YAML (.yaml，推薦) 和 Python (.py，legacy) config 檔案。
YAML configs 自動從此目錄發現，不需手動註冊。

使用方式:
  --experiment_config wd_sa2_a2c_aux          # 用名字（自動找 .yaml）
  --experiment_config /path/to/my_config.yaml # 用 YAML 路徑
  --experiment_config /path/to/my_config.py   # 用 Python 路徑 (legacy)
"""

from __future__ import annotations

import os
from pathlib import Path

from rnn_car_modular.experiment_config import (
    ExperimentConfig,
    load_experiment_config_from_file,
    load_experiment_config_from_yaml,
)

# Directory containing built-in config files
_CONFIGS_DIR = Path(__file__).parent


def _discover_yaml_configs() -> dict[str, Path]:
    """Auto-discover all .yaml configs in this directory."""
    configs = {}
    for p in sorted(_CONFIGS_DIR.glob("*.yaml")):
        configs[p.stem] = p
    return configs


# Lazy singleton
_YAML_CONFIGS: dict[str, Path] | None = None


def _get_yaml_configs() -> dict[str, Path]:
    global _YAML_CONFIGS
    if _YAML_CONFIGS is None:
        _YAML_CONFIGS = _discover_yaml_configs()
    return _YAML_CONFIGS


def get_experiment_config(name_or_path: str) -> ExperimentConfig:
    """Get an ExperimentConfig by registry name or file path.

    Resolution order:
    1. YAML file in configs/ directory (by stem name)
    2. Direct file path (.yaml or .py)

    Args:
        name_or_path: Either a config name (e.g. "wd_sa2_a2c_aux")
                      or a path to a .yaml/.py file.

    Returns:
        The resolved ExperimentConfig.
    """
    # 1. Try as registered YAML config name
    yaml_configs = _get_yaml_configs()
    if name_or_path in yaml_configs:
        return load_experiment_config_from_yaml(str(yaml_configs[name_or_path]))

    # 2. Try as file path
    if os.path.isfile(name_or_path):
        if name_or_path.endswith((".yaml", ".yml")):
            return load_experiment_config_from_yaml(name_or_path)
        if name_or_path.endswith(".py"):
            return load_experiment_config_from_file(name_or_path)

    available = ", ".join(sorted(yaml_configs.keys()))
    raise KeyError(
        f"Unknown experiment config: {name_or_path!r}. "
        f"Available: [{available}]. Or provide a .yaml/.py file path."
    )
