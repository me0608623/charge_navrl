# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航任務 - Agents 配置模組（SKRL 專用）

此模組包含 SKRL 強化學習算法的 YAML 配置。

SKRL YAML 配置加載機制說明：
===============================
YAML 文件通過 gym.register 的 skrl_cfg_entry_point 參數引用。
格式為："{模組名}:{文件名}.yaml"

加載流程：
1. gym.register 註冊環境時，記錄 skrl_cfg_entry_point
2. parse_cfg.load_cfg_from_registry() 解析配置：
   - 檢查是否以 .yaml 結尾
   - 使用 importlib 導入模組
   - 獲取模組路徑並拼接 YAML 文件路徑
   - 使用 yaml.full_load() 加載配置為字典

SKRL YAML 配置清單：
- skrl_ppo_cfg_phase0.yaml：Phase 0 基礎 PPO 配置
- skrl_ppo_cfg_navrl.yaml：NavRL 風格 CNN 架構配置
- skrl_ppo_cfg_charge_cnn.yaml：多分支 CNN 特徵萃取器配置
"""


def _verify_skrl_configs() -> dict[str, bool]:
    """驗證 SKRL YAML 配置文件是否存在

    Returns:
        dict: {"文件名": 是否存在}
    """
    import os
    current_dir = os.path.dirname(__file__)
    configs = {
        "skrl_ppo_cfg_phase0.yaml": os.path.exists(os.path.join(current_dir, "skrl_ppo_cfg_phase0.yaml")),
        "skrl_ppo_cfg_navrl.yaml": os.path.exists(os.path.join(current_dir, "skrl_ppo_cfg_navrl.yaml")),
        "skrl_ppo_cfg_charge_cnn.yaml": os.path.exists(os.path.join(current_dir, "skrl_ppo_cfg_charge_cnn.yaml")),
    }
    return configs


_SKRL_CONFIGS_STATUS = _verify_skrl_configs()
if not all(_SKRL_CONFIGS_STATUS.values()):
    missing = [k for k, v in _SKRL_CONFIGS_STATUS.items() if not v]
    import warnings
    warnings.warn(
        f"[Charge SKRL] Some SKRL YAML config files are missing: {missing}. "
        f"This may cause errors when using SKRL environments."
    )

__all__ = [
    "_verify_skrl_configs",
    "_SKRL_CONFIGS_STATUS",
]
