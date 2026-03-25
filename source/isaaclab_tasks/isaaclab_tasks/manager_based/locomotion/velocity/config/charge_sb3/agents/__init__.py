# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航任務 - Agents 配置模組

此模組包含所有強化學習算法的配置：
- RSL-RL PPO 配置（.py 文件）
- Stable Baselines 3 PPO 配置（.yaml 文件）

SB3 YAML 配置加載機制說明：
===============================
YAML 文件通過 gym.register 的 sb3_cfg_entry_point 參數引用。
格式為："{模組名}:{文件名}.yaml"

加載流程：
1. gym.register 註冊環境時，記錄 sb3_cfg_entry_point
2. parse_cfg.load_cfg_from_registry() 解析配置：
   - 檢查是否以 .yaml 結尾
   - 使用 importlib 導入模組
   - 獲取模組路徑並拼接 YAML 文件路徑
   - 使用 yaml.full_load() 加載配置為字典

示例：
    gym.register(
        id="Isaac-Navigation-Charge-SB3-v1",
        kwargs={
            "sb3_cfg_entry_point": "isaaclab_tasks...config.charge_sb3.agents:sb3_ppo_cfg.yaml",
        },
    )

    # 加載過程：
    # 1. mod_name = "isaaclab_tasks...config.charge_sb3.agents"
    # 2. file_name = "sb3_ppo_cfg.yaml"
    # 3. mod_path = "/.../charge_sb3/agents/"
    # 4. yaml_path = mod_path + file_name
    # 5. cfg = yaml.full_load(yaml_path)
"""

# 導出 RSL-RL 配置類
from .rsl_rl_ppo_cfg import ChargeNavigationPPORunnerCfg
from .rsl_rl_ppo_cfg_v2 import ChargeNavigationPPORunnerCfgPhase2
from .rsl_rl_ppo_cfg_v2_5 import ChargeNavigationPPORunnerCfgPhase2_5
from .rsl_rl_ppo_cfg_v3 import ChargeNavigationPPORunnerCfgPhase3

# 驗證 YAML 配置文件的存在性（用於調試）
def _verify_sb3_configs() -> dict[str, bool]:
    """驗證 SB3 YAML 配置文件是否存在

    Returns:
        dict: {"文件名": 是否存在}
    """
    import os
    current_dir = os.path.dirname(__file__)
    configs = {
        "sb3_ppo_cfg.yaml": os.path.exists(os.path.join(current_dir, "sb3_ppo_cfg.yaml")),
        "sb3_ppo_cfg_v2.yaml": os.path.exists(os.path.join(current_dir, "sb3_ppo_cfg_v2.yaml")),
        "sb3_ppo_cfg_v3.yaml": os.path.exists(os.path.join(current_dir, "sb3_ppo_cfg_v3.yaml")),
        "sb3_ppo_cfg_breakthrough.yaml": os.path.exists(os.path.join(current_dir, "sb3_ppo_cfg_breakthrough.yaml")),
    }
    return configs

# 在模組加載時驗證（可選，用於調試）
_SB3_CONFIGS_STATUS = _verify_sb3_configs()
if not all(_SB3_CONFIGS_STATUS.values()):
    missing = [k for k, v in _SB3_CONFIGS_STATUS.items() if not v]
    import warnings
    warnings.warn(
        f"[Charge SB3] Some SB3 YAML config files are missing: {missing}. "
        f"This may cause errors when using SB3 environments."
    )

__all__ = [
    # RSL-RL 配置類
    "ChargeNavigationPPORunnerCfg",
    "ChargeNavigationPPORunnerCfgPhase2",
    "ChargeNavigationPPORunnerCfgPhase2_5",
    "ChargeNavigationPPORunnerCfgPhase3",
    # 工具函數
    "_verify_sb3_configs",
    "_SB3_CONFIGS_STATUS",
]
