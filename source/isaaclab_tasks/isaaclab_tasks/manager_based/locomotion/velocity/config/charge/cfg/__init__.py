"""配置模組

包含所有環境配置相關的文件：
- charge_env_cfg.py: 基礎配置（Phase 1）
- charge_env_cfg_v2.py: Phase 2 配置
- charge_env_cfg_v3.py: Phase 3 配置（自適應課程學習）
- charge_cfg.py: 機器人配置
- charge_env.py: 環境類
"""

# 導出所有配置類
from .charge_env_cfg import (
    ChargeNavigationEnvCfg,
    ChargeNavigationEnvCfg_PLAY,
)

from .charge_env_cfg_v2 import (
    ChargeNavigationEnvCfgV2,
    ChargeNavigationEnvCfgV2_PLAY,
)

from .charge_env_cfg_v3 import (
    ChargeNavigationEnvCfgV3,
    ChargeNavigationEnvCfgV3_PLAY,
)

from .charge_env import (
    ChargeNavigationEnv,
)

__all__ = [
    # Phase 1 配置
    "ChargeNavigationEnvCfg",
    "ChargeNavigationEnvCfg_PLAY",
    # Phase 2 配置
    "ChargeNavigationEnvCfgV2",
    "ChargeNavigationEnvCfgV2_PLAY",
    # Phase 3 配置（自適應課程學習）
    "ChargeNavigationEnvCfgV3",
    "ChargeNavigationEnvCfgV3_PLAY",
    # 環境類
    "ChargeNavigationEnv",
]

