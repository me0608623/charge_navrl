"""配置模組

包含所有環境配置相關的文件：

新的地形 Curriculum:
- charge_env_cfg_phase0.py: Phase 0 - 空房間（車輛動力學校準）
- charge_env_cfg_phase1.py: Phase 1 - 內牆結構（U型牆、隔間）
- charge_env_cfg_phase2.py: Phase 2 - 走廊與窄通道
- charge_env_cfg_phase3.py: Phase 3 - 複雜地形 + 動態障礙物

NavRL 風格配置 🆕:
- charge_env_cfg_phase0_navrl.py: NavRL 獎勵函數移植自 NavRL (IEEE RA-L 2025)

舊版配置（保留兼容性）:
- charge_env_cfg.py: 基礎配置（舊 Phase 1）
- charge_env_cfg_v2.py: Phase 2 配置
- charge_env_cfg_v3.py: Phase 3 配置（自適應課程學習）
- charge_cfg.py: 機器人配置
- charge_env.py: 環境類
"""

# ============================================================================
# 新地形 Curriculum 導出
# ============================================================================

from .charge_env_cfg_phase0 import (
    ChargeNavigationEnvCfgPhase0,
    ChargeNavigationEnvCfgPhase0_PLAY,
)

from .charge_env_cfg_phase1 import (
    ChargeNavigationEnvCfgPhase1,
    ChargeNavigationEnvCfgPhase1_PLAY,
)

from .charge_env_cfg_phase2 import (
    ChargeNavigationEnvCfgPhase2,
    ChargeNavigationEnvCfgPhase2_PLAY,
)

from .charge_env_cfg_phase3 import (
    ChargeNavigationEnvCfgPhase3,
    ChargeNavigationEnvCfgPhase3_PLAY,
)

# 🆕 NavRL 風格配置
from .charge_env_cfg_phase0_navrl import (
    ChargeNavigationEnvCfgPhase0NavRL,
)

# ============================================================================
# 舊版配置導出（保留兼容性）
# ============================================================================

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
    HierarchicalChargeNavigationEnv,
)

__all__ = [
    # ========================================
    # 新地形 Curriculum (推薦使用)
    # ========================================
    # Phase 0: 空房間（車輛動力學校準）
    "ChargeNavigationEnvCfgPhase0",
    "ChargeNavigationEnvCfgPhase0_PLAY",
    # Phase 1: 內牆結構（U型牆、隔間）
    "ChargeNavigationEnvCfgPhase1",
    "ChargeNavigationEnvCfgPhase1_PLAY",
    # Phase 2: 走廊與窄通道
    "ChargeNavigationEnvCfgPhase2",
    "ChargeNavigationEnvCfgPhase2_PLAY",
    # Phase 3: 複雜地形 + 動態障礙物
    "ChargeNavigationEnvCfgPhase3",
    "ChargeNavigationEnvCfgPhase3_PLAY",

    # ========================================
    # 🆕 NavRL 風格配置
    # ========================================
    "ChargeNavigationEnvCfgPhase0NavRL",

    # ========================================
    # 舊版配置（保留兼容性）
    # ========================================
    "ChargeNavigationEnvCfg",
    "ChargeNavigationEnvCfg_PLAY",
    "ChargeNavigationEnvCfgV2",
    "ChargeNavigationEnvCfgV2_PLAY",
    "ChargeNavigationEnvCfgV3",
    "ChargeNavigationEnvCfgV3_PLAY",

    # 環境類
    "ChargeNavigationEnv",
    "HierarchicalChargeNavigationEnv",
]

