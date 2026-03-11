"""配置模組（SKRL 版本）

包含所有環境配置相關的文件：

Phase 0 配置（基礎訓練）:
- charge_env_cfg_phase0.py: Phase 0 - 空房間 + 混合障礙物（車輛動力學校準）
- charge_env_cfg_phase0_navrl.py: Phase 0 NavRL - NavRL 風格獎勵函數

VLP-16 配置（離散動作 + Multi-Branch CNN）:
- charge_env_cfg_vlp16.py: VLP-16 離散動作環境 v2（79D Policy / 79D Critic）

競爭式配置:
- charge_env_cfg_competitive.py: 競爭式多 Agent 訓練

迷宮配置:
- charge_env_cfg_maze.py: 迷宮導航環境

共用模組:
- charge_cfg.py: 機器人配置
- charge_env_cfg.py: 基礎環境配置（觀測/獎勵/終止/事件）
"""

# ============================================================================
# Phase 0 配置
# ============================================================================

from .charge_env_cfg_phase0 import (
    ChargeNavigationEnvCfgPhase0,
    ChargeNavigationEnvCfgPhase0_PLAY,
)

# NavRL 風格配置
from .charge_env_cfg_phase0_navrl import (
    ChargeNavigationEnvCfgPhase0NavRL,
)

# ============================================================================
# VLP-16 配置
# ============================================================================

from .charge_env_cfg_vlp16 import (
    ChargeNavigationEnvCfgVLP16,
)

# ============================================================================
# 競爭式配置（實驗性，competitive_rewards 模組尚未完成）
# ============================================================================
try:
    from .charge_env_cfg_competitive import (
        ChargeNavigationEnvCfgCompetitive,
    )
except ImportError:
    ChargeNavigationEnvCfgCompetitive = None  # type: ignore[assignment, misc]

__all__ = [
    # Phase 0
    "ChargeNavigationEnvCfgPhase0",
    "ChargeNavigationEnvCfgPhase0_PLAY",
    # NavRL
    "ChargeNavigationEnvCfgPhase0NavRL",
    # VLP-16
    "ChargeNavigationEnvCfgVLP16",
    # Competitive
    "ChargeNavigationEnvCfgCompetitive",
]
