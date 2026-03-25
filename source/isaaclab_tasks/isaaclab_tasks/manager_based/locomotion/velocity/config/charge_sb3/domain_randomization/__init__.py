"""
域隨機化模組

實現完整的域隨機化系統，提高 Sim-to-Real 遷移效果。

功能：
1. 物理參數隨機化：質量 ±10%、摩擦力 ±20%
2. 傳感器噪聲增強：LiDAR 真實傳感器特性
3. 外部擾動：隨機推力模擬不平整地面
4. 初始狀態隨機：重置時速度隨機

使用方式：
    # 在環境配置中
    from .domain_randomization import (
        DomainRandConfig,
        apply_domain_randomization,
    )

    # 在事件配置中添加重置事件
    class EventCfg:
        domain_randomization = EventTerm(
            func=apply_domain_randomization,
            mode="reset",
            params={},
        )
"""

from .dr_manager import DomainRandConfig, DomainRandomizationManager
from .dr_events import apply_domain_randomization, domain_randomization_pre_step

__all__ = [
    "DomainRandConfig",
    "DomainRandomizationManager",
    "apply_domain_randomization",
    "domain_randomization_pre_step",
]
