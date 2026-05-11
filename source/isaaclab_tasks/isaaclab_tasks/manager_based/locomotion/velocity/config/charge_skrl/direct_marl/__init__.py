"""DirectMARLEnv for Charge Navigation — Phase 1 (single car)

DirectMARLEnv 版本的 Charge 導航任務，為 Multi-Agent RL 做準備。
Phase 1: 單車驗證框架正確性，確保訓練曲線與 ManagerBased 版本一致。
"""

import gymnasium as gym

gym.register(
    id="Isaac-Navigation-Charge-MARL-VLP16",
    entry_point=f"{__name__}.charge_marl_env:DirectMARLChargeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.charge_marl_env:DirectMARLChargeCfg",
    },
)
