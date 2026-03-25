"""CBF 修正懲罰函數

用於與 SB3 PPO + CBF 整合的懲罰機制。
當 CBF 安全濾鏡修改 PPO 的動作時，給予懲罰以防止 PPO 依賴 CBF。

核心概念：
1. PPO 是 On-Policy 演算法，假設訓練數據由當前策略產生
2. 當 CBF 修改動作 (u_safe ≠ u_rl)，PPO 會誤以為 u_safe 是自己想出來的
3. 若無懲罰，PPO 會學到「衝向障礙物也沒關係，CBF 會救我」
4. 懲罰機制告訴 PPO：「被 CBF 修改動作代表你做錯了」

使用方式：
    # 在 CBFActionWrapper 中記錄修正量
    penalty = penalty_coeff * ||u_rl - u_safe||^2
    reward -= penalty
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .utils import _check_reward_term


def cbf_correction_penalty(
    env: ManagerBasedRLEnv,
    policy_cfg: SceneEntityCfg,
    penalty_coeff: float = 0.5,
    use_squared_penalty: bool = True,
    min_correction_threshold: float = 0.01,
) -> torch.Tensor:
    """CBF 修正懲罰（基於動作差異）

    計算 CBF 安全濾鏡對動作的修正量，並給予懲罰。
    這個函數需要在環境中記錄原始動作和修正後動作的差異。

    設計理念：
    - 修正量 = ||u_rl - u_safe||
    - 懲罰 = penalty_coeff * (修正量)^2
    - 只有修正量超過閾值才給懲罰

    Args:
        env: 環境實例
        policy_cfg: 策略資產配置（用於獲取動作）
        penalty_coeff: 懲罰係數（建議 0.1 ~ 2.0）
        use_squared_penalty: 是否使用平方懲罰（True: 更大懲罰）
        min_correction_threshold: 最小修正閾值（小於此值不懲罰）

    Returns:
        shape [num_envs]：懲罰值 [0, +inf)
        0 = 無修正或修正量很小
        >0 = CBF 修改了動作，給予懲罰

    注意：
        此函數需要環境在 step() 時記錄原始動作和修正後動作。
        通常配合 CBFActionWrapper 使用。
    """
    # 嘗試從環境獲取 CBF 修正資訊
    # 這需要在環境中實現相應的追蹤機制
    correction = _get_cbf_correction(env)

    if correction is None:
        # 如果環境沒有 CBF 修正資訊，返回零懲罰
        return torch.zeros(env.num_envs, device=env.device)

    # 應用最小修正閾值
    correction = torch.clamp(correction - min_correction_threshold, min=0.0)

    if use_squared_penalty:
        # 使用平方懲罰（更強的懲罰）
        penalty = penalty_coeff * (correction ** 2)
    else:
        # 線性懲罰
        penalty = penalty_coeff * correction

    # 安全處理
    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=10.0, neginf=0.0)
    penalty = torch.clamp(penalty, 0.0, 10.0)

    # 檢查 reward term
    penalty = _check_reward_term("cbf_correction_penalty", penalty, env, raise_on_error=False)

    return penalty


def cbf_intervention_penalty(
    env: ManagerBasedRLEnv,
    policy_cfg: SceneEntityCfg,
    fixed_penalty: float = 0.5,
) -> torch.Tensor:
    """CBF 介入懲罰（二元懲罰）

    當 CBF 介入修改動作時，給予固定懲罰。
    這是一個簡化版本，只關注「是否有介入」而不關介入「介入了多少」。

    Args:
        env: 環境實例
        policy_cfg: 策略資產配置
        fixed_penalty: 固定懲罰值（當 CBF 介入時）

    Returns:
        shape [num_envs]：懲罰值 {0, fixed_penalty}
        0 = CBF 未介入
        fixed_penalty = CBF 介入了

    適用場景：
    - 想要強烈懲罰任何 CBF 介入
    - 不關心修正量的大小
    """
    triggered = _get_cbf_triggered(env)

    if triggered is None:
        return torch.zeros(env.num_envs, device=env.device)

    penalty = torch.where(triggered, torch.tensor(fixed_penalty, device=env.device),
                         torch.zeros(env.num_envs, device=env.device))

    penalty = _check_reward_term("cbf_intervention_penalty", penalty, env, raise_on_error=False)

    return penalty


def cbf_adaptive_penalty(
    env: ManagerBasedRLEnv,
    policy_cfg: SceneEntityCfg,
    base_penalty_coeff: float = 0.5,
    distance_aware: bool = True,
    sensor_cfg: Optional[SceneEntityCfg] = None,
) -> torch.Tensor:
    """CBF 自適應懲罰（根據危險程度調整）

    根據當前狀態的危險程度動態調整懲罰係數：
    - 在安全區域：較小的懲罰（允許探索）
    - 在危險區域：較大的懲罰（強調安全性）

    Args:
        env: 環境實例
        policy_cfg: 策略資產配置
        base_penalty_coeff: 基礎懲罰係數
        distance_aware: 是否根據障礙物距離調整懲罰
        sensor_cfg: 雷達感測器配置（用於獲取障礙物距離）

    Returns:
        shape [num_envs]：自適應懲罰值
    """
    correction = _get_cbf_correction(env)

    if correction is None:
        return torch.zeros(env.num_envs, device=env.device)

    # 基礎懲罰
    penalty = base_penalty_coeff * (correction ** 2)

    # 根據危險程度調整
    if distance_aware and sensor_cfg is not None:
        from isaaclab.sensors import RayCaster

        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        sensor_pos_2d = sensor.data.pos_w[:, :2]
        hit_points_2d = sensor.data.ray_hits_w[:, :, :2]
        distances_2d = torch.norm(hit_points_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
        distances_2d = torch.nan_to_num(distances_2d, nan=sensor.cfg.max_distance,
                                       posinf=sensor.cfg.max_distance)
        min_distance = torch.min(distances_2d, dim=1)[0]

        # 危險係數：距離越近，係數越大
        # 距離 < 0.5m：係數 = 2.0
        # 距離 > 2.0m：係數 = 1.0
        danger_coefficient = torch.clamp(2.0 - min_distance, 1.0, 2.0)
        penalty = penalty * danger_coefficient

    penalty = torch.nan_to_num(penalty, nan=0.0, posinf=10.0, neginf=0.0)
    penalty = torch.clamp(penalty, 0.0, 10.0)
    penalty = _check_reward_term("cbf_adaptive_penalty", penalty, env, raise_on_error=False)

    return penalty


# ============================================================================
# 輔助函數
# ============================================================================

def _get_cbf_correction(env: ManagerBasedRLEnv) -> Optional[torch.Tensor]:
    """從環境獲取 CBF 修正量

    嘗試多種方式獲取 CBF 修正資訊：
    1. 從 env 的 cbf_correction 屬性
    2. 從 env 的 extras 字典
    3. 返回 None（表示無資訊）

    Args:
        env: 環境實例

    Returns:
        CBF 修正量 tensor [num_envs]，或 None
    """
    # 方法 1：直接從環境屬性獲取
    if hasattr(env, "cbf_correction"):
        correction = env.cbf_correction
        if isinstance(correction, torch.Tensor):
            return correction
        elif isinstance(correction, (list, tuple)):
            return torch.tensor(correction, device=env.device, dtype=torch.float32)

    # 方法 2：從最後的 extras 獲取
    if hasattr(env, "_last_extras") and isinstance(env._last_extras, dict):
        if "cbf_correction" in env._last_extras:
            correction = env._last_extras["cbf_correction"]
            if isinstance(correction, torch.Tensor):
                return correction

    # 方法 3：從 wrapper 鏈獲取
    current = env
    while hasattr(current, "env"):
        if hasattr(current, "cbf_correction"):
            correction = current.cbf_correction
            if isinstance(correction, torch.Tensor):
                return correction
        if hasattr(current, "_last_corrections"):
            correction = current._last_corrections
            if isinstance(correction, torch.Tensor):
                return correction
            elif isinstance(correction, (list, tuple, np.ndarray)):
                return torch.tensor(correction, device=env.device, dtype=torch.float32)
        current = current.env

    return None


def _get_cbf_triggered(env: ManagerBasedRLEnv) -> Optional[torch.Tensor]:
    """從環境獲取 CBF 觸發狀態

    Returns:
        CBF 觸發布林 tensor [num_envs]，或 None
    """
    if hasattr(env, "cbf_triggered"):
        triggered = env.cbf_triggered
        if isinstance(triggered, torch.Tensor):
            return triggered.bool()
        elif isinstance(triggered, (list, tuple)):
            return torch.tensor(triggered, device=env.device, dtype=torch.bool)

    if hasattr(env, "_last_extras") and isinstance(env._last_extras, dict):
        if "cbf_triggered" in env._last_extras:
            triggered = env._last_extras["cbf_triggered"]
            if isinstance(triggered, torch.Tensor):
                return triggered.bool()

    current = env
    while hasattr(current, "env"):
        if hasattr(current, "_last_triggered"):
            triggered = current._last_triggered
            if isinstance(triggered, torch.Tensor):
                return triggered.bool()
            elif isinstance(triggered, (list, tuple, np.ndarray)):
                return torch.tensor(triggered, device=env.device, dtype=torch.bool)
        current = current.env

    return None


# 導入 numpy 用於類型檢查
import numpy as np


__all__ = [
    "cbf_correction_penalty",
    "cbf_intervention_penalty",
    "cbf_adaptive_penalty",
]
