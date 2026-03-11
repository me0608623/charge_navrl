"""類型定義模組

此模組定義 MDP 相關的共用類型,用於:
1. 避免循環導入
2. 提供清晰的類型提示
3. 統一類型定義

注意: 目前大部分類型來自 Isaac Lab,這裡主要定義專案特有的類型。
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union

import torch

# ============================================================================
# 類型檢查導入 (避免運行時循環導入)
# ============================================================================
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation
    from isaaclab.sensors import RayCaster, ContactSensor
    from isaaclab.managers import SceneEntityCfg

# ============================================================================
# 基礎類型別名
# ============================================================================

# 環境相關
EnvType = "ManagerBasedRLEnv"
EnvIndices = Union[torch.Tensor, list[int], int, None]

# 資產相關
AssetType = "Articulation"
EntityCfg = "SceneEntityCfg"

# 感測器相關
LidarType = "RayCaster"
ContactSensorType = "ContactSensor"

# ============================================================================
# 障礙物相關類型
# ============================================================================

# 障礙物元數據: (數量, 尺寸列表)
ObstacleMetadata = tuple[int, list[float]]

# 障礙物位置: shape [num_envs, num_obstacles, 2]
ObstaclePositions = Optional[torch.Tensor]

# 障礙物方向: shape [num_envs, num_obstacles]
ObstacleDirections = Optional[torch.Tensor]

# ============================================================================
# 獎勵與終止相關類型
# ============================================================================

# 獎勵值: shape [num_envs]
RewardTensor = torch.Tensor

# 終止標記: shape [num_envs]
DoneTensor = torch.Tensor

# 觀測值: shape [num_envs, obs_dim]
ObservationTensor = torch.Tensor

# 動作值: shape [num_envs, action_dim]
ActionTensor = torch.Tensor

# ============================================================================
# 配置相關類型
# ============================================================================

# 數值範圍: (min, max)
Range = tuple[float, float]

# 距離參數
Distance = float

# 速度參數
Velocity = float

# ============================================================================
# 實用類型守衛 (Type Guards)
# ============================================================================

def is_valid_env_indices(env_ids: EnvIndices) -> bool:
    """檢查環境索引是否有效
    
    Args:
        env_ids: 環境索引
        
    Returns:
        是否為有效的環境索引
    """
    if env_ids is None:
        return True
    if isinstance(env_ids, (int, list)):
        return True
    if isinstance(env_ids, torch.Tensor):
        return env_ids.dtype == torch.long
    return False


def normalize_env_indices(
    env_ids: EnvIndices,
    num_envs: int,
    device: torch.device
) -> torch.Tensor:
    """將環境索引標準化為 Tensor
    
    Args:
        env_ids: 環境索引 (可以是 None, int, list, 或 Tensor)
        num_envs: 環境總數
        device: 設備
        
    Returns:
        標準化後的環境索引 Tensor
    """
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    
    if isinstance(env_ids, int):
        return torch.tensor([env_ids], device=device, dtype=torch.long)
    
    if isinstance(env_ids, (list, tuple)):
        return torch.tensor(env_ids, device=device, dtype=torch.long)
    
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    
    raise TypeError(f"不支援的 env_ids 類型: {type(env_ids)}")


# ============================================================================
# 未來擴展預留
# ============================================================================

# 當需要定義更多專案特有的類型時,可以在這裡添加
# 例如:
# - 課程學習相關類型
# - 多智能體相關類型
# - 自定義觀測空間類型