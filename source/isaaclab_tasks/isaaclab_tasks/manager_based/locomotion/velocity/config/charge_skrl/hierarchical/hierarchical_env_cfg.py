"""層級式導航環境配置 (Hierarchical Navigation Environment Configuration)

整合 AIT* + RL (PPO) 的完整環境配置。
"""

import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.sensors import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

# 導入基礎配置
from ..cfg.charge_env import ChargeNavigationEnvCfg, MySceneCfg
from ..mdp.observations import (
    lidar_scan_2d_sweep,
    base_velocity_xy,
    safe_last_action,
    time_remaining_ratio,
    alive_flag,
)

# 導入層級式導航組件
from ..mdp.observations.hierarchical_navigation import (
    navigation_features,
)
from ..mdp.rewards.hierarchical_rewards import (
    local_goal_reached_reward,
    progress_to_local_goal,
    heading_alignment_reward,
    hierarchical_navigation_reward,
)
from .hierarchical_navigation_manager import HierarchicalNavigationManager


@configclass
class HierarchicalObservationsCfg:
    """層級式導航觀測配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組：LiDAR + 導航特征"""
        
        # LiDAR 感知（避障）
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        
        # 導航特征（跟隨 AIT* 路徑）
        nav_features = ObsTerm(
            func=navigation_features,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "goal_cfg": SceneEntityCfg("goal_command"),
            },
        )
        
        # 機器人速度
        velocity = ObsTerm(
            func=base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        
        # 上一幀動作
        actions = ObsTerm(func=safe_last_action)
        
        # 時間剩餘
        time_remaining = ObsTerm(func=time_remaining_ratio)
        
        # 存活標誌
        alive = ObsTerm(func=alive_flag)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    policy: PolicyCfg = PolicyCfg()


@configclass
class HierarchicalRewardsCfg:
    """層級式導航獎勵配置"""
    
    # 抵達局部目標獎勵
    local_goal_reached = RewTerm(
        func=local_goal_reached_reward,
        weight=10.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_cfg": SceneEntityCfg("goal_command"),
            "threshold": 0.5,
        },
    )
    
    # 前進獎勵（靠近目標）
    progress = RewTerm(
        func=progress_to_local_goal,
        weight=2.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_cfg": SceneEntityCfg("goal_command"),
        },
    )
    
    # 航向對齊獎勵
    tracking = RewTerm(
        func=heading_alignment_reward,
        weight=0.5,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "goal_cfg": SceneEntityCfg("goal_command"),
        },
    )


@configclass
class HierarchicalNavigationEnvCfg(ChargeNavigationEnvCfg):
    """層級式導航環境配置
    
    整合 AIT* 全域規劃器 + RL 局部控制器。
    """
    
    scene: MySceneCfg = MySceneCfg(num_envs=128, env_spacing=25.0)
    observations: HierarchicalObservationsCfg = HierarchicalObservationsCfg()
    rewards: HierarchicalRewardsCfg = HierarchicalRewardsCfg()


__all__ = [
    "HierarchicalObservationsCfg",
    "HierarchicalRewardsCfg",
    "HierarchicalNavigationEnvCfg",
]
