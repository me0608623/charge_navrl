# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Navigation-Carter-v0 --num_envs 5

"""
Carter 機器人配置模組

此模組定義了 Carter 機器人的完整配置，包括：
- USD 模型文件路徑
- 物理屬性（剛體屬性、關節屬性）
- 初始狀態（位置、關節角度、關節速度）
- 執行器配置（輪子速度控制）
"""

import isaaclab.sim as sim_utils  # Isaac Lab 模擬工具，提供物理配置類別
from isaaclab.actuators import ImplicitActuatorCfg  # 隱式執行器配置（用於速度控制）
from isaaclab.assets.articulation import ArticulationCfg  # 關節式機器人配置基類
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # Isaac Nucleus 資產目錄路徑（預設機器人模型）

##
# 配置定義 (Configuration)
##

CARTER_CFG = ArticulationCfg(
    # 生成配置：定義如何從 USD 文件載入機器人模型
    spawn=sim_utils.UsdFileCfg(
        # Nova Carter Base - 差速驅動底盤
        # USD 文件路徑：本地自定義的機器人模型文件
        usd_path="/home/aa/usd/Carter/Carter/nova_carter_sensors.usd",
        # 備用路徑：使用 Isaac Nucleus 中的預設模型（如果本地文件不存在）
        # usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Carter/nova_carter_sensors.usd",
        activate_contact_sensors=False,  # 不啟用接觸感測器（此任務不需要）
        # 剛體屬性配置：定義機器人各剛體的物理屬性
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 不禁用重力：機器人受重力影響（貼地）
            max_depenetration_velocity=1.0,  # 最大去穿透速度：1.0 m/s（防止物體快速穿透）
        ),
        # 關節屬性配置：定義關節求解器的參數
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,  # 禁用自碰撞檢測：機器人各部件不會相互碰撞（提高性能）
            solver_position_iteration_count=4,  # 位置求解器迭代次數：4 次（平衡精度和性能）
            solver_velocity_iteration_count=1,  # 速度求解器迭代次數：1 次（足夠用於簡單場景）
        ),
    ),
    # 初始狀態配置：定義機器人載入時的初始狀態
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # 初始位置：X=0, Y=0, Z=0（貼地）
        joint_pos={
            ".*": 0.0,  # 所有關節初始角度：0.0（使用正則表達式 ".*" 匹配所有關節）
        },
        joint_vel={
            ".*": 0.0,  # 所有關節初始速度：0.0（靜止開始）
        },
    ),
    # 執行器配置：定義如何控制機器人的關節
    actuators={
        # 輪子速度控制執行器
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel.*"],  # 關節名稱模式：匹配所有包含 "wheel" 的關節（正則表達式）
            stiffness=0.0,  # 剛度：0.0（速度控制模式，不需要位置剛度）
            damping=10.0,  # 阻尼：10.0（高阻尼，模擬速度控制器的響應特性）
            velocity_limit=100.0,  # 速度限制：100.0 rad/s（輪子的最大角速度，實際會被動作空間限制）
        ),
    },
)