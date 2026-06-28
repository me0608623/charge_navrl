# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Charge 機器人配置（SKRL 版本）

定義 Charge 差速驅動機器人的：
1. 物理模型（USD 文件、質量、碰撞體）
2. 初始狀態（出生位置、姿態）
3. 執行器（輪子驅動方式）
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# USD 路徑優先序: 環境變數 CHARGE_USD_PATH > repo 內自包含 robot (本檔上層 ../charge.usd) > 本機 fallback
#   ../charge.usd = 自包含完整機器人 (Mesh + RigidBody, 無 payload)，已隨 repo (de-LFS) 一起 clone，
#                   fresh clone 即可用、不依賴 LFS auth 也不依賴本機檔。
#   ⚠ 舊 assets/usd/charge/charge.usd 是「殼 + payload」，單獨載入沒有 rigid body
#     (ValueError: ...no rigid bodies)，故不再列入候選。
_USD_CANDIDATES = [
    os.environ.get("CHARGE_USD_PATH", ""),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "charge.usd")),
    "/home/aa/usd/charge/charge.usd",
]
_USD_PATH = next((p for p in _USD_CANDIDATES if p and os.path.isfile(p)), _USD_CANDIDATES[1])

CHARGE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=0.5,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint", "left_wheel_joint"],
            stiffness=0.0,
            damping=10.0,
            velocity_limit=100.0,
        ),
    },
)
