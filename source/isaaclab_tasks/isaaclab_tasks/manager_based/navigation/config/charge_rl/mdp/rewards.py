# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
導航任務的獎勵函數模組 (Reward Functions for Navigation Task)
此模組定義了用於引導導航策略學習的獎勵函數，包括：
- 位置追蹤獎勵：鼓勵機器人接近目標位置
- 方向追蹤獎勵：鼓勵機器人朝向目標方向
"""

from __future__ import annotations

import torch  # PyTorch 核心庫，用於張量運算
from typing import TYPE_CHECKING  # 用於類型檢查時的條件導入

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv  # 僅在類型檢查時導入，避免循環依賴


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """
    使用 tanh 核函數計算位置追蹤獎勵 (Reward position tracking with tanh kernel)
    
    此函數計算機器人當前位置與目標位置之間的距離，並使用 tanh 函數將其轉換為平滑的獎勵信號。
    tanh 函數的特性：
    - 當距離接近 0 時，獎勵接近 1（最大獎勵）
    - 當距離增大時，獎勵平滑遞減
    - 當距離遠大於 std 時，獎勵趨於 0（飽和）
    
    這種設計可以：
    1. 提供平滑的獎勵梯度，有利於策略學習
    2. 避免獎勵信號過大或過小，保持訓練穩定性
    3. 通過調整 std 參數，控制獎勵的敏感範圍
    
    Args:
        env: 強化學習環境實例，包含命令管理器等
        std: 標準差參數，控制 tanh 函數的敏感範圍（單位：米）
             - 較小的 std（如 0.2）：在接近目標時提供較大的獎勵梯度，適合精確追蹤
             - 較大的 std（如 2.0）：在較遠距離時仍提供獎勵梯度，適合長距離導航
        command_name: 命令名稱，用於從命令管理器中獲取目標位置命令
    
    Returns:
        torch.Tensor: 形狀為 (num_envs,) 的獎勵張量，每個元素對應一個環境的獎勵值
                     獎勵範圍：0 到 1（距離為 0 時獎勵為 1，距離增大時獎勵遞減）
    
    數學公式：
        distance = ||desired_position - current_position||
        reward = 1 - tanh(distance / std)
    
    範例：
        如果 std=2.0，距離為 2 米時：
        reward = 1 - tanh(2.0 / 2.0) = 1 - tanh(1.0) ≈ 1 - 0.76 = 0.24
        
        如果 std=0.2，距離為 0.2 米時：
        reward = 1 - tanh(0.2 / 0.2) = 1 - tanh(1.0) ≈ 0.24
    """
    # 從命令管理器中獲取指定名稱的命令（目標姿勢命令）
    # 命令格式通常為 [x, y, z, heading]，其中 x, y, z 是目標位置相對於當前位置的偏移
    command = env.command_manager.get_command(command_name)
    # 提取目標位置的前三個元素（x, y, z），這是在基座座標系中的相對位置
    # command[:, :3] 的形狀為 (num_envs, 3)
    des_pos_b = command[:, :3]
    # 計算目標位置向量的模長（歐幾里得距離）
    # torch.norm(..., dim=1) 計算每個環境中目標位置的距離
    # 結果形狀為 (num_envs,)，表示每個環境中機器人到目標位置的距離
    distance = torch.norm(des_pos_b, dim=1)
    # 使用 tanh 函數將距離轉換為獎勵值
    # tanh(distance / std) 將距離標準化，然後映射到 [0, 1) 區間
    # 1 - tanh(...) 確保距離為 0 時獎勵為 1，距離增大時獎勵遞減
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """
    計算方向追蹤誤差的絕對值，作為懲罰項 (Penalize tracking orientation error)
    
    此函數計算機器人當前朝向與目標朝向之間的誤差（絕對值），用於懲罰方向偏差。
    在獎勵配置中，此函數通常與負權重結合使用，表示誤差越大，懲罰越大（獎勵越小）。
    
    設計目的：
    1. 確保機器人不僅到達目標位置，還要朝向正確的方向
    2. 提供平滑的懲罰信號，避免方向誤差過大
    3. 與位置追蹤獎勵配合，實現完整的姿勢（位置+朝向）追蹤
    
    Args:
        env: 強化學習環境實例，包含命令管理器等
        command_name: 命令名稱，用於從命令管理器中獲取目標姿勢命令
    
    Returns:
        torch.Tensor: 形狀為 (num_envs,) 的誤差張量，每個元素對應一個環境的方向誤差（單位：弧度）
                     誤差範圍：0 到 π（0 表示完全對齊，π 表示完全相反）
    
    注意：
        - 此函數返回的是誤差值（懲罰），不是獎勵值
        - 在獎勵配置中使用時，通常會設置負權重（如 weight=-0.2）
        - 最終獎勵 = weight * heading_error，誤差越大，獎勵越小
    
    範例：
        如果目標朝向為 0 弧度（正前方），當前朝向為 0.1 弧度：
        error = |0.1| = 0.1 弧度
        
        如果目標朝向為 0 弧度，當前朝向為 π 弧度（完全相反）：
        error = |π| = π 弧度（最大誤差）
    """
    # 從命令管理器中獲取指定名稱的命令（目標姿勢命令）
    command = env.command_manager.get_command(command_name)
    # 提取目標朝向（heading）角度，這是命令的第四個元素（索引 3）
    # command[:, 3] 的形狀為 (num_envs,)，表示每個環境的目標朝向誤差（相對於當前朝向）
    # 注意：這裡的 heading_b 已經是誤差值，不是絕對朝向
    heading_b = command[:, 3]
    # 計算朝向誤差的絕對值
    # 使用 abs() 確保誤差為正數，無論機器人是順時針還是逆時針偏離目標
    # 結果形狀為 (num_envs,)，表示每個環境的方向誤差（單位：弧度）
    return heading_b.abs()
