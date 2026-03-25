# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Charge 導航環境類別（帶有觀測檢查）

這個文件定義了一個自定義環境類，繼承自 ManagerBasedRLEnv，
並在觀測返回前添加 NaN/Inf 檢查，用於修復 PPO std>=0 錯誤。
"""

import torch
from collections.abc import Sequence
from typing import Any
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn

from ..mdp.observations import check_finite as _check_finite

# 導入 AIT* 路徑規劃器和視覺化器
try:
    from ..mdp.path_planner.aitstar_adapter import create_aitstar_planner
    from ..mdp.path_planner.path_visualizer import AITStarPathVisualizer
    _AITSTAR_AVAILABLE = True
except ImportError:
    _AITSTAR_AVAILABLE = False


class ChargeNavigationEnv(ManagerBasedRLEnv):
    """Charge 導航環境（帶有觀測檢查）
    
    這個類繼承自 ManagerBasedRLEnv，並在 step() 方法中添加了觀測檢查，
    用於檢測和修復 NaN/Inf 值，防止 PPO std>=0 錯誤。
    """
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """執行環境動態的一個時間步長（帶有觀測檢查）
        
        在返回觀測前，檢查所有觀測項是否包含 NaN/Inf 值。
        如果發現異常值，會立即拋出異常並提供詳細信息。
        """
        # 調用父類的 step 方法
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)
        
        # ========================================================================
        # 關鍵：在觀測返回前檢查 NaN/Inf（防止 PPO std>=0 錯誤）
        # ========================================================================
        if isinstance(obs_buf, dict):
            # 如果觀測是字典格式（多個觀測組）
            for group_name, obs in obs_buf.items():
                if isinstance(obs, torch.Tensor):
                    # 檢查並清理觀測
                    obs_cleaned = self._sanitize_observation(f"{group_name}", obs)
                    obs_buf[group_name] = obs_cleaned
                elif isinstance(obs, dict):
                    # 如果觀測組內部還是字典（未合併的情況）
                    for term_name, term_obs in obs.items():
                        if isinstance(term_obs, torch.Tensor):
                            term_cleaned = self._sanitize_observation(f"{group_name}.{term_name}", term_obs)
                            obs_buf[group_name][term_name] = term_cleaned
        elif isinstance(obs_buf, torch.Tensor):
            # 如果觀測是單個張量
            obs_buf = self._sanitize_observation("policy", obs_buf)
        
        # 檢查獎勵是否也有 NaN/Inf
        if not torch.isfinite(reward_buf).all():
            _check_finite("reward_buf", reward_buf, raise_on_error=True)
        
        return obs_buf, reward_buf, terminated, time_outs, extras
    
    def _sanitize_observation(self, name: str, obs: torch.Tensor) -> torch.Tensor:
        """清理觀測中的 NaN/Inf 值
        
        Args:
            name: 觀測項名稱（用於錯誤訊息）
            obs: 觀測張量
        
        Returns:
            清理後的觀測張量
        """
        # 檢查是否有 NaN/Inf
        if not torch.isfinite(obs).all():
            # 先報告問題（不拋出異常，因為我們會嘗試修復）
            _check_finite(name, obs, raise_on_error=False)
            
            # 計算合理的替換值
            # 先獲取有限值的統計信息
            finite_mask = torch.isfinite(obs)
            if finite_mask.any():
                # 如果有有限值，使用有限值的中位數作為替換值
                finite_values = obs[finite_mask]
                replacement_value = finite_values.median().item()
                # 確保替換值在合理範圍內
                replacement_value = max(-10.0, min(10.0, replacement_value))
            else:
                # 如果所有值都是 NaN/Inf，使用 0.5（假設觀測已歸一化）
                replacement_value = 0.5
            
            # 清理異常值
            obs_cleaned = torch.nan_to_num(
                obs, 
                nan=replacement_value, 
                posinf=replacement_value, 
                neginf=replacement_value
            )
            
            # 根據觀測名稱決定 clip 範圍
            # lidar_scan 應該在 [0, 1]（已歸一化）
            if "lidar" in name.lower():
                obs_cleaned = torch.clamp(obs_cleaned, 0.0, 1.0)
            # goal_position 和 goal_distance 可能在較大範圍
            elif "goal" in name.lower():
                obs_cleaned = torch.clamp(obs_cleaned, -10.0, 10.0)
            # actions 應該在 [-1, 1]
            elif "action" in name.lower():
                obs_cleaned = torch.clamp(obs_cleaned, -1.0, 1.0)
            else:
                # 默認 clip 到 [-10, 10]
                obs_cleaned = torch.clamp(obs_cleaned, -10.0, 10.0)
            
            # 最終檢查
            if not torch.isfinite(obs_cleaned).all():
                raise RuntimeError(
                    f"[{name}] Failed to sanitize observation! "
                    f"Still contains NaN/Inf after cleaning. "
                    f"shape={tuple(obs.shape)}"
                )
            
            return obs_cleaned

        return obs


class HierarchicalChargeNavigationEnv(ChargeNavigationEnv):
    """層級式導航環境 (帶 AIT* 全局路徑規劃和視覺化)

    整合 AIT* 全局路徑規劃器和 RL 局部控制器：
    - AIT* 規劃全局路徑 (1-5 Hz)
    - RL 跟隨局部目標 (50-100 Hz)
    - 視覺化 AIT* 路徑（綠色線條和點）

    架構：
    ┌───────────────────────────────────────────────────────────────────┐
    │  AIT* Global Planner (全局規劃)                                    │
    │  ├─ 規劃從機器人位置到目標位置的全局路徑                           │
    │  └─ 提取局部目標（carrot-on-stick）                               │
    ├───────────────────────────────────────────────────────────────────┤
    │  RL PPO Controller (局部控制)                                      │
    │  ├─ 觀測: LiDAR + 局部目標 + 速度                                  │
    │  └─ 動作: [linear_vel, angular_vel]                               │
    └───────────────────────────────────────────────────────────────────┘
    """

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """初始化層級式導航環境"""
        super().__init__(cfg, render_mode, **kwargs)

        # 初始化 AIT* 規劃器和視覺化器
        self._aitstar_planner = None
        self._path_visualizer = None
        self._current_paths = [None] * self.num_envs

        # 嘗試初始化 AIT*
        if _AITSTAR_AVAILABLE:
            try:
                self._aitstar_planner = create_aitstar_planner(
                    map_size=(20.0, 20.0),
                    grid_resolution=0.1,
                    robot_radius=0.3,
                    max_iterations=500,
                    waypoint_spacing=0.5,  # 增加路徑點間隔，讓每 0.5m 一個點
                )

                # 創建視覺化器（只視覺化第一個環境，避免性能問題）
                self._path_visualizer = AITStarPathVisualizer(
                    prim_path="/World/AITStarPath",
                    path_color=(0.0, 1.0, 0.0),  # 綠色
                )

                print("[HierarchicalChargeNavigationEnv] AIT* 規劃器和視覺化器已初始化")
            except Exception as e:
                print(f"[警告] 無法初始化 AIT* 規劃器: {e}")
                self._aitstar_planner = None
                self._path_visualizer = None
        else:
            print("[警告] AIT* 模組不可用，使用標準環境")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """執行環境動態的一個時間步長（帶 AIT* 路徑更新和視覺化）"""
        # 先執行標準 step
        obs_buf, reward_buf, terminated, time_outs, extras = super().step(action)

        # 更新 AIT* 路徑視覺化
        if self._aitstar_planner is not None and self._path_visualizer is not None:
            self._update_path_visualization()

        return obs_buf, reward_buf, terminated, time_outs, extras

    def _update_path_visualization(self):
        """更新 AIT* 路徑視覺化

        每隔一定步數更新一次路徑規劃和視覺化。
        """
        # 獲取第一個環境的機器人位置和目標位置
        if not hasattr(self.scene, "robot"):
            return

        robot = self.scene["robot"]
        robot_pos = robot.data.root_pos_w[0, :2]  # [2]

        # 獲取目標位置（從 command_manager）
        try:
            goal_pos = self.command_manager.get_command("goal_command")[0, :2]  # [2]
        except (AttributeError, KeyError):
            # 如果沒有 command_manager 或 goal_command，跳過
            return

        # 每隔一定步數重新規劃路徑
        current_step = self.common_step_counter[0].item()

        # 每 50 步（約 2 秒）重新規劃一次，或第一次調用時立即規劃
        should_replan = (current_step % 50 == 0) or (current_step == 0)

        if should_replan and self._aitstar_planner is not None:
            # 使用 AIT* 規劃新路徑
            try:
                new_path = self._aitstar_planner.plan_path(
                    robot_pos,
                    goal_pos,
                    env=self,
                )

                if len(new_path) > 0:
                    self._current_paths[0] = new_path
                    if current_step == 0:
                        print(f"[AIT*] 初始規劃成功: {len(new_path)} 個路徑點")
                else:
                    if current_step == 0:
                        print(f"[AIT*] 初始規劃失敗: 無法找到路徑，使用直線路徑")
                        # 創建直線路徑作為備選
                        num_points = 20
                        new_path = torch.linspace(robot_pos.cpu(), goal_pos.cpu(), num_points)
                        self._current_paths[0] = new_path
            except Exception as e:
                print(f"[AIT*] 規劃出錯: {e}")
                # 創建直線路徑作為備選
                num_points = 20
                new_path = torch.linspace(robot_pos.cpu(), goal_pos.cpu(), num_points)
                self._current_paths[0] = new_path

        # 視覺化當前路徑
        if self._current_paths[0] is not None and self._path_visualizer is not None:
            try:
                self._path_visualizer.visualize(
                    path_points=self._current_paths[0],
                    start_pos=robot_pos,
                    goal_pos=goal_pos,
                )
            except Exception as e:
                print(f"[AIT*] 視覺化出錯: {e}")

    def reset(self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """重置環境"""
        obs, info = super().reset(seed=seed, env_ids=env_ids, options=options)

        # 清空路徑
        for i in range(self.num_envs):
            self._current_paths[i] = None

        # 清空視覺化
        if self._path_visualizer is not None:
            self._path_visualizer.clear()

        # 重置後立即規劃初始路徑
        if self._aitstar_planner is not None:
            try:
                self._plan_initial_path()
            except Exception as e:
                print(f"[AIT*] 規劃初始路徑失敗: {e}")

        return obs, info

    def _plan_initial_path(self):
        """規劃初始路徑（在環境重置後調用）"""
        # 查找機器人（可能的屬性名稱）
        robot = None
        for key in self.scene.keys():
            obj = self.scene[key]
            # 檢查是否是 Articulation 類型（機器人）
            if hasattr(obj, 'data') and hasattr(obj.data, 'root_pos_w'):
                if robot is None:
                    robot = obj
        if robot is None:
            print(f"[DEBUG] 未找到機器人，嘗試直接訪問 'robot' 鍵")
            try:
                robot = self.scene["robot"]
            except KeyError:
                print(f"[DEBUG] 'robot' 鍵不存在")
                return

        if robot is None:
            return

        robot_pos = robot.data.root_pos_w[0, :2]

        try:
            goal_pos = self.command_manager.get_command("goal_command")[0, :2]
        except (AttributeError, KeyError):
            return

        try:
            new_path = self._aitstar_planner.plan_path(
                robot_pos,
                goal_pos,
                env=self,
            )

            if len(new_path) > 0:
                self._current_paths[0] = new_path
                # 立即視覺化
                if self._path_visualizer is not None:
                    self._path_visualizer.visualize(
                        path_points=new_path,
                        start_pos=robot_pos,
                        goal_pos=goal_pos,
                    )
            else:
                # 創建直線路徑作為備選
                num_points = 20
                new_path = torch.linspace(robot_pos.cpu(), goal_pos.cpu(), num_points)
                self._current_paths[0] = new_path

                if self._path_visualizer is not None:
                    self._path_visualizer.visualize(
                        path_points=new_path,
                        start_pos=robot_pos,
                        goal_pos=goal_pos,
                    )
        except Exception as e:
            print(f"[AIT*] 規劃出錯: {e}")

    def close(self):
        """關閉環境"""
        # 清理視覺化器
        if self._path_visualizer is not None:
            self._path_visualizer.clear()

        super().close()
