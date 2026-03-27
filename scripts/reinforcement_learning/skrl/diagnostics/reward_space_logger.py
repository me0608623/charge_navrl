"""Reward Space Logger — 最小侵入式 episode-level 統計

記錄每個 episode 的關鍵 reward space 指標：
- lidar_min: LiDAR 最小距離
- d_front: 前方扇區平均距離
- collision_flag: 是否觸發碰撞 (lidar_min <= 0.45m)
- front_block_flag: 是否前方堵塞 (d_front < 1.2m)

使用方式:
    from diagnostics.reward_space_logger import RewardSpaceLogger
    logger = RewardSpaceLogger(env, output_dir=logs_dir)
    # 每步:
    logger.step(env_ids, lidar_72)
    # episode 結束:
    logger.finalize_episodes(reset_env_ids, infos)
    # rollout 結束:
    metrics = logger.get_rollout_metrics()
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RewardSpaceLogger:
    """Reward Space 關鍵指標收集器。

    為每個 env 維護 episode 累積統計，episode 結束時輸出到 CSV。
    同時提供 rollout-level 聚合指標給 WandB。
    """

    # LiDAR 配置
    NUM_LIDAR_BINS = 72
    LIDAR_FOV_DEG = 360.0
    BIN_SIZE_DEG = LIDAR_FOV_DEG / NUM_LIDAR_BINS  # 5°
    CENTER_BIN = NUM_LIDAR_BINS // 2  # 36 (正前方)

    # Flag 閾檻
    COLLISION_THRESHOLD = 0.45  # m
    FRONT_WARN_DIST = 1.2  # m

    # Front arc 配置
    FRONT_HALF_ANGLE_DEG = 30.0  # ±30°
    FRONT_NEAREST_K = 5  # 取前方最近 5 bins 平均

    def __init__(self, env: ManagerBasedRLEnv, output_dir: str | None = None):
        """初始化 logger。

        Args:
            env: Isaac Lab 環境 (unwrapped)
            output_dir: CSV 輸出目錄 (默認: logs/reward_space/)
        """
        self._env = env.unwrapped if hasattr(env, "unwrapped") else env
        self._num_envs = self._env.num_envs
        self._device = self._env.device

        # 設置輸出目錄
        if output_dir is None:
            output_dir = "logs/reward_space"
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # CSV 路徑
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = self._output_dir / f"reward_space_episodes_{timestamp}.csv"

        # CSV 列名
        self._csv_columns = [
            "episode_id",
            "env_id",
            "num_steps",
            "lidar_min_mean",
            "lidar_min_min",
            "d_front_mean",
            "collision_count",
            "front_block_count",
            "collision_trigger_ratio",
            "front_block_trigger_ratio",
            "success",
            "collision_done",
            "timeout",
        ]

        # Per-env episode 累積統計
        self._episode_id = 0
        self._env_episode_ids = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._step_counts = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._lidar_min_sum = torch.zeros(self._num_envs, device=self._device)
        self._lidar_min_min = torch.full((self._num_envs,), float("inf"), device=self._device)
        self._d_front_sum = torch.zeros(self._num_envs, device=self._device)
        self._collision_counts = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self._front_block_counts = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        # Front arc 預計算索引
        half_bins = int(self.FRONT_HALF_ANGLE_DEG / self.BIN_SIZE_DEG)  # 6
        self._front_start = max(0, self.CENTER_BIN - half_bins)  # 30
        self._front_end = min(self.NUM_LIDAR_BINS, self.CENTER_BIN + half_bins)  # 42
        self._front_num_bins = self._front_end - self._front_start  # 12

        # Rollout-level 累計 (用於 WandB)
        self._rollout_episodes = 0
        self._rollout_lidar_min_sum = 0.0
        self._rollout_d_front_sum = 0.0
        self._rollout_collision_triggers = 0
        self._rollout_front_block_triggers = 0
        self._rollout_total_steps = 0

        # CSV 檔案句柄
        self._csv_file = None
        self._csv_writer = None

        # 首次寫入標題
        self._init_csv()

    def _init_csv(self):
        """初始化 CSV 檔案並寫入標題。"""
        try:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_columns, extrasaction="ignore"
            )
            self._csv_writer.writeheader()
            self._csv_file.flush()
            print(f"[RewardSpaceLogger] CSV: {self._csv_path}")
        except Exception as e:
            print(f"[RewardSpaceLogger] CSV 初始化失敗: {e}")

    def step(
        self,
        env_ids: torch.Tensor,
        lidar_72: torch.Tensor,
    ):
        """每步更新。快速 GPU 操作。

        Args:
            env_ids: 本次更新的 env 索引 [M]
            lidar_72: LiDAR 72-bin 距離 [N, 72] 或 [B, N, 72]
        """
        # 移除多餘的 batch 維度 (如果來自 unsqueeze(0))
        if lidar_72.dim() == 3 and lidar_72.shape[0] == 1:
            lidar_72 = lidar_72.squeeze(0)

        # 驗證輸入形狀
        if lidar_72.dim() != 2:
            raise ValueError(f"lidar_72 必須是 2D tensor [N, 72]，實際: {lidar_72.shape}")
        if lidar_72.shape[1] != self.NUM_LIDAR_BINS:
            raise ValueError(f"lidar_72 最後維度必須是 {self.NUM_LIDAR_BINS}，實際: {lidar_72.shape[1]}")

        # 處理 lidar_72 形狀
        if lidar_72.shape[0] == self._num_envs:
            # 全部 env
            active_lidar = lidar_72
            active_ids = torch.arange(self._num_envs, device=self._device)
        else:
            # 部分 env (env_ids)
            active_lidar = lidar_72
            active_ids = env_ids

        if active_ids.numel() == 0:
            return

        # 1. 計算 lidar_min (全局最小) — 對 LiDAR bins 維度取 min
        lidar_min = active_lidar.amin(dim=-1)  # [M]

        # 2. 計算 d_front (前方扇區最近 5 bins 平均)
        front_sector = active_lidar[:, self._front_start : self._front_end]  # [M, 12]
        k = min(self.FRONT_NEAREST_K, front_sector.shape[1])
        d_front = torch.topk(front_sector, k=k, dim=-1, largest=False).values.mean(dim=-1)  # [M]

        # 3. 計算 flags
        collision_flag = lidar_min <= self.COLLISION_THRESHOLD  # [M] bool
        front_block_flag = d_front < self.FRONT_WARN_DIST  # [M] bool

        # 4. 累積到 per-env 統計
        ids = active_ids.long()
        self._step_counts[ids] += 1
        self._lidar_min_sum[ids] += lidar_min
        self._lidar_min_min[ids] = torch.minimum(self._lidar_min_min[ids], lidar_min)
        self._d_front_sum[ids] += d_front
        self._collision_counts[ids] += collision_flag.long()
        self._front_block_counts[ids] += front_block_flag.long()

    def finalize_episodes(
        self,
        reset_env_ids: torch.Tensor,
        infos: dict | None = None,
    ):
        """Episode 結束時輸出 summary rows。

        Args:
            reset_env_ids: 完成 episode 的 env 索引 [M]
            infos:環境 infos dict (可選，用於提取 outcome)
        """
        if reset_env_ids.numel() == 0:
            return

        ids = reset_env_ids.long()
        num_episodes = ids.numel()

        # 準備批量寫入的 rows
        rows = []
        for i in range(num_episodes):
            env_id = ids[i].item()
            episode_id = self._env_episode_ids[env_id].item()
            steps = self._step_counts[env_id].item()

            if steps == 0:
                continue  # 跳過空 episode

            lidar_min_mean = (self._lidar_min_sum[env_id] / steps).item()
            lidar_min_min = self._lidar_min_min[env_id].item()
            d_front_mean = (self._d_front_sum[env_id] / steps).item()
            collision_count = self._collision_counts[env_id].item()
            front_block_count = self._front_block_counts[env_id].item()

            row = {
                "episode_id": episode_id,
                "env_id": env_id,
                "num_steps": steps,
                "lidar_min_mean": round(lidar_min_mean, 4),
                "lidar_min_min": round(lidar_min_min, 4),
                "d_front_mean": round(d_front_mean, 4),
                "collision_count": collision_count,
                "front_block_count": front_block_count,
                "collision_trigger_ratio": round(collision_count / steps, 4) if steps > 0 else 0.0,
                "front_block_trigger_ratio": round(front_block_count / steps, 4) if steps > 0 else 0.0,
                "success": None,  # 從 infos 填充
                "collision_done": None,
                "timeout": None,
            }

            # 嘗試從 infos 提取 outcome
            if infos is not None:
                try:
                    log_data = infos.get(self._environment_info_key(), {})
                    if log_data:
                        # 檢查 termination 類型
                        for key in log_data:
                            if "success" in key.lower() or "goal_reached" in key.lower():
                                row["success"] = bool(log_data[key])
                            elif "collision" in key.lower() and key.endswith("/count"):
                                row["collision_done"] = bool(log_data[key])
                            elif "timeout" in key.lower() and key.endswith("/count"):
                                row["timeout"] = bool(log_data[key])
                except Exception:
                    pass

            rows.append(row)

            # 累加到 rollout 統計
            self._rollout_episodes += 1
            self._rollout_lidar_min_sum += lidar_min_mean
            self._rollout_d_front_sum += d_front_mean
            self._rollout_collision_triggers += collision_count
            self._rollout_front_block_triggers += front_block_count
            self._rollout_total_steps += steps

            # 重設該 env 的統計
            self._reset_env(env_id)
            self._env_episode_ids[env_id] += 1

        # 批量寫入 CSV
        if rows and self._csv_writer is not None:
            try:
                self._csv_writer.writerows(rows)
                self._csv_file.flush()
            except Exception as e:
                print(f"[RewardSpaceLogger] CSV 寫入失敗: {e}")

    def get_rollout_metrics(self) -> dict:
        """返回 rollout-level 聚合指標 (給 WandB)。

        Returns:
            dict: 聚合指標
        """
        if self._rollout_episodes == 0:
            return {
                "reward_space/lidar_min_mean": 0.0,
                "reward_space/d_front_mean": 0.0,
                "reward_space/collision_trigger_ratio": 0.0,
                "reward_space/front_block_trigger_ratio": 0.0,
            }

        total_steps = max(self._rollout_total_steps, 1)

        metrics = {
            "reward_space/lidar_min_mean": self._rollout_lidar_min_sum / self._rollout_episodes,
            "reward_space/d_front_mean": self._rollout_d_front_sum / self._rollout_episodes,
            "reward_space/collision_trigger_ratio": self._rollout_collision_triggers / total_steps,
            "reward_space/front_block_trigger_ratio": self._rollout_front_block_triggers / total_steps,
        }

        # 重置 rollout 統計
        self._rollout_episodes = 0
        self._rollout_lidar_min_sum = 0.0
        self._rollout_d_front_sum = 0.0
        self._rollout_collision_triggers = 0
        self._rollout_front_block_triggers = 0
        self._rollout_total_steps = 0

        return metrics

    def _reset_env(self, env_id: int):
        """重置單個 env 的統計。"""
        idx = torch.tensor(env_id, device=self._device)
        self._step_counts[idx] = 0
        self._lidar_min_sum[idx] = 0.0
        self._lidar_min_min[idx] = float("inf")
        self._d_front_sum[idx] = 0.0
        self._collision_counts[idx] = 0
        self._front_block_counts[idx] = 0

    def _environment_info_key(self) -> str:
        """獲取 environment_info 的 key。"""
        # 嘗試從 env 獲取
        if hasattr(self._env, "environment_info"):
            return self._env.environment_info
        return "/__log__"

    def flush(self):
        """刷新 CSV 並關閉檔案。"""
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
            except Exception:
                pass

    def close(self):
        """關閉 CSV 檔案。"""
        if self._csv_file is not None:
            try:
                self._csv_file.close()
                print(f"[RewardSpaceLogger] Closed: {self._csv_path}")
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

    def __del__(self):
        """析構時確保檔案關閉。"""
        self.close()
