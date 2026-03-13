# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Console Summary Logger for SKRL Training.

訓練過程中輸出結構化的終端摘要（中文），方便複製給 AI debug。
"""

import time
from datetime import datetime
from typing import Optional, Union

import numpy as np
import torch


class ConsoleSummaryLogger:
    """終端摘要記錄器，輸出結構化訓練資訊供 AI 調試使用。"""

    def __init__(
        self,
        print_every: int = 10000,
        task_name: str = "unknown",
        num_envs: int = 1,
        headless: bool = True,
        device: str = "cuda:0",
        seed: int = 42,
        algo: str = "PPO",
    ):
        self.print_every = print_every
        self.task_name = task_name
        self.num_envs = num_envs
        self.headless = headless
        self.device = device
        self.seed = seed
        self.algo = algo

        self.initial_timestamp = None
        self.last_print_timestep = 0
        self._first_print = True
        self._last_metrics = {}

    def should_print(self, timestep: int) -> bool:
        if timestep == 0:
            return True
        if timestep - self.last_print_timestep >= self.print_every:
            return True
        return False

    def print_summary(
        self,
        timestep: int,
        timesteps_total: int,
        tracking_data: dict,
        infos: dict = None,
        checkpoint_path: str = None,
        loss_data: dict = None,
    ) -> None:
        if not self.should_print(timestep):
            return

        self.last_print_timestep = timestep

        if self.initial_timestamp is None:
            self.initial_timestamp = time.time()

        current_time = time.time()
        elapsed_time = current_time - self.initial_timestamp
        if elapsed_time > 0:
            fps = timestep / elapsed_time
            steps_per_sec = fps * self.num_envs
        else:
            fps = 0.0
            steps_per_sec = 0.0

        metrics = self._extract_metrics_from_tracking_data(tracking_data)

        if loss_data:
            loss_metrics = self._extract_metrics_from_tracking_data(loss_data)
            for key, value in loss_metrics.items():
                if key.startswith("Loss /") or key.startswith("Policy /") or key.startswith("Learning /"):
                    metrics[key] = value

        if infos:
            info_metrics = self._extract_metrics_from_infos(infos)
            metrics.update(info_metrics)

        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_env_steps = timestep * self.num_envs

        summary_lines = [
            "",
            "=" * 60,
            "[TRAIN_SUMMARY]",
            f"  時間: {timestamp_str}",
            f"  任務: {self.task_name}",
            f"  環境數: {self.num_envs}",
            f"  無頭模式: {self.headless}",
            f"  演算法: {self.algo}",
            f"  裝置: {self.device}",
            f"  種子: {self.seed}",
            "",
            "=== 訓練進度 ===",
            f"  Trainer 總步數: {timesteps_total}",
            f"  Trainer 當前步: {timestep}",
            f"  總環境步數: {total_env_steps:,} (= {timestep} x {self.num_envs} envs)",
            f"  進度: {100.0 * timestep / timesteps_total:.2f}%",
            f"  預計剩餘時間: {self._format_time((timesteps_total - timestep) / fps if fps > 0 else 0)}",
            "",
            "=== 效能 ===",
            f"  FPS (trainer): {fps:.2f}",
            f"  環境步/秒: {steps_per_sec:.2f}",
            f"  已用時間: {self._format_time(elapsed_time)}",
            "",
            "=== Episode 指標 ===",
        ]

        # Episode 指標
        episode_metrics = [
            ("平均 Episode 獎勵", "Reward / Total reward (mean)"),
            ("平均 Episode 長度", "Episode / Total timesteps (mean)"),
        ]

        for metric_key, tracking_key in episode_metrics:
            value = metrics.get(tracking_key)
            if value is not None:
                summary_lines.append(f"  {metric_key}: {value:.4f}")
            else:
                summary_lines.append(f"  {metric_key}: N/A (尚無完整 episode)")

        # 即時獎勵
        inst_metrics = [
            ("即時獎勵 (平均)", "Reward / Instantaneous reward (mean)"),
            ("即時獎勵 (最大)", "Reward / Instantaneous reward (max)"),
            ("即時獎勵 (最小)", "Reward / Instantaneous reward (min)"),
        ]

        summary_lines.append("")
        summary_lines.append("=== 即時獎勵 (每步) ===")
        has_inst = False
        for metric_key, tracking_key in inst_metrics:
            value = metrics.get(tracking_key)
            if value is not None:
                summary_lines.append(f"  {metric_key}: {value:.6f}")
                has_inst = True
        if not has_inst:
            summary_lines.append("  (尚無數據)")

        # 任務指標
        task_metrics = [
            ("成功率", "nav/success_rate"),
            ("碰撞率", "nav/collision_rate"),
            ("超時率", "nav/timeout_rate"),
            ("總 Episode 數", "nav/total_episodes"),
            ("平均 Episode 長度", "nav/episode_length_mean"),
        ]

        has_task_metrics = False
        for metric_key, tracking_key in task_metrics:
            value = metrics.get(tracking_key)
            if value is not None:
                if not has_task_metrics:
                    summary_lines.append("")
                    summary_lines.append("=== 任務指標 ===")
                    has_task_metrics = True
                if metric_key == "總 Episode 數":
                    summary_lines.append(f"  {metric_key}: {int(value)}")
                else:
                    summary_lines.append(f"  {metric_key}: {value:.4f}")

        # 終止原因分析
        termination_components = []
        for key in metrics.keys():
            if key.startswith("Info / Episode_Termination/"):
                component_name = key.replace("Info / Episode_Termination/", "")
                termination_components.append((component_name, key))

        if termination_components:
            summary_lines.append("")
            summary_lines.append("=== 終止原因分析 ===")
            for name, key in sorted(termination_components):
                value = metrics.get(key, 0.0)
                summary_lines.append(f"  {name}: {value:.4f}")

        # 獎勵分項
        reward_components = []
        for key in metrics.keys():
            if key.startswith("Info / Episode_Reward/"):
                component_name = key.replace("Info / Episode_Reward/", "")
                reward_components.append((component_name, key))

        if reward_components:
            summary_lines.append("")
            summary_lines.append("=== 獎勵分項 ===")
            for component_name, tracking_key in sorted(reward_components):
                value = metrics.get(tracking_key, 0.0)
                summary_lines.append(f"  {component_name}: {value:.6f}")

        # Loss 指標
        summary_lines.append("")
        summary_lines.append("=== 損失函數 ===")

        loss_metrics_list = [
            ("策略損失", "Loss / Policy loss"),
            ("價值損失", "Loss / Value loss"),
            ("熵損失", "Loss / Entropy loss"),
            ("KL 散度", "Info / KL divergence"),
        ]

        for metric_key, tracking_key in loss_metrics_list:
            value = metrics.get(tracking_key)
            if value is not None:
                summary_lines.append(f"  {metric_key}: {value:.6f}")
            else:
                summary_lines.append(f"  {metric_key}: N/A")

        # 策略指標
        policy_metrics = [
            ("標準差", "Policy / Standard deviation"),
            ("學習率", "Learning / Learning rate"),
        ]

        has_policy = False
        for metric_key, tracking_key in policy_metrics:
            value = metrics.get(tracking_key)
            if value is not None:
                if not has_policy:
                    summary_lines.append("")
                    summary_lines.append("=== 策略指標 ===")
                    has_policy = True
                summary_lines.append(f"  {metric_key}: {value:.6f}")

        # 檢查點
        if checkpoint_path:
            summary_lines.append("")
            summary_lines.append("=== 檢查點 ===")
            summary_lines.append(f"  路徑: {checkpoint_path}")

        # 變化偵測
        if self._last_metrics:
            changed_keys = []
            for key, value in metrics.items():
                if key in self._last_metrics:
                    last_value = self._last_metrics[key]
                    if abs(value - last_value) > 1e-6:
                        changed_keys.append(f"{key}: {last_value:.4f} -> {value:.4f}")

            if changed_keys:
                summary_lines.append("")
                summary_lines.append("=== 變化指標 (與上次比較) ===")
                for change in changed_keys[:10]:
                    summary_lines.append(f"  {change}")
                if len(changed_keys) > 10:
                    summary_lines.append(f"  ... 另有 {len(changed_keys) - 10} 項變化")

        self._last_metrics = metrics.copy()
        self._first_print = False

        summary_lines.append("[/TRAIN_SUMMARY]")
        summary_lines.append("=" * 60)
        summary_lines.append("")

        print("\n".join(summary_lines))

    def _extract_metrics_from_tracking_data(self, tracking_data: dict) -> dict:
        metrics = {}
        if not tracking_data:
            return metrics

        for key, values in tracking_data.items():
            if values and len(values) > 0:
                try:
                    metrics[key] = np.mean(values)
                except:
                    pass

        return metrics

    def _extract_metrics_from_infos(self, infos: dict) -> dict:
        metrics = {}
        if not infos:
            return metrics
        return metrics

    def _format_time(self, seconds: float) -> str:
        if seconds <= 0:
            return "00:00:00"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
