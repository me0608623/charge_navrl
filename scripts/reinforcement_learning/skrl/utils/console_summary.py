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

        # 課程階段
        cur_stage = metrics.get("Curriculum / stage")
        cur_level = metrics.get("Curriculum / difficulty_level")
        if cur_level is not None and cur_level > 0:
            stage_str = f"OE-L{int(cur_level)}"
        elif cur_stage is not None:
            stage_str = f"B{int(cur_stage)}"
        else:
            stage_str = "N/A"

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
            f"  課程階段: Phase {stage_str}",
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

        # Agent 速度
        speed_avg = metrics.get("behavior/speed_avg")
        speed_near = metrics.get("behavior/speed_near_obstacle")
        if speed_avg is not None:
            summary_lines.append("")
            summary_lines.append("=== Agent 速度 ===")
            summary_lines.append(f"  平均速度: {speed_avg:.4f} m/s")
            if speed_near is not None:
                summary_lines.append(f"  近障礙速度: {speed_near:.4f} m/s")

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
            ("成功率", "perf/success_rate"),
            ("碰撞率", "perf/collision_rate"),
            ("超時率", "perf/timeout_rate"),
            ("總 Episode 數", "perf/total_episodes"),
            ("平均 Episode 長度", "perf/episode_length"),
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
            if key.startswith("Termination / "):
                component_name = key.replace("Termination / ", "")
                termination_components.append((component_name, key))

        if termination_components:
            summary_lines.append("")
            summary_lines.append("=== 終止原因分析 ===")
            for name, key in sorted(termination_components):
                value = metrics.get(key, 0.0)
                summary_lines.append(f"  {name}: {value:.4f}")

        # 獎勵分項 — NavRL-Ground v4 活躍項 + 預期走勢
        # 格式: (顯示名, WandB key, 預期走勢)
        reward_terms = [
            ("reaching_goal",   "Reward / reaching_goal",   "↑ 隨 SR 上升 (terminal w=500)"),
            ("goal_velocity",   "Reward / goal_velocity",   "↑ 朝目標速度 (r_vel, 核心驅動力)"),
            ("goal_progress",   "Reward / goal_progress",   "↑ 距離縮減 PBRS"),
            ("static_safety",   "Reward / static_safety",   "↑ LiDAR 72-bin log clearance"),
            ("dynamic_safety",  "Reward / dynamic_safety",  "↑ 動態障礙物安全 (Stage 6+)"),
            ("smoothness",      "Reward / smoothness",      "→ 微量負值 (控制平滑)"),
            ("collision_ground","Reward / collision_ground", "↓ 隨 CR 下降 (terminal w=-50)"),
        ]

        summary_lines.append("")
        summary_lines.append("=== 獎勵分項（每秒平均） ===")
        has_any = False
        for display_name, key, trend in reward_terms:
            value = metrics.get(key)
            if value is not None:
                summary_lines.append(f"  {display_name:30s} {value:+.6f}  {trend}")
                has_any = True
        if not has_any:
            summary_lines.append("  (尚無數據)")

        # 列出非零的其他 Reward 項（跳過 weight=0 的零值項）
        known_keys = {t[1] for t in reward_terms}
        skip_prefixes = ("Reward / Total", "Reward / Instantaneous")
        extra_rewards = []
        for key in sorted(metrics.keys()):
            if key.startswith("Reward / ") and key not in known_keys and not any(key.startswith(p) for p in skip_prefixes):
                value = metrics.get(key, 0.0)
                if value != 0.0:  # 跳過 weight=0 的項
                    extra_rewards.append((key.replace("Reward / ", ""), key))
        if extra_rewards:
            summary_lines.append("  --- 其他 (非零) ---")
            for name, key in extra_rewards:
                value = metrics.get(key, 0.0)
                summary_lines.append(f"  {name:30s} {value:+.6f}")

        # 課程學習指標
        cur_sr = metrics.get("Curriculum / success_rate")
        cur_cr = metrics.get("Curriculum / collision_rate")
        cur_to = metrics.get("Curriculum / timeout_rate")
        cur_ep = metrics.get("Curriculum / num_episodes")
        cur_sep = metrics.get("Curriculum / stage_episodes")
        cur_fill = metrics.get("Curriculum / window_fill")
        if cur_sr is not None:
            summary_lines.append("")
            summary_lines.append("=== 課程學習 ===")
            summary_lines.append(f"  階段: Phase {stage_str}")

            # 環境配置摘要：Goals / Obstacles / Walls
            cur_goals = metrics.get("Curriculum / num_goals")
            cur_obs_s = metrics.get("Curriculum / num_obstacles_static")
            cur_obs_d = metrics.get("Curriculum / num_obstacles_dynamic")
            cur_min_w = metrics.get("Curriculum / min_walls")
            cur_max_w = metrics.get("Curriculum / max_walls")
            if cur_goals is not None:
                goals_str = f"{int(cur_goals)}G"
                obs_str = f"{int(cur_obs_s or 0)}S+{int(cur_obs_d or 0)}D"
                walls_str = f"{int(cur_min_w or 0)}-{int(cur_max_w or 0)}" if cur_min_w is not None else "N/A"
                summary_lines.append(f"  配置: {goals_str} goals / {obs_str} obstacles / {walls_str} walls")

            # 升級差距
            sr_gap = metrics.get("Curriculum / sr_gap")
            cr_gap = metrics.get("Curriculum / cr_gap")
            to_gap = metrics.get("Curriculum / to_gap")
            up_sr_t = metrics.get("Curriculum / upgrade_sr_target")
            up_cr_t = metrics.get("Curriculum / upgrade_cr_target")
            up_to_t = metrics.get("Curriculum / upgrade_to_target")
            pass_count = metrics.get("Curriculum / upgrade_pass_count", 0)
            if up_sr_t is not None:
                def _flag(val, target, higher_is_better=True):
                    if higher_is_better:
                        return "OK" if val >= target else f"{val-target:+.2%}"
                    else:
                        return "OK" if val <= target else f"{val-target:+.2%}"
                sr_s = _flag(cur_sr, up_sr_t, True)
                cr_s = _flag(cur_cr, up_cr_t, False) if up_cr_t is not None and up_cr_t < 1.0 else "--"
                to_s = _flag(cur_to, up_to_t, False)
                summary_lines.append(
                    f"  目前:   SR={cur_sr:.2%}  CR={cur_cr:.2%}  TO={cur_to:.2%}"
                )
                summary_lines.append(
                    f"  門檻:   SR>{up_sr_t:.0%}({sr_s})  "
                    f"CR<{up_cr_t:.0%}({cr_s})  "
                    f"TO<{up_to_t:.0%}({to_s})  "
                    f"pass: {int(pass_count)}/5"
                )
            else:
                summary_lines.append(f"  成功率: {cur_sr:.4f}  碰撞率: {cur_cr:.4f}  超時率: {cur_to:.4f}")

            if cur_ep is not None:
                summary_lines.append(f"  總 episodes: {int(cur_ep)}  階段 episodes: {int(cur_sep or 0)}  窗口填充: {cur_fill:.2f}")

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
