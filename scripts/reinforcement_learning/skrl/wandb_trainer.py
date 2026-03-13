# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
WandB-enabled SKRL Trainer for IsaacLab.

扩展 SKRL SequentialTrainer，添加 WandB logging 支持。
"""

import csv
import os
import sys
import tqdm
from typing import Union, Optional

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.trainers.torch import SequentialTrainer

# Import console summary logger
from console_summary import ConsoleSummaryLogger


class WandBSequentialTrainer(SequentialTrainer):
    """SKRL SequentialTrainer with WandB logging and console summary.

    在原有 SKRL SequentialTrainer 基础上添加：
    - 自动记录 tracking_data 到 WandB
    - 记录自定义指标（FPS、环境统计等）
    - 输出可复制的终端训练摘要
    - TrainingDebugLogger 集成（action/ + robot/ 指标）
    - 保持与原有 Trainer 完全兼容
    """

    def __init__(
        self,
        env: Union[Wrapper, MultiAgentEnvWrapper],
        agents: Union[Agent, tuple, list],
        agents_scope: list = None,
        cfg: dict = None,
        wandb_run=None,
        console_summary_logger: Optional[ConsoleSummaryLogger] = None,
        debug_logger=None,
        log_dir: Optional[str] = None,
    ):
        """Initialize WandB-enabled trainer.

        Args:
            env: Environment to train on
            agents: Agent(s) to train
            agents_scope: Number of environments for each agent
            cfg: Trainer configuration
            wandb_run: WandB run instance (optional)
            console_summary_logger: ConsoleSummaryLogger instance (optional)
            debug_logger: TrainingDebugLogger instance (optional)
            log_dir: Directory for local CSV metric log (optional)
        """
        super().__init__(env, agents, agents_scope, cfg)
        self.wandb_run = wandb_run
        self.console_summary_logger = console_summary_logger
        self.debug_logger = debug_logger
        self._initial_timestamp = None
        self._last_log_time = None

        # ── Local CSV logger ──
        self._csv_path: Optional[str] = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_columns: Optional[list] = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            self._csv_path = os.path.join(log_dir, "debug_metrics.csv")
            print(f"[INFO] Local debug metrics CSV: {self._csv_path}")

        # 保存最后一次完整的tracking_data状态（因为record_transition会清空tracking_data）
        self._last_tracking_data = {}
        self._last_loss_data = {}

        # Running counters for task metrics
        self._episode_count = 0
        self._goal_reached_count = 0.0
        self._collision_count = 0.0
        self._timeout_count = 0.0
        self._episode_length_sum = 0.0

        # Self-managed episode length counter (fixes Isaac Lab reset-inside-step bug)
        self._ep_len_counter: Optional[torch.Tensor] = None

        # Fix 5: Early stopping — 監控 lin_vel_mean，連續凍結時停止訓練
        self._frozen_steps = 0
        self._frozen_threshold_vel = 0.02     # 低於此速度視為凍結
        self._frozen_max_steps = 5000         # 連續凍結超過此步數停止訓練
        self._frozen_check_interval = 128     # 每 N 步檢查一次

        # Hook agent.track_data()来捕获loss数据
        self._hook_agent_track_data()

    def _check_frozen_agent(self, timestep: int, next_states: torch.Tensor) -> bool:
        """Fix 5: 檢查 agent 是否凍結（持續不動）。

        透過監控觀測中的速度資訊，偵測 agent 是否學會了「不動」策略。
        由於觀測是高維且結構各異，改用 env 的實際速度數據。

        Args:
            timestep: 當前 timestep
            next_states: 觀測張量（用於 fallback）

        Returns:
            True = 應該停止訓練
        """
        if (timestep + 1) % self._frozen_check_interval != 0:
            return False

        # 從 env 取得實際速度
        try:
            env = self.env
            while hasattr(env, '_env'):
                env = env._env
            if hasattr(env, 'scene') and hasattr(env.scene, '__getitem__'):
                robot = env.scene["robot"]
                vel_xy = robot.data.root_lin_vel_w[:, :2]
                mean_speed = torch.norm(vel_xy, dim=-1).mean().item()
            else:
                # Fallback: 用 next_states 的方差估算活躍度
                mean_speed = next_states.std().item()
        except Exception:
            return False

        if mean_speed < self._frozen_threshold_vel:
            self._frozen_steps += self._frozen_check_interval
        else:
            self._frozen_steps = 0

        if self._frozen_steps >= self._frozen_max_steps:
            print(f"\n{'!'*70}")
            print(f"[EARLY STOP] Agent frozen detected!")
            print(f"  mean_speed={mean_speed:.6f} < threshold={self._frozen_threshold_vel}")
            print(f"  Frozen for {self._frozen_steps} consecutive steps (max={self._frozen_max_steps})")
            print(f"  Stopping training at timestep {timestep}")
            print(f"{'!'*70}\n")
            return True

        return False

    def _hook_agent_track_data(self) -> None:
        """Hook agent.track_data()方法来拦截并记录loss数据.

        SKRL的agent.track_data()用于记录数据到tracking_data。
        我们拦截这个方法来保存loss相关数据，因为tracking_data会被清空。
        """
        agents = self.agents if self.num_simultaneous_agents != 1 else [self.agents]

        for agent in agents:
            if hasattr(agent, 'track_data'):
                # 保存原始的track_data方法
                original_track_data = agent.track_data

                # 创建包装函数
                def hooked_track_data(tag: str, value: float) -> None:
                    # 先调用原始方法
                    original_track_data(tag, value)

                    # 如果是loss相关数据，保存到_last_loss_data
                    if tag.startswith("Loss /") or tag.startswith("Policy /") or tag.startswith("Learning /"):
                        if tag not in self._last_loss_data:
                            self._last_loss_data[tag] = []
                        self._last_loss_data[tag].append(value)

                        # 限制保存的历史数据量（只保留最新的100个）
                        if len(self._last_loss_data[tag]) > 100:
                            self._last_loss_data[tag] = self._last_loss_data[tag][-100:]

                # 替换agent的track_data方法
                agent.track_data = hooked_track_data

    def _capture_tracking_data(self, single_agent: bool = False) -> dict:
        """捕获当前tracking_data的快照，并更新_last_tracking_data。

        Args:
            single_agent: 是否为单agent训练

        Returns:
            tracking_data的深拷贝字典（包含_last_tracking_data中仍然有效的数据）
        """
        snapshot = {}

        if single_agent:
            agents = [self.agents]
        else:
            agents = self.agents

        for agent in agents:
            if hasattr(agent, 'tracking_data'):
                # 深拷贝当前tracking_data
                for key, values in agent.tracking_data.items():
                    if key not in snapshot:
                        snapshot[key] = []
                    snapshot[key].extend(list(values))

                    # 更新_last_tracking_data（保存所有看到过的数据）
                    self._last_tracking_data[key] = list(values)

                    # 保存loss相关数据
                    if key.startswith("Loss /") or key.startswith("Policy /") or key.startswith("Learning /"):
                        self._last_loss_data[key] = list(values)

        # 合并_last_tracking_data：将_last_tracking_data中的数据合并到snapshot
        # 这样即使当前tracking_data被清空，我们仍然有之前的数据
        for key, values in self._last_tracking_data.items():
            if key not in snapshot or not snapshot[key]:
                snapshot[key] = list(values)

        return snapshot

    def train(self) -> None:
        """Train the agents sequentially with WandB logging.

        扩展原训练循环，在每个 timestep 后将 tracking_data 记录到 WandB。
        """
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # non-simultaneous agents
        if self.num_simultaneous_agents == 1:
            # single-agent
            if self.env.num_agents == 1:
                return self._single_agent_train_with_wandb()
            # multi-agent
            else:
                return self._multi_agent_train_with_wandb()

        # simultaneous agents
        # reset env
        states, infos = self.env.reset()

        self._initial_timestamp = None
        self._last_log_time = None

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = torch.vstack(
                    [
                        agent.act(states[scope[0] : scope[1]], timestep=timestep, timesteps=self.timesteps)[0]
                        for agent, scope in zip(self.agents, self.agents_scope)
                    ]
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # 保存tracking_data（在record_transition清空之前）
                # PPO的record_transition会在内部调用post_interaction清空tracking_data
                # 所以我们需要在调用record_transition之前捕获
                tracking_data_snapshot = self._capture_tracking_data()

                # record the environments' transitions
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                    # record_transition之后立即检查是否有新的loss数据
                    # PPO会在record_transition中调用update()记录loss
                    # 然后调用post_interaction()清空tracking_data
                    # 所以我们需要在record_transition之后立即捕获
                    if hasattr(agent, 'tracking_data'):
                        for key, values in agent.tracking_data.items():
                            if key.startswith("Loss /") or key.startswith("Policy /") or key.startswith("Learning /"):
                                if values:  # 如果有数据，保存到last_loss_data
                                    self._last_loss_data[key] = list(values)
                                    # 同时添加到tracking_data_snapshot
                                    tracking_data_snapshot[key] = list(values)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())
                        elif isinstance(v, (int, float)):
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", float(v))

            # update running task metrics
            self._update_task_metrics(infos, terminated, truncated)

            # post-interaction
            for agent in self.agents:
                agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # WandB logging（使用快照数据）
            self._log_to_wandb(timestep, tracking_data_snapshot=tracking_data_snapshot)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

    def _single_agent_train_with_wandb(self) -> None:
        """Single-agent training loop with WandB logging + debug logger."""
        agent = self.agents
        agent.set_running_mode("train")

        # reset env
        states, infos = self.env.reset()

        self._initial_timestamp = None
        self._last_log_time = None

        # Determine rollout length for debug logger flush interval
        _rollouts = getattr(agent, '_rollouts', 128)

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            # pre-interaction
            agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = agent.act(states, timestep=timestep, timesteps=self.timesteps)[0]

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # ── Debug logger: per-step collection ──
                if self.debug_logger is not None:
                    self.debug_logger.step(actions, rewards, terminated, truncated, infos)

                # 保存tracking_data（在record_transition清空之前）
                tracking_data_snapshot = self._capture_tracking_data(single_agent=True)

                # record the environments' transitions
                agent.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                # record_transition之后立即检查是否有新的loss数据
                if hasattr(agent, 'tracking_data'):
                    for key, values in agent.tracking_data.items():
                        if key.startswith("Loss /") or key.startswith("Policy /") or key.startswith("Learning /"):
                            if values:
                                self._last_loss_data[key] = list(values)
                                tracking_data_snapshot[key] = list(values)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            agent.track_data(f"Info / {k}", v.item())
                        elif isinstance(v, (int, float)):
                            agent.track_data(f"Info / {k}", float(v))

            # post-interaction
            # PPO 的 _update 在 post_interaction 中被调用
            agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # post_interaction 後立即捕獲新的 loss 數據
            if hasattr(agent, 'tracking_data'):
                for key, values in agent.tracking_data.items():
                    if key.startswith("Loss /") or key.startswith("Policy /") or key.startswith("Learning /"):
                        if values:
                            self._last_loss_data[key] = list(values)
                            tracking_data_snapshot[key] = list(values)

            # update running task metrics
            self._update_task_metrics(infos, terminated, truncated)

            # ── Flush all metrics every rollout (NOT every step) ──
            # This is the ONLY place wandb.log / CSV write occurs, avoiding
            # per-step I/O overhead and reducing CUDA sync frequency.
            if (timestep + 1) % _rollouts == 0:
                # Debug logger: flush aggregated GPU metrics
                if self.debug_logger is not None:
                    # Scatter plot (built BEFORE get_and_reset clears buffers)
                    if self.wandb_run is not None:
                        scatter_table = self.debug_logger.get_accel_scatter_table()
                        if scatter_table is not None:
                            try:
                                import wandb
                                wandb.log(
                                    {"action/accel_scatter": wandb.plot.scatter(
                                        scatter_table,
                                        x="linear_acceleration",
                                        y="angular_acceleration",
                                        title="speed distributed",
                                    )},
                                    step=timestep,
                                )
                            except ImportError:
                                pass

                    debug_metrics = self.debug_logger.get_and_reset()
                    tracking_data_snapshot.update(
                        {k: [v] for k, v in debug_metrics.items()}
                    )

                # WandB + CSV + console logging (flush-only, not every step)
                self._log_to_wandb(timestep, tracking_data_snapshot=tracking_data_snapshot, single_agent=True)

            # Fix 5: Early stopping check
            if self._check_frozen_agent(timestep, next_states):
                break

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

    def _update_task_metrics(self, infos: dict, terminated, truncated) -> None:
        """Update running task metrics from infos["log"] data.

        Accumulates episode counts and termination reason counts to compute
        running averages for success_rate, collision_rate, timeout_rate.

        Args:
            infos: Environment infos dict containing "log" key
            terminated: Terminated tensor from env.step
            truncated: Truncated tensor from env.step
        """
        log_data = infos.get(self.environment_info, {})

        # Count how many envs finished this step
        dones = terminated.view(-1) | truncated.view(-1)

        # Self-managed episode length counter (GPU, no CPU transfer per step)
        num_envs = terminated.shape[0]
        if self._ep_len_counter is None:
            self._ep_len_counter = torch.ones(num_envs, device=terminated.device, dtype=torch.long)
        else:
            self._ep_len_counter += 1

        # Single GPU→CPU transfer: done count + episode lengths
        num_done_t = dones.sum()
        if num_done_t.item() == 0:
            return
        num_done = int(num_done_t.item())

        self._episode_count += num_done

        # Read from Episode_Termination/* (infos values are often already scalar)
        for k, v in log_data.items():
            if k.startswith("Episode_Termination/"):
                term_name = k.split("/", 1)[1]
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                count = val * num_done
                if "goal" in term_name.lower():
                    self._goal_reached_count += count
                elif "collision" in term_name.lower():
                    self._collision_count += count
                elif "time" in term_name.lower():
                    self._timeout_count += count

        # Record episode lengths from self-managed counter, then reset done envs
        done_lengths = self._ep_len_counter[dones].float()
        if done_lengths.numel() > 0:
            self._episode_length_sum += done_lengths.sum().cpu().item()
        self._ep_len_counter[dones] = 0

    def _get_task_metrics(self) -> dict:
        """Return current running task metrics.

        Returns:
            Dict of task metric names to values.
        """
        metrics = {}
        if self._episode_count > 0:
            metrics["nav/success_rate"] = self._goal_reached_count / self._episode_count
            metrics["nav/collision_rate"] = self._collision_count / self._episode_count
            metrics["nav/timeout_rate"] = self._timeout_count / self._episode_count
            metrics["nav/total_episodes"] = self._episode_count
            metrics["nav/episode_length_mean"] = self._episode_length_sum / self._episode_count
        return metrics

    def _multi_agent_train_with_wandb(self) -> None:
        """Multi-agent training loop with WandB logging."""
        for agent in self.agents:
            agent.set_running_mode("train")

        # reset env
        states, infos = self.env.reset()

        self._initial_timestamp = None
        self._last_log_time = None

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):
            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = torch.vstack(
                    [
                        agent.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                        for agent in self.agents
                    ]
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)

                # render scene
                if not self.headless:
                    self.env.render()

                # 保存tracking_data（在record_transition清空之前）
                tracking_data_snapshot = self._capture_tracking_data()

                # record the environments' transitions
                for agent in self.agents:
                    agent.record_transition(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        terminated=terminated,
                        truncated=truncated,
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                    # record_transition之后立即检查是否有新的loss数据
                    if hasattr(agent, 'tracking_data'):
                        for key, values in agent.tracking_data.items():
                            if key.startswith("Loss /") or key.startswith("Policy /") or key.startswith("Learning /"):
                                if values:
                                    self._last_loss_data[key] = list(values)
                                    tracking_data_snapshot[key] = list(values)

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())
                        elif isinstance(v, (int, float)):
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", float(v))

            # update running task metrics
            self._update_task_metrics(infos, terminated, truncated)

            # post-interaction
            # PPO已经在record_transition中调用了post_interaction
            # 这里不再调用，避免重复
            # 但为了兼容其他agent类型，保留调用
            # for agent in self.agents:
            #     agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # WandB logging（使用快照数据）
            self._log_to_wandb(timestep, tracking_data_snapshot=tracking_data_snapshot)

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

    def _log_to_wandb(self, timestep: int, tracking_data_snapshot: dict = None, single_agent: bool = False) -> None:
        """Log tracking data to WandB and print console summary.

        从 tracking_data_snapshot 提取指标并记录到 WandB。
        同时输出终端摘要到控制台。

        Args:
            timestep: Current timestep
            tracking_data_snapshot: tracking_data的快照（在post_interaction清空之前捕获）
            single_agent: Whether this is single-agent training
        """
        # 如果没有提供快照，尝试从agent获取（兼容旧代码）
        if tracking_data_snapshot is None:
            agents = [self.agents] if single_agent else self.agents
            tracking_data_snapshot = {}
            for agent in agents:
                if hasattr(agent, 'tracking_data'):
                    for key, values in agent.tracking_data.items():
                        if key not in tracking_data_snapshot:
                            tracking_data_snapshot[key] = []
                        tracking_data_snapshot[key].extend(list(values))

        # 合并最后一次的loss数据（因为PPO的record_transition会清空tracking_data）
        # 这样即使当前tracking_data_snapshot中没有loss数据，也能显示上一次的loss
        if self._last_loss_data:
            for key, values in self._last_loss_data.items():
                if key not in tracking_data_snapshot or not tracking_data_snapshot[key]:
                    tracking_data_snapshot[key] = list(values)

        # 从快照收集metrics
        metrics = {}
        import numpy as np
        for key, values in tracking_data_snapshot.items():
            if values:  # 只记录有数据的key
                metrics[key] = np.mean(values)

        # 计算FPS
        import time
        if self._initial_timestamp is None:
            self._initial_timestamp = time.time()
        current_time = time.time()
        elapsed_time = current_time - self._initial_timestamp

        if elapsed_time > 0:
            fps = timestep / elapsed_time
            metrics["train/fps"] = fps
            metrics["train/elapsed_time"] = elapsed_time

        # 记录timestep作为step
        metrics["train/timestep"] = timestep

        # Merge running task metrics
        task_metrics = self._get_task_metrics()
        metrics.update(task_metrics)

        # Inject task metrics into snapshot so console summary can display them
        for k, v in task_metrics.items():
            tracking_data_snapshot[k] = [v]

        # 记录到WandB
        if self.wandb_run is not None:
            try:
                import wandb
                if metrics:
                    wandb.log(metrics, step=timestep)
            except ImportError:
                pass

        # ── 写入本地 CSV ──
        if self._csv_path is not None and metrics:
            self._write_csv_row(timestep, metrics)

        # 输出终端摘要
        if self.console_summary_logger is not None:
            self.console_summary_logger.print_summary(
                timestep=timestep,
                timesteps_total=self.timesteps,
                tracking_data=tracking_data_snapshot,
                loss_data=self._last_loss_data,
            )

    # ──────────────────────────────────────────────────────────────────────
    # Local CSV Metric Logger
    # ──────────────────────────────────────────────────────────────────────

    def _write_csv_row(self, timestep: int, metrics: dict) -> None:
        """Append one row of metrics to the local CSV file.

        On first call, writes header row with sorted column names.
        Subsequent calls append data rows. New columns discovered later
        are handled by rewriting the header (rare, only if metric set changes).

        Args:
            timestep: Current global timestep
            metrics: Flat dict of metric_name -> float value
        """
        # Build the row dict (timestep always first)
        row = {"timestep": timestep}
        row.update(metrics)

        # First call: create file and write header
        if self._csv_columns is None:
            self._csv_columns = sorted(row.keys())
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_columns, extrasaction="ignore"
            )
            self._csv_writer.writeheader()
            self._csv_file.flush()

        # Check if new columns appeared (happens when debug metrics first flush)
        current_keys = set(row.keys())
        if not current_keys.issubset(set(self._csv_columns)):
            # New columns discovered — rewrite with expanded header
            new_columns = sorted(current_keys | set(self._csv_columns))
            self._csv_columns = new_columns
            # Close old file, rewrite header by reading existing data
            self._csv_file.close()
            self._rewrite_csv_with_new_columns(new_columns)
            self._csv_file = open(self._csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=self._csv_columns, extrasaction="ignore"
            )

        # Write the row (missing columns become empty string)
        self._csv_writer.writerow(row)
        # Flush every row for crash safety
        self._csv_file.flush()

    def _rewrite_csv_with_new_columns(self, new_columns: list) -> None:
        """Rewrite existing CSV with expanded column set.

        Reads all existing rows, writes them back with the new header.
        Missing values in old rows become empty strings.
        """
        # Read existing data
        rows = []
        try:
            with open(self._csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception:
            pass

        # Rewrite with new columns
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=new_columns, extrasaction="ignore")
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    def __del__(self):
        """Close CSV file on cleanup."""
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
