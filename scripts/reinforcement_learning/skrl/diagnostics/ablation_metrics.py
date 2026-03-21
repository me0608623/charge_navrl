"""消融實驗共用診斷指標 — 15 個行為 metrics

用於 NavRL01-04 消融實驗，記錄 stuck/freeze/oscillation/retreat 等行為。
透過 wandb_trainer.py 的 ablation_logger hook 接入。

使用方式:
    logger = AblationMetricsLogger(env)
    # 每步:
    logger.step(actions, rewards, terminated, truncated, infos)
    # 每 rollout flush:
    metrics = logger.get_and_reset()  # → dict for wandb.log()
"""

from __future__ import annotations

import numpy as np
import torch
from collections import deque


class AblationMetricsLogger:
    """消融實驗行為診斷收集器。"""

    # stuck 判定: 連續 N 步 speed < threshold 且 goal dist 變化 < epsilon
    STUCK_WINDOW = 5
    STUCK_SPEED_THRESH = 0.02
    STUCK_DIST_EPS = 0.01

    # 近障礙物判定門檻
    NEAR_OBS_D_SAFE = 1.5
    DANGER_D_SAFE = 0.8

    # oscillation 窗口
    OSC_WINDOW = 10

    def __init__(self, env):
        """
        Args:
            env: unwrapped ManagerBasedRLEnv (or wrapped env with .unwrapped)
        """
        self._env = env.unwrapped if hasattr(env, 'unwrapped') else env
        N = self._env.num_envs
        device = self._env.device

        # Per-env 計數器
        self._consecutive_slow = torch.zeros(N, device=device)
        self._prev_goal_dist = None

        # Rollout 累計
        self._total_steps = 0
        self._freeze_steps = 0
        self._stuck_count = 0
        self._retreat_near_steps = 0
        self._near_obs_steps = 0
        self._danger_zone_steps = 0
        self._progress_near_obs = []
        self._v_toward_near_obs = []
        self._speed_near_obs = []
        self._d_safe_values = []
        self._front_clearance_values = []
        self._episode_lengths_at_collision = []
        self._episode_lengths_all = []

        # Oscillation: 記錄 v_toward 符號歷史
        self._v_toward_sign_history = deque(maxlen=self.OSC_WINDOW)
        self._sign_changes_total = 0
        self._sign_checks_total = 0

        # Shield (從 action term 讀取)
        self._shield_stats_collected = False

    def step(self, actions, rewards, terminated, truncated, infos):
        """每 env.step() 後呼叫。快速 GPU 操作，不做 CPU sync。"""
        env = self._env
        N = env.num_envs
        device = env.device
        self._total_steps += N

        # === 讀取環境狀態 ===
        robot = env.scene["robot"]
        robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
        robot_vel = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)
        speed = torch.norm(robot_vel, dim=-1)  # [N]

        # Goal position
        if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
            goal_pos = env._local_goal_world
        else:
            try:
                goal_pos = env.command_manager.get_command("goal_command")[:, :2]
            except Exception:
                goal_pos = robot_pos  # fallback

        goal_dist = torch.norm(goal_pos - robot_pos, dim=1)  # [N]

        # d_safe (bottom-10 LiDAR)
        try:
            sensor = env.scene.sensors["lidar"]
            sensor_pos_2d = sensor.data.pos_w[:, :2]
            hit_2d = sensor.data.ray_hits_w[:, :, :2]
            dists_2d = torch.norm(hit_2d - sensor_pos_2d.unsqueeze(1), dim=-1)
            dists_2d = torch.nan_to_num(dists_2d, nan=20.0, posinf=20.0)
            k = min(10, dists_2d.shape[1])
            bottom_k = torch.topk(dists_2d, k=k, dim=1, largest=False).values
            d_safe = (bottom_k.mean(dim=1) - 0.35).clamp(min=1e-4)
        except Exception:
            d_safe = torch.full((N,), 10.0, device=device)

        # v_toward_goal
        diff = goal_pos - robot_pos
        goal_dist_safe = goal_dist.clamp(min=1e-6)
        goal_dir = diff / goal_dist_safe.unsqueeze(1)
        v_toward = (robot_vel * goal_dir).sum(dim=1)  # [N]

        # === 1. Freeze ratio ===
        is_slow = speed < self.STUCK_SPEED_THRESH
        self._freeze_steps += is_slow.sum().item()

        # === 2. Stuck count ===
        if self._prev_goal_dist is not None:
            dist_change = (self._prev_goal_dist - goal_dist).abs()
            is_stuck_step = is_slow & (dist_change < self.STUCK_DIST_EPS)
            self._consecutive_slow += is_stuck_step.float()
            self._consecutive_slow[~is_stuck_step] = 0
            # 達到 STUCK_WINDOW 時記一次 stuck
            just_stuck = self._consecutive_slow >= self.STUCK_WINDOW
            self._stuck_count += just_stuck.sum().item()
            self._consecutive_slow[just_stuck] = 0  # 重置避免重複計數
        self._prev_goal_dist = goal_dist.clone()

        # Episode reset 時重置
        done = terminated | truncated
        if done.any():
            self._consecutive_slow[done] = 0
            self._prev_goal_dist[done] = goal_dist[done]

        # === 3. Oscillation score ===
        v_sign = (v_toward > 0).float().mean().item()  # 正向比例 [0,1]
        self._v_toward_sign_history.append(v_sign)
        if len(self._v_toward_sign_history) >= 2:
            changes = sum(
                1 for i in range(1, len(self._v_toward_sign_history))
                if abs(self._v_toward_sign_history[i] - self._v_toward_sign_history[i-1]) > 0.3
            )
            self._sign_changes_total += changes
            self._sign_checks_total += len(self._v_toward_sign_history) - 1

        # === 4-6. Near-obstacle metrics ===
        near_mask = d_safe < self.NEAR_OBS_D_SAFE
        n_near = near_mask.sum().item()
        self._near_obs_steps += n_near

        if n_near > 0:
            # retreat ratio
            retreat = near_mask & (v_toward < 0)
            self._retreat_near_steps += retreat.sum().item()

            # progress near obstacle
            if self._prev_goal_dist is not None:
                progress = (self._prev_goal_dist - goal_dist)[near_mask]
                self._progress_near_obs.append(progress.mean().item())

            # v_toward near obstacle
            self._v_toward_near_obs.append(v_toward[near_mask].mean().item())

            # speed near obstacle
            self._speed_near_obs.append(speed[near_mask].mean().item())

        # === 9-11. d_safe stats ===
        self._d_safe_values.append(d_safe.mean().item())
        self._danger_zone_steps += (d_safe < self.DANGER_D_SAFE).sum().item()

        # === 13. Front clearance (rough: use d_safe as proxy) ===
        # 精確版需要 bin-level 操作，這裡用 d_safe 近似
        self._front_clearance_values.append(d_safe.mean().item())

        # === 14-15. Episode length at termination ===
        if done.any():
            ep_lens = env.episode_length_buf[done].float()
            self._episode_lengths_all.extend(ep_lens.cpu().tolist())

            # 區分碰撞 vs 其他
            try:
                tm = env.termination_manager
                for name in tm._term_names:
                    if "collision" in name:
                        coll_buf = tm.get_term(name)
                        coll_done = coll_buf[done]
                        if coll_done.any():
                            coll_lens = ep_lens[coll_done]
                            self._episode_lengths_at_collision.extend(coll_lens.cpu().tolist())
                        break
            except Exception:
                pass

    def get_and_reset(self) -> dict:
        """回傳聚合指標並清空 buffer。"""
        total = max(self._total_steps, 1)
        near_total = max(self._near_obs_steps, 1)

        metrics = {
            "ablation/stuck_count": self._stuck_count,
            "ablation/freeze_ratio": self._freeze_steps / total,
            "ablation/oscillation_score": (
                self._sign_changes_total / max(self._sign_checks_total, 1)
            ),
            "ablation/retreat_ratio": self._retreat_near_steps / near_total,
            "ablation/progress_near_obs": (
                float(np.mean(self._progress_near_obs)) if self._progress_near_obs else 0.0
            ),
            "ablation/v_toward_near_obs": (
                float(np.mean(self._v_toward_near_obs)) if self._v_toward_near_obs else 0.0
            ),
            "ablation/d_safe_mean": (
                float(np.mean(self._d_safe_values)) if self._d_safe_values else 0.0
            ),
            "ablation/d_safe_min": (
                float(np.min(self._d_safe_values)) if self._d_safe_values else 0.0
            ),
            "ablation/danger_ratio": self._danger_zone_steps / total,
            "ablation/speed_near_obs": (
                float(np.mean(self._speed_near_obs)) if self._speed_near_obs else 0.0
            ),
            "ablation/front_clearance": (
                float(np.mean(self._front_clearance_values)) if self._front_clearance_values else 0.0
            ),
            "ablation/avg_ep_length": (
                float(np.mean(self._episode_lengths_all)) if self._episode_lengths_all else 0.0
            ),
            "ablation/collision_ep_len": (
                float(np.mean(self._episode_lengths_at_collision)) if self._episode_lengths_at_collision else 0.0
            ),
            "ablation/step_count": self._total_steps,
        }

        # Shield stats (if available)
        try:
            action_term = list(self._env.action_manager._terms.values())[0]
            if hasattr(action_term, 'shield_stats'):
                shield = action_term.shield_stats
                metrics["ablation/shield_rate"] = shield["shield_intervention_rate"]
                metrics["ablation/action_override_rate"] = shield["shield_override_rate"]
                action_term.reset_shield_stats()
        except Exception:
            pass

        self._reset()
        return metrics

    def _reset(self):
        """清空所有 buffer。"""
        N = self._env.num_envs
        self._consecutive_slow.zero_()
        self._total_steps = 0
        self._freeze_steps = 0
        self._stuck_count = 0
        self._retreat_near_steps = 0
        self._near_obs_steps = 0
        self._danger_zone_steps = 0
        self._progress_near_obs.clear()
        self._v_toward_near_obs.clear()
        self._speed_near_obs.clear()
        self._d_safe_values.clear()
        self._front_clearance_values.clear()
        self._episode_lengths_at_collision.clear()
        self._episode_lengths_all.clear()
        self._v_toward_sign_history.clear()
        self._sign_changes_total = 0
        self._sign_checks_total = 0
