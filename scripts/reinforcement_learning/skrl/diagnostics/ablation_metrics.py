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


class AblationMetricsLogger:
    """消融實驗行為診斷收集器。"""

    # stuck 判定: 連續 N 步 speed < threshold 且 goal dist 變化 < epsilon
    STUCK_WINDOW = 5
    STUCK_SPEED_THRESH = 0.02
    STUCK_DIST_EPS = 0.01

    # 近障礙物判定門檻
    NEAR_OBS_D_SAFE = 1.5
    DANGER_D_SAFE = 0.8

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
        # 自行追蹤 episode length（因為 Isaac Lab auto-reset 會在返回前清零 episode_length_buf）
        self._ep_len_counter = torch.zeros(N, device=device, dtype=torch.long)

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
        self._speed_all_values = []
        self._d_safe_values = []
        self._front_clearance_values = []
        self._episode_lengths_at_collision = []
        self._episode_lengths_all = []

        # Oscillation: per-env v_toward 符號翻轉追蹤
        self._prev_v_sign = None
        self._sign_changes_total = 0
        self._sign_checks_total = 0

        # Shield (從 action term 讀取)
        self._shield_stats_collected = False

        # --- 新增: 障礙物密度診斷 ---
        # B: 路徑效率 (per-env)
        self._prev_pos = None
        self._distance_traveled = torch.zeros(N, device=device)
        self._initial_goal_dist = torch.zeros(N, device=device)
        self._path_efficiency_values = []
        # C: 局部障礙密度 (LiDAR bins < 2m proxy)
        self._local_density_values = []
        # D: 動態障礙遭遇率
        self._dynamic_encounter_steps = 0
        # A: 碰撞類型分類
        self._collision_static_count = 0
        self._collision_dynamic_count = 0
        self._collision_total_count = 0
        # Action 分布 (init 時不需要 zero_，直接設值)

        # --- 新增: Action 分布診斷 ---
        self._action_saturation_count = 0
        self._action_total_count = 0
        self._action_center_count = 0
        self._lin_action_hist = torch.zeros(19, device=device, dtype=torch.long)
        self._ang_action_hist = torch.zeros(19, device=device, dtype=torch.long)

    def step(self, actions, rewards, terminated, truncated, infos):
        """每 env.step() 後呼叫。快速 GPU 操作，不做 CPU sync。"""
        env = self._env
        N = env.num_envs
        device = env.device
        self._total_steps += N

        # === 追蹤 episode 長度 ===
        self._ep_len_counter += 1

        # === 讀取環境狀態 ===
        robot = env.scene["robot"]
        robot_pos = torch.nan_to_num(robot.data.root_pos_w[:, :2], nan=0.0)
        robot_vel = torch.nan_to_num(robot.data.root_lin_vel_w[:, :2], nan=0.0)
        speed = torch.norm(robot_vel, dim=-1)  # [N]
        self._speed_all_values.append(speed.mean().item())

        # === Action 分布統計 ===
        if actions is not None:
            acts = actions.view(-1, 2) if actions.dim() >= 2 and actions.shape[-1] == 2 else None
            if acts is not None:
                lin_acts = acts[:, 0].long().clamp(0, 18)
                ang_acts = acts[:, 1].long().clamp(0, 18)
                # Saturation: 任一軸在邊界 (0 or 18)
                lin_edge = (lin_acts == 0) | (lin_acts == 18)
                ang_edge = (ang_acts == 0) | (ang_acts == 18)
                any_edge = lin_edge | ang_edge
                self._action_saturation_count += any_edge.sum().item()
                self._action_total_count += acts.shape[0]
                # Center: 雙軸都在中心 (9,9)
                is_center = (lin_acts == 9) & (ang_acts == 9)
                self._action_center_count += is_center.sum().item()
                # Histogram
                self._lin_action_hist += torch.bincount(lin_acts, minlength=19)[:19]
                self._ang_action_hist += torch.bincount(ang_acts, minlength=19)[:19]

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
        done = (terminated | truncated).squeeze(-1) if (terminated | truncated).dim() > 1 else (terminated | truncated)
        done = done.bool()
        if done.any():
            self._consecutive_slow[done] = 0
            self._prev_goal_dist[done] = goal_dist[done]

        # === 3. Oscillation score (per-env) ===
        # 每個 env 獨立計算 v_toward 符號翻轉，而非跨 env 平均（避免大數定律平滑掉震盪信號）
        v_sign_per_env = (v_toward > 0).float()  # [N] 每 env 0 or 1
        if self._prev_v_sign is not None:
            sign_flipped = (v_sign_per_env != self._prev_v_sign)  # [N] bool
            self._sign_changes_total += sign_flipped.sum().item()
            self._sign_checks_total += N
        self._prev_v_sign = v_sign_per_env.clone()

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

        # === B. 路徑效率 ===
        if self._prev_pos is not None:
            delta = torch.norm(robot_pos - self._prev_pos, dim=1)
            self._distance_traveled += delta
        else:
            # 首次 step: 記錄初始 goal 距離
            self._initial_goal_dist = goal_dist.clone()
        self._prev_pos = robot_pos.clone()

        # === C. 局部障礙密度 (LiDAR bins < 2m 數量) ===
        try:
            self._local_density_values.append(
                (dists_2d < 2.0).float().sum(dim=1).mean().item()
            )
        except Exception:
            pass

        # === D. 動態障礙遭遇率 ===
        try:
            obs_vel = getattr(env, "_obstacle_velocities", None)
            if obs_vel is not None:
                # 找有速度的障礙 (dynamic)
                vel_mag = torch.norm(obs_vel[:, :, :2], dim=-1)  # [N, num_obs]
                is_dynamic = vel_mag > 0.05  # speed > 0.05 m/s = dynamic
                if is_dynamic.any():
                    # 計算 robot 到每個障礙的距離
                    all_obs_pos = torch.zeros_like(obs_vel[:, :, :2])
                    num_obs = min(obs_vel.shape[1], getattr(env, "_num_obstacles", 50))
                    for i in range(num_obs):
                        obs_name = f"obstacle_{i}"
                        if obs_name in env.scene.keys():
                            p = env.scene[obs_name].data.root_pos_w
                            if p[:, 2].mean() > 0:  # visible
                                all_obs_pos[:, i] = torch.nan_to_num(p[:, :2], nan=999.0)
                            else:
                                all_obs_pos[:, i] = 999.0
                        else:
                            all_obs_pos[:, i] = 999.0
                    dists_to_obs = torch.norm(
                        all_obs_pos - robot_pos.unsqueeze(1), dim=-1
                    )  # [N, num_obs]
                    # 只看 dynamic 障礙, 非 dynamic 設為遠距
                    dists_to_obs[~is_dynamic] = 999.0
                    min_dyn_dist = dists_to_obs.min(dim=1).values  # [N]
                    self._dynamic_encounter_steps += (min_dyn_dist < 1.5).sum().item()
        except Exception:
            pass

        # === 14-15. Episode length at termination ===
        if done.any():
            # 使用自行追蹤的 episode length（Isaac Lab auto-reset 會清零 episode_length_buf）
            ep_lens = self._ep_len_counter[done].float()
            self._episode_lengths_all.extend(ep_lens.cpu().tolist())

            # B: 路徑效率 — episode 結束時計算
            init_dist = self._initial_goal_dist[done]
            traveled = self._distance_traveled[done]
            valid = init_dist > 1.0  # 至少 1m 才計算
            if valid.any():
                efficiency = traveled[valid] / init_dist[valid]
                self._path_efficiency_values.extend(efficiency.clamp(max=10.0).cpu().tolist())
            # reset per-env trackers
            self._distance_traveled[done] = 0.0
            self._initial_goal_dist[done] = goal_dist[done]
            self._ep_len_counter[done] = 0

            # 區分碰撞 vs 其他 + A: 碰撞類型分類
            # 使用 terminated (非 truncated) 判定碰撞 — 此值在 auto-reset 前已保存
            # terminated=True 且非 timeout 的就是碰撞終止
            terminated_1d = terminated.squeeze(-1) if terminated.dim() > 1 else terminated
            is_collision = done & terminated_1d.bool()  # terminated (非 timeout) = collision
            if is_collision.any():
                coll_lens = self._ep_len_counter[is_collision].float()
                # ep_len_counter 已重置，改用 ep_lens 中對應的值
                coll_mask_in_done = terminated_1d[done].bool()
                if coll_mask_in_done.any():
                    self._episode_lengths_at_collision.extend(ep_lens[coll_mask_in_done].cpu().tolist())
                n_coll = is_collision.sum().item()
                self._collision_total_count += n_coll
                # 碰撞類型分類：最近障礙物是 static 還是 dynamic
                obs_vel = getattr(env, "_obstacle_velocities", None)
                if obs_vel is not None:
                    coll_indices = torch.where(is_collision)[0]
                    num_obs = min(obs_vel.shape[1], getattr(env, "_num_obstacles", 100))
                    # 快取障礙物位置
                    cache = getattr(env, "_obstacle_cache", None)
                    for idx in coll_indices:
                        idx_i = idx.item()
                        min_dist = 999.0
                        nearest_is_dynamic = False
                        for oi in range(num_obs):
                            try:
                                if cache is not None and oi < len(cache):
                                    obj = cache[oi]
                                else:
                                    obs_name = f"obstacle_{oi}"
                                    obj = env.scene[obs_name] if obs_name in env.scene.keys() else None
                                if obj is None:
                                    continue
                                p = obj.data.root_pos_w[idx_i]
                                if p[2] < 0:
                                    continue
                                d = torch.norm(p[:2] - robot_pos[idx_i]).item()
                                if d < min_dist:
                                    min_dist = d
                                    v = torch.norm(obs_vel[idx_i, oi, :2]).item()
                                    nearest_is_dynamic = v > 0.05
                            except Exception:
                                continue
                        if nearest_is_dynamic:
                            self._collision_dynamic_count += 1
                        else:
                            self._collision_static_count += 1

    def get_and_reset(self) -> dict:
        """回傳聚合指標並清空 buffer。"""
        total = max(self._total_steps, 1)
        near_total = max(self._near_obs_steps, 1)

        metrics = {
            # Tier 2: 行為診斷
            "behavior/stuck_events": self._stuck_count,
            "behavior/freeze_ratio": self._freeze_steps / total,
            "behavior/oscillation": (
                self._sign_changes_total / max(self._sign_checks_total, 1)
            ),
            "behavior/retreat_ratio": self._retreat_near_steps / near_total,
            "behavior/progress_near_obstacle": (
                float(np.mean(self._progress_near_obs)) if self._progress_near_obs else 0.0
            ),
            "behavior/goal_velocity_near_obstacle": (
                float(np.mean(self._v_toward_near_obs)) if self._v_toward_near_obs else 0.0
            ),
            "behavior/obstacle_distance_avg": (
                float(np.mean(self._d_safe_values)) if self._d_safe_values else 0.0
            ),
            "behavior/obstacle_distance_min": (
                float(np.min(self._d_safe_values)) if self._d_safe_values else 0.0
            ),
            "behavior/danger_zone_ratio": self._danger_zone_steps / total,
            "behavior/speed_avg": (
                float(np.mean(self._speed_all_values)) if self._speed_all_values else 0.0
            ),
            "behavior/speed_near_obstacle": (
                float(np.mean(self._speed_near_obs)) if self._speed_near_obs else 0.0
            ),
            "behavior/front_clearance": (
                float(np.mean(self._front_clearance_values)) if self._front_clearance_values else 0.0
            ),
            "behavior/avg_episode_length": (
                float(np.mean(self._episode_lengths_all)) if self._episode_lengths_all else 0.0
            ),
            "behavior/collision_episode_length": (
                float(np.mean(self._episode_lengths_at_collision)) if self._episode_lengths_at_collision else 0.0
            ),
            # 新增: 障礙物密度診斷
            "behavior/path_efficiency": (
                float(np.mean(self._path_efficiency_values)) if self._path_efficiency_values else 0.0
            ),
            "behavior/local_obstacle_density": (
                float(np.mean(self._local_density_values)) if self._local_density_values else 0.0
            ),
            "behavior/dynamic_encounter_ratio": self._dynamic_encounter_steps / total,
            "behavior/collision_static_ratio": (
                self._collision_static_count / max(self._collision_total_count, 1)
            ),
            "behavior/collision_dynamic_ratio": (
                self._collision_dynamic_count / max(self._collision_total_count, 1)
            ),
        }

        # Action 分布
        action_total = max(self._action_total_count, 1)
        metrics["action/saturation_rate"] = self._action_saturation_count / action_total
        metrics["action/center_fraction"] = self._action_center_count / action_total

        def _hist_entropy(hist):
            p = hist.float() / hist.sum().clamp(min=1)
            p = p[p > 0]
            return -(p * torch.log(p)).sum().item()

        metrics["action/linear_entropy"] = _hist_entropy(self._lin_action_hist)
        metrics["action/angular_entropy"] = _hist_entropy(self._ang_action_hist)
        lin_total = max(self._lin_action_hist.sum().item(), 1)
        ang_total = max(self._ang_action_hist.sum().item(), 1)
        arange = torch.arange(19, device=self._lin_action_hist.device, dtype=torch.float)
        metrics["action/linear_mean"] = (self._lin_action_hist.float() * arange).sum().item() / lin_total
        metrics["action/angular_mean"] = (self._ang_action_hist.float() * arange).sum().item() / ang_total

        # Tier 3: Shield stats (if available)
        try:
            action_term = list(self._env.action_manager._terms.values())[0]
            if hasattr(action_term, 'shield_stats'):
                shield = action_term.shield_stats
                metrics["shield/intervention_rate"] = shield["shield_intervention_rate"]
                metrics["shield/action_modified_rate"] = shield["shield_override_rate"]
                action_term.reset_shield_stats()
        except Exception:
            pass

        self._reset()
        return metrics

    def _reset(self):
        """清空所有 buffer。"""
        N = self._env.num_envs
        self._consecutive_slow.zero_()
        # 注意: 不重置 _ep_len_counter — 它是 per-env 跨 rollout 持續追蹤的
        self._total_steps = 0
        self._freeze_steps = 0
        self._stuck_count = 0
        self._retreat_near_steps = 0
        self._near_obs_steps = 0
        self._danger_zone_steps = 0
        self._progress_near_obs.clear()
        self._v_toward_near_obs.clear()
        self._speed_near_obs.clear()
        self._speed_all_values.clear()
        self._d_safe_values.clear()
        self._front_clearance_values.clear()
        self._episode_lengths_at_collision.clear()
        self._episode_lengths_all.clear()
        self._prev_v_sign = None
        self._sign_changes_total = 0
        self._sign_checks_total = 0
        # 新增指標 reset
        self._path_efficiency_values.clear()
        self._local_density_values.clear()
        self._dynamic_encounter_steps = 0
        self._collision_static_count = 0
        self._collision_dynamic_count = 0
        self._collision_total_count = 0
        # Action 分布
        self._action_saturation_count = 0
        self._action_total_count = 0
        self._action_center_count = 0
        self._lin_action_hist.zero_()
        self._ang_action_hist.zero_()
