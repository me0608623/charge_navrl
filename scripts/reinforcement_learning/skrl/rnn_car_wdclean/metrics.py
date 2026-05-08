"""MetricsCollector -- extracted from train_rnn_car_wdclip.py."""

import math

import numpy as np
import torch


class MetricsCollector:
    """Warp Drive style complete metrics collector.

    Metric naming follows Warp Drive custom_trainer.py:
      spot -> charge, car -> charge
      goal -> goal (unchanged)
      obstacle -> obstacle
    """

    def __init__(self, num_envs: int, max_obstacles: int, device):
        self.num_envs = num_envs
        self.max_obstacles = max_obstacles
        self.device = device

        # --- Per-env episode accumulators ---
        self._ep_reward = torch.zeros(num_envs, device=device)
        self._ep_length = torch.zeros(num_envs, device=device)
        self._ep_steps_alive = torch.zeros(num_envs, device=device)

        # --- Per-env WD reward decomposition accumulators ---
        self._ep_goal_reward = torch.zeros(num_envs, device=device)
        self._ep_wall_hit_reward = torch.zeros(num_envs, device=device)
        self._ep_obs_hit_reward = torch.zeros(num_envs, device=device)
        self._ep_floor_reward = torch.zeros(num_envs, device=device)
        self._ep_action_reward = torch.zeros(num_envs, device=device)

        # --- Completed episode reward decomposition ---
        self._completed_goal_reward: list[float] = []
        self._completed_wall_hit_reward: list[float] = []
        self._completed_obs_hit_reward: list[float] = []
        self._completed_floor_reward: list[float] = []
        self._completed_action_reward: list[float] = []

        # --- Completed episode stats ---
        self._completed_rewards: list[float] = []
        self._completed_lengths: list[float] = []
        self._completed_alive: list[float] = []

        # --- Per-outcome expected value tracking (WD 期望值) ---
        # 每個 episode 完成時，根據結局（goal/collision/timeout）分別記錄 reward 和 length，
        # 用於計算條件期望值 E[reward | outcome]。
        self._ev_reward_goal: list[float] = []       # E[reward | goal_reached]
        self._ev_reward_collision: list[float] = []  # E[reward | collision]
        self._ev_reward_timeout: list[float] = []    # E[reward | timeout]
        self._ev_length_goal: list[float] = []       # E[length | goal_reached]
        self._ev_length_collision: list[float] = []  # E[length | collision]
        self._ev_length_timeout: list[float] = []    # E[length | timeout]
        self._ev_alive_ratio: list[float] = []       # alive_steps / total_steps per episode

        # --- Goal-directed behavior diagnostics ---
        self._ep_goal_start_dist = torch.full((num_envs,), float("nan"), device=device)
        self._ep_goal_prev_dist = torch.full((num_envs,), float("nan"), device=device)
        self._ep_goal_progress_sum = torch.zeros(num_envs, device=device)
        self._ep_goal_velocity_sum = torch.zeros(num_envs, device=device)
        self._ep_goal_heading_abs_sum = torch.zeros(num_envs, device=device)
        self._ep_goal_diag_steps = torch.zeros(num_envs, device=device)
        self._ep_goal_prev_target_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._ep_goal_switch_count = torch.zeros(num_envs, device=device)
        self._completed_goal_start_dist: list[float] = []
        self._completed_goal_end_dist: list[float] = []
        self._completed_goal_distance_delta: list[float] = []
        self._completed_goal_progress_mean: list[float] = []
        self._completed_velocity_to_goal_mean: list[float] = []
        self._completed_heading_error_abs_mean: list[float] = []
        self._completed_target_switch_rate: list[float] = []

        # --- Termination counters ---
        self._goal_reached = 0
        self._collision = 0
        self._wall_collision = 0
        self._obstacle_collision = 0
        self._timeout = 0
        self._tipped_over = 0
        self._total_eps = 0
        self._first_step_deaths = 0

        # --- Reward term accumulators ---
        self._reward_terms: dict[str, list[float]] = {}

        # --- Curriculum info ---
        self._curriculum_info: dict[str, float] = {}

        # --- Obstacle policy per-step stats ---
        self._obs_speed_limit: float = 0.8
        self._obs_speeds: list[float] = []
        self._obs_distances: list[float] = []

    def compute_goal_diagnostics(self, env_unwrapped) -> dict[str, torch.Tensor] | None:
        """Return per-env goal-directed diagnostics before env.step()."""
        try:
            robot = env_unwrapped.scene["robot"]
            robot_pos = robot.data.root_pos_w[:, :2]
            robot_vel = robot.data.root_lin_vel_w[:, :2]
            robot_quat = robot.data.root_quat_w

            if hasattr(env_unwrapped, "_local_goal_world") and env_unwrapped._local_goal_world is not None:
                goal_pos = env_unwrapped._local_goal_world[:, :2]
            else:
                goal_pos = env_unwrapped.command_manager.get_command("goal_command")[:, :2]

            target_id = None
            try:
                goal_term = env_unwrapped.command_manager.get_term("goal_command")
                if hasattr(goal_term, "nearest_goal_idx"):
                    target_id = goal_term.nearest_goal_idx.detach().long().to(robot_pos.device)
            except (AttributeError, KeyError, RuntimeError, ValueError):
                target_id = None
            if target_id is None or target_id.shape[0] != robot_pos.shape[0]:
                quantized_goal = torch.round(goal_pos * 100.0).long()
                target_id = quantized_goal[:, 0] * 1000003 + quantized_goal[:, 1]

            diff = torch.nan_to_num(goal_pos - robot_pos, nan=0.0, posinf=0.0, neginf=0.0)
            dist = torch.norm(diff, dim=1).clamp_min(1e-6)
            goal_dir = diff / dist.unsqueeze(-1)
            velocity_to_goal = (robot_vel * goal_dir).sum(dim=1)

            w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            goal_angle = torch.atan2(diff[:, 1], diff[:, 0])
            heading_error = torch.atan2(torch.sin(goal_angle - yaw), torch.cos(goal_angle - yaw))

            return {
                "distance": dist,
                "velocity_to_goal": velocity_to_goal,
                "heading_error_abs": heading_error.abs(),
                "target_id": target_id,
            }
        except (AttributeError, KeyError, RuntimeError):
            return None

    def step(self, obs, reward, done, info,
             obs_obs: torch.Tensor | None = None,
             obs_actions: torch.Tensor | None = None,
             reward_breakdown: dict[str, torch.Tensor] | None = None,
             goal_diagnostics: dict[str, torch.Tensor] | None = None):
        """Record one env step."""
        self._ep_reward += reward.reshape(-1)
        self._ep_length += 1.0
        self._ep_steps_alive += (1.0 - done.reshape(-1).float())

        # --- Accumulate WD reward decomposition ---
        if reward_breakdown is not None:
            self._ep_goal_reward += reward_breakdown["goal_reward"]
            self._ep_wall_hit_reward += reward_breakdown["wall_hit_reward"]
            self._ep_obs_hit_reward += reward_breakdown["obs_hit_reward"]
            self._ep_floor_reward += reward_breakdown["floor_reward"]
            self._ep_action_reward += reward_breakdown["action_reward"]

        # --- Goal-directed diagnostics ---
        if goal_diagnostics is not None:
            goal_dist = goal_diagnostics["distance"].detach()
            first_valid = ~torch.isfinite(self._ep_goal_start_dist)
            self._ep_goal_start_dist[first_valid] = goal_dist[first_valid]
            prev_valid = torch.isfinite(self._ep_goal_prev_dist)
            step_progress = torch.zeros_like(goal_dist)
            step_progress[prev_valid] = self._ep_goal_prev_dist[prev_valid] - goal_dist[prev_valid]
            self._ep_goal_prev_dist = goal_dist
            self._ep_goal_progress_sum += step_progress
            self._ep_goal_velocity_sum += goal_diagnostics["velocity_to_goal"].detach()
            self._ep_goal_heading_abs_sum += goal_diagnostics["heading_error_abs"].detach()
            target_id = goal_diagnostics.get("target_id")
            if target_id is not None:
                target_id = target_id.detach().long()
                prev_target_valid = self._ep_goal_prev_target_id >= 0
                target_switched = prev_target_valid & (self._ep_goal_prev_target_id != target_id)
                self._ep_goal_switch_count += target_switched.float()
                self._ep_goal_prev_target_id = target_id
            self._ep_goal_diag_steps += 1.0

        # --- Parse info["log"] ---
        if "log" in info:
            for key, val in info["log"].items():
                if key.startswith("Curriculum/"):
                    k = key.replace("Curriculum/", "")
                    try:
                        self._curriculum_info[k] = val.item() if isinstance(val, torch.Tensor) else float(val)
                    except (ValueError, TypeError):
                        pass

                if key.startswith("Episode_Reward/"):
                    k = key.replace("Episode_Reward/", "")
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    self._reward_terms.setdefault(k, []).append(v)

                if key.startswith("Episode_Termination/"):
                    k = key.replace("Episode_Termination/", "")
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    num_done = done.reshape(-1).bool().sum().item()
                    count = max(0, round(v * num_done))
                    if "goal_reached" in k:
                        self._goal_reached += count
                    elif "obstacle_collision" in k:
                        self._obstacle_collision += count
                        self._collision += count
                    elif "wall_collision" in k:
                        self._wall_collision += count
                        self._collision += count
                    elif "collision" in k:
                        self._collision += count
                    elif "time_out" in k:
                        self._timeout += count
                    elif "tipped_over" in k or "physics_explosion" in k:
                        self._tipped_over += count

        # --- Obstacle policy metrics ---
        if obs_actions is not None:
            actual_vel = obs_actions.reshape(-1, 2) * self._obs_speed_limit
            speed = actual_vel.norm(dim=-1).mean().item()
            self._obs_speeds.append(speed)
        if obs_obs is not None:
            robot_rel = obs_obs[:, :, 4:6]
            dist = robot_rel.norm(dim=-1).mean().item()
            self._obs_distances.append(dist)

        # --- Episode completion ---
        done_mask = done.reshape(-1).bool()
        if done_mask.any():
            ids = done_mask.nonzero(as_tuple=False).reshape(-1)
            for idx in ids:
                self._completed_rewards.append(self._ep_reward[idx].item())
                self._completed_lengths.append(self._ep_length[idx].item())
                self._completed_alive.append(self._ep_steps_alive[idx].item())
                self._completed_goal_reward.append(self._ep_goal_reward[idx].item())
                self._completed_wall_hit_reward.append(self._ep_wall_hit_reward[idx].item())
                self._completed_obs_hit_reward.append(self._ep_obs_hit_reward[idx].item())
                self._completed_floor_reward.append(self._ep_floor_reward[idx].item())
                self._completed_action_reward.append(self._ep_action_reward[idx].item())
                if torch.isfinite(self._ep_goal_start_dist[idx]) and torch.isfinite(self._ep_goal_prev_dist[idx]):
                    steps = max(self._ep_goal_diag_steps[idx].item(), 1.0)
                    start_dist = self._ep_goal_start_dist[idx].item()
                    end_dist = self._ep_goal_prev_dist[idx].item()
                    self._completed_goal_start_dist.append(start_dist)
                    self._completed_goal_end_dist.append(end_dist)
                    self._completed_goal_distance_delta.append(start_dist - end_dist)
                    self._completed_goal_progress_mean.append(self._ep_goal_progress_sum[idx].item() / steps)
                    self._completed_velocity_to_goal_mean.append(self._ep_goal_velocity_sum[idx].item() / steps)
                    self._completed_heading_error_abs_mean.append(self._ep_goal_heading_abs_sum[idx].item() / steps)
                    switch_steps = max(steps - 1.0, 1.0)
                    self._completed_target_switch_rate.append(self._ep_goal_switch_count[idx].item() / switch_steps)
                # --- Per-outcome expected value tracking ---
                ep_rwd = self._ep_reward[idx].item()
                ep_len = self._ep_length[idx].item()
                alive_ratio = self._ep_steps_alive[idx].item() / max(ep_len, 1.0)
                self._ev_alive_ratio.append(alive_ratio)
                if reward_breakdown is not None:
                    if reward_breakdown["goal_reached"][idx]:
                        self._ev_reward_goal.append(ep_rwd)
                        self._ev_length_goal.append(ep_len)
                    elif reward_breakdown["wall_collision"][idx] or reward_breakdown["obs_collision"][idx]:
                        self._ev_reward_collision.append(ep_rwd)
                        self._ev_length_collision.append(ep_len)
                    else:
                        self._ev_reward_timeout.append(ep_rwd)
                        self._ev_length_timeout.append(ep_len)

                self._total_eps += 1
                if self._ep_length[idx].item() <= 2:
                    self._first_step_deaths += 1
            self._ep_reward[ids] = 0.0
            self._ep_length[ids] = 0.0
            self._ep_steps_alive[ids] = 0.0
            self._ep_goal_reward[ids] = 0.0
            self._ep_wall_hit_reward[ids] = 0.0
            self._ep_obs_hit_reward[ids] = 0.0
            self._ep_floor_reward[ids] = 0.0
            self._ep_action_reward[ids] = 0.0
            self._ep_goal_start_dist[ids] = float("nan")
            self._ep_goal_prev_dist[ids] = float("nan")
            self._ep_goal_progress_sum[ids] = 0.0
            self._ep_goal_velocity_sum[ids] = 0.0
            self._ep_goal_heading_abs_sum[ids] = 0.0
            self._ep_goal_diag_steps[ids] = 0.0
            self._ep_goal_prev_target_id[ids] = -1
            self._ep_goal_switch_count[ids] = 0.0

    def get_gamma(self):
        return self._curriculum_info.get("gamma", None)

    def collect(self) -> dict[str, float]:
        """Collect all metrics under structured WandB groups."""
        m: dict[str, float] = {}
        eps = 1e-8
        te = self._total_eps + eps

        if self._completed_rewards:
            m["charge/reward_mean"] = np.mean(self._completed_rewards)
            m["charge/reward_max"] = np.max(self._completed_rewards)
            m["charge/reward_min"] = np.min(self._completed_rewards)
        if self._completed_lengths:
            m["charge/episode_length_mean"] = np.mean(self._completed_lengths)
        m["charge/total_episodes"] = self._total_eps
        m["charge/goal_reach_rate"] = self._goal_reached / te
        m["charge/hit_probability"] = self._collision / te
        m["charge/timeout_rate"] = self._timeout / te
        m["charge/survival_probability"] = 1.0 - self._collision / te
        m["charge/dies_at_birth_rate"] = self._first_step_deaths / te

        m["charge/wall_collision_rate"] = self._wall_collision / te
        m["charge/obstacle_collision_rate"] = self._obstacle_collision / te
        total_col = self._collision + eps
        m["charge/VR_wall"] = self._wall_collision / total_col
        m["charge/VR_obstacle"] = self._obstacle_collision / total_col

        if self._completed_goal_start_dist:
            m["goal_diagnostics/start_distance_mean"] = np.mean(self._completed_goal_start_dist)
            m["goal_diagnostics/end_distance_mean"] = np.mean(self._completed_goal_end_dist)
            m["goal_diagnostics/distance_delta_mean"] = np.mean(self._completed_goal_distance_delta)
            m["goal_diagnostics/progress_per_step_mean"] = np.mean(self._completed_goal_progress_mean)
            m["goal_diagnostics/velocity_to_goal_mean"] = np.mean(self._completed_velocity_to_goal_mean)
            m["goal_diagnostics/heading_error_abs_mean_rad"] = np.mean(self._completed_heading_error_abs_mean)
            m["goal_diagnostics/heading_error_abs_mean_deg"] = (
                np.mean(self._completed_heading_error_abs_mean) * 180.0 / math.pi
            )
        if self._completed_target_switch_rate:
            m["goal_diagnostics/target_switch_rate"] = np.mean(self._completed_target_switch_rate)

        m["goal_diagnostics/reached_count"] = self._goal_reached

        m["obstacle/collision_count"] = self._collision
        m["obstacle/collision_rate"] = self._collision / te
        if self._obs_speeds:
            m["obstacle/mean_speed"] = np.mean(self._obs_speeds)
        if self._obs_distances:
            m["obstacle/mean_distance_to_robot"] = np.mean(self._obs_distances)

        for k, vals in self._reward_terms.items():
            if vals:
                m[f"charge/reward_term/{k}"] = np.mean(vals)

        if "stage" in self._curriculum_info:
            m["curriculum/stage"] = self._curriculum_info["stage"]

        # --- Expected Value metrics (WD 期望值) ---
        # V_total: 總體期望值 = 所有 episode 的平均回報 (WD: V_Spot)
        if self._completed_rewards:
            m["expect_value/V_total"] = np.mean(self._completed_rewards)

        # V_navigate: 導航期望值 = E[reward | goal_reached] (WD: V_Move)
        # 衡量「成功 episode 的回報品質」— 是快速到達還是勉強到達？
        if self._ev_reward_goal:
            m["expect_value/V_navigate"] = np.mean(self._ev_reward_goal)

        # V_collision: 碰撞期望值 = E[reward | collision]
        # 衡量「碰撞 episode 積累了多少回報才死」— 越接近 0 代表早期就碰撞
        if self._ev_reward_collision:
            m["expect_value/V_collision"] = np.mean(self._ev_reward_collision)

        # V_timeout: 超時期望值 = E[reward | timeout]
        # 衡量「超時 episode 是接近成功（reward ≈ 0）還是完全卡住（reward << 0）」
        if self._ev_reward_timeout:
            m["expect_value/V_timeout"] = np.mean(self._ev_reward_timeout)

        # V_survive: 存活期望值 = E[alive_steps / ep_length] (WD: V_Survive)
        # 衡量 agent 在 episode 中的平均存活比例
        if self._ev_alive_ratio:
            m["expect_value/V_survive"] = np.mean(self._ev_alive_ratio)

        # p_goal: 到達目標機率 (WD: p_Spot-Goal)
        m["expect_value/p_goal"] = self._goal_reached / te

        # p_collision: 碰撞機率 (WD: p_Spot-Obs + p_Spot-Map)
        m["expect_value/p_collision"] = self._collision / te

        # p_timeout: 超時機率
        m["expect_value/p_timeout"] = self._timeout / te

        # VR_wall / VR_obs: 碰撞歸因佔比 (WD: VR_Spot-Map / VR_Spot-Obs)
        # 「碰撞中有多少比例來自牆壁 vs 動態障礙」— 用於診斷 policy 弱點
        total_col = self._collision + eps
        m["expect_value/VR_wall"] = self._wall_collision / total_col
        m["expect_value/VR_obs"] = self._obstacle_collision / total_col

        # ep_length_goal: 成功 episode 平均步數 — 導航效率
        if self._ev_length_goal:
            m["expect_value/ep_length_goal"] = np.mean(self._ev_length_goal)

        # ep_length_collision: 碰撞 episode 平均步數 — 碰撞前能存活多久
        if self._ev_length_collision:
            m["expect_value/ep_length_collision"] = np.mean(self._ev_length_collision)

        # V_goal_reward: 目標獎勵期望值 = E[goal_reward 分量] (WD: V_Spot-Goal)
        # 衡量每 episode 平均獲得多少 goal reward（≈ p_goal × reward_get_goal）
        if self._completed_goal_reward:
            m["expect_value/V_goal_reward"] = np.mean(self._completed_goal_reward)

        # V_penalty: 碰撞懲罰期望值 = E[wall_hit + obs_hit 分量]
        # 衡量每 episode 平均承受多少碰撞懲罰（≈ p_collision × penalty_hit）
        if self._completed_wall_hit_reward and self._completed_obs_hit_reward:
            wall_arr = np.array(self._completed_wall_hit_reward)
            obs_arr = np.array(self._completed_obs_hit_reward)
            m["expect_value/V_penalty"] = np.mean(wall_arr + obs_arr)

        # V_action_cost: 操作成本期望值 = E[action_reward 分量]
        # 衡量 action cost 對 episode reward 的平均貢獻（Phase 1 有 cost_operate 時才有意義）
        if self._completed_action_reward:
            action_mean = np.mean(self._completed_action_reward)
            if abs(action_mean) > 1e-6:
                m["expect_value/V_action_cost"] = action_mean

        return m

    def reset(self):
        self._completed_rewards.clear()
        self._completed_lengths.clear()
        self._completed_alive.clear()
        self._completed_goal_reward.clear()
        self._completed_wall_hit_reward.clear()
        self._completed_obs_hit_reward.clear()
        self._completed_floor_reward.clear()
        self._completed_action_reward.clear()
        self._completed_goal_start_dist.clear()
        self._completed_goal_end_dist.clear()
        self._completed_goal_distance_delta.clear()
        self._completed_goal_progress_mean.clear()
        self._completed_velocity_to_goal_mean.clear()
        self._completed_heading_error_abs_mean.clear()
        self._completed_target_switch_rate.clear()
        self._goal_reached = 0
        self._collision = 0
        self._wall_collision = 0
        self._obstacle_collision = 0
        self._timeout = 0
        self._tipped_over = 0
        self._total_eps = 0
        self._first_step_deaths = 0
        self._reward_terms.clear()
        self._obs_speeds.clear()
        self._obs_distances.clear()
        self._ev_reward_goal.clear()
        self._ev_reward_collision.clear()
        self._ev_reward_timeout.clear()
        self._ev_length_goal.clear()
        self._ev_length_collision.clear()
        self._ev_length_timeout.clear()
        self._ev_alive_ratio.clear()
