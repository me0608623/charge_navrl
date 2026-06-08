"""Reward functions -- extracted from train_rnn_car_wdclip.py."""

import torch


def compute_obstacle_reward(env_unwrapped, obs_obs: torch.Tensor, max_obstacles: int,
                            mode: str = "zero") -> torch.Tensor:
    """Compute per-obstacle reward.

    Args:
        obs_obs: [num_envs, N, 9] obstacle observations
        mode: "zero" (Warp Drive) or "approach" (weak adversarial)
    Returns:
        [num_envs * N] flat reward
    """
    num_envs = obs_obs.shape[0]
    N = max_obstacles

    if mode == "zero":
        return torch.zeros(num_envs * N, device=obs_obs.device)

    # "approach" mode: small reward for being close to robot
    robot_rel = obs_obs[:, :, 4:6]  # [E, N, 2]
    dist_to_robot = robot_rel.norm(dim=-1)  # [E, N]

    # Reward: closer -> higher, max 0.1 at distance 0
    reward = (0.1 * (2.0 - dist_to_robot).clamp(0.0, 2.0) / 2.0)  # [E, N]

    # Speed penalty
    obs_vel = obs_obs[:, :, 2:4]
    speed = obs_vel.norm(dim=-1)
    reward = reward - 0.01 * speed

    return reward.reshape(-1)  # [E*N]


def compute_wd_charge_reward(
    env_unwrapped,
    actions: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    penalty_hit: float,
    reward_get_goal: float,
    cost_operate: float,
    penalty_timeout: float = 0.0,
    rl_fps: float = 5.0,
    cost_turn_rate: float = 0.5,
    penalty_smoothness: float = 0.0,
    prev_actions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute Warp Drive style sparse reward for charge agent.

    Uses termination_manager per-term buffers to detect goal_reached vs collision.

    Returns:
        (reward, breakdown) where breakdown contains per-component tensors:
          - goal_reward: [N] goal reaching reward (WD: car goal reward)
          - wall_hit_reward: [N] wall/static collision penalty (WD: car static obstacle reward)
          - obs_hit_reward: [N] dynamic obstacle collision penalty (WD: car dynamic obstacle reward)
          - timeout_reward: [N] timeout penalty (truncated & not terminated)
          - floor_reward: [N] always 0 for flat terrain (WD: car floor reward)
          - action_reward: [N] action cost (WD: car dynamic reward / cost_operate)
          - goal_reached: [N] bool
          - wall_collision: [N] bool
          - obs_collision: [N] bool
          - other_death: [N] bool (tipped/explosion)
    """
    N = terminated.shape[0]
    device = terminated.device
    reward = torch.zeros(N, device=device)

    terminated_flat = terminated.squeeze(-1).bool() if terminated.dim() > 1 else terminated.bool()
    truncated_flat = truncated.squeeze(-1).bool() if truncated.dim() > 1 else truncated.bool()

    # --- Detect goal_reached vs collision type from termination_manager ---
    goal_reached = torch.zeros(N, dtype=torch.bool, device=device)
    wall_collision = torch.zeros(N, dtype=torch.bool, device=device)
    obs_collision = torch.zeros(N, dtype=torch.bool, device=device)
    other_death = torch.zeros(N, dtype=torch.bool, device=device)

    try:
        tm = env_unwrapped.termination_manager
        for name in tm._term_names:
            buf = tm.get_term(name)
            if buf is None:
                continue
            if "goal_reached" in name or "reaching_goal" in name:
                goal_reached = goal_reached | buf.bool()
            elif "wall_collision" in name:
                wall_collision = wall_collision | buf.bool()
            elif "obstacle_collision" in name:
                obs_collision = obs_collision | buf.bool()
            elif "collision" in name:
                # Generic collision -- attribute to obstacle if not wall
                obs_collision = obs_collision | buf.bool()
            elif "tipped" in name or "explosion" in name or "flying" in name:
                other_death = other_death | buf.bool()
    except (AttributeError, RuntimeError):
        # Fallback
        obs_collision = terminated_flat & ~truncated_flat

    any_collision = wall_collision | obs_collision | other_death  # noqa: F841

    # --- Static vs dynamic obstacle collision attribution ---
    static_obs_collision = getattr(
        env_unwrapped, "_obs_collision_static_mask",
        torch.zeros(N, dtype=torch.bool, device=device),
    )
    dynamic_obs_collision = getattr(
        env_unwrapped, "_obs_collision_dynamic_mask",
        torch.zeros(N, dtype=torch.bool, device=device),
    )

    # --- Per-component reward ---
    goal_reward = torch.zeros(N, device=device)
    wall_hit_reward = torch.zeros(N, device=device)
    obs_hit_reward = torch.zeros(N, device=device)
    action_reward = torch.zeros(N, device=device)

    # Goal reached: +reward_get_goal
    goal_reward[goal_reached] = reward_get_goal
    reward += goal_reward

    # Wall/static collision: penalty (WD: car static obstacle reward)
    wall_hit_reward[wall_collision] = penalty_hit
    reward += wall_hit_reward

    # Obstacle/dynamic collision: penalty (WD: car dynamic obstacle reward)
    obs_hit_reward[obs_collision & ~wall_collision] = penalty_hit
    reward += obs_hit_reward

    # Other death (tipped/explosion): penalty
    other_hit = other_death & ~wall_collision & ~obs_collision
    reward[other_hit] += penalty_hit

    # --- Action cost (WD: only meaningful when cost_operate > 0, typically Phase 1) ---
    if cost_operate > 0 and actions is not None:
        cost_per_step = cost_operate / rl_fps
        nomal_acc = (actions[:, 0].float() - 9.0).abs() / 9.0
        nomal_turn = (actions[:, 1].float() - 9.0).abs() / 9.0
        action_reward = cost_per_step * (1.0 - nomal_acc) ** 2
        action_reward = action_reward + cost_per_step * cost_turn_rate * (1.0 - nomal_turn) ** 2
        alive = ~terminated_flat
        action_reward = action_reward * alive.float()
        reward += action_reward

    # --- Smoothness penalty (v3, 2026-06-08): 抑制單幀抽動 ---
    # penalty_smoothness=0.005, |Δratio_ang|∈[0,2]
    #   → 最壞每幀 -0.01，一 episode (300步) 最壞 -3.0 ≈ 20% collision penalty
    #   → 實際平均 ~-0.3~-0.5/episode，僅作 inductive bias，不破壞主目標
    smoothness_reward = torch.zeros(N, device=device)
    if penalty_smoothness > 0 and actions is not None and prev_actions is not None:
        ratio_ang_t = (actions[:, 1].float() - 9.0) / 9.0          # [-1, 1]
        ratio_ang_prev = (prev_actions[:, 1].float() - 9.0) / 9.0  # [-1, 1]
        delta_ang = (ratio_ang_t - ratio_ang_prev).abs()           # [0, 2]
        smoothness_reward = -penalty_smoothness * delta_ang
        alive = ~terminated_flat
        smoothness_reward = smoothness_reward * alive.float()
        reward += smoothness_reward

    # --- Timeout penalty (truncated but not terminated = episode 時間到但未碰撞/未到達目標) ---
    timeout_reward = torch.zeros(N, device=device)
    if penalty_timeout != 0.0:
        is_timeout = truncated_flat & ~terminated_flat
        timeout_reward[is_timeout] = penalty_timeout
        reward += timeout_reward

    breakdown = {
        "goal_reward": goal_reward,              # WD: car goal reward
        "wall_hit_reward": wall_hit_reward,      # WD: car static obstacle reward
        "obs_hit_reward": obs_hit_reward,        # WD: car dynamic obstacle reward
        "smoothness_reward": smoothness_reward,  # v3: anti-jitter penalty
        "timeout_reward": timeout_reward,        # timeout penalty
        "floor_reward": torch.zeros(N, device=device),  # WD: car floor reward (0 for flat)
        "action_reward": action_reward,          # WD: car dynamic reward (action cost)
        "goal_reached": goal_reached,
        "wall_collision": wall_collision,
        "obs_collision": obs_collision,
        "static_obs_collision": static_obs_collision,
        "dynamic_obs_collision": dynamic_obs_collision,
        "other_death": other_death,
    }

    return reward, breakdown
