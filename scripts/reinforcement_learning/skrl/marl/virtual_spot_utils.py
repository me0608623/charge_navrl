"""
Virtual Spot Utilities — Multi-Agent RL support for Isaac Lab

WarpDrive 的 num_spot=2-6 表示場景中有多個 policy-controlled car 互相競爭。
Isaac Lab 只有 1 個物理 charge robot，所以用 "虛擬 spot" (kinematic rigid body)
來模擬額外的 car agent。

虛擬 spot:
  - 使用 obstacle slot 的物理實體（kinematic rigid body）
  - 與 charge 共享同一個 policy network (parameter sharing, 和 WD 一致)
  - 觀測: 合成 79D (ego + goal + analytical LiDAR + time)
  - 動作: Discrete 19×19 (accel × omega) → 運動學更新 position/heading
  - 互相之間 + 對 charge 都是障礙物

Slot 分區:
  obstacle_0 ~ obstacle_{N_vs-1}   → virtual spots
  obstacle_{N_vs} ~ obstacle_{N_vs+N_obs-1} → regular obstacles (obstacle policy)
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional

# Matching modular_rnn_models.py constants
NUM_BINS = 19
LIDAR_DIM = 72
USED_OBS_DIM = 79  # ego(4) + goal(2) + lidar(72) + time(1)
FULL_OBS_DIM = 139  # env output dim (padded to match charge)


# ============================================================================
# Virtual Spot State (per-env, per-spot kinematic state)
# ============================================================================

class VirtualSpotState:
    """管理所有虛擬 spot 的運動學狀態。

    每個 spot: position(x,y), velocity(vx,vy), heading(theta), angular_vel(omega)

    WD car_mode 物理:
      spot_max_speed_x = 1.4 m/s, spot_min_speed_x = -1.05 m/s
      spot_max_speed_y = 0.01 (幾乎純前後)
      spot_max_acceleration_x = max_speed_x / rl_fps
      spot_max_turn_pi_ = 3 * rl_fps
      action: 19 bins for accel, 19 bins for turn

    我們的 charge 物理 (Isaac Lab):
      v_max = 1.0 m/s, a_max = 0.5 m/s², omega_max = 0.25π rad/s
      dt = 0.2s (rl_fps=5)

    虛擬 spot 用 charge 的物理參數，保持一致。
    """

    def __init__(
        self,
        num_envs: int,
        max_spots: int,
        device: torch.device,
        v_max: float = 1.0,
        a_max: float = 0.5,
        omega_max: float = 0.25 * math.pi,
        dt: float = 0.2,
        body_radius: float = 0.35,
    ):
        self.num_envs = num_envs
        self.max_spots = max_spots
        self.device = device
        self.v_max = v_max
        self.a_max = a_max
        self.omega_max = omega_max
        self.dt = dt
        self.body_radius = body_radius

        # State tensors [num_envs, max_spots]
        self.pos_x = torch.zeros(num_envs, max_spots, device=device)
        self.pos_y = torch.zeros(num_envs, max_spots, device=device)
        self.vel = torch.zeros(num_envs, max_spots, device=device)  # forward speed
        self.heading = torch.zeros(num_envs, max_spots, device=device)  # radians
        self.omega = torch.zeros(num_envs, max_spots, device=device)  # angular vel
        self.alive = torch.zeros(num_envs, max_spots, dtype=torch.bool, device=device)

    def reset(self, env_ids: torch.Tensor, num_active: int,
              bound_limit: float = 7.0, min_dist: float = 1.5):
        """Reset virtual spots for given envs.

        Spawns them at random positions avoiding charge robot and each other.
        """
        if len(env_ids) == 0 or num_active == 0:
            return
        E = len(env_ids)
        n = min(num_active, self.max_spots)

        self.alive[env_ids] = False
        self.alive[env_ids, :n] = True
        self.vel[env_ids] = 0.0
        self.omega[env_ids] = 0.0
        self.heading[env_ids] = torch.rand(E, self.max_spots, device=self.device) * 2 * math.pi

        # Random positions within bounds, with rejection sampling for overlap
        for s in range(n):
            for attempt in range(10):
                px = (torch.rand(E, device=self.device) * 2 - 1) * (bound_limit - 1.0)
                py = (torch.rand(E, device=self.device) * 2 - 1) * (bound_limit - 1.0)
                # Check distance to previous spots
                ok = torch.ones(E, dtype=torch.bool, device=self.device)
                for prev in range(s):
                    dx = px - self.pos_x[env_ids, prev]
                    dy = py - self.pos_y[env_ids, prev]
                    dist = (dx ** 2 + dy ** 2).sqrt()
                    ok = ok & (dist > min_dist)
                if ok.all() or attempt == 9:
                    self.pos_x[env_ids, s] = px
                    self.pos_y[env_ids, s] = py
                    break

    def apply_actions(self, actions: torch.Tensor, num_active: int,
                      bound_limit: float = 7.0):
        """Apply discrete actions (accel_bin, omega_bin) to update kinematic state.

        Args:
            actions: [num_envs, max_spots, 2] — accel_bin(0-18), omega_bin(0-18)
            num_active: number of active virtual spots
        """
        n = min(num_active, self.max_spots)
        if n == 0:
            return

        center = (NUM_BINS - 1) / 2.0  # 9.0

        # Decode accel from bin
        accel_norm = (actions[:, :n, 0].float() - center) / center  # [-1, 1]
        accel = accel_norm * self.a_max  # [-a_max, a_max]

        # Decode omega from bin
        omega_norm = (actions[:, :n, 1].float() - center) / center  # [-1, 1]
        new_omega = omega_norm * self.omega_max  # [-omega_max, omega_max]

        # Dynamic acceleration bounds (matching charge physics)
        max_accel = torch.clamp((self.v_max - self.vel[:, :n]) / self.dt, -self.a_max, self.a_max)
        min_accel = torch.clamp((-self.v_max - self.vel[:, :n]) / self.dt, -self.a_max, self.a_max)
        accel = torch.clamp(accel, min_accel, max_accel)

        # Update velocity
        new_vel = self.vel[:, :n] + accel * self.dt
        new_vel = torch.clamp(new_vel, -self.v_max, self.v_max)

        # Update heading
        new_heading = self.heading[:, :n] + new_omega * self.dt

        # Update position (forward kinematics)
        dx = new_vel * torch.cos(new_heading) * self.dt
        dy = new_vel * torch.sin(new_heading) * self.dt
        new_x = self.pos_x[:, :n] + dx
        new_y = self.pos_y[:, :n] + dy

        # Geofence: soft bounce at boundaries
        for dim_vals, new_vals in [(new_x, new_x), (new_y, new_y)]:
            over = new_vals > bound_limit
            under = new_vals < -bound_limit
            new_vals[over] = 2 * bound_limit - new_vals[over]
            new_vals[under] = -2 * bound_limit - new_vals[under]
            new_vals.clamp_(-bound_limit, bound_limit)

        # Apply only to alive spots
        alive_mask = self.alive[:, :n]
        self.pos_x[:, :n] = torch.where(alive_mask, new_x, self.pos_x[:, :n])
        self.pos_y[:, :n] = torch.where(alive_mask, new_y, self.pos_y[:, :n])
        self.vel[:, :n] = torch.where(alive_mask, new_vel, self.vel[:, :n])
        self.heading[:, :n] = torch.where(alive_mask, new_heading, self.heading[:, :n])
        self.omega[:, :n] = torch.where(alive_mask, new_omega, self.omega[:, :n])

    def write_to_scene(self, env_unwrapped, spot_slot_offset: int = 0):
        """Write virtual spot positions to Isaac Lab scene (obstacle slots).

        虛擬 spot 佔用 obstacle slot 的前 N 個:
          obstacle_{spot_slot_offset} ~ obstacle_{spot_slot_offset + max_spots - 1}
        """
        env_origins = env_unwrapped.scene.env_origins[:, :2]  # [E, 2]

        for i in range(self.max_spots):
            name = f"obstacle_{spot_slot_offset + i}"
            if name not in env_unwrapped.scene.keys():
                continue

            entity = env_unwrapped.scene[name]
            pos_w = entity.data.root_pos_w.clone()  # [E, 3]

            alive_i = self.alive[:, i]

            # Set world position
            pos_w[:, 0] = self.pos_x[:, i] + env_origins[:, 0]
            pos_w[:, 1] = self.pos_y[:, i] + env_origins[:, 1]
            pos_w[:, 2] = torch.where(alive_i, torch.ones_like(pos_w[:, 2]) * 0.5,
                                      torch.ones_like(pos_w[:, 2]) * -10.0)

            # Set heading as quaternion (yaw only)
            quat = torch.zeros(self.num_envs, 4, device=self.device)
            half_angle = self.heading[:, i] / 2.0
            quat[:, 0] = torch.cos(half_angle)  # w
            quat[:, 3] = torch.sin(half_angle)  # z

            pose = torch.cat([pos_w, quat], dim=-1)  # [E, 7]
            entity.write_root_pose_to_sim(pose)


# ============================================================================
# Analytical LiDAR (for virtual spots without PhysX raycaster)
# ============================================================================

def _analytical_lidar(
    spot_x: torch.Tensor,      # [E]
    spot_y: torch.Tensor,      # [E]
    spot_heading: torch.Tensor, # [E]
    obstacle_xy: torch.Tensor,  # [E, M, 2] all obstacle positions (including other spots, charge)
    obstacle_r: torch.Tensor,   # [E, M] obstacle radii
    obstacle_active: torch.Tensor,  # [E, M] bool
    wall_segments: Optional[torch.Tensor],  # [E, W, 4] (x1,y1,x2,y2) or None
    bound_limit: float,
    num_bins: int = LIDAR_DIM,
    max_range: float = 10.0,
    r_min: float = 0.5,
) -> torch.Tensor:
    """Compute analytical LiDAR scan from a spot's perspective.

    Returns: [E, num_bins] distance values, clipped to [r_min, max_range]
    """
    E = spot_x.shape[0]
    device = spot_x.device

    # Initialize with max range
    lidar = torch.full((E, num_bins), max_range, device=device)

    # Ray angles (relative to heading, spread over 360°)
    angles = torch.linspace(0, 2 * math.pi * (1 - 1 / num_bins), num_bins, device=device)
    # Absolute angles = heading + relative
    abs_angles = spot_heading.unsqueeze(-1) + angles.unsqueeze(0)  # [E, num_bins]

    ray_dx = torch.cos(abs_angles)  # [E, num_bins]
    ray_dy = torch.sin(abs_angles)  # [E, num_bins]

    # --- Check obstacles ---
    if obstacle_xy is not None and obstacle_xy.shape[1] > 0:
        M = obstacle_xy.shape[1]
        # Relative position from spot to each obstacle
        rel_x = obstacle_xy[:, :, 0] - spot_x.unsqueeze(-1)  # [E, M]
        rel_y = obstacle_xy[:, :, 1] - spot_y.unsqueeze(-1)  # [E, M]

        # For each ray, compute closest intersection with each circular obstacle
        # Project obstacle center onto ray: t = dot(rel, ray_dir)
        # Distance to ray axis: d_perp = |cross(rel, ray_dir)|
        for m in range(M):
            rx = rel_x[:, m]  # [E]
            ry = rel_y[:, m]  # [E]
            r = obstacle_r[:, m]  # [E]
            active = obstacle_active[:, m]  # [E]

            # Projection along each ray
            t = rx.unsqueeze(-1) * ray_dx + ry.unsqueeze(-1) * ray_dy  # [E, num_bins]
            # Perpendicular distance
            d_perp = (rx.unsqueeze(-1) * ray_dy - ry.unsqueeze(-1) * ray_dx).abs()  # [E, num_bins]

            # Hit if d_perp < r and t > 0 (in front)
            hit = (d_perp < r.unsqueeze(-1)) & (t > 0) & active.unsqueeze(-1)

            # Distance to hit point (simplified: t - sqrt(r^2 - d_perp^2))
            inside = (r.unsqueeze(-1) ** 2 - d_perp ** 2).clamp(min=0)
            hit_dist = t - inside.sqrt()
            hit_dist = hit_dist.clamp(min=r_min)

            lidar = torch.where(hit & (hit_dist < lidar), hit_dist, lidar)

    # --- Check boundary walls (4 sides) ---
    # North wall: y = +bound_limit
    # South wall: y = -bound_limit
    # East wall: x = +bound_limit
    # West wall: x = -bound_limit
    for wall_pos, dim, ray_d in [
        (bound_limit, 1, ray_dy),    # North
        (-bound_limit, 1, ray_dy),   # South
        (bound_limit, 0, ray_dx),    # East
        (-bound_limit, 0, ray_dx),   # West
    ]:
        spot_coord = spot_y if dim == 1 else spot_x
        dist_to_wall = wall_pos - spot_coord  # [E]
        # t = dist_to_wall / ray_d (only valid when ray_d has same sign)
        t_wall = dist_to_wall.unsqueeze(-1) / (ray_d + 1e-8)  # [E, num_bins]
        valid = t_wall > r_min
        lidar = torch.where(valid & (t_wall < lidar), t_wall, lidar)

    # Clamp to range
    lidar = lidar.clamp(r_min, max_range)

    return lidar  # [E, num_bins]


# ============================================================================
# Build Virtual Spot Observations (79D, matching charge)
# ============================================================================

def build_virtual_spot_obs(
    vs_state: VirtualSpotState,
    env_unwrapped,
    num_active: int,
    charge_pos_local: torch.Tensor,  # [E, 2] charge robot local xy
    charge_vel: torch.Tensor,        # [E, 2] charge robot (vx, vy) world
    goal_pos_local: torch.Tensor,    # [E, 2] current goal local xy
    obstacle_pos_local: torch.Tensor,  # [E, N_obs, 2] regular obstacle local xy
    obstacle_radii: torch.Tensor,      # [E, N_obs] obstacle radii
    obstacle_active: torch.Tensor,     # [E, N_obs] bool
    episode_remaining: torch.Tensor,   # [E] time remaining ratio
    bound_limit: float = 7.0,
    max_range: float = 10.0,
    r_min: float = 0.5,
) -> torch.Tensor:
    """Build 79D observations for all active virtual spots.

    Obs layout (matching charge):
      ego(4): [accel, vel, omega, radius]
      goal(2): [rel_goal_x, rel_goal_y] (body frame)
      lidar(72): analytical LiDAR distances
      time(1): episode remaining ratio

    Returns: [num_envs, num_active, 79]
    """
    E = vs_state.num_envs
    n = min(num_active, vs_state.max_spots)
    device = vs_state.device

    if n == 0:
        return torch.zeros(E, 0, USED_OBS_DIM, device=device)

    obs = torch.zeros(E, n, USED_OBS_DIM, device=device)

    for s in range(n):
        alive = vs_state.alive[:, s]  # [E]

        # --- Ego (4D) ---
        # accel: approximate from velocity change (0 at first step)
        obs[:, s, 0] = 0.0  # accel (will be computed in training loop from prev vel)
        obs[:, s, 1] = vs_state.vel[:, s]  # forward velocity
        obs[:, s, 2] = vs_state.omega[:, s]  # angular velocity
        obs[:, s, 3] = vs_state.body_radius  # collision radius

        # --- Goal (2D) — relative to spot, in body frame ---
        heading = vs_state.heading[:, s]
        dx = goal_pos_local[:, 0] - vs_state.pos_x[:, s]
        dy = goal_pos_local[:, 1] - vs_state.pos_y[:, s]
        # Rotate to body frame
        cos_h = torch.cos(-heading)
        sin_h = torch.sin(-heading)
        obs[:, s, 4] = dx * cos_h - dy * sin_h  # body-frame goal x
        obs[:, s, 5] = dx * sin_h + dy * cos_h  # body-frame goal y

        # --- LiDAR (72D) — analytical ---
        # Collect all obstacles for this spot's LiDAR
        # Obstacles = charge robot + other virtual spots + regular obstacles
        all_obs_list = []
        all_r_list = []
        all_active_list = []

        # Charge robot as obstacle
        all_obs_list.append(charge_pos_local.unsqueeze(1))  # [E, 1, 2]
        all_r_list.append(torch.full((E, 1), vs_state.body_radius, device=device))
        all_active_list.append(torch.ones(E, 1, dtype=torch.bool, device=device))

        # Other virtual spots as obstacles
        for other_s in range(n):
            if other_s == s:
                continue
            other_xy = torch.stack([vs_state.pos_x[:, other_s],
                                    vs_state.pos_y[:, other_s]], dim=-1)  # [E, 2]
            all_obs_list.append(other_xy.unsqueeze(1))
            all_r_list.append(torch.full((E, 1), vs_state.body_radius, device=device))
            all_active_list.append(vs_state.alive[:, other_s].unsqueeze(1))

        # Regular obstacles
        if obstacle_pos_local is not None and obstacle_pos_local.shape[1] > 0:
            all_obs_list.append(obstacle_pos_local)
            all_r_list.append(obstacle_radii)
            all_active_list.append(obstacle_active)

        all_obs_xy = torch.cat(all_obs_list, dim=1)  # [E, M, 2]
        all_obs_r = torch.cat(all_r_list, dim=1)  # [E, M]
        all_obs_active = torch.cat(all_active_list, dim=1)  # [E, M]

        lidar = _analytical_lidar(
            vs_state.pos_x[:, s], vs_state.pos_y[:, s], heading,
            all_obs_xy, all_obs_r, all_obs_active,
            None, bound_limit, LIDAR_DIM, max_range, r_min,
        )  # [E, 72]
        obs[:, s, 6:78] = lidar

        # --- Time (1D) ---
        obs[:, s, 78] = episode_remaining

        # Zero out dead spots
        if not alive.all():
            obs[~alive, s, :] = 0.0

    return obs  # [E, n, 79]


# ============================================================================
# Collision Detection (analytical, for virtual spots)
# ============================================================================

def check_virtual_spot_collisions(
    vs_state: VirtualSpotState,
    num_active: int,
    charge_pos_local: torch.Tensor,  # [E, 2]
    obstacle_pos_local: torch.Tensor,  # [E, N_obs, 2]
    obstacle_radii: torch.Tensor,  # [E, N_obs]
    obstacle_active: torch.Tensor,  # [E, N_obs]
    charge_radius: float = 0.35,
    collision_threshold: float = 0.45,  # body_radius + buffer
    bound_limit: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Check collisions for each virtual spot.

    Returns:
        goal_reached: [E, n] (placeholder, filled by caller)
        collision: [E, n] bool — hit obstacle/wall/charge/other spot
        wall_collision: [E, n] bool — hit boundary wall
    """
    E = vs_state.num_envs
    n = min(num_active, vs_state.max_spots)
    device = vs_state.device

    collision = torch.zeros(E, n, dtype=torch.bool, device=device)
    wall_collision = torch.zeros(E, n, dtype=torch.bool, device=device)
    goal_reached = torch.zeros(E, n, dtype=torch.bool, device=device)

    for s in range(n):
        alive = vs_state.alive[:, s]
        sx = vs_state.pos_x[:, s]
        sy = vs_state.pos_y[:, s]

        # --- Wall collision ---
        wall_hit = (sx.abs() > bound_limit - collision_threshold) | \
                   (sy.abs() > bound_limit - collision_threshold)
        wall_collision[:, s] = wall_hit & alive

        # --- Charge collision ---
        dx = sx - charge_pos_local[:, 0]
        dy = sy - charge_pos_local[:, 1]
        dist = (dx ** 2 + dy ** 2).sqrt()
        charge_hit = dist < (collision_threshold + charge_radius)
        collision[:, s] = collision[:, s] | (charge_hit & alive)

        # --- Other virtual spot collision ---
        for other in range(n):
            if other == s:
                continue
            other_alive = vs_state.alive[:, other]
            dx = sx - vs_state.pos_x[:, other]
            dy = sy - vs_state.pos_y[:, other]
            dist = (dx ** 2 + dy ** 2).sqrt()
            spot_hit = dist < (2 * collision_threshold)
            collision[:, s] = collision[:, s] | (spot_hit & alive & other_alive)

        # --- Obstacle collision ---
        if obstacle_pos_local is not None and obstacle_pos_local.shape[1] > 0:
            N_obs = obstacle_pos_local.shape[1]
            dx = sx.unsqueeze(-1) - obstacle_pos_local[:, :, 0]  # [E, N_obs]
            dy = sy.unsqueeze(-1) - obstacle_pos_local[:, :, 1]
            dist = (dx ** 2 + dy ** 2).sqrt()
            obs_thresh = collision_threshold + obstacle_radii  # [E, N_obs]
            obs_hit = (dist < obs_thresh) & obstacle_active
            collision[:, s] = collision[:, s] | obs_hit.any(dim=-1) & alive

        # Combine
        collision[:, s] = collision[:, s] | wall_collision[:, s]

    return goal_reached, collision, wall_collision


def check_virtual_spot_goal_reached(
    vs_state: VirtualSpotState,
    num_active: int,
    goal_pos_local: torch.Tensor,  # [E, 2]
    goal_radius: float = 0.5,
) -> torch.Tensor:
    """Check if any virtual spot reached the goal.

    Returns: [E, n] bool
    """
    n = min(num_active, vs_state.max_spots)
    E = vs_state.num_envs
    device = vs_state.device
    reached = torch.zeros(E, n, dtype=torch.bool, device=device)

    for s in range(n):
        alive = vs_state.alive[:, s]
        dx = vs_state.pos_x[:, s] - goal_pos_local[:, 0]
        dy = vs_state.pos_y[:, s] - goal_pos_local[:, 1]
        dist = (dx ** 2 + dy ** 2).sqrt()
        reached[:, s] = (dist < goal_radius) & alive

    return reached


# ============================================================================
# Virtual Spot Reward (WD style sparse)
# ============================================================================

def compute_virtual_spot_reward(
    goal_reached: torch.Tensor,  # [E, n]
    collision: torch.Tensor,  # [E, n]
    penalty_hit: float = -5.0,
    reward_get_goal: float = 40.0,
) -> torch.Tensor:
    """Compute WD-style sparse reward for virtual spots.

    Returns: [E, n] reward
    """
    reward = torch.zeros_like(goal_reached, dtype=torch.float32)
    reward[goal_reached] = reward_get_goal
    reward[collision] = penalty_hit
    return reward


# ============================================================================
# Pad 79D → 139D (for shared policy compatibility)
# ============================================================================

def pad_obs_79_to_139(obs_79: torch.Tensor) -> torch.Tensor:
    """Pad 79D virtual spot obs to 139D for compatibility with charge's extractor.

    Layout: ego(4) + goal(2) + lidar(72) + zeros(60) + time(1) = 139D
    The extractor only uses indices [0:78] + [138], so padding doesn't affect policy.
    """
    B = obs_79.shape[0]
    device = obs_79.device

    # obs_79 = [ego(4), goal(2), lidar(72), time(1)]
    obs_139 = torch.zeros(B, FULL_OBS_DIM, device=device)
    obs_139[:, :78] = obs_79[:, :78]       # ego + goal + lidar
    obs_139[:, 138] = obs_79[:, 78]        # time
    return obs_139


# ============================================================================
# Helper: Extract obstacle positions from scene
# ============================================================================

def get_obstacle_positions(env_unwrapped, slot_start: int, slot_end: int,
                           device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get obstacle positions, radii, and active mask from scene.

    Returns:
        pos_local: [E, N, 2] local xy
        radii: [E, N] collision radius
        active: [E, N] bool (Z > 0)
    """
    E = env_unwrapped.num_envs
    env_origins = env_unwrapped.scene.env_origins[:, :2]
    N = slot_end - slot_start

    pos_local = torch.zeros(E, N, 2, device=device)
    radii = torch.zeros(E, N, device=device)
    active = torch.zeros(E, N, dtype=torch.bool, device=device)

    for i in range(N):
        name = f"obstacle_{slot_start + i}"
        if name not in env_unwrapped.scene.keys():
            continue
        entity = env_unwrapped.scene[name]
        pos_w = entity.data.root_pos_w[:, :3]
        pos_local[:, i, :] = pos_w[:, :2] - env_origins
        active[:, i] = pos_w[:, 2] > 0.0

        # Estimate radius from collision geometry or default
        radii[:, i] = 0.3  # default obstacle radius

    return pos_local, radii, active
