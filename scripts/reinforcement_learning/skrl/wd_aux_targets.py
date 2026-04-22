"""
wd_aux_targets.py — WD-style 7D Privileged Geometry Target + Module Loss

Port from Warp Drive:
  - Target: new_warp_drive/custom_envs/spot_3d/spot_3d_step.cu (preprocess_data_arr)
  - Loss:   new_warp_drive/warp_drive/training/algorithms/module_loss.py

7D target layout (per env):
  [near_1_x, near_1_y, near_1_d,   # nearest obstacle body-frame (x, y, surface_dist)
   near_2_x, near_2_y, near_2_d,   # 2nd nearest obstacle body-frame (x, y, surface_dist)
   timestep]                        # current episode step (raw integer, not normalized)

Design:
  - Training-only privileged signal — NOT available at inference
  - Target comes from simulator geometry, NOT from policy observation
  - Coordinate frame: robot body-frame (forward=+x, left=+y)
  - Distance: surface distance (center_dist - obj_radius - robot_radius), clamped >= 0
  - Only considers active obstacles (Z > 0)
  - Missing obstacles filled with default=10.0 (WD convention: far away)

WD bug fix:
  - WD spot_3d_step.cu line 671: dim[4] writes obs_spot_x_ instead of obs_spot_y_
  - This port correctly writes (x, y, d) for both nearest obstacles
"""

import torch
import torch.nn as nn


# WD default weight: [1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.0]
# dim 0-2: nearest obstacle (full weight)
# dim 3-5: 2nd nearest obstacle (0.7 weight)
# dim 6: timestep (weight=0, connectivity check only)
WD_DEFAULT_WEIGHT = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.0]

# WD default fill value for missing obstacles (spot_3d_step.cu lines 219-224)
_FAR_DEFAULT = 10.0


def build_wd_preprocess_targets(
    env_unwrapped,
    max_obstacles: int,
    device: torch.device,
    robot_radius: float = 0.33,
) -> torch.Tensor:
    """Build WD-style 7D privileged geometry target from simulator state.

    This is a training-only privileged signal. At inference time this function
    is NOT called — the predict_head output is discarded.

    Target comes from simulator geometry (scene entity positions), not from
    policy observations.

    Geometry sources (reusing existing Isaac Lab patterns):
      - Robot pos/quat: env.scene["robot"].data.root_pos_w / root_quat_w
      - Obstacle pos:   env._obs_policy_cache[i].data.root_pos_w
      - Obstacle radii:  env._obstacle_radii[:, i]
      - Episode step:   env.episode_length_buf

    Body-frame rotation follows obs_functions.py convention:
      px_body =  cos(yaw) * dx + sin(yaw) * dy
      py_body = -sin(yaw) * dx + cos(yaw) * dy

    Args:
        env_unwrapped: unwrapped Isaac Lab env (ManagerBasedRLEnv)
        max_obstacles: maximum number of obstacle entities in scene
        device: torch device
        robot_radius: robot body radius for surface distance (m)

    Returns:
        [num_envs, 7] float tensor — WD preprocess_real_data equivalent
    """
    num_envs = env_unwrapped.num_envs

    # --- Robot state ---
    robot_pos_w = env_unwrapped.scene["robot"].data.root_pos_w[:, :2]  # [E, 2]
    robot_quat = env_unwrapped.scene["robot"].data.root_quat_w         # [E, 4] (w,x,y,z)

    # Yaw extraction (obs_functions.py convention)
    w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))  # [E]
    cos_yaw = torch.cos(yaw)  # [E]
    sin_yaw = torch.sin(yaw)  # [E]

    # --- Build obstacle cache if needed ---
    if not hasattr(env_unwrapped, "_obs_policy_cache"):
        env_unwrapped._obs_policy_cache = []
        for i in range(max_obstacles):
            name = f"obstacle_{i}"
            if name in env_unwrapped.scene.keys():
                env_unwrapped._obs_policy_cache.append(env_unwrapped.scene[name])
            else:
                env_unwrapped._obs_policy_cache.append(None)

    # --- Collect all obstacle body-frame positions + surface distances ---
    # Shape: [E, N_active, 3] where 3 = (body_x, body_y, surface_d)
    n_found = 0
    # Pre-allocate for max_obstacles
    all_body_x = torch.full((num_envs, max_obstacles), _FAR_DEFAULT, device=device)
    all_body_y = torch.full((num_envs, max_obstacles), _FAR_DEFAULT, device=device)
    all_surf_d = torch.full((num_envs, max_obstacles), _FAR_DEFAULT, device=device)
    all_active = torch.zeros(num_envs, max_obstacles, dtype=torch.bool, device=device)

    # Obstacle radii (per-env, per-obstacle)
    has_radii = hasattr(env_unwrapped, "_obstacle_radii")

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue
        if i >= max_obstacles:
            break

        obs_pos_w = obstacle.data.root_pos_w  # [E, 3]
        active = obs_pos_w[:, 2] > 0.0        # hidden obstacles at Z=-10

        if not active.any():
            continue

        # World-frame displacement
        dx_w = obs_pos_w[:, 0] - robot_pos_w[:, 0]  # [E]
        dy_w = obs_pos_w[:, 1] - robot_pos_w[:, 1]  # [E]

        # Body-frame rotation (obs_functions.py: R(yaw)^T * delta)
        bx = cos_yaw * dx_w + sin_yaw * dy_w         # [E]
        by = -sin_yaw * dx_w + cos_yaw * dy_w        # [E]

        # Center distance
        center_d = torch.sqrt(bx * bx + by * by).clamp(min=1e-6)  # [E]

        # Obstacle radius
        if has_radii:
            # _obstacle_radii stores collision distance (center-to-center threshold)
            # = obj_radius + robot_radius approximately, so obj_radius ≈ radii/2
            # But for WD convention: surface_d = center_d - obj_radius - robot_radius
            # We use _obstacle_radii as the total collision distance directly
            obs_r = env_unwrapped._obstacle_radii[:, i]  # [E]
            # surface_d = center_d - collision_radius (which already includes robot+obj)
            # But WD uses: center_d - obj_radius - robot_radius
            # _obstacle_radii ≈ obs_collision_base (default 0.9) which is ~obj_r + robot_r
            # So surface_d = center_d - _obstacle_radii is close to WD convention
            surf_d = (center_d - obs_r).clamp(min=0.0)
        else:
            # Fallback: assume obstacle radius ≈ 0.5m
            surf_d = (center_d - 0.5 - robot_radius).clamp(min=0.0)

        all_body_x[:, i] = torch.where(active, bx, torch.full_like(bx, _FAR_DEFAULT))
        all_body_y[:, i] = torch.where(active, by, torch.full_like(by, _FAR_DEFAULT))
        all_surf_d[:, i] = torch.where(active, surf_d, torch.full_like(surf_d, _FAR_DEFAULT))
        all_active[:, i] = active
        n_found += 1

    # --- Sort by surface distance, take nearest 2 ---
    # For inactive obstacles, surf_d = _FAR_DEFAULT = 10.0, so they sort last
    _, sort_idx = all_surf_d.sort(dim=1)  # [E, N], ascending

    # Gather sorted body-frame coordinates
    near1_idx = sort_idx[:, 0]  # [E]
    near2_idx = sort_idx[:, 1] if max_obstacles >= 2 else sort_idx[:, 0]

    # Advanced indexing: gather per-env
    batch_idx = torch.arange(num_envs, device=device)

    near1_x = all_body_x[batch_idx, near1_idx]
    near1_y = all_body_y[batch_idx, near1_idx]
    near1_d = all_surf_d[batch_idx, near1_idx]

    near2_x = all_body_x[batch_idx, near2_idx]
    near2_y = all_body_y[batch_idx, near2_idx]  # WD bug fix: correctly use y, not x
    near2_d = all_surf_d[batch_idx, near2_idx]

    # --- Timestep (raw integer, WD: env_timestep_arr) ---
    # WD: env_timestep_arr is 1-indexed (incremented at START of CudaSpot_3dStep,
    #   before obs/preprocess are computed). After reset → 0, first step → 1.
    #   Source: spot_3d_step.cu line 1181: env_timestep_arr[kEnvId] += 1
    # IL: episode_length_buf is incremented in env.step() post-step phase, then
    #   done envs are reset to 0. We read it AFTER env.step(), so:
    #   - Non-done envs: step count (1-indexed, same as WD)
    #   - Done envs: 0 (post-reset, same as WD; excluded from aux loss by done filter)
    timestep = env_unwrapped.episode_length_buf.float()  # [E]

    # --- Assemble 7D target ---
    target = torch.stack([
        near1_x, near1_y, near1_d,
        near2_x, near2_y, near2_d,
        timestep,
    ], dim=-1)  # [E, 7]

    return target


def compute_wd_module_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: list[float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute WD module prediction loss — direct port from PreProcess_Module_Loss.

    Source: new_warp_drive/warp_drive/training/algorithms/module_loss.py

    Per-dimension formula (BRANCHED by index):
      raw_L1    = |pred[..., i] - target[..., i].detach()|    (element-wise)

      if idx < 6:   (obstacle dims)
        clamped   = clamp(raw_L1, min=0.01)                   (floor to avoid log(0))
        loss_p    = log(clamped)                               (log scale)
      else:          (goal/timestep dims, idx >= 6)
        loss_p    = raw_L1 * raw_L1                            (squared L1)

      weighted  = loss_p * weight[i]                           (per-dim weight)
      dim_loss  = mean(weighted)                               (reduce to scalar)

    Total: preprocess_loss = sum(dim_loss for all dims)

    Source: module_loss.py lines 24-27 — `if idx < 6: log(clamp(L1))` else `L1 * L1`.

    Except for the target semantic fix (dim[4] = y not x), this loss formula is
    unchanged from WD original.

    Args:
        pred:   [B, 7] model prediction
        target: [B, 7] privileged geometry target (will be detached internally)
        weight: per-dim weight list, length 7. Default: [1,1,1, 0.7,0.7,0.7, 0]

    Returns:
        (preprocess_loss, display_loss) where:
          preprocess_loss: scalar tensor (backprop-ready)
          display_loss: dict with per-dim and total loss values
            "module_feture_{i}_loss": float  (WD original typo preserved)
            "preprcess_loss": tensor          (WD original typo preserved)
    """
    if weight is None:
        weight = WD_DEFAULT_WEIGHT

    weight_len = pred.size(-1)
    assert len(weight) == weight_len, (
        f"Weight length {len(weight)} must match pred dim {weight_len}"
    )

    display_loss: dict[str, float | torch.Tensor] = {}
    preprocess_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    for idx in range(weight_len):
        input_flat = pred[..., idx].reshape(-1)
        target_flat = target[..., idx].reshape(-1).detach()

        # Element-wise L1
        loss_ = nn.L1Loss(reduction='none')(input_flat, target_flat)

        # WD branch: idx < 6 → log(clamp(L1)), idx >= 6 → L1² (module_loss.py:24-27)
        if idx < 6:
            loss_ = torch.clamp(loss_, min=0.01)
            loss_ = torch.log(loss_)
        else:
            loss_ = loss_ * loss_

        # Apply per-dim weight
        loss_ = loss_ * weight[idx]

        display_loss[f"module_feture_{idx}_loss"] = loss_.mean().item()
        preprocess_loss = preprocess_loss + loss_.mean()

    display_loss["preprcess_loss"] = preprocess_loss

    return preprocess_loss, display_loss
