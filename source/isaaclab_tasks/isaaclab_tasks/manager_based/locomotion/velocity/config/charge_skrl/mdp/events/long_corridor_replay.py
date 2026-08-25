"""Reset injector for a 10 m interaction corridor sealed to room boundaries."""

from __future__ import annotations

import math

import torch

from .corridor_density import (
    CENTER_STRIP_X,
    INTERACTION_INDEPENDENT,
    INTERACTION_NAMES,
    INTERACTION_SIDE_BY_SIDE,
    INTERACTION_CROSSING,
    NO_PAIR,
    apply_interaction_geometry,
    assign_families_and_pairs,
    audit_installed_pairs,
    crossing_axis_and_side,
    crossing_progress,
    sample_interaction_types,
    validate_pair_families,
    MAX_CORRIDOR_DYNAMIC,
    MAX_CORRIDOR_STATIC,
    active_masks_from_counts,
    dynamic_layout_for_families,
    dynamic_waypoints_for_families,
    sample_density_counts_systematic,
    sample_speed_density_profiles_systematic,
    validate_density_mix,
    validate_speed_density_mix,
)
from .long_corridor_replay_geometry import (
    MIXED_MAX_DYNAMIC,
    MIXED_MAX_STATIC,
    MOTION_RANDOM_2D,
    LongCorridorSpec,
    layout_is_constructively_solvable,
    apply_pause_override,
    sample_conflict_free_layout,
    corridor_penetration_masks,
    corridor_wander_bounds,
    validate_random_2d_kinematics,
    sample_dynamic_trajectories,
    normalize_dynamic_motion_weights,
    sample_obstacle_layout,
    static_layout_id,
    validate_dynamic_motion_mode,
    validate_obstacle_counts,
    validate_spec,
    wall_boundary_overlap,
    wall_geometry,
)


#: 正式 Gate 的題型：固定 4 靜態 + 2 動態，四個模式各自純化（不混 family）。
_GATE_ALIGNED_STATIC = 4
_GATE_ALIGNED_DYNAMIC = 2
_GATE_ALIGNED_MODES = ("lateral", "longitudinal", "random_2d", "mixed_iid")

_ASSET_NAMES = ("long_corridor_wall_0", "long_corridor_wall_1")
_HIDDEN_Z = -10.0
_INTERACTION_OVERRIDE_IDS = {
    "independent": INTERACTION_INDEPENDENT,
    "crossing": INTERACTION_CROSSING,
    "side_by_side": INTERACTION_SIDE_BY_SIDE,
}


def _normalize_interaction_override(value: str | None) -> int | None:
    """Translate the eval-only interaction override into its sampler ID."""
    if value is None or str(value).strip() in ("", "sample"):
        return None
    normalized = str(value).strip().lower()
    if normalized not in _INTERACTION_OVERRIDE_IDS:
        raise ValueError(
            "interaction_override must be sample, independent, crossing, or "
            f"side_by_side; got {value!r}"
        )
    return _INTERACTION_OVERRIDE_IDS[normalized]


def configure_long_corridor_assets(
    env_cfg,
    *,
    fraction: float,
    free_width: float = 4.0,
    length: float = 10.0,
    room_half_extent: float,
    boundary_wall_width: float = 1.0,
    static_obstacles: int = 4,
    dynamic_obstacles: int = 2,
    dynamic_speed_range: tuple[float, float] = (0.30, 0.60),
    dynamic_motion_mode: str = "lateral",
    dynamic_motion_weights: tuple[float, float, float] | None = None,
    random_2d_kinematics: str = "patrol",
    obstacle_count_mix=None,
    speed_density_mix=None,
    interaction_override: str | None = None,
    gate_aligned_share: float = 0.0,
) -> None:
    """Add dedicated corridor walls and configure the reset event.

    ``obstacle_count_mix`` 不是 None 時走高密度混合場：每個 env 的 (S, D) 由
    這組權重逐 env 抽，``static_obstacles`` / ``dynamic_obstacles`` 失效。
    """
    from isaaclab.assets import RigidObjectCfg
    import isaaclab.sim as sim_utils

    wall_span_length = 2.0 * float(room_half_extent)
    spec = LongCorridorSpec(
        free_width=float(free_width),
        length=float(length),
        wall_span_length=wall_span_length,
    )
    validate_spec(spec)
    boundary_overlap = wall_boundary_overlap(
        spec,
        room_half_extent=float(room_half_extent),
        boundary_wall_width=float(boundary_wall_width),
    )
    if boundary_overlap < 0.0:
        raise ValueError(
            "long-corridor side walls do not reach the north/south "
            f"boundary walls: overlap={boundary_overlap:.3f}m"
        )
    if obstacle_count_mix is not None and speed_density_mix is not None:
        raise ValueError(
            "obstacle_count_mix and speed_density_mix are mutually exclusive"
        )
    if obstacle_count_mix is None and speed_density_mix is None:
        validate_obstacle_counts(static_obstacles, dynamic_obstacles)
        required_scheduler_capacity = max(
            int(static_obstacles), 4 + int(dynamic_obstacles)
        )
    else:
        if speed_density_mix is not None:
            validate_speed_density_mix(speed_density_mix)
        else:
            validate_density_mix(obstacle_count_mix)
        required_scheduler_capacity = (
            MAX_CORRIDOR_STATIC + MAX_CORRIDOR_DYNAMIC
        )
    motion_mode = validate_dynamic_motion_mode(dynamic_motion_mode)
    motion_weights = normalize_dynamic_motion_weights(
        dynamic_motion_weights, motion_mode
    )
    kinematics = validate_random_2d_kinematics(random_2d_kinematics)
    override_id = _normalize_interaction_override(interaction_override)
    if (
        override_id is not None
        and obstacle_count_mix is None
        and speed_density_mix is None
    ):
        raise ValueError(
            "interaction_override requires obstacle_count_mix or speed_density_mix"
        )
    speed_min, speed_max = map(float, dynamic_speed_range)
    if not (0.0 < speed_min <= speed_max):
        raise ValueError("dynamic corridor speed range must be positive and ordered")

    # BehaviorScheduler historically sized its tensors to the current stage's
    # active obstacle count. SA1 has only two active slots, while corridor
    # replay can require ten. Reserve tensor capacity without changing how many
    # obstacles the native stage activates.
    current_capacity = int(
        getattr(env_cfg, "behavior_scheduler_capacity", 0) or 0
    )
    env_cfg.behavior_scheduler_capacity = max(
        current_capacity, required_scheduler_capacity
    )

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True, disable_gravity=True
    )
    collision_props = sim_utils.CollisionPropertiesCfg()
    visual = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.32, 0.38, 0.42), metallic=0.1
    )
    for index, name in enumerate(_ASSET_NAMES):
        setattr(
            env_cfg.scene,
            name,
            RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Wall_LongCorridor_{index}",
                spawn=sim_utils.CuboidCfg(
                    size=(
                        spec.wall_thickness,
                        spec.physical_wall_span,
                        spec.wall_height,
                    ),
                    rigid_props=rigid_props,
                    collision_props=collision_props,
                    visual_material=visual,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, _HIDDEN_Z)
                ),
            ),
        )

    event = getattr(env_cfg.events, "long_corridor_replay", None)
    if event is None:
        raise RuntimeError("environment config has no long_corridor_replay reset event")
    event.params.update(
        {
            "fraction": float(fraction),
            "free_width": spec.free_width,
            "length": spec.length,
            "wall_span_length": spec.physical_wall_span,
            "static_obstacles": int(static_obstacles),
            "dynamic_obstacles": int(dynamic_obstacles),
            "dynamic_speed_min": speed_min,
            "dynamic_speed_max": speed_max,
            "dynamic_motion_mode": motion_mode,
            "dynamic_motion_weights": motion_weights,
            # Must ride in the event params: the reset event re-invokes
            # setup_long_corridor_replay on every auto-reset, and a missing
            # key silently falls back to "patrol", overwriting the direct
            # call's setting after the very first reset. That exact bug
            # produced a "wander" gate whose episodes were patrol from the
            # second reset onward.
            "random_2d_kinematics": kinematics,
            # 同上，必須隨 event params 走：auto-reset 會重跑 setup，
            # 少了這個 key 會從第二次 reset 起悄悄掉回 legacy 密度。
            "obstacle_count_mix": obstacle_count_mix,
            # Joint speed-density curriculum. None is the historical no-op.
            "speed_density_mix": speed_density_mix,
            # 同上，必須隨 event params 走：auto-reset 會重跑 setup。
            "gate_aligned_share": float(gate_aligned_share),
            # Eval/GUI only. None keeps the production 60/20/20 sampler.
            "interaction_override": interaction_override,
        }
    )
    # 密度描述必須反映**實際生效**的那一組：混合場一開，
    # static_obstacles/dynamic_obstacles 就失效了，照印會讓 log 說謊。
    if speed_density_mix is not None:
        density = "speed_density_mix[" + ",".join(
            f"{s}S{d}D@{lo:.2f}-{hi:.2f}:{w:.2f}"
            for (s, d), (lo, hi), w in speed_density_mix
        ) + "]"
    elif obstacle_count_mix is None:
        density = f"{static_obstacles}S+{dynamic_obstacles}D"
    else:
        density = "mix[" + ",".join(
            f"{s}S{d}D:{w:.2f}" for (s, d), w in obstacle_count_mix
        ) + "]"
    print(
        "[LONG-CORRIDOR-CONFIG] "
        f"fraction={fraction:.3f} free_width={spec.free_width:.2f}m "
        f"interaction_length={spec.length:.2f}m "
        f"wall_span={spec.physical_wall_span:.2f}m "
        f"boundary_overlap={boundary_overlap:.2f}m "
        f"obstacles={density} "
        f"speed=[{speed_min:.2f},{speed_max:.2f}]m/s "
        f"motion={motion_mode} random_2d_kinematics={kinematics} "
        f"interaction_override={interaction_override or 'sample'} "
        f"scheduler_capacity={env_cfg.behavior_scheduler_capacity} "
        f"configured_motion_weights={motion_weights} reward_unchanged=True",
        flush=True,
    )


def _install_corridor_wander(
    scheduler,
    selected: torch.Tensor,
    dynamic_slots: torch.Tensor,
    wander_mask: torch.Tensor,
    spec: LongCorridorSpec,
    speeds: torch.Tensor,
) -> None:
    """Switch the masked dynamic slots from patrol to a bounded random walk."""
    from .behavior_scheduler import BEHAVIOR_RANDOM_WALK

    device = scheduler.device
    # ``dynamic_slots`` is a slice into the obstacle axis; materialise it so the
    # masked slots can be gathered as explicit [env, slot] index pairs.
    slot_ids = torch.arange(
        dynamic_slots.start, dynamic_slots.stop, device=device, dtype=torch.long
    )
    env_idx = selected[:, None].expand_as(wander_mask)[wander_mask]
    slot_idx = slot_ids[None, :].expand_as(wander_mask)[wander_mask]
    count = int(env_idx.numel())
    if count == 0:
        return

    scheduler.behavior_type[env_idx, slot_idx] = BEHAVIOR_RANDOM_WALK
    # Patrol state must be cleared, otherwise a stale waypoint could be read by
    # any downstream consumer that keys off the slot rather than the behaviour.
    scheduler.patrol_pause_remaining[env_idx, slot_idx] = 0
    scheduler.patrol_num_waypoints[env_idx, slot_idx] = 0

    x_min, x_max, y_min, y_max = corridor_wander_bounds(spec)
    bounds = torch.tensor(
        [x_min, x_max, y_min, y_max], device=device, dtype=scheduler.rw_bounds.dtype
    )
    scheduler.rw_bounds[env_idx, slot_idx] = bounds
    # Static-conflict resolver participation: a blind wander penetrated a
    # static obstacle in 16% of measured slot-frames. Two shared radii =
    # centre distance at exact touch. Dynamic-dynamic overlap is deliberately
    # not resolved (independent scripted pedestrians; audited, not blocked).
    scheduler.pairwise_clearance[env_idx, slot_idx] = 2.0 * spec.obstacle_radius

    heading = torch.rand(count, device=device) * 2.0 * math.pi
    slot_speeds = speeds[wander_mask]
    scheduler.rw_heading[env_idx, slot_idx] = heading
    scheduler.rw_target_heading[env_idx, slot_idx] = heading
    scheduler.rw_speed[env_idx, slot_idx] = slot_speeds
    scheduler.rw_timer[env_idx, slot_idx] = 0
    scheduler.rw_turn_remaining[env_idx, slot_idx] = 0
    cfg = scheduler.cfg.random_walk
    scheduler.rw_change_interval[env_idx, slot_idx] = torch.randint(
        cfg.direction_change_interval_range[0],
        cfg.direction_change_interval_range[1] + 1,
        (count,),
        device=device,
    )
    scheduler.velocities[env_idx, slot_idx, 0] = slot_speeds * torch.cos(heading)
    scheduler.velocities[env_idx, slot_idx, 1] = slot_speeds * torch.sin(heading)
    # Start inside the reflecting box so the very first step cannot begin out
    # of bounds.
    scheduler.positions[env_idx, slot_idx, 0] = scheduler.positions[
        env_idx, slot_idx, 0
    ].clamp(x_min, x_max)
    scheduler.positions[env_idx, slot_idx, 1] = scheduler.positions[
        env_idx, slot_idx, 1
    ].clamp(y_min, y_max)


def _as_env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _ensure_state(env) -> None:
    if hasattr(env, "_long_corridor_wall_centers"):
        return
    env._long_corridor_wall_centers = torch.zeros(
        env.num_envs, 2, 2, device=env.device
    )
    env._long_corridor_wall_sizes = torch.zeros(
        env.num_envs, 2, 2, device=env.device
    )
    env._long_corridor_wall_mask = torch.zeros(
        env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    env._long_corridor_active = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._long_corridor_pending_obstacles = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._long_corridor_goal_w = torch.zeros(env.num_envs, 3, device=env.device)
    # 寬度取高密度上限；legacy 路徑一律用 ``[:, :dynamic_obstacles]`` 前綴，
    # 多出來的欄位保持 0/-1，不影響既有數字。
    env._long_corridor_dynamic_start = torch.zeros(
        env.num_envs, MAX_CORRIDOR_DYNAMIC, 2, device=env.device
    )
    env._long_corridor_dynamic_motion_type = torch.full(
        (env.num_envs, MAX_CORRIDOR_DYNAMIC), -1, dtype=torch.long, device=env.device
    )
    # Static-skeleton identity, per env, for the episode currently installed.
    # The sampler places static obstacles on a fixed lattice (|x| = static_x,
    # y = static_y rows) plus a small jitter, and mirrors the whole x pattern
    # with p = 0.5. The discrete identity is therefore the sign pattern of the
    # static x column, encoded as a bitmask: bit i is set when slot i sits at
    # +x. -1 means "no corridor layout installed".
    # Diagnostic only: nothing reads this to choose actions or rewards.
    env._long_corridor_static_layout_id = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    # per-env 狀態，不是「最後一批 reset」的快照。用全域欄位存 counts 的話，
    # 每次 reset 都會被不同長度的張量整個蓋掉，事後無法按 global env ID
    # 把 episode 歸因到它當時真正的密度/互動型態。
    # 累計審計：首批 8 個 env 的比例毫無統計意義（裁決要求 >= 500 次指派）。
    # 逐次 reset 累加，才看得出小批次反覆重置下的真實分佈。
    env._long_corridor_density_combo_total = torch.zeros(
        (MAX_CORRIDOR_STATIC + 1) * (MAX_CORRIDOR_DYNAMIC + 1),
        dtype=torch.long, device=env.device,
    )
    env._long_corridor_interaction_total = torch.zeros(
        3, dtype=torch.long, device=env.device
    )
    env._long_corridor_interaction_by_d = torch.zeros(
        MAX_CORRIDOR_DYNAMIC + 1, 3, dtype=torch.long, device=env.device
    )
    env._long_corridor_interaction_geometry_ok = torch.zeros(
        2, 2, dtype=torch.long, device=env.device
    )
    env._long_corridor_wander_total = torch.zeros(
        2, dtype=torch.long, device=env.device
    )
    env._long_corridor_assignment_total = 0
    # gate-aligned 分流的曝光審計：四模式各自被指派幾次，以及 count-mix 有幾次。
    env._long_corridor_gate_aligned_counts = [0] * len(_GATE_ALIGNED_MODES)
    env._long_corridor_mixed_env_total = 0
    env._long_corridor_density_counts = torch.zeros(
        env.num_envs, 2, dtype=torch.long, device=env.device
    )
    env._long_corridor_speed_density_profile = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    env._long_corridor_sampled_speed_range = torch.full(
        (env.num_envs, 2), float("nan"), dtype=torch.float32, device=env.device
    )
    env._long_corridor_interaction_type = torch.full(
        (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    env._long_corridor_interaction_pair = torch.full(
        (env.num_envs, 2), NO_PAIR, dtype=torch.long, device=env.device
    )
    # 實際 family 的累計比例，與互動比例**分開**輸出 —— 互動會強制吃掉
    # longitudinal 額度，把兩者混在一起看不出配額是否仍然平衡。
    env._long_corridor_actual_family_total = torch.zeros(
        3, dtype=torch.long, device=env.device
    )
    # 跨步複驗：安裝當下正確不代表之後仍然正確。
    env._long_corridor_pair_step_checks = torch.zeros(
        2, 2, dtype=torch.long, device=env.device
    )
    env._long_corridor_pair_step_breakdown = {}
    # 跨批次記憶：密度 carry 降低短期變異；family debt 補回配對強制吃掉的
    # longitudinal 額度（只在當批補償補不回來）。
    env._long_corridor_density_carry = {}
    env._long_corridor_speed_density_carry = {}
    env._long_corridor_speed_density_profile_total = torch.zeros(
        0,
        dtype=torch.long,
        device=env.device,
    )
    env._long_corridor_family_debt = {}
    # crossing 狀態機：固定交點 + 起始側別 + 各自是否已穿越。
    env._long_corridor_cross_point = torch.zeros(env.num_envs, 2, device=env.device)
    env._long_corridor_cross_side = torch.zeros(env.num_envs, 2, device=env.device)
    env._long_corridor_cross_done = torch.zeros(
        env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    env._long_corridor_cross_completed_total = 0
    env._long_corridor_cross_pairs_total = 0
    # 已結束並結算的 crossing 配對數。這才是完成率的正確分母 ——
    # 用「已安裝」當分母會把「尚未結束的回合」算成未完成，低估完成率。
    env._long_corridor_cross_harvested_total = 0
    env._long_corridor_motion_mode = "lateral"
    env._long_corridor_motion_weights = None
    # Cumulative motion-family audit across every successful install. The
    # one-shot injector log only covers the first batch, which cannot reveal a
    # sampling bias that only shows up over many small reset batches.
    env._long_corridor_motion_env_counts_total = torch.zeros(
        3, dtype=torch.long, device=env.device
    )
    env._long_corridor_motion_slot_counts_total = torch.zeros(
        3, dtype=torch.long, device=env.device
    )
    env._long_corridor_pure_env_count_total = 0
    # True only after _install_obstacles succeeded for the env. The activity
    # flag flips True before installation, so audits keyed on it would count
    # pre-install (non-corridor) frames.
    env._long_corridor_obstacles_ready = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    env._long_corridor_reset_count = 0
    env._long_corridor_injected_count = 0
    env._long_corridor_unsolvable_count = 0


def _clear_corridor_metadata(env, env_ids: torch.Tensor) -> None:
    """Drop per-env density/interaction attribution when an env leaves the corridor.

    留著舊值的話，之後在非走廊場景結束的 episode 會被歸因到上一次的密度與
    互動型態 —— 分項指標會把不相干的結果算進某個 (S,D) 組合。
    """
    _harvest_completed_crossings(env, env_ids)
    env._long_corridor_density_counts[env_ids] = 0
    env._long_corridor_speed_density_profile[env_ids] = -1
    env._long_corridor_sampled_speed_range[env_ids] = float("nan")
    env._long_corridor_interaction_type[env_ids] = -1
    env._long_corridor_interaction_pair[env_ids] = NO_PAIR
    env._long_corridor_cross_done[env_ids] = False


def _hide_corridor_walls(env, env_ids: torch.Tensor) -> None:
    env._long_corridor_wall_mask[env_ids] = False
    env._long_corridor_active[env_ids] = False
    env._long_corridor_pending_obstacles[env_ids] = False
    env._long_corridor_obstacles_ready[env_ids] = False
    env._long_corridor_dynamic_motion_type[env_ids] = -1
    env._long_corridor_static_layout_id[env_ids] = -1
    _clear_corridor_metadata(env, env_ids)
    origins = env.scene.env_origins[env_ids]
    for name in _ASSET_NAMES:
        pose = torch.zeros(env_ids.numel(), 7, device=env.device)
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = _HIDDEN_Z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)


def _hide_original_geometry(env, env_ids: torch.Tensor) -> None:
    """Remove random walls and obstacles from selected replay envs."""
    origins = env.scene.env_origins[env_ids]
    if hasattr(env, "_maze_wall_mask"):
        env._maze_wall_mask[env_ids] = False
    wall_slots = getattr(env, "_maze_wall_mask", torch.empty(0, 0)).shape[1]
    for slot in range(wall_slots):
        name = f"wall_internal_{slot}"
        if name not in env.scene.keys():
            continue
        pose = torch.zeros(env_ids.numel(), 7, device=env.device)
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = _HIDDEN_Z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=env_ids)

    scheduler = getattr(env.unwrapped, "_behavior_scheduler", None)
    obstacle_slots = scheduler.max_obstacles if scheduler is not None else 100
    if scheduler is not None:
        from .behavior_scheduler import BEHAVIOR_INACTIVE

        scheduler.behavior_type[env_ids] = BEHAVIOR_INACTIVE
        scheduler.positions[env_ids] = 0.0
        scheduler.velocities[env_ids] = 0.0
        scheduler.phase_timer[env_ids] = 0
        scheduler.patrol_pause_remaining[env_ids] = 0

    for slot in range(obstacle_slots):
        name = f"obstacle_{slot}"
        if name not in env.scene.keys():
            continue
        obstacle = env.scene[name]
        pose = obstacle.data.default_root_state[env_ids, :7].clone()
        pose[:, :2] = origins[:, :2]
        pose[:, 2] = _HIDDEN_Z
        obstacle.write_root_pose_to_sim(pose, env_ids=env_ids)
        obstacle.write_root_velocity_to_sim(
            torch.zeros(env_ids.numel(), 6, device=env.device), env_ids=env_ids
        )


#: 高密度混合場的未啟用 slot 全留在原點 (0,0)，直接送去檢查會被誤判成
#: 一堆互相穿透的障礙。兩種檢查要用**不同**的替身，因為它們問的事不同：
#:
#: * 重疊檢查問「有沒有兩個實體長在一起」-> 把未啟用的挪到走廊外最乾淨。
#: * 可解性檢查問「還在牆內嗎、中央路線通不通」-> 挪到走廊外會被判出界，
#:   所以改停在貼牆的界內停車位（不擋中央路線）。
_INACTIVE_FAR_SENTINEL = 1.0e4
#: 哨兵之間的間距，遠大於 2 * obstacle_radius = 0.7 m。
_INACTIVE_SENTINEL_STRIDE = 10.0


def _sample_mixed_density_layout(
    count: int,
    spec: LongCorridorSpec,
    device,
    count_mix,
    max_tries: int = 20,
    density_carry: dict | None = None,
    family_debt: dict | None = None,
    global_env_ids: torch.Tensor | None = None,
    interaction_override: str | None = None,
    counts_override: torch.Tensor | None = None,
    family_weights: tuple[float, float, float] | None = None,
):
    """Sample a per-env variable-density corridor layout.

    Returns ``(counts, families, static, dynamic, waypoints, targets,
    static_mask, dynamic_mask)``.

    數量、family 配額與 active mask 由 ``corridor_density`` 決定（policy），
    位置與路徑由幾何層產生。未啟用的 slot 一律留在原點並標成 INACTIVE，
    但**檢查時**先挪到哨兵位置，避免假重疊。
    """
    if counts_override is None:
        counts = sample_density_counts_systematic(
            count, mix=count_mix, device=device, carry=density_carry
        )
    else:
        counts = counts_override.to(device=device, dtype=torch.long)
        if counts.shape != (count, 2):
            raise ValueError(
                f"counts_override must have shape {(count, 2)}, got "
                f"{tuple(counts.shape)}"
            )
        active_masks_from_counts(counts)  # fail closed on capacity/count errors
    dynamic_counts = counts[:, 1]
    # 順序不可調換：互動型態是契約，family 是為了實現它而指派的。
    # 反過來（先 family 再問做得出什麼互動）只能默默降級，湊不出 60/20/20。
    override_id = _normalize_interaction_override(interaction_override)
    if override_id is None:
        interaction_types = sample_interaction_types(
            dynamic_counts=dynamic_counts, device=device
        )
    else:
        if override_id != INTERACTION_INDEPENDENT and bool((dynamic_counts < 2).any()):
            raise ValueError(
                f"{interaction_override} interaction requires at least 2 dynamic "
                "obstacles in every sampled environment"
            )
        interaction_types = torch.full_like(dynamic_counts, override_id)
    # 配對 slot 的 family 一開始就正確（crossing=lateral+longitudinal、
    # side_by_side=longitudinal+longitudinal），random_2d 永不參與配對 ——
    # wander 會重抽獨立 heading，會把擺好的互動幾何整個蓋掉。
    families, pairs = assign_families_and_pairs(
        dynamic_counts, interaction_types, device=device,
        family_debt=family_debt,
        family_weights=family_weights,
        # global env ID 必須傳下去 —— 沒有它，dump 出來的 batch-local 索引
        # 無法對回是哪些環境，重現時只能猜。
        global_env_ids=global_env_ids,
    )
    problems = validate_pair_families(interaction_types, pairs, families)
    if problems:
        raise RuntimeError(
            "corridor interaction pairing violated its family contract: "
            + "; ".join(problems[:5])
        )
    static_mask, dynamic_mask = active_masks_from_counts(counts)

    lateral_lanes = tuple(float(y) for y in spec.dynamic_y)
    longitudinal_y_limit = min(
        3.4, 0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance
    )

    clean = None
    speeds = None
    for attempt in range(max_tries + 1):
        static, _, _ = sample_obstacle_layout(
            count, spec, device,
            max_static=MIXED_MAX_STATIC, max_dynamic=MIXED_MAX_DYNAMIC,
            permute_slots=True,
        )
        dynamic = dynamic_layout_for_families(
            families,
            lateral_lanes=lateral_lanes,
            center_strip_x=CENTER_STRIP_X,
            lateral_x_limit=spec.dynamic_x_limit,
            device=device,
        )
        waypoints, targets = dynamic_waypoints_for_families(
            families, dynamic,
            lateral_x_limit=spec.dynamic_x_limit,
            longitudinal_y_limit=longitudinal_y_limit,
            center_strip_x=CENTER_STRIP_X,
            device=device,
        )
        bounds_probe = _park_inactive(
            static, dynamic, waypoints, static_mask, dynamic_mask, spec
        )
        solvable = layout_is_constructively_solvable(*bounds_probe, spec)
        far_static, far_dynamic, _ = _banish_inactive(
            static, dynamic, waypoints, static_mask, dynamic_mask
        )
        ds, dd = corridor_penetration_masks(
            far_dynamic, far_static, spec.obstacle_radius
        )
        clean = solvable & ~ds.any(dim=1) & ~dd.any(dim=1)
        if bool(clean.all()):
            return (
                counts, families, static, dynamic, waypoints, targets,
                static_mask, dynamic_mask, interaction_types, pairs,
            )
        if attempt == max_tries:
            break
    raise RuntimeError(
        "mixed-density corridor resampling failed to clear overlaps within "
        f"{max_tries} tries for {int((~clean).sum().item())} envs; "
        "refusing to install an overlapped scene"
    )


def _banish_inactive(static, dynamic, waypoints, static_mask, dynamic_mask):
    """Move masked-off slots far outside the corridor (overlap probe).

    每個 slot 的哨兵座標必須**互不相同**。全部塞同一點的話，未啟用的 slot 之間
    距離為 0，重疊檢查會把空 slot 判成互相穿透 —— 密度越低反而越不合格。
    靜態與動態也各自佔一個哨兵區，避免跨類假重疊。
    """
    far = _INACTIVE_FAR_SENTINEL
    stride = _INACTIVE_SENTINEL_STRIDE

    def banish(tensor, mask, sign):
        slots = tensor.shape[1]
        offsets = torch.arange(slots, device=tensor.device, dtype=tensor.dtype)
        sentinel = torch.empty_like(tensor)
        sentinel[..., 0] = sign * far
        sentinel[..., 1] = far + offsets.reshape(
            (1, slots) + (1,) * (tensor.ndim - 3)
        ) * stride
        expand = mask.reshape(mask.shape + (1,) * (tensor.ndim - mask.ndim))
        return torch.where(expand, tensor, sentinel)

    return (
        banish(static, static_mask, 1.0),
        banish(dynamic, dynamic_mask, -1.0),
        banish(waypoints, dynamic_mask, -1.0),
    )


def _park_inactive(static, dynamic, waypoints, static_mask, dynamic_mask, spec):
    """Park masked-off slots against the wall, in bounds (solvability probe).

    停車位貼在走廊**末端**的左右兩角，靜態與動態各佔一角：

    * 仍在牆內 -> 過得了邊界檢查。
    * ``|x|`` 遠離中央線 -> 不會假性擋住建構路線。
    * 離最近的真實靜態（y 最外一列 3.0 m）有 1.3 m -> 不會被巡邏線段的
      「不得掠過靜態」檢查誤判。停在 ``y=0`` 只有 0.67 m，剛好低於
      2 * 0.35 m 的門檻，會讓**每一個**有空 slot 的場景都判成不可解。
    * 靜態與動態分佔左右 -> 兩邊的空 slot 之間也不會互相誤判。
    """
    park_x = spec.inner_half_width - spec.obstacle_radius - spec.wall_clearance
    park_y = 0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance - 0.15
    static_park = torch.tensor([park_x, park_y], device=static.device)
    dynamic_park = torch.tensor([-park_x, park_y], device=static.device)
    return (
        torch.where(static_mask[..., None], static, static_park),
        torch.where(dynamic_mask[..., None], dynamic, dynamic_park),
        torch.where(dynamic_mask[..., None, None], waypoints, dynamic_park),
    )


def _install_obstacles(
    env,
    selected: torch.Tensor,
    spec: LongCorridorSpec,
    speed_min: float,
    speed_max: float,
    static_obstacles: int,
    dynamic_obstacles: int,
    dynamic_motion_mode: str,
    dynamic_motion_weights: tuple[float, float, float] | None = None,
    count_mix=None,
    speed_density_mix=None,
    interaction_override: str | None = None,
) -> bool:
    scheduler = getattr(env.unwrapped, "_behavior_scheduler", None)
    if scheduler is None:
        env._long_corridor_pending_obstacles[selected] = True
        return False
    if count_mix is not None or speed_density_mix is not None:
        return _install_mixed_density_obstacles(
            env, selected, spec, speed_min, speed_max, count_mix, scheduler,
            speed_density_mix=speed_density_mix,
            dynamic_motion_weights=dynamic_motion_weights,
            interaction_override=interaction_override,
        )
    validate_obstacle_counts(static_obstacles, dynamic_obstacles)
    if scheduler.max_obstacles < 4 + dynamic_obstacles:
        raise RuntimeError(
            "long corridor requires enough BehaviorScheduler obstacle slots"
        )

    from .behavior_scheduler import (
        BEHAVIOR_INACTIVE,
        BEHAVIOR_PATROL,
        BEHAVIOR_RANDOM_WALK,
        BEHAVIOR_STATIC,
    )

    count = selected.numel()
    # Spawn-conflict-free sampling: two obstacles spawning inside each other
    # merge into one, so overlapped draws are rejected and redrawn with their
    # motion families held fixed; the sampler raises after max_tries instead
    # of installing an overlapped scene (postcondition).
    static, dynamic, waypoints, motion_types, target_indices = (
        sample_conflict_free_layout(
            count,
            spec,
            env.device,
            dynamic_motion_mode,
            dynamic_motion_weights,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
        )
    )

    scheduler.behavior_type[selected] = BEHAVIOR_INACTIVE
    scheduler.positions[selected] = 0.0
    scheduler.velocities[selected] = 0.0
    scheduler.phase_timer[selected] = 0
    scheduler.patrol_pause_remaining[selected] = 0
    env._long_corridor_dynamic_start[selected] = 0.0
    env._long_corridor_dynamic_motion_type[selected] = -1

    if static_obstacles > 0:
        static_slots = slice(0, static_obstacles)
        scheduler.behavior_type[selected, static_slots] = BEHAVIOR_STATIC
        scheduler.positions[selected, static_slots] = static[
            :, :static_obstacles
        ]
        env._long_corridor_static_layout_id[selected] = static_layout_id(
            static[:, :static_obstacles]
        )
    else:
        env._long_corridor_static_layout_id[selected] = 0

    if dynamic_obstacles > 0:
        dynamic_slots = slice(4, 4 + dynamic_obstacles)
        active_dynamic = dynamic[:, :dynamic_obstacles]
        active_waypoints = waypoints[:, :dynamic_obstacles]
        active_motion_types = motion_types[:, :dynamic_obstacles]
        active_target_indices = target_indices[:, :dynamic_obstacles]
        # Cumulative audit; counted here so a pending (deferred) install is not
        # double counted when it later succeeds.
        env._long_corridor_motion_env_counts_total += torch.bincount(
            active_motion_types[:, 0], minlength=3
        )
        env._long_corridor_motion_slot_counts_total += torch.bincount(
            active_motion_types.flatten(), minlength=3
        )
        env._long_corridor_pure_env_count_total += int(
            (active_motion_types == active_motion_types[:, :1])
            .all(dim=1)
            .sum()
            .item()
        )
        speeds = torch.empty(
            count, dynamic_obstacles, device=env.device
        ).uniform_(float(speed_min), float(speed_max))

        scheduler.positions[selected, dynamic_slots] = active_dynamic
        env._long_corridor_dynamic_start[selected, :dynamic_obstacles] = (
            active_dynamic
        )
        env._long_corridor_dynamic_motion_type[
            selected, :dynamic_obstacles
        ] = active_motion_types

        # All motion families use bounded two-point patrols. Random-2D varies
        # both endpoints per episode without allowing unbounded wall tunneling.
        scheduler.behavior_type[selected, dynamic_slots] = BEHAVIOR_PATROL
        scheduler.patrol_num_waypoints[selected, dynamic_slots] = 2
        scheduler.patrol_speed[selected, dynamic_slots] = speeds
        scheduler.patrol_pause_remaining[selected, dynamic_slots] = 0
        scheduler.patrol_wp_index[
            selected, dynamic_slots
        ] = active_target_indices
        scheduler.patrol_waypoints[
            selected, dynamic_slots, :2
        ] = active_waypoints
        scheduler.patrol_waypoints[selected, dynamic_slots, 2:] = 0.0
        target = torch.gather(
            active_waypoints,
            2,
            active_target_indices[..., None, None].expand(
                -1, -1, 1, 2
            ),
        ).squeeze(2)
        direction = target - active_dynamic
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        ).clamp_min(1e-6)
        scheduler.velocities[selected, dynamic_slots] = (
            direction * speeds[..., None]
        )

        # Optional: give the random_2d family a genuine bounded random walk
        # instead of a two-point patrol. The patrol reverses on a fixed ~1.5 m
        # leg roughly every 3.3 s, which sits inside the 1.5 s future-occupancy
        # horizon and makes a large share of predictions structurally wrong.
        # A wander has no fixed turnaround point. Default is "patrol", so every
        # historical gate is untouched.
        if getattr(env, "_long_corridor_random_2d_kinematics", "patrol") == "wander":
            wander = active_motion_types == MOTION_RANDOM_2D
            if bool(wander.any()):
                _install_corridor_wander(
                    scheduler, selected, dynamic_slots, wander, spec, speeds
                )
    env._long_corridor_pending_obstacles[selected] = False
    env._long_corridor_obstacles_ready[selected] = True
    scheduler._write_positions_to_sim(env)
    return True


def _install_mixed_density_obstacles(
    env,
    selected: torch.Tensor,
    spec: LongCorridorSpec,
    speed_min: float,
    speed_max: float,
    count_mix,
    scheduler,
    speed_density_mix=None,
    dynamic_motion_weights: tuple[float, float, float] | None = None,
    interaction_override: str | None = None,
) -> bool:
    """Install a per-env variable-density corridor scene (2026-07-27 混合場).

    與 legacy 路徑的關鍵差別：每個 env 的 (S, D) **不同**，所以 slot 佈局固定
    在 ``[0, 5)`` 靜態 + ``[5, 10)`` 動態，靠 active mask 逐 env 關掉多餘的，
    而不是靠 ``slice(0, S)`` / ``slice(4, 4 + D)`` 這種全批共用的前綴切片。
    """
    from .behavior_scheduler import (
        BEHAVIOR_INACTIVE,
        BEHAVIOR_PATROL,
        BEHAVIOR_RANDOM_WALK,
        BEHAVIOR_STATIC,
    )

    required = MAX_CORRIDOR_STATIC + MAX_CORRIDOR_DYNAMIC
    if scheduler.max_obstacles < required:
        raise RuntimeError(
            f"mixed-density corridor needs {required} BehaviorScheduler obstacle "
            f"slots, scheduler has {scheduler.max_obstacles}"
        )

    count = selected.numel()
    sampled_counts = None
    sampled_speed_ranges = None
    sampled_profile_ids = None
    if speed_density_mix is not None:
        (
            sampled_counts,
            sampled_speed_ranges,
            sampled_profile_ids,
        ) = sample_speed_density_profiles_systematic(
            count,
            mix=speed_density_mix,
            device=env.device,
            carry=getattr(env, "_long_corridor_speed_density_carry", None),
        )

    (
        counts, families, static, dynamic, waypoints, targets,
        static_mask, dynamic_mask, interaction_types, pairs,
    ) = _sample_mixed_density_layout(
        count, spec, env.device, count_mix,
        density_carry=getattr(env, "_long_corridor_density_carry", None),
        family_debt=getattr(env, "_long_corridor_family_debt", None),
        global_env_ids=selected,
        interaction_override=interaction_override,
        counts_override=sampled_counts,
        family_weights=dynamic_motion_weights,
    )

    if sampled_speed_ranges is None:
        speeds = torch.empty(
            count, MAX_CORRIDOR_DYNAMIC, device=env.device
        ).uniform_(float(speed_min), float(speed_max))
        env._long_corridor_speed_density_profile[selected] = -1
        env._long_corridor_sampled_speed_range[selected, 0] = float(speed_min)
        env._long_corridor_sampled_speed_range[selected, 1] = float(speed_max)
    else:
        unit = torch.rand(
            count, MAX_CORRIDOR_DYNAMIC, device=env.device
        )
        speed_lo = sampled_speed_ranges[:, 0:1]
        speed_hi = sampled_speed_ranges[:, 1:2]
        speeds = speed_lo + unit * (speed_hi - speed_lo)
        env._long_corridor_speed_density_profile[selected] = sampled_profile_ids
        env._long_corridor_sampled_speed_range[selected] = sampled_speed_ranges
        env._long_corridor_speed_density_profile_total += torch.bincount(
            sampled_profile_ids,
            minlength=len(speed_density_mix),
        )
    speeds = speeds * dynamic_mask

    # 互動幾何必須在寫進 scheduler **之前**套用，而且要 fail-fast：
    # 抽到 crossing 卻蓋不出 crossing 的話，指標會報 20% 交叉、場景裡卻是
    # 各走各的。降級比缺功能更糟 —— 它讓錯誤的數字看起來是對的。
    longitudinal_y_limit = min(
        3.4, 0.5 * spec.length - spec.obstacle_radius - spec.wall_clearance
    )
    dynamic, waypoints, targets, speeds, realized = apply_interaction_geometry(
        interaction_types, pairs, families,
        dynamic, waypoints, targets, speeds,
        longitudinal_y_limit=longitudinal_y_limit,
        center_strip_x=CENTER_STRIP_X, device=env.device,
    )
    if not bool(realized.all()):
        missing = (~realized).nonzero(as_tuple=False).flatten()
        kinds = sorted(
            {INTERACTION_NAMES[int(interaction_types[i])] for i in missing.tolist()}
        )
        raise RuntimeError(
            f"corridor interaction sampling produced {missing.numel()} env(s) whose "
            f"interaction could not be built ({kinds}); refusing to install a scene "
            "whose reported interaction mix does not match its geometry"
        )

    scheduler.behavior_type[selected] = BEHAVIOR_INACTIVE
    scheduler.positions[selected] = 0.0
    scheduler.velocities[selected] = 0.0
    scheduler.phase_timer[selected] = 0
    scheduler.patrol_pause_remaining[selected] = 0
    env._long_corridor_dynamic_start[selected] = 0.0
    env._long_corridor_dynamic_motion_type[selected] = -1

    static_slots = slice(0, MAX_CORRIDOR_STATIC)
    dynamic_slots = slice(
        MAX_CORRIDOR_STATIC, MAX_CORRIDOR_STATIC + MAX_CORRIDOR_DYNAMIC
    )

    scheduler.behavior_type[selected, static_slots] = torch.where(
        static_mask,
        torch.full_like(static_mask, BEHAVIOR_STATIC, dtype=torch.long),
        torch.full_like(static_mask, BEHAVIOR_INACTIVE, dtype=torch.long),
    ).to(scheduler.behavior_type.dtype)
    scheduler.positions[selected, static_slots] = static * static_mask[..., None]
    # Mixed density installs every slot, so the active count must be passed in
    # explicitly; without it two densities would share one layout id.
    env._long_corridor_static_layout_id[selected] = static_layout_id(
        static * static_mask[..., None], static_mask.sum(dim=-1)
    )

    scheduler.behavior_type[selected, dynamic_slots] = torch.where(
        dynamic_mask,
        torch.full_like(dynamic_mask, BEHAVIOR_PATROL, dtype=torch.long),
        torch.full_like(dynamic_mask, BEHAVIOR_INACTIVE, dtype=torch.long),
    ).to(scheduler.behavior_type.dtype)
    scheduler.positions[selected, dynamic_slots] = dynamic * dynamic_mask[..., None]
    scheduler.patrol_num_waypoints[selected, dynamic_slots] = 2
    scheduler.patrol_speed[selected, dynamic_slots] = speeds
    scheduler.patrol_pause_remaining[selected, dynamic_slots] = 0
    scheduler.patrol_wp_index[selected, dynamic_slots] = targets
    scheduler.patrol_waypoints[selected, dynamic_slots, :2] = waypoints
    scheduler.patrol_waypoints[selected, dynamic_slots, 2:] = 0.0

    target_xy = torch.gather(
        waypoints, 2, targets[..., None, None].expand(-1, -1, 1, 2)
    ).squeeze(2)
    direction = target_xy - dynamic
    direction = direction / torch.linalg.vector_norm(
        direction, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    scheduler.velocities[selected, dynamic_slots] = (
        direction * speeds[..., None] * dynamic_mask[..., None]
    )

    env._long_corridor_dynamic_start[selected] = dynamic * dynamic_mask[..., None]
    env._long_corridor_dynamic_motion_type[selected] = torch.where(
        dynamic_mask, families, torch.full_like(families, -1)
    )
    env._long_corridor_density_counts[selected] = counts
    env._long_corridor_interaction_type[selected] = interaction_types
    env._long_corridor_interaction_pair[selected] = pairs

    # 並排 pair 不得隨機暫停 —— 兩人各自暫停幾步就散開了。
    scheduler.patrol_no_pause[selected, dynamic_slots] = False
    side_rows = (
        interaction_types == INTERACTION_SIDE_BY_SIDE
    ).nonzero(as_tuple=False).flatten()
    if side_rows.numel() > 0:
        base = MAX_CORRIDOR_STATIC
        for row in side_rows.tolist():
            env_id = int(selected[row])
            for k in range(2):
                scheduler.patrol_no_pause[env_id, base + int(pairs[row, k])] = True

    point, side = crossing_axis_and_side(interaction_types, pairs, families, dynamic)
    env._long_corridor_cross_point[selected] = point
    env._long_corridor_cross_side[selected] = side
    env._long_corridor_cross_done[selected] = False
    env._long_corridor_cross_pairs_total += int(
        (interaction_types == INTERACTION_CROSSING).sum()
    )
    env._long_corridor_actual_family_total += torch.bincount(
        families[dynamic_mask], minlength=3
    )

    # legacy 路徑會把 random_2d 換成 wander，mixed 路徑漏掉就等於 banner 寫
    # wander、實際卻全是 patrol —— 兩點巡邏每 ~3.3 s 在固定點折返，落在
    # future-occupancy 的 1.5 s 視界內，正是 W1 要擺脫的東西。
    if getattr(env, "_long_corridor_random_2d_kinematics", "patrol") == "wander":
        wander = (families == MOTION_RANDOM_2D) & dynamic_mask
        # 硬閘：被配對的 slot 絕不能變成 RANDOM_WALK —— wander 會重抽獨立
        # heading，把互動幾何蓋掉，而 audit 若量安裝前的張量就會報假陽性。
        paired_slots = torch.zeros_like(wander)
        valid = pairs >= 0
        if bool(valid.any()):
            rows = valid.nonzero(as_tuple=False)
            paired_slots[rows[:, 0], pairs[rows[:, 0], rows[:, 1]]] = True
        if bool((wander & paired_slots).any()):
            raise RuntimeError(
                "corridor wander would overwrite an interaction pair slot; "
                "paired slots must never be random_2d"
            )
        if bool(wander.any()):
            _install_corridor_wander(
                scheduler, selected, dynamic_slots, wander, spec, speeds
            )

    # ---- 累計審計 ----
    env._long_corridor_assignment_total += int(count)
    env._long_corridor_density_combo_total += torch.bincount(
        counts[:, 0] * (MAX_CORRIDOR_DYNAMIC + 1) + counts[:, 1],
        minlength=(MAX_CORRIDOR_STATIC + 1) * (MAX_CORRIDOR_DYNAMIC + 1),
    )
    env._long_corridor_interaction_total += torch.bincount(
        interaction_types, minlength=3
    )
    for d in range(MAX_CORRIDOR_DYNAMIC + 1):
        rows = counts[:, 1] == d
        if bool(rows.any()):
            env._long_corridor_interaction_by_d[d] += torch.bincount(
                interaction_types[rows], minlength=3
            )

    # 複驗必須讀**安裝後**的 scheduler 狀態：wander 在幾何之後才覆寫 heading，
    # 量安裝前的暫存張量會讓已被蓋掉的互動看起來完好（假陽性）。
    env._long_corridor_interaction_audit = audit_installed_pairs(
        behavior_type=scheduler.behavior_type[selected, dynamic_slots],
        positions=scheduler.positions[selected, dynamic_slots],
        velocities=scheduler.velocities[selected, dynamic_slots],
        interaction_types=interaction_types,
        pairs=pairs,
        families=families,
        random_walk_id=BEHAVIOR_RANDOM_WALK,
    )
    if env._long_corridor_interaction_audit["paired_is_random_walk"]:
        raise RuntimeError(
            "interaction pair slot was installed as RANDOM_WALK; the wander "
            "override must never touch a paired slot"
        )
    # runtime 證明：random_2d 的 slot 在 wander 模式下必須真的是 RANDOM_WALK。
    # 只看 min_speed > 0.1 證明不了 —— patrol 的速度一樣 > 0.1。
    want_wander = (
        getattr(env, "_long_corridor_random_2d_kinematics", "patrol") == "wander"
    )
    rnd = (families == MOTION_RANDOM_2D) & dynamic_mask
    installed = scheduler.behavior_type[selected, dynamic_slots]
    env._long_corridor_wander_audit = (
        int(rnd.sum()),
        int((installed[rnd] == BEHAVIOR_RANDOM_WALK).sum()) if bool(rnd.any()) else 0,
        bool(want_wander),
    )
    env._long_corridor_wander_total[0] += int(rnd.sum())
    env._long_corridor_wander_total[1] += (
        int((installed[rnd] == BEHAVIOR_RANDOM_WALK).sum()) if bool(rnd.any()) else 0
    )
    audit = env._long_corridor_interaction_audit
    env._long_corridor_interaction_geometry_ok += torch.tensor(
        [
            [audit["crossing_ok"], audit["crossing_n"]],
            [audit["side_ok"], audit["side_n"]],
        ],
        dtype=torch.long, device=env.device,
    )
    # future-occupancy 只認 speed > 0.10 m/s 的障礙。高密度場多出來的
    # slot 若速度為 0，會變成 reward 看不見的隱形障礙 —— 必須量，不能推論。
    active_speed = speeds[dynamic_mask]
    env._long_corridor_density_speed_audit = (
        float(active_speed.min()) if active_speed.numel() else float("nan"),
        int((active_speed <= 0.10).sum()),
        int(active_speed.numel()),
    )

    _maybe_log_cumulative_density_audit(env)

    env._long_corridor_pending_obstacles[selected] = False
    env._long_corridor_obstacles_ready[selected] = True
    scheduler._write_positions_to_sim(env)
    return True


#: 累計審計的回報門檻。裁決要求「累積至少 500 個 corridor assignments
#: 後再驗比例，不只看首批 8 env」。
_DENSITY_AUDIT_MIN_ASSIGNMENTS = 500


def _verify_pairs_this_step(env, selected: torch.Tensor) -> None:
    """Re-check interaction pair invariants against the **live** scheduler state.

    安裝當下正確不代表之後仍然正確 —— 例如 wander 的 heading 重抽、或任何
    下游改寫速度的邏輯，都只會在 episode 進行中才顯現。因此每一步重量一次，
    累積通過率，而不是只在注入時量一次就宣告成功。
    """
    scheduler = getattr(env.unwrapped, "_behavior_scheduler", None)
    if scheduler is None or selected.numel() == 0:
        return
    if (
        getattr(env, "_long_corridor_obstacle_count_mix", None) is None
        and getattr(env, "_long_corridor_speed_density_mix", None) is None
    ):
        return
    from .behavior_scheduler import BEHAVIOR_RANDOM_WALK

    ready = selected[env._long_corridor_obstacles_ready[selected]]
    if ready.numel() == 0:
        return
    pairs = env._long_corridor_interaction_pair[ready]
    has_pair = (pairs[:, 0] >= 0).nonzero(as_tuple=False).flatten()
    if has_pair.numel() == 0:
        return
    rows = ready[has_pair]
    dynamic_slots = slice(
        MAX_CORRIDOR_STATIC, MAX_CORRIDOR_STATIC + MAX_CORRIDOR_DYNAMIC
    )
    audit = audit_installed_pairs(
        behavior_type=scheduler.behavior_type[rows, dynamic_slots],
        positions=scheduler.positions[rows, dynamic_slots],
        velocities=scheduler.velocities[rows, dynamic_slots],
        interaction_types=env._long_corridor_interaction_type[rows],
        pairs=pairs[has_pair],
        families=env._long_corridor_dynamic_motion_type[rows],
        random_walk_id=BEHAVIOR_RANDOM_WALK,
    )
    # crossing 改記**事件**：兩人各自穿過固定交點一次就算完成。
    # 逐幀要求朝交點走是錯的 —— 穿過去或在 waypoint 暫停都會自然不符合
    # （實測 past_crossing 4665、zero_speed 1179 佔了失敗的絕大多數）。
    env._long_corridor_cross_done[rows] = crossing_progress(
        env._long_corridor_interaction_type[rows],
        pairs[has_pair],
        env._long_corridor_dynamic_motion_type[rows],
        scheduler.positions[rows, dynamic_slots],
        env._long_corridor_cross_point[rows],
        env._long_corridor_cross_side[rows],
        env._long_corridor_cross_done[rows],
    )

    env._long_corridor_pair_step_checks += torch.tensor(
        [
            [audit["crossing_ok"], audit["crossing_n"]],
            [audit["side_ok"], audit["side_n"]],
        ],
        dtype=torch.long, device=env.device,
    )
    for key in (
        "x_wrong_family", "x_zero_speed", "x_past_crossing",
        "s_wrong_family", "s_spacing", "s_speed_delta", "s_opposite_phase",
    ):
        env._long_corridor_pair_step_breakdown[key] = (
            env._long_corridor_pair_step_breakdown.get(key, 0) + audit[key]
        )
    if audit["paired_is_random_walk"]:
        raise RuntimeError(
            "an interaction pair slot became RANDOM_WALK mid-episode; the "
            "wander override must never reach a paired slot"
        )


def _harvest_completed_crossings(env, env_ids: torch.Tensor) -> None:
    """Count crossings that completed before the env leaves the corridor."""
    done = env._long_corridor_cross_done[env_ids]
    was_crossing = (
        env._long_corridor_interaction_type[env_ids] == INTERACTION_CROSSING
    )
    env._long_corridor_cross_completed_total += int(
        (done.all(dim=1) & was_crossing).sum()
    )
    env._long_corridor_cross_harvested_total += int(was_crossing.sum())


def _maybe_log_cumulative_density_audit(env) -> None:
    """Emit the cumulative density/interaction audit once it is statistically real."""
    total = env._long_corridor_assignment_total
    if total < _DENSITY_AUDIT_MIN_ASSIGNMENTS:
        return
    if getattr(env, "_long_corridor_density_audit_logged", False):
        return
    env._long_corridor_density_audit_logged = True

    combo = env._long_corridor_density_combo_total
    stride = MAX_CORRIDOR_DYNAMIC + 1
    realized = {
        f"{s}S{d}D": round(float(combo[s * stride + d]) / total, 4)
        for s in range(MAX_CORRIDOR_STATIC + 1)
        for d in range(stride)
        if int(combo[s * stride + d]) > 0
    }
    profile_realized = None
    profile_total = getattr(env, "_long_corridor_speed_density_profile_total", None)
    speed_density_mix = getattr(env, "_long_corridor_speed_density_mix", None)
    if (
        speed_density_mix is not None
        and profile_total is not None
        and int(profile_total.sum()) > 0
    ):
        profile_denominator = int(profile_total.sum())
        profile_realized = {
            (
                f"{counts[0]}S{counts[1]}D@"
                f"{speed_range[0]:.2f}-{speed_range[1]:.2f}"
            ): round(float(profile_total[index]) / profile_denominator, 4)
            for index, (counts, speed_range, _) in enumerate(speed_density_mix)
        }
    names = [INTERACTION_NAMES[k] for k in range(3)]
    overall = env._long_corridor_interaction_total
    by_d = {
        d: [round(float(v) / max(int(env._long_corridor_interaction_by_d[d].sum()), 1), 3)
            for v in env._long_corridor_interaction_by_d[d]]
        for d in range(stride)
        if int(env._long_corridor_interaction_by_d[d].sum()) > 0
    }
    geom_ok = env._long_corridor_interaction_geometry_ok
    wander = env._long_corridor_wander_total
    fam_total = env._long_corridor_actual_family_total
    step_checks = env._long_corridor_pair_step_checks
    print(
        "[LONG-CORRIDOR-AUDIT] "
        f"assignments={total} density={realized} "
        f"speed_density_profiles={profile_realized} "
        f"interaction_order={names} "
        f"interaction_overall={[int(v) for v in overall]} "
        f"interaction_by_dynamic_count={by_d} "
        f"geometry_ok=crossing {int(geom_ok[0][0])}/{int(geom_ok[0][1])},"
        f"side_by_side {int(geom_ok[1][0])}/{int(geom_ok[1][1])} "
        f"random_2d_slots={int(wander[0])} "
        f"installed_as_random_walk={int(wander[1])} "
        f"actual_family_fractions={[round(float(v) / max(int(fam_total.sum()), 1), 4) for v in fam_total]} "
        f"crossing_completed={int(env._long_corridor_cross_completed_total)}/"
        f"{int(env._long_corridor_cross_harvested_total)}"
        f"(installed={int(env._long_corridor_cross_pairs_total)}) "
        f"side_formation_steps={int(step_checks[1][0])}/{int(step_checks[1][1])} "
        f"pair_step_breakdown={getattr(env, '_long_corridor_pair_step_breakdown', {})}",
        flush=True,
    )


def _write_goal(
    env, selected: torch.Tensor, *, update_markers: bool = False
) -> None:
    goal = env._long_corridor_goal_w[selected]
    goal_term = env.command_manager.get_term("goal_command")
    goal_term.goal_pos_w[selected] = goal
    if hasattr(goal_term, "all_goals_pos_w"):
        goal_term.all_goals_pos_w[selected] = goal[:, None, :]
    if hasattr(env, "_local_goal_world") and env._local_goal_world is not None:
        env._local_goal_world[selected, :2] = goal[:, :2]
    if update_markers and hasattr(goal_term, "_update_goal_markers"):
        goal_term._update_goal_markers()


def setup_long_corridor_replay(
    env,
    env_ids,
    *,
    fraction: float = 0.0,
    free_width: float = 4.0,
    length: float = 10.0,
    wall_span_length: float | None = None,
    static_obstacles: int = 4,
    dynamic_obstacles: int = 2,
    dynamic_speed_min: float = 0.30,
    dynamic_speed_max: float = 0.60,
    dynamic_motion_mode: str = "lateral",
    dynamic_motion_weights: tuple[float, float, float] | None = None,
    dynamic_pause_steps_range: tuple[int, int] | None = None,
    obstacle_count_mix=None,
    speed_density_mix=None,
    gate_aligned_share: float = 0.0,
    random_2d_kinematics: str = "patrol",
    interaction_override: str | None = None,
    wall_z: float = 1.5,
) -> None:
    """Replace a fraction of resets with the frozen deployment corridor.

    ``dynamic_pause_steps_range`` is an **eval-only** override for the patrol
    waypoint pause. ``None`` (the default) leaves the scheduler configuration
    untouched, so training and every historical gate keep the stock 0-5 step
    behaviour. It exists so a pause-vs-no-pause A/B can isolate whether the
    future-occupancy reward's blindness to stationary obstacles (velocity is
    hard-zeroed on waypoint arrival, below the 0.1 m/s validity threshold)
    drives the random_2d failure.
    """
    if fraction <= 0.0:
        return
    if obstacle_count_mix is not None and speed_density_mix is not None:
        raise ValueError(
            "obstacle_count_mix and speed_density_mix are mutually exclusive"
        )
    if obstacle_count_mix is None and speed_density_mix is None:
        validate_obstacle_counts(static_obstacles, dynamic_obstacles)
    elif speed_density_mix is not None:
        validate_speed_density_mix(speed_density_mix)
    else:
        validate_density_mix(obstacle_count_mix)

    ids = _as_env_ids(env, env_ids)
    if ids.numel() == 0:
        return
    spec = LongCorridorSpec(
        free_width=float(free_width),
        length=float(length),
        wall_span_length=(
            float(wall_span_length)
            if wall_span_length is not None
            else None
        ),
    )
    validate_spec(spec)
    motion_mode = validate_dynamic_motion_mode(dynamic_motion_mode)
    motion_weights = normalize_dynamic_motion_weights(
        dynamic_motion_weights, motion_mode
    )
    _ensure_state(env)
    env._long_corridor_fraction = float(fraction)
    env._long_corridor_spec = spec
    env._long_corridor_speed_range = (
        float(dynamic_speed_min),
        float(dynamic_speed_max),
    )
    env._long_corridor_motion_mode = motion_mode
    env._long_corridor_motion_weights = motion_weights
    env._long_corridor_obstacle_count_mix = obstacle_count_mix
    env._long_corridor_speed_density_mix = speed_density_mix
    expected_profiles = len(speed_density_mix) if speed_density_mix is not None else 0
    profile_total = env._long_corridor_speed_density_profile_total
    if profile_total.numel() == 0 and expected_profiles > 0:
        env._long_corridor_speed_density_profile_total = torch.zeros(
            expected_profiles, dtype=torch.long, device=env.device
        )
    elif profile_total.numel() != expected_profiles:
        raise RuntimeError(
            "speed-density profile count changed after environment setup: "
            f"{profile_total.numel()} -> {expected_profiles}"
        )
    _normalize_interaction_override(interaction_override)
    if (
        interaction_override not in (None, "", "sample")
        and obstacle_count_mix is None
        and speed_density_mix is None
    ):
        raise ValueError(
            "interaction_override requires obstacle_count_mix or speed_density_mix"
        )
    env._long_corridor_interaction_override = interaction_override
    env._long_corridor_pause_steps_range = apply_pause_override(env, dynamic_pause_steps_range)
    env._long_corridor_random_2d_kinematics = validate_random_2d_kinematics(
        random_2d_kinematics
    )
    env._long_corridor_obstacle_counts = (
        int(static_obstacles),
        int(dynamic_obstacles),
    )
    _hide_corridor_walls(env, ids)
    env._long_corridor_reset_count += int(ids.numel())

    # Previous-stage replay runs immediately before this event. Exclude those
    # envs and compensate the Bernoulli probability so `fraction` remains the
    # absolute long-corridor share of all resets.
    eligible = ids
    conditional_fraction = float(fraction)
    if hasattr(env, "_previous_stage_replay_active"):
        eligible = ids[~env._previous_stage_replay_active[ids]]
        previous_fraction = float(
            getattr(env, "_previous_stage_replay_fraction", 0.0)
        )
        conditional_fraction = min(
            float(fraction) / max(1.0 - previous_fraction, 1e-6),
            1.0,
        )
    if eligible.numel() == 0:
        return
    selected = eligible[
        torch.rand(eligible.numel(), device=env.device)
        < conditional_fraction
    ]
    if selected.numel() == 0:
        return

    _hide_original_geometry(env, selected)
    origins = env.scene.env_origins[selected]
    centers, sizes = wall_geometry(selected.numel(), spec, env.device)
    for index, name in enumerate(_ASSET_NAMES):
        pose = torch.zeros(selected.numel(), 7, device=env.device)
        pose[:, 0] = origins[:, 0] + centers[:, index, 0]
        pose[:, 1] = origins[:, 1] + centers[:, index, 1]
        pose[:, 2] = wall_z
        pose[:, 3] = 1.0
        env.scene[name].write_root_pose_to_sim(pose, env_ids=selected)
    env._long_corridor_wall_centers[selected] = centers
    env._long_corridor_wall_sizes[selected] = sizes
    env._long_corridor_wall_mask[selected] = True

    robot = env.scene["robot"]
    robot_pose = robot.data.default_root_state[selected, :7].clone()
    robot_pose[:, 0] = origins[:, 0]
    robot_pose[:, 1] = origins[:, 1] + spec.robot_start_y
    robot_pose[:, 3:7] = 0.0
    robot_pose[:, 3] = math.cos(math.pi / 4.0)
    robot_pose[:, 6] = math.sin(math.pi / 4.0)
    robot.write_root_pose_to_sim(robot_pose, env_ids=selected)
    robot.write_root_velocity_to_sim(
        torch.zeros(selected.numel(), 6, device=env.device), env_ids=selected
    )

    goal = origins.clone()
    goal[:, 1] = origins[:, 1] + spec.goal_y
    goal[:, 2] = 0.0
    env._long_corridor_goal_w[selected] = goal
    env._long_corridor_active[selected] = True
    _write_goal(env, selected, update_markers=True)
    # gate-aligned 分流：SA7 的診斷顯示訓練走 count-mix（逐 env 抽 D=1..5、
    # 每 env 混三種 family、24.6% 有強制配對），而正式 Gate 是**固定 4S+2D
    # 且四個模式各自純化** —— 兩者是不同題型，訓練分佈幾乎不含 Gate 的場景。
    # `gate_aligned_share` 把一部分走廊 env 換成 Gate 題型，四模式均衡輪派。
    _gate_share = float(gate_aligned_share or 0.0)
    #: count-mix 取樣器的審計母體。Gate 題型是固定 4S+2D，混進來會讓
    #: 「取樣器有沒有重現凍結表」這個問題失去意義（4S2D 會被灌爆）。
    mix_selected = selected
    _mixed_profiles_enabled = (
        obstacle_count_mix is not None or speed_density_mix is not None
    )
    if _gate_share > 0.0 and _mixed_profiles_enabled and selected.numel() > 0:
        is_gate = torch.rand(selected.numel(), device=env.device) < _gate_share
        gate_ids = selected[is_gate]
        mix_ids = selected[~is_gate]
        mix_selected = mix_ids
        installed = True
        if gate_ids.numel() > 0:
            # Gate 題型走 legacy install，不經過 count-mix 取樣器，所以它不會
            # 寫 `_long_corridor_density_counts`。若放著不管，這些 env 會以
            # (0,0) 留在帳本裡，被 per-env 歸因與首發審計讀成一個不存在的
            # 「0S0D」密度組合 —— 跟先前「56 個非走廊 env 被算成 0S0D」
            # 完全同型的帳本脫節。這裡直接寫入它們真正的固定密度。
            env._long_corridor_density_counts[gate_ids, 0] = _GATE_ALIGNED_STATIC
            env._long_corridor_density_counts[gate_ids, 1] = _GATE_ALIGNED_DYNAMIC
            order = torch.randperm(gate_ids.numel(), device=env.device)
            for k, gate_mode in enumerate(_GATE_ALIGNED_MODES):
                sub = gate_ids[order[k::len(_GATE_ALIGNED_MODES)]]
                if sub.numel() == 0:
                    continue
                installed &= bool(_install_obstacles(
                    env, sub, spec, dynamic_speed_min, dynamic_speed_max,
                    _GATE_ALIGNED_STATIC, _GATE_ALIGNED_DYNAMIC,
                    gate_mode, None, count_mix=None,
                    speed_density_mix=None,
                    interaction_override=None,
                ))
                env._long_corridor_gate_aligned_counts[k] += int(sub.numel())
        if mix_ids.numel() > 0:
            installed &= bool(_install_obstacles(
                env, mix_ids, spec, dynamic_speed_min, dynamic_speed_max,
                static_obstacles, dynamic_obstacles, motion_mode, motion_weights,
                count_mix=obstacle_count_mix,
                speed_density_mix=speed_density_mix,
                interaction_override=interaction_override,
            ))
            env._long_corridor_mixed_env_total += int(mix_ids.numel())
    else:
        installed = _install_obstacles(
            env,
            selected,
            spec,
            dynamic_speed_min,
            dynamic_speed_max,
            static_obstacles,
            dynamic_obstacles,
            motion_mode,
            motion_weights,
            count_mix=obstacle_count_mix,
            speed_density_mix=speed_density_mix,
            interaction_override=interaction_override,
        )

    env._long_corridor_injected_count += int(selected.numel())
    if not getattr(env, "_long_corridor_logged", False):
        env._long_corridor_logged = True
        pending = 0 if installed else selected.numel()
        motion_audit = ""
        if installed and _mixed_profiles_enabled:
            # 只統計**這批真的裝進走廊**的 env。density_counts 現在是
            # per-env 全域欄位，整片拿去 bincount 會把 56 個非走廊 env
            # 算成一個叫「0S0D」的密度組合。
            counts = env._long_corridor_density_counts[mix_selected]
            combo = torch.bincount(
                counts[:, 0] * (MAX_CORRIDOR_DYNAMIC + 1) + counts[:, 1],
                minlength=(MAX_CORRIDOR_STATIC + 1) * (MAX_CORRIDOR_DYNAMIC + 1),
            )
            realized = {
                f"{s}S{d}D": int(combo[s * (MAX_CORRIDOR_DYNAMIC + 1) + d])
                for s in range(MAX_CORRIDOR_STATIC + 1)
                for d in range(MAX_CORRIDOR_DYNAMIC + 1)
                if int(combo[s * (MAX_CORRIDOR_DYNAMIC + 1) + d]) > 0
            }
            profile_audit = ""
            if speed_density_mix is not None and mix_selected.numel() > 0:
                profile_ids = env._long_corridor_speed_density_profile[mix_selected]
                if bool((profile_ids < 0).any()):
                    raise RuntimeError(
                        "joint speed-density install left an unassigned profile ID"
                    )
                profile_counts = torch.bincount(
                    profile_ids, minlength=len(speed_density_mix)
                )
                profile_realized = {
                    (
                        f"{counts_[0]}S{counts_[1]}D@"
                        f"{speed_range[0]:.2f}-{speed_range[1]:.2f}"
                    ): int(profile_counts[index])
                    for index, (counts_, speed_range, _) in enumerate(
                        speed_density_mix
                    )
                }
                expected_ranges = torch.tensor(
                    [speed_range for _, speed_range, _ in speed_density_mix],
                    dtype=env._long_corridor_sampled_speed_range.dtype,
                    device=env.device,
                )[profile_ids]
                actual_ranges = env._long_corridor_sampled_speed_range[mix_selected]
                range_mismatch = int(
                    (~torch.isclose(actual_ranges, expected_ranges)).any(dim=1).sum()
                )
                if range_mismatch:
                    raise RuntimeError(
                        "joint speed-density profile/range pairing drifted for "
                        f"{range_mismatch} env(s)"
                    )
                profile_audit = (
                    f" speed_density_profiles={profile_realized}"
                    f" profile_range_mismatch={range_mismatch}"
                )
            slow_min, slow_n, slow_total = getattr(
                env, "_long_corridor_density_speed_audit", (float("nan"), -1, -1)
            )
            interaction = getattr(env, "_long_corridor_interaction_audit", {})
            rnd_n, rnd_walk, want_wander = getattr(
                env, "_long_corridor_wander_audit", (0, 0, False)
            )
            gate_audit = ""
            if _gate_share > 0.0:
                gate_audit = (
                    " gate_aligned="
                    + ",".join(
                        f"{mode} {n}"
                        for mode, n in zip(
                            _GATE_ALIGNED_MODES,
                            env._long_corridor_gate_aligned_counts,
                        )
                    )
                    + f" mix_envs={int(mix_selected.numel())}"
                )
            motion_audit = (
                f"{gate_audit}"
                f" density_mix_realized={realized}"
                f"{profile_audit}"
                f" interaction_geometry_ok="
                f"crossing {interaction.get('crossing_ok', 0)}/"
                f"{interaction.get('crossing_n', 0)},"
                f"side_by_side {interaction.get('side_ok', 0)}/"
                f"{interaction.get('side_n', 0)}"
                f" random_2d_slots={rnd_n}"
                f" installed_as_random_walk={rnd_walk}"
                f" wander_requested={want_wander}"
                f" active_dynamic_slots={slow_total}"
                f" min_speed={slow_min:.3f}m/s"
                f" below_future_occupancy_threshold={slow_n}"
            )
        elif installed and dynamic_obstacles > 0:
            active_types = env._long_corridor_dynamic_motion_type[
                selected, :dynamic_obstacles
            ]
            type_counts = torch.bincount(
                active_types.flatten(), minlength=3
            ).tolist()
            pure_env_fraction = float(
                (active_types == active_types[:, :1])
                .all(dim=1)
                .float()
                .mean()
                .item()
            )
            env_counts = torch.bincount(
                active_types[:, 0], minlength=3
            ).tolist()
            motion_audit = (
                f" configured_motion_weights="
                f"{getattr(env, '_long_corridor_motion_weights', None)} "
                f"motion_env_counts={env_counts} "
                f"motion_slot_counts={type_counts} "
                f"pure_env_fraction={pure_env_fraction:.3f}"
            )
        print(
            "[LONG-CORRIDOR] injector FIRED: "
            f"{selected.numel()}/{ids.numel()} envs "
            f"free_width={spec.free_width:.2f}m length={spec.length:.2f}m "
            f"walls_x=+/-{spec.wall_center_offset:.2f}m "
            f"obstacles="
            f"{'joint speed-density mix' if speed_density_mix is not None else ('per-env mix' if obstacle_count_mix is not None else f'{static_obstacles}S+{dynamic_obstacles}D')} "
            f"speed=[{dynamic_speed_min:.2f},"
            f"{dynamic_speed_max:.2f}]m/s motion={motion_mode} "
            f"pending_scheduler={pending} "
            f"constructive_solvability=100%{motion_audit}",
            flush=True,
        )


def maintain_long_corridor_goal(env, env_ids=None) -> None:
    """Pin the goal and finish delayed obstacle installation if necessary."""
    if not hasattr(env, "_long_corridor_active"):
        return
    active = env._long_corridor_active
    if env_ids is None:
        selected = active.nonzero(as_tuple=False).flatten()
    else:
        ids = _as_env_ids(env, env_ids)
        selected = ids[active[ids]]
    if selected.numel() == 0:
        return

    _verify_pairs_this_step(env, selected)

    pending = selected[env._long_corridor_pending_obstacles[selected]]
    if pending.numel() > 0:
        speed_min, speed_max = env._long_corridor_speed_range
        _install_obstacles(
            env,
            pending,
            env._long_corridor_spec,
            speed_min,
            speed_max,
            *env._long_corridor_obstacle_counts,
            env._long_corridor_motion_mode,
            getattr(env, "_long_corridor_motion_weights", None),
            count_mix=getattr(env, "_long_corridor_obstacle_count_mix", None),
            speed_density_mix=getattr(
                env, "_long_corridor_speed_density_mix", None
            ),
            interaction_override=getattr(
                env, "_long_corridor_interaction_override", None
            ),
        )
    _write_goal(env, selected)
