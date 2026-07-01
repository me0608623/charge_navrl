"""Shared env_cfg overrides for Charge training entrypoints.

This module centralizes the runtime env_cfg patches that were historically
implemented only in train_charge_ac.py, so alternative trainers (for example
the modular RNN trainer) can reproduce the same v21-style experiment settings.
"""


def apply_charge_env_overrides(env_cfg, args_cli):
    """Apply CLI-driven reward/action/curriculum overrides to env_cfg."""
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return

    changed = False

    # --- v_gate_mode ---
    if args_cli.v_gate_mode != "baseline":
        vt = getattr(rewards, "velocity_to_goal", None)
        if vt is not None:
            if args_cli.v_gate_mode == "floor":
                vt.params["v_gate_floor"] = 0.2
            elif args_cli.v_gate_mode == "softer":
                vt.params["d_attenuate"] = 0.6
            print(f"[ABLATION] v_gate_mode={args_cli.v_gate_mode}: {vt.params}")
            changed = True

    # --- progress_gate_mode ---
    if args_cli.progress_gate_mode != "baseline":
        sp = getattr(rewards, "safe_progress", None)
        if sp is not None:
            if args_cli.progress_gate_mode == "delayed_negative":
                sp.params["d_danger"] = 0.55
            elif args_cli.progress_gate_mode == "weaken_negative":
                sp.params["negative_scale"] = 0.2
            print(f"[ABLATION] progress_gate_mode={args_cli.progress_gate_mode}: {sp.params}")
            changed = True

    # --- gap reward ---
    if args_cli.use_gap_reward:
        from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.rewards.gap_rewards import (
            forward_clearance_improvement_reward,
            heading_to_gap_reward,
        )

        body_r = 0.35
        weight = args_cli.gap_reward_weight
        if args_cli.gap_reward_type in ("heading", "both"):
            rewards.heading_to_gap = RewTerm(
                func=heading_to_gap_reward,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg("lidar"),
                    "body_radius": body_r,
                    "min_gap_width": 0.9,
                    "activation_d_safe": 2.0,
                    "speed_threshold": 0.05,
                },
                weight=weight,
            )
        if args_cli.gap_reward_type in ("clearance", "both"):
            rewards.forward_clearance = RewTerm(
                func=forward_clearance_improvement_reward,
                params={
                    "robot_cfg": SceneEntityCfg("robot"),
                    "sensor_cfg": SceneEntityCfg("lidar"),
                    "body_radius": body_r,
                    "front_arc_bins": 12,
                    "activation_d_safe": 2.0,
                },
                weight=weight,
            )
        print(f"[ABLATION] gap_reward: type={args_cli.gap_reward_type} weight={weight}")
        changed = True

    # --- safety shield / VO shield ---
    use_safety_shield = getattr(args_cli, "use_safety_shield", False)
    use_vo_shield = getattr(args_cli, "use_vo_shield", False)
    if use_safety_shield or use_vo_shield:
        import math as _math

        from isaaclab.utils import configclass as _configclass

        body_r = 0.35
        mode = args_cli.shield_mode

        if use_vo_shield:
            from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.actions.vo_safety_shield import (
                VOShieldedDiscreteDifferentialDriveActionCfg,
            )

            @_configclass
            class _ShieldedActions:
                diff_drive = VOShieldedDiscreteDifferentialDriveActionCfg(
                    asset_name="robot",
                    debug_vis=True,
                    num_bins=19,
                    max_linear_velocity=1.0,
                    max_linear_accel=0.5,
                    max_angular_vel=0.25 * _math.pi,
                    shield_mode=mode,
                    shield_d_danger=0.55,
                    shield_d_safe=1.2,
                    sensor_name="lidar",
                    body_radius=body_r,
                    vo_horizon=getattr(args_cli, "vo_horizon", 1.0),
                    vo_safety_radius=getattr(args_cli, "vo_safety_radius", 0.45),
                    vo_evade_gain=getattr(args_cli, "vo_evade_gain", 1.0),
                )

            env_cfg.actions = _ShieldedActions()
            print(
                f"[ABLATION] vo_shield: mode={mode} "
                f"horizon={getattr(args_cli, 'vo_horizon', 1.0)}s "
                f"safety_r={getattr(args_cli, 'vo_safety_radius', 0.45)}m "
                f"evade_gain={getattr(args_cli, 'vo_evade_gain', 1.0)}"
            )
        else:
            from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.actions.safety_shield import (
                ShieldedDiscreteDifferentialDriveActionCfg,
            )

            @_configclass
            class _ShieldedActions:
                diff_drive = ShieldedDiscreteDifferentialDriveActionCfg(
                    asset_name="robot",
                    debug_vis=True,
                    num_bins=19,
                    max_linear_velocity=1.0,
                    max_linear_accel=0.5,
                    max_angular_vel=0.25 * _math.pi,
                    shield_mode=mode,
                    shield_d_danger=0.55,
                    shield_d_safe=1.2,
                    sensor_name="lidar",
                    body_radius=body_r,
                )

            env_cfg.actions = _ShieldedActions()
            print(f"[ABLATION] safety_shield: mode={mode}")
        changed = True

    # --- directional gate ---
    if args_cli.directional_gate:
        dir_params = {
            "cone_half_bins": args_cli.gate_cone_half_bins,
            "cone_bottom_k": args_cli.gate_cone_bottom_k,
            "omni_blend": args_cli.gate_omni_blend,
        }
        gv = getattr(rewards, "goal_velocity", None)
        if gv is not None:
            gv.params["directional_gate"] = True
            gv.params.update(dir_params)
        gp = getattr(rewards, "goal_progress", None)
        if gp is not None:
            gp.params["directional_scale"] = True
            gp.params.update(dir_params)
        if gv or gp:
            print(
                f"[ABLATION] directional_gate: cone=±{args_cli.gate_cone_half_bins}bins "
                f"bottom_k={args_cli.gate_cone_bottom_k} blend={args_cli.gate_omni_blend}"
            )
            changed = True

    reward_mode = getattr(args_cli, "reward_mode", "current")

    # --- reward_mode variants ---
    if reward_mode == "navrl_ground_v1":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGround,
        )

        env_cfg.rewards = RewardsCfgVLP16NavRLGround()
        r = env_cfg.rewards
        r.reaching_goal.weight = args_cli.w_goal
        r.goal_velocity.weight = args_cli.w_vel
        r.goal_progress.weight = args_cli.w_prog
        r.static_safety.weight = args_cli.w_ss
        r.dynamic_safety.weight = args_cli.w_ds
        r.smoothness.weight = args_cli.w_smooth
        r.time_penalty.weight = args_cli.w_time
        r.collision_ground.weight = args_cli.w_collision
        r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
        r.goal_progress.params["scale_gamma"] = args_cli.progress_scale_gamma
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode
        if args_cli.directional_gate:
            for term_name, dir_key in (
                ("goal_velocity", "directional_gate"),
                ("goal_progress", "directional_scale"),
            ):
                term = getattr(r, term_name, None)
                if term is not None:
                    term.params[dir_key] = True
                    term.params["cone_half_bins"] = args_cli.gate_cone_half_bins
                    term.params["cone_bottom_k"] = args_cli.gate_cone_bottom_k
                    term.params["omni_blend"] = args_cli.gate_omni_blend
        dir_tag = ""
        if args_cli.directional_gate:
            dir_tag = (
                f" | dir_gate: cone=±{args_cli.gate_cone_half_bins}bins "
                f"bottom_k={args_cli.gate_cone_bottom_k} blend={args_cli.gate_omni_blend}"
            )
        print(
            f"[REWARD_MODE] navrl_ground_v1 | "
            f"w: goal={args_cli.w_goal} vel={args_cli.w_vel} prog={args_cli.w_prog} "
            f"ss={args_cli.w_ss} ds={args_cli.w_ds} smooth={args_cli.w_smooth} "
            f"time={args_cli.w_time} collision={args_cli.w_collision} | "
            f"beta={args_cli.goal_vel_gate_beta} gamma={args_cli.progress_scale_gamma} "
            f"ds_mode={args_cli.dynamic_safety_mode}{dir_tag}"
        )
        changed = True

    if reward_mode == "navrl_ground_v2":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV2,
        )

        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV2()
        r = env_cfg.rewards
        r.alive.weight = args_cli.w_alive
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode
        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
            print(f"[ABLATION] goal_velocity: soft_gate ON, beta={args_cli.goal_vel_gate_beta}")
        print(
            f"[REWARD_MODE] navrl_ground_v2 (no gate + alive) | "
            f"w: alive={args_cli.w_alive} vel=2.0 prog=3.0 ss=2.0 ds=2.0 "
            f"smooth=-0.1 goal=100 collision=-50 | "
            f"ds_mode={args_cli.dynamic_safety_mode}"
        )
        changed = True

    if reward_mode == "navrl_ground_v3":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV3,
        )

        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV3()
        r = env_cfg.rewards
        r.alive.weight = args_cli.w_alive
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode
        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
        print(
            f"[REWARD_MODE] navrl_ground_v3 (v2 + 20obs + density weights) | "
            f"w: alive={args_cli.w_alive} | "
            f"ds_mode={args_cli.dynamic_safety_mode} | "
            f"weights controlled by curriculum stage"
        )
        changed = True

    if reward_mode == "navrl_ground_v4":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV3,
        )

        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV3()
        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode
        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
        print(
            f"[REWARD_MODE] navrl_ground_v4 (v3 + goal=500 + no alive) | "
            f"w: reaching_goal={r.reaching_goal.weight} alive={r.alive.weight} | "
            f"ds_mode={args_cli.dynamic_safety_mode} | "
            f"weights controlled by curriculum stage"
        )
        changed = True

    if reward_mode == "navrl_ground_v5":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV5,
        )

        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV5()
        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode
        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
        print(
            f"[REWARD_MODE] navrl_ground_v5 (collision=-100 + ss_boost Stage 3-6) | "
            f"w: reaching_goal={r.reaching_goal.weight} "
            f"collision={r.collision_ground.weight} alive={r.alive.weight} | "
            f"ds_mode={args_cli.dynamic_safety_mode} | "
            f"搭配 goal_first_v3 使用"
        )
        changed = True

    if reward_mode == "navrl_ground_v6":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV6,
        )

        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV6()
        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode
        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
        print(
            f"[REWARD_MODE] navrl_ground_v6 (open-ended, front_block=0.4) | "
            f"w: goal={r.reaching_goal.weight} coll={r.collision_ground.weight} "
            f"alive={r.alive.weight} ss_front={r.static_safety.params['a_front_block']} | "
            f"ds_mode={args_cli.dynamic_safety_mode}"
        )
        changed = True

    if reward_mode == "navrl_ground_v7":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV7,
        )

        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV7()
        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode
        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
        print(
            f"[REWARD_MODE] navrl_ground_v7 (碰撞遞增: -5→-80) | "
            f"w: goal={r.reaching_goal.weight} coll_base={r.collision_ground.weight} "
            f"alive={r.alive.weight} ss_front={r.static_safety.params['a_front_block']} | "
            f"ds_mode={args_cli.dynamic_safety_mode} | "
            f"collision 由 curriculum 動態調整"
        )
        changed = True

    if reward_mode == "navrl_ground_v8":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            RewardsCfgVLP16NavRLGroundV8,
        )

        env_cfg.rewards = RewardsCfgVLP16NavRLGroundV8()
        r = env_cfg.rewards
        r.dynamic_safety.params["mode"] = args_cli.dynamic_safety_mode
        if args_cli.goal_vel_use_soft_gate:
            r.goal_velocity.params["use_soft_gate"] = True
            r.goal_velocity.params["gate_beta"] = args_cli.goal_vel_gate_beta
        print(
            f"[REWARD_MODE] navrl_ground_v8 (safety↓ vel↑ 修正比例失衡) | "
            f"w: goal={r.reaching_goal.weight} coll_base={r.collision_ground.weight} | "
            f"ss/ds weight 由 curriculum 控制 (上限 ss=0.5 ds=0.4) | "
            f"vel 下限=3.5"
        )
        changed = True

    # --- curriculum_version ---
    cv = getattr(args_cli, "curriculum_version", None)
    if cv is not None:
        cur = getattr(env_cfg, "curriculum", None)
        if cur is not None:
            term = getattr(cur, "goal_obstacle_curriculum", None)
            if term is not None:
                term.params["curriculum_version"] = cv
                print(f"[CURRICULUM] version={cv}")
                changed = True

    # --- no_walls ---
    if getattr(args_cli, "no_walls", False):
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            CURRICULUM_CONFIGS,
        )

        cv_key = cv or "baseline_v1"
        if cv_key in CURRICULUM_CONFIGS:
            for stage_cfg in CURRICULUM_CONFIGS[cv_key]["stages"]:
                stage_cfg["min_walls"] = 0
                stage_cfg["max_walls"] = 0
            print("[NO_WALLS] 所有 stage 的 min/max_walls 已設為 0（保留外牆）")
            changed = True

    # --- no_domain_randomization ---
    if getattr(args_cli, "no_domain_randomization", False):
        events = getattr(env_cfg, "events", None)
        if events is not None:
            dr = getattr(events, "domain_randomization", None)
            if dr is not None:
                dr.params["enable_physics"] = False
                dr.params["enable_sensor_noise"] = False
                dr.params["enable_external_force"] = False
                print("[NO_DR] domain_randomization event: physics/sensor_noise/external_force = False")
                print("[NO_DR] reset_base 保留原樣（pose/velocity range 不動）")
        changed = True

    # --- reward_speed_v05 ---
    if getattr(args_cli, "reward_speed_v05", False):
        rewards = getattr(env_cfg, "rewards", None)
        if rewards is not None and hasattr(rewards, "time_penalty"):
            rewards.time_penalty.weight = -0.6
            print("[REWARD_SPEED_V05] time_penalty weight: -0.2 → -0.6")

        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            CURRICULUM_CONFIGS,
        )

        cv_key = cv or "open_ended_v1"
        if cv_key in CURRICULUM_CONFIGS:
            stages = CURRICULUM_CONFIGS[cv_key]["stages"]
            for stage_cfg in stages:
                rw = stage_cfg.get("reward_weights")
                if rw is None:
                    continue
                if "goal_velocity" in rw:
                    rw["goal_velocity"] = round(rw["goal_velocity"] * 3.0, 2)
                if "static_safety" in rw:
                    rw["static_safety"] = round(rw["static_safety"] * 0.375, 3)
            print(
                f"[REWARD_SPEED_V05] {len(stages)} bootstrap stages: "
                f"goal_velocity ×3, static_safety ×0.375"
            )

        import importlib

        curr_mod = importlib.import_module(
            "isaaclab_tasks.manager_based.locomotion.velocity.config."
            "charge_skrl.curriculum.goal_obstacle_curriculum"
        )
        orig_oe_weights = curr_mod._open_ended_reward_weights

        def _patched_oe_weights(level: int) -> dict:
            rw = orig_oe_weights(level)
            rw["goal_velocity"] = round(rw.get("goal_velocity", 4.0) * 3.0, 2)
            rw["static_safety"] = round(rw.get("static_safety", 0.4) * 0.375, 3)
            return rw

        curr_mod._open_ended_reward_weights = _patched_oe_weights
        print("[REWARD_SPEED_V05] OE stages: _open_ended_reward_weights monkey-patched")
        print("[REWARD_SPEED_V05] 目標 speed ≈ 0.4-0.6 m/s, 預期 SR ≈ 88-92%")
        changed = True

    # --- ds rebalance ---
    need_ds_patch = (
        args_cli.ds_weight_boost != 1.0
        or args_cli.risk_sigma is not None
        or args_cli.b_risk is not None
    )
    if need_ds_patch:
        rewards = getattr(env_cfg, "rewards", None)
        if rewards is not None and hasattr(rewards, "dynamic_safety"):
            r = rewards
            ds_params = r.dynamic_safety.params
            if args_cli.risk_sigma is not None:
                old_sigma = ds_params.get("risk_sigma", "?")
                ds_params["risk_sigma"] = args_cli.risk_sigma
                print(f"[M1.4_DS] risk_sigma: {old_sigma} → {args_cli.risk_sigma}")
            if args_cli.b_risk is not None:
                old_b = ds_params.get("b_risk", "?")
                ds_params["b_risk"] = args_cli.b_risk
                print(f"[M1.4_DS] b_risk: {old_b} → {args_cli.b_risk}")
            if args_cli.ds_weight_boost != 1.0:
                import importlib as _il

                curr_mod_m14 = _il.import_module(
                    "isaaclab_tasks.manager_based.locomotion.velocity.config."
                    "charge_skrl.curriculum.goal_obstacle_curriculum"
                )
                orig_oe_for_ds = curr_mod_m14._open_ended_reward_weights
                ds_boost = args_cli.ds_weight_boost

                def _patched_oe_with_ds(level: int) -> dict:
                    rw = orig_oe_for_ds(level)
                    rw["dynamic_safety"] = round(rw.get("dynamic_safety", 0.4) * ds_boost, 3)
                    return rw

                curr_mod_m14._open_ended_reward_weights = _patched_oe_with_ds
                print(f"[M1.4_DS] OE dynamic_safety weight boost ×{ds_boost} (chained)")

                from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
                    CURRICULUM_CONFIGS,
                )

                cv_key = cv or "open_ended_v1"
                if cv_key in CURRICULUM_CONFIGS:
                    stages = CURRICULUM_CONFIGS[cv_key]["stages"]
                    patched_count = 0
                    for stage_cfg in stages:
                        rw = stage_cfg.get("reward_weights")
                        if rw is None or "dynamic_safety" not in rw:
                            continue
                        if rw["dynamic_safety"] > 0:
                            rw["dynamic_safety"] = round(rw["dynamic_safety"] * ds_boost, 3)
                            patched_count += 1
                    print(
                        f"[M1.4_DS] {patched_count} bootstrap stages: dynamic_safety ×{ds_boost}"
                    )
            print("[M1.4_DS] M1.4 ds rebalance applied. 預期 ds effective weight ↑ 4-10x")
            changed = True

    # --- Sim-to-Real LiDAR noise configuration ---
    # lidar_no_noise=True → all zeros (legacy shortcut)
    # lidar_no_noise=False → apply per-param values from YAML
    _apply_lidar_noise_config(env_cfg, args_cli)
    # Always count as changed if any lidar param was set
    if getattr(args_cli, "lidar_no_noise", False) or not getattr(args_cli, "lidar_no_noise", True):
        changed = True

    # --- Actuator DR (action delay, velocity scaling, motor lag) ---
    _apply_actuator_dr_config(env_cfg, args_cli)

    # --- Physics / Disturbance / Actuator DR param overrides ---
    _apply_dr_param_overrides(env_cfg, args_cli)
    changed = True

    # --- Ablation 5: safety weights ---
    ss_lower = getattr(args_cli, "ss_lower_mode", None)
    ss_raise = getattr(args_cli, "ss_raise_mode", False)
    if ss_lower or ss_raise:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.goal_obstacle_curriculum import (
            set_ablation_params,
        )

        set_ablation_params(ss_lower_mode=ss_lower, ss_raise=ss_raise)
        mode_str = f"ss_lower={ss_lower}" if ss_lower else f"ss_raise={ss_raise}"
        print(f"[ABLATION_5] Safety 權重調整: {mode_str}")
        changed = True

    if not changed:
        print("[ABLATION] baseline (no overrides)")


# ══════════════════════════════════════════════════════════════════════════════
# Sim-to-Real DR helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_lidar_obs_terms(env_cfg):
    """Yield (group_name, term_name, term) for all LiDAR ObsTerms."""
    obs_root = getattr(env_cfg, "observations", None)
    if obs_root is None:
        return
    for group_name in dir(obs_root):
        if group_name.startswith("_"):
            continue
        group = getattr(obs_root, group_name, None)
        if group is None or not hasattr(group, "__dict__"):
            continue
        for term_name in dir(group):
            if term_name.startswith("_"):
                continue
            term = getattr(group, term_name, None)
            if term is None or not hasattr(term, "params"):
                continue
            if "lidar" not in term_name.lower():
                continue
            yield group_name, term_name, term


# ── VLP-16 empirical noise (measured white_wall, 2026-07-01) ─────────────────
# Source: vlp16_noise/isaac_lab_noise_params.py. σ is distance-independent (R²=0.078),
# so it maps to the fixed-σ "soft" slot. The per-ring systematic bias lives in
# obs_functions.wd_like_sweep_72 (activated by per_ring_bias=True); the global bias is NOT
# added separately, to avoid double-counting the common-mode already in the per-ring array.
_VLP16_SIGMA_FIXED_M = 0.008672     # measured range σ (point-to-plane residual std, ~8.67mm)
_VLP16_HOLE_RATE = 0.194859         # measured dropout (low-reflectivity / intensity<thr, ~19.5%)
_VLP16_DISTRACTOR_RATE = 0.002515   # measured mixed-pixel / edge-outlier rate

# --- Per-material: HUMAN (soft target) on dynamic-obstacle rays. Measured 2026-07-01. ---
# dropout rises with distance (1m .209 / 2m .223 / 3m .236) → linear dropout(d)=slope·d+intercept.
# σ = null (measured 6.5cm is an analyze_gap fallback, NOT sensor precision — do NOT use).
# ⚠ Absolute values are scene-level / PROVISIONAL, pending ROI return-rate re-measure; trend is usable.
_VLP16_HUMAN_DROPOUT_SLOPE = 0.0135
_VLP16_HUMAN_DROPOUT_INTERCEPT = 0.196
_VLP16_HUMAN_MIXED_PIXEL = 0.178    # high vs white_wall (0.0025) but noisy → provisional

_VLP16_NOISE_MODES = ("ideal", "sigma", "bias", "dropout", "full", "full_material")


def _apply_vlp16_ablation_mode(env_cfg, mode: str):
    """Override LiDAR ObsTerms with the measured VLP-16 preset for one ablation arm.

    Faithful (fixed) injection — no domain-randomization ranges. Reuses wd_like_sweep_72;
    adds no new noise math. Modes (README §5 in vlp16_noise/):
      ideal | sigma | bias | dropout | full
    """
    mode = str(mode).lower()
    if mode not in _VLP16_NOISE_MODES:
        raise ValueError(
            f"vlp16_noise_mode={mode!r} invalid; choose one of {_VLP16_NOISE_MODES}"
        )
    # full_material = full (white_wall on all rays) + human noise on dynamic-obstacle rays
    on_sigma = mode in ("sigma", "full", "full_material")
    on_bias = mode in ("bias", "full", "full_material")
    on_drop = mode in ("dropout", "full", "full_material")
    human = mode == "full_material"
    for _gn, _tn, term in _find_lidar_obs_terms(env_cfg):
        p = term.params
        # random range σ (fixed, distance-independent) → soft slot; kill distance-scaled path
        p["displacement_std_per_meter"] = 0.0
        p["displacement_std"] = 0.0
        p["displacement_std_soft"] = _VLP16_SIGMA_FIXED_M if on_sigma else 0.0
        # dropout family: hole (low-reflectivity) + mixed-pixel ghost
        p["hole_rate"] = _VLP16_HOLE_RATE if on_drop else 0.0
        p["distractor_rate"] = _VLP16_DISTRACTOR_RATE if on_drop else 0.0
        # systematic bias = measured per-ring array (carries common-mode); no separate global bias
        p["per_ring_bias"] = on_bias
        p["distance_bias_k"] = 0.0
        p["distance_bias_b"] = 0.0
        # faithful/fixed → clear every per-episode DR range key
        for k in list(p.keys()):
            if k.endswith("_dr_min") or k.endswith("_dr_max"):
                p[k] = 0.0
        # measured model has no L2 per-bin blanket and no block dropout
        p["block_dropout_prob"] = 0.0
        # per-material: human dropout(d) + mixed-pixel on rays hitting DYNAMIC obstacles (no σ)
        p["human_dynamic_dropout"] = human
        if human:
            p["human_dropout_slope"] = _VLP16_HUMAN_DROPOUT_SLOPE
            p["human_dropout_intercept"] = _VLP16_HUMAN_DROPOUT_INTERCEPT
            p["human_mixed_pixel_rate"] = _VLP16_HUMAN_MIXED_PIXEL
        if hasattr(term, "noise"):
            term.noise = None
    print(
        f"[SIM2REAL][VLP16-ablation] mode={mode} "
        f"σ={'ON' if on_sigma else 'off'} "
        f"bias={'ON' if on_bias else 'off'} "
        f"dropout={'ON' if on_drop else 'off'} "
        f"human_dyn={'ON' if human else 'off'} "
        f"(fixed measured values, no DR)"
    )


def _apply_lidar_noise_config(env_cfg, args_cli):
    """Apply LiDAR noise params from YAML/CLI to env_cfg ObsTerms.

    When vlp16_noise_mode is set, the measured VLP-16 ablation preset takes precedence.
    Otherwise: when lidar_no_noise=True all random noise is zeroed (bias kept);
    when lidar_no_noise=False, individual params are applied from YAML.
    """
    mode = getattr(args_cli, "vlp16_noise_mode", None)
    if mode:
        _apply_vlp16_ablation_mode(env_cfg, mode)
        return
    if getattr(args_cli, "lidar_no_noise", False):
        # Noise-only path: 歸零隨機 noise（每 step 變化），但保留 bias（硬體屬性）
        # bias 是「這台機器人的固定特性」，連 SA1 bootstrap 也該訓 policy 適應
        # 只關掉真正的「random noise」: displacement / hole / distractor / obs term noise
        for gn, tn, term in _find_lidar_obs_terms(env_cfg):
            params = term.params
            if "displacement_std" in params:
                params["displacement_std"] = 0.0
            if "displacement_std_per_meter" in params:
                params["displacement_std_per_meter"] = 0.0
            if "displacement_std_soft" in params:
                params["displacement_std_soft"] = 0.0
            if "hole_rate" in params:
                params["hole_rate"] = 0.0
            if "distractor_rate" in params:
                params["distractor_rate"] = 0.0
            if hasattr(term, "noise"):
                term.noise = None
            # 仍處理 bias（如果 yaml 啟用）— 不視為 "noise"
            if getattr(args_cli, "lidar_distance_bias", False):
                dr_k = getattr(args_cli, "lidar_distance_bias_k_dr", None)
                dr_b = getattr(args_cli, "lidar_distance_bias_b_dr", None)
                if dr_k and dr_b:
                    params["distance_bias_k_dr_min"] = dr_k[0]
                    params["distance_bias_k_dr_max"] = dr_k[1]
                    params["distance_bias_b_dr_min"] = dr_b[0]
                    params["distance_bias_b_dr_max"] = dr_b[1]
                else:
                    params["distance_bias_k"] = 0.021
                    params["distance_bias_b"] = -0.030
            if getattr(args_cli, "lidar_per_ring_bias", False):
                params["per_ring_bias"] = True
        _dist_on = getattr(args_cli, "lidar_distance_bias", False)
        _ring_on = getattr(args_cli, "lidar_per_ring_bias", False)
        print(
            f"[SIM2REAL] lidar_no_noise=True → random noise off, "
            f"bias kept: dist_bias={'ON' if _dist_on else 'OFF'} "
            f"ring_bias={'ON' if _ring_on else 'OFF'}"
        )
        return

    # Fine-grained per-param configuration
    disp_std = getattr(args_cli, "lidar_displacement_std", None)
    disp_std_per_meter = getattr(args_cli, "lidar_displacement_std_per_meter", None)
    hole_rate = getattr(args_cli, "lidar_hole_rate", None)
    distractor_rate = getattr(args_cli, "lidar_distractor_rate", None)
    obs_noise_std = getattr(args_cli, "lidar_obs_noise_std", None)
    distance_bias = getattr(args_cli, "lidar_distance_bias", False)
    per_ring_bias = getattr(args_cli, "lidar_per_ring_bias", False)
    block_dropout_prob = getattr(args_cli, "lidar_block_dropout_prob", 0.0)
    block_dropout_width = getattr(args_cli, "lidar_block_dropout_width", (3, 8))
    disp_std_dr = getattr(args_cli, "lidar_displacement_std_dr", None)
    disp_std_per_meter_dr = getattr(args_cli, "lidar_displacement_std_per_meter_dr", None)
    disp_std_soft = getattr(args_cli, "lidar_displacement_std_soft", None)
    disp_std_soft_dr = getattr(args_cli, "lidar_displacement_std_soft_dr", None)
    hole_rate_dr = getattr(args_cli, "lidar_hole_rate_dr", None)

    any_set = False
    for gn, tn, term in _find_lidar_obs_terms(env_cfg):
        params = term.params

        # Layer 1: Per-Ray params
        if disp_std_per_meter is not None:
            params["displacement_std_per_meter"] = disp_std_per_meter
            params["displacement_std"] = 0.0  # disable legacy fixed-σ
            any_set = True
        elif disp_std is not None and "displacement_std" in params:
            params["displacement_std"] = disp_std
            any_set = True
        if disp_std_soft is not None:
            params["displacement_std_soft"] = disp_std_soft
            any_set = True
        if hole_rate is not None and "hole_rate" in params:
            params["hole_rate"] = hole_rate
            any_set = True
        if distractor_rate is not None and "distractor_rate" in params:
            params["distractor_rate"] = distractor_rate
            any_set = True

        # Layer 1 NEW: distance-dependent bias & per-ring bias
        # distance_bias 兩種模式：
        #   (a) DR 範圍：per-episode 採樣 k,b（不假設模型形狀，推薦）
        #   (b) 固定 k,b：legacy linear fit（R²=0.58，已知殘差有結構）
        if distance_bias:
            dr_k = getattr(args_cli, "lidar_distance_bias_k_dr", None)
            dr_b = getattr(args_cli, "lidar_distance_bias_b_dr", None)
            if dr_k and dr_b:
                # DR mode: per-episode sampling
                params["distance_bias_k_dr_min"] = dr_k[0]
                params["distance_bias_k_dr_max"] = dr_k[1]
                params["distance_bias_b_dr_min"] = dr_b[0]
                params["distance_bias_b_dr_max"] = dr_b[1]
            else:
                # Legacy fixed mode
                params["distance_bias_k"] = 0.021
                params["distance_bias_b"] = -0.030
            any_set = True
        if per_ring_bias:
            params["per_ring_bias"] = True
            any_set = True

        # Layer 2: Block dropout
        if block_dropout_prob > 0:
            params["block_dropout_prob"] = block_dropout_prob
            params["block_dropout_width_min"] = block_dropout_width[0]
            params["block_dropout_width_max"] = block_dropout_width[1]
            any_set = True

        # Layer 2: ObsTerm Gaussian noise (replace Unoise with Gnoise)
        if obs_noise_std is not None and obs_noise_std > 0:
            from isaaclab.utils.noise import GaussianNoiseCfg
            term.noise = GaussianNoiseCfg(std=obs_noise_std)
            any_set = True
        elif obs_noise_std == 0.0:
            term.noise = None
            any_set = True

        # Layer 3: Per-episode DR ranges (distance-dependent, preferred)
        if disp_std_per_meter_dr is not None:
            params["displacement_std_per_meter_dr_min"] = disp_std_per_meter_dr[0]
            params["displacement_std_per_meter_dr_max"] = disp_std_per_meter_dr[1]
            any_set = True
        elif disp_std_dr is not None:
            # legacy fixed-σ DR (kept for backward compat, absorbed by function)
            params["displacement_std_dr_min"] = disp_std_dr[0]
            params["displacement_std_dr_max"] = disp_std_dr[1]
            any_set = True
        if disp_std_soft_dr is not None:
            params["displacement_std_soft_dr_min"] = disp_std_soft_dr[0]
            params["displacement_std_soft_dr_max"] = disp_std_soft_dr[1]
            any_set = True
        if hole_rate_dr is not None:
            params["hole_rate_dr_min"] = hole_rate_dr[0]
            params["hole_rate_dr_max"] = hole_rate_dr[1]
            any_set = True

    if any_set:
        print(
            f"[SIM2REAL] LiDAR noise config: "
            f"disp_σ={disp_std} hole={hole_rate} ghost={distractor_rate} "
            f"obs_noise_σ={obs_noise_std} "
            f"dist_bias={'ON' if distance_bias else 'OFF'} "
            f"ring_bias={'ON' if per_ring_bias else 'OFF'} "
            f"block_drop={block_dropout_prob}"
        )
        if disp_std_dr:
            print(f"[SIM2REAL]   L3 DR: disp_σ∈{disp_std_dr} hole∈{hole_rate_dr}")
        if disp_std_soft_dr:
            print(f"[SIM2REAL]   L3 DR: soft_σ∈{disp_std_soft_dr} (human target noise)")


def _apply_actuator_dr_config(env_cfg, args_cli):
    """Wire actuator DR (action delay, velocity scaling, motor lag) into ActionsCfg.

    Reads enable_actuator_dr / actuator_delay_range / actuator_velocity_scale /
    actuator_motor_lag from args_cli and sets them on env_cfg.actions.diff_drive.
    Safe no-op if action term does not expose these fields (legacy configs).
    """
    if not getattr(args_cli, "enable_actuator_dr", False):
        return

    actions = getattr(env_cfg, "actions", None)
    if actions is None:
        return
    diff = getattr(actions, "diff_drive", None)
    if diff is None or not hasattr(diff, "enable_actuator_dr"):
        return  # action term doesn't support actuator DR (e.g. legacy continuous drive)

    diff.enable_actuator_dr = True
    delay_range = getattr(args_cli, "actuator_delay_range", None)
    if delay_range is not None:
        diff.actuator_delay_range = tuple(delay_range)
    vel_scale = getattr(args_cli, "actuator_velocity_scale", None)
    if vel_scale is not None:
        diff.actuator_velocity_scale = tuple(vel_scale)
    motor_lag = getattr(args_cli, "actuator_motor_lag", None)
    if motor_lag is not None:
        diff.actuator_motor_lag = float(motor_lag)

    print(
        f"[SIM2REAL] Actuator DR: "
        f"delay={diff.actuator_delay_range} steps, "
        f"vel_scale={diff.actuator_velocity_scale}, "
        f"motor_lag α={diff.actuator_motor_lag}"
    )


def _apply_dr_param_overrides(env_cfg, args_cli):
    """Apply physics/disturbance/actuator DR params from YAML to env_cfg EventTerms."""
    if getattr(args_cli, "no_domain_randomization", False):
        return  # DR disabled globally, handled elsewhere

    events = getattr(env_cfg, "events", None)
    if events is None:
        return

    dr_term = getattr(events, "domain_randomization", None)
    if dr_term is None:
        return

    params = dr_term.params
    any_set = False

    # Physics DR ranges
    mass_dr = getattr(args_cli, "physics_mass_dr", None)
    if mass_dr is not None:
        params["mass_scale"] = tuple(mass_dr)
        any_set = True

    friction_dr = getattr(args_cli, "physics_friction_dr", None)
    if friction_dr is not None:
        params["friction_scale"] = tuple(friction_dr)
        any_set = True

    com_offset = getattr(args_cli, "physics_com_offset", None)
    if com_offset is not None:
        params["com_offset"] = com_offset
        any_set = True

    # Disturbance DR
    wind_force = getattr(args_cli, "disturbance_wind_force", None)
    if wind_force is not None:
        params["wind_force_range"] = tuple(wind_force)
        any_set = True

    push_force = getattr(args_cli, "disturbance_push_force", None)
    if push_force is not None:
        params["push_force_range"] = tuple(push_force)
        any_set = True

    push_ratio = getattr(args_cli, "disturbance_push_ratio", None)
    if push_ratio is not None:
        params["push_env_ratio"] = push_ratio
        any_set = True

    # Actuator DR
    enable_act = getattr(args_cli, "enable_actuator_dr", False)
    params["enable_actuator_dr"] = enable_act

    if any_set or enable_act:
        print(
            f"[SIM2REAL] DR overrides: "
            f"mass={params.get('mass_scale')} "
            f"friction={params.get('friction_scale')} "
            f"com={params.get('com_offset')} "
            f"wind={params.get('wind_force_range')} "
            f"push={params.get('push_force_range')}@{params.get('push_env_ratio')} "
            f"actuator={'ON' if enable_act else 'OFF'}"
        )
