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

    # --- lidar_no_noise ---
    if getattr(args_cli, "lidar_no_noise", False):
        obs_root = getattr(env_cfg, "observations", None)
        if obs_root is not None:
            cleared_terms = []
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
                    params = term.params
                    if "displacement_std" in params:
                        params["displacement_std"] = 0.0
                    if "hole_rate" in params:
                        params["hole_rate"] = 0.0
                    if "distractor_rate" in params:
                        params["distractor_rate"] = 0.0
                    if hasattr(term, "noise"):
                        term.noise = None
                    cleared_terms.append(f"{group_name}.{term_name}")
            print(f"[NO_LIDAR_NOISE] LiDAR observation noise disabled: {cleared_terms}")
            print(
                "[NO_LIDAR_NOISE] displacement_std=0, hole_rate=0, "
                "distractor_rate=0, Unoise=None"
            )
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
