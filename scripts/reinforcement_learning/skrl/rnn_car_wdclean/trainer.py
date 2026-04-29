"""Main training loop and PPO utilities -- extracted from train_rnn_car_wdclip.py."""

import math
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from buffers import ChargeRolloutBuffer, ObstacleRolloutBuffer
from rewards import compute_wd_charge_reward, compute_obstacle_reward
from obstacle_agent import build_obstacle_obs, apply_obstacle_actions
from metrics import MetricsCollector
from checkpoint import save_checkpoint, load_checkpoint
from utils import (
    RunningNormalizer,
    _AUX_DIM_NAMES,
    _param_l2_norm,
    _grad_l2_norm,
    _scale_grads,
    _snapshot_params,
    _param_delta_norm,
)
from models import (
    LidarStateExtractor,
    PreprocessRNN,
    PolicyHead,
    ValueHead,
    RNNStateManager,
    ObstaclePolicyFC,
    ObstacleValueFC,
    OBS_POLICY_OBS_DIM,
    NUM_BINS,
)
from aux_targets import build_wd_preprocess_targets, compute_wd_module_loss


# ============================================================================
# PPO Utilities
# ============================================================================

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def sample_action(logits):
    logits_a = logits[:, :NUM_BINS]
    logits_w = logits[:, NUM_BINS:]
    dist_a = Categorical(logits=logits_a)
    dist_w = Categorical(logits=logits_w)
    action_a = dist_a.sample()
    action_w = dist_w.sample()
    log_prob = dist_a.log_prob(action_a) + dist_w.log_prob(action_w)
    entropy = dist_a.entropy() + dist_w.entropy()
    return torch.stack([action_a, action_w], dim=-1), log_prob, entropy


def evaluate_actions(logits, actions):
    """Returns (log_prob, entropy_linear, entropy_angular)."""
    logits_a = logits[:, :NUM_BINS]
    logits_w = logits[:, NUM_BINS:]
    dist_a = Categorical(logits=logits_a)
    dist_w = Categorical(logits=logits_w)
    log_prob = dist_a.log_prob(actions[:, 0]) + dist_w.log_prob(actions[:, 1])
    return log_prob, dist_a.entropy(), dist_w.entropy()


def ppo_update_continuous(policy, value_fn, buffer, optimizer, epochs, mini_batches,
                          clip_eps, vf_coeff, ent_coeff, max_grad_norm, gamma, gae_lambda):
    """Generic PPO update for continuous action policies (obstacle)."""
    T = buffer.ptr
    B = buffer.B

    with torch.no_grad():
        last_obs = buffer.obs[T - 1]
        last_value = value_fn(last_obs).squeeze(-1)

    advantages, returns = compute_gae(
        buffer.rewards[:T], buffer.values[:T], buffer.dones[:T],
        last_value, gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    flat_obs = buffer.obs[:T].reshape(-1, buffer.obs.shape[-1])
    flat_actions = buffer.actions[:T].reshape(-1, buffer.actions.shape[-1])
    flat_log_probs = buffer.log_probs[:T].reshape(-1)
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)

    batch_size = T * B
    mini_batch_size = batch_size // mini_batches

    p_losses, v_losses, ent_vals, total_losses = [], [], [], []
    for _ in range(epochs):
        indices = torch.randperm(batch_size, device=flat_obs.device)
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            idx = indices[start:end]

            mb_obs = flat_obs[idx]
            mb_act = flat_actions[idx]
            mb_old_lp = flat_log_probs[idx]
            mb_adv = flat_adv[idx]
            mb_ret = flat_ret[idx]

            new_lp, entropy = policy.evaluate(mb_obs, mb_act)
            new_val = value_fn(mb_obs).squeeze(-1)

            ratio = (new_lp - mb_old_lp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(new_val, mb_ret)
            loss = p_loss + vf_coeff * v_loss - ent_coeff * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_fn.parameters()), max_grad_norm)
            optimizer.step()
            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
            ent_vals.append(entropy.mean().item())
            total_losses.append(loss.item())

    return {
        "policy_loss": np.mean(p_losses) if p_losses else 0.0,
        "value_loss": np.mean(v_losses) if v_losses else 0.0,
        "entropy": np.mean(ent_vals) if ent_vals else 0.0,
        "total_loss": np.mean(total_losses) if total_losses else 0.0,
    }


# ============================================================================
# Main entry point
# ============================================================================

def run(args_cli, headless_mode: bool):
    """Run the full training loop. Called from train.py after AppLauncher init."""

    import gymnasium as gym
    from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    import isaaclab_tasks  # noqa: register tasks
    from isaaclab_tasks.utils.hydra import hydra_task_config

    print("[INFO] Multi-Agent Modular RNN Training v5 -- WD-Principle (A2CK + vanilla RNN)")

    @hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):

        # --- Seed ---
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        torch.manual_seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        random.seed(args_cli.seed)

        # --- Env config overrides ---
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs

        if args_cli.lidar_no_noise:
            try:
                lidar_cfg = env_cfg.observations.policy.lidar_ranges
                lidar_cfg.params["displacement_std"] = 0.0
                lidar_cfg.params["hole_rate"] = 0.0
                lidar_cfg.params["distractor_rate"] = 0.0
                lidar_cfg.params["use_Unoise"] = False
                print("[INFO] LiDAR noise disabled")
            except Exception as e:
                print(f"[WARN] Failed to disable LiDAR noise: {e}")

        # --- Curriculum version ---
        cv = args_cli.curriculum_version
        cur = getattr(env_cfg, 'curriculum', None)
        if cur is not None:
            term = getattr(cur, 'goal_obstacle_curriculum', None)
            if term is not None:
                term.params["curriculum_version"] = cv
                term.params["initial_stage"] = args_cli.initial_stage
                term.params["fixed_stage"] = args_cli.fixed_stage
                print(
                    f"[INFO] Curriculum version: {cv}, initial_stage: {args_cli.initial_stage}, "
                    f"fixed_stage={args_cli.fixed_stage}"
                )

        # --- Create env ---
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = SkrlVecEnvWrapper(env, ml_framework="torch")
        num_envs = env.num_envs
        device = env.device

        # Activate obstacle policy mode
        env.unwrapped._obstacle_policy_active = True
        print(f"[INFO] Obstacle policy ACTIVE -- scripted motion DISABLED")

        obs_dim = env.observation_space.shape[-1]  # 139
        N_obs = args_cli.max_active_obstacles

        # --- Obstacle size + scene bound randomization ---
        _obs_size_rand = args_cli.obs_size_rand
        _scene_bound_rand = args_cli.scene_bound_rand
        _obs_collision_base = args_cli.obs_collision_base
        _scene_bound_base = args_cli.scene_bound_base

        env.unwrapped._obstacle_radii = torch.full(
            (num_envs, N_obs), _obs_collision_base, device=device)
        env.unwrapped._scene_bounds = torch.full(
            (num_envs,), _scene_bound_base, device=device)

        def _randomize_obstacle_sizes(env_ids: torch.Tensor, rand_range: float):
            if rand_range <= 0.0:
                return
            n = env_ids.shape[0]
            noise = (torch.rand(n, N_obs, device=device) - 0.5) * rand_range
            env.unwrapped._obstacle_radii[env_ids] = (_obs_collision_base + noise).clamp(min=0.3)

        def _randomize_scene_bounds(env_ids: torch.Tensor, rand_range: float):
            if rand_range <= 0.0:
                return
            n = env_ids.shape[0]
            noise = (torch.rand(n, device=device) - 0.5) * rand_range
            env.unwrapped._scene_bounds[env_ids] = (_scene_bound_base + noise).clamp(min=4.0)

        all_ids = torch.arange(num_envs, device=device)
        _randomize_obstacle_sizes(all_ids, _obs_size_rand)
        _randomize_scene_bounds(all_ids, _scene_bound_rand)

        print(f"[INFO] Randomization: obs_size_rand={_obs_size_rand} (base={_obs_collision_base}), "
              f"scene_bound_rand={_scene_bound_rand} (base={_scene_bound_base})")
        print(f"[INFO] Env: {args_cli.task}, {num_envs} envs, device={device}")
        print(f"[INFO] obs_dim={obs_dim}, max_active_obstacles={N_obs}")

        # --- Charge policy obs ---
        POLICY_OBS_INDICES = list(range(0, 78)) + [138]
        policy_obs_dim = len(POLICY_OBS_INDICES)  # 79

        # --- Build Charge models ---
        use_extractor = (args_cli.charge_encoder_mode == "extractor_rnn")
        if use_extractor:
            extractor = LidarStateExtractor().to(device)
            rnn_input_dim = extractor.output_dim  # 96
        else:
            extractor = None
            rnn_input_dim = policy_obs_dim  # 79
        preprocess_rnn = PreprocessRNN(
            input_dim=rnn_input_dim, fc_dim=args_cli.fc_dim,
            hidden_dim=args_cli.hidden_dim, preprocess_dim=args_cli.preprocess_dim,
            predict_dim=7,
            rnn_type=args_cli.rnn_type,
        ).to(device)
        _policy_obs_idx = torch.tensor(POLICY_OBS_INDICES, dtype=torch.long, device=device)
        rl_input_dim = policy_obs_dim + args_cli.preprocess_dim  # 79+12=91
        policy_head = PolicyHead(input_dim=rl_input_dim).to(device)
        value_head = ValueHead(input_dim=rl_input_dim).to(device)
        rnn_state = RNNStateManager(num_envs, args_cli.hidden_dim, device)
        obs_normalizer = RunningNormalizer(obs_dim, device)

        # --- RL optimizer: only policy/value heads ---
        charge_params_actor = list(policy_head.parameters())
        charge_params_critic = list(value_head.parameters())
        charge_params_rl = charge_params_actor + charge_params_critic
        charge_opt_rl = torch.optim.Adam(charge_params_rl, lr=args_cli.lr, eps=1e-5)

        # --- Aux optimizer: fine-grained param groups ---
        charge_params_rnn_cell = list(preprocess_rnn.rnn.parameters())
        charge_params_fc_front = list(preprocess_rnn.fc_front.parameters())
        charge_params_fc_middle = list(preprocess_rnn.fc_middle.parameters())
        charge_params_predict_head = list(preprocess_rnn.predict_head.parameters())
        charge_params_extractor = list(extractor.parameters()) if use_extractor else []
        charge_params_aux = charge_params_extractor + list(preprocess_rnn.parameters())

        _lr_predict_head = args_cli.aux_lr_predict_head
        _lr_fc_middle = args_cli.aux_lr_fc_middle
        _lr_fc_front = args_cli.aux_lr_fc_front
        _lr_extractor = args_cli.aux_lr_extractor

        _aux_param_groups = [
            {"params": charge_params_rnn_cell,     "lr": args_cli.rnn_lr},
            {"params": charge_params_predict_head, "lr": _lr_predict_head},
            {"params": charge_params_fc_middle,    "lr": _lr_fc_middle},
            {"params": charge_params_fc_front,     "lr": _lr_fc_front},
        ]
        if use_extractor:
            _aux_param_groups.append(
                {"params": charge_params_extractor, "lr": _lr_extractor},
            )
        charge_opt_aux = torch.optim.Adam(_aux_param_groups, eps=1e-5)

        for pg in charge_opt_rl.param_groups:
            pg["initial_lr"] = pg["lr"]
        for pg in charge_opt_aux.param_groups:
            pg["initial_lr"] = pg["lr"]

        total_charge_params = sum(p.numel() for p in charge_params_rl + charge_params_aux)
        _n_aux_groups = len(_aux_param_groups)
        print(f"[INFO] Encoder mode: {args_cli.charge_encoder_mode}")
        print(f"[INFO]   fc_front input dim: {rnn_input_dim}")
        print(f"[INFO]   Using extractor: {use_extractor}")
        print(f"[INFO] Charge: {policy_obs_dim}D + {args_cli.rnn_type} {args_cli.preprocess_dim}D = {rl_input_dim}D, {total_charge_params:,} params")
        print(f"[INFO] RL optimizer: policy_head+value_head lr={args_cli.lr}")
        print(f"[INFO] Grad clip: rl={args_cli.max_grad_norm}, aux={args_cli.aux_grad_clip if args_cli.aux_grad_clip is not None else args_cli.max_grad_norm}")
        print(f"[INFO] Aux optimizer ({_n_aux_groups} groups):")
        print(f"  rnn_cell:     lr={args_cli.rnn_lr}")
        print(f"  predict_head: lr={_lr_predict_head}")
        print(f"  fc_middle:    lr={_lr_fc_middle}")
        print(f"  fc_front:     lr={_lr_fc_front}  {'(frozen)' if _lr_fc_front == 0 else ''}")
        if use_extractor:
            print(f"  extractor:    lr={_lr_extractor}  {'(frozen)' if _lr_extractor == 0 else ''}")
        else:
            print(f"  extractor:    N/A (raw_fc_rnn mode, no extractor)")

        # --- Build Obstacle models ---
        obs_policy = ObstaclePolicyFC().to(device)
        obs_value = ObstacleValueFC().to(device)
        obs_params = list(obs_policy.parameters()) + list(obs_value.parameters())
        obs_optimizer = torch.optim.Adam(obs_params, lr=args_cli.obs_lr, eps=1e-5)

        total_obs_params = sum(p.numel() for p in obs_params)
        print(f"[INFO] Obstacle: {OBS_POLICY_OBS_DIM}D obs, 2D act, {total_obs_params:,} params")
        print(f"[INFO] Alternating: train_goal_rate={args_cli.train_goal_rate} "
              f"(charge:{args_cli.train_goal_rate-1}, obstacle:1)")

        # --- Training config ---
        RL = args_cli.rollout_length
        batch_size = num_envs * RL
        mini_batch_size = batch_size // args_cli.mini_batches
        total_timesteps = args_cli.timesteps
        num_iterations = total_timesteps // RL
        current_gamma = args_cli.gamma

        print(f"[INFO] {num_iterations} iterations, {RL} rollout, {batch_size:,} batch, "
              f"gamma={current_gamma} (WD constant)")
        print(f"[INFO] A2C mode: {'ON (no clipping)' if args_cli.use_a2c else 'OFF (PPO clip={})'.format(args_cli.clip_eps)}")
        print(
            f"[INFO] WD update monitor/clipping: "
            f"enabled={args_cli.wd_update_clip and not args_cli.wd_update_monitor_only} "
            f"monitor_only={args_cli.wd_update_monitor_only} "
            f"actor_grad_cap={args_cli.wd_actor_update_clip} critic_grad_cap={args_cli.wd_critic_update_clip}"
        )

        # --- Buffers ---
        charge_buf = ChargeRolloutBuffer(RL, num_envs, rl_input_dim, obs_dim, args_cli.hidden_dim, device)
        obs_buf = ObstacleRolloutBuffer(RL, num_envs, N_obs, OBS_POLICY_OBS_DIM, 2, device)

        # --- Metrics ---
        metrics = MetricsCollector(num_envs, N_obs, device)

        # --- Log dir + WandB ---
        run_name = args_cli.run_name or f"marl_{datetime.now().strftime('%m%d_%H%M')}"
        log_dir = os.path.join("logs", "rnn_car", run_name)
        os.makedirs(log_dir, exist_ok=True)

        wandb_run = None
        if headless_mode:
            try:
                import wandb
                wandb.init(
                    project="charge_skrl", name=run_name,
                    config={
                        "agent": "MARL-ModularRNN-A2CK-v5" if args_cli.use_a2c else "MARL-ModularRNN-PPO-v5",
                        "charge_encoder_mode": args_cli.charge_encoder_mode,
                        "num_envs": num_envs, "seed": args_cli.seed,
                        "charge_lr": args_cli.lr, "rnn_lr": args_cli.rnn_lr,
                        "aux_lr": args_cli.aux_lr,
                        "aux_lr_predict_head": _lr_predict_head,
                        "aux_lr_fc_middle": _lr_fc_middle,
                        "aux_lr_fc_front": _lr_fc_front,
                        "aux_lr_extractor": _lr_extractor if use_extractor else "N/A",
                        "max_grad_norm": args_cli.max_grad_norm,
                        "aux_grad_clip": args_cli.aux_grad_clip if args_cli.aux_grad_clip is not None else args_cli.max_grad_norm,
                        "wd_update_clip": args_cli.wd_update_clip and not args_cli.wd_update_monitor_only,
                        "wd_actor_grad_cap": args_cli.wd_actor_update_clip,
                        "wd_critic_grad_cap": args_cli.wd_critic_update_clip,
                        "vf_coeff": args_cli.vf_coeff,
                        "normalize_return": args_cli.normalize_return,
                        "gamma": args_cli.gamma, "use_a2c": args_cli.use_a2c,
                        "rnn_type": args_cli.rnn_type,
                        "hidden_dim": args_cli.hidden_dim,
                        "preprocess_dim": args_cli.preprocess_dim,
                        "obs_lr": args_cli.obs_lr, "obs_ent_coeff": args_cli.obs_ent_coeff,
                        "obs_speed_limit": args_cli.obs_speed_limit,
                        "train_goal_rate": args_cli.train_goal_rate,
                        "obs_reward_mode": args_cli.obs_reward_mode,
                        "rollout_length": RL, "ppo_epochs": args_cli.ppo_epochs,
                        "tbptt_len": args_cli.tbptt_len,
                        "lr_decay": args_cli.lr_decay,
                        "total_charge_params": total_charge_params,
                        "total_obs_params": total_obs_params,
                        "obs_size_rand": args_cli.obs_size_rand,
                        "obs_collision_base": args_cli.obs_collision_base,
                        "scene_bound_rand": args_cli.scene_bound_rand,
                        "scene_bound_base": args_cli.scene_bound_base,
                    },
                    tags=["marl", "obstacle-policy", "v5", "wd-principle"],
                )
                wandb_run = wandb.run
                print(f"[INFO] WandB: {wandb.run.name}")
            except Exception as e:
                print(f"[WARN] WandB init failed: {e}")

        # --- Load checkpoint ---
        if args_cli.checkpoint:
            load_checkpoint(
                args_cli.checkpoint,
                extractor=extractor, preprocess_rnn=preprocess_rnn,
                policy_head=policy_head, value_head=value_head,
                obs_policy=obs_policy, obs_value=obs_value,
                obs_normalizer=obs_normalizer, device=device,
                use_extractor=use_extractor,
            )
            print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

        # --- Initial reset ---
        obs, info = env.reset()
        start_time = time.time()

        # --- WD reward params (initial, Phase 1 defaults) ---
        _spot_penalty_hit = -5.0
        _spot_reward_get_goal = 40.0
        _spot_cost_operate = 0.03

        # --- WD entropy params ---
        _ent_coeff_linear = args_cli.ent_coeff_linear if args_cli.ent_coeff_linear > 0 else 0.10
        _ent_coeff_angular = args_cli.ent_coeff_angular if args_cli.ent_coeff_angular > 0 else 0.375
        if args_cli.ent_coeff > 0:
            _ent_coeff_linear = args_cli.ent_coeff
            _ent_coeff_angular = args_cli.ent_coeff

        print(
            f"[INFO] Warp Drive reward: "
            f"penalty_hit={_spot_penalty_hit}, get_goal={_spot_reward_get_goal}, "
            f"cost_operate={_spot_cost_operate}"
        )
        print(
            f"[INFO] Warp Drive entropy (A2CK per-head): "
            f"linear={_ent_coeff_linear:.3f}, angular={_ent_coeff_angular:.3f} "
            f"(Phase 1 defaults, auto-sync from curriculum)"
        )

        # ====================================================================
        # Training Loop
        # ====================================================================

        _prev_stage = -1
        _prev_obs_agent_active = True
        _prev_rnn_feature_mean = None

        for iteration in range(num_iterations):
            iter_start = time.time()
            charge_buf.reset()
            obs_buf.reset()
            metrics.reset()

            # === Determine who trains this iteration ===
            _n_dynamic = int(metrics._curriculum_info.get("num_obstacles_dynamic", N_obs))
            _obs_agent_active = _n_dynamic > 0
            train_charge = (iteration % args_cli.train_goal_rate != 1)
            train_obstacle = (iteration % args_cli.train_goal_rate == 1) and _obs_agent_active
            if not _obs_agent_active:
                train_charge = True

            # === WD: Reset optimizer momentum on phase change ===
            _cur_stage = int(metrics._curriculum_info.get("stage", 1))
            if _cur_stage != _prev_stage and _prev_stage > 0:
                for opt in [charge_opt_rl, charge_opt_aux, obs_optimizer]:
                    for group in opt.param_groups:
                        for p in group["params"]:
                            state = opt.state.get(p)
                            if state:
                                if "exp_avg" in state:
                                    state["exp_avg"].zero_()
                                if "exp_avg_sq" in state:
                                    state["exp_avg_sq"].zero_()
                print(f"[INFO] Phase {_prev_stage}->{_cur_stage}: optimizer momentum reset (WD: reset_model_mentum)")
                if _obs_agent_active != _prev_obs_agent_active:
                    print(f"[INFO] obs_agent {'ACTIVATED' if _obs_agent_active else 'DEACTIVATED'} "
                          f"(dynamic: {_n_dynamic})")
            _prev_stage = _cur_stage
            _prev_obs_agent_active = _obs_agent_active

            # === Sync params from curriculum ===
            if args_cli.obs_size_rand == 0.0:
                _obs_size_rand = metrics._curriculum_info.get("obs_size_rand", 0.0)
            if args_cli.scene_bound_rand == 0.0:
                _scene_bound_rand = metrics._curriculum_info.get("scene_bound_rand", 0.0)

            _spot_penalty_hit = metrics._curriculum_info.get("spot_penalty_hit", -5.0)
            _spot_reward_get_goal = metrics._curriculum_info.get("spot_reward_get_goal", 40.0)
            _spot_cost_operate = metrics._curriculum_info.get("spot_cost_operate", 0.0)

            if args_cli.ent_coeff_linear == 0.0:
                _ent_coeff_linear = metrics._curriculum_info.get("ent_coeff_linear", 0.30)
            if args_cli.ent_coeff_angular == 0.0:
                _ent_coeff_angular = metrics._curriculum_info.get("ent_coeff_angular", 0.375)
            if args_cli.ent_coeff > 0:
                _ent_coeff_linear = args_cli.ent_coeff
                _ent_coeff_angular = args_cli.ent_coeff

            _obs_speed_limit = metrics._curriculum_info.get(
                "obstacle_speed_rate", args_cli.obs_speed_limit)
            metrics._obs_speed_limit = _obs_speed_limit

            # === LR decay ===
            if args_cli.lr_decay > 0 and iteration > 0:
                decay = max(0.01, 1.0 - args_cli.lr_decay * iteration)
                for opt in [charge_opt_rl, charge_opt_aux]:
                    for pg in opt.param_groups:
                        pg["lr"] = pg.get("initial_lr", pg["lr"]) * decay

            # === Rollout ===
            if use_extractor: extractor.eval()
            preprocess_rnn.eval(); policy_head.eval(); value_head.eval()
            obs_policy.eval(); obs_value.eval()

            for step in range(RL):
                # --- 1. Charge forward ---
                with torch.no_grad():
                    obs_normalizer.update(obs)
                    obs_normed = obs_normalizer.normalize(obs)
                    features = extractor(obs_normed) if use_extractor else obs_normed[:, _policy_obs_idx]
                    hidden = rnn_state.get()
                    rnn_feat, _, new_hidden = preprocess_rnn(features, hidden)
                    p_obs = obs_normed[:, POLICY_OBS_INDICES]
                    _rnn_for_rl = (torch.zeros_like(rnn_feat)
                                   if args_cli.zero_preprocess_feature_for_rl else rnn_feat)
                    rl_in = torch.cat([p_obs, _rnn_for_rl], dim=-1)
                    logits = policy_head(rl_in)
                    value = value_head(rl_in).squeeze(-1)
                    actions, log_prob, _ = sample_action(logits)
                    goal_diagnostics = metrics.compute_goal_diagnostics(env.unwrapped)

                # --- 2. Env step ---
                next_obs, reward, terminated, truncated, info = env.step(actions.float())
                done = (terminated.squeeze(-1) | truncated.squeeze(-1)).float()
                env_reward_flat = reward.squeeze(-1)  # noqa: F841

                reward_flat, reward_breakdown = compute_wd_charge_reward(
                    env.unwrapped, actions, terminated, truncated,
                    _spot_penalty_hit, _spot_reward_get_goal, _spot_cost_operate,
                )

                # --- 3. Obstacle forward + apply ---
                if _obs_agent_active:
                    with torch.no_grad():
                        obs_obs = build_obstacle_obs(env.unwrapped, N_obs, device)
                        obs_flat = obs_obs.reshape(-1, OBS_POLICY_OBS_DIM)
                        obs_act, obs_lp, obs_ent = obs_policy.sample(obs_flat)
                        obs_val = obs_value(obs_flat).squeeze(-1)

                    apply_obstacle_actions(env.unwrapped, obs_act.reshape(num_envs, N_obs, 2),
                                           N_obs, dt=0.2, speed_limit=_obs_speed_limit)

                    obs_rew = compute_obstacle_reward(env.unwrapped, obs_obs, N_obs, args_cli.obs_reward_mode)
                    obs_done = done.unsqueeze(-1).expand(-1, N_obs).reshape(-1)
                else:
                    obs_obs = torch.zeros(num_envs, N_obs, OBS_POLICY_OBS_DIM, device=device)
                    obs_flat = obs_obs.reshape(-1, OBS_POLICY_OBS_DIM)
                    obs_act = torch.zeros(num_envs * N_obs, 2, dtype=torch.long, device=device)
                    obs_lp = torch.zeros(num_envs * N_obs, device=device)
                    obs_val = torch.zeros(num_envs * N_obs, device=device)
                    obs_rew = torch.zeros(num_envs * N_obs, device=device)
                    obs_done = done.unsqueeze(-1).expand(-1, N_obs).reshape(-1)

                # --- 4. Store transitions ---
                with torch.no_grad():
                    wd_aux_tgt = build_wd_preprocess_targets(
                        env.unwrapped, N_obs, device)
                charge_buf.add(rl_in, actions, log_prob, reward_flat, value, done, obs, hidden,
                               aux_target=wd_aux_tgt)
                obs_buf.add(obs_flat, obs_act, obs_lp, obs_rew, obs_val, obs_done)

                # --- 5. Metrics ---
                metrics.step(obs, reward, done, info,
                             obs_obs=obs_obs, obs_actions=obs_act.reshape(num_envs, N_obs, 2),
                             reward_breakdown=reward_breakdown,
                             goal_diagnostics=goal_diagnostics)

                # --- 6. Update states ---
                rnn_state.update(new_hidden)
                done_mask = done.bool()
                if done_mask.any():
                    done_ids = done_mask.nonzero(as_tuple=False).reshape(-1)
                    rnn_state.reset(done_ids)
                    if hasattr(env.unwrapped, "_obstacle_velocities"):
                        env.unwrapped._obstacle_velocities[done_ids] = 0.0
                    _randomize_obstacle_sizes(done_ids, _obs_size_rand)
                    _randomize_scene_bounds(done_ids, _scene_bound_rand)
                obs = next_obs

            # === Charge PPO Update ===
            charge_ppo_loss = 0.0
            charge_vf_loss = 0.0
            charge_entropy = 0.0
            aux_loss_val = 0.0
            if args_cli.play:
                train_charge = False
                train_obstacle = False
            aux_per_feature = {}
            wd_update_monitor = {}

            if train_charge:
                # Bootstrap
                with torch.no_grad():
                    obs_normed = obs_normalizer.normalize(obs)
                    features = extractor(obs_normed) if use_extractor else obs_normed[:, _policy_obs_idx]
                    hidden = rnn_state.get()
                    rnn_feat, _, _ = preprocess_rnn(features, hidden)
                    p_obs = obs_normed[:, POLICY_OBS_INDICES]
                    _rnn_for_rl = (torch.zeros_like(rnn_feat)
                                   if args_cli.zero_preprocess_feature_for_rl else rnn_feat)
                    rl_in = torch.cat([p_obs, _rnn_for_rl], dim=-1)
                    last_value = value_head(rl_in).squeeze(-1)

                advantages, returns = compute_gae(
                    charge_buf.rewards, charge_buf.values, charge_buf.dones,
                    last_value, current_gamma, args_cli.gae_lambda)
                if args_cli.normalize_return:
                    value_targets = (returns - returns.mean()) / (returns.std() + 1e-8)
                else:
                    value_targets = returns
                _raw_adv_std = advantages.std().item()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                policy_head.train(); value_head.train()

                flat_ri = charge_buf.rl_inputs.reshape(-1, rl_input_dim)
                flat_act = charge_buf.actions.reshape(-1, 2)
                flat_lp = charge_buf.log_probs.reshape(-1)
                flat_adv = advantages.reshape(-1)
                flat_ret_raw = returns.reshape(-1)
                flat_value_target = value_targets.reshape(-1)

                _ret_mean = flat_ret_raw.mean().item()
                _ret_std = flat_ret_raw.std().item()
                _ret_min = flat_ret_raw.min().item()
                _ret_max = flat_ret_raw.max().item()
                _target_mean = flat_value_target.mean().item()
                _target_std = flat_value_target.std().item()
                _target_min = flat_value_target.min().item()
                _target_max = flat_value_target.max().item()
                flat_value_pred = charge_buf.values[:RL].reshape(-1)
                value_error = flat_value_target - flat_value_pred
                _value_pred_mean = flat_value_pred.mean().item()
                _value_pred_std = flat_value_pred.std().item()
                _value_pred_min = flat_value_pred.min().item()
                _value_pred_max = flat_value_pred.max().item()
                _value_error_mean = value_error.mean().item()
                _value_error_std = value_error.std().item()
                _value_error_abs_max = value_error.abs().max().item()
                _adv_mean = flat_adv.mean().item()
                _adv_std = flat_adv.std().item()
                _adv_min = flat_adv.min().item()
                _adv_max = flat_adv.max().item()

                n_epochs = 1 if args_cli.use_a2c else args_cli.ppo_epochs

                ppo_l, vf_l, ent_l = [], [], []
                vf_raw_l, pl_clamped_l, vf_clamped_l = [], [], []
                pl_clamp_triggered_l, vf_clamp_triggered_l = [], []
                total_loss_l = []
                approx_kl_l, ent_lin_l, ent_ang_l = [], [], []
                ppo_clip_frac_l, ppo_ratio_l = [], []
                wd_actor_grad_l, wd_critic_grad_l = [], []
                wd_actor_update_l, wd_critic_update_l = [], []
                wd_actor_delta_l, wd_critic_delta_l = [], []
                wd_actor_clip_l, wd_critic_clip_l = [], []
                wd_module_entropy_l = []
                wd_actor_grad_post_l, wd_critic_grad_post_l = [], []
                wd_merged_grad_pre_l, wd_merged_grad_post_l = [], []
                wd_actor_update_ratio_l, wd_critic_update_ratio_l = [], []
                wd_actor_param_norm_l, wd_critic_param_norm_l = [], []

                for _ in range(n_epochs):
                    if args_cli.use_a2c:
                        batches = [(torch.arange(batch_size, device=device),)]
                    else:
                        idx = torch.randperm(batch_size, device=device)
                        batches = [(idx[s:min(s + mini_batch_size, batch_size)],)
                                   for s in range(0, batch_size, mini_batch_size)]

                    for (mb,) in batches:
                        nl = policy_head(flat_ri[mb])
                        nlp, ent_lin, ent_ang = evaluate_actions(nl, flat_act[mb])
                        nv = value_head(flat_ri[mb]).squeeze(-1)

                        with torch.no_grad():
                            _approx_kl = (flat_lp[mb] - nlp).mean().item()

                        if args_cli.use_a2c:
                            pl = -(nlp * flat_adv[mb].detach()).mean()
                        else:
                            ratio = (nlp - flat_lp[mb]).exp()
                            s1 = ratio * flat_adv[mb]
                            s2 = torch.clamp(ratio, 1 - args_cli.clip_eps, 1 + args_cli.clip_eps) * flat_adv[mb]
                            pl = -torch.min(s1, s2).mean()
                            with torch.no_grad():
                                _clip_frac = ((ratio - 1.0).abs() > args_cli.clip_eps).float().mean().item()
                                ppo_clip_frac_l.append(_clip_frac)
                                ppo_ratio_l.append(ratio.mean().item())

                        vl = F.mse_loss(nv, flat_value_target[mb])
                        vl_raw = vl.item()

                        entropy_loss = (_ent_coeff_linear * ent_lin.mean()
                                        + _ent_coeff_angular * ent_ang.mean())

                        _pl_clamp_triggered = abs(pl.item()) > 20.0
                        pl_clamped = pl
                        if _pl_clamp_triggered:
                            pl_clamped = pl * (20.0 / abs(pl.item()))
                        vf_term = args_cli.vf_coeff * vl
                        _vf_clamp_triggered = abs(vf_term.item()) > 30.0
                        if _vf_clamp_triggered:
                            max_30 = abs(max(min(30.0, abs(pl.item())), 10.0) / vl.item())
                            vl = vl * max_30

                        loss = pl_clamped + args_cli.vf_coeff * vl - entropy_loss
                        actor_before = _snapshot_params(charge_params_actor)
                        critic_before = _snapshot_params(charge_params_critic)

                        charge_opt_rl.zero_grad()
                        loss.backward()

                        actor_grad = _grad_l2_norm(charge_params_actor)
                        critic_grad = _grad_l2_norm(charge_params_critic)
                        actor_update_est = args_cli.lr * actor_grad
                        critic_update_est = args_cli.lr * critic_grad

                        actor_scale = 1.0
                        critic_scale = 1.0
                        if args_cli.wd_update_clip and not args_cli.wd_update_monitor_only:
                            if actor_grad > args_cli.wd_actor_update_clip:
                                actor_scale = args_cli.wd_actor_update_clip / (actor_grad + 1e-12)
                            if critic_grad > args_cli.wd_critic_update_clip:
                                critic_scale = args_cli.wd_critic_update_clip / (critic_grad + 1e-12)
                            _scale_grads(charge_params_actor, actor_scale)
                            _scale_grads(charge_params_critic, critic_scale)

                        _merged_pre = _grad_l2_norm(charge_params_rl)
                        if args_cli.wd_update_clip and not args_cli.wd_update_monitor_only:
                            _merged_post = _grad_l2_norm(charge_params_rl)
                        else:
                            _merged_post = nn.utils.clip_grad_norm_(charge_params_rl, args_cli.max_grad_norm)
                        _actor_grad_post = _grad_l2_norm(charge_params_actor)
                        _critic_grad_post = _grad_l2_norm(charge_params_critic)

                        charge_opt_rl.step()

                        actor_delta = _param_delta_norm(actor_before, charge_params_actor)
                        critic_delta = _param_delta_norm(critic_before, charge_params_critic)
                        module_entropy = math.log10(actor_update_est + args_cli.wd_module_entropy_eps) - math.log10(
                            critic_update_est + args_cli.wd_module_entropy_eps)

                        _actor_pnorm = _param_l2_norm(charge_params_actor)
                        _critic_pnorm = _param_l2_norm(charge_params_critic)

                        ppo_l.append(pl.item()); vf_l.append(vl.item())
                        ent_l.append((ent_lin.mean() + ent_ang.mean()).item())
                        vf_raw_l.append(vl_raw)
                        pl_clamped_l.append(pl_clamped.item())
                        vf_clamped_l.append(vl.item())
                        pl_clamp_triggered_l.append(float(_pl_clamp_triggered))
                        vf_clamp_triggered_l.append(float(_vf_clamp_triggered))
                        approx_kl_l.append(_approx_kl)
                        ent_lin_l.append(ent_lin.mean().item())
                        ent_ang_l.append(ent_ang.mean().item())
                        total_loss_l.append(loss.item())
                        wd_actor_grad_l.append(actor_grad)
                        wd_critic_grad_l.append(critic_grad)
                        wd_actor_update_l.append(actor_update_est)
                        wd_critic_update_l.append(critic_update_est)
                        wd_actor_delta_l.append(actor_delta)
                        wd_critic_delta_l.append(critic_delta)
                        wd_actor_clip_l.append(float(actor_scale < 1.0))
                        wd_critic_clip_l.append(float(critic_scale < 1.0))
                        wd_module_entropy_l.append(module_entropy)
                        wd_actor_grad_post_l.append(_actor_grad_post)
                        wd_critic_grad_post_l.append(_critic_grad_post)
                        wd_merged_grad_pre_l.append(_merged_pre)
                        wd_merged_grad_post_l.append(_merged_post if isinstance(_merged_post, float) else _merged_post.item())
                        wd_actor_param_norm_l.append(_actor_pnorm)
                        wd_critic_param_norm_l.append(_critic_pnorm)
                        wd_actor_update_ratio_l.append(actor_delta / (_actor_pnorm + 1e-12))
                        wd_critic_update_ratio_l.append(critic_delta / (_critic_pnorm + 1e-12))

                charge_ppo_loss = np.mean(ppo_l)
                charge_vf_loss = np.mean(vf_l)
                charge_entropy = np.mean(ent_l)
                wd_update_monitor = {
                    "wd_update/actor_grad_norm": float(np.mean(wd_actor_grad_l)) if wd_actor_grad_l else 0.0,
                    "wd_update/critic_grad_norm": float(np.mean(wd_critic_grad_l)) if wd_critic_grad_l else 0.0,
                    "wd_update/actor_update_est": float(np.mean(wd_actor_update_l)) if wd_actor_update_l else 0.0,
                    "wd_update/critic_update_est": float(np.mean(wd_critic_update_l)) if wd_critic_update_l else 0.0,
                    "wd_update/actor_param_delta_norm": float(np.mean(wd_actor_delta_l)) if wd_actor_delta_l else 0.0,
                    "wd_update/critic_param_delta_norm": float(np.mean(wd_critic_delta_l)) if wd_critic_delta_l else 0.0,
                    "wd_update/actor_clip_fraction": float(np.mean(wd_actor_clip_l)) if wd_actor_clip_l else 0.0,
                    "wd_update/critic_clip_fraction": float(np.mean(wd_critic_clip_l)) if wd_critic_clip_l else 0.0,
                    "wd_update/module_entropy": float(np.mean(wd_module_entropy_l)) if wd_module_entropy_l else 0.0,
                    "wd_update/actor_grad_norm_post_clip": float(np.mean(wd_actor_grad_post_l)) if wd_actor_grad_post_l else 0.0,
                    "wd_update/critic_grad_norm_post_clip": float(np.mean(wd_critic_grad_post_l)) if wd_critic_grad_post_l else 0.0,
                    "wd_update/merged_grad_norm_pre_clip": float(np.mean(wd_merged_grad_pre_l)) if wd_merged_grad_pre_l else 0.0,
                    "wd_update/merged_grad_norm_post_clip": float(np.mean(wd_merged_grad_post_l)) if wd_merged_grad_post_l else 0.0,
                    "wd_update/actor_param_norm": float(np.mean(wd_actor_param_norm_l)) if wd_actor_param_norm_l else 0.0,
                    "wd_update/critic_param_norm": float(np.mean(wd_critic_param_norm_l)) if wd_critic_param_norm_l else 0.0,
                    "wd_update/actor_update_ratio": float(np.mean(wd_actor_update_ratio_l)) if wd_actor_update_ratio_l else 0.0,
                    "wd_update/critic_update_ratio": float(np.mean(wd_critic_update_ratio_l)) if wd_critic_update_ratio_l else 0.0,
                    "rl/raw_vf_loss": float(np.mean(vf_raw_l)) if vf_raw_l else 0.0,
                    "rl/clamped_policy_loss": float(np.mean(pl_clamped_l)) if pl_clamped_l else 0.0,
                    "rl/clamped_vf_loss": float(np.mean(vf_clamped_l)) if vf_clamped_l else 0.0,
                    "rl/vf_coeff": args_cli.vf_coeff,
                    "rl/vf_coeff_times_vf_loss": args_cli.vf_coeff * float(np.mean(vf_raw_l)) if vf_raw_l else 0.0,
                    "rl/policy_clamp_triggered": float(np.mean(pl_clamp_triggered_l)) if pl_clamp_triggered_l else 0.0,
                    "rl/vf_clamp_triggered": float(np.mean(vf_clamp_triggered_l)) if vf_clamp_triggered_l else 0.0,
                    "rl/approx_kl": float(np.mean(approx_kl_l)) if approx_kl_l else 0.0,
                    "rl/entropy_linear": float(np.mean(ent_lin_l)) if ent_lin_l else 0.0,
                    "rl/entropy_angular": float(np.mean(ent_ang_l)) if ent_ang_l else 0.0,
                    "rl/total_loss": float(np.mean(total_loss_l)) if total_loss_l else 0.0,
                    "rl/returns_mean": _ret_mean,
                    "rl/returns_std": _ret_std,
                    "rl/returns_min": _ret_min,
                    "rl/returns_max": _ret_max,
                    "rl/value_target_mean": _target_mean,
                    "rl/value_target_std": _target_std,
                    "rl/value_target_min": _target_min,
                    "rl/value_target_max": _target_max,
                    "rl/value_pred_mean": _value_pred_mean,
                    "rl/value_pred_std": _value_pred_std,
                    "rl/value_pred_min": _value_pred_min,
                    "rl/value_pred_max": _value_pred_max,
                    "rl/value_error_mean": _value_error_mean,
                    "rl/value_error_std": _value_error_std,
                    "rl/value_error_abs_max": _value_error_abs_max,
                    "rl/advantage_mean": _adv_mean,
                    "rl/advantage_std": _adv_std,
                    "rl/advantage_min": _adv_min,
                    "rl/advantage_max": _adv_max,
                    "rl/raw_advantage_std": _raw_adv_std,
                }
                if ppo_clip_frac_l:
                    wd_update_monitor["rl/clip_fraction"] = float(np.mean(ppo_clip_frac_l))
                    wd_update_monitor["rl/ratio_mean"] = float(np.mean(ppo_ratio_l))

                # ============================================================
                # Auxiliary loss (WD module loss)
                # ============================================================
                aux_per_feature = {}
                aux_monitor = {}
                if use_extractor: extractor.train()
                preprocess_rnn.train()

                _mon_modules = {
                    "rnn": list(preprocess_rnn.rnn.parameters()),
                    "fc_front": list(preprocess_rnn.fc_front.parameters()),
                    "fc_middle": list(preprocess_rnn.fc_middle.parameters()),
                    "predict_head": list(preprocess_rnn.predict_head.parameters()),
                }
                if use_extractor:
                    _mon_modules["extractor"] = list(extractor.parameters())
                _grad_norms = {k: 0.0 for k in _mon_modules}
                _delta_norms = {k: 0.0 for k in _mon_modules}
                _aux_valid_seq_count = 0
                _aux_actual_batch = 0

                if args_cli.aux_mode == "tbptt":
                    _seq_len = args_cli.aux_seq_len
                    _burn_in = args_cli.aux_burn_in
                    _seq_bs = args_cli.aux_seq_batch_size

                    sampled = charge_buf.sample_aux_sequences(_seq_len, _seq_bs, _burn_in)
                    if sampled is not None:
                        obs_seq, target_seq, h0, _aux_valid_seq_count = sampled
                        B_seq, L_seq = obs_seq.shape[0], obs_seq.shape[1]
                        _aux_actual_batch = B_seq

                        obs_flat_aux = obs_seq.reshape(B_seq * L_seq, -1)
                        obs_normed_aux = obs_normalizer.normalize(obs_flat_aux)

                        if use_extractor:
                            feat_flat = extractor(obs_normed_aux)
                        else:
                            feat_flat = obs_normed_aux[:, _policy_obs_idx]
                        feat_seq = feat_flat.reshape(L_seq, B_seq, -1)

                        _, pred_seq, _ = preprocess_rnn(
                            feat_seq, h0, training=True)

                        effective_start = _burn_in
                        effective_len = L_seq - effective_start
                        pred_eff = pred_seq[effective_start:]
                        tgt_eff = target_seq.permute(1, 0, 2)[effective_start:]

                        total_loss_aux = torch.tensor(0.0, device=device)
                        n_loss_steps = 0
                        last_display = {}
                        for t_idx in range(effective_len):
                            l_t, disp_t = compute_wd_module_loss(
                                pred_eff[t_idx], tgt_eff[t_idx])
                            total_loss_aux = total_loss_aux + l_t
                            n_loss_steps += 1
                            last_display = disp_t
                        total_loss_aux = total_loss_aux / max(n_loss_steps, 1)

                        _snaps = {k: _snapshot_params(ps) for k, ps in _mon_modules.items()}
                        charge_opt_aux.zero_grad()
                        total_loss_aux.backward()
                        nn.utils.clip_grad_norm_(
                            charge_params_aux,
                            args_cli.aux_grad_clip if args_cli.aux_grad_clip is not None else args_cli.max_grad_norm,
                        )
                        for k, ps in _mon_modules.items():
                            _grad_norms[k] = _grad_l2_norm(ps)
                        charge_opt_aux.step()
                        for k, ps in _mon_modules.items():
                            _delta_norms[k] = _param_delta_norm(_snaps[k], ps)

                        aux_loss_val = total_loss_aux.item()
                        for k, v in last_display.items():
                            if k != "preprcess_loss":
                                aux_per_feature[f"charge_{k}"] = v
                        aux_monitor["aux/loss_per_step"] = aux_loss_val / max(effective_len, 1)

                else:
                    # --- Step mode ---
                    valid_t, valid_e = [], []
                    for t in range(RL):
                        nd = (charge_buf.dones[t] < 0.5).nonzero(as_tuple=False).reshape(-1)
                        if len(nd) > 0:
                            valid_t.append(torch.full_like(nd, t)); valid_e.append(nd)
                    _aux_valid_seq_count = sum(len(v) for v in valid_e) if valid_t else 0

                    if valid_t:
                        at = torch.cat(valid_t); ae = torch.cat(valid_e)
                        a_obs = charge_buf.raw_obs[at, ae]
                        a_hid = charge_buf.hiddens[at, ae].unsqueeze(0)
                        a_tgt = charge_buf.aux_targets[at, ae]
                        abs_ = min(4096, len(at))
                        alv = 0.0; nab = 0
                        _aux_actual_batch = len(at)
                        perm = torch.randperm(len(at), device=device)
                        for ab_s in range(0, len(at), abs_):
                            ab_e = min(ab_s + abs_, len(at))
                            ab_i = perm[ab_s:ab_e]
                            mo = obs_normalizer.normalize(a_obs[ab_i])
                            ft = extractor(mo) if use_extractor else mo[:, _policy_obs_idx]
                            _, pred, _ = preprocess_rnn(ft, a_hid[:, ab_i, :], training=True)
                            l, aux_display = compute_wd_module_loss(pred, a_tgt[ab_i])

                            _snaps = {k: _snapshot_params(ps) for k, ps in _mon_modules.items()}
                            charge_opt_aux.zero_grad(); l.backward()
                            nn.utils.clip_grad_norm_(
                                charge_params_aux,
                                args_cli.aux_grad_clip if args_cli.aux_grad_clip is not None else args_cli.max_grad_norm,
                            )
                            for k, ps in _mon_modules.items():
                                _grad_norms[k] += _grad_l2_norm(ps)
                            charge_opt_aux.step()
                            for k, ps in _mon_modules.items():
                                _delta_norms[k] += _param_delta_norm(_snaps[k], ps)

                            alv += l.item(); nab += 1
                            if ab_s + abs_ >= len(at):
                                for k, v in aux_display.items():
                                    if k != "preprcess_loss":
                                        aux_per_feature[f"charge_{k}"] = v
                        aux_loss_val = alv / max(nab, 1)
                        for k in _mon_modules:
                            _grad_norms[k] /= max(nab, 1)
                            _delta_norms[k] /= max(nab, 1)
                        aux_monitor["aux/loss_per_step"] = aux_loss_val

                # --- Common monitoring ---
                for k in _mon_modules:
                    aux_monitor[f"aux/{k}_grad_norm"] = _grad_norms[k]
                    aux_monitor[f"aux/{k}_param_delta_norm"] = _delta_norms[k]
                aux_monitor["aux/rnn_param_norm"] = _param_l2_norm(_mon_modules["rnn"])
                aux_monitor["aux/predict_head_param_norm"] = _param_l2_norm(_mon_modules["predict_head"])

                for raw_key, readable_key in _AUX_DIM_NAMES.items():
                    charge_key = f"charge_{raw_key}"
                    if charge_key in aux_per_feature:
                        aux_monitor[readable_key] = aux_per_feature[charge_key]
                aux_monitor["aux/preprocess_loss"] = aux_loss_val
                aux_monitor["aux/mode"] = 1.0 if args_cli.aux_mode == "tbptt" else 0.0
                aux_monitor["aux/seq_len"] = float(args_cli.aux_seq_len if args_cli.aux_mode == "tbptt" else 1)
                aux_monitor["aux/burn_in"] = float(args_cli.aux_burn_in if args_cli.aux_mode == "tbptt" else 0)
                aux_monitor["aux/valid_seq_count"] = float(_aux_valid_seq_count)
                aux_monitor["aux/seq_batch_size_actual"] = float(_aux_actual_batch)

                with torch.no_grad():
                    rnn_features = charge_buf.rl_inputs[:RL, :, policy_obs_dim:].reshape(-1, args_cli.preprocess_dim)
                    rnn_feature_mean_vec = rnn_features.mean(dim=0)
                    aux_monitor["aux/rnn_feature_mean"] = rnn_features.mean().item()
                    aux_monitor["aux/rnn_feature_std"] = rnn_features.std().item()
                    aux_monitor["aux/rnn_feature_abs_mean"] = rnn_features.abs().mean().item()
                    aux_monitor["aux/rnn_feature_norm_mean"] = rnn_features.norm(dim=1).mean().item()
                    if _prev_rnn_feature_mean is None:
                        aux_monitor["aux/rnn_feature_delta_norm"] = 0.0
                    else:
                        aux_monitor["aux/rnn_feature_delta_norm"] = (
                            rnn_feature_mean_vec - _prev_rnn_feature_mean).norm().item()
                    _prev_rnn_feature_mean = rnn_feature_mean_vec.detach().clone()

                _value_residual = value_targets - charge_buf.values[:RL]
                _var_expl = max(-1.0, 1.0 - (_value_residual.var() / (value_targets.var() + 1e-8)).item())
                aux_monitor["rl/variance_explained"] = _var_expl

                if iteration == 0:
                    _rnn_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in preprocess_rnn.rnn.parameters())
                    _head_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in policy_head.parameters())
                    _ph_lr = charge_opt_aux.param_groups[1]["lr"]
                    _fm_lr = charge_opt_aux.param_groups[2]["lr"]
                    _ff_lr = charge_opt_aux.param_groups[3]["lr"]
                    print(f"[gradient verification] RL head={'Y' if _head_grad else 'N'} (PPO), "
                          f"RNN cell={'Y' if _rnn_grad else 'N'} (aux)")
                    _ext_info = ""
                    if use_extractor:
                        _ext_lr = charge_opt_aux.param_groups[4]["lr"]
                        _ext_info = f" | extractor lr={_ext_lr} {'(frozen)' if _ext_lr == 0 else ''}"
                    else:
                        _ext_info = " | extractor: N/A (raw_fc_rnn)"
                    print(f"  predict_head lr={_ph_lr} | fc_middle lr={_fm_lr} | "
                          f"fc_front lr={_ff_lr} {'(frozen)' if _ff_lr == 0 else ''}"
                          f"{_ext_info}")
                    if args_cli.zero_preprocess_feature_for_rl:
                        print("[ABLATION] --zero_preprocess_feature_for_rl ACTIVE: "
                              "rl_in uses zero instead of 12D preprocess feature")
                    print(f"[AUX] mode={args_cli.aux_mode}, seq_len={args_cli.aux_seq_len}, "
                          f"burn_in={args_cli.aux_burn_in}, seq_batch={args_cli.aux_seq_batch_size}")

            # === Obstacle PPO Update ===
            obs_metrics = None
            if train_obstacle:
                obs_policy.train(); obs_value.train()
                obs_metrics = ppo_update_continuous(
                    obs_policy, obs_value, obs_buf, obs_optimizer,
                    epochs=args_cli.ppo_epochs, mini_batches=args_cli.mini_batches,
                    clip_eps=args_cli.clip_eps, vf_coeff=args_cli.vf_coeff,
                    ent_coeff=args_cli.obs_ent_coeff, max_grad_norm=args_cli.max_grad_norm,
                    gamma=current_gamma, gae_lambda=args_cli.gae_lambda)

            # === Logging ===
            elapsed = time.time() - iter_start
            total_steps = (iteration + 1) * RL
            fps = num_envs * RL / elapsed
            wd = metrics.collect()

            _timeout_rate = wd.get("charge/timeout_rate", 0)

            if (iteration + 1) % args_cli.log_interval == 0 or iteration == 0:
                stage = wd.get("curriculum/stage", 0)
                sr = wd.get("charge/goal_reach_rate", 0)
                cr = wd.get("charge/hit_probability", 0)
                rwd = wd.get("charge/reward_mean", 0)
                goal_v = wd.get("goal_diagnostics/velocity_to_goal_mean", 0)
                goal_d = wd.get("goal_diagnostics/distance_delta_mean", 0)
                goal_h = wd.get("goal_diagnostics/heading_error_abs_mean_deg", 0)
                goal_sw = wd.get("goal_diagnostics/target_switch_rate", 0)
                who = "CHARGE" if train_charge else "OBS"
                obs_tag = f" obs_agent={'ON' if _obs_agent_active else 'OFF'}" if not _obs_agent_active else ""
                if train_charge:
                    print(
                        f"[{iteration+1}/{num_iterations}] {who} "
                        f"S{int(stage)} | fps={fps:.0f} | "
                        f"R={rwd:.1f} SR={sr:.1%} CR={cr:.1%} TO={_timeout_rate:.1%} | "
                        f"ppo={charge_ppo_loss:.4f} vf={charge_vf_loss:.4f} "
                        f"ent={charge_entropy:.3f} | "
                        f"gV={goal_v:+.3f} gD={goal_d:+.2f} h={goal_h:.0f}deg "
                        f"sw={goal_sw:.3f}{obs_tag}")
                else:
                    _obs_ent = obs_metrics["entropy"] if obs_metrics else 0.0
                    _obs_pl = obs_metrics["policy_loss"] if obs_metrics else 0.0
                    print(
                        f"[{iteration+1}/{num_iterations}] {who} "
                        f"S{int(stage)} | fps={fps:.0f} | "
                        f"R={rwd:.1f} SR={sr:.1%} CR={cr:.1%} TO={_timeout_rate:.1%} | "
                        f"obs_ppo={_obs_pl:.4f} obs_ent={_obs_ent:.3f} | "
                        f"gV={goal_v:+.3f} gD={goal_d:+.2f} h={goal_h:.0f}deg "
                        f"sw={goal_sw:.3f}{obs_tag}")
                if train_charge:
                    _rnn_gn = aux_monitor.get("aux/rnn_grad_norm", 0)
                    _rnn_dn = aux_monitor.get("aux/rnn_param_delta_norm", 0)
                    _ph_dn = aux_monitor.get("aux/predict_head_param_delta_norm", 0)
                    _fm_dn = aux_monitor.get("aux/fc_middle_param_delta_norm", 0)
                    _n1d = aux_monitor.get("aux/near1_d_loss", 0)
                    _n2d = aux_monitor.get("aux/near2_d_loss", 0)
                    _vsc = int(aux_monitor.get("aux/valid_seq_count", 0))
                    _asl = int(aux_monitor.get("aux/seq_len", 1))
                    _ve = aux_monitor.get("rl/variance_explained", 0)
                    print(
                        f"  AUX({args_cli.aux_mode} L={_asl}): "
                        f"loss={aux_loss_val:.4f} "
                        f"n1d={_n1d:.3f} n2d={_n2d:.3f} "
                        f"valid={_vsc} | "
                        f"rnn_grad={_rnn_gn:.4f} rnn_delta={_rnn_dn:.6f} "
                        f"ph_delta={_ph_dn:.6f} fm_delta={_fm_dn:.6f} | "
                        f"VE={_ve:.3f}")

            if wandb_run is not None:
                log_data = {}

                log_data.update({
                    "rl/return_mean": wd.get("charge/reward_mean", 0),
                    "rl/success_rate": wd.get("charge/goal_reach_rate", 0),
                    "rl/collision_rate": wd.get("charge/hit_probability", 0),
                    "rl/timeout_rate": _timeout_rate,
                    "rl/stage_idx": float(wd.get("curriculum/stage", 0)),
                })
                if train_charge:
                    log_data.update({
                        "rl/policy_loss": charge_ppo_loss,
                        "rl/value_loss": charge_vf_loss,
                        "rl/entropy": charge_entropy,
                    })

                if obs_metrics is not None:
                    log_data.update({
                        "obstacle/policy_loss": obs_metrics["policy_loss"],
                        "obstacle/value_loss": obs_metrics["value_loss"],
                        "obstacle/entropy": obs_metrics["entropy"],
                        "obstacle/total_loss": obs_metrics["total_loss"],
                    })

                if train_charge:
                    log_data.update(aux_monitor)
                    log_data.update(wd_update_monitor)

                log_data.update({
                    "train/fps": fps,
                    "train/gamma": current_gamma,
                    "train/active_agent": "charge" if train_charge else "obstacle",
                    "train/obs_agent_active": float(_obs_agent_active),
                    "train/zero_preprocess_for_rl": float(args_cli.zero_preprocess_feature_for_rl),
                    "train/ent_coeff_linear": _ent_coeff_linear,
                    "train/ent_coeff_angular": _ent_coeff_angular,
                })

                log_data.update(wd)
                wandb_run.log(log_data, step=total_steps)

            # === Save checkpoint ===
            if (iteration + 1) % args_cli.save_interval == 0 or iteration == num_iterations - 1:
                ckpt_path = os.path.join(log_dir, f"checkpoint_{total_steps}.pt")
                save_checkpoint(
                    ckpt_path,
                    extractor=extractor, preprocess_rnn=preprocess_rnn,
                    policy_head=policy_head, value_head=value_head,
                    obs_policy=obs_policy, obs_value=obs_value,
                    charge_opt_rl=charge_opt_rl, charge_opt_aux=charge_opt_aux,
                    obs_optimizer=obs_optimizer, obs_normalizer=obs_normalizer,
                    iteration=iteration, total_steps=total_steps,
                    args_cli=args_cli, use_extractor=use_extractor,
                )
                print(f"[SAVE] {ckpt_path}")

        # === Finish ===
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete: {total_timesteps:,} steps in {total_time:.0f}s ({total_time/60:.1f}min)")
        print(f"{'='*60}")
        if wandb_run is not None:
            import wandb
            wandb.finish()
        env.close()

    main()
