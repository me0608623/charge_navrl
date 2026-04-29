"""CLI argument definitions -- extracted from train_rnn_car_wdclip.py (lines 49-201)."""

import argparse


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser. Caller adds AppLauncher args afterwards."""

    parser = argparse.ArgumentParser(
        description="Train Charge + Obstacle with Modular RNN (Warp Drive port)")

    # --- Isaac Lab 標準 ---
    parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1)

    # --- Charge 訓練超參 ---
    parser.add_argument("--timesteps", type=int, default=262_144,
                        help="Total timesteps. 262144 = 1024 updates x 256 rollout")
    parser.add_argument("--rollout_length", type=int, default=256, help="Steps per rollout")
    parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO learning epochs")
    parser.add_argument("--mini_batches", type=int, default=32, help="Mini-batches per epoch")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Charge RL head LR. WD: spot_lr=0.0002")
    parser.add_argument("--rnn_lr", type=float, default=5e-4,
                        help="Charge RNN module LR. WD: spot_rnn_model_lr=0.0005")
    parser.add_argument("--aux_lr", type=float, default=0.0,
                        help="(Legacy fallback) Preprocess FC + extractor LR in aux optimizer. "
                             "Overridden by fine-grained --aux_lr_* args when they differ from default. "
                             "WD: spot_preprocess_model_lr=0 (frozen). RNN cell uses --rnn_lr.")
    parser.add_argument("--aux_lr_predict_head", type=float, default=0.0,
                        help="Aux optimizer LR for predict_head. WD baseline: 0 (frozen). "
                             "Unfreeze experiment: 1e-4.")
    parser.add_argument("--aux_lr_fc_middle", type=float, default=0.0,
                        help="Aux optimizer LR for fc_middle (post-RNN FC). WD baseline: 0 (frozen). "
                             "Unfreeze experiment: 5e-5.")
    parser.add_argument("--aux_lr_fc_front", type=float, default=0.0,
                        help="Aux optimizer LR for fc_front (pre-RNN FC). Frozen: changing RNN "
                             "input distribution risks destabilizing RNN learning.")
    parser.add_argument("--aux_lr_extractor", type=float, default=0.0,
                        help="Aux optimizer LR for extractor (Conv1d+MLP). Frozen: isolate "
                             "'output mapping' fix from 'input feature' changes.")
    parser.add_argument("--gamma", type=float, default=0.984,
                        help="Discount factor. WD: 1-(1-0.92)/fps = 0.984 (fps=5)")
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coeff", type=float, default=0.025,
                        help="Value loss coefficient. WD: spot_vf_loss_coeff=0.025")
    parser.add_argument("--normalize_return", "--normalize_returns", dest="normalize_return",
                        action="store_true", default=False,
                        help="Normalize critic value targets per rollout before value MSE. "
                             "This keeps vf_coeff unchanged but tests whether raw return scale "
                             "is driving critic loss/gradient spikes.")
    parser.add_argument("--ent_coeff", type=float, default=0.0,
                        help="Legacy single entropy coeff. 0=use per-head WD coeffs from curriculum")
    parser.add_argument("--ent_coeff_linear", type=float, default=0.0,
                        help="Head1 (linear accel) entropy coeff. 0=auto from curriculum. "
                             "WD: 0.10 (Phase 1), 0.30 (Phase 2+)")
    parser.add_argument("--ent_coeff_angular", type=float, default=0.0,
                        help="Head2 (angular vel) entropy coeff. 0=auto from curriculum. "
                             "WD: 0.375 (all phases)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--aux_grad_clip", type=float, default=None,
                        help="Optional gradient clip just for aux/RNN updates. None uses --max_grad_norm. "
                             "Use values like 0.5 to suppress RNN spikes without changing RL clipping.")
    parser.add_argument("--wd_update_clip", dest="wd_update_clip", action="store_true",
                        help="Enable WD-style actor/critic separated gradient caps. "
                             "When enabled, actor and critic grads are clipped independently "
                             "to --wd_actor_update_clip / --wd_critic_update_clip, avoiding "
                             "critic spikes suppressing actor updates through merged grad clipping.")
    parser.add_argument("--no_wd_update_clip", dest="wd_update_clip", action="store_false",
                        help="Disable WD-style actor/critic gradient scaling while keeping the experiment entrypoint.")
    parser.set_defaults(wd_update_clip=False)
    parser.add_argument("--wd_update_monitor_only", action="store_true", default=False,
                        help="Deprecated alias for the default behavior: log WD-style update metrics but do not scale gradients.")
    parser.add_argument("--wd_actor_update_clip", type=float, default=8.0,
                        help="WD-style policy gradient cap k. Applied directly to actor grad norm.")
    parser.add_argument("--wd_critic_update_clip", type=float, default=30.0,
                        help="WD-style critic gradient cap q. Applied directly to critic grad norm.")
    parser.add_argument("--wd_module_entropy_eps", type=float, default=1e-12,
                        help="Numerical epsilon for module_entropy = log10(actor_update) - log10(critic_update).")
    parser.add_argument("--use_a2c", action="store_true", default=False,
                        help="Use A2C (no PPO clipping). WD: A2CK mode")
    parser.add_argument("--tbptt_len", type=int, default=0,
                        help="TBPTT sequence length. 0=disabled (step-by-step). WD: 15")
    parser.add_argument("--lr_decay", type=float, default=0.0,
                        help="Linear LR decay factor per iteration. 0=no decay. "
                             "WD: uses ParamScheduler")

    # --- Modular RNN ---
    parser.add_argument("--charge_encoder_mode", type=str, default="extractor_rnn",
                        choices=["extractor_rnn", "raw_fc_rnn"],
                        help="Charge encoder mode. "
                             "extractor_rnn: 79D -> Conv1d+MLP extractor(96D) -> FC -> RNN -> FC -> 12D (current). "
                             "raw_fc_rnn: 79D -> FC -> RNN -> FC -> 12D (closer to WD original).")
    parser.add_argument("--hidden_dim", type=int, default=30,
                        help="RNN hidden state dim. WD: memory_dim=30")
    parser.add_argument("--preprocess_dim", type=int, default=12,
                        help="Preprocess feature dim. WD: module_connect_dim=12")
    parser.add_argument("--fc_dim", type=int, default=48, help="FC front dim before RNN")
    parser.add_argument("--rnn_type", type=str, default="RNN", choices=["RNN", "GRU"],
                        help="RNN type. WD: vanilla RNN")

    # --- Obstacle Policy ---
    parser.add_argument("--obs_lr", type=float, default=3e-4, help="Obstacle policy LR")
    parser.add_argument("--obs_ent_coeff", type=float, default=0.1,
                        help="Obstacle entropy coeff (high = explore more)")
    parser.add_argument("--obs_speed_limit", type=float, default=0.8,
                        help="Obstacle max speed (m/s), relative to charge max 1.0")
    parser.add_argument("--train_goal_rate", type=int, default=3,
                        help="Warp Drive: every N iters, 1 trains obstacle (rest train charge)")
    parser.add_argument("--obs_reward_mode", type=str, default="zero",
                        choices=["zero", "approach"],
                        help="zero=Warp Drive style, approach=weak adversarial")
    parser.add_argument("--max_active_obstacles", type=int, default=10)

    # --- Randomization (Warp Drive: obs_size_rand + floor_width_bias) ---
    parser.add_argument("--obs_size_rand", type=float, default=0.0,
                        help="Obstacle collision radius randomization range (m). "
                             "0=auto from curriculum. WD: 0.1->0.4 across phases")
    parser.add_argument("--obs_collision_base", type=float, default=0.9,
                        help="Base collision distance (m). Randomized +/- obs_size_rand/2")
    parser.add_argument("--scene_bound_rand", type=float, default=0.0,
                        help="Scene bound randomization range (m). "
                             "0=auto from curriculum. Effective bound = base +/- rand/2")
    parser.add_argument("--scene_bound_base", type=float, default=7.0,
                        help="Base obstacle movement boundary (m)")

    # --- Logging ---
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N iterations")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint path")

    # --- Env config overrides ---
    parser.add_argument("--reward_mode", type=str, default="current")
    parser.add_argument("--curriculum_version", type=str, default="warp_drive_goal_first",
                        help="Curriculum version (default: warp_drive_goal_first -- goal->static->dynamic)")
    parser.add_argument("--lidar_no_noise", action="store_true", default=False)
    parser.add_argument("--no_domain_randomization", action="store_true", default=False)
    parser.add_argument("--reward_speed_v05", action="store_true", default=False)
    parser.add_argument("--play", action="store_true", default=False,
                        help="Inference only -- no PPO training, just run rollout with loaded checkpoint")
    parser.add_argument("--initial_stage", type=int, default=1,
                        help="Force curriculum to start at this stage (1-8)")
    parser.add_argument("--fixed_stage", action="store_true", default=False,
                        help="Keep curriculum at initial_stage and disable promote/demote transitions. "
                             "Useful for WD-style fixed phase experiments.")
    parser.add_argument("--zero_preprocess_feature_for_rl", action="store_true", default=False,
                        help="Ablation: replace 12D preprocess feature with zeros in rl_in. "
                             "Aux path trains normally. Tests whether policy uses RNN features.")

    # --- Aux TBPTT ---
    parser.add_argument("--aux_mode", type=str, default="tbptt", choices=["step", "tbptt"],
                        help="Aux training mode. step=cached hidden single-step (old); "
                             "tbptt=truncated BPTT sequence unroll (WD-style)")
    parser.add_argument("--aux_seq_len", type=int, default=15,
                        help="TBPTT sequence length for aux training. WD commonly uses 15")
    parser.add_argument("--aux_burn_in", type=int, default=0,
                        help="Burn-in steps at start of each sequence (update hidden, skip loss)")
    parser.add_argument("--aux_seq_batch_size", type=int, default=256,
                        help="Number of sequences per aux mini-batch")

    return parser
