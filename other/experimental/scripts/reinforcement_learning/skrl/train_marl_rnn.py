"""
MARL RNN Training — DirectMARLChargeEnv + Parameter Sharing + Auxiliary Loss

N 台車共享一個 policy（Parameter Sharing），使用 vanilla RNN 做時序記憶。
完全對齊 WD 原始設計（module_connected.py + custom_trainer.py）。

WD 核心設計：
  - RL loss（A2CK）只更新 policy/value head（extractor/RNN 被 .detach() 隔離）
  - Aux loss（module loss）是 extractor/RNN 的唯一梯度來源
  - 雙 optimizer：opt_rl (RL params lr=lr, module lr=0)
                   opt_module (RL params lr=0, module lr=aux_lr)

架構:
  Obs [E, N, 79] → flatten [E*N, 79]
    → LidarStateExtractor (96D)
    → PreprocessRNN (preprocess_dim, detached for RL path)
    → cat(79D_obs, preprocess_dim)
    → PolicyHead → 38 logits (19 accel + 19 omega)
    → ValueHead → scalar

RNN Hidden: [1, E*N, 30] — per-car respawn 時 reset 該車的 hidden
Auxiliary Loss: regress env-computed preprocess_real_data (WD-style privileged targets)
                — 唯一訓練 RNN 的路徑
"""

# ============================================================================
# 1. Pre-launch imports & CLI
# ============================================================================
import sys
import os
import time
import math
from pathlib import Path

import argparse
parser = argparse.ArgumentParser(description="MARL RNN Training — Parameter Sharing")
# -- Env --
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--num_cars", type=int, default=6)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--headless", action="store_true")
# -- Training --
parser.add_argument("--timesteps", type=int, default=262_144,
                    help="Total env steps (across all cars)")
parser.add_argument("--rollout_length", type=int, default=64)
parser.add_argument("--ppo_epochs", type=int, default=4)
parser.add_argument("--mini_batches", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-4, help="RL head LR")
parser.add_argument("--rnn_lr", type=float, default=5e-4,
                    help="[DEPRECATED, use --aux_lr] RNN + extractor LR")
parser.add_argument("--aux_lr", type=float, default=5e-4,
                    help="Auxiliary preprocess-target LR (WD: RNN/extractor ONLY "
                         "trained by aux loss, never by RL loss)")
parser.add_argument("--gamma", type=float, default=0.984)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_eps", type=float, default=0.2)
parser.add_argument("--vf_coeff", type=float, default=0.025)
parser.add_argument("--ent_coeff_linear", type=float, default=0.10)
parser.add_argument("--ent_coeff_angular", type=float, default=0.375)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--grad-clip-mode", type=str, default="separate",
                    choices=["merged", "separate"],
                    help="Gradient clipping mode for RL head optimizer")
parser.add_argument("--use_a2c", action="store_true", default=False)
parser.add_argument("--entropy_floor", type=float, default=0.3,
                    help="Min entropy threshold; below this, coeff is boosted")
parser.add_argument("--entropy_floor_boost", type=float, default=3.0,
                    help="Multiplier for entropy coeff when H < floor")
# -- Model --
parser.add_argument("--hidden_dim", type=int, default=30)
parser.add_argument("--preprocess_dim", type=int, default=8,
                    help="Auxiliary target dimension (default: 8 WD-style geometry dims)")
parser.add_argument("--fc_dim", type=int, default=48)
parser.add_argument("--rnn_type", type=str, default="RNN", choices=["RNN", "GRU"])
# -- Obstacle agent --
parser.add_argument("--obs_lr", type=float, default=3e-4, help="Obstacle policy LR")
parser.add_argument("--obs_ent_coeff", type=float, default=0.1)
parser.add_argument("--obs_speed_limit", type=float, default=0.8)
parser.add_argument("--train_goal_rate", type=int, default=3,
                    help="Charge trains N-1 iters, obstacle trains 1 iter per N")
parser.add_argument("--obs_reward_mode", type=str, default="move",
                    choices=["zero", "move", "approach"],
                    help="WD default: 'move' — reward for speed+accel, keeps obstacles moving")
# -- Curriculum --
parser.add_argument("--curriculum_version", type=str, default="auto_7phase",
                    help="Curriculum version (auto_7phase, baseline_v1, etc.)")
parser.add_argument("--initial_stage", type=int, default=1)
parser.add_argument("--no_curriculum", action="store_true",
                    help="Disable curriculum (fixed difficulty)")
parser.add_argument("--eval_interval", type=int, default=20,
                    help="Eval every N iterations for phase promotion")
parser.add_argument("--phase_patience", type=int, default=3,
                    help="Number of consecutive evals to average for promotion")
parser.add_argument("--min_iters_per_phase", type=int, default=200)
parser.add_argument("--max_iters_per_phase", type=int, default=18000)
# -- Logging --
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=100)
parser.add_argument("--checkpoint", type=str, default=None)

args_cli, extra = parser.parse_known_args()

# Isaac Sim launch
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args_cli.headless or "--headless" in sys.argv})


# ============================================================================
# 2. Post-launcher imports
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).parent))
from modular_rnn_models import (
    LidarStateExtractor,
    PreprocessRNN,
    PolicyHead,
    ValueHead,
    RNNStateManager,
    ObstaclePolicyFC,
    ObstacleValueFC,
    OBS_POLICY_OBS_DIM,
    NUM_BINS,
    USED_OBS_DIM,
)

from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.direct_marl.charge_marl_env import (
    DirectMARLChargeCfg,
    DirectMARLChargeEnv,
    _build_scene_cfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.direct_marl.marl_curriculum import (
    MARLCurriculum,
)

print(f"[MARL-RNN] num_envs={args_cli.num_envs}, num_cars={args_cli.num_cars}, "
      f"rnn_type={args_cli.rnn_type}, hidden_dim={args_cli.hidden_dim}, "
      f"grad_clip_mode={args_cli.grad_clip_mode}")


# ============================================================================
# Constants
# ============================================================================
# Obs layout: ego(4) + goal(2) + lidar(72) + time(1) = 79D  [WD-aligned]
# All 79D used directly by policy (no index remapping needed)
OBS_DIM = 79
PREPROCESS_WEIGHTS = (1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 0.5, 0.5)


# ============================================================================
# Rollout Buffer (for E*N flattened agents)
# ============================================================================
class RolloutBuffer:
    """Rollout buffer for parameter-shared MARL PPO.

    All tensors have shape [T, E*N, ...] where T=rollout_length, E*N=total agents.
    """

    def __init__(self, num_steps: int, num_agents: int, rl_input_dim: int,
                 obs_dim: int, hidden_dim: int, preprocess_dim: int,
                 device: torch.device):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.device = device

        self.rl_inputs = torch.zeros(num_steps, num_agents, rl_input_dim, device=device)
        self.actions = torch.zeros(num_steps, num_agents, 2, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(num_steps, num_agents, device=device)
        self.rewards = torch.zeros(num_steps, num_agents, device=device)
        self.values = torch.zeros(num_steps, num_agents, device=device)
        self.dones = torch.zeros(num_steps, num_agents, device=device)
        self.raw_obs = torch.zeros(num_steps, num_agents, obs_dim, device=device)
        self.hiddens = torch.zeros(num_steps, num_agents, hidden_dim, device=device)
        self.preprocess_targets = torch.zeros(
            num_steps, num_agents, preprocess_dim, device=device,
        )
        self.ptr = 0

    def add(self, rl_input, action, log_prob, reward, value, done, raw_obs, hidden,
            preprocess_target):
        i = self.ptr
        self.rl_inputs[i] = rl_input
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.raw_obs[i] = raw_obs
        self.hiddens[i] = hidden.squeeze(0)
        self.preprocess_targets[i] = preprocess_target
        self.ptr += 1

    def reset(self):
        self.ptr = 0


# ============================================================================
# Obstacle Rollout Buffer (flat: all obstacles as independent samples)
# ============================================================================
class ObstacleRolloutBuffer:
    def __init__(self, num_steps: int, num_agents: int, obs_dim: int,
                 act_dim: int, device: torch.device):
        self.num_steps = num_steps
        self.B = num_agents
        self.device = device
        self.obs = torch.zeros(num_steps, num_agents, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_agents, act_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_agents, device=device)
        self.rewards = torch.zeros(num_steps, num_agents, device=device)
        self.values = torch.zeros(num_steps, num_agents, device=device)
        self.dones = torch.zeros(num_steps, num_agents, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def reset(self):
        self.ptr = 0


def ppo_update_obstacle(policy, value_fn, buf, optimizer, epochs, mini_batches,
                        clip_eps, vf_coeff, ent_coeff, max_grad_norm, gamma, gae_lambda):
    """PPO update for continuous-action obstacle policy."""
    T = buf.ptr
    B = buf.B
    if T == 0 or B == 0:
        return 0.0

    # GAE
    with torch.no_grad():
        last_val = value_fn(buf.obs[T - 1]).squeeze(-1)
    advantages, returns = compute_gae(
        buf.rewards[:T], buf.values[:T], buf.dones[:T], last_val, gamma, gae_lambda,
    )
    # Flatten
    flat_obs = buf.obs[:T].reshape(T * B, -1)
    flat_act = buf.actions[:T].reshape(T * B, -1)
    flat_lp = buf.log_probs[:T].reshape(T * B)
    flat_adv = advantages.reshape(T * B)
    flat_ret = returns.reshape(T * B)

    flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
    batch_size = T * B
    mb_size = max(1, batch_size // mini_batches)

    losses = []
    for _ in range(epochs):
        idx = torch.randperm(batch_size, device=buf.device)
        for start in range(0, batch_size, mb_size):
            end = min(start + mb_size, batch_size)
            mb = idx[start:end]
            new_lp, entropy = policy.evaluate(flat_obs[mb], flat_act[mb])
            new_val = value_fn(flat_obs[mb]).squeeze(-1)

            ratio = (new_lp - flat_lp[mb]).exp()
            s1 = ratio * flat_adv[mb]
            s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * flat_adv[mb]
            pl = -torch.min(s1, s2).mean()
            vl = F.mse_loss(new_val, flat_ret[mb])
            el = entropy.mean()

            loss = pl + vf_coeff * vl - ent_coeff * el
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(value_fn.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(pl.item())

    return sum(losses) / max(len(losses), 1)


# ============================================================================
# Running Normalizer (Welford's online)
# ============================================================================
class RunningNormalizer:
    def __init__(self, shape, device, clip=5.0):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = 1e-4
        self.clip = clip

    @torch.no_grad()
    def update(self, x):
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        return torch.clamp(
            (x - self.mean) / (self.var.sqrt() + 1e-8), -self.clip, self.clip,
        )


# ============================================================================
# Action utilities
# ============================================================================
def sample_action(logits: torch.Tensor):
    """Sample from 2-head Categorical. Returns (actions [B,2], log_prob [B], entropy [B])."""
    logits_a = logits[:, :NUM_BINS]
    logits_w = logits[:, NUM_BINS:]
    dist_a = Categorical(logits=logits_a)
    dist_w = Categorical(logits=logits_w)
    action_a = dist_a.sample()
    action_w = dist_w.sample()
    log_prob = dist_a.log_prob(action_a) + dist_w.log_prob(action_w)
    entropy = dist_a.entropy() + dist_w.entropy()
    return torch.stack([action_a, action_w], dim=-1), log_prob, entropy


def evaluate_actions(logits: torch.Tensor, actions: torch.Tensor):
    """Evaluate log_prob and per-head entropy for given actions."""
    logits_a = logits[:, :NUM_BINS]
    logits_w = logits[:, NUM_BINS:]
    dist_a = Categorical(logits=logits_a)
    dist_w = Categorical(logits=logits_w)
    log_prob = dist_a.log_prob(actions[:, 0]) + dist_w.log_prob(actions[:, 1])
    return log_prob, dist_a.entropy(), dist_w.entropy()


# ============================================================================
# GAE
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


# ============================================================================
# MARL ↔ Flat conversion
# ============================================================================
def dict_to_flat(obs_dict: dict[str, torch.Tensor], n_active: int) -> torch.Tensor:
    """Stack ACTIVE agent obs → [E*n_active, obs_dim]."""
    stacked = torch.stack([obs_dict[f"car_{i}"] for i in range(n_active)], dim=1)
    E, Na, D = stacked.shape
    return stacked.reshape(E * Na, D)


def flat_to_dict(flat: torch.Tensor, num_envs: int, n_active: int,
                 num_cars_total: int) -> dict[str, torch.Tensor]:
    """[E*n_active, D] → dict for ALL cars (inactive get zeros)."""
    D = flat.shape[-1]
    reshaped = flat.reshape(num_envs, n_active, D)
    result = {}
    for i in range(num_cars_total):
        if i < n_active:
            result[f"car_{i}"] = reshaped[:, i]
        else:
            result[f"car_{i}"] = torch.zeros(num_envs, D, device=flat.device)
    return result


def rewards_to_flat(reward_dict: dict[str, torch.Tensor], n_active: int) -> torch.Tensor:
    """Stack ACTIVE reward dicts → [E*n_active]."""
    return torch.stack([reward_dict[f"car_{i}"] for i in range(n_active)], dim=1).reshape(-1)


def dones_to_flat(done_dict: dict[str, torch.Tensor], n_active: int) -> torch.Tensor:
    """Stack ACTIVE done dicts → [E*n_active]."""
    return torch.stack([done_dict[f"car_{i}"] for i in range(n_active)], dim=1).reshape(-1).float()


# ============================================================================
# Main Training
# ============================================================================
def main():
    torch.manual_seed(args_cli.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = args_cli.num_cars
    # Auto num_envs: WD uses ~1200 total agents (envs × cars_per_env)
    # GPU budget ~768 agents for RTX 5090 (6 LiDARs + 100 obs slots per env)
    MAX_TOTAL_AGENTS = 768
    if args_cli.num_envs > 0:
        E = args_cli.num_envs
    else:
        E = max(4, MAX_TOTAL_AGENTS // N)
        print(f"[MARL-RNN] Auto num_envs={E} (budget={MAX_TOTAL_AGENTS}, cars={N})")
    EN = E * N  # total agents (max, when all cars active)

    # ------------------------------------------------------------------
    # 1. Create environment
    # ------------------------------------------------------------------
    cfg = DirectMARLChargeCfg()
    cfg.scene = _build_scene_cfg(num_envs=E, num_cars=N)
    if N != cfg.num_cars:
        cfg.num_cars = N
        cfg.possible_agents = [f"car_{i}" for i in range(N)]
        cfg.observation_spaces = {f"car_{i}": 79 for i in range(N)}
        cfg.action_spaces = {f"car_{i}": 2 for i in range(N)}

    # Enable learned obstacle control (disables scripted waypoint-following)
    cfg.use_learned_obstacles = True
    env = DirectMARLChargeEnv(cfg)
    print(f"[MARL-RNN] Env ready: {E} envs × {N} cars = {EN} agents, learned_obs=True")

    # ------------------------------------------------------------------
    # 1b. Curriculum
    # ------------------------------------------------------------------
    curriculum = None
    if not args_cli.no_curriculum:
        curriculum = MARLCurriculum(
            version=args_cli.curriculum_version,
            initial_stage=args_cli.initial_stage,
            eval_interval=args_cli.eval_interval,
            patience=args_cli.phase_patience,
            min_iters_per_phase=args_cli.min_iters_per_phase,
            max_iters_per_phase=args_cli.max_iters_per_phase,
        )
        curriculum.apply_to_env(env)
        print(f"[MARL-RNN] Curriculum: {args_cli.curriculum_version}, "
              f"initial_stage={args_cli.initial_stage}")

    # Current gamma and active cars (updated by curriculum)
    _gamma = args_cli.gamma
    # n_active = 1 charge + num_virtual_spots (from curriculum phase)
    # Default: all N cars active (no curriculum or first phase)
    n_active = N

    # ------------------------------------------------------------------
    # 2. Create shared models
    # ------------------------------------------------------------------
    extractor = LidarStateExtractor().to(device)
    preprocess_rnn = PreprocessRNN(
        input_dim=96,
        fc_dim=args_cli.fc_dim,
        hidden_dim=args_cli.hidden_dim,
        preprocess_dim=args_cli.preprocess_dim,
        predict_dim=args_cli.preprocess_dim,
        rnn_type=args_cli.rnn_type,
    ).to(device)
    policy_head = PolicyHead(input_dim=USED_OBS_DIM + args_cli.preprocess_dim).to(device)
    value_head = ValueHead(input_dim=USED_OBS_DIM + args_cli.preprocess_dim).to(device)

    rl_input_dim = USED_OBS_DIM + args_cli.preprocess_dim  # 79 + 12 = 91

    # ------------------------------------------------------------------
    # 3. Optimizers (WD-aligned: dual optimizer with LR=0 masks)
    #    - opt_rl: RL loss updates ONLY policy/value head (preprocess/RNN lr=0)
    #    - opt_module: aux loss updates ONLY extractor/RNN (RL params lr=0)
    #    This matches WD custom_trainer.py lines 336-349
    # ------------------------------------------------------------------
    params_rl = list(policy_head.parameters()) + list(value_head.parameters())
    params_module = list(preprocess_rnn.parameters()) + list(extractor.parameters())

    # RL optimizer: only updates policy/value head
    opt_rl = torch.optim.Adam([
        {"params": params_rl, "lr": args_cli.lr},
        {"params": params_module, "lr": 0},  # WD: preprocess/RNN lr=0 in RL opt
    ], eps=1e-5)

    # Module optimizer: only updates extractor/RNN via aux loss
    opt_module = torch.optim.Adam([
        {"params": params_rl, "lr": 0},  # WD: RL params lr=0 in module opt
        {"params": params_module, "lr": args_cli.aux_lr},
    ], eps=1e-5)

    # ------------------------------------------------------------------
    # 4. RNN state manager & normalizer & buffer
    # ------------------------------------------------------------------
    rnn_state = RNNStateManager(EN, args_cli.hidden_dim, device)
    obs_normalizer = RunningNormalizer(OBS_DIM, device)

    buf = RolloutBuffer(
        num_steps=args_cli.rollout_length,
        num_agents=EN,
        rl_input_dim=rl_input_dim,
        obs_dim=OBS_DIM,
        hidden_dim=args_cli.hidden_dim,
        preprocess_dim=args_cli.preprocess_dim,
        device=device,
    )
    aux_weights = torch.tensor(PREPROCESS_WEIGHTS, device=device)
    if aux_weights.numel() != args_cli.preprocess_dim:
        raise ValueError(
            f"preprocess weights mismatch: weights={aux_weights.numel()}, "
            f"preprocess_dim={args_cli.preprocess_dim}",
        )

    # -- Obstacle agent (learned dynamic obstacle policy) --
    n_obs_active = cfg.num_static_obstacles + cfg.num_dynamic_obstacles
    obs_policy = ObstaclePolicyFC(obs_dim=OBS_POLICY_OBS_DIM, act_dim=2).to(device)
    obs_value_fn = ObstacleValueFC(obs_dim=OBS_POLICY_OBS_DIM).to(device)
    opt_obs = torch.optim.Adam(
        list(obs_policy.parameters()) + list(obs_value_fn.parameters()),
        lr=args_cli.obs_lr, eps=1e-5,
    )
    obs_buf = ObstacleRolloutBuffer(
        num_steps=args_cli.rollout_length,
        num_agents=E * n_obs_active,
        obs_dim=OBS_POLICY_OBS_DIM,
        act_dim=2,
        device=device,
    )
    _obs_speed_limit = args_cli.obs_speed_limit

    # ------------------------------------------------------------------
    # 5. Checkpoint loading
    # ------------------------------------------------------------------
    global_step = 0
    if args_cli.checkpoint:
        ckpt = torch.load(args_cli.checkpoint, map_location=device)
        extractor.load_state_dict(ckpt["extractor"])
        preprocess_rnn.load_state_dict(ckpt["preprocess_rnn"])
        policy_head.load_state_dict(ckpt["policy_head"])
        value_head.load_state_dict(ckpt["value_head"])
        if "obs_normalizer" in ckpt:
            obs_normalizer.mean = ckpt["obs_normalizer"]["mean"]
            obs_normalizer.var = ckpt["obs_normalizer"]["var"]
            obs_normalizer.count = ckpt["obs_normalizer"]["count"]
        global_step = ckpt.get("global_step", 0)
        print(f"[MARL-RNN] Loaded checkpoint: step={global_step}")

    if hasattr(env, "preprocess_target_dim") and env.preprocess_target_dim != args_cli.preprocess_dim:
        raise ValueError(
            f"preprocess_dim mismatch: script={args_cli.preprocess_dim}, "
            f"env={env.preprocess_target_dim}",
        )

    # ------------------------------------------------------------------
    # 6. Logging setup
    # ------------------------------------------------------------------
    run_name = args_cli.run_name or f"marl_rnn_N{N}_E{E}_s{args_cli.seed}"
    log_dir = Path(f"logs/marl_rnn/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save command
    with open(log_dir / "command.txt", "w") as f:
        f.write(" ".join(sys.argv) + "\n")

    # CSV header
    csv_path = log_dir / "metrics.csv"
    if not csv_path.exists():
        with open(csv_path, "w") as f:
            f.write("step,iter,total_reward,policy_loss,value_loss,entropy,"
                    "goals_per_ep,collisions_per_ep,car_collisions_per_ep,"
                    "grad_norm,grad_norm_policy,grad_norm_value,fps,stage,success_rate\n")

    print(f"[MARL-RNN] Logging to {log_dir}")

    # WandB init (headless only)
    wandb_run = None
    if args_cli.headless:
        try:
            import wandb
            wandb.init(
                project="charge_skrl", name=run_name,
                config={
                    "agent": "DirectMARL-RNN-PPO",
                    "num_envs": E, "num_cars": N, "seed": args_cli.seed,
                    "lr": args_cli.lr, "rnn_lr": args_cli.rnn_lr,
                    "aux_lr": args_cli.aux_lr, "vf_coeff": args_cli.vf_coeff,
                    "gamma": args_cli.gamma, "use_a2c": args_cli.use_a2c,
                    "rnn_type": args_cli.rnn_type, "hidden_dim": args_cli.hidden_dim,
                    "preprocess_dim": args_cli.preprocess_dim,
                    "rollout_length": args_cli.rollout_length,
                    "ppo_epochs": args_cli.ppo_epochs,
                    "obs_lr": args_cli.obs_lr, "obs_ent_coeff": args_cli.obs_ent_coeff,
                    "obs_speed_limit": args_cli.obs_speed_limit,
                    "train_goal_rate": args_cli.train_goal_rate,
                    "curriculum_version": args_cli.curriculum_version,
                },
                tags=["directmarl", "rnn", "wd-aligned"],
            )
            wandb_run = wandb.run
            print(f"[WandB] Run: {wandb.run.name}")
        except Exception as e:
            print(f"[WandB] Init failed: {e}")

    # ------------------------------------------------------------------
    # 7. Training loop
    # ------------------------------------------------------------------
    obs_dict, info = env.reset()
    total_timesteps = args_cli.timesteps
    steps_per_iter = args_cli.rollout_length * EN  # agent-steps per iteration
    # WD uses num_iterations = timesteps // RL (timesteps = rollout steps, not agent-steps)
    # We follow the same convention: timesteps counts rollout steps
    num_iters = max(1, total_timesteps // args_cli.rollout_length)

    print(f"[MARL-RNN] timesteps={total_timesteps}, rollout={args_cli.rollout_length}, "
          f"num_iters={num_iters}, agents_per_iter={EN}", flush=True)
    print(f"[MARL-RNN] Training: {num_iters} iters × {args_cli.rollout_length} steps "
          f"× {EN} agents = {num_iters * steps_per_iter} total agent-steps", flush=True)

    for iteration in range(1, num_iters + 1):
        iter_start = time.time()
        buf.reset()

        # ============================================================
        # Rollout collection
        # ============================================================
        extractor.eval()
        preprocess_rnn.eval()
        policy_head.eval()
        value_head.eval()

        obs_buf.reset()
        # Alternating training: charge trains most iters, obstacle trains every N-th
        train_charge = (iteration % args_cli.train_goal_rate != 1)
        train_obstacle = not train_charge

        # Recreate charge buffer sized for current n_active
        EA = E * n_active
        buf = RolloutBuffer(
            num_steps=args_cli.rollout_length,
            num_agents=EA,
            rl_input_dim=rl_input_dim,
            obs_dim=OBS_DIM,
            hidden_dim=args_cli.hidden_dim,
            preprocess_dim=args_cli.preprocess_dim,
            device=device,
        )

        ep_rewards = torch.zeros(EA, device=device)

        EA = E * n_active  # active agents this iteration

        for step in range(args_cli.rollout_length):
            with torch.no_grad():
                # --- Flatten obs: dict → [E*n_active, 79] ---
                obs_flat = dict_to_flat(obs_dict, n_active)

                # NaN debug (first 3 steps only)
                if step < 3:
                    n_nan = obs_flat.isnan().sum().item()
                    n_inf = obs_flat.isinf().sum().item()
                    if n_nan > 0 or n_inf > 0:
                        print(f"[NaN DEBUG] step={step} obs_flat: {n_nan} NaN, {n_inf} Inf, "
                              f"shape={obs_flat.shape}", flush=True)
                        # Which dimensions have NaN?
                        nan_dims = obs_flat.isnan().any(dim=0).nonzero(as_tuple=False).squeeze(-1).tolist()
                        print(f"[NaN DEBUG]   NaN dims: {nan_dims}", flush=True)

                obs_flat = torch.nan_to_num(obs_flat, nan=0.0, posinf=0.0, neginf=0.0)

                # --- Normalize ---
                obs_normalizer.update(obs_flat)
                obs_normed = obs_normalizer.normalize(obs_flat)
                obs_normed = torch.nan_to_num(obs_normed, nan=0.0, posinf=5.0, neginf=-5.0)
                preprocess_targets = env.build_preprocess_targets(n_active).reshape(EA, -1)
                preprocess_targets = torch.nan_to_num(
                    preprocess_targets, nan=0.0, posinf=20.0, neginf=-20.0,
                )

                # --- Feature extraction ---
                features = extractor(obs_normed)
                features = torch.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)

                # --- RNN forward (only active agents) ---
                hidden = rnn_state.get()[:, :EA, :]
                # WD: RNN output always detached for RL path
                rnn_feat, _, new_hidden = preprocess_rnn(
                        features, hidden, detach_output=True)
                rnn_feat = torch.nan_to_num(rnn_feat, nan=0.0, posinf=5.0, neginf=-5.0)
                new_hidden = torch.nan_to_num(new_hidden, nan=0.0, posinf=5.0, neginf=-5.0)
                full_hidden = rnn_state.get().clone()
                full_hidden[:, :EA, :] = new_hidden.detach()
                rnn_state.hidden = full_hidden

                # --- Compose RL input ---
                p_obs = obs_normed
                rl_input = torch.cat([p_obs, rnn_feat], dim=-1)
                rl_input = torch.nan_to_num(rl_input, nan=0.0, posinf=5.0, neginf=-5.0)

                # --- Policy & value ---
                logits = policy_head(rl_input)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                value = value_head(rl_input).squeeze(-1)
                actions_flat, log_prob, _ = sample_action(logits)

            # --- Convert actions to dict for env ---
            action_dict = flat_to_dict(actions_flat.float(), E, n_active, N)

            # --- Obstacle agent forward + apply ---
            with torch.no_grad():
                o_obs = env.build_obstacle_obs(n_obs_active)  # [E, N_obs, 9]
                o_flat = o_obs.reshape(E * n_obs_active, OBS_POLICY_OBS_DIM)
                o_flat = torch.nan_to_num(o_flat, nan=0.0, posinf=0.0, neginf=0.0)
                o_act, o_lp, _ = obs_policy.sample(o_flat)
                o_val = obs_value_fn(o_flat).squeeze(-1)
            env.apply_obstacle_actions(
                o_act.reshape(E, n_obs_active, 2), n_obs_active,
                speed_limit=_obs_speed_limit,
            )

            # --- Step env ---
            next_obs_dict, reward_dict, term_dict, trunc_dict, info = env.step(action_dict)

            # --- Obstacle buffer ---
            # All obstacles share the episode timeout (from any car_0's trunc)
            o_done_per_env = trunc_dict["car_0"].float()  # [E]
            o_done = o_done_per_env.unsqueeze(1).expand(
                -1, n_obs_active,
            ).reshape(-1)  # [E*N_obs]
            if args_cli.obs_reward_mode == "zero":
                o_rew = torch.zeros(E * n_obs_active, device=device)
            elif args_cli.obs_reward_mode == "move":
                # WD original: reward = 0.3 * (norm_speed + 0.3 * norm_accel)
                # Vectorized: no Python loop over obstacles
                O = n_obs_active
                all_vel = env._obstacle_velocities[:, :O]        # [E, O, 2]
                all_speed = all_vel.norm(dim=2)                   # [E, O]
                norm_speed = all_speed / _obs_speed_limit         # [E, O]
                o_act_2d = o_act.reshape(O, E, 2)                 # [O, E, 2]
                norm_accel = o_act_2d.abs().mean(dim=2).t()       # [E, O]
                o_rew_2d = 0.3 * (norm_speed + 0.3 * norm_accel)  # [E, O]
                o_rew = o_rew_2d.t().reshape(-1)                  # [E*O]
            else:
                # "approach": reward for being near any car — vectorized
                O = n_obs_active
                all_car = torch.stack(
                    [r.data.root_pos_w[:, :2] for r in env.robots], dim=1,
                )                                                  # [E, N_car, 2]
                all_obs_pos = torch.stack(
                    [env._obstacle_entities[oi].data.root_pos_w[:, :2]
                     for oi in range(O)], dim=1,
                )                                                  # [E, O, 2]
                d = torch.norm(
                    all_car.unsqueeze(1) - all_obs_pos.unsqueeze(2), dim=3,
                )                                                  # [E, O, N_car]
                d_min = d.min(dim=2).values                        # [E, O]
                o_rew_2d = 0.1 * (2.0 - d_min).clamp(0, 2) / 2.0  # [E, O]
                o_rew = o_rew_2d.t().reshape(-1)                   # [E*O]
            obs_buf.add(o_flat, o_act, o_lp, o_rew, o_val, o_done)

            # --- Flatten rewards & dones ---
            rewards_flat = rewards_to_flat(reward_dict, n_active)  # [E*n_active]
            # WarpDrive-style: terminated is always False, only trunc (timeout) matters
            dones_flat = dones_to_flat(trunc_dict, n_active)  # [E*n_active]

            # --- Store in buffer ---
            buf.add(rl_input, actions_flat, log_prob, rewards_flat, value,
                    dones_flat, obs_flat, hidden, preprocess_targets)

            # --- RNN hidden reset for respawned cars ---
            if "respawn_mask" in info:
                # info["respawn_mask"]: [E, N_total] bool — only use active cars
                respawn_mask = info["respawn_mask"][:, :n_active]  # [E, n_active]
                respawn_flat = respawn_mask.reshape(-1)  # [E*n_active]
                respawn_ids = respawn_flat.nonzero(as_tuple=False).squeeze(-1)
                rnn_state.reset(respawn_ids)

            # --- RNN hidden reset for timed-out episodes ---
            timeout_flat = dones_flat > 0.5
            if timeout_flat.any():
                timeout_ids = timeout_flat.nonzero(as_tuple=False).squeeze(-1)
                rnn_state.reset(timeout_ids)

            # --- Track episode stats ---
            ep_rewards += rewards_flat

            obs_dict = next_obs_dict
            global_step += EA

        # ============================================================
        # Compute last value for GAE
        # ============================================================
        with torch.no_grad():
            obs_flat = dict_to_flat(obs_dict, n_active)
            obs_flat = torch.nan_to_num(obs_flat, nan=0.0, posinf=0.0, neginf=0.0)
            obs_normed = obs_normalizer.normalize(obs_flat)
            obs_normed = torch.nan_to_num(obs_normed, nan=0.0, posinf=5.0, neginf=-5.0)
            features = extractor(obs_normed)
            features = torch.nan_to_num(features, nan=0.0)
            hidden = rnn_state.get()[:, :EA, :]
            rnn_feat, _, _ = preprocess_rnn(
                features, hidden, detach_output=True)
            rnn_feat = torch.nan_to_num(rnn_feat, nan=0.0)
            p_obs = obs_normed
            rl_input = torch.cat([p_obs, rnn_feat], dim=-1)
            last_value = value_head(rl_input).squeeze(-1)
            last_value = torch.nan_to_num(last_value, nan=0.0)

        # ============================================================
        # GAE & PPO Update
        # ============================================================
        # NaN debug: check buffer before GAE
        _r = buf.rewards[:buf.ptr]
        _v = buf.values[:buf.ptr]
        _d = buf.dones[:buf.ptr]
        print(f"[NaN DEBUG GAE] rewards: nan={_r.isnan().sum().item()}, "
              f"values: nan={_v.isnan().sum().item()}, "
              f"dones: nan={_d.isnan().sum().item()}, "
              f"last_value: nan={last_value.isnan().sum().item()}, "
              f"rewards range=[{_r.min():.2f}, {_r.max():.2f}], "
              f"values range=[{_v.min():.2f}, {_v.max():.2f}]", flush=True)
        # Clamp NaN in buffer
        buf.rewards[:buf.ptr] = torch.nan_to_num(_r, nan=0.0)
        buf.values[:buf.ptr] = torch.nan_to_num(_v, nan=0.0)
        buf.dones[:buf.ptr] = torch.nan_to_num(_d, nan=0.0)
        last_value = torch.nan_to_num(last_value, nan=0.0)

        advantages, returns = compute_gae(
            buf.rewards[:buf.ptr], buf.values[:buf.ptr], buf.dones[:buf.ptr],
            last_value, _gamma, args_cli.gae_lambda,
        )

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Flatten time × active agents → single batch
        T = buf.ptr
        flat_rl = buf.rl_inputs[:T].reshape(T * EA, -1)  # detached from rollout
        flat_act = buf.actions[:T].reshape(T * EA, 2)
        flat_lp = buf.log_probs[:T].reshape(T * EA)
        flat_adv = advantages.reshape(T * EA)
        flat_ret = returns.reshape(T * EA)
        batch_size = T * EA

        # NaN debug: check buffer data before PPO
        _nan_rl = flat_rl.isnan().sum().item()
        _nan_adv = flat_adv.isnan().sum().item()
        _nan_ret = flat_ret.isnan().sum().item()
        if _nan_rl > 0 or _nan_adv > 0 or _nan_ret > 0:
            print(f"[NaN DEBUG PPO] flat_rl: {_nan_rl} NaN, flat_adv: {_nan_adv}, flat_ret: {_nan_ret}, "
                  f"batch_size={batch_size}", flush=True)
            # Which rl_input dims have NaN?
            nan_dims = flat_rl.isnan().any(dim=0).nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"[NaN DEBUG PPO] NaN dims in rl_input: {nan_dims}", flush=True)
            # Clamp to fix
            flat_rl = torch.nan_to_num(flat_rl, nan=0.0, posinf=5.0, neginf=-5.0)
            flat_adv = torch.nan_to_num(flat_adv, nan=0.0)
            flat_ret = torch.nan_to_num(flat_ret, nan=0.0)

        extractor.train()
        preprocess_rnn.train()
        policy_head.train()
        value_head.train()

        n_epochs = 1 if args_cli.use_a2c else args_cli.ppo_epochs
        mb_size = max(1, batch_size // args_cli.mini_batches)

        # GPU accumulators — avoid .item() per mini-batch
        _acc_pl = torch.tensor(0.0, device=device)
        _acc_vl = torch.tensor(0.0, device=device)
        _acc_ent = torch.tensor(0.0, device=device)
        _acc_gn = torch.tensor(0.0, device=device)
        _acc_gn_policy = torch.tensor(0.0, device=device)
        _acc_gn_value = torch.tensor(0.0, device=device)
        n_updates = 0

        for epoch in range(n_epochs):
            idx = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                end = min(start + mb_size, batch_size)
                mb_idx = idx[start:end]

                # WD-aligned: RL loss only updates policy/value head
                # RNN/extractor trained separately by aux loss (module_connected.py:505 detach)
                mb_rl = flat_rl[mb_idx]  # already detached from rollout buffer

                new_logits = policy_head(mb_rl)
                new_lp, ent_lin, ent_ang = evaluate_actions(new_logits, flat_act[mb_idx])
                new_value = value_head(mb_rl).squeeze(-1)

                if args_cli.use_a2c:
                    policy_loss = -(new_lp * flat_adv[mb_idx]).mean()
                else:
                    ratio = (new_lp - flat_lp[mb_idx]).exp()
                    s1 = ratio * flat_adv[mb_idx]
                    s2 = torch.clamp(ratio, 1 - args_cli.clip_eps,
                                     1 + args_cli.clip_eps) * flat_adv[mb_idx]
                    policy_loss = -torch.min(s1, s2).mean()

                value_loss_raw = F.mse_loss(new_value, flat_ret[mb_idx])
                value_loss = value_loss_raw

                # Per-head entropy (WD A2CK) with adaptive floor
                # Use tensor comparison to avoid .item() sync per mini-batch
                _ent_mean_lin = ent_lin.mean()
                _ent_mean_ang = ent_ang.mean()
                _ecl = args_cli.ent_coeff_linear
                _eca = args_cli.ent_coeff_angular
                if args_cli.entropy_floor > 0:
                    _floor_t = torch.tensor(args_cli.entropy_floor, device=device)
                    _boost = args_cli.entropy_floor_boost
                    _ecl = torch.where(_ent_mean_lin < _floor_t, _ecl * _boost, _ecl)
                    _eca = torch.where(_ent_mean_ang < _floor_t, _eca * _boost, _eca)
                entropy_loss = _ecl * _ent_mean_lin + _eca * _ent_mean_ang

                # Policy safety clamp (WD a2c_ken.py:123, rarely triggers with PPO)
                pl_abs = policy_loss.detach().abs()
                policy_loss = torch.where(
                    pl_abs > 20.0,
                    policy_loss * (20.0 / pl_abs.clamp(min=1e-8)),
                    policy_loss,
                )
                # NOTE: WD's critic clamp (q=30) removed — it was designed for A2C
                # where policy_loss ≈ 5-30. With PPO (policy_loss ≈ 0.002), it
                # forces value_loss to 10.0 every step, killing critic learning.
                # Since policy_head and value_head have separate params, no
                # cross-contamination risk — standard PPO vf_coeff is sufficient.

                # WD a2c_ken.py:130-133: first iteration = critic-only warmup
                # (WD skips entire iteration, not just first mini-batch)
                if iteration == 1:
                    loss = 0 * value_loss  # critic warmup: zero RL loss
                else:
                    loss = policy_loss + args_cli.vf_coeff * value_loss - entropy_loss

                opt_rl.zero_grad()
                loss.backward()
                # -- Measure per-head grad norms BEFORE clipping (for A/B comparison) --
                gn_policy_pre = nn.utils.clip_grad_norm_(
                    policy_head.parameters(), float('inf'),  # inf = measure only
                )
                gn_value_pre = nn.utils.clip_grad_norm_(
                    value_head.parameters(), float('inf'),
                )
                # Undo the inf-clip (it didn't change anything, just measured)

                if args_cli.grad_clip_mode == "merged":
                    grad_norm = nn.utils.clip_grad_norm_(
                        list(policy_head.parameters()) + list(value_head.parameters()),
                        args_cli.max_grad_norm,
                    )
                else:
                    gn_policy = nn.utils.clip_grad_norm_(
                        policy_head.parameters(), args_cli.max_grad_norm,
                    )
                    gn_value = nn.utils.clip_grad_norm_(
                        value_head.parameters(), args_cli.max_grad_norm,
                    )
                    grad_norm = gn_policy
                opt_rl.step()

                # Accumulate on GPU — .item() deferred to logging
                _acc_pl += policy_loss.detach()
                _acc_vl += value_loss_raw.detach()  # log raw, not clamped
                _acc_ent += entropy_loss.detach()
                _acc_gn += grad_norm if isinstance(grad_norm, torch.Tensor) else torch.tensor(grad_norm, device=device)
                _acc_gn_policy += gn_policy_pre if isinstance(gn_policy_pre, torch.Tensor) else torch.tensor(gn_policy_pre, device=device)
                _acc_gn_value += gn_value_pre if isinstance(gn_value_pre, torch.Tensor) else torch.tensor(gn_value_pre, device=device)
                n_updates += 1

        # Gradient sanity check (iteration 2: after first real RL update + first aux update)
        if iteration == 2:
            head_gn = sum(p.grad.norm().item() for p in policy_head.parameters() if p.grad is not None)
            print(f"[GRAD CHECK] policy_head={head_gn:.6f} (RL loss only)", flush=True)
            # extractor/RNN grads come from aux loss below, checked separately

        # ============================================================
        # Auxiliary Loss — WD core: ONLY path for RNN/extractor gradients
        # (WD custom_trainer.py:975-997, module_connected.py:505 detach)
        # ============================================================
        aux_loss_val = 0.0
        if T > 0:
            a_obs = buf.raw_obs[:T].reshape(T * EA, OBS_DIM)
            a_hid = buf.hiddens[:T].reshape(T * EA, args_cli.hidden_dim).unsqueeze(0)
            a_tgt = buf.preprocess_targets[:T].reshape(T * EA, args_cli.preprocess_dim)

            # WD: aux loss runs BEFORE RL loss (custom_trainer.py:975-997)
            # We run it after RL since both use separate optimizers.
            n_aux = a_obs.size(0)
            aux_mb = max(1, n_aux // 8)
            aux_idx = torch.randperm(n_aux, device=device)

            for astart in range(0, n_aux, aux_mb):
                aend = min(astart + aux_mb, n_aux)
                bi = aux_idx[astart:aend]

                mo = obs_normalizer.normalize(a_obs[bi])
                ft = extractor(mo)
                _, pred, _ = preprocess_rnn(ft, a_hid[:, bi, :], training=True)
                per_dim_loss = F.smooth_l1_loss(pred, a_tgt[bi], reduction="none")
                aux_loss = (per_dim_loss * aux_weights.unsqueeze(0)).mean()

                opt_module.zero_grad()
                aux_loss.backward()
                nn.utils.clip_grad_norm_(params_module, args_cli.max_grad_norm)
                opt_module.step()
                aux_loss_val = aux_loss.item()

            # Gradient check for module (iteration 2)
            if iteration == 2:
                ext_gn = sum(p.grad.norm().item() for p in extractor.parameters() if p.grad is not None)
                rnn_gn = sum(p.grad.norm().item() for p in preprocess_rnn.parameters() if p.grad is not None)
                print(f"[GRAD CHECK] extractor={ext_gn:.6f}, preprocess_rnn={rnn_gn:.6f} "
                      f"(aux loss only — WD design)", flush=True)

        # ============================================================
        # Obstacle PPO Update (on obstacle training iterations)
        # ============================================================
        obs_loss = 0.0
        if train_obstacle and obs_buf.ptr > 0:
            obs_policy.train()
            obs_value_fn.train()
            obs_loss = ppo_update_obstacle(
                obs_policy, obs_value_fn, obs_buf, opt_obs,
                epochs=args_cli.ppo_epochs,
                mini_batches=args_cli.mini_batches,
                clip_eps=args_cli.clip_eps,
                vf_coeff=args_cli.vf_coeff,
                ent_coeff=args_cli.obs_ent_coeff,
                max_grad_norm=args_cli.max_grad_norm,
                gamma=args_cli.gamma,
                gae_lambda=args_cli.gae_lambda,
            )
            obs_policy.eval()
            obs_value_fn.eval()

        # ============================================================
        # Curriculum Update
        # ============================================================
        cur_stage = 0
        cur_sr = 0.0
        cur_metrics = {}
        if curriculum is not None:
            goals, obs_coll, car_coll, born_died, timeouts = env.get_and_reset_event_counts()
            curriculum.record_events(goals, obs_coll, car_coll, timeouts=timeouts)
            # Pass entropy/value_loss for health checks (auto_7phase)
            _nu_c = max(n_updates, 1)
            _ent_for_cur = (_acc_ent / _nu_c).item() if n_updates > 0 else 1.0
            _vl_for_cur = (_acc_vl / _nu_c).item() if n_updates > 0 else 0.0
            cur_metrics = curriculum.update(
                iteration, entropy=_ent_for_cur, value_loss=_vl_for_cur,
            )
            cur_stage = cur_metrics["stage"]
            cur_sr = cur_metrics["success_rate"]

            if cur_metrics["stage_changed"]:
                curriculum.apply_to_env(env)
                _gamma = cur_metrics["gamma"]
                # Sync num_virtual_spots → n_active_cars
                stage_cfg = curriculum.current_stage_cfg
                n_vs = stage_cfg.get("num_virtual_spots", 0)
                new_active = min(1 + n_vs, N)
                if new_active != n_active:
                    n_active = new_active
                    env.n_active_cars = n_active
                    print(f"[Curriculum] n_active_cars → {n_active} "
                          f"(1 charge + {n_active - 1} virtual spots)")
                # Update obs agent active count from new phase
                new_obs_active = (cur_metrics["num_obstacles_static"]
                                  + cur_metrics["num_obstacles_dynamic"])
                new_obs_active = min(new_obs_active, env._num_obstacle_slots)
                if new_obs_active != n_obs_active:
                    print(f"[Curriculum] n_obs_active → {new_obs_active} "
                          f"(was {n_obs_active})")
                    n_obs_active = new_obs_active
                # Reset optimizer momentum on stage transition (WD custom_trainer.py:1042-1048)
                opt_rl.zero_grad(set_to_none=True)
                opt_module.zero_grad(set_to_none=True)
                for param in opt_rl.state.values():
                    if 'exp_avg' in param: param['exp_avg'].zero_()
                    if 'exp_avg_sq' in param: param['exp_avg_sq'].zero_()
                for param in opt_module.state.values():
                    if 'exp_avg' in param: param['exp_avg'].zero_()
                    if 'exp_avg_sq' in param: param['exp_avg_sq'].zero_()
                # Reset RNN hidden state
                rnn_state.hidden.zero_()
                # Rollout buffer will be recreated at top of next iteration
                buf.reset()

        # ============================================================
        # Logging
        # ============================================================
        iter_time = time.time() - iter_start
        fps = steps_per_iter / iter_time

        if iteration % args_cli.log_interval == 0 or iteration == 1:
            # Single .item() batch — one GPU→CPU sync for all 4 accumulators
            _nu = max(n_updates, 1)
            avg_rew = buf.rewards[:T].mean().item()
            avg_pl = (_acc_pl / _nu).item()
            avg_vl = (_acc_vl / _nu).item()
            avg_ent = (_acc_ent / _nu).item()
            avg_gn = (_acc_gn / _nu).item()
            avg_gn_policy = (_acc_gn_policy / _nu).item()
            avg_gn_value = (_acc_gn_value / _nu).item()

            # Per-episode metrics from env extras
            goals_ep = 0.0
            coll_ep = 0.0
            car_coll_ep = 0.0
            for i in range(N):
                agent = f"car_{i}"
                if agent in env.extras and "log" in env.extras[agent]:
                    log = env.extras[agent]["log"]
                    goals_ep += log.get("goals_per_episode", 0.0)
                    coll_ep += log.get("collisions_per_episode", 0.0)
                    car_coll_ep += log.get("car_collisions_per_episode", 0.0)
            goals_ep /= N
            coll_ep /= N
            car_coll_ep /= N

            _pi = cur_metrics.get("phase_iters", 0)
            stage_str = f"  P{cur_stage}({_pi}) sr={cur_sr:.1%}" if curriculum else ""
            print(f"[iter {iteration:5d} | step {global_step:>10d}] "
                  f"rew={avg_rew:+.3f}  pl={avg_pl:.4f}  vl={avg_vl:.4f}  "
                  f"ent={avg_ent:.4f}  gn={avg_gn:.2f}  "
                  f"gn_p={avg_gn_policy:.3f}  gn_v={avg_gn_value:.3f}  "
                  f"aux={aux_loss_val:.4f}  "
                  f"goals/ep={goals_ep:.2f}  coll/ep={coll_ep:.2f}  "
                  f"car_coll/ep={car_coll_ep:.2f}  obs_loss={obs_loss:.4f}  "
                  f"fps={fps:.0f}{stage_str}", flush=True)

            with open(csv_path, "a") as f:
                f.write(f"{global_step},{iteration},{avg_rew:.4f},"
                        f"{avg_pl:.6f},{avg_vl:.6f},{avg_ent:.6f},"
                        f"{goals_ep:.4f},{coll_ep:.4f},{car_coll_ep:.4f},"
                        f"{avg_gn:.4f},{avg_gn_policy:.4f},{avg_gn_value:.4f},"
                        f"{fps:.0f},{cur_stage},{cur_sr:.4f}\n")

            # WandB logging — WD-aligned comprehensive metrics
            if wandb_run is not None:
                log_dict = {
                    # -- Loss (WD key names) --
                    "charge loss": avg_pl,
                    "charge vf_loss": avg_vl,
                    "charge loss coefficient entropy": avg_ent,
                    "charge aux_loss": aux_loss_val,
                    "obstacle loss": obs_loss,
                    # -- Reward (WD key names) --
                    "charge reward": avg_rew,
                    # -- Probabilities (WD key names) --
                    "charge/goals_per_ep": goals_ep,
                    "charge/collisions_per_ep": coll_ep,
                    "charge/car_collisions_per_ep": car_coll_ep,
                    # -- Training --
                    "train/fps": fps,
                    "train/grad_norm": avg_gn,
                    "train/grad_norm_policy": avg_gn_policy,
                    "train/grad_norm_value": avg_gn_value,
                    "train/grad_clip_mode": 0 if args_cli.grad_clip_mode == "merged" else 1,
                    "train/gamma": _gamma,
                    "train/n_active_cars": n_active,
                    # -- Curriculum --
                    "curriculum/stage": cur_stage,
                    "curriculum/success_rate": cur_sr,
                    "curriculum/eval_sr": cur_metrics.get("eval_sr", 0.0) if curriculum else 0.0,
                    "curriculum/eval_coll": cur_metrics.get("eval_coll", 0.0) if curriculum else 0.0,
                    "curriculum/phase_iters": cur_metrics.get("phase_iters", 0) if curriculum else 0,
                }

                # -- Reward decomposition (WD-aligned, all "charge" prefix) --
                # Average across all cars
                rc_avg = {}
                for i in range(N):
                    rc = env._last_reward_components[i]
                    for k, v in rc.items():
                        rc_avg[k] = rc_avg.get(k, 0.0) + v / N

                if rc_avg:
                    # WD new_car equivalent keys (spot→charge)
                    log_dict["charge dynamic obstacle reward"] = rc_avg.get("obs_collision", 0)
                    log_dict["charge static obstacle reward"] = 0.0  # not separated
                    log_dict["charge goal reward"] = rc_avg.get("goal_bonus", 0)
                    log_dict["charge dynamic reward"] = rc_avg.get("velocity_to_goal", 0)  # WD: car dynamic reward
                    log_dict["charge floor reward"] = 0.0  # no floor boundary in Isaac Lab
                    # WD-aligned probabilities
                    log_dict["charge hit probability"] = rc_avg.get("hit_probability", 0)
                    goal_remain = 1.0 - rc_avg.get("goal_reach_probability", 0)
                    log_dict["goal remain probability"] = goal_remain
                    log_dict["charge dies at birth probability"] = rc_avg.get("born_died_probability", 0)
                    log_dict["charge time goal remain probability"] = goal_remain  # WD: car time goal remain probability
                    # Extra reward components (not in WD, but useful)
                    log_dict["charge safety log reward"] = rc_avg.get("safety_log", 0)
                    log_dict["charge safe progress reward"] = rc_avg.get("safe_progress", 0)
                    log_dict["charge step reward"] = rc_avg.get("time_penalty", 0)
                    log_dict["charge car collision reward"] = rc_avg.get("car_collision", 0)
                    # Physics diagnostics
                    log_dict["charge/d_safe_mean"] = rc_avg.get("d_safe_mean", 0)
                    log_dict["charge/speed_mean"] = rc_avg.get("speed_mean", 0)
                    log_dict["charge/v_toward_mean"] = rc_avg.get("v_toward_mean", 0)

                wandb_run.log(log_dict, step=global_step)

        # ============================================================
        # Save checkpoint
        # ============================================================
        if iteration % args_cli.save_interval == 0:
            ckpt_dir = log_dir / "checkpoints"
            ckpt_dir.mkdir(exist_ok=True)
            ckpt_path = ckpt_dir / f"step_{global_step}.pt"
            ckpt_data = {
                "global_step": global_step,
                "iteration": iteration,
                "extractor": extractor.state_dict(),
                "preprocess_rnn": preprocess_rnn.state_dict(),
                "policy_head": policy_head.state_dict(),
                "value_head": value_head.state_dict(),
                "obs_normalizer": {
                    "mean": obs_normalizer.mean,
                    "var": obs_normalizer.var,
                    "count": obs_normalizer.count,
                },
            }
            if curriculum:
                ckpt_data["curriculum_stage"] = curriculum.stage
                ckpt_data["gamma"] = _gamma
            torch.save(ckpt_data, ckpt_path)
            print(f"[MARL-RNN] Saved checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # Final save
    # ------------------------------------------------------------------
    final_path = log_dir / "final_model.pt"
    torch.save({
        "global_step": global_step,
        "extractor": extractor.state_dict(),
        "preprocess_rnn": preprocess_rnn.state_dict(),
        "policy_head": policy_head.state_dict(),
        "value_head": value_head.state_dict(),
        "obs_normalizer": {
            "mean": obs_normalizer.mean,
            "var": obs_normalizer.var,
            "count": obs_normalizer.count,
        },
    }, final_path)
    print(f"[MARL-RNN] Training complete. Final model: {final_path}")

    if wandb_run is not None:
        import wandb
        wandb.finish()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
