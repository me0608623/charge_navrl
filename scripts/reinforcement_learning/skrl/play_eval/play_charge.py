#!/usr/bin/env python3
"""
play_charge.py — Charge RL 統一 Play 入口

用法:
  # 自動從 checkpoint 推斷 phase
  python play_charge.py --checkpoint logs/rnn_car/xxx/checkpoint_210000.pt

  # 指定 phase（覆寫場景配置）
  python play_charge.py --checkpoint xxx.pt --phase 3

  # 手動覆寫場景參數
  python play_charge.py --checkpoint xxx.pt --phase 5 --num_dynamic_obs 8 --num_walls 3

場景參數按 phase 自動設定（elif stage == x 結構），CLI 可覆寫。
BEV + RNN aux 7D 對比永遠啟用。
"""

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher


# ============================================================================
# CLI 參數
# ============================================================================
parser = argparse.ArgumentParser(description="Charge RL Play — 統一視覺化入口")

# --- 基礎 ---
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-WD")
parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint .pt 路徑")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=3000, help="最大 play 步數")
parser.add_argument("--deterministic", action="store_true", default=True,
                    help="使用 argmax（預設 True）")
parser.add_argument("--no_deterministic", action="store_true", default=False,
                    help="使用 sampling（探索模式）")
parser.add_argument("--real_time", action="store_true", default=False,
                    help="以真實時間步進")

# --- Phase / 場景控制 ---
parser.add_argument("--phase", type=int, default=None,
                    help="指定 curriculum phase (1-5)。None=從 checkpoint 推斷")
parser.add_argument("--num_goals", type=int, default=None, help="覆寫目標數量")
parser.add_argument("--num_static_obs", type=int, default=None, help="覆寫靜態障礙物數")
parser.add_argument("--num_dynamic_obs", type=int, default=None, help="覆寫動態障礙物數")
parser.add_argument("--num_walls", type=int, default=None, help="覆寫內部牆壁數")
parser.add_argument("--wall_length", type=float, default=None, help="覆寫牆壁長度 (m)")
parser.add_argument("--goal_distance_min", type=float, default=None, help="覆寫最小目標距離")
parser.add_argument("--goal_distance_max", type=float, default=None, help="覆寫最大目標距離")
parser.add_argument("--scripted_obstacles", action="store_true", default=True,
                    help="動態障礙物移動（預設開啟）")
parser.add_argument("--no_scripted_obstacles", action="store_true", default=False,
                    help="凍結動態障礙物")

# --- VO Shield ---
parser.add_argument("--use_vo_shield", action="store_true", default=False)
parser.add_argument("--shield_mode", type=str, default="soft", choices=["soft", "hard"])
parser.add_argument("--vo_horizon", type=float, default=1.0)
parser.add_argument("--vo_safety_radius", type=float, default=0.45)
parser.add_argument("--vo_evade_gain", type=float, default=1.0)

# --- BEV 顯示 ---
parser.add_argument("--bev_frame", type=str, default="world", choices=["body", "world"])
parser.add_argument("--bev_max_range", type=float, default=20.0)
parser.add_argument("--bev_update_interval", type=int, default=2)
parser.add_argument("--no_aux_panel", action="store_true", default=False,
                    help="關閉 RNN aux 7D 對比面板")

# --- 其他 ---
parser.add_argument("--lidar_no_noise", action="store_true", default=False)
parser.add_argument("--camera", type=str, default="top", choices=["top", "follow", "side"])

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.no_deterministic:
    args_cli.deterministic = False
if args_cli.no_scripted_obstacles:
    args_cli.scripted_obstacles = False

if not getattr(args_cli, "headless", False):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============================================================================
# Post-AppLauncher imports
# ============================================================================
import gymnasium as gym
import torch
from torch.distributions import Categorical

from isaaclab.envs import ViewerCfg
import isaaclab_tasks  # noqa: F401

_skrl_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(_skrl_root / "models"))
sys.path.insert(0, str(_skrl_root / "utils"))

from bev_renderer import ChargeBEVRenderer
from charge_env_overrides import apply_charge_env_overrides
from modular_rnn_models import PolicyHead, PreprocessRNN, RNNStateManager, ValueHead
from wd_aux_targets import build_wd_preprocess_targets


# ============================================================================
# Phase 場景預設值（elif stage == x 結構）
# ============================================================================
def get_phase_defaults(phase: int) -> dict:
    """依 phase 回傳場景預設參數。"""
    if phase == 1:
        return {
            "name": "SA1_目標學習+少量動態",
            "num_goals": 10,
            "num_static_obs": 0,
            "num_dynamic_obs": 2,
            "num_dynamic_obs_min": 1,
            "walls_min": 0,
            "walls_max": 1,
            "wall_length": 5.0,
            "goal_distance": (2.0, 6.0),
            "episode_s": 60,
            "obstacle_speed": 0.8,
        }
    elif phase == 2:
        return {
            "name": "SA2_遠距探索+動態維持",
            "num_goals": 8,
            "num_static_obs": 0,
            "num_dynamic_obs": 2,
            "num_dynamic_obs_min": 1,
            "walls_min": 0,
            "walls_max": 1,
            "wall_length": 5.0,
            "goal_distance": (2.0, 7.0),
            "episode_s": 60,
            "obstacle_speed": 0.8,
        }
    elif phase == 3:
        return {
            "name": "SA3_動態增加+牆壁引入",
            "num_goals": 6,
            "num_static_obs": 0,
            "num_dynamic_obs": 3,
            "num_dynamic_obs_min": 2,
            "walls_min": 1,
            "walls_max": 2,
            "wall_length": 4.5,
            "goal_distance": (2.0, 8.0),
            "episode_s": 60,
            "obstacle_speed": 0.85,
        }
    elif phase == 4:
        return {
            "name": "SA4_長距導航+持續干擾",
            "num_goals": 4,
            "num_static_obs": 0,
            "num_dynamic_obs": 4,
            "num_dynamic_obs_min": 3,
            "walls_min": 1,
            "walls_max": 2,
            "wall_length": 4.0,
            "goal_distance": (2.0, 9.0),
            "episode_s": 75,
            "obstacle_speed": 0.85,
        }
    elif phase == 5:
        return {
            "name": "SA5_高密度動態+最終難度",
            "num_goals": 2,
            "num_static_obs": 0,
            "num_dynamic_obs": 6,
            "num_dynamic_obs_min": 4,
            "walls_min": 1,
            "walls_max": 2,
            "wall_length": 3.5,
            "goal_distance": (2.0, 10.0),
            "episode_s": 90,
            "obstacle_speed": 0.85,
        }
    else:
        raise ValueError(f"不支援的 phase: {phase}（有效範圍 1-5）")


# ============================================================================
# 環境設定
# ============================================================================
def resolve_env_cfg(task: str):
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum-WD":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_wd_sparse import (
            ChargeNavigationEnvCfgVLP16CurriculumWD,
        )
        return ChargeNavigationEnvCfgVLP16CurriculumWD()
    if task == "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL":
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
            ChargeNavigationEnvCfgVLP16CurriculumNavRL,
        )
        return ChargeNavigationEnvCfgVLP16CurriculumNavRL()
    raise ValueError(f"不支援的 task: {task}")


def apply_phase_to_env(env_cfg, phase_cfg: dict, cli_args):
    """將 phase 預設 + CLI override 套用到 env_cfg"""
    num_goals = cli_args.num_goals if cli_args.num_goals is not None else phase_cfg["num_goals"]
    num_dynamic = cli_args.num_dynamic_obs if cli_args.num_dynamic_obs is not None else phase_cfg["num_dynamic_obs"]
    num_static = cli_args.num_static_obs if cli_args.num_static_obs is not None else phase_cfg["num_static_obs"]
    walls_max = cli_args.num_walls if cli_args.num_walls is not None else phase_cfg["walls_max"]
    wall_length = cli_args.wall_length if cli_args.wall_length is not None else phase_cfg["wall_length"]
    goal_dist_min = cli_args.goal_distance_min if cli_args.goal_distance_min is not None else phase_cfg["goal_distance"][0]
    goal_dist_max = cli_args.goal_distance_max if cli_args.goal_distance_max is not None else phase_cfg["goal_distance"][1]

    # 套用到 curriculum term
    cur = getattr(env_cfg, "curriculum", None)
    term = getattr(cur, "goal_obstacle_curriculum", None) if cur is not None else None
    if term is not None:
        params = term.params
        params["curriculum_version"] = "warp_drive_single_agent_v1"
        params["initial_stage"] = (cli_args.phase or 1) - 1  # 0-indexed

    # 套用到 goal command
    try:
        goal_cmd = env_cfg.commands.goal_command
        goal_cmd.num_goals = num_goals
        if hasattr(goal_cmd, "goal_distance_range"):
            goal_cmd.goal_distance_range = (goal_dist_min, goal_dist_max)
    except AttributeError:
        pass

    return {
        "num_goals": num_goals,
        "num_static_obs": num_static,
        "num_dynamic_obs": num_dynamic,
        "walls_max": walls_max,
        "wall_length": wall_length,
        "goal_distance": (goal_dist_min, goal_dist_max),
    }


def configure_camera(env_cfg, mode: str):
    if mode == "top":
        env_cfg.viewer = ViewerCfg(eye=(0.0, 0.0, 25.0), lookat=(0.0, 0.0, 0.0))
    elif mode == "follow":
        env_cfg.viewer = ViewerCfg(eye=(0.0, -5.0, 5.0), lookat=(0.0, 0.0, 0.5))
    elif mode == "side":
        env_cfg.viewer = ViewerCfg(eye=(15.0, 0.0, 10.0), lookat=(0.0, 0.0, 0.0))


# ============================================================================
# 模型工具
# ============================================================================
POLICY_OBS_INDICES = list(range(0, 78)) + [138]


def build_wd_like_obs(obs_normed: torch.Tensor) -> torch.Tensor:
    """139D IsaacLab obs → 113D WD car layout"""
    leading = obs_normed.shape[:-1]
    base = torch.zeros(*leading, 17, dtype=obs_normed.dtype, device=obs_normed.device)
    base[..., 0] = obs_normed[..., 1]        # speed_x
    base[..., 2:4] = obs_normed[..., 4:6]    # body-frame goal x/y
    base[..., 15] = obs_normed[..., 138]     # time
    base[..., 16] = 1.0                      # in_game
    obstacles = obs_normed[..., 78:138]
    lidar72 = obs_normed[..., 6:78]
    lidar36 = lidar72.reshape(*leading, 36, 2).mean(dim=-1)
    return torch.cat([base, obstacles, lidar36], dim=-1)


def sample_action(logits: torch.Tensor, deterministic: bool) -> torch.Tensor:
    logits_a, logits_w = logits[:, :19], logits[:, 19:]
    if deterministic:
        act_a = logits_a.argmax(dim=-1)
        act_w = logits_w.argmax(dim=-1)
    else:
        act_a = Categorical(logits=logits_a).sample()
        act_w = Categorical(logits=logits_w).sample()
    return torch.stack([act_a, act_w], dim=-1)


# ============================================================================
# 主程式
# ============================================================================
def main():
    ckpt_path = os.path.abspath(args_cli.checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    # 讀取 checkpoint 設定
    ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt_meta.get("args", {})
    del ckpt_meta

    # 自動 lidar_no_noise
    if ckpt_args.get("lidar_no_noise", False):
        args_cli.lidar_no_noise = True
        print("[PLAY] 自動啟用 --lidar_no_noise（來自 checkpoint）")

    # 推斷 phase
    phase = args_cli.phase
    if phase is None:
        phase = 1
        print(f"[PLAY] 未指定 --phase，預設 phase={phase}")
    phase_cfg = get_phase_defaults(phase)
    print(f"[PLAY] Phase {phase}: {phase_cfg['name']}")

    # 環境設定
    env_cfg = resolve_env_cfg(args_cli.task)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    apply_charge_env_overrides(env_cfg, args_cli)
    scene_params = apply_phase_to_env(env_cfg, phase_cfg, args_cli)
    configure_camera(env_cfg, args_cli.camera)

    if hasattr(env_cfg.scene, "lidar"):
        env_cfg.scene.lidar.debug_vis = True
    if not args_cli.headless:
        env_cfg.sim.render_interval = 4

    print(f"[PLAY] checkpoint: {ckpt_path}")
    print(f"[PLAY] 場景: 目標={scene_params['num_goals']} "
          f"靜態={scene_params['num_static_obs']} 動態={scene_params['num_dynamic_obs']} "
          f"牆壁≤{scene_params['walls_max']} 牆長={scene_params['wall_length']:.1f}m "
          f"目標距離={scene_params['goal_distance']}")

    # 建立環境
    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    obs, _ = env.reset()
    device = raw_env.device

    def policy_obs(x):
        return x["policy"] if isinstance(x, dict) else x

    obs_tensor = policy_obs(obs)

    # 載入模型
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    hidden_dim = int(ckpt_args.get("hidden_dim", 30))
    preprocess_dim = int(ckpt_args.get("preprocess_dim", 12))
    fc_dim = int(ckpt_args.get("fc_dim", 48))
    rnn_type = ckpt_args.get("rnn_type", "RNN")
    encoder_mode = ckpt_args.get("charge_encoder_mode", "wd_exact_rnn")
    middle_dim = int(ckpt_args.get("wd_middle_dim", 32))

    wd_exact_mode = (encoder_mode == "wd_exact_rnn")
    policy_obs_dim = 113 if wd_exact_mode else len(POLICY_OBS_INDICES)

    preprocess_rnn = PreprocessRNN(
        input_dim=policy_obs_dim,
        hidden_dim=hidden_dim,
        preprocess_dim=preprocess_dim,
        fc_dim=fc_dim,
        rnn_type=rnn_type,
        middle_dim=middle_dim,
    ).to(device)
    policy_head = PolicyHead(input_dim=policy_obs_dim + preprocess_dim).to(device)
    value_head = ValueHead(input_dim=policy_obs_dim + preprocess_dim).to(device)

    preprocess_rnn.load_state_dict(ckpt["preprocess_rnn"])
    policy_head.load_state_dict(ckpt["policy_head"])
    value_head.load_state_dict(ckpt["value_head"])
    preprocess_rnn.eval()
    policy_head.eval()
    value_head.eval()

    print(f"[PLAY] 模型: encoder={encoder_mode} obs_dim={policy_obs_dim} "
          f"hidden={hidden_dim} preprocess={preprocess_dim}")

    # Normalizer
    obs_norm = ckpt.get("obs_normalizer", {})
    mean = obs_norm.get("mean", torch.zeros(obs_tensor.shape[-1], device=device))
    var = obs_norm.get("var", torch.ones(obs_tensor.shape[-1], device=device))
    rnn_state = RNNStateManager(raw_env.num_envs, hidden_dim, device)

    # 障礙物動作
    raw_env._obstacle_policy_active = not args_cli.scripted_obstacles
    print(f"[PLAY] 動態障礙物: {'移動中（scripted）' if args_cli.scripted_obstacles else '凍結'}")

    step_dt = env.step_dt if hasattr(env, "step_dt") else raw_env.step_dt

    # BEV renderer（永遠啟用）
    show_aux = not args_cli.no_aux_panel
    try:
        bev = ChargeBEVRenderer(
            raw_env, max_range=args_cli.bev_max_range,
            frame=args_cli.bev_frame, show_aux=show_aux,
        )
        print(f"[PLAY] BEV 啟動 (frame={args_cli.bev_frame}, aux={'開' if show_aux else '關'})")
    except Exception as exc:
        print(f"[PLAY] 警告: BEV 初始化失敗: {exc}")
        bev = None

    # 統計
    episode_reward = torch.zeros(raw_env.num_envs, device=device)
    episode_step = torch.zeros(raw_env.num_envs, dtype=torch.long, device=device)
    stats = {"goal": 0, "wall": 0, "obs": 0, "timeout": 0, "other": 0, "total": 0}
    steps_list = []

    def normalize(x):
        return torch.clamp((x - mean) / (var.sqrt() + 1e-8), -5.0, 5.0)

    def get_obs_for_model(obs_normed):
        if wd_exact_mode:
            return build_wd_like_obs(obs_normed)
        return obs_normed[:, POLICY_OBS_INDICES]

    def compute_aux_ground_truth():
        """從 simulator 計算 aux 7D ground truth"""
        try:
            return build_wd_preprocess_targets(raw_env, device=device)
        except Exception:
            return None

    def detect_termination_cause(terminated_flat, truncated_flat):
        N = terminated_flat.shape[0]
        cause = torch.zeros(N, dtype=torch.long, device=device)
        try:
            tm = raw_env.termination_manager
            for name in tm._term_names:
                buf = tm.get_term(name)
                if buf is None:
                    continue
                fired = buf.bool()
                if "goal_reached" in name or "reaching_goal" in name:
                    cause[fired & (cause == 0)] = 1
                elif "wall_collision" in name:
                    cause[fired & (cause == 0)] = 2
                elif "obstacle_collision" in name or "collision" in name:
                    cause[fired & (cause == 0)] = 3
        except (AttributeError, RuntimeError):
            pass
        trunc = truncated_flat.bool() if truncated_flat.dim() == 1 else truncated_flat.squeeze(-1).bool()
        cause[trunc & (cause == 0)] = 4
        return cause

    CAUSE_NAMES = {0: "進行中", 1: "到達目標", 2: "撞牆", 3: "撞障礙", 4: "超時", 5: "其他"}

    # ========================================================================
    # 主迴圈
    # ========================================================================
    step = 0
    print(f"\n{'='*60}")
    print("開始 Play — 關閉 BEV 視窗或按 Ctrl+C 停止")
    print(f"{'='*60}\n")

    while simulation_app.is_running() and step < args_cli.steps:
        start = time.time()
        with torch.inference_mode():
            obs_tensor = policy_obs(obs)
            obs_normed = normalize(obs_tensor)
            p_obs = get_obs_for_model(obs_normed)
            hidden = rnn_state.get()
            rnn_feat, aux_pred, new_hidden = preprocess_rnn(p_obs, hidden, training=show_aux)
            rl_in = torch.cat([p_obs, rnn_feat], dim=-1)
            logits = policy_head(rl_in)
            actions = sample_action(logits, args_cli.deterministic)

        new_hidden = new_hidden.clone()

        # Aux ground truth
        aux_gt = compute_aux_ground_truth() if show_aux else None

        # BEV 更新
        if bev is not None and step % max(1, args_cli.bev_update_interval) == 0:
            if not bev.update(step, obs_tensor, actions, aux_pred=aux_pred, aux_gt=aux_gt):
                print("[PLAY] BEV 視窗已關閉。")
                break

        # Step
        next_obs, reward, terminated, truncated, info = env.step(actions.float())
        done = (terminated.squeeze(-1) | truncated.squeeze(-1)) if terminated.ndim > 1 else (terminated | truncated)

        episode_reward += reward.squeeze(-1) if reward.ndim > 1 else reward
        episode_step += 1
        rnn_state.update(new_hidden)

        if done.any():
            done_ids = done.nonzero(as_tuple=False).reshape(-1)
            terminated_flat = terminated.squeeze(-1) if terminated.ndim > 1 else terminated
            truncated_flat = truncated.squeeze(-1) if truncated.ndim > 1 else truncated
            cause = detect_termination_cause(terminated_flat, truncated_flat)

            for env_id in done_ids.tolist():
                c = cause[env_id].item()
                cause_name = CAUSE_NAMES.get(c, "?")
                ep_steps = episode_step[env_id].item()
                ep_rew = episode_reward[env_id].item()
                stats["total"] += 1
                steps_list.append(ep_steps)
                if c == 1:
                    stats["goal"] += 1
                elif c == 2:
                    stats["wall"] += 1
                elif c == 3:
                    stats["obs"] += 1
                elif c == 4:
                    stats["timeout"] += 1
                else:
                    stats["other"] += 1
                print(
                    f"  [回合 {stats['total']:3d}] {cause_name:6s} "
                    f"步數={ep_steps:4d} 獎勵={ep_rew:.1f}"
                )

            rnn_state.reset(done_ids)
            episode_reward[done_ids] = 0.0
            episode_step[done_ids] = 0

        obs = next_obs
        step += 1

        if args_cli.real_time:
            elapsed = time.time() - start
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

    # ========================================================================
    # 最終統計
    # ========================================================================
    print(f"\n{'='*60}")
    print("Play 統計摘要")
    print(f"{'='*60}")
    if stats["total"] > 0:
        sr = stats["goal"] / stats["total"] * 100
        cr = (stats["wall"] + stats["obs"]) / stats["total"] * 100
        to = stats["timeout"] / stats["total"] * 100
        avg_steps = sum(steps_list) / len(steps_list)
        print(f"  回合數:       {stats['total']}")
        print(f"  成功率 (SR):  {stats['goal']:4d} ({sr:.1f}%)")
        print(f"  撞牆:        {stats['wall']:4d} ({stats['wall']/stats['total']*100:.1f}%)")
        print(f"  撞障礙:      {stats['obs']:4d} ({stats['obs']/stats['total']*100:.1f}%)")
        print(f"  碰撞率 (CR):  {stats['wall']+stats['obs']:4d} ({cr:.1f}%)")
        print(f"  超時:        {stats['timeout']:4d} ({to:.1f}%)")
        print(f"  平均步數/回合: {avg_steps:.1f}")
    else:
        print("  未完成任何回合。")
    print(f"{'='*60}")

    if bev is not None:
        bev.close()
    env.close()


if __name__ == "__main__":
    main()
