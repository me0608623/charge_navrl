# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
訓練參數自動記錄器

在每次訓練啟動時自動生成完整參數檔，包含：
- 獎勵設計（項目名、函數、權重、物理意義、預期走勢）
- 環境配置（num_envs、episode_length、dt 等）
- PPO 超參數（LR、rollouts、mini_batches 等）
- 訓練控制（timesteps、PPO updates、total env steps）
- 動作空間與觀測空間

輸出至 {log_dir}/training_params_{run_name}.yaml
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from isaaclab.managers import RewardTermCfg


# 獎勵項目物理意義與預期走勢（手動維護，覆蓋所有已知 reward term）
REWARD_DESCRIPTIONS = {
    "reaching_goal": {
        "meaning": "到達目標點的一次性獎勵",
        "expected_trend": "隨訓練上升，反映導航成功率提升",
        "category": "成功",
    },
    "potential_progress": {
        "meaning": "PBRS 距離縮減獎勵 — Phi(s')−Phi(s)，鼓勵靠近目標",
        "expected_trend": "訓練初期上升，穩定後維持正值；過高可能衝向障礙物",
        "category": "前進",
    },
    "near_obstacle_penalty": {
        "meaning": "線性近障礙物懲罰 — d_min < safe_distance 時線性增加",
        "expected_trend": "訓練中下降（學會繞路），過低=沒有避障壓力",
        "category": "安全",
    },
    "proximity_brake": {
        "meaning": "近距離制動懲罰 — d_min 在 [collision_threshold, safe_distance] 內線性懲罰",
        "expected_trend": "Phase 2 初期增大（開始感知障礙物），後期下降（學會提前減速）",
        "category": "安全（Phase 2）",
    },
    "risk_speed": {
        "meaning": "風險速度懲罰 — 靠近障礙物時高速行駛的乘積懲罰 (speed × proximity)",
        "expected_trend": "Phase 2 初期增大，後期下降（學會靠近時減速）",
        "category": "安全（Phase 2）",
    },
    "exponential_obstacle_penalty": {
        "meaning": "指數型排斥力 — d_min < safe_distance 時指數增加，靠近時極大懲罰",
        "expected_trend": "訓練中下降（學會保持距離）",
        "category": "安全",
    },
    "collision_terminal": {
        "meaning": "碰撞終止懲罰 — LiDAR 偵測 d_min ≤ threshold 時的一次性懲罰",
        "expected_trend": "下降（碰撞次數減少）；始終為負，接近 0 = 幾乎不碰撞",
        "category": "安全",
    },
    "time_penalty": {
        "meaning": "每步固定時間懲罰 — 鼓勵更快完成任務",
        "expected_trend": "穩定負值，episode 越短累計越少",
        "category": "效率",
    },
    "velocity_too_low": {
        "meaning": "靜止懲罰 — 速度低於閾值時懲罰，防止 agent 學會不動",
        "expected_trend": "訓練中下降（學會移動）；持續高 = agent 凍結風險",
        "category": "防呆",
    },
    "acceleration_penalty": {
        "meaning": "加速度平方懲罰 — |a_t|²，鼓勵平滑加速",
        "expected_trend": "穩定在低值，過高 = 動作劇烈抖動",
        "category": "平滑",
    },
    "angular_velocity_penalty": {
        "meaning": "角速度平方懲罰 — |ω_t|²，鼓勵平滑轉向",
        "expected_trend": "穩定在低值，過高 = 原地打轉",
        "category": "平滑",
    },
}


def dump_training_params(
    log_dir: str,
    run_name: str,
    env_cfg,
    agent_cfg: dict,
    args_cli,
    extra_info: Optional[dict] = None,
) -> str:
    """生成完整訓練參數檔。

    Args:
        log_dir: 訓練日誌目錄
        run_name: 訓練 run 名稱（用於檔名）
        env_cfg: ManagerBasedRLEnvCfg 實例
        agent_cfg: SKRL agent 配置字典
        args_cli: CLI 參數
        extra_info: 額外資訊字典（可選）

    Returns:
        生成的檔案路徑
    """
    os.makedirs(log_dir, exist_ok=True)

    # 安全檔名
    safe_name = run_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    output_path = os.path.join(log_dir, f"training_params_{safe_name}.yaml")

    lines = []
    lines.append("# " + "=" * 78)
    lines.append(f"# Training Parameters — {run_name}")
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("# " + "=" * 78)
    lines.append("")

    # ── 1. 基本資訊 ──
    lines.append("# " + "-" * 78)
    lines.append("# 1. Basic Info")
    lines.append("# " + "-" * 78)
    lines.append(f"task: {getattr(args_cli, 'task', 'unknown')}")
    lines.append(f"phase: {getattr(args_cli, 'phase', 1)}")
    lines.append(f"checkpoint: {getattr(args_cli, 'checkpoint', None)}")
    lines.append(f"seed: {agent_cfg.get('seed', 'N/A')}")
    lines.append(f"device: {getattr(args_cli, 'device', 'N/A')}")
    lines.append("")

    # ── 2. 環境配置 ──
    lines.append("# " + "-" * 78)
    lines.append("# 2. Environment Configuration")
    lines.append("# " + "-" * 78)
    num_envs = getattr(env_cfg.scene, 'num_envs', 'N/A')
    decimation = getattr(env_cfg, 'decimation', 'N/A')
    sim_dt = getattr(env_cfg.sim, 'dt', 'N/A') if hasattr(env_cfg, 'sim') else 'N/A'
    episode_length_s = getattr(env_cfg, 'episode_length_s', 'N/A')

    env_dt = "N/A"
    if isinstance(decimation, (int, float)) and isinstance(sim_dt, (int, float)):
        env_dt = round(decimation * sim_dt, 4)

    max_episode_steps = "N/A"
    if isinstance(episode_length_s, (int, float)) and isinstance(env_dt, (int, float)) and env_dt > 0:
        max_episode_steps = int(episode_length_s / env_dt)

    lines.append(f"num_envs: {num_envs}")
    lines.append(f"decimation: {decimation}")
    lines.append(f"sim_dt: {sim_dt}")
    lines.append(f"env_dt: {env_dt}  # decimation × sim_dt")
    lines.append(f"episode_length_s: {episode_length_s}")
    lines.append(f"max_episode_steps: {max_episode_steps}")
    lines.append(f"env_spacing: {getattr(env_cfg.scene, 'env_spacing', 'N/A')}")
    lines.append("")

    # ── 3. 動作空間 ──
    lines.append("# " + "-" * 78)
    lines.append("# 3. Action Space")
    lines.append("# " + "-" * 78)
    if hasattr(env_cfg, 'actions'):
        actions_cfg = env_cfg.actions
        for attr_name in dir(actions_cfg):
            if attr_name.startswith('_'):
                continue
            attr = getattr(actions_cfg, attr_name, None)
            if attr is None or callable(attr):
                continue
            if hasattr(attr, '__dataclass_fields__') or hasattr(attr, '__dict__'):
                lines.append(f"{attr_name}:")
                for field_name in dir(attr):
                    if field_name.startswith('_'):
                        continue
                    val = getattr(attr, field_name, None)
                    if val is not None and not callable(val) and not field_name.startswith('__'):
                        if isinstance(val, (int, float, str, bool)):
                            lines.append(f"  {field_name}: {val}")
    lines.append("")

    # ── 4. PPO 超參數 ──
    lines.append("# " + "-" * 78)
    lines.append("# 4. PPO Hyperparameters")
    lines.append("# " + "-" * 78)
    agent_params = agent_cfg.get("agent", {})
    rollouts = agent_params.get("rollouts", 128)
    timesteps = agent_cfg.get("trainer", {}).get("timesteps", "N/A")

    ppo_updates = "N/A"
    total_env_steps = "N/A"
    if isinstance(timesteps, (int, float)) and isinstance(rollouts, (int, float)):
        ppo_updates = int(timesteps) // int(rollouts)
    if isinstance(timesteps, (int, float)) and isinstance(num_envs, (int, float)):
        total_env_steps = int(timesteps) * int(num_envs)

    batch_size = "N/A"
    mini_batch_size = "N/A"
    mini_batches = agent_params.get("mini_batches", 8)
    if isinstance(rollouts, (int, float)) and isinstance(num_envs, (int, float)):
        batch_size = int(rollouts) * int(num_envs)
        if isinstance(mini_batches, (int, float)) and mini_batches > 0:
            mini_batch_size = batch_size // int(mini_batches)

    lines.append(f"rollouts: {rollouts}")
    lines.append(f"learning_epochs: {agent_params.get('learning_epochs', 'N/A')}")
    lines.append(f"mini_batches: {mini_batches}")
    lines.append(f"batch_size: {batch_size}  # rollouts × num_envs")
    lines.append(f"mini_batch_size: {mini_batch_size}  # batch_size / mini_batches")
    lines.append(f"discount_factor: {agent_params.get('discount_factor', 'N/A')}")
    lines.append(f"lambda_gae: {agent_params.get('lambda', 'N/A')}")
    lines.append(f"learning_rate: {agent_params.get('learning_rate', 'N/A')}")
    lines.append(f"entropy_loss_scale: {agent_params.get('entropy_loss_scale', 'N/A')}")
    lines.append(f"value_loss_scale: {agent_params.get('value_loss_scale', 'N/A')}")
    lines.append(f"grad_norm_clip: {agent_params.get('grad_norm_clip', 'N/A')}")
    lines.append(f"ratio_clip: {agent_params.get('ratio_clip', 'N/A')}")
    lines.append(f"value_clip: {agent_params.get('value_clip', 'N/A')}")
    lines.append(f"clip_predicted_values: {agent_params.get('clip_predicted_values', 'N/A')}")
    lines.append(f"rewards_shaper_scale: {agent_params.get('rewards_shaper_scale', 'N/A')}")
    lines.append(f"time_limit_bootstrap: {agent_params.get('time_limit_bootstrap', 'N/A')}")
    lines.append(f"state_preprocessor: {agent_params.get('state_preprocessor', 'N/A')}")
    lines.append(f"value_preprocessor: {agent_params.get('value_preprocessor', 'N/A')}")
    lines.append("")

    # LR scheduler
    lr_sched = agent_params.get("learning_rate_scheduler", None)
    if lr_sched:
        lines.append("# LR Scheduler")
        lines.append(f"lr_scheduler: {lr_sched}")
        lr_kwargs = agent_params.get("learning_rate_scheduler_kwargs", {})
        if lr_kwargs:
            for k, v in lr_kwargs.items():
                lines.append(f"  {k}: {v}")
        lines.append("")

    # ── 5. 訓練控制 ──
    lines.append("# " + "-" * 78)
    lines.append("# 5. Training Control")
    lines.append("# " + "-" * 78)
    lines.append(f"timesteps: {timesteps}")
    lines.append(f"ppo_updates: {ppo_updates}  # timesteps / rollouts")
    lines.append(f"total_env_steps: {total_env_steps}  # timesteps × num_envs")
    lines.append("")

    # ── 6. 獎勵設計 ──
    lines.append("# " + "-" * 78)
    lines.append("# 6. Reward Design")
    lines.append("# " + "-" * 78)
    lines.append(f"# env_dt = {env_dt}s — RewardManager 計算: reward = func() × weight × dt")
    lines.append(f"# WandB key 格式: Info / Episode_Reward/{{term_name}}")
    lines.append(f"#   值 = episode_sum / max_episode_length_s (即每秒平均 reward)")
    lines.append("")

    if hasattr(env_cfg, 'rewards'):
        rewards_cfg = env_cfg.rewards
        lines.append("rewards:")
        for attr_name in sorted(dir(rewards_cfg)):
            if attr_name.startswith('_'):
                continue
            term = getattr(rewards_cfg, attr_name, None)
            if term is None:
                # 被設為 None 的 term（如 Phase 2 移除的 near_obstacle_penalty）
                lines.append(f"  {attr_name}: null  # 已移除")
                continue
            if not isinstance(term, RewardTermCfg):
                continue

            func_name = getattr(term.func, '__name__', str(term.func))
            weight = term.weight
            params = term.params if term.params else {}

            desc = REWARD_DESCRIPTIONS.get(attr_name, {})
            meaning = desc.get("meaning", "（未註冊說明）")
            trend = desc.get("expected_trend", "（未註冊）")
            category = desc.get("category", "其他")

            # 計算 per-step 量級範例
            per_step = ""
            if isinstance(weight, (int, float)) and isinstance(env_dt, (int, float)):
                per_step_val = abs(weight) * env_dt
                per_step = f"  # |weight×dt| = {per_step_val:.3f}/step (when func=1.0)"

            lines.append(f"  {attr_name}:")
            lines.append(f"    function: {func_name}")
            lines.append(f"    weight: {weight}{per_step}")
            lines.append(f"    category: {category}")
            lines.append(f"    meaning: \"{meaning}\"")
            lines.append(f"    expected_trend: \"{trend}\"")
            lines.append(f"    wandb_key: \"Info / Episode_Reward/{attr_name}\"")
            if params:
                lines.append(f"    params:")
                for pk, pv in params.items():
                    # SceneEntityCfg 不能直接 serialize
                    if hasattr(pv, 'name'):
                        lines.append(f"      {pk}: \"{pv.name}\"")
                    else:
                        lines.append(f"      {pk}: {pv}")
            lines.append("")

    # ── 7. 終止條件 ──
    lines.append("# " + "-" * 78)
    lines.append("# 7. Termination Conditions")
    lines.append("# " + "-" * 78)
    if hasattr(env_cfg, 'terminations'):
        term_cfg = env_cfg.terminations
        lines.append("terminations:")
        for attr_name in sorted(dir(term_cfg)):
            if attr_name.startswith('_'):
                continue
            term = getattr(term_cfg, attr_name, None)
            if term is None or not hasattr(term, 'func'):
                continue
            func_name = getattr(term.func, '__name__', str(term.func))
            is_timeout = getattr(term, 'time_out', False)
            lines.append(f"  {attr_name}:")
            lines.append(f"    function: {func_name}")
            lines.append(f"    time_out: {is_timeout}")
    lines.append("")

    # ── 8. 環境分布 ──
    lines.append("# " + "-" * 78)
    lines.append("# 8. Environment Distribution (obstacle mix)")
    lines.append("# " + "-" * 78)
    if hasattr(env_cfg, 'events'):
        events = env_cfg.events
        for attr_name in ['randomize_obstacles', 'randomize_obstacles_startup']:
            evt = getattr(events, attr_name, None)
            if evt is not None and hasattr(evt, 'params'):
                p = evt.params
                lines.append(f"{attr_name}:")
                for k in ['empty_ratio', 'static_ratio', 'dynamic_ratio',
                           'num_obstacles_static', 'num_obstacles_dynamic',
                           'max_obstacles', 'speed_range', 'min_speed']:
                    if k in p:
                        lines.append(f"  {k}: {p[k]}")
                lines.append("")

    # ── 9. 模型架構 ──
    lines.append("# " + "-" * 78)
    lines.append("# 9. Model Architecture")
    lines.append("# " + "-" * 78)
    models_cfg = agent_cfg.get("models", {})
    lines.append(f"separate: {models_cfg.get('separate', 'N/A')}")
    for role in ['policy', 'value']:
        if role in models_cfg:
            lines.append(f"{role}:")
            for k, v in models_cfg[role].items():
                lines.append(f"  {k}: {v}")
    lines.append("")

    # ── 10. WandB Key 速查表 ──
    lines.append("# " + "-" * 78)
    lines.append("# 10. WandB Key Quick Reference")
    lines.append("# " + "-" * 78)
    lines.append("wandb_keys:")
    lines.append("  # ── Reward per-term (episode 平均，每秒) ──")
    if hasattr(env_cfg, 'rewards'):
        for attr_name in sorted(dir(env_cfg.rewards)):
            term = getattr(env_cfg.rewards, attr_name, None)
            if term is None or not isinstance(term, RewardTermCfg):
                continue
            lines.append(f"  - \"Info / Episode_Reward/{attr_name}\"")
    lines.append("")
    lines.append("  # ── Navigation metrics (running average) ──")
    lines.append("  - nav/success_rate")
    lines.append("  - nav/collision_rate")
    lines.append("  - nav/timeout_rate")
    lines.append("  - nav/total_episodes")
    lines.append("  - nav/episode_length_mean")
    lines.append("")
    lines.append("  # ── Per-env-type metrics (rolling window) ──")
    lines.append("  # 分場景成功/碰撞/超時率，需 _env_difficulty tensor 存在")
    for env_type in ["empty", "static", "dynamic"]:
        lines.append(f"  - eval/success_rate_{env_type}")
        lines.append(f"  - eval/collision_rate_{env_type}")
        lines.append(f"  - eval/timeout_rate_{env_type}")
    lines.append("")
    lines.append("  # ── Training metrics ──")
    lines.append("  - train/fps")
    lines.append("  - train/timestep")
    lines.append("")

    # ── 額外資訊 ──
    if extra_info:
        lines.append("# " + "-" * 78)
        lines.append("# 10. Extra Info")
        lines.append("# " + "-" * 78)
        for k, v in extra_info.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    # 寫入檔案
    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[INFO] Training parameters saved to: {output_path}")
    return output_path
