# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
動態障礙物避障能力測試 — 驗證 v17 訓練的真實能力。

問題假設：
    Stage 14 訓練設定為 25 static + 10 dynamic（mixed mode）
    agent 報告 SR=71.8%, CR=0.18%
    但這是因為「真的學會避開動態障礙」嗎？
    還是因為「靜態障礙物已經佔了大部分空間，agent 只要避開靜態 + 偶爾運氣好」？

測試方法：
    用同一個 best checkpoint，依序測試以下配置：
    1. Stage 14 baseline (25S + 10D mixed)  ← 重現訓練分數
    2. 純動態 5D                            ← 簡單動態
    3. 純動態 10D                           ← 與 Stage 14 動態數量相同
    4. 純動態 25D                           ← 與 Stage 14 靜態數量相同
    5. 純動態 35D                           ← 與 Stage 14 總數相同

    每個 config 跑 600 steps × 6 envs，計算 SR/CR/TO

預期結果：
    - 若 SR 大致相同 → agent 真的學會了動態避障
    - 若 SR 在純動態下大幅下降 → agent 主要靠靜態 pattern，未學會動態預測

使用方法：
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_dynamic_test.py --headless

    # 指定特定 checkpoint
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_dynamic_test.py \
        --checkpoint logs/skrl/.../checkpoints/agent_46872.pt --headless

    # 跑更多 steps 增加統計可信度
    ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play_charge_dynamic_test.py \
        --steps_per_test 1200 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from isaaclab.app import AppLauncher

# ============================================================================
# CLI Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Dynamic obstacle avoidance test for v17 best checkpoint.")
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL")
parser.add_argument("--num_envs", type=int, default=6, help="Number of parallel envs (default 6).")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint. Default: best_model/best_agent.pt of v17.")
parser.add_argument("--steps_per_test", type=int, default=600,
                    help="Steps to run per test config (default 600).")
parser.add_argument("--episode_length_s", type=float, default=80.0,
                    help="Max episode length in seconds (default 80s).")
parser.add_argument("--no_walls", action="store_true", default=True,
                    help="Disable walls (matches v17 training).")
parser.add_argument("--deterministic", action="store_true", default=False,
                    help="Use argmax instead of sample.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force headless if not specified (this is a benchmark, not visual)
if not getattr(args_cli, "headless", False):
    print("[INFO] Running with rendering enabled. Use --headless for faster benchmark.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import skrl
from packaging import version
from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.skrl import SkrlVecEnvWrapper

# Custom model modules
sys.path.insert(0, str(Path(__file__).parent))
import vlp16_models as _vlp16_models_module
import charge_models as _charge_models_module
_CUSTOM_MODEL_MODULES = {
    "charge_models": _charge_models_module,
    "vlp16_models": _vlp16_models_module,
}

import isaaclab_tasks  # noqa: F401

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    print(f"[ERROR] skrl >= {SKRL_VERSION} required, got {skrl.__version__}")
    exit()


# ============================================================================
# Patch Runner for custom model classes (same as play_charge_ac_curriculum.py)
# ============================================================================
def patch_runner_for_custom_models() -> None:
    original_generate_models = Runner._generate_models

    def patched_generate_models(self, env, cfg):
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' defined in cfg")
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True

            if not separate:
                raise ValueError("Shared models not supported")

            for role in models_cfg:
                model_class_name = models_cfg[role].get("class")
                if not model_class_name:
                    raise ValueError(f"No 'class' in 'models:{role}'")
                del models_cfg[role]["class"]

                if "." in model_class_name:
                    parts = model_class_name.split(".")
                    module = _CUSTOM_MODEL_MODULES.get(parts[0])
                    if module is not None:
                        model_class = getattr(module, parts[1])
                    else:
                        raise ValueError(f"Module '{parts[0]}' not found")
                else:
                    model_class = self._component(model_class_name)

                models[agent_id][role] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    **self._process_cfg(models_cfg[role]),
                )
        return models

    Runner._generate_models = patched_generate_models


patch_runner_for_custom_models()


# ============================================================================
# Test configurations
# ============================================================================
TEST_CONFIGS: list[dict] = [
    {
        "name": "Stage14 baseline",
        "static": 25, "dynamic": 10, "mode": "mixed",
        "desc": "重現訓練配置 (25S + 10D mixed) — 應該達到 ~71% SR",
    },
    {
        "name": "Pure dynamic 5D",
        "static": 0, "dynamic": 5, "mode": "dynamic",
        "desc": "純動態少量 — agent 應能輕鬆處理（如果學會了動態避障）",
    },
    {
        "name": "Pure dynamic 10D",
        "static": 0, "dynamic": 10, "mode": "dynamic",
        "desc": "與 Stage 14 動態數量相同 — 真實動態避障測試",
    },
    {
        "name": "Pure dynamic 15D",
        "static": 0, "dynamic": 15, "mode": "dynamic",
        "desc": "中等密度純動態",
    },
    {
        "name": "Pure dynamic 25D",
        "static": 0, "dynamic": 25, "mode": "dynamic",
        "desc": "與 Stage 14 靜態數量相同 — 高密度純動態",
    },
    {
        "name": "Pure dynamic 35D",
        "static": 0, "dynamic": 35, "mode": "dynamic",
        "desc": "與 Stage 14 總數相同 — 極端壓力測試",
    },
]


# ============================================================================
# Helpers
# ============================================================================
def _find_default_checkpoint() -> str | None:
    """v17 best_agent.pt 預設路徑。"""
    candidates = [
        "logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-AC/"
        "rw_groundv8_openendedv1__seed1_nowalls_diag_v17/best_model/best_agent.pt",
        "logs/skrl/Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-AC/"
        "rw_groundv8_openendedv1__seed1_nowalls_diag_v17/checkpoints/best_agent.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _apply_obstacle_config(raw_env, cfg: dict) -> None:
    """更新 randomize_obstacles_startup 的參數，下次 reset 生效。"""
    n_s = cfg["static"]
    n_d = cfg["dynamic"]
    mode = cfg["mode"]

    if mode == "mixed":
        ratios = {"empty_ratio": 0.0, "static_ratio": 0.0,
                  "dynamic_ratio": 0.0, "mixed_ratio": 1.0}
    elif mode == "dynamic":
        ratios = {"empty_ratio": 0.0, "static_ratio": 0.0,
                  "dynamic_ratio": 1.0, "mixed_ratio": 0.0}
    elif mode == "static":
        ratios = {"empty_ratio": 0.0, "static_ratio": 1.0,
                  "dynamic_ratio": 0.0, "mixed_ratio": 0.0}
    else:
        raise ValueError(f"unknown mode: {mode}")

    obs_params = {
        **ratios,
        "num_obstacles_static": n_s,
        "num_obstacles_dynamic": n_d,
    }

    evt = raw_env.event_manager
    for name in ["randomize_obstacles", "randomize_obstacles_startup"]:
        try:
            ec = evt.get_term_cfg(name)
            ec.params.update(obs_params)
            evt.set_term_cfg(name, ec)
        except Exception:
            continue

    # 同步 env 內部的 _num_obstacles 計數
    raw_env._num_obstacles = n_s + n_d
    if hasattr(raw_env, "_obstacle_cache"):
        del raw_env._obstacle_cache

    # 更新 goal_command
    try:
        cmd = raw_env.command_manager.get_term("goal_command")
        cmd.cfg.num_obstacles = n_s + n_d
    except Exception:
        pass


def _count_episodes(raw_env, counts: dict) -> None:
    """從 termination_manager 累積結果。"""
    try:
        tm = raw_env.termination_manager
        for name in tm._term_names:
            buf = tm.get_term(name)
            if buf is None:
                continue
            c = int(buf.sum().item())
            if c == 0:
                continue
            if "goal_reached" in name:
                counts["success"] += c
            elif "collision" in name:
                counts["collision"] += c
            elif "time_out" in name:
                counts["timeout"] += c
    except Exception:
        pass


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    # --- Load env config ---
    from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.cfg.charge_env_cfg_vlp16_curriculum import (
        ChargeNavigationEnvCfgVLP16CurriculumNavRL,
    )
    env_cfg = ChargeNavigationEnvCfgVLP16CurriculumNavRL()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # 禁用課程，我們手動切換 config
    env_cfg.curriculum = None

    # 對齊訓練：no walls
    if args_cli.no_walls:
        wall_evt = getattr(env_cfg.events, "randomize_wall_positions", None)
        if wall_evt is not None:
            wall_evt.params["min_walls"] = 0
            wall_evt.params["max_walls"] = 0

    # Episode 長度
    env_cfg.episode_length_s = args_cli.episode_length_s

    # Goal 距離 (使用更新後的 6m 設定)
    env_cfg.commands.goal_command.ranges.distance = (2.0, 6.0)
    env_cfg.commands.goal_command.ranges.angle = (-math.pi, math.pi)
    env_cfg.commands.goal_command.num_goals = 1
    env_cfg.commands.goal_command.wall_safe_margin = 1.0
    env_cfg.commands.goal_command.num_obstacles = TEST_CONFIGS[0]["static"] + TEST_CONFIGS[0]["dynamic"]

    # 套用第一個 config 作為初始 env 創建參數
    init_cfg = TEST_CONFIGS[0]
    if init_cfg["mode"] == "mixed":
        init_ratios = {"empty_ratio": 0.0, "static_ratio": 0.0,
                       "dynamic_ratio": 0.0, "mixed_ratio": 1.0}
    elif init_cfg["mode"] == "dynamic":
        init_ratios = {"empty_ratio": 0.0, "static_ratio": 0.0,
                       "dynamic_ratio": 1.0, "mixed_ratio": 0.0}
    else:
        init_ratios = {"empty_ratio": 0.0, "static_ratio": 1.0,
                       "dynamic_ratio": 0.0, "mixed_ratio": 0.0}

    for evt_attr in ["randomize_obstacles", "randomize_obstacles_startup"]:
        evt_term = getattr(env_cfg.events, evt_attr, None)
        if evt_term is not None:
            evt_term.params.update({
                **init_ratios,
                "num_obstacles_static": init_cfg["static"],
                "num_obstacles_dynamic": init_cfg["dynamic"],
            })

    # --- Load agent yaml ---
    agent_yaml = Path(__file__).parent.parent.parent.parent / (
        "source/isaaclab_tasks/isaaclab_tasks/manager_based/"
        "locomotion/velocity/config/charge_skrl/agents/skrl_ppo_cfg_vlp16.yaml"
    )
    with open(agent_yaml) as f:
        agent_cfg = yaml.safe_load(f)
    agent_cfg["agent"]["experiment"]["directory"] = "logs/skrl/play"
    agent_cfg["agent"]["experiment"]["experiment_name"] = "dynamic_test"
    agent_cfg["trainer"]["timesteps"] = 0
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # --- Create env ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    # --- Create agent ---
    runner = Runner(env, agent_cfg)

    # --- Resolve checkpoint ---
    checkpoint_path = args_cli.checkpoint or _find_default_checkpoint()
    if checkpoint_path is None:
        print("[ERROR] No checkpoint found. Specify --checkpoint <path>.")
        env.close()
        simulation_app.close()
        return

    checkpoint_path = retrieve_file_path(checkpoint_path)
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        env.close()
        simulation_app.close()
        return

    runner.agent.load(checkpoint_path)
    print(f"[OK] Loaded checkpoint: {checkpoint_path}")
    runner.agent.set_running_mode("eval")

    # --- Deterministic mode ---
    if args_cli.deterministic:
        import types

        def _deterministic_act(self_policy, inputs, role=""):
            net_output, outputs = self_policy.compute(inputs, role)
            nvec = self_policy.action_space.nvec.tolist()
            splits = torch.split(net_output, nvec, dim=-1)
            actions = torch.stack([s.argmax(dim=-1) for s in splits], dim=-1)
            dists = [torch.distributions.Categorical(logits=s) for s in splits]
            log_prob = torch.stack(
                [d.log_prob(a) for d, a in zip(dists, torch.unbind(actions, dim=-1))],
                dim=-1,
            ).sum(dim=-1, keepdim=True)
            outputs["net_output"] = net_output
            return actions, log_prob, outputs

        runner.agent.policy.act = types.MethodType(_deterministic_act, runner.agent.policy)
        print("[INFO] Deterministic mode (argmax) enabled")

    # --- Get raw env for config updates ---
    raw_env = env.unwrapped
    while hasattr(raw_env, "unwrapped") and raw_env is not raw_env.unwrapped:
        raw_env = raw_env.unwrapped

    # ====================================================================
    # Run all test configurations sequentially
    # ====================================================================
    print(f"\n{'=' * 76}")
    print(f"  動態障礙物避障能力測試 — v17 best checkpoint")
    print(f"{'=' * 76}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Envs:        {args_cli.num_envs}")
    print(f"  Steps/test:  {args_cli.steps_per_test}")
    print(f"  Episode len: {args_cli.episode_length_s}s")
    print(f"  Goal dist:   2.0 - 6.0m")
    print(f"  Walls:       {'off (no_walls)' if args_cli.no_walls else 'on'}")
    print(f"  Determ:      {args_cli.deterministic}")
    print(f"{'=' * 76}\n")

    results: list[dict] = []
    global_start = time.time()

    for i, cfg in enumerate(TEST_CONFIGS):
        print(f"\n{'─' * 76}")
        print(f"  Test {i + 1}/{len(TEST_CONFIGS)}: {cfg['name']}")
        print(f"  {cfg['desc']}")
        print(f"  Config: {cfg['static']}S + {cfg['dynamic']}D ({cfg['mode']} mode)")
        print(f"{'─' * 76}")

        # 套用 config 後 reset env
        _apply_obstacle_config(raw_env, cfg)
        obs, _ = env.reset()

        counts = {"success": 0, "collision": 0, "timeout": 0}
        test_start = time.time()

        for step in range(args_cli.steps_per_test):
            with torch.no_grad():
                actions = runner.agent.act(obs, timestep=0, timesteps=0)[0]
            obs, _reward, _term, _trunc, _infos = env.step(actions)
            _count_episodes(raw_env, counts)

        elapsed = time.time() - test_start
        total_ep = sum(counts.values())
        sr = counts["success"] / total_ep * 100 if total_ep > 0 else 0.0
        cr = counts["collision"] / total_ep * 100 if total_ep > 0 else 0.0
        tr = counts["timeout"] / total_ep * 100 if total_ep > 0 else 0.0

        results.append({
            "name": cfg["name"],
            "static": cfg["static"],
            "dynamic": cfg["dynamic"],
            "mode": cfg["mode"],
            "episodes": total_ep,
            "success": counts["success"],
            "collision": counts["collision"],
            "timeout": counts["timeout"],
            "sr": sr,
            "cr": cr,
            "tr": tr,
            "elapsed": elapsed,
        })

        print(f"  → Ep={total_ep} | SR={sr:.1f}% | CR={cr:.1f}% | TO={tr:.1f}% | {elapsed:.1f}s")

    # ====================================================================
    # Summary table
    # ====================================================================
    total_elapsed = time.time() - global_start
    print(f"\n{'=' * 76}")
    print(f"  測試總結 — Total time: {total_elapsed:.0f}s")
    print(f"{'=' * 76}")
    print(f"  {'Config':<22} {'Mode':<8} {'S+D':<8} {'Ep':>4} {'SR':>7} {'CR':>7} {'TO':>7}")
    print(f"  {'─' * 70}")
    for r in results:
        sd = f"{r['static']}+{r['dynamic']}"
        print(f"  {r['name']:<22} {r['mode']:<8} {sd:<8} {r['episodes']:>4} "
              f"{r['sr']:>6.1f}% {r['cr']:>6.1f}% {r['tr']:>6.1f}%")
    print(f"{'=' * 76}\n")

    # ====================================================================
    # Diagnostic conclusion
    # ====================================================================
    baseline_sr = results[0]["sr"]
    pure_dynamic_results = [r for r in results if r["mode"] == "dynamic"]

    print(f"{'=' * 76}")
    print(f"  診斷結論")
    print(f"{'=' * 76}")
    print(f"  Baseline (Stage 14 mixed) SR = {baseline_sr:.1f}%")
    print()
    print(f"  純動態場景表現：")
    for r in pure_dynamic_results:
        delta = r["sr"] - baseline_sr
        symbol = "→" if abs(delta) < 5 else ("↑" if delta > 0 else "↓")
        print(f"    {r['dynamic']:>2}D: SR={r['sr']:>5.1f}%  CR={r['cr']:>5.1f}%  "
              f"({symbol} {delta:+.1f}% vs baseline)")
    print()

    # 判斷
    pure_10d = next((r for r in results if r["name"] == "Pure dynamic 10D"), None)
    if pure_10d:
        sr_10d = pure_10d["sr"]
        cr_10d = pure_10d["cr"]
        if sr_10d >= baseline_sr - 5 and cr_10d <= 5:
            print(f"  ✅ 結論：agent 真的學會了動態避障")
            print(f"     pure 10D SR ({sr_10d:.1f}%) ≈ baseline ({baseline_sr:.1f}%)")
            print(f"     CR={cr_10d:.1f}% 仍然很低")
        elif sr_10d < baseline_sr - 15 or cr_10d > 15:
            print(f"  ❌ 結論：agent 未真正學會動態避障")
            print(f"     pure 10D SR ({sr_10d:.1f}%) 遠低於 baseline ({baseline_sr:.1f}%)")
            print(f"     CR={cr_10d:.1f}% 顯示碰撞大幅增加")
            print()
            print(f"  代表 Stage 14 訓練的高 SR 主要來自：")
            print(f"  1. 靜態障礙物已佔據大部分空間（25/35 = 71%）")
            print(f"  2. agent 學會的是 'avoid static patterns + freeze on detect'")
            print(f"  3. 對少量動態障礙的處理是「運氣 + 速度太慢撞不到」")
            print(f"  4. v17 訓練分數 71.8% 高估了真實能力")
        else:
            print(f"  ⚠ 結論：部分學會動態避障，但不完全")
            print(f"     pure 10D SR ({sr_10d:.1f}%) 比 baseline ({baseline_sr:.1f}%) 低 "
                  f"{baseline_sr - sr_10d:.1f}%")
    print(f"{'=' * 76}\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
