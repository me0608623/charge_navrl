"""Goal-Obstacle 聯動課程學習 — Multi-Phase RL

教學理念：分階段教技能，不同時教所有東西。
  Phase 1: 密集探索 → 學會「走到目標」
  Phase 2: 稀疏導航 → 學會「遠距離穩定導航」
  Phase 3: 安全避障 → 在已有導航能力上，學會「避開障礙物」
  Phase 4: 終極挑戰 → 密集動態障礙 + 最嚴懲罰

核心設計：
  - Phase 1/2 只教導航，碰撞懲罰極輕（不干擾學習方向感）
  - Phase 3/4 大幅提高碰撞懲罰權重（agent 已會導航，專注學避障）
  - 所有階段統一 80% SR 門檻升級
  - Phase 3/4 額外要求 CR < 閾值才能升級
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ============================================================================
# 4 階段定義
# ============================================================================
STAGES = {
    # ------------------------------------------------------------------
    # Phase 1: 密集探索 — 學會走到目標
    #   大量 goal + 近距離 → 頻繁觸發成功獎勵
    #   幾乎無障礙物，碰撞懲罰極輕
    # ------------------------------------------------------------------
    1: {
        "num_goals": 8,
        "goal_distance": (2.0, 5.0),
        "num_obstacles_static": 0,
        "num_obstacles_dynamic": 0,
        "empty_ratio": 1.00,
        "static_ratio": 0.00,
        "dynamic_ratio": 0.00,
        # 獎勵權重（Phase 1/2: 導航為主，碰撞輕罰）
        "collision_terminal_weight": -30.0,
        "near_obstacle_weight": -1.0,
        # 升級：SR > 80%（無 CR 約束，Phase 1 無障礙物）
        "upgrade_sr": 0.80,
        "upgrade_max_cr": 1.0,       # 不限
        "downgrade_sr": 0.0,         # 不降級
        "downgrade_min_cr": 1.0,     # 不限
    },
    # ------------------------------------------------------------------
    # Phase 2: 稀疏導航 — 擴大探索範圍，學會遠距離導航
    #   目標變少 + 距離拉遠 → 不能靠運氣或短距離得分
    #   少量障礙物開始出現，但碰撞懲罰仍輕
    # ------------------------------------------------------------------
    2: {
        "num_goals": 3,
        "goal_distance": (4.0, 8.0),
        "num_obstacles_static": 2,
        "num_obstacles_dynamic": 1,
        "empty_ratio": 0.80,
        "static_ratio": 0.15,
        "dynamic_ratio": 0.05,
        # 獎勵權重（仍以導航為主）
        "collision_terminal_weight": -50.0,
        "near_obstacle_weight": -2.0,
        # 升級：SR > 80%（CR 寬鬆，agent 還在學導航）
        "upgrade_sr": 0.80,
        "upgrade_max_cr": 0.30,      # 允許 30% 碰撞（還沒專注學避障）
        "downgrade_sr": 0.25,
        "downgrade_min_cr": 0.60,
    },
    # ------------------------------------------------------------------
    # Phase 3: 安全避障 — 已會導航，專注學避障
    #   碰撞懲罰大幅提高（-200 → 讓 agent 感受痛）
    #   obstacle penalty 也拉高（靠近就扣分）
    # ------------------------------------------------------------------
    3: {
        "num_goals": 2,
        "goal_distance": (3.0, 8.0),
        "num_obstacles_static": 5,
        "num_obstacles_dynamic": 3,
        "empty_ratio": 0.25,
        "static_ratio": 0.40,
        "dynamic_ratio": 0.35,
        # 獎勵權重（避障為主）
        "collision_terminal_weight": -200.0,
        "near_obstacle_weight": -5.0,
        # 升級：SR > 80% AND CR < 20%
        "upgrade_sr": 0.80,
        "upgrade_max_cr": 0.20,
        "downgrade_sr": 0.25,
        "downgrade_min_cr": 0.50,
    },
    # ------------------------------------------------------------------
    # Phase 4: 終極挑戰 — 密集動態障礙 + 最嚴懲罰
    #   最終部署條件：1 goal + 大量動態障礙物
    #   碰撞懲罰拉到最高
    # ------------------------------------------------------------------
    4: {
        "num_goals": 1,
        "goal_distance": (3.0, 8.0),
        "num_obstacles_static": 5,
        "num_obstacles_dynamic": 8,
        "empty_ratio": 0.15,
        "static_ratio": 0.35,
        "dynamic_ratio": 0.50,
        # 獎勵權重（最嚴避障）
        "collision_terminal_weight": -300.0,
        "near_obstacle_weight": -8.0,
        # 不升級（最終階段）
        "upgrade_sr": 1.0,
        "upgrade_max_cr": 0.0,
        # 降級：SR < 20% 或 CR > 50%
        "downgrade_sr": 0.20,
        "downgrade_min_cr": 0.50,
    },
}


def goal_obstacle_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    window_size: int = 2000,
    min_stage_episodes: int = 5000,
    initial_stage: int = 1,
) -> dict[str, float]:
    """Goal-Obstacle 聯動課程學習。

    升級條件（全部同時滿足）：
      1. 窗口填滿 + 最低停留 episodes
      2. SR > 80%（所有階段統一）
      3. Phase 3/4 額外要求 CR < max_collision_rate

    降級條件（Phase 2+ 任一觸發）：
      SR < downgrade_sr  OR  CR > downgrade_min_cr
    """
    if not hasattr(env, "_goal_obs_curriculum"):
        n = env.num_envs
        effective_window = max(window_size, n * 5)
        effective_min = max(min_stage_episodes, n * 8)
        env._goal_obs_curriculum = {
            "stage": initial_stage,
            "outcome_window": deque(maxlen=effective_window),
            "effective_window_size": effective_window,
            "min_stage_episodes": effective_min,
            "total_episodes": 0,
            "stage_episodes": 0,
            "stage_transitions": 0,
        }
        _apply_stage(env, initial_stage)
        s = STAGES[initial_stage]
        print(
            f"\n{'='*70}\n"
            f"[Curriculum] 初始化 — Phase {initial_stage}: "
            f"{_phase_name(initial_stage)}\n"
            f"  num_envs={n}  |  窗口={effective_window}  |  "
            f"最低停留={effective_min} episodes\n"
            f"  Goals={s['num_goals']}  距離={s['goal_distance']}m  "
            f"障礙物={s['num_obstacles_static']}靜/{s['num_obstacles_dynamic']}動\n"
            f"  碰撞懲罰={s['collision_terminal_weight']}  "
            f"障礙物懲罰={s['near_obstacle_weight']}\n"
            f"  升級: SR>{s['upgrade_sr']:.0%}"
            f"{'  AND CR<' + format(s['upgrade_max_cr'], '.0%') if s['upgrade_max_cr'] < 1.0 else ''}\n"
            f"{'='*70}",
            flush=True,
        )

    state = env._goal_obs_curriculum

    if env_ids is not None and not isinstance(env_ids, slice):
        _collect_episode_results(env, env_ids, state)

    window = state["outcome_window"]
    effective_window = state["effective_window_size"]
    if len(window) >= 10:
        total = len(window)
        success_rate = sum(1 for x in window if x == 1) / total
        collision_rate = sum(1 for x in window if x == -1) / total
    else:
        success_rate = 0.0
        collision_rate = 0.0

    current_stage = state["stage"]
    stage_cfg = STAGES[current_stage]

    can_transition = (
        len(window) >= effective_window
        and state["stage_episodes"] >= state["min_stage_episodes"]
    )

    if can_transition:
        up_sr = stage_cfg["upgrade_sr"]
        up_cr = stage_cfg["upgrade_max_cr"]
        down_sr = stage_cfg["downgrade_sr"]
        down_cr = stage_cfg["downgrade_min_cr"]

        # 升級
        if (success_rate > up_sr and collision_rate < up_cr
                and current_stage < 4):
            old = current_stage
            current_stage += 1
            state["stage"] = current_stage
            state["outcome_window"].clear()
            state["stage_episodes"] = 0
            state["stage_transitions"] += 1
            _apply_stage(env, current_stage)
            s = STAGES[current_stage]
            cr_info = (f" AND CR={collision_rate:.1%}<{up_cr:.0%}"
                       if up_cr < 1.0 else "")
            print(
                f"\n{'='*70}\n"
                f"[Curriculum] ▲ Phase {old} → Phase {current_stage}: "
                f"{_phase_name(current_stage)}\n"
                f"  觸發: SR={success_rate:.1%}>{up_sr:.0%}{cr_info}\n"
                f"  (累計 {state['total_episodes']} ep, "
                f"第 {state['stage_transitions']} 次轉換)\n"
                f"  Goals={s['num_goals']}  距離={s['goal_distance']}m  "
                f"障礙物={s['num_obstacles_static']}靜/{s['num_obstacles_dynamic']}動\n"
                f"  碰撞懲罰: {STAGES[old]['collision_terminal_weight']} → "
                f"{s['collision_terminal_weight']}  "
                f"障礙物懲罰: {STAGES[old]['near_obstacle_weight']} → "
                f"{s['near_obstacle_weight']}\n"
                f"  下一階段: SR>{s['upgrade_sr']:.0%}"
                f"{'  AND CR<' + format(s['upgrade_max_cr'], '.0%') if s['upgrade_max_cr'] < 1.0 else ''}\n"
                f"{'='*70}",
                flush=True,
            )
            stage_cfg = s

        # 降級
        elif ((success_rate < down_sr or collision_rate > down_cr)
                and current_stage > 1):
            old = current_stage
            reason = (f"SR={success_rate:.1%}<{down_sr:.0%}"
                      if success_rate < down_sr
                      else f"CR={collision_rate:.1%}>{down_cr:.0%}")
            current_stage -= 1
            state["stage"] = current_stage
            state["outcome_window"].clear()
            state["stage_episodes"] = 0
            state["stage_transitions"] += 1
            _apply_stage(env, current_stage)
            s = STAGES[current_stage]
            print(
                f"\n{'='*70}\n"
                f"[Curriculum] ▼ Phase {old} → Phase {current_stage}: "
                f"{_phase_name(current_stage)}\n"
                f"  觸發: {reason}  (SR={success_rate:.1%} CR={collision_rate:.1%})\n"
                f"  碰撞懲罰: {STAGES[old]['collision_terminal_weight']} → "
                f"{s['collision_terminal_weight']}\n"
                f"{'='*70}",
                flush=True,
            )
            stage_cfg = s

    return {
        "stage": float(current_stage),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": max(0.0, 1.0 - success_rate - collision_rate),
        "num_goals": float(stage_cfg["num_goals"]),
        "collision_penalty": float(stage_cfg["collision_terminal_weight"]),
        "obstacle_penalty": float(stage_cfg["near_obstacle_weight"]),
        "num_episodes": float(state["total_episodes"]),
        "stage_episodes": float(state["stage_episodes"]),
        "window_fill": float(len(window)) / float(effective_window),
    }


def _phase_name(stage: int) -> str:
    return {1: "密集探索", 2: "稀疏導航", 3: "安全避障", 4: "終極挑戰"}[stage]


def _collect_episode_results(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    state: dict,
):
    """收集 per-env episode 結局：1=success, -1=collision, 0=timeout"""
    import torch

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if ids.numel() == 0:
        return

    try:
        tm = env.termination_manager
        goal_buf = None
        coll_buf = None
        for name in tm._term_names:
            if "goal_reached" in name and goal_buf is None:
                goal_buf = tm.get_term(name)
            elif "collision" in name and coll_buf is None:
                coll_buf = tm.get_term(name)

        for idx in ids:
            eid = idx.item() if isinstance(idx, torch.Tensor) else int(idx)
            if goal_buf is not None and goal_buf[eid].item():
                outcome = 1
            elif coll_buf is not None and coll_buf[eid].item():
                outcome = -1
            else:
                outcome = 0
            state["outcome_window"].append(outcome)
            state["total_episodes"] += 1
            state["stage_episodes"] += 1

    except Exception:
        n = ids.numel() if hasattr(ids, 'numel') else len(ids)
        for _ in range(n):
            state["outcome_window"].append(0)
            state["total_episodes"] += 1
            state["stage_episodes"] += 1


def _apply_stage(env: ManagerBasedRLEnv, stage: int):
    """套用指定階段的全部參數：命令、事件、獎勵權重。"""
    cfg = STAGES[stage]

    # --- 1. 更新 MultiGoalCommand ---
    try:
        cmd = env.command_manager.get_term("goal_command")
        cmd.cfg.num_goals = cfg["num_goals"]
        cmd.cfg.ranges.distance = cfg["goal_distance"]
    except Exception:
        pass

    # --- 2. 更新障礙物事件 ---
    try:
        evt = env.event_manager
        for name in ["randomize_obstacles", "randomize_obstacles_startup"]:
            try:
                ec = evt.get_term_cfg(name)
                ec.params["empty_ratio"] = cfg["empty_ratio"]
                ec.params["static_ratio"] = cfg["static_ratio"]
                ec.params["dynamic_ratio"] = cfg["dynamic_ratio"]
                ec.params["num_obstacles_static"] = cfg["num_obstacles_static"]
                ec.params["num_obstacles_dynamic"] = cfg["num_obstacles_dynamic"]
                evt.set_term_cfg(name, ec)
            except Exception:
                continue
    except Exception:
        pass

    # --- 3. 動態修改獎勵權重（核心：Phase 1/2 輕罰 → Phase 3/4 重罰）---
    try:
        rm = env.reward_manager
        # 碰撞終止懲罰
        try:
            rc = rm.get_term_cfg("collision_terminal")
            rc.weight = cfg["collision_terminal_weight"]
            rm.set_term_cfg("collision_terminal", rc)
        except Exception:
            pass
        # 障礙物接近懲罰
        try:
            rc = rm.get_term_cfg("near_obstacle_penalty")
            rc.weight = cfg["near_obstacle_weight"]
            rm.set_term_cfg("near_obstacle_penalty", rc)
        except Exception:
            pass
    except Exception:
        pass


__all__ = ["goal_obstacle_curriculum"]
