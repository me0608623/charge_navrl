"""Goal-Obstacle 聯動課程學習 — v10 (NavRL + TO-aware)

v9 → v10 變更：
  1. Stage 4 拆分為 4a/4b，緩衝 Stage 3→4 的難度斷崖
  2. 升降級條件加入 TO（timeout rate），直接偵測「不動策略」

5 階段定義：
  Phase 1: 0 障礙物 → 純導航（建立 V > 0）
  Phase 2: 少量障礙物 → 死亡機制啟動
  Phase 3: 密集靜態+少量動態 → 避障為必經手段
  Phase 4a: 中等動態 → 過渡階段（5 static + 5 dynamic）
  Phase 4b: 密集動態 → 終極挑戰（5 static + 8 dynamic）

獎勵函數在所有階段完全不變。課程只改變環境參數。
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ============================================================================
# 5 階段定義 — 只有環境參數，沒有獎勵權重
# ============================================================================
STAGES = {
    # ------------------------------------------------------------------
    # Phase 1: 密集探索 — 建立 V(s) > 0
    #   大量 goal + 近距離 → 頻繁觸發成功獎勵
    #   無障礙物 → agent 快速學會「到達目標 = 高期望值」
    # ------------------------------------------------------------------
    1: {
        "num_goals": 8,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": 0,
        "num_obstacles_dynamic": 0,
        "empty_ratio": 1.00,
        "static_ratio": 0.00,
        "dynamic_ratio": 0.00,
        # 升級：SR > 72% AND TO < 30%
        "upgrade_sr": 0.72,
        "upgrade_max_cr": 1.0,
        "upgrade_max_to": 0.30,
        "downgrade_sr": 0.0,
        "downgrade_min_cr": 1.0,
        "downgrade_min_to": 1.0,
    },
    # ------------------------------------------------------------------
    # Phase 2: 死亡機制啟動 — 障礙物出現
    #   agent 已知「到達目標 = 高獎勵」(V > 0)
    #   現在碰撞 = 死亡 = 失去所有未來獎勵
    #   agent 被迫學習：「避障是獲取核心獎勵的前提」
    # ------------------------------------------------------------------
    2: {
        "num_goals": 3,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": 3,
        "num_obstacles_dynamic": 2,
        "empty_ratio": 0.55,
        "static_ratio": 0.30,
        "dynamic_ratio": 0.15,
        # 升級：SR > 65% AND CR < 40% AND TO < 35%
        "upgrade_sr": 0.65,
        "upgrade_max_cr": 0.40,
        "upgrade_max_to": 0.35,
        "downgrade_sr": 0.20,
        "downgrade_min_cr": 0.75,
        "downgrade_min_to": 0.70,
    },
    # ------------------------------------------------------------------
    # Phase 3: 密集避障 — 避障成為必經手段
    #   障礙物密度大幅提高 → 直衝策略大機率死亡
    #   agent 必須學會繞路才能存活並到達目標
    # ------------------------------------------------------------------
    3: {
        "num_goals": 2,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": 5,
        "num_obstacles_dynamic": 3,
        "empty_ratio": 0.25,
        "static_ratio": 0.40,
        "dynamic_ratio": 0.35,
        # 升級：SR > 60% AND CR < 40% AND TO < 40%
        "upgrade_sr": 0.60,
        "upgrade_max_cr": 0.40,
        "upgrade_max_to": 0.40,
        "downgrade_sr": 0.20,
        "downgrade_min_cr": 0.65,
        "downgrade_min_to": 0.65,
    },
    # ------------------------------------------------------------------
    # Phase 4a: 中等動態 — Stage 3→4b 的過渡
    #   動態障礙物增加到 5（從 3），dynamic ratio 40%
    #   緩衝難度跳躍，讓 agent 逐步適應動態環境
    # ------------------------------------------------------------------
    4: {
        "num_goals": 1,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": 5,
        "num_obstacles_dynamic": 5,
        "empty_ratio": 0.20,
        "static_ratio": 0.40,
        "dynamic_ratio": 0.40,
        # 升級：SR > 55% AND CR < 35% AND TO < 40%
        "upgrade_sr": 0.55,
        "upgrade_max_cr": 0.35,
        "upgrade_max_to": 0.40,
        "downgrade_sr": 0.15,
        "downgrade_min_cr": 0.60,
        "downgrade_min_to": 0.60,
    },
    # ------------------------------------------------------------------
    # Phase 4b: 終極挑戰 — 密集動態障礙物
    #   最終部署條件：1 goal + 8 動態障礙物 + 50% dynamic
    # ------------------------------------------------------------------
    5: {
        "num_goals": 1,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": 5,
        "num_obstacles_dynamic": 8,
        "empty_ratio": 0.15,
        "static_ratio": 0.35,
        "dynamic_ratio": 0.50,
        # 不升級（最終階段）
        "upgrade_sr": 1.0,
        "upgrade_max_cr": 0.0,
        "upgrade_max_to": 0.0,
        "downgrade_sr": 0.15,
        "downgrade_min_cr": 0.60,
        "downgrade_min_to": 0.60,
    },
}

MAX_STAGE = max(STAGES.keys())


def goal_obstacle_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    window_size: int = 2000,
    min_stage_episodes: int = 5000,
    initial_stage: int = 1,
) -> dict[str, float]:
    """v10 課程學習 — TO-aware 升降級。

    v9 → v10 變更：
    - Stage 4 拆為 4a(stage=4) / 4b(stage=5)
    - 升級條件新增 TO < threshold
    - 降級條件新增 TO > threshold
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
            f"[Curriculum v10] 初始化 — Phase {_stage_label(initial_stage)}: "
            f"{_phase_name(initial_stage)}\n"
            f"  num_envs={n}  |  窗口={effective_window}  |  "
            f"最低停留={effective_min} episodes\n"
            f"  Goals={s['num_goals']}  距離={s['goal_distance']}m  "
            f"障礙物={s['num_obstacles_static']}靜/{s['num_obstacles_dynamic']}動\n"
            f"  獎勵：固定不變\n"
            f"  升級: SR>{s['upgrade_sr']:.0%}"
            f"{'  CR<' + format(s['upgrade_max_cr'], '.0%') if s['upgrade_max_cr'] < 1.0 else ''}"
            f"  TO<{s['upgrade_max_to']:.0%}\n"
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
        timeout_rate = sum(1 for x in window if x == 0) / total
    else:
        success_rate = 0.0
        collision_rate = 0.0
        timeout_rate = 0.0

    current_stage = state["stage"]
    stage_cfg = STAGES[current_stage]

    can_transition = (
        len(window) >= effective_window
        and state["stage_episodes"] >= state["min_stage_episodes"]
    )

    if can_transition:
        up_sr = stage_cfg["upgrade_sr"]
        up_cr = stage_cfg["upgrade_max_cr"]
        up_to = stage_cfg["upgrade_max_to"]
        down_sr = stage_cfg["downgrade_sr"]
        down_cr = stage_cfg["downgrade_min_cr"]
        down_to = stage_cfg["downgrade_min_to"]

        # 升級：SR > threshold AND CR < threshold AND TO < threshold
        if (success_rate > up_sr
                and collision_rate < up_cr
                and timeout_rate < up_to
                and current_stage < MAX_STAGE):
            old = current_stage
            current_stage += 1
            state["stage"] = current_stage
            state["outcome_window"].clear()
            state["stage_episodes"] = 0
            state["stage_transitions"] += 1
            _apply_stage(env, current_stage)
            s = STAGES[current_stage]
            cr_info = (f" CR={collision_rate:.1%}<{up_cr:.0%}"
                       if up_cr < 1.0 else "")
            print(
                f"\n{'='*70}\n"
                f"[Curriculum v10] ▲ Phase {_stage_label(old)} → "
                f"Phase {_stage_label(current_stage)}: "
                f"{_phase_name(current_stage)}\n"
                f"  觸發: SR={success_rate:.1%}>{up_sr:.0%}"
                f"{cr_info} TO={timeout_rate:.1%}<{up_to:.0%}\n"
                f"  (累計 {state['total_episodes']} ep, "
                f"第 {state['stage_transitions']} 次轉換)\n"
                f"  Goals={s['num_goals']}  距離={s['goal_distance']}m  "
                f"障礙物={s['num_obstacles_static']}靜/{s['num_obstacles_dynamic']}動\n"
                f"  獎勵：固定不變\n"
                f"  下一階段: SR>{s['upgrade_sr']:.0%}"
                f"{'  CR<' + format(s['upgrade_max_cr'], '.0%') if s['upgrade_max_cr'] < 1.0 else ''}"
                f"  TO<{s['upgrade_max_to']:.0%}\n"
                f"{'='*70}",
                flush=True,
            )
            stage_cfg = s

        # 降級：SR 太低 OR CR 太高 OR TO 太高
        elif current_stage > 1:
            reason = None
            if success_rate < down_sr:
                reason = f"SR={success_rate:.1%}<{down_sr:.0%}"
            elif collision_rate > down_cr:
                reason = f"CR={collision_rate:.1%}>{down_cr:.0%}"
            elif timeout_rate > down_to:
                reason = f"TO={timeout_rate:.1%}>{down_to:.0%}"

            if reason is not None:
                old = current_stage
                current_stage -= 1
                state["stage"] = current_stage
                state["outcome_window"].clear()
                state["stage_episodes"] = 0
                state["stage_transitions"] += 1
                _apply_stage(env, current_stage)
                s = STAGES[current_stage]
                print(
                    f"\n{'='*70}\n"
                    f"[Curriculum v10] ▼ Phase {_stage_label(old)} → "
                    f"Phase {_stage_label(current_stage)}: "
                    f"{_phase_name(current_stage)}\n"
                    f"  觸發: {reason}  "
                    f"(SR={success_rate:.1%} CR={collision_rate:.1%} "
                    f"TO={timeout_rate:.1%})\n"
                    f"{'='*70}",
                    flush=True,
                )
                stage_cfg = s

    return {
        "stage": float(current_stage),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "num_goals": float(stage_cfg["num_goals"]),
        "num_episodes": float(state["total_episodes"]),
        "stage_episodes": float(state["stage_episodes"]),
        "window_fill": float(len(window)) / float(effective_window),
    }


def _stage_label(stage: int) -> str:
    """Stage 數字 → 顯示標籤（4→4a, 5→4b）"""
    return {1: "1", 2: "2", 3: "3", 4: "4a", 5: "4b"}[stage]


def _phase_name(stage: int) -> str:
    return {
        1: "密集探索（建立 V>0）",
        2: "死亡機制啟動（避障萌芽）",
        3: "密集避障（避障為必經手段）",
        4: "中等動態（過渡階段）",
        5: "終極挑戰（動態密集障礙）",
    }[stage]


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
    """套用指定階段的環境參數。

    只改環境（命令 + 障礙物），不改獎勵權重。
    """
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

    # --- 3. 不修改獎勵權重 ---
    # 獎勵在所有階段保持固定


__all__ = ["goal_obstacle_curriculum"]
