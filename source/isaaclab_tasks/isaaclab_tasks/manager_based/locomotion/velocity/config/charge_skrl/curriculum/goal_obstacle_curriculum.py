"""Goal-Obstacle 聯動課程學習 — v11 (穩定升降級機制)

v10 → v11 變更：
  1. 連續通過 K=3 次才升級（anti-fluke gate）
  2. 高階 Stage 門檻收緊（Stage 3→4a, 4a→4b）
  3. 最少 rollout cycles 門檻（min_stage_updates）
  4. Dynamic-only SR 子集門檻（Stage 3+ 升級用）

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
# 連續通過次數要求
# ============================================================================
UPGRADE_PASS_REQUIRED = 3

# ============================================================================
# 5 階段定義 — 只有環境參數，沒有獎勵權重
# ============================================================================
STAGES = {
    # ------------------------------------------------------------------
    # Phase 1: 密集探索 — 建立 V(s) > 0
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
        "upgrade_min_dyn_sr": 0.0,   # 不適用
        "min_stage_updates": 0,       # 不適用
        "downgrade_sr": 0.0,
        "downgrade_min_cr": 1.0,
        "downgrade_min_to": 1.0,
    },
    # ------------------------------------------------------------------
    # Phase 2: 死亡機制啟動 — 障礙物出現
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
        "upgrade_min_dyn_sr": 0.0,   # 不適用
        "min_stage_updates": 0,       # 不適用
        "downgrade_sr": 0.20,
        "downgrade_min_cr": 0.75,
        "downgrade_min_to": 0.70,
    },
    # ------------------------------------------------------------------
    # Phase 3: 密集避障 — 避障成為必經手段
    # ------------------------------------------------------------------
    3: {
        "num_goals": 2,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": 5,
        "num_obstacles_dynamic": 3,
        "empty_ratio": 0.25,
        "static_ratio": 0.40,
        "dynamic_ratio": 0.35,
        # 升級：SR > 65% AND CR < 35% AND TO < 30% AND dynSR > 40%
        "upgrade_sr": 0.65,
        "upgrade_max_cr": 0.35,
        "upgrade_max_to": 0.30,
        "upgrade_min_dyn_sr": 0.40,
        "min_stage_updates": 50,
        "downgrade_sr": 0.20,
        "downgrade_min_cr": 0.65,
        "downgrade_min_to": 0.65,
    },
    # ------------------------------------------------------------------
    # Phase 4a: 中等動態 — Stage 3→4b 的過渡
    # ------------------------------------------------------------------
    4: {
        "num_goals": 1,
        "goal_distance": (2.0, 13.0),
        "num_obstacles_static": 5,
        "num_obstacles_dynamic": 5,
        "empty_ratio": 0.20,
        "static_ratio": 0.40,
        "dynamic_ratio": 0.40,
        # 升級：SR > 60% AND CR < 30% AND TO < 25% AND dynSR > 45%
        "upgrade_sr": 0.60,
        "upgrade_max_cr": 0.30,
        "upgrade_max_to": 0.25,
        "upgrade_min_dyn_sr": 0.45,
        "min_stage_updates": 50,
        "downgrade_sr": 0.15,
        "downgrade_min_cr": 0.60,
        "downgrade_min_to": 0.60,
    },
    # ------------------------------------------------------------------
    # Phase 4b: 終極挑戰 — 密集動態障礙物
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
        "upgrade_min_dyn_sr": 0.0,
        "min_stage_updates": 0,
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
    """v11 課程學習 — 穩定升降級機制。

    v10 → v11 變更：
    - 連續通過 K=3 次才升級（anti-fluke gate）
    - 高階 Stage 門檻收緊
    - 最少 rollout cycles 門檻
    - Dynamic-only SR 子集門檻（Stage 3+）
    """
    if not hasattr(env, "_goal_obs_curriculum"):
        n = env.num_envs
        effective_window = max(window_size, n * 5)
        effective_min = max(min_stage_episodes, n * 8)
        env._goal_obs_curriculum = {
            "stage": initial_stage,
            "outcome_window": deque(maxlen=effective_window),  # (outcome, env_type) tuples
            "effective_window_size": effective_window,
            "min_stage_episodes": effective_min,
            "total_episodes": 0,
            "stage_episodes": 0,
            "stage_transitions": 0,
            "upgrade_pass_count": 0,
        }
        _apply_stage(env, initial_stage)
        s = STAGES[initial_stage]
        print(
            f"\n{'='*70}\n"
            f"[Curriculum v11] 初始化 — Phase {_stage_label(initial_stage)}: "
            f"{_phase_name(initial_stage)}\n"
            f"  num_envs={n}  |  窗口={effective_window}  |  "
            f"最低停留={effective_min} episodes\n"
            f"  Goals={s['num_goals']}  距離={s['goal_distance']}m  "
            f"障礙物={s['num_obstacles_static']}靜/{s['num_obstacles_dynamic']}動\n"
            f"  獎勵：固定不變\n"
            f"  升級: SR>{s['upgrade_sr']:.0%}"
            f"{'  CR<' + format(s['upgrade_max_cr'], '.0%') if s['upgrade_max_cr'] < 1.0 else ''}"
            f"  TO<{s['upgrade_max_to']:.0%}"
            f"  需連續通過 {UPGRADE_PASS_REQUIRED} 次\n"
            f"{'='*70}",
            flush=True,
        )

    state = env._goal_obs_curriculum

    if env_ids is not None and not isinstance(env_ids, slice):
        _collect_episode_results(env, env_ids, state)

    window = state["outcome_window"]
    effective_window = state["effective_window_size"]

    # 計算整體 rates
    if len(window) >= 10:
        total = len(window)
        success_rate = sum(1 for o, _ in window if o == 1) / total
        collision_rate = sum(1 for o, _ in window if o == -1) / total
        timeout_rate = sum(1 for o, _ in window if o == 0) / total
    else:
        success_rate = 0.0
        collision_rate = 0.0
        timeout_rate = 0.0

    # 計算 dynamic-only SR
    dyn_outcomes = [(o, t) for o, t in window if t == 2]
    dynamic_sr = sum(1 for o, _ in dyn_outcomes if o == 1) / max(len(dyn_outcomes), 1)

    current_stage = state["stage"]
    stage_cfg = STAGES[current_stage]

    # rollout cycles 近似
    approx_rollout_cycles = state["stage_episodes"] // max(env.num_envs, 1)
    min_stage_updates = stage_cfg.get("min_stage_updates", 0)

    can_transition = (
        len(window) >= effective_window
        and state["stage_episodes"] >= state["min_stage_episodes"]
        and approx_rollout_cycles >= min_stage_updates
    )

    if can_transition:
        up_sr = stage_cfg["upgrade_sr"]
        up_cr = stage_cfg["upgrade_max_cr"]
        up_to = stage_cfg["upgrade_max_to"]
        up_dyn_sr = stage_cfg.get("upgrade_min_dyn_sr", 0.0)
        down_sr = stage_cfg["downgrade_sr"]
        down_cr = stage_cfg["downgrade_min_cr"]
        down_to = stage_cfg["downgrade_min_to"]

        # 原始計數（用於 debug log）
        total = len(window)
        success_count = sum(1 for o, _ in window if o == 1)
        collision_count = sum(1 for o, _ in window if o == -1)
        timeout_count = sum(1 for o, _ in window if o == 0)

        def _debug_log(direction: str, old_stage: int, new_stage: int):
            """升降級前後印出完整 debug log"""
            sr_pass = f"{'PASS' if success_rate > up_sr else 'FAIL'}"
            cr_pass = f"{'PASS' if collision_rate < up_cr else 'FAIL'}" if up_cr < 1.0 else "N/A"
            to_pass = f"{'PASS' if timeout_rate < up_to else 'FAIL'}"
            dyn_pass = f"{'PASS' if dynamic_sr > up_dyn_sr else 'FAIL'}" if up_dyn_sr > 0 else "N/A"
            s = STAGES[new_stage]
            print(
                f"\n{'='*70}\n"
                f"[Curriculum v11] {direction} Phase {_stage_label(old_stage)} → "
                f"Phase {_stage_label(new_stage)}: {_phase_name(new_stage)}\n"
                f"  --- raw counts (window={total}) ---\n"
                f"  success={success_count}  collision={collision_count}  timeout={timeout_count}\n"
                f"  --- rates (unrounded) ---\n"
                f"  SR={success_rate:.6f}  CR={collision_rate:.6f}  TO={timeout_rate:.6f}\n"
                f"  dynSR={dynamic_sr:.4f} ({len(dyn_outcomes)} dynamic episodes)\n"
                f"  --- upgrade conditions (strict > / <) ---\n"
                f"  SR={success_rate:.6f} > {up_sr} ? {sr_pass}\n"
                f"  CR={collision_rate:.6f} < {up_cr} ? {cr_pass}\n"
                f"  TO={timeout_rate:.6f} < {up_to} ? {to_pass}\n"
                f"  dynSR={dynamic_sr:.4f} > {up_dyn_sr} ? {dyn_pass}\n"
                f"  rollout_cycles={approx_rollout_cycles} (min={min_stage_updates})\n"
                f"  upgrade_pass_count={state['upgrade_pass_count']}\n"
                f"  --- meta ---\n"
                f"  累計 {state['total_episodes']} ep, "
                f"階段 {state['stage_episodes']} ep, "
                f"第 {state['stage_transitions']} 次轉換\n"
                f"  Goals={s['num_goals']}  距離={s['goal_distance']}m  "
                f"障礙物={s['num_obstacles_static']}靜/{s['num_obstacles_dynamic']}動\n"
                f"  下一升級: SR>{s['upgrade_sr']}"
                f"{'  CR<' + str(s['upgrade_max_cr']) if s['upgrade_max_cr'] < 1.0 else ''}"
                f"  TO<{s['upgrade_max_to']}"
                f"{'  dynSR>' + str(s.get('upgrade_min_dyn_sr', 0)) if s.get('upgrade_min_dyn_sr', 0) > 0 else ''}\n"
                f"{'='*70}",
                flush=True,
            )

        # 升級檢查：SR + CR + TO + dynamic SR
        sr_ok = success_rate > up_sr
        cr_ok = collision_rate < up_cr
        to_ok = timeout_rate < up_to
        dyn_ok = dynamic_sr > up_dyn_sr if (up_dyn_sr > 0 and len(dyn_outcomes) > 0) else True
        all_pass = sr_ok and cr_ok and to_ok and dyn_ok

        if all_pass and current_stage < MAX_STAGE:
            state["upgrade_pass_count"] += 1
            if state["upgrade_pass_count"] >= UPGRADE_PASS_REQUIRED:
                # 真正升級
                old = current_stage
                current_stage += 1
                state["stage"] = current_stage
                state["outcome_window"].clear()
                state["stage_episodes"] = 0
                state["stage_transitions"] += 1
                state["upgrade_pass_count"] = 0
                _apply_stage(env, current_stage)
                _debug_log(f"▲ ({state['upgrade_pass_count']}/{UPGRADE_PASS_REQUIRED} confirmed)", old, current_stage)
                stage_cfg = STAGES[current_stage]
            else:
                # 待確認
                print(
                    f"[Curriculum v11] 升級待確認 "
                    f"{state['upgrade_pass_count']}/{UPGRADE_PASS_REQUIRED} — "
                    f"SR={success_rate:.3f} CR={collision_rate:.3f} "
                    f"TO={timeout_rate:.3f} dynSR={dynamic_sr:.3f} "
                    f"rollout_cycles={approx_rollout_cycles}",
                    flush=True,
                )
        else:
            # 條件不通過 → 歸零連續計數
            if state["upgrade_pass_count"] > 0:
                print(
                    f"[Curriculum v11] 升級連續計數歸零 "
                    f"(was {state['upgrade_pass_count']}/{UPGRADE_PASS_REQUIRED}) — "
                    f"SR={success_rate:.3f} CR={collision_rate:.3f} "
                    f"TO={timeout_rate:.3f} dynSR={dynamic_sr:.3f}",
                    flush=True,
                )
            state["upgrade_pass_count"] = 0

            # 降級：SR 太低 OR CR 太高 OR TO 太高（一次即降）
            if current_stage > 1:
                reason = None
                if success_rate < down_sr:
                    reason = f"SR={success_rate:.6f} < {down_sr}"
                elif collision_rate > down_cr:
                    reason = f"CR={collision_rate:.6f} > {down_cr}"
                elif timeout_rate > down_to:
                    reason = f"TO={timeout_rate:.6f} > {down_to}"

                if reason is not None:
                    old = current_stage
                    current_stage -= 1
                    state["stage"] = current_stage
                    state["outcome_window"].clear()
                    state["stage_episodes"] = 0
                    state["stage_transitions"] += 1
                    state["upgrade_pass_count"] = 0
                    _apply_stage(env, current_stage)
                    _debug_log(f"▼ ({reason})", old, current_stage)
                    stage_cfg = STAGES[current_stage]

    return {
        "stage": float(current_stage),
        "success_rate": success_rate,
        "collision_rate": collision_rate,
        "timeout_rate": timeout_rate,
        "dynamic_sr": dynamic_sr,
        "num_goals": float(stage_cfg["num_goals"]),
        "num_episodes": float(state["total_episodes"]),
        "stage_episodes": float(state["stage_episodes"]),
        "window_fill": float(len(window)) / float(effective_window),
        "upgrade_pass_count": float(state["upgrade_pass_count"]),
        "approx_rollout_cycles": float(approx_rollout_cycles),
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
    """收集 per-env episode 結局：(outcome, env_type)

    outcome: 1=success, -1=collision, 0=timeout
    env_type: 0=empty, 1=static, 2=dynamic, -1=unknown
    """
    import torch

    if isinstance(env_ids, torch.Tensor):
        ids = env_ids
    else:
        ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    if ids.numel() == 0:
        return

    # 取得 env difficulty type
    difficulty = None
    if hasattr(env, '_env_difficulty') and env._env_difficulty is not None:
        difficulty = env._env_difficulty

    try:
        tm = env.termination_manager
        goal_buf = None
        coll_bufs = []
        for name in tm._term_names:
            if "goal_reached" in name and goal_buf is None:
                goal_buf = tm.get_term(name)
            elif "collision" in name:
                coll_bufs.append(tm.get_term(name))

        for idx in ids:
            eid = idx.item() if isinstance(idx, torch.Tensor) else int(idx)
            if goal_buf is not None and goal_buf[eid].item():
                outcome = 1
            elif any(buf[eid].item() for buf in coll_bufs):
                outcome = -1
            else:
                outcome = 0

            env_type = int(difficulty[eid].item()) if difficulty is not None else -1
            state["outcome_window"].append((outcome, env_type))
            state["total_episodes"] += 1
            state["stage_episodes"] += 1

    except Exception:
        n = ids.numel() if hasattr(ids, 'numel') else len(ids)
        for _ in range(n):
            state["outcome_window"].append((0, -1))
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
