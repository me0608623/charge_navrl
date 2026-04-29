"""MARL 7-Phase Auto Curriculum — eval-based promotion

7 phase 設計：
  - 每 eval_interval iterations 做固定 eval
  - 用最近 patience 次 eval 的 EMA 判斷升 phase
  - 升 phase 條件：sr >= threshold AND coll <= threshold
  - min_iters / max_iters 包住每個 phase
  - 不看 aux loss，只看 eval sr + collision
  - 不訓練 goal，只訓練 spot (cars)

Usage:
    curriculum = MARLCurriculum("auto_7phase")
    ...
    # 每個 training iteration:
    curriculum.record_events(goals, obs_coll, car_coll)
    metrics = curriculum.update(iteration, entropy=ent, value_loss=vl)
    if metrics["stage_changed"]:
        curriculum.apply_to_env(env)
"""

from __future__ import annotations

from collections import deque
from typing import Any

from ..curriculum.goal_obstacle_curriculum import CURRICULUM_CONFIGS


# ============================================================================
# 7-Phase table with eval-based gates
# ============================================================================
AUTO_7PHASE_TABLE: list[dict[str, Any]] = [
    # ---------------------------------------------------------------
    # 對齊 WD train_rnn_car.py Phase 1-7 (lines 168-325)
    # All obstacles are obs-agent-controlled (learned policy)
    # num_obstacles_dynamic = WD num_obstacle
    # num_obstacles_static = 0 (no scripted obstacles)
    # WD 共用預設: num_wall=1, wall_length=6, num_stairs=1
    # 我們無 stairs；wall 用 min_walls=max_walls 對齊固定值
    # ---------------------------------------------------------------

    # WD P1: 6 spot, 20 goals, 3 obs, 1 wall(6.0), 1 stairs
    {
        "name": "P1-basic",
        "num_goals": 20, "num_obstacles_static": 0, "num_obstacles_dynamic": 3,
        "min_walls": 1, "max_walls": 1, "gamma": 0.990,
        "episode_length_s": 45.0, "empty_ratio": 0.80,
        "sr_threshold": 0.05, "coll_threshold": 25.0,
        "spot_penalty_hit": -5.0, "spot_reward_get_goal": 40.0,
        "ent_coeff_linear": 0.10, "ent_coeff_angular": 0.375,
        "obstacle_speed_rate": 0.8, "num_virtual_spots": 0,
        "goal_speed_rate": 0.7,
    },
    # WD P2: 3 spot, 16 goals, 3 obs, 1 wall(6.0), 1 stairs
    {
        "name": "P2-avoid",
        "num_goals": 16, "num_obstacles_static": 0, "num_obstacles_dynamic": 3,
        "min_walls": 1, "max_walls": 1, "gamma": 0.992,
        "episode_length_s": 51.0, "empty_ratio": 0.80,
        "sr_threshold": 0.08, "coll_threshold": 20.0,
        "spot_penalty_hit": -8.0, "spot_reward_get_goal": 40.0,
        "ent_coeff_linear": 0.30, "ent_coeff_angular": 0.375,
        "obstacle_speed_rate": 0.85, "num_virtual_spots": 0,
        "goal_speed_rate": 0.7,
    },
    # WD P3: 3 spot, 1 goal, 2 obs, 1 wall(6.0), 1 stairs
    {
        "name": "P3-precise",
        "num_goals": 1, "num_obstacles_static": 0, "num_obstacles_dynamic": 2,
        "min_walls": 1, "max_walls": 1, "gamma": 0.994,
        "episode_length_s": 56.0, "empty_ratio": 0.74,
        "sr_threshold": 0.12, "coll_threshold": 16.0,
        "spot_penalty_hit": -20.0, "spot_reward_get_goal": 40.0,
        "ent_coeff_linear": 0.30, "ent_coeff_angular": 0.375,
        "obstacle_speed_rate": 0.85, "num_virtual_spots": 0,
        "goal_speed_rate": 0.7,
    },
    # WD P4: 3 spot, 1 goal, 2 obs, 1 wall(4.5), 1 stairs
    {
        "name": "P4-short-wall",
        "num_goals": 1, "num_obstacles_static": 0, "num_obstacles_dynamic": 2,
        "min_walls": 1, "max_walls": 1, "gamma": 0.995,
        "episode_length_s": 61.0, "empty_ratio": 0.61,
        "sr_threshold": 0.15, "coll_threshold": 14.0,
        "spot_penalty_hit": -40.0, "spot_reward_get_goal": 40.0,
        "ent_coeff_linear": 0.30, "ent_coeff_angular": 0.375,
        "obstacle_speed_rate": 0.85, "num_virtual_spots": 0,
        "goal_speed_rate": 0.7,
        "wall_length": 4.5,
    },
    # WD P5: 2 spot, 1 goal, 2 obs, 1 wall(5.0), 1 stairs
    {
        "name": "P5-complex",
        "num_goals": 1, "num_obstacles_static": 0, "num_obstacles_dynamic": 2,
        "min_walls": 1, "max_walls": 1, "gamma": 0.996,
        "episode_length_s": 67.0, "empty_ratio": 0.48,
        "sr_threshold": 0.18, "coll_threshold": 12.0,
        "spot_penalty_hit": -80.0, "spot_reward_get_goal": 40.0,
        "ent_coeff_linear": 0.30, "ent_coeff_angular": 0.375,
        "obstacle_speed_rate": 0.85, "num_virtual_spots": 0,
        "goal_speed_rate": 0.7,
        "wall_length": 5.0,
    },
    # WD P6: 3 spot, 1 goal, 10 obs, 1 wall(6.0), 1 stairs
    {
        "name": "P6-dense",
        "num_goals": 1, "num_obstacles_static": 0, "num_obstacles_dynamic": 10,
        "min_walls": 1, "max_walls": 1, "gamma": 0.997,
        "episode_length_s": 72.0, "empty_ratio": 0.35,
        "sr_threshold": 0.22, "coll_threshold": 10.0,
        "spot_penalty_hit": -120.0, "spot_reward_get_goal": 40.0,
        "ent_coeff_linear": 0.30, "ent_coeff_angular": 0.375,
        "obstacle_speed_rate": 1.15, "num_virtual_spots": 0,
        "goal_speed_rate": 0.7,
    },
    # WD P7: 2 spot, 1 goal, 10 obs, 2 wall(3.5), 1 stairs — final
    {
        "name": "P7-ultimate",
        "num_goals": 1, "num_obstacles_static": 0, "num_obstacles_dynamic": 10,
        "min_walls": 2, "max_walls": 2, "gamma": 0.998,
        "episode_length_s": 90.0, "empty_ratio": 0.22,
        "sr_threshold": 1.0, "coll_threshold": 0.0,  # never promote
        "spot_penalty_hit": -200.0, "spot_reward_get_goal": 40.0,
        "ent_coeff_linear": 0.30, "ent_coeff_angular": 0.375,
        "obstacle_speed_rate": 1.15, "num_virtual_spots": 0,
        "goal_speed_rate": 0.7,
        "wall_length": 3.5,
    },
]


class MARLCurriculum:
    """7-phase eval-based auto curriculum for DirectMARLChargeEnv."""

    def __init__(
        self,
        version: str = "auto_7phase",
        initial_stage: int = 1,
        eval_interval: int = 20,
        patience: int = 3,
        min_iters_per_phase: int = 200,
        max_iters_per_phase: int = 18000,
        entropy_dead_threshold: float = 0.10,
        value_loss_explode: float = 1e6,
        **kwargs,
    ):
        # Support both old config-based and new auto mode
        if version in CURRICULUM_CONFIGS:
            self._legacy = True
            self._init_legacy(version, initial_stage, **kwargs)
            return

        self._legacy = False
        self.phases = AUTO_7PHASE_TABLE
        self.max_stage = len(self.phases)
        self.stage = min(initial_stage, self.max_stage)

        # Eval scheduling
        self.eval_interval = eval_interval
        self.patience = patience
        self.min_iters = min_iters_per_phase
        self.max_iters = max_iters_per_phase

        # Health check thresholds
        self.entropy_dead = entropy_dead_threshold
        self.vl_explode = value_loss_explode

        # Per-phase state
        self.phase_iters = 0
        self.stage_transitions = 0

        # Eval history: deque of (sr, coll_per_ep) from fixed evals
        self._eval_history: deque[tuple[float, float]] = deque(maxlen=patience)

        # Rolling event accumulators (between evals)
        self._goals_acc = 0
        self._coll_acc = 0
        self._car_coll_acc = 0
        self._ep_count_acc = 0
        self._timeout_acc = 0  # actual episode completions (timeout events)

        # Latest eval metrics (for logging)
        self._last_eval_sr = 0.0
        self._last_eval_coll = 0.0

    # ------------------------------------------------------------------
    # Legacy support (backward compat with CURRICULUM_CONFIGS)
    # ------------------------------------------------------------------
    def _init_legacy(self, version: str, initial_stage: int, **kwargs):
        """Old-style MARLCurriculum init for CURRICULUM_CONFIGS."""
        self.config = CURRICULUM_CONFIGS[version]
        self.stages_legacy = self.config["stages"]
        self.max_stage = len(self.stages_legacy)
        self.pass_required = self.config.get("upgrade_pass_required", 5)
        self.clear_on_promote = self.config.get("clear_window_on_promote", True)
        self.stage = initial_stage
        window_size = kwargs.get("window_size", 50)
        self.window: deque[tuple[float, float, float]] = deque(maxlen=max(window_size, 10))
        self.min_events = kwargs.get("min_events", 10)
        self.total_events = 0
        self.stage_events = 0
        self.stage_iters = 0
        self.upgrade_pass_count = 0
        self.stage_transitions = 0
        self._prev_stage = initial_stage

    @property
    def current_stage_cfg(self) -> dict:
        if self._legacy:
            return self.stages_legacy[self.stage - 1]
        return self.phases[self.stage - 1]

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------
    def record_events(
        self,
        goals_reached: int,
        obstacle_collisions: int,
        car_collisions: int,
        timeouts: int = 0,
    ):
        if self._legacy:
            self._record_events_legacy(goals_reached, obstacle_collisions, car_collisions)
            return

        self._goals_acc += goals_reached
        self._coll_acc += obstacle_collisions
        self._car_coll_acc += car_collisions
        self._timeout_acc += timeouts
        total = goals_reached + obstacle_collisions + car_collisions
        if total > 0:
            self._ep_count_acc += total

    # ------------------------------------------------------------------
    # Main update — called every iteration
    # ------------------------------------------------------------------
    def update(
        self,
        iteration: int,
        entropy: float = 1.0,
        value_loss: float = 0.0,
    ) -> dict[str, Any]:
        if self._legacy:
            return self._update_legacy(iteration)

        self.phase_iters += 1
        stage_changed = False
        cfg = self.current_stage_cfg

        # --- Periodic eval ---
        is_eval_iter = (self.phase_iters % self.eval_interval == 0)
        if is_eval_iter and self._ep_count_acc > 0:
            # SR = fraction of car-lives ending with goal (WD-aligned)
            sr = self._goals_acc / max(self._ep_count_acc, 1)
            # Coll per episode = total collisions / timeout episodes
            # (each timeout = one complete episode for all active cars)
            coll = self._coll_acc + self._car_coll_acc
            coll_per_ep = coll / max(self._timeout_acc, 1)

            self._eval_history.append((sr, coll_per_ep))
            self._last_eval_sr = sr
            self._last_eval_coll = coll_per_ep

            # Reset accumulators
            self._goals_acc = 0
            self._coll_acc = 0
            self._car_coll_acc = 0
            self._ep_count_acc = 0
            self._timeout_acc = 0

        # --- Promotion check (only at eval points) ---
        if is_eval_iter and len(self._eval_history) >= self.patience and self.stage < self.max_stage:
            # EMA of last `patience` evals
            avg_sr = sum(e[0] for e in self._eval_history) / len(self._eval_history)
            avg_coll = sum(e[1] for e in self._eval_history) / len(self._eval_history)

            ready = self.phase_iters >= self.min_iters

            # Health checks
            entropy_alive = entropy > self.entropy_dead
            vl_ok = not (value_loss != value_loss or value_loss > self.vl_explode)  # NaN or explode

            promote = (
                ready
                and avg_sr >= cfg["sr_threshold"]
                and avg_coll <= cfg["coll_threshold"]
                and entropy_alive
                and vl_ok
            )

            if promote:
                stage_changed = True
                print(f"[Curriculum] ↑ Phase {self.stage}→{self.stage + 1} "
                      f"(avg_sr={avg_sr:.3f}>={cfg['sr_threshold']}, "
                      f"avg_coll={avg_coll:.1f}<={cfg['coll_threshold']}) "
                      f"after {self.phase_iters} iters")
                self.stage += 1
                self.stage_transitions += 1
                self.phase_iters = 0
                self._eval_history.clear()

        # --- Max iters force promotion ---
        if not stage_changed and self.phase_iters >= self.max_iters:
            if self.stage < self.max_stage:
                print(f"[Curriculum] ⚠ Phase {self.stage} hit max_iters={self.max_iters}, "
                      f"force promoting to Phase {self.stage + 1} "
                      f"(sr={self._last_eval_sr:.3f}, coll={self._last_eval_coll:.1f})")
                self.stage += 1
                self.stage_transitions += 1
                self.phase_iters = 0
                self._eval_history.clear()
                stage_changed = True
            else:
                print(f"[Curriculum] ⚠ Final phase {self.stage} hit max_iters={self.max_iters}. "
                      f"Training continues but no further promotion possible.")

        new_cfg = self.current_stage_cfg
        return {
            "stage": self.stage,
            "stage_name": new_cfg["name"],
            "stage_changed": stage_changed,
            "stage_transitions": self.stage_transitions,
            "success_rate": self._last_eval_sr,
            "collision_rate": 0.0,  # compat
            "car_collision_rate": 0.0,
            "window_size": len(self._eval_history),
            # Stage parameters
            "num_goals": new_cfg["num_goals"],
            "num_obstacles_static": new_cfg["num_obstacles_static"],
            "num_obstacles_dynamic": new_cfg["num_obstacles_dynamic"],
            "min_walls": new_cfg["min_walls"],
            "max_walls": new_cfg["max_walls"],
            "gamma": new_cfg["gamma"],
            "episode_length_s": new_cfg["episode_length_s"],
            "empty_ratio": new_cfg["empty_ratio"],
            "upgrade_sr_target": new_cfg["sr_threshold"],
            "upgrade_pass": 0,
            # WD compat
            "ent_coeff_linear": new_cfg.get("ent_coeff_linear", 0.30),
            "ent_coeff_angular": new_cfg.get("ent_coeff_angular", 0.375),
            "obstacle_speed_rate": new_cfg.get("obstacle_speed_rate", 0.8),
            "spot_penalty_hit": new_cfg.get("spot_penalty_hit", -5.0),
            "spot_reward_get_goal": new_cfg.get("spot_reward_get_goal", 40.0),
            # Eval metrics
            "eval_sr": self._last_eval_sr,
            "eval_coll": self._last_eval_coll,
            "phase_iters": self.phase_iters,
        }

    # ------------------------------------------------------------------
    # Apply to env (shared for both modes)
    # ------------------------------------------------------------------
    def apply_to_env(self, env) -> None:
        """Apply current stage parameters to DirectMARLChargeEnv."""
        cfg = self.current_stage_cfg

        env.cfg.num_static_obstacles = cfg["num_obstacles_static"]
        env.cfg.num_dynamic_obstacles = cfg["num_obstacles_dynamic"]
        env.cfg.obs_count_randomize = True

        new_ep_s = cfg["episode_length_s"]
        env.cfg.episode_length_s = new_ep_s

        speed_rate = cfg.get("obstacle_speed_rate", 0.8)
        env.cfg.obstacle_speed_range = (
            round(0.3 * speed_rate, 2),
            round(1.2 * speed_rate, 2),
        )

        env.cfg.obs_size_rand = cfg.get("obs_size_rand", 0.0)
        env.cfg.scene_bound_rand = cfg.get("scene_bound_rand", 0.0)

        # Wall config (WD: num_wall fixed per phase, wall_length varies)
        if hasattr(env.cfg, "min_walls"):
            env.cfg.min_walls = cfg.get("min_walls", 1)
            env.cfg.max_walls = cfg.get("max_walls", 1)
        if "wall_length" in cfg and hasattr(env.cfg, "wall_length"):
            env.cfg.wall_length = cfg["wall_length"]

        # Goal count
        if hasattr(env.cfg, "num_goals"):
            env.cfg.num_goals = cfg.get("num_goals", 1)

        penalty = cfg.get("spot_penalty_hit", -5.0)
        env.cfg.reward_collision = penalty
        env.cfg.reward_car_collision = penalty * 0.5

        goal_reward = cfg.get("spot_reward_get_goal", 40.0)
        env.cfg.reward_reaching_goal = goal_reward

        n_vs = cfg.get("num_virtual_spots", 0)
        new_active = min(1 + n_vs, env.cfg.num_cars)
        env.n_active_cars = new_active

        print(f"[Curriculum] Applied Phase {self.stage}: "
              f"n_active={new_active}, "
              f"obs_agent={cfg['num_obstacles_dynamic']}, "
              f"goals={cfg.get('num_goals', '?')}, "
              f"walls={cfg.get('min_walls', 1)}, "
              f"ep_len={new_ep_s}s, penalty={penalty}")

    # ------------------------------------------------------------------
    # Legacy methods (for backward compat with old curriculum configs)
    # ------------------------------------------------------------------
    def _record_events_legacy(self, goals, obs_coll, car_coll):
        n = goals + obs_coll + car_coll
        if n > 0:
            sr = goals / n
            cr = obs_coll / n
            ccr = car_coll / n
        else:
            sr, cr, ccr = 0.0, 0.0, 0.0
        self.window.append((sr, cr, ccr))
        self.total_events += n
        self.stage_events += n

    def _update_legacy(self, iteration: int) -> dict[str, Any]:
        self.stage_iters += 1
        if len(self.window) == 0:
            rates = {"success_rate": 0.0, "collision_rate": 0.0, "car_collision_rate": 0.0, "window_size": 0}
        else:
            n = len(self.window)
            rates = {
                "success_rate": sum(t[0] for t in self.window) / n,
                "collision_rate": sum(t[1] for t in self.window) / n,
                "car_collision_rate": sum(t[2] for t in self.window) / n,
                "window_size": n,
            }

        cfg = self.stages_legacy[self.stage - 1]
        stage_changed = False
        sr = rates["success_rate"]
        cr = rates["collision_rate"] + rates["car_collision_rate"]

        ready = (
            len(self.window) >= self.min_events
            and self.stage_iters >= cfg.get("min_stage_updates", 50)
        )

        if ready and self.stage < self.max_stage:
            upgrade_ok = sr >= cfg["upgrade_sr"] and cr <= cfg["upgrade_max_cr"]
            if upgrade_ok:
                self.upgrade_pass_count += 1
                if self.upgrade_pass_count >= self.pass_required:
                    self.stage += 1
                    stage_changed = True
                    self.stage_transitions += 1
                    self.upgrade_pass_count = 0
                    self.stage_events = 0
                    self.stage_iters = 0
                    if self.clear_on_promote:
                        self.window.clear()
                    print(f"[Curriculum] ↑ PROMOTED to Stage {self.stage}/{self.max_stage} "
                          f"({self.stages_legacy[self.stage - 1]['name']})")
            else:
                self.upgrade_pass_count = 0

        new_cfg = self.stages_legacy[self.stage - 1]
        return {
            "stage": self.stage,
            "stage_name": new_cfg["name"],
            "stage_changed": stage_changed,
            "stage_transitions": self.stage_transitions,
            **rates,
            "num_goals": new_cfg["num_goals"],
            "num_obstacles_static": new_cfg["num_obstacles_static"],
            "num_obstacles_dynamic": new_cfg["num_obstacles_dynamic"],
            "min_walls": new_cfg["min_walls"],
            "max_walls": new_cfg["max_walls"],
            "gamma": new_cfg["gamma"],
            "episode_length_s": new_cfg["episode_length_s"],
            "empty_ratio": new_cfg["empty_ratio"],
            "upgrade_sr_target": new_cfg["upgrade_sr"],
            "upgrade_pass": getattr(self, "upgrade_pass_count", 0),
            "ent_coeff_linear": new_cfg.get("ent_coeff_linear", 0.30),
            "ent_coeff_angular": new_cfg.get("ent_coeff_angular", 0.375),
            "obstacle_speed_rate": new_cfg.get("obstacle_speed_rate", 0.8),
            "spot_penalty_hit": new_cfg.get("spot_penalty_hit", -5.0),
            "spot_reward_get_goal": new_cfg.get("spot_reward_get_goal", 40.0),
        }
