#!/usr/bin/env python3
"""wandblook — W&B RL Training Analyzer for Charge-SKRL project.

A complete diagnostic tool that fetches W&B runs, extracts RL training metrics,
applies heuristic rules, and produces structured analysis reports.

Usage:
    # Single latest run
    python wandblook.py

    # Specific run by path
    python wandblook.py me0608623-none/charge_skrl/runs/ldvz34m4

    # Latest 5 runs with comparison
    python wandblook.py --latest 5

    # Filter by name pattern, last 7 days
    python wandblook.py --filter "abl1_*" --since-days 7

    # Custom metrics focus + CSV output
    python wandblook.py --latest 3 --metrics kl,entropy,reward_mean --csv

    # Save to reports/ directory
    python wandblook.py --latest 3 --output-dir reports/
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("wandblook")


# ══════════════════════════════════════════════════════════════════════════
#  Section 1: Configuration & Thresholds (centralized)
# ══════════════════════════════════════════════════════════════════════════

DEFAULT_ENTITY = "me0608623-none"
DEFAULT_PROJECT = "charge_skrl"
DEFAULT_REPORT_DIR = Path("/tmp")

# ── Diagnostic rule thresholds ────────────────────────────────────────────
# All heuristic thresholds are collected here for easy tuning.
# Each is labeled as [HEURISTIC] — these are empirical rules of thumb,
# NOT proven facts. Adjust based on your specific training setup.

@dataclass(frozen=True)
class DiagnosticThresholds:
    """Centralized thresholds for all heuristic diagnostic rules.

    [HEURISTIC] All values are empirical defaults for PPO with discrete
    actions in navigation tasks. They may need adjustment for different
    task domains or algorithm variants.
    """
    # ── PPO stability ──
    kl_warn: float = 0.05           # KL above this → update too aggressive
    kl_trend_ratio: float = 1.5     # recent/earlier KL ratio → trending up
    clip_fraction_warn: float = 0.3 # clip_fraction above this → updates clipped too often
    ratio_max_warn: float = 3.0     # importance ratio above this → extreme update

    # ── Entropy ──
    entropy_collapse_ratio: float = 0.5  # recent/earlier < this → premature collapse
    entropy_min_absolute: float = 0.1    # entropy below this → nearly deterministic

    # ── Advantage signal ──
    adv_std_weak: float = 0.01      # adv_raw_std below this → actor has weak signal
    adv_std_collapse_ratio: float = 0.3  # recent/earlier < this → advantage shrinking

    # ── Critic ──
    explained_var_bad: float = 0.0   # EV below 0 → worse than mean baseline
    explained_var_low: float = 0.3   # EV below this → critic underfitting

    # ── Grad norms ──
    grad_norm_explode: float = 100.0  # grad norm above this → likely exploding
    grad_norm_vanish: float = 1e-6    # grad norm below this → likely vanishing

    # ── Task performance ──
    success_stagnation_delta: float = 0.02  # change < this → stagnating
    success_stagnation_ceil: float = 0.5    # only flag if below this
    conservative_coll_drop: float = 0.8     # collision recent < earlier*this
    conservative_tout_rise: float = 1.2     # timeout recent > earlier*this

    # ── Ablation-specific ──
    freeze_ratio_warn: float = 0.3
    oscillation_warn: float = 0.4
    retreat_ratio_warn: float = 0.5
    stuck_count_warn: float = 50.0

    # ── Trend detection ──
    trend_up_threshold: float = 0.10    # >10% change → rising
    trend_down_threshold: float = -0.10 # <-10% change → falling
    trend_recent_ratio: float = 0.2     # last 20% vs first 80%
    min_points_for_trend: int = 4       # need at least N points for trend

    # ── Data sufficiency ──
    min_points_for_diagnosis: int = 10  # need at least N points per metric
    min_history_for_report: int = 20    # need at least N rows total


THRESHOLDS = DiagnosticThresholds()


# ══════════════════════════════════════════════════════════════════════════
#  Section 2: Field Alias Mappings
# ══════════════════════════════════════════════════════════════════════════

# Each canonical name maps to a list of possible W&B column names.
# Order matters: first match wins.

PPO_FIELD_ALIASES: dict[str, list[str]] = {
    # ── KL divergence ──
    # IMPORTANT: "diag/real_approx_kl" is the actual per-update KL in charge_skrl.
    # "Loss / KL divergence" is the SKRL built-in (often missing).
    "kl": [
        "diag/real_approx_kl",
        "kl", "approx_kl", "train/kl", "ppo/kl", "train/approx_kl",
        "Loss / KL divergence", "metrics/kl",
    ],
    # ── Clip fraction ──
    "clip_fraction": [
        "diag/real_clip_fraction",
        "clip_fraction", "train/clip_fraction", "ppo/clip_fraction",
        "Loss / Clip fraction",
    ],
    # ── Importance sampling ratio ──
    "ratio_max": [
        "diag/real_ratio_max",
        "ratio_max", "train/ratio_max", "ppo/ratio_max",
    ],
    "ratio_mean": [
        "diag/real_ratio_mean",
    ],
    "ratio_std": [
        "diag/real_ratio_std",
    ],
    # ── Grad norms ──
    "actor_grad_norm": [
        "train/actor_grad_norm",
        "actor_grad_norm", "grad_norm/actor",
        "Gradient / Actor gradient norm",
    ],
    "critic_grad_norm": [
        "train/critic_grad_norm",
        "critic_grad_norm", "grad_norm/critic",
        "Gradient / Critic gradient norm",
    ],
    # ── Entropy ──
    # In charge_skrl with MultiDiscrete actions:
    #   - "action/linear_entropy" and "action/angular_entropy" are the TRUE per-dim H(π)
    #     (range ~0–2.94 for 19 bins, max = ln(19) ≈ 2.944)
    #   - "train/module_entropy" is SKRL's internal entropy (can be negative, not raw H(π))
    #   - "Loss / Entropy loss" is -coef*H(π), always negative
    "entropy_linear": [
        "action/linear_entropy",
    ],
    "entropy_angular": [
        "action/angular_entropy",
    ],
    "entropy_module": [
        "train/module_entropy",
        "module_entropy", "entropy", "train/entropy", "ppo/entropy",
    ],
    "entropy_loss": [
        "Loss / Entropy loss", "entropy_loss",
    ],
    "entropy_state_code": [
        "train/module_entropy_state_code",
        "module_entropy_state_code", "entropy_state_code",
    ],
    # ── Advantage ──
    "adv_raw_mean": [
        "diag/advantage_raw_mean",
        "adv_raw_mean", "train/adv_raw_mean", "advantage_mean",
    ],
    "adv_raw_std": [
        "diag/advantage_raw_std",
        "adv_raw_std", "train/adv_raw_std", "advantage_std",
    ],
    "adv_mean": [
        "diag/advantage_mean",
    ],
    "adv_std": [
        "diag/advantage_std",
    ],
    # ── Losses ──
    "value_loss": [
        "Loss / Value loss",
        "value_loss", "train/value_loss", "critic_loss", "train/critic_loss",
    ],
    "policy_loss": [
        "Loss / Policy loss",
        "policy_loss", "train/policy_loss", "actor_loss", "train/actor_loss",
    ],
    # ── Value estimates ──
    "explained_variance": [
        "explained_variance", "train/explained_variance",
    ],
    "value_mean": [
        "diag/value_mean",
    ],
    "value_std": [
        "diag/value_std",
    ],
    "return_mean": [
        "diag/return_mean",
    ],
    "return_std": [
        "diag/return_std",
    ],
    # ── Learning rate ──
    "learning_rate": [
        "Learning / Learning rate",
        "learning_rate", "train/learning_rate", "lr",
    ],
    # ── Network health ──
    "erank_actor": [
        "diag/erank_actor",
    ],
    "erank_critic": [
        "diag/erank_critic",
    ],
    "update_ratio_ac": [
        "train/update_ratio_actor_over_critic",
    ],
}

TASK_FIELD_ALIASES: dict[str, list[str]] = {
    # ── Performance rates ──
    # charge_skrl uses "perf/*" for rolling rates, "Curriculum/*" for windowed rates
    "success_rate": [
        "perf/success_rate", "Curriculum / success_rate",
        "eval/success_rate_empty", "eval/success_rate",
        "success_rate", "task/success_rate", "episode/success_rate",
    ],
    "collision_rate": [
        "perf/collision_rate", "Curriculum / collision_rate",
        "eval/collision_rate_empty", "eval/collision_rate",
        "collision_rate", "task/collision_rate", "episode/collision_rate",
    ],
    "timeout_rate": [
        "perf/timeout_rate", "Curriculum / timeout_rate",
        "eval/timeout_rate_empty", "eval/timeout_rate",
        "timeout_rate", "task/timeout_rate", "episode/timeout_rate",
    ],
    # ── Reward ──
    "reward_mean": [
        "Reward / Total reward (mean)",
        "reward_mean", "episode/reward_mean", "rollout/ep_rew_mean",
    ],
    "reward_max": [
        "Reward / Total reward (max)",
    ],
    "reward_min": [
        "Reward / Total reward (min)",
    ],
    # ── Episode length ──
    "episode_length": [
        "perf/episode_length", "behavior/avg_episode_length",
        "Episode / Total timesteps (mean)",
        "episode_length", "episode/length_mean", "rollout/ep_len_mean",
    ],
    # ── Curriculum ──
    "stage": [
        "Curriculum / stage", "Curriculum / difficulty_level",
        "stage", "curriculum/stage", "curriculum/phase",
    ],
    "num_goals": [
        "Curriculum / num_goals",
    ],
    "num_obstacles_static": [
        "Curriculum / num_obstacles_static",
    ],
    "num_obstacles_dynamic": [
        "Curriculum / num_obstacles_dynamic",
    ],
    "total_episodes": [
        "perf/total_episodes", "Curriculum / num_episodes",
    ],
    # ── Termination breakdown ──
    "goal_reached": [
        "Termination / goal_reached",
        "goal_reached", "task/goal_reached",
    ],
    "wall_collision": [
        "Termination / wall_collision",
    ],
}

ABLATION_FIELDS: list[str] = [
    # charge_skrl uses "behavior/*" prefix for diagnostic behavior metrics
    "behavior/stuck_events",
    "behavior/freeze_ratio",
    "behavior/oscillation",
    "behavior/retreat_ratio",
    "behavior/progress_near_obstacle",
    "behavior/obstacle_distance_avg",
    "behavior/obstacle_distance_min",
    "behavior/speed_avg",
    "behavior/speed_near_obstacle",
    "behavior/danger_zone_ratio",
    "behavior/path_efficiency",
    "behavior/goal_velocity_near_obstacle",
    "behavior/front_clearance",
    # Legacy "ablation/*" prefix (older runs)
    "ablation/stuck_count",
    "ablation/freeze_ratio",
    "ablation/oscillation_score",
    "ablation/retreat_ratio",
    "ablation/progress_near_obs",
    "ablation/d_safe_mean",
    "ablation/shield_rate",
    # Reward space diagnostics
    "reward_space/collision_trigger_ratio",
    "reward_space/front_block_trigger_ratio",
    "reward_space/lidar_min_mean",
    # Diag context metrics (key ones)
    "diag/ctx_near_obs_fwd_adv_mean",
    "diag/ctx_near_obs_retreat_adv_mean",
    "diag/ctx_goal_front_fwd_adv_mean",
    "diag/ctx_goal_front_retreat_adv_mean",
    "diag/effective_horizon_gae",
]

CONFIG_KEYS_OF_INTEREST: list[str] = [
    "algorithm", "learning_rate", "learning_epochs", "gamma", "lambda", "lam",
    "entropy_coef", "value_loss_scale", "rollout_length", "rollouts",
    "num_envs", "reward_mode", "curriculum_version", "seed",
    "mini_batches", "discount_factor", "learning_rate_scheduler",
    "v_gate_mode", "progress_gate_mode", "use_gap_reward", "gap_reward_type",
    "gap_reward_weight", "use_safety_shield", "shield_mode",
    "clip_ratio", "grad_norm_clip", "max_grad_norm",
]


# ══════════════════════════════════════════════════════════════════════════
#  Section 3: Utility Functions
# ══════════════════════════════════════════════════════════════════════════

def resolve_field(row: dict[str, Any], aliases: list[str]) -> float | None:
    """Return first matching numeric value from a history row, or None."""
    for alias in aliases:
        val = row.get(alias)
        if val is None:
            continue
        try:
            f = float(val)
            # Guard against NaN/Inf which poison statistics
            if f != f or f == float("inf") or f == float("-inf"):
                continue
            return f
        except (ValueError, TypeError):
            continue
    return None


def safe_mean(values: list[float]) -> float | None:
    """Mean that returns None for empty lists."""
    if not values:
        return None
    return sum(values) / len(values)


def safe_std(values: list[float]) -> float | None:
    """Population std that returns None for < 2 values."""
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def trend_label(recent: float | None, earlier: float | None, t: DiagnosticThresholds = THRESHOLDS) -> str:
    """Classify trend direction based on relative change."""
    if recent is None or earlier is None:
        return "N/A"
    if abs(earlier) < 1e-9:
        if recent > 1e-9:
            return "↑ 上升"
        elif recent < -1e-9:
            return "↓ 下降"
        return "→ 持平"
    ratio = (recent - earlier) / abs(earlier)
    if ratio > t.trend_up_threshold:
        return "↑ 上升"
    elif ratio < t.trend_down_threshold:
        return "↓ 下降"
    return "→ 持平"


def fmt_val(v: float | None, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if abs(v) > 1e6:
        return f"{v:.2e}"
    return f"{v:.{decimals}f}"


def flatten_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested config dict, keeping leaf key names."""
    flat: dict[str, Any] = {}
    def _walk(d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _walk(v, full_key)
            else:
                flat[k] = v       # leaf key (short)
                flat[full_key] = v  # full path key
    _walk(cfg)
    return flat


# ══════════════════════════════════════════════════════════════════════════
#  Section 4: W&B Data Fetching (with error handling)
# ══════════════════════════════════════════════════════════════════════════

def check_wandb_auth() -> bool:
    """Verify W&B API key is available."""
    key = os.environ.get("WANDB_API_KEY", "")
    if key:
        return True
    # wandb also stores key in ~/.netrc or ~/.config/wandb/settings
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "api.wandb.ai" in netrc.read_text(errors="ignore"):
        return True
    wandb_settings = Path.home() / ".config" / "wandb" / "settings"
    if wandb_settings.exists():
        return True
    # Last resort: try the wandb dir
    wandb_dir = Path.home() / ".wandb"
    if wandb_dir.exists():
        return True
    return False


def create_api() -> Any:
    """Create W&B API client with proper error handling."""
    try:
        import wandb  # noqa: F811
    except ImportError:
        log.error("❌ wandb 套件未安裝。請執行: pip install wandb")
        sys.exit(1)

    if not check_wandb_auth():
        log.warning(
            "⚠ 未偵測到 W&B API key。如果接下來連線失敗，請執行:\n"
            "  export WANDB_API_KEY=<your-key>\n"
            "  或: wandb login"
        )

    try:
        api = wandb.Api(timeout=60)
        return api
    except Exception as e:
        log.error(f"❌ 無法連接 W&B API: {e}")
        sys.exit(1)


def fetch_runs(
    api: Any,
    entity: str,
    project: str,
    run_paths: list[str] | None = None,
    latest: int = 1,
    name_filter: str | None = None,
    since_days: int | None = None,
    state_filter: str | None = None,
) -> list[Any]:
    """Fetch runs from W&B API with comprehensive error handling.

    Args:
        api: wandb.Api instance
        entity: W&B entity (user or team)
        project: W&B project name
        run_paths: specific run paths to fetch (bypasses other filters)
        latest: number of recent runs to fetch
        name_filter: glob-style name pattern (e.g. "abl1_*")
        since_days: only fetch runs created within N days
        state_filter: filter by run state (e.g. "running", "finished", "crashed")
    """
    # ── Mode A: specific run paths ──
    if run_paths:
        runs = []
        for rp in run_paths:
            rp = rp.strip().rstrip("/")
            try:
                runs.append(api.run(rp))
                log.info(f"  ✓ Fetched: {rp}")
            except Exception as e:
                log.warning(f"  ✗ Could not fetch '{rp}': {e}")
        if not runs:
            log.error("❌ 所有指定的 run path 都無法取得。")
        return runs

    # ── Mode B: query by project ──
    project_path = f"{entity}/{project}"
    filters: dict[str, Any] = {}

    if name_filter:
        # Convert glob-style to regex
        regex = name_filter.replace("*", ".*").replace("?", ".")
        filters["display_name"] = {"$regex": regex}

    if since_days is not None and since_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        filters["created_at"] = {"$gte": cutoff.isoformat()}

    if state_filter:
        filters["state"] = state_filter

    try:
        runs = api.runs(
            project_path,
            filters=filters if filters else None,
            order="-created_at",
            per_page=min(latest, 50),
        )
        result = list(runs)[:latest]
        if not result:
            log.warning(
                f"⚠ 在 {project_path} 中未找到符合條件的 runs。\n"
                f"  Filters: {json.dumps(filters, default=str)}"
            )
        return result
    except Exception as e:
        err_str = str(e).lower()
        if "not found" in err_str or "404" in err_str:
            log.error(f"❌ Project 不存在: {project_path}")
        elif "permission" in err_str or "403" in err_str or "401" in err_str:
            log.error(f"❌ 無權限存取 {project_path}，請確認 API key 和 entity/project 設定。")
        else:
            log.error(f"❌ 查詢 runs 失敗: {e}")
        return []


def fetch_history(run: Any, max_samples: int = 2000) -> list[dict[str, Any]]:
    """Fetch run history using scan_history (memory-efficient).

    Falls back to run.history() if scan_history fails.
    Handles large histories by capping at max_samples rows.
    """
    rows: list[dict[str, Any]] = []

    # Strategy 1: scan_history (streaming, memory-efficient)
    try:
        for row in run.scan_history(page_size=500):
            rows.append(dict(row))
            if len(rows) >= max_samples:
                log.info(f"    (capped at {max_samples} rows)")
                break
        if rows:
            return rows
    except Exception as e:
        log.warning(f"  ⚠ scan_history failed for {run.id}: {e}")

    # Strategy 2: run.history() with sampling
    try:
        df = run.history(samples=min(max_samples, 500))
        rows = df.to_dict("records") if not df.empty else []
        return rows
    except Exception as e:
        log.warning(f"  ⚠ history() also failed for {run.id}: {e}")

    # Strategy 3: summary only (last resort)
    if run.summary:
        log.warning(f"  ⚠ 只取得 summary (無 history) for {run.id}")
        return [dict(run.summary)]

    log.warning(f"  ⚠ 完全無法取得 {run.id} 的 history 資料")
    return []


# ══════════════════════════════════════════════════════════════════════════
#  Section 5: Run Analysis Engine
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TrendRow:
    """One metric's trend analysis result."""
    metric: str
    latest: float | None
    mean: float | None
    std: float | None
    recent_mean: float | None
    earlier_mean: float | None
    trend: str
    min_val: float | None
    max_val: float | None
    count: int


class RunAnalysis:
    """Analyze a single W&B run's metrics, config, and health."""

    def __init__(self, run: Any, history: list[dict[str, Any]],
                 thresholds: DiagnosticThresholds = THRESHOLDS,
                 extra_metrics: list[str] | None = None):
        self.run = run
        self.history = history
        self.n = len(history)
        self.t = thresholds

        # ── Extract time series ──
        self.ppo_series: dict[str, list[float]] = {}
        self.task_series: dict[str, list[float]] = {}
        self.ablation_series: dict[str, list[float]] = {}
        self.extra_series: dict[str, list[float]] = {}

        for canonical, aliases in PPO_FIELD_ALIASES.items():
            vals = [v for row in history if (v := resolve_field(row, aliases)) is not None]
            if vals:
                self.ppo_series[canonical] = vals

        for canonical, aliases in TASK_FIELD_ALIASES.items():
            vals = [v for row in history if (v := resolve_field(row, aliases)) is not None]
            if vals:
                self.task_series[canonical] = vals

        for fld in ABLATION_FIELDS:
            vals = []
            for row in history:
                raw = row.get(fld)
                if raw is not None:
                    try:
                        f = float(raw)
                        if f == f:  # skip NaN
                            vals.append(f)
                    except (ValueError, TypeError):
                        pass
            if vals:
                self.ablation_series[fld] = vals

        # Extra user-requested metrics (pass-through, no alias mapping)
        if extra_metrics:
            for m in extra_metrics:
                # Check if it's already captured
                if m in self.ppo_series or m in self.task_series or m in self.ablation_series:
                    continue
                vals = []
                for row in history:
                    raw = row.get(m)
                    if raw is not None:
                        try:
                            f = float(raw)
                            if f == f:
                                vals.append(f)
                        except (ValueError, TypeError):
                            pass
                if vals:
                    self.extra_series[m] = vals

        # ── Detect which aliases actually matched (for debugging) ──
        self._matched_aliases: dict[str, str] = {}
        if history:
            sample = history[0]
            for canonical, aliases in {**PPO_FIELD_ALIASES, **TASK_FIELD_ALIASES}.items():
                for alias in aliases:
                    if alias in sample and sample[alias] is not None:
                        self._matched_aliases[canonical] = alias
                        break

    # ── Properties ──

    @property
    def meta(self) -> dict[str, Any]:
        r = self.run
        summary = dict(r.summary) if r.summary else {}
        return {
            "name": r.name,
            "id": r.id,
            "state": r.state,
            "created_at": str(r.created_at),
            "url": r.url,
            "tags": list(r.tags) if r.tags else [],
            "history_rows": self.n,
            "summary_keys": len(summary),
            "has_summary": bool(summary),
        }

    @property
    def config_snapshot(self) -> dict[str, Any]:
        cfg = self.run.config or {}
        flat = flatten_config(cfg)
        result: dict[str, Any] = {}
        for k in CONFIG_KEYS_OF_INTEREST:
            if k in flat:
                result[k] = flat[k]
        # Also capture ablation-related config keys
        for k, v in flat.items():
            if any(tok in k.lower() for tok in ["gate", "gap", "shield", "ablat", "curriculum"]):
                short_key = k.split(".")[-1] if "." in k else k
                result[short_key] = v
        return result

    # ── Trend computation ──

    def _segment_stats(self, series: list[float]) -> tuple[float | None, float | None, float | None]:
        """Split into earlier (80%) vs recent (20%) and return (overall_mean, earlier_mean, recent_mean)."""
        if len(series) < self.t.min_points_for_trend:
            return safe_mean(series), None, None
        split = max(1, int(len(series) * (1 - self.t.trend_recent_ratio)))
        return safe_mean(series), safe_mean(series[:split]), safe_mean(series[split:])

    def compute_trends(self, series_dict: dict[str, list[float]]) -> list[TrendRow]:
        rows = []
        for name, vals in series_dict.items():
            overall, earlier, recent = self._segment_stats(vals)
            rows.append(TrendRow(
                metric=name,
                latest=vals[-1] if vals else None,
                mean=overall,
                std=safe_std(vals),
                recent_mean=recent,
                earlier_mean=earlier,
                trend=trend_label(recent, earlier, self.t),
                min_val=min(vals) if vals else None,
                max_val=max(vals) if vals else None,
                count=len(vals),
            ))
        return rows

    # ── Diagnostic rules ──

    def diagnose(self) -> list[dict[str, str]]:
        """Apply heuristic rules and return structured findings.

        Each finding is a dict with:
            - level: "confirmed" | "suspicious" | "info"
            - message: human-readable description
            - rule: which rule triggered
        """
        findings: list[dict[str, str]] = []
        t = self.t
        ppo = self.ppo_series
        task = self.task_series

        def add(level: str, msg: str, rule: str) -> None:
            findings.append({"level": level, "message": msg, "rule": rule})

        # ── Rule 1: KL divergence ──────────────────────────────────
        if "kl" in ppo and len(ppo["kl"]) >= t.min_points_for_diagnosis:
            _, earlier, recent = self._segment_stats(ppo["kl"])
            if recent is not None and recent > t.kl_warn:
                add("confirmed", f"KL = {recent:.4f} > {t.kl_warn} — 策略更新幅度偏大", "kl_high")
            if recent and earlier and recent > earlier * t.kl_trend_ratio:
                add("suspicious",
                    f"KL 趨勢上升 ({earlier:.4f} → {recent:.4f}) — PPO 更新可能過猛，"
                    "考慮降低 learning_rate 或增加 mini_batches",
                    "kl_rising")

        # ── Rule 2: Clip fraction ──────────────────────────────────
        if "clip_fraction" in ppo and len(ppo["clip_fraction"]) >= t.min_points_for_diagnosis:
            recent_vals = ppo["clip_fraction"][-20:]
            recent_clip = safe_mean(recent_vals)
            if recent_clip and recent_clip > t.clip_fraction_warn:
                add("confirmed",
                    f"clip_fraction = {recent_clip:.3f} > {t.clip_fraction_warn} — 策略更新頻繁被 clip",
                    "clip_high")

        # ── Rule 3: Ratio max ──────────────────────────────────────
        if "ratio_max" in ppo and len(ppo["ratio_max"]) >= t.min_points_for_diagnosis:
            max_ratio = max(ppo["ratio_max"][-20:])
            if max_ratio > t.ratio_max_warn:
                add("suspicious",
                    f"ratio_max = {max_ratio:.2f} > {t.ratio_max_warn} — importance sampling ratio 過大",
                    "ratio_extreme")

        # ── Rule 4: Entropy collapse ───────────────────────────────
        # Use action-level entropy (entropy_linear / entropy_angular) which are true H(π).
        # For MultiDiscrete(19,19): max entropy = ln(19) ≈ 2.944 per dimension.
        # "entropy_module" / "entropy_loss" are SKRL internals and may be negative — skip them.
        entropy_key = None
        for ek in ("entropy_linear", "entropy_angular"):
            if ek in ppo and len(ppo[ek]) >= t.min_points_for_diagnosis:
                entropy_key = ek
                break
        if entropy_key:
            _, earlier, recent = self._segment_stats(ppo[entropy_key])
            if recent is not None and recent < t.entropy_min_absolute:
                add("confirmed",
                    f"{entropy_key} = {recent:.4f} < {t.entropy_min_absolute} — 策略幾乎確定性，exploration 極低",
                    "entropy_collapsed")
            elif recent and earlier and recent < earlier * t.entropy_collapse_ratio:
                success_rising = False
                if "success_rate" in task and len(task["success_rate"]) >= t.min_points_for_diagnosis:
                    _, se, sr = self._segment_stats(task["success_rate"])
                    if sr and se and sr > se * 1.1:
                        success_rising = True
                if not success_rising:
                    add("suspicious",
                        f"{entropy_key} 快速下降 ({earlier:.4f} → {recent:.4f}) 且 success 未同步上升 — "
                        "可能 premature policy collapse，考慮提高 entropy_coef",
                        "entropy_premature_collapse")

        # ── Rule 5: Advantage signal ───────────────────────────────
        if "adv_raw_std" in ppo and len(ppo["adv_raw_std"]) >= t.min_points_for_diagnosis:
            _, earlier, recent = self._segment_stats(ppo["adv_raw_std"])
            if recent is not None and recent < t.adv_std_weak:
                add("confirmed",
                    f"adv_raw_std = {recent:.5f} < {t.adv_std_weak} — actor 可用 advantage 訊號極弱",
                    "adv_weak")
            elif recent and earlier and recent < earlier * t.adv_std_collapse_ratio:
                add("suspicious",
                    f"adv_raw_std 持續縮小 ({earlier:.4f} → {recent:.4f}) — advantage 區分度下降",
                    "adv_shrinking")

        # ── Rule 6: Explained variance ─────────────────────────────
        if "explained_variance" in ppo and ppo["explained_variance"]:
            ev_latest = ppo["explained_variance"][-1]
            if ev_latest < t.explained_var_bad:
                add("confirmed",
                    f"explained_variance = {ev_latest:.3f} < 0 — critic 預測比 mean baseline 更差",
                    "ev_negative")
            elif ev_latest < t.explained_var_low:
                add("suspicious",
                    f"explained_variance = {ev_latest:.3f} < {t.explained_var_low} — critic fitting 不足",
                    "ev_low")

        # ── Rule 7: Gradient norms ─────────────────────────────────
        for gname in ("actor_grad_norm", "critic_grad_norm"):
            if gname in ppo and ppo[gname]:
                recent_max = max(ppo[gname][-20:])
                recent_min = min(ppo[gname][-20:])
                if recent_max > t.grad_norm_explode:
                    add("confirmed",
                        f"{gname} max = {recent_max:.1f} > {t.grad_norm_explode} — gradient 爆炸風險",
                        f"{gname}_explode")
                if recent_min < t.grad_norm_vanish and recent_max < t.grad_norm_vanish * 10:
                    add("suspicious",
                        f"{gname} < {t.grad_norm_vanish * 10:.1e} — gradient 可能消失",
                        f"{gname}_vanish")

        # ── Rule 8: Success stagnation ─────────────────────────────
        if "success_rate" in task and len(task["success_rate"]) >= t.min_points_for_diagnosis:
            _, earlier, recent = self._segment_stats(task["success_rate"])
            if (recent is not None and earlier is not None
                    and abs(recent - earlier) < t.success_stagnation_delta
                    and recent < t.success_stagnation_ceil):
                add("suspicious",
                    f"success_rate 停滯 ({earlier:.3f} → {recent:.3f}, < {t.success_stagnation_ceil}) — "
                    "考慮調整 reward 或 curriculum",
                    "success_stagnation")

        # ── Rule 9: Conservative policy (coll↓ timeout↑) ──────────
        if "collision_rate" in task and "timeout_rate" in task:
            coll, tout = task["collision_rate"], task["timeout_rate"]
            if len(coll) >= t.min_points_for_diagnosis and len(tout) >= t.min_points_for_diagnosis:
                _, ce, cr = self._segment_stats(coll)
                _, te, tr = self._segment_stats(tout)
                if cr and ce and tr and te:
                    if cr < ce * t.conservative_coll_drop and tr > te * t.conservative_tout_rise:
                        add("suspicious",
                            f"collision ↓ ({ce:.3f}→{cr:.3f}) 但 timeout ↑ ({te:.3f}→{tr:.3f}) — "
                            "agent 變得過於保守",
                            "conservative_policy")

        # ── Rule 10: Ablation-specific ─────────────────────────────
        for field, vals in self.ablation_series.items():
            if not vals:
                continue
            latest = vals[-1]
            if "freeze_ratio" in field and latest > t.freeze_ratio_warn:
                add("confirmed",
                    f"{field} = {latest:.2f} > {t.freeze_ratio_warn} — agent 大量時間凍結",
                    "freeze_high")
            if "oscillation_score" in field and latest > t.oscillation_warn:
                add("suspicious",
                    f"{field} = {latest:.2f} > {t.oscillation_warn} — agent 來回振盪",
                    "oscillation_high")
            if "retreat_ratio" in field and latest > t.retreat_ratio_warn:
                add("suspicious",
                    f"{field} = {latest:.2f} > {t.retreat_ratio_warn} — 近障礙時過度後退",
                    "retreat_high")
            if "stuck_count" in field and latest > t.stuck_count_warn:
                add("confirmed",
                    f"{field} = {latest:.0f} > {t.stuck_count_warn} — 頻繁卡住",
                    "stuck_high")

        # ── Rule 11: Value loss spike ──────────────────────────────
        if "value_loss" in ppo and len(ppo["value_loss"]) >= t.min_points_for_diagnosis:
            _, earlier, recent = self._segment_stats(ppo["value_loss"])
            if recent and earlier and recent > earlier * 2.0:
                add("suspicious",
                    f"value_loss 急升 ({earlier:.4f} → {recent:.4f}) — critic 可能 overfitting 或 reward 結構改變",
                    "value_loss_spike")

        # ── Fallback ──
        if not findings:
            if self.n < t.min_history_for_report:
                add("info",
                    f"資料量不足 ({self.n} rows < {t.min_history_for_report})，無法做可靠趨勢判斷",
                    "insufficient_data")
            else:
                add("info", "未發現明顯異常", "all_clear")

        return findings

    def suggest_next_checks(self) -> list[str]:
        """Generate actionable suggestions based on analysis gaps."""
        suggestions: list[str] = []
        diag = self.diagnose()
        rules_triggered = {d["rule"] for d in diag}

        if "explained_variance" not in self.ppo_series:
            suggestions.append("確認 W&B 是否記錄 `explained_variance` — 這是判斷 critic 品質的關鍵指標")
        if "success_rate" not in self.task_series:
            suggestions.append("檢查是否有 `success_rate` / `collision_rate` 欄位")
        if not self.ablation_series:
            suggestions.append("確認 `AblationMetricsLogger` 是否啟用 — ablation/* 欄位未出現")
        if self.n < 50:
            suggestions.append("History 點數較少，建議等訓練更久後再做趨勢分析")

        if "kl_rising" in rules_triggered or "kl_high" in rules_triggered:
            suggestions.append("嘗試降低 `learning_rate` 或增加 `mini_batches` 來穩定 PPO 更新")
        if "entropy_premature_collapse" in rules_triggered or "entropy_collapsed" in rules_triggered:
            suggestions.append("嘗試提高 `entropy_coef` 來維持 exploration")
        if "adv_weak" in rules_triggered or "adv_shrinking" in rules_triggered:
            suggestions.append("檢查 reward 設計是否提供足夠區分度，或 critic 是否過度平滑 value baseline")
        if "conservative_policy" in rules_triggered:
            suggestions.append("考慮降低 collision penalty 或增加 goal-reaching reward 來平衡 risk/reward")
        if "freeze_high" in rules_triggered or "stuck_high" in rules_triggered:
            suggestions.append("檢查 action space 是否允許足夠的機動性，或 obstacle 配置是否過於密集")

        if not suggestions:
            suggestions.append("目前狀態良好，繼續監控 reward_mean 和 success_rate 的趨勢")

        return suggestions


# ══════════════════════════════════════════════════════════════════════════
#  Section 6: Output Formatters
# ══════════════════════════════════════════════════════════════════════════

LEVEL_ICONS = {
    "confirmed": "🔴",
    "suspicious": "🟡",
    "info": "ℹ️",
}


def terminal_summary(analyses: list[RunAnalysis]) -> str:
    """Compact terminal output for quick review."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 72)
    lines.append("  wandblook — W&B RL Training Analysis")
    lines.append("=" * 72)

    for a in analyses:
        m = a.meta
        lines.append(f"\n{'─' * 64}")
        lines.append(f"  Run: {m['name']}  ({m['id']})")
        lines.append(f"  State: {m['state']}  |  History: {m['history_rows']} rows  |  Created: {m['created_at']}")
        lines.append(f"  URL: {m['url']}")

        if a.ppo_series:
            lines.append(f"\n  PPO Metrics (latest):")
            for name, vals in a.ppo_series.items():
                lines.append(f"    {name:25s} = {fmt_val(vals[-1])}")

        if a.task_series:
            lines.append(f"\n  Task Metrics (latest):")
            for name, vals in a.task_series.items():
                lines.append(f"    {name:25s} = {fmt_val(vals[-1])}")

        if a.ablation_series:
            lines.append(f"\n  Ablation Metrics (latest):")
            for name, vals in a.ablation_series.items():
                lines.append(f"    {name:25s} = {fmt_val(vals[-1])}")

        if a.extra_series:
            lines.append(f"\n  Extra Metrics (latest):")
            for name, vals in a.extra_series.items():
                lines.append(f"    {name:25s} = {fmt_val(vals[-1])}")

        lines.append(f"\n  Diagnosis:")
        for d in a.diagnose():
            icon = LEVEL_ICONS.get(d["level"], "?")
            lines.append(f"    {icon} [{d['level']}] {d['message']}")

    lines.append(f"\n{'=' * 72}")
    return "\n".join(lines)


def markdown_report(analyses: list[RunAnalysis]) -> str:
    """Full markdown diagnostic report."""
    lines: list[str] = []
    lines.append("# W&B Run Analysis Report")
    lines.append(f"\n> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> Runs analyzed: {len(analyses)}")
    lines.append(f"> Analyzer: wandblook v2.0\n")

    for i, a in enumerate(analyses):
        m = a.meta

        if len(analyses) > 1:
            lines.append(f"\n---\n\n## Run {i+1}: {m['name']}\n")

        # ── 1. Metadata ──
        lines.append("### 1. Run Metadata\n")
        lines.append("| Field | Value |")
        lines.append("|-------|-------|")
        for key, val in [
            ("Name", f"`{m['name']}`"),
            ("ID", f"`{m['id']}`"),
            ("State", f"**{m['state']}**"),
            ("Created", m["created_at"]),
            ("URL", m["url"]),
            ("Tags", ", ".join(m["tags"]) if m["tags"] else "—"),
            ("History rows", str(m["history_rows"])),
        ]:
            lines.append(f"| {key} | {val} |")

        # State warning
        if m["state"] == "crashed":
            lines.append("\n> ⚠️ **此 run 已 crash**，以下分析僅涵蓋 crash 前的資料。\n")
        elif m["state"] == "running":
            lines.append("\n> ℹ️ 此 run 仍在執行中，數據持續更新。\n")

        # ── 2. Config ──
        cfg = a.config_snapshot
        if cfg:
            lines.append("\n### 2. Config Snapshot\n")
            lines.append("| Key | Value |")
            lines.append("|-----|-------|")
            for k, v in sorted(cfg.items()):
                lines.append(f"| `{k}` | `{v}` |")
        else:
            lines.append("\n### 2. Config Snapshot\n\n_No config data available._\n")

        # ── 3. Metrics ──
        lines.append("\n### 3. Key Metrics Snapshot\n")

        def _metrics_table(title: str, trend_rows: list[TrendRow]) -> None:
            lines.append(f"\n#### {title}\n")
            if not trend_rows:
                lines.append("_No data available._\n")
                return
            lines.append("| Metric | Latest | Mean | Std | Recent 20% | Trend | Min | Max | N |")
            lines.append("|--------|--------|------|-----|------------|-------|-----|-----|---|")
            for r in trend_rows:
                lines.append(
                    f"| `{r.metric}` "
                    f"| {fmt_val(r.latest)} "
                    f"| {fmt_val(r.mean)} "
                    f"| {fmt_val(r.std)} "
                    f"| {fmt_val(r.recent_mean)} "
                    f"| {r.trend} "
                    f"| {fmt_val(r.min_val)} "
                    f"| {fmt_val(r.max_val)} "
                    f"| {r.count} |"
                )

        _metrics_table("PPO / Actor-Critic", a.compute_trends(a.ppo_series))
        _metrics_table("Task Performance", a.compute_trends(a.task_series))
        _metrics_table("Ablation Diagnostics", a.compute_trends(a.ablation_series))
        if a.extra_series:
            _metrics_table("Extra Metrics", a.compute_trends(a.extra_series))

        # ── 4. Trend Analysis ──
        lines.append("\n### 4. Trend Analysis\n")
        all_trends = (
            a.compute_trends(a.ppo_series)
            + a.compute_trends(a.task_series)
            + a.compute_trends(a.ablation_series)
            + a.compute_trends(a.extra_series)
        )
        rising = [r for r in all_trends if "上升" in r.trend]
        falling = [r for r in all_trends if "下降" in r.trend]
        stable = [r for r in all_trends if "持平" in r.trend]

        if rising:
            lines.append("**↑ 上升趨勢:**")
            for r in rising:
                lines.append(f"- `{r.metric}`: {fmt_val(r.earlier_mean)} → {fmt_val(r.recent_mean)}")
        if falling:
            lines.append("\n**↓ 下降趨勢:**")
            for r in falling:
                lines.append(f"- `{r.metric}`: {fmt_val(r.earlier_mean)} → {fmt_val(r.recent_mean)}")
        if stable:
            lines.append(f"\n**→ 持平:** {', '.join(f'`{r.metric}`' for r in stable)}")
        if not (rising or falling or stable):
            lines.append("_資料不足，無法判斷趨勢。_")

        # ── 5. Diagnostic Conclusion ──
        lines.append("\n### 5. Diagnostic Conclusion\n")
        diag = a.diagnose()
        confirmed = [d for d in diag if d["level"] == "confirmed"]
        suspicious = [d for d in diag if d["level"] == "suspicious"]
        info = [d for d in diag if d["level"] == "info"]

        if confirmed:
            lines.append("**已確認 (Confirmed):**")
            for d in confirmed:
                lines.append(f"- 🔴 {d['message']}")
        if suspicious:
            lines.append("\n**高度可疑 (Suspicious) — [HEURISTIC]:**")
            for d in suspicious:
                lines.append(f"- 🟡 {d['message']}")
        if info:
            lines.append("\n**資訊:**")
            for d in info:
                lines.append(f"- ℹ️ {d['message']}")

        # ── 6. Suggested Next Checks ──
        lines.append("\n### 6. Suggested Next Checks\n")
        for s in a.suggest_next_checks():
            lines.append(f"- {s}")

        # ── Field alias debug info ──
        if a._matched_aliases:
            lines.append("\n<details><summary>Field Alias Mapping (debug)</summary>\n")
            for canonical, actual in sorted(a._matched_aliases.items()):
                lines.append(f"- `{canonical}` ← `{actual}`")
            lines.append("\n</details>\n")

    return "\n".join(lines)


def comparison_section(analyses: list[RunAnalysis]) -> str:
    """Cross-run comparison table for multi-run analysis."""
    if len(analyses) < 2:
        return ""

    lines: list[str] = ["\n---\n\n## Multi-Run Comparison\n"]

    # Collect all metrics
    all_metrics: set[str] = set()
    for a in analyses:
        all_metrics.update(a.ppo_series.keys())
        all_metrics.update(a.task_series.keys())
        all_metrics.update(a.ablation_series.keys())
        all_metrics.update(a.extra_series.keys())

    # Table header
    names = [a.meta["name"][:25] for a in analyses]
    header = "| Metric | " + " | ".join(names) + " |"
    sep = "|--------|" + "|".join("-----:" for _ in analyses) + "|"
    lines.append(header)
    lines.append(sep)

    for metric in sorted(all_metrics):
        row = f"| `{metric}` |"
        for a in analyses:
            combined = {**a.ppo_series, **a.task_series, **a.ablation_series, **a.extra_series}
            vals = combined.get(metric)
            row += f" {fmt_val(vals[-1]) if vals else '—'} |"
        lines.append(row)

    # Diagnosis comparison
    lines.append("\n### Diagnosis Summary\n")
    lines.append("| Run | Confirmed | Suspicious | Status |")
    lines.append("|-----|-----------|------------|--------|")
    for a in analyses:
        diag = a.diagnose()
        n_conf = sum(1 for d in diag if d["level"] == "confirmed")
        n_susp = sum(1 for d in diag if d["level"] == "suspicious")
        status = "🔴" if n_conf > 0 else ("🟡" if n_susp > 0 else "✅")
        lines.append(f"| {a.meta['name'][:30]} | {n_conf} | {n_susp} | {status} |")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  Section 7: CSV / JSON Export
# ══════════════════════════════════════════════════════════════════════════

def export_json(analyses: list[RunAnalysis], path: str) -> None:
    """Export structured analysis data as JSON."""
    data = []
    for a in analyses:
        entry: dict[str, Any] = {
            "meta": a.meta,
            "config": a.config_snapshot,
            "ppo_latest": {k: vals[-1] for k, vals in a.ppo_series.items()},
            "task_latest": {k: vals[-1] for k, vals in a.task_series.items()},
            "ablation_latest": {k: vals[-1] for k, vals in a.ablation_series.items()},
            "extra_latest": {k: vals[-1] for k, vals in a.extra_series.items()},
            "diagnosis": a.diagnose(),
            "suggestions": a.suggest_next_checks(),
        }
        # Add trend summaries
        all_trends = (
            a.compute_trends(a.ppo_series)
            + a.compute_trends(a.task_series)
            + a.compute_trends(a.ablation_series)
        )
        entry["trends"] = {
            t.metric: {"latest": t.latest, "mean": t.mean, "trend": t.trend}
            for t in all_trends
        }
        data.append(entry)

    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info(f"📦 JSON saved: {path}")


def export_csv(analyses: list[RunAnalysis], path: str) -> None:
    """Export a flat CSV with one row per run, columns = metrics."""
    all_metrics: set[str] = set()
    for a in analyses:
        all_metrics.update(a.ppo_series.keys())
        all_metrics.update(a.task_series.keys())
        all_metrics.update(a.ablation_series.keys())
        all_metrics.update(a.extra_series.keys())

    sorted_metrics = sorted(all_metrics)
    fieldnames = ["run_name", "run_id", "state", "history_rows"] + sorted_metrics

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()

    for a in analyses:
        combined = {**a.ppo_series, **a.task_series, **a.ablation_series, **a.extra_series}
        row: dict[str, Any] = {
            "run_name": a.meta["name"],
            "run_id": a.meta["id"],
            "state": a.meta["state"],
            "history_rows": a.meta["history_rows"],
        }
        for m in sorted_metrics:
            vals = combined.get(m)
            row[m] = f"{vals[-1]:.6f}" if vals else ""
        writer.writerow(row)

    Path(path).write_text(buf.getvalue(), encoding="utf-8")
    log.info(f"📊 CSV saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
#  Section 8: CLI Entry Point
# ══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wandblook",
        description="W&B RL Training Analyzer — 自動抓取、分析、診斷 RL 訓練 runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  %(prog)s                                           # 最新 1 個 run
  %(prog)s me0608623-none/charge_skrl/runs/abc123    # 指定 run
  %(prog)s --latest 5                                # 最新 5 個比較
  %(prog)s --filter "abl1_*" --since-days 3          # 3 天內 abl1_* runs
  %(prog)s --latest 3 --metrics kl,entropy --csv     # 自訂指標 + CSV
  %(prog)s --state running                           # 只看執行中的 runs
""",
    )

    # ── Target selection ──
    p.add_argument("run_paths", nargs="*", default=[],
                   help="W&B run path(s), e.g. entity/project/runs/id")
    p.add_argument("--entity", default=DEFAULT_ENTITY,
                   help=f"W&B entity (default: {DEFAULT_ENTITY})")
    p.add_argument("--project", default=DEFAULT_PROJECT,
                   help=f"W&B project (default: {DEFAULT_PROJECT})")
    p.add_argument("--latest", type=int, default=1,
                   help="Fetch N latest runs (default: 1)")
    p.add_argument("--filter", default=None,
                   help="Filter runs by display_name pattern (glob-style, e.g. 'abl1_*')")
    p.add_argument("--since-days", type=int, default=None,
                   help="Only fetch runs created within N days")
    p.add_argument("--state", default=None, choices=["running", "finished", "crashed", "failed"],
                   help="Filter by run state")

    # ── Analysis options ──
    p.add_argument("--max-history", type=int, default=2000,
                   help="Max history rows per run (default: 2000)")
    p.add_argument("--metrics", default=None,
                   help="Comma-separated extra metric names to track (e.g. 'my_metric,custom/loss')")

    # ── Output options ──
    p.add_argument("--output-dir", default=None,
                   help="Output directory for reports (default: /tmp)")
    p.add_argument("--json", action="store_true",
                   help="Also export JSON summary")
    p.add_argument("--csv", action="store_true",
                   help="Also export CSV summary (one row per run)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress terminal summary, only write files")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ── Resolve output directory ──
    out_dir = Path(args.output_dir) if args.output_dir else DEFAULT_REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Parse extra metrics ──
    extra_metrics: list[str] | None = None
    if args.metrics:
        extra_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # ── Create API client ──
    api = create_api()

    # ── Fetch runs ──
    log.info(f"🔍 Fetching runs from {args.entity}/{args.project} ...")
    runs = fetch_runs(
        api=api,
        entity=args.entity,
        project=args.project,
        run_paths=args.run_paths if args.run_paths else None,
        latest=args.latest,
        name_filter=args.filter,
        since_days=args.since_days,
        state_filter=args.state,
    )

    if not runs:
        log.error("❌ No runs found. 請確認 entity/project/filter 設定。")
        return 1

    log.info(f"📊 Analyzing {len(runs)} run(s) ...\n")

    # ── Analyze each run ──
    analyses: list[RunAnalysis] = []
    for run in runs:
        log.info(f"  → {run.name} ({run.id}) [{run.state}]")
        t0 = time.monotonic()
        history = fetch_history(run, max_samples=args.max_history)
        elapsed = time.monotonic() - t0
        log.info(f"    fetched {len(history)} rows in {elapsed:.1f}s")
        analyses.append(RunAnalysis(run, history, extra_metrics=extra_metrics))

    # ── Terminal output ──
    if not args.quiet:
        print(terminal_summary(analyses))

    # ── Markdown report ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_name = f"wandblook_{timestamp}.md" if len(analyses) > 1 else f"wandblook_{analyses[0].meta['id']}_{timestamp}.md"
    md_path = str(out_dir / md_name)

    report = markdown_report(analyses)
    if len(analyses) > 1:
        report += "\n" + comparison_section(analyses)

    Path(md_path).write_text(report, encoding="utf-8")
    log.info(f"\n📝 Report saved: {md_path}")

    # Also write a stable "latest" symlink/copy
    latest_path = str(out_dir / "wandblook_report.md")
    Path(latest_path).write_text(report, encoding="utf-8")

    # ── Optional exports ──
    if args.json:
        json_path = md_path.replace(".md", ".json")
        export_json(analyses, json_path)

    if args.csv:
        csv_path = md_path.replace(".md", ".csv")
        export_csv(analyses, csv_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
