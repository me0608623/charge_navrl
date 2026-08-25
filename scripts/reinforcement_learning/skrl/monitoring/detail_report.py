#!/usr/bin/env python
"""每 10 分鐘產生一份繁中訓練詳細監督報告。

設計原則
--------
1. **只回報，永不介入。** 本模組不含任何終止進程的手段；由
   `test_detail_report.py::test_module_never_kills_a_process` 以來源契約保證。
2. **判定邏輯是純函式。** `evaluate` / `trend` / `check_stall` 等只吃 metric
   rows，不碰檔案系統，因此崩潰情境可以用假資料測，不必真的跑壞一次訓練。
3. **欄位不存在 ≠ 檢查通過。** `core_health` 對缺欄位發 RED，避免把「讀不到」
   誤讀成「沒問題」。
4. **門檻集中在 `Thresholds`**，不散落在判定式裡。

用法::

    python detail_report.py                 # 自動找最新 run，附加到報告檔
    python detail_report.py --stdout        # 只印到終端，不寫檔
    python detail_report.py --run <name>    # 指定 run
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[4]
LOGS_DIR_DEFAULT = REPO_DEFAULT / "logs" / "rnn_car"
STATE_DIR_DEFAULT = REPO_DEFAULT / "logs" / "training_supervisor"

METRICS_FILENAME = "supervisor_metrics.jsonl"
SCENE_NAMES = ("native", "narrow", "corridor")
CORRIDOR_FAMILY_NAMES = (
    "lateral",
    "longitudinal",
    "random_2d",
    "mixed",
    "no_dynamic",
    "unready",
)
CORE_FIELDS = ("sr", "cr", "entropy", "kl", "vf", "policy_loss")

#: 與 cron_training_supervisor.sh 相同的收斂式 regex：conda wrapper 不得誤配。
TRAINER_PATTERN = (
    r"^/[^ ]*/python(3([.][0-9]+)?)? ([-][^ ]+ )*[^ ]*train_rnn_car_wdclip[.]py( |$)"
)

ERROR_PATTERN = (
    r"Traceback|CUDA error|out of memory|\bOOM\b|RuntimeError|Segmentation|"
    r"AssertionError|(^|[^A-Za-z])nan([^A-Za-z]|$)"
)

LEVEL_ORDER = {"GREEN": 0, "YELLOW": 1, "RED": 2}


@dataclass(frozen=True)
class Thresholds:
    """所有判定門檻。改門檻只改這裡，判定式不得寫死數字。"""

    kl_red: float = 0.020
    kl_yellow: float = 0.010
    clip_red: float = 0.30
    clip_yellow: float = 0.20
    #: entropy 在 `trend_lookback` 個 iteration 內的下降量
    entropy_drop_red: float = 0.50
    entropy_drop_yellow: float = 0.25
    #: CR 在 `trend_lookback` 個 iteration 內的絕對回彈量
    cr_rebound_red: float = 0.05
    cr_rebound_yellow: float = 0.02
    #: 近期 fps 中位數低於基準中位數的比例
    fps_drop_yellow: float = 0.30
    fps_recent_window: int = 3
    #: 單一場景 episode 數低於此值時，該場景的比率不足以支撐結論
    min_scene_episodes: int = 30
    trend_lookback: int = 50
    table_window: int = 6


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str


@dataclass(frozen=True)
class Trend:
    key: str
    span: int
    requested_lookback: int
    first: float | None
    last: float | None
    delta: float | None


@dataclass(frozen=True)
class SceneRow:
    name: str
    episodes: float | None
    sr: float | None
    cr: float | None
    timeout: float | None


@dataclass(frozen=True)
class CorridorFamilyRow:
    name: str
    episodes: float | None
    sr: float | None
    cr: float | None
    timeout: float | None
    active_step_share: float | None
    reset_share: float | None
    completed_episode_share: float | None
    linear_speed_abs_mean_mps: float | None
    stop_command_fraction: float | None
    reverse_command_fraction: float | None
    high_turn_fraction: float | None
    extreme_turn_fraction: float | None
    wall_cr: float | None
    obstacle_cr: float | None
    static_obstacle_cr: float | None
    dynamic_obstacle_cr: float | None
    reward_mean_per_step: float | None
    progress_mean_per_step: float | None


# ---------------------------------------------------------------------------
# 純函式：數值處理
# ---------------------------------------------------------------------------
def _num(value):
    """只有有限實數才回傳 float，其餘一律 None（含 None / NaN / inf / 字串）。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _fmt(value, spec=".4f", dash="—"):
    number = _num(value)
    return dash if number is None else format(number, spec)


def _pct(value, dash="—"):
    number = _num(value)
    return dash if number is None else f"{number * 100:.2f}%"


def _duration(seconds):
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


# ---------------------------------------------------------------------------
# 純函式：健康檢查
# ---------------------------------------------------------------------------
def core_health(row):
    """核心欄位的 NaN / None / 缺漏檢查。"""
    findings = []
    for key in CORE_FIELDS:
        if key not in row:
            findings.append(
                Finding(
                    "RED",
                    "core_missing",
                    f"metrics 缺少核心欄位 {key} —— 欄位不存在不等於檢查通過",
                )
            )
            continue
        value = row[key]
        if value is None:
            findings.append(Finding("RED", "core_none", f"核心欄位 {key} 是 None"))
        elif _num(value) is None:
            findings.append(
                Finding("RED", "core_nonfinite", f"核心欄位 {key} 非有限值：{value!r}")
            )
    return findings


def trend(rows, key, lookback):
    """回傳 `key` 在最近 `lookback` 個 iteration 的變化。

    span 是**實際**跨度，不是要求的跨度 —— 歷史不足時不假裝跨滿。
    """
    samples = [v for v in (_num(r.get(key)) for r in rows) if v is not None]
    if not samples:
        return None
    window = samples[-(lookback + 1) :]
    first, last = window[0], window[-1]
    return Trend(
        key=key,
        span=len(window) - 1,
        requested_lookback=lookback,
        first=first,
        last=last,
        delta=last - first,
    )


def _normalized_delta(tr, lookback):
    """把實際跨度換算回 lookback 尺度，避免短歷史被低估或高估。"""
    if tr is None or tr.delta is None or tr.span <= 0:
        return None
    return tr.delta * (lookback / tr.span)


def scene_snapshot(row):
    """抽出分場景 SR/CR/TO；沒有 scene/* 欄位的舊 run 回傳空 list。"""
    scenes = []
    for name in SCENE_NAMES:
        episodes_key = f"scene/{name}/episodes"
        if episodes_key not in row:
            continue
        scenes.append(
            SceneRow(
                name=name,
                episodes=_num(row.get(episodes_key)),
                sr=_num(row.get(f"scene/{name}/sr")),
                cr=_num(row.get(f"scene/{name}/cr")),
                timeout=_num(row.get(f"scene/{name}/timeout")),
            )
        )
    return scenes


def corridor_family_snapshot(row):
    """抽出走廊 family 指標；舊 run 沒有欄位時回傳空 list。"""
    families = []
    for name in CORRIDOR_FAMILY_NAMES:
        prefix = f"corridor_family/{name}"
        if (
            f"{prefix}/active_steps" not in row
            and f"{prefix}/episodes" not in row
        ):
            continue
        families.append(
            CorridorFamilyRow(
                name=name,
                episodes=_num(row.get(f"{prefix}/episodes")),
                sr=_num(row.get(f"{prefix}/sr")),
                cr=_num(row.get(f"{prefix}/cr")),
                timeout=_num(row.get(f"{prefix}/timeout")),
                active_step_share=_num(
                    row.get(f"{prefix}/active_step_share")
                ),
                reset_share=_num(row.get(f"{prefix}/reset_share")),
                completed_episode_share=_num(
                    row.get(f"{prefix}/completed_episode_share")
                ),
                linear_speed_abs_mean_mps=_num(
                    row.get(f"{prefix}/linear_speed_abs_mean_mps")
                ),
                stop_command_fraction=_num(
                    row.get(f"{prefix}/stop_command_fraction")
                ),
                reverse_command_fraction=_num(
                    row.get(f"{prefix}/reverse_command_fraction")
                ),
                high_turn_fraction=_num(
                    row.get(f"{prefix}/high_turn_fraction")
                ),
                extreme_turn_fraction=_num(
                    row.get(f"{prefix}/extreme_turn_fraction")
                ),
                wall_cr=_num(row.get(f"{prefix}/wall_cr")),
                obstacle_cr=_num(row.get(f"{prefix}/obstacle_cr")),
                static_obstacle_cr=_num(
                    row.get(f"{prefix}/static_obstacle_cr")
                ),
                dynamic_obstacle_cr=_num(
                    row.get(f"{prefix}/dynamic_obstacle_cr")
                ),
                reward_mean_per_step=_num(
                    row.get(
                        f"{prefix}/signal/total_reward_mean_per_step"
                    )
                ),
                progress_mean_per_step=_num(
                    row.get(
                        f"{prefix}/signal/progress_reward_mean_per_step"
                    )
                ),
            )
        )
    return families


def evaluate(rows, thresholds=None):
    """對最後一列 + 趨勢做完整判定，回傳 findings。"""
    limits = thresholds or Thresholds()
    if not rows:
        return [Finding("RED", "no_metrics", "supervisor_metrics.jsonl 沒有任何資料列")]

    last = rows[-1]
    findings = list(core_health(last))
    lookback = limits.trend_lookback

    kl = _num(last.get("kl"))
    if kl is not None:
        if kl > limits.kl_red:
            findings.append(
                Finding("RED", "kl_high", f"KL={kl:.4f} 超過紅線 {limits.kl_red}")
            )
        elif kl > limits.kl_yellow:
            findings.append(
                Finding("YELLOW", "kl_high", f"KL={kl:.4f} 高於 {limits.kl_yellow}")
            )

    clip = _num(last.get("clip_fraction"))
    if clip is not None:
        if clip > limits.clip_red:
            findings.append(
                Finding(
                    "RED",
                    "clip_high",
                    f"clip_fraction={clip:.4f} 超過紅線 {limits.clip_red}",
                )
            )
        elif clip > limits.clip_yellow:
            findings.append(
                Finding(
                    "YELLOW",
                    "clip_high",
                    f"clip_fraction={clip:.4f} 高於 {limits.clip_yellow}",
                )
            )

    entropy_trend = trend(rows, "entropy", lookback)
    drop = _normalized_delta(entropy_trend, lookback)
    if drop is not None and drop < 0:
        magnitude = -drop
        span = entropy_trend.span
        if magnitude >= limits.entropy_drop_red:
            findings.append(
                Finding(
                    "RED",
                    "entropy_drop",
                    f"entropy 於 {span} iter 內下降 {-entropy_trend.delta:.3f}"
                    f"（換算每 {lookback} iter {magnitude:.3f}），達塌縮紅線",
                )
            )
        elif magnitude >= limits.entropy_drop_yellow:
            findings.append(
                Finding(
                    "YELLOW",
                    "entropy_drop",
                    f"entropy 於 {span} iter 內下降 {-entropy_trend.delta:.3f}"
                    f"（換算每 {lookback} iter {magnitude:.3f}），下降偏快",
                )
            )

    cr_trend = trend(rows, "cr", lookback)
    rebound = _normalized_delta(cr_trend, lookback)
    if rebound is not None and rebound > 0:
        span = cr_trend.span
        if rebound >= limits.cr_rebound_red:
            findings.append(
                Finding(
                    "RED",
                    "cr_rebound",
                    f"CR 於 {span} iter 內上升 {cr_trend.delta:+.4f}"
                    f"（換算每 {lookback} iter {rebound:+.4f}），回彈顯著",
                )
            )
        elif rebound >= limits.cr_rebound_yellow:
            findings.append(
                Finding(
                    "YELLOW",
                    "cr_rebound",
                    f"CR 於 {span} iter 內上升 {cr_trend.delta:+.4f}"
                    f"（換算每 {lookback} iter {rebound:+.4f}）",
                )
            )

    fps_values = [v for v in (_num(r.get("fps")) for r in rows[-lookback:]) if v]
    recent = fps_values[-limits.fps_recent_window :]
    baseline_median = _median(fps_values)
    recent_median = _median(recent)
    if baseline_median and recent_median is not None:
        floor = baseline_median * (1.0 - limits.fps_drop_yellow)
        if recent_median < floor:
            findings.append(
                Finding(
                    "YELLOW",
                    "fps_drop",
                    f"fps 近期中位數 {recent_median:.0f} 低於基準 {baseline_median:.0f} "
                    f"的 {(1 - limits.fps_drop_yellow) * 100:.0f}%；"
                    "先看 nvidia-smi 有沒有別人的 process 再談 stall",
                )
            )

    for scene in scene_snapshot(last):
        if scene.episodes is not None and scene.episodes < limits.min_scene_episodes:
            findings.append(
                Finding(
                    "YELLOW",
                    "scene_low_n",
                    f"場景 {scene.name} 本輪僅 {scene.episodes:.0f} 個 episode"
                    f"（< {limits.min_scene_episodes}），其比率不足以支撐結論",
                )
            )

    families = corridor_family_snapshot(last)
    if families:
        reconciliation = _num(
            last.get("corridor_family/accounting/reconciliation_ok")
        )
        if reconciliation != 1.0:
            findings.append(
                Finding(
                    "RED",
                    "corridor_family_reconciliation",
                    "走廊 family 帳本未通過 scene/corridor 對帳",
                )
            )
        for family in families:
            if (
                family.episodes is not None
                and family.episodes < limits.min_scene_episodes
            ):
                findings.append(
                    Finding(
                        "YELLOW",
                        "corridor_family_low_n",
                        f"走廊 family {family.name} 本輪僅 "
                        f"{family.episodes:.0f} 個 episode"
                        f"（< {limits.min_scene_episodes}），"
                        "其比率不足以支撐結論",
                    )
                )

    return findings


def verdict(findings):
    """整體判定 = 所有 findings 中最嚴重的一級。"""
    if not findings:
        return "GREEN"
    return max(findings, key=lambda f: LEVEL_ORDER.get(f.level, 0)).level


# ---------------------------------------------------------------------------
# 純函式：進度 / stall / 進程
# ---------------------------------------------------------------------------
def iteration_rate(previous, current):
    """iterations per second；無前次觀測或沒有前進時回傳 None。"""
    if not previous:
        return None
    delta_iter = current["iteration"] - previous["iteration"]
    delta_time = current["observed_at"] - previous["observed_at"]
    if delta_iter <= 0 or delta_time <= 0:
        return None
    return delta_iter / delta_time


def eta_seconds(rate, remaining):
    if not rate or rate <= 0 or remaining is None or remaining < 0:
        return None
    return remaining / rate


def check_stall(previous, current, process_alive):
    """iteration 與 console log 同時凍結才算 stall —— 單看 iteration 會誤判。"""
    if not process_alive or not previous:
        return []
    frozen_iteration = current["iteration"] == previous["iteration"]
    frozen_log = current.get("log_size") == previous.get("log_size")
    elapsed = current["observed_at"] - previous["observed_at"]
    if frozen_iteration and frozen_log and elapsed > 0:
        return [
            Finding(
                "RED",
                "stall",
                f"iteration 停在 {current['iteration']} 且 console log "
                f"{int(elapsed)} 秒無成長 —— 疑似 stall（先確認 GPU 上有無他人 job）",
            )
        ]
    return []


def check_process(iteration, target, process_alive):
    if process_alive:
        return []
    if target is not None and iteration is not None and iteration >= target:
        return [
            Finding("GREEN", "run_complete", f"訓練已完成 {iteration}/{target}，進程正常結束")
        ]
    return [
        Finding(
            "RED",
            "process_gone",
            f"進程消失但只跑到 {iteration}/{target} —— 異常結束，須查死因",
        )
    ]


# ---------------------------------------------------------------------------
# I/O 外殼
# ---------------------------------------------------------------------------
def find_active_run(logs_dir):
    """最新 mtime、且含 supervisor_metrics.jsonl 的 run 目錄。"""
    logs_dir = Path(logs_dir)
    if not logs_dir.is_dir():
        return None
    candidates = [
        (metrics.stat().st_mtime, run)
        for run in logs_dir.iterdir()
        if run.is_dir() and (metrics := run / METRICS_FILENAME).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def load_rows(path):
    """讀 jsonl；壞掉的行必須吵出來，不得安靜跳過整段歷史。"""
    rows = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return rows


def read_state(path):
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def write_state(path, observation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(observation, ensure_ascii=False, indent=2))


def previous_for_run(path, run_name):
    """換 run 之後不得沿用舊 run 的 iteration 計算速率。"""
    state = read_state(path)
    if not state or state.get("run") != run_name:
        return None
    return state


def _run_command(argv):
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout


def detect_trainer_pids():
    output = _run_command(["pgrep", "-f", TRAINER_PATTERN])
    return [line.strip() for line in output.splitlines() if line.strip()]


def gpu_summary():
    lines = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader",
        ]
    ).splitlines()
    procs = []
    for line in _run_command(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"]
    ).splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2:
            procs.append((parts[0], parts[1]))
    return [line.strip() for line in lines if line.strip()], procs


def scan_console(console_log, tail_lines=800):
    """回傳 (錯誤行, 最後一筆 REPLAY-MIX, log 位元組數)。"""
    path = Path(console_log)
    if not path.is_file():
        return ["console log 不存在：" + str(path)], None, None
    size = path.stat().st_size
    text = path.read_text(errors="replace").splitlines()
    tail = text[-tail_lines:]
    pattern = re.compile(ERROR_PATTERN, re.IGNORECASE)
    errors = [line for line in tail if pattern.search(line)]
    mixes = [line.strip() for line in text if "REPLAY-MIX" in line]
    return errors[-10:], (mixes[-1] if mixes else None), size


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------
def _metric_table(rows, window):
    keys = [
        ("iteration", "it", "d", 5),
        ("sr", "SR", ".4f", 7),
        ("cr", "CR", ".4f", 7),
        ("timeout", "TO", ".4f", 7),
        ("entropy", "ent", ".4f", 7),
        ("kl", "KL", ".5f", 8),
        ("clip_fraction", "clip", ".4f", 7),
        ("policy_loss", "ppo", ".5f", 9),
        ("vf", "vf", ".3f", 7),
        ("actor_param_delta_norm", "Δθa", ".4f", 7),
        ("fps", "fps", ".0f", 6),
    ]
    header = " ".join(f"{label:>{width}}" for _, label, _, width in keys)
    lines = [header, "-" * len(header)]
    for row in rows[-window:]:
        cells = []
        for key, _, spec, width in keys:
            value = row.get(key)
            if spec == "d":
                text = str(value)
            else:
                text = _fmt(value, spec)
            cells.append(f"{text:>{width}}")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def _scene_table(rows, window):
    present = [r for r in rows[-window:] if scene_snapshot(r)]
    if not present:
        return None
    header = f"{'it':>5} | " + " | ".join(
        f"{name:^28}" for name in SCENE_NAMES
    )
    sub = f"{'':>5} | " + " | ".join(
        f"{'n':>6} {'SR':>6} {'CR':>6} {'TO':>6}" for _ in SCENE_NAMES
    )
    lines = [header, sub, "-" * len(sub)]
    for row in present:
        by_name = {s.name: s for s in scene_snapshot(row)}
        cells = []
        for name in SCENE_NAMES:
            scene = by_name.get(name)
            if scene is None:
                cells.append(f"{'—':>6} {'—':>6} {'—':>6} {'—':>6}")
                continue
            cells.append(
                f"{_fmt(scene.episodes, '.0f'):>6} "
                f"{_fmt(scene.sr):>6} {_fmt(scene.cr):>6} {_fmt(scene.timeout):>6}"
            )
        lines.append(f"{row.get('iteration'):>5} | " + " | ".join(cells))
    return "\n".join(lines)


def _corridor_family_outcome_table(rows, window):
    present = [
        (row, corridor_family_snapshot(row))
        for row in rows[-window:]
        if corridor_family_snapshot(row)
    ]
    if not present:
        return None
    header = (
        f"{'it':>5} {'family':<13} {'n':>6} "
        f"{'SR':>6} {'CR':>6} {'TO':>6} "
        f"{'occ':>6} {'reset':>6} {'done':>6}"
    )
    lines = [header, "-" * len(header)]
    for row, families in present:
        for family in families:
            lines.append(
                f"{row.get('iteration'):>5} {family.name:<13} "
                f"{_fmt(family.episodes, '.0f'):>6} "
                f"{_fmt(family.sr):>6} {_fmt(family.cr):>6} "
                f"{_fmt(family.timeout):>6} "
                f"{_fmt(family.active_step_share):>6} "
                f"{_fmt(family.reset_share):>6} "
                f"{_fmt(family.completed_episode_share):>6}"
            )
    return "\n".join(lines)


def _corridor_family_behavior_table(row):
    families = corridor_family_snapshot(row)
    if not families:
        return None
    header = (
        f"{'family':<13} {'|v|':>6} {'stop':>6} {'rev':>6} "
        f"{'turn':>6} {'xturn':>6} {'wall':>6} {'obs':>6} "
        f"{'static':>6} {'dyn':>6} {'R/step':>8} {'prog':>7}"
    )
    lines = [header, "-" * len(header)]
    for family in families:
        lines.append(
            f"{family.name:<13} "
            f"{_fmt(family.linear_speed_abs_mean_mps):>6} "
            f"{_fmt(family.stop_command_fraction):>6} "
            f"{_fmt(family.reverse_command_fraction):>6} "
            f"{_fmt(family.high_turn_fraction):>6} "
            f"{_fmt(family.extreme_turn_fraction):>6} "
            f"{_fmt(family.wall_cr):>6} "
            f"{_fmt(family.obstacle_cr):>6} "
            f"{_fmt(family.static_obstacle_cr):>6} "
            f"{_fmt(family.dynamic_obstacle_cr):>6} "
            f"{_fmt(family.reward_mean_per_step, '.4f'):>8} "
            f"{_fmt(family.progress_mean_per_step, '.4f'):>7}"
        )
    return "\n".join(lines)


def render_report(
    run_name,
    rows,
    process_alive,
    gpu_lines,
    gpu_procs,
    errors,
    replay_mix,
    rate,
    findings,
    now_text,
    thresholds=None,
    pids=(),
):
    limits = thresholds or Thresholds()
    last = rows[-1] if rows else {}
    iteration = last.get("iteration")
    target = last.get("iterations_target")
    remaining = None
    if isinstance(iteration, int) and isinstance(target, int):
        remaining = max(0, target - iteration)
    eta = eta_seconds(rate, remaining)

    level = verdict(findings)
    badge = {"GREEN": "🟢 GREEN", "YELLOW": "🟡 YELLOW", "RED": "🔴 RED"}[level]

    out = [
        "=" * 78,
        f"訓練詳細監督報告  {now_text}",
        f"run       : {run_name}",
        f"判定      : {badge}",
        f"進度      : {iteration}/{target}"
        + (f"    剩餘 {remaining} iter" if remaining is not None else ""),
        f"速率      : "
        + (f"{1 / rate:.1f} 秒/iter" if rate else "—（本輪無前次觀測或未前進）")
        + f"    ETA {_duration(eta)}",
        f"進程      : " + ("存活 " + ",".join(pids) if process_alive else "不存在"),
        "=" * 78,
        "",
        "【1】最近 %d 個 iteration" % limits.table_window,
        _metric_table(rows, limits.table_window) if rows else "（無資料）",
        "",
    ]

    scene_table = _scene_table(rows, limits.table_window)
    if scene_table:
        out += ["【2】分場景 SR / CR / TO", scene_table, ""]
    else:
        out += ["【2】分場景 SR / CR / TO：此 run 未輸出 scene/* 指標", ""]

    family_outcomes = _corridor_family_outcome_table(
        rows, limits.table_window
    )
    family_behavior = _corridor_family_behavior_table(last)
    if family_outcomes:
        out += [
            "【3】走廊 motion family（occ/reset/done 皆為各自分母內占比）",
            family_outcomes,
            "",
            "    最新一輪行為與碰撞分解",
            family_behavior,
            "",
        ]
    else:
        out += [
            "【3】走廊 motion family：此 run 未輸出 corridor_family/* 指標",
            "",
        ]

    out += ["【4】趨勢（實際跨度，非要求跨度）"]
    for key, label in (("entropy", "entropy"), ("cr", "CR"), ("sr", "SR"), ("vf", "vf")):
        tr = trend(rows, key, limits.trend_lookback)
        if tr is None:
            out.append(f"  {label:8s} —")
            continue
        out.append(
            f"  {label:8s} {tr.first:.4f} → {tr.last:.4f}  "
            f"Δ={tr.delta:+.4f}  (跨 {tr.span} iter，要求 {tr.requested_lookback})"
        )
    out.append("")

    out += ["【5】GPU"]
    out += [f"  {line}" for line in gpu_lines] or ["  （查不到）"]
    for pid, mem in gpu_procs:
        tag = " ← 本 run" if pid in pids else " ← 他人 process"
        out.append(f"  pid {pid:>8}  {mem}{tag}")
    out.append("")

    out += ["【6】REPLAY-MIX（占用率，非表現）", f"  {replay_mix or '（無）'}", ""]

    out += ["【7】錯誤掃描"]
    if errors:
        out += [f"  ⚠ {line[:160]}" for line in errors]
    else:
        out.append("  無 Traceback / CUDA / OOM / NaN / assert")
    out.append("")

    out += ["【8】判定明細"]
    if findings:
        for finding in sorted(
            findings, key=lambda f: -LEVEL_ORDER.get(f.level, 0)
        ):
            out.append(f"  [{finding.level}] {finding.code}: {finding.message}")
    else:
        out.append("  全部檢查通過")
    out += ["", "本報告只回報，不會停止任何訓練。", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_report(args):
    logs_dir = Path(args.logs_dir)
    run_dir = (logs_dir / args.run) if args.run else find_active_run(logs_dir)
    now_text = time.strftime("%F %T %Z")
    if run_dir is None or not (run_dir / METRICS_FILENAME).is_file():
        message = (
            f"{'=' * 78}\n"
            f"訓練詳細監督報告  {now_text}\n"
            f"找不到任何含 {METRICS_FILENAME} 的 run（logs_dir={logs_dir}）\n"
        )
        return None, message

    run_name = run_dir.name
    rows = load_rows(run_dir / METRICS_FILENAME)
    console_log = logs_dir / f"{run_name}.console.log"
    errors, replay_mix, log_size = scan_console(console_log)
    pids = detect_trainer_pids()
    process_alive = bool(pids)
    gpu_lines, gpu_procs = gpu_summary()

    last = rows[-1] if rows else {}
    observation = {
        "run": run_name,
        "observed_at": time.time(),
        "iteration": last.get("iteration"),
        "log_size": log_size,
    }
    previous = previous_for_run(args.state, run_name)
    rate = (
        iteration_rate(previous, observation)
        if isinstance(observation["iteration"], int)
        else None
    )

    findings = list(evaluate(rows))
    findings += check_process(
        last.get("iteration"), last.get("iterations_target"), process_alive
    )
    if isinstance(observation["iteration"], int):
        findings += check_stall(previous, observation, process_alive)

    text = render_report(
        run_name=run_name,
        rows=rows,
        process_alive=process_alive,
        gpu_lines=gpu_lines,
        gpu_procs=gpu_procs,
        errors=errors,
        replay_mix=replay_mix,
        rate=rate,
        findings=findings,
        now_text=now_text,
        pids=pids,
    )
    return observation, text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-dir", default=str(LOGS_DIR_DEFAULT))
    parser.add_argument("--run", default=None, help="指定 run 目錄名，預設自動挑最新")
    parser.add_argument(
        "--state", default=str(STATE_DIR_DEFAULT / "detail_report_state.json")
    )
    parser.add_argument(
        "--output", default=str(STATE_DIR_DEFAULT / "detail_10min.log")
    )
    parser.add_argument(
        "--latest", default=str(STATE_DIR_DEFAULT / "detail_latest.txt")
    )
    parser.add_argument("--stdout", action="store_true", help="只印出，不寫檔")
    args = parser.parse_args(argv)

    observation, text = build_report(args)
    if args.stdout:
        print(text)
        return 0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")
    Path(args.latest).write_text(text, encoding="utf-8")
    if observation:
        write_state(args.state, observation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
