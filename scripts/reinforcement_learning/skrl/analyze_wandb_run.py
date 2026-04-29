#!/usr/bin/env python3
"""Analyze train_rnn_car runs from local WandB artifacts and optional WandB API.

Supports three workflows:
1. monitor-log: parse logs/ablation_4way.log and summarize current health
2. report: summarize one or more runs, generate plots when history is available
3. compare: compare multiple runs and rank them on key metrics

This script is designed for the IsaacLab train_rnn_car.py workflow and understands
the current rl/*, aux/*, charge/*, curriculum/* metric layout.

Quick usage:
    # Monitor live log
    python analyze_wandb_run.py monitor-log

    # Report single run (uses WandB API by default)
    python analyze_wandb_run.py report --run-name abl_step_normal_s1

    # Compare multiple runs
    python analyze_wandb_run.py compare --run-name abl_step_normal_s1 --run-name abl_tbptt_normal_s1

    # Compare with regex pattern
    python analyze_wandb_run.py compare --pattern "abl_.*_s1"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Avoid importing local ./wandb directory as a namespace package when using the API.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path = [p for p in sys.path if Path(p).resolve() != _REPO_ROOT]
try:
    import wandb as wandb_sdk  # type: ignore
except Exception:
    wandb_sdk = None


DEFAULT_KEYS = [
    "rl/return_mean",
    "rl/success_rate",
    "rl/collision_rate",
    "rl/timeout_rate",
    "rl/value_loss",
    "rl/variance_explained",
    "aux/preprocess_loss",
    "aux/loss_per_step",
    "aux/rnn_param_delta_norm",
    "aux/valid_seq_count",
    "aux/mode",
    "aux/seq_len",
]

PLOT_KEYS = [
    "rl/success_rate",
    "rl/collision_rate",
    "rl/return_mean",
    "rl/variance_explained",
    "aux/loss_per_step",
    "aux/preprocess_loss",
]


def _unwrap_cfg(v: Any) -> Any:
    if isinstance(v, dict) and "value" in v:
        return v["value"]
    return v


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        if math.isfinite(float(v)):
            return float(v)
        return None
    return None


@dataclass
class LocalRun:
    run_dir: Path
    run_id: str
    run_name: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    output_log: Path | None = None
    inferred_log_dir: str | None = None


def discover_local_runs(wandb_root: Path) -> list[LocalRun]:
    runs: list[LocalRun] = []
    if not wandb_root.exists():
        return runs
    for run_dir in sorted(wandb_root.glob("run-*")):
        files_dir = run_dir / "files"
        cfg_path = files_dir / "config.yaml"
        summary_path = files_dir / "wandb-summary.json"
        output_path = files_dir / "output.log"
        config: dict[str, Any] = {}
        summary: dict[str, Any] = {}
        run_name = None
        if cfg_path.exists():
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw_cfg = yaml.safe_load(f) or {}
                config = {k: _unwrap_cfg(v) for k, v in raw_cfg.items()}
                run_name = config.get("run_name")
            except Exception:
                pass
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                pass
        inferred_log_dir = None
        if output_path.exists():
            try:
                txt = output_path.read_text(encoding="utf-8", errors="ignore")
                m = re.findall(r"logs/rnn_car/([^/\s]+)/checkpoint_\d+\.pt", txt)
                if m:
                    inferred_log_dir = m[-1]
            except Exception:
                pass
        runs.append(
            LocalRun(
                run_dir=run_dir,
                run_id=run_dir.name.split("-")[-1],
                run_name=run_name,
                summary=summary,
                config=config,
                output_log=output_path if output_path.exists() else None,
                inferred_log_dir=inferred_log_dir,
            )
        )
    return runs


def find_local_runs(run_names: list[str], wandb_root: Path) -> list[LocalRun]:
    local_runs = discover_local_runs(wandb_root)
    matched: list[LocalRun] = []
    for name in run_names:
        candidates = [
            r for r in local_runs
            if r.run_name == name
            or r.inferred_log_dir == name
            or r.run_id == name
            or r.run_dir.name == name
        ]
        if candidates:
            matched.append(sorted(candidates, key=lambda r: r.run_dir.name)[-1])
    return matched


DEFAULT_ENTITY = "me0608623-none"
DEFAULT_PROJECT = "charge_skrl"


def fetch_api_run(entity: str, project: str, run_name: str):
    if wandb_sdk is None:
        return None
    api = wandb_sdk.Api(timeout=30)
    path = f"{entity}/{project}"
    # Use display_name filter for efficient lookup (no full scan)
    try:
        runs = list(api.runs(path, filters={"display_name": run_name}, per_page=5))
        if runs:
            return runs[-1]  # most recent if duplicates
    except Exception:
        pass
    # Fallback: try config.run_name or run id
    try:
        runs = list(api.runs(path, filters={"config.run_name": run_name}, per_page=5))
        if runs:
            return runs[-1]
    except Exception:
        pass
    return None


def fetch_api_runs_by_pattern(entity: str, project: str, pattern: str) -> list:
    """Fetch runs matching a regex pattern."""
    if wandb_sdk is None:
        return []
    api = wandb_sdk.Api(timeout=30)
    path = f"{entity}/{project}"
    try:
        return list(api.runs(path, filters={"display_name": {"$regex": pattern}}, per_page=50))
    except Exception:
        return []


def fetch_history(api_run, keys: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for row in api_run.scan_history(keys=list(set(keys + ["_step"]))):
            rows.append(dict(row))
    except Exception:
        return []
    return rows


def metric_series(history: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for i, row in enumerate(history):
        y = _safe_float(row.get(key))
        if y is None:
            continue
        x = _safe_float(row.get("_step"))
        xs.append(float(i if x is None else x))
        ys.append(y)
    return xs, ys


def simple_trend(values: list[float]) -> str:
    if len(values) < 6:
        return "insufficient"
    n = len(values)
    first = sum(values[: max(2, n // 5)]) / max(2, n // 5)
    last = sum(values[-max(2, n // 5):]) / max(2, n // 5)
    delta = last - first
    scale = max(abs(first), 1e-6)
    rel = delta / scale
    if rel > 0.1:
        return "up"
    if rel < -0.1:
        return "down"
    return "flat"


def health_summary(latest: dict[str, float | None], trends: dict[str, str]) -> list[str]:
    msgs: list[str] = []
    sr = latest.get("rl/success_rate")
    cr = latest.get("rl/collision_rate")
    ve = latest.get("rl/variance_explained")
    aux = latest.get("aux/loss_per_step")
    rnn_delta = latest.get("aux/rnn_param_delta_norm")

    if sr is not None and cr is not None:
        if sr > 0.6 and cr < 0.3:
            msgs.append("RL 表現偏健康：success 高且 collision 低")
        elif sr < 0.2 and cr > 0.5:
            msgs.append("RL 可能卡住：success 低且 collision 高")
    if ve is not None:
        if ve > 0.4:
            msgs.append("critic 品質不錯：variance_explained 偏高")
        elif ve < 0:
            msgs.append("critic 偏弱：variance_explained < 0")
    if aux is not None:
        trend = trends.get("aux/loss_per_step")
        if trend == "down":
            msgs.append("aux module 在學：loss_per_step 有下降")
        elif trend == "flat":
            msgs.append("aux module 學習有限：loss_per_step 近乎持平")
    if rnn_delta is not None:
        if rnn_delta > 0:
            msgs.append("RNN cell 有實際更新（param delta > 0）")
        else:
            msgs.append("RNN cell 沒有明顯更新（請檢查 aux path / lr）")
    return msgs


def plot_histories(run_histories: dict[str, list[dict[str, Any]]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for key in PLOT_KEYS:
        plt.figure(figsize=(8, 4))
        drawn = False
        for run_name, history in run_histories.items():
            xs, ys = metric_series(history, key)
            if ys:
                plt.plot(xs, ys, label=run_name)
                drawn = True
        if not drawn:
            plt.close()
            continue
        plt.title(key)
        plt.xlabel("step")
        plt.ylabel(key)
        plt.grid(True, alpha=0.3)
        plt.legend()
        out_path = out_dir / f"{key.replace('/', '__')}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        paths.append(out_path)
    return paths


RL_LINE_RE = re.compile(
    r"^\[(?P<iter>\d+)/(?P<total>\d+)\]\s+\w+\s+S(?P<stage>\d+)\s+\|\s+fps=(?P<fps>[\d.]+)\s+\|\s+"
    r"R=(?P<R>[-\d.]+)\s+SR=(?P<SR>[\d.]+)%\s+CR=(?P<CR>[\d.]+)%\s+TO=(?P<TO>[\d.]+)%\s+\|\s+"
    r"ppo=(?P<ppo>[-\d.]+)\s+vf=(?P<vf>[-\d.]+)\s+ent=(?P<ent>[-\d.]+)"
)
AUX_LINE_RE = re.compile(
    r"^\s+AUX\((?P<mode>\w+)\s+L=(?P<L>\d+)\):\s+loss=(?P<loss>[-\d.]+)\s+"
    r"n1d=(?P<n1d>[-\d.]+)\s+n2d=(?P<n2d>[-\d.]+)\s+valid=(?P<valid>\d+)\s+\|\s+"
    r"rnn_grad=(?P<rnn_grad>[-\d.]+)\s+rnn_delta=(?P<rnn_delta>[-\d.]+)\s+"
    r"ph_delta=(?P<ph_delta>[-\d.]+)\s+\|\s+VE=(?P<ve>[-\d.]+)"
)


def monitor_log(log_file: Path, tail: int = 120) -> str:
    if not log_file.exists():
        return f"log file not found: {log_file}"
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-tail:]

    current_block = None
    latest_rl = None
    latest_aux = None
    issues: list[str] = []
    for line in lines:
        if "==========" in line and "[" in line:
            current_block = line.strip()
        if "Traceback" in line or "[Error]" in line or "NaN" in line or "PhysX error" in line:
            issues.append(line.strip())
        m = RL_LINE_RE.match(line.strip())
        if m:
            latest_rl = m.groupdict()
        m2 = AUX_LINE_RE.match(line.rstrip("\n"))
        if m2:
            latest_aux = m2.groupdict()

    out = []
    if current_block:
        out.append(f"Current block: {current_block}")
    if latest_rl:
        out.append(
            "Latest RL: iter={iter}/{total} stage={stage} R={R} SR={SR}% CR={CR}% TO={TO}% "
            "ppo={ppo} vf={vf} ent={ent} fps={fps}".format(**latest_rl)
        )
    if latest_aux:
        out.append(
            "Latest AUX: mode={mode} L={L} loss={loss} n1d={n1d} n2d={n2d} valid={valid} "
            "rnn_grad={rnn_grad} rnn_delta={rnn_delta} ph_delta={ph_delta} VE={ve}".format(**latest_aux)
        )
    if issues:
        out.append("Recent issues:")
        out.extend(f"- {x}" for x in issues[-10:])
    else:
        out.append("Recent issues: none detected in tail window")
    return "\n".join(out)


def parse_latest_from_output_log(output_log: Path) -> dict[str, float]:
    latest: dict[str, float] = {}
    if not output_log.exists():
        return latest
    try:
        lines = output_log.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return latest
    for line in lines:
        m = RL_LINE_RE.match(line.strip())
        if m:
            g = m.groupdict()
            latest.update({
                "rl/return_mean": float(g["R"]),
                "rl/success_rate": float(g["SR"]) / 100.0,
                "rl/collision_rate": float(g["CR"]) / 100.0,
                "rl/timeout_rate": float(g["TO"]) / 100.0,
                "rl/policy_loss": float(g["ppo"]),
                "rl/value_loss": float(g["vf"]),
                "rl/entropy": float(g["ent"]),
                "rl/stage_idx": float(g["stage"]),
                "train/fps": float(g["fps"]),
            })
        m2 = AUX_LINE_RE.match(line.rstrip("\n"))
        if m2:
            g = m2.groupdict()
            latest.update({
                "aux/preprocess_loss": float(g["loss"]),
                "aux/near1_d_loss": float(g["n1d"]),
                "aux/near2_d_loss": float(g["n2d"]),
                "aux/valid_seq_count": float(g["valid"]),
                "aux/rnn_grad_norm": float(g["rnn_grad"]),
                "aux/rnn_param_delta_norm": float(g["rnn_delta"]),
                "aux/predict_head_param_delta_norm": float(g["ph_delta"]),
                "rl/variance_explained": float(g["ve"]),
                "aux/seq_len": float(g["L"]),
                "aux/mode": 1.0 if g["mode"] == "tbptt" else 0.0,
            })
    return latest


def build_report(run_names: list[str], wandb_root: Path, out_dir: Path,
                 entity: str | None, project: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_runs = {r.run_name or r.run_id: r for r in find_local_runs(run_names, wandb_root)}
    run_histories: dict[str, list[dict[str, Any]]] = {}
    sections: list[str] = []

    for name in run_names:
        local = local_runs.get(name)
        history: list[dict[str, Any]] = []
        summary = local.summary if local else {}
        config = local.config if local else {}
        source = "local-summary"
        if entity and project:
            api_run = fetch_api_run(entity, project, name)
            if api_run is not None:
                history = fetch_history(api_run, DEFAULT_KEYS)
                try:
                    summary = dict(api_run.summary)
                except Exception:
                    pass
                try:
                    cfg = dict(api_run.config)
                    cfg.setdefault("run_name", api_run.name)
                    config = cfg
                except Exception:
                    pass
                source = "wandb-api"
        run_histories[name] = history

        latest = {k: _safe_float(summary.get(k)) for k in DEFAULT_KEYS}
        if local and (all(v is None for v in latest.values()) or not summary):
            output_latest = parse_latest_from_output_log(local.output_log) if local.output_log else {}
            if output_latest:
                for k in DEFAULT_KEYS:
                    if latest.get(k) is None and k in output_latest:
                        latest[k] = output_latest[k]
                source = "local-output-log"
        trends: dict[str, str] = {}
        for key in DEFAULT_KEYS:
            _, ys = metric_series(history, key)
            trends[key] = simple_trend(ys) if ys else "summary-only"

        sections.append(f"## {name}")
        sections.append(f"- source: `{source}`")
        if local:
            sections.append(f"- local run dir: `{local.run_dir}`")
        if config:
            for k in ("task", "curriculum_version", "seed", "num_envs", "aux_mode", "zero_preprocess_feature_for_rl"):
                if k in config:
                    sections.append(f"- {k}: `{config[k]}`")
        sections.append("")
        sections.append("### 主要指標")
        for key in DEFAULT_KEYS:
            sections.append(f"- `{key}`: latest={latest.get(key)} trend={trends.get(key)}")
        sections.append("")
        sections.append("### 健康摘要")
        msgs = health_summary(latest, trends)
        if msgs:
            sections.extend(f"- {m}" for m in msgs)
        else:
            sections.append("- 資料不足，無法自動判讀")
        sections.append("")

    plot_paths = plot_histories({k: v for k, v in run_histories.items() if v}, out_dir / "plots")
    if plot_paths:
        sections.append("# 圖表")
        for p in plot_paths:
            sections.append(f"- `{p}`")
        sections.append("")

    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# analyze_wandb_run report\n\n")
        f.write("\n".join(sections))
        f.write("\n")
    return report_path


def compare_runs(run_names: list[str], wandb_root: Path, out_dir: Path,
                 entity: str | None, project: str | None) -> Path:
    report_path = build_report(run_names, wandb_root, out_dir, entity, project)
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Analyze train_rnn_car WandB runs")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mon = sub.add_parser("monitor-log", help="Parse training log and summarize current state")
    p_mon.add_argument("--log-file", default="logs/ablation_4way.log")
    p_mon.add_argument("--tail", type=int, default=160)

    p_rep = sub.add_parser("report", help="Generate report for one or more runs")
    p_rep.add_argument("--run-name", action="append", required=False, default=[])
    p_rep.add_argument("--pattern", type=str, help="Regex pattern to match run names")
    p_rep.add_argument("--wandb-root", default="wandb")
    p_rep.add_argument("--out-dir", default="logs/run_analysis")
    p_rep.add_argument("--entity", default=DEFAULT_ENTITY)
    p_rep.add_argument("--project", default=DEFAULT_PROJECT)

    p_cmp = sub.add_parser("compare", help="Compare multiple runs")
    p_cmp.add_argument("--run-name", action="append", required=False, default=[])
    p_cmp.add_argument("--pattern", type=str, help="Regex pattern to match run names")
    p_cmp.add_argument("--wandb-root", default="wandb")
    p_cmp.add_argument("--out-dir", default="logs/run_analysis_compare")
    p_cmp.add_argument("--entity", default=DEFAULT_ENTITY)
    p_cmp.add_argument("--project", default=DEFAULT_PROJECT)

    args = parser.parse_args()

    if args.cmd == "monitor-log":
        print(monitor_log(Path(args.log_file), tail=args.tail))
        return

    if args.cmd in ("report", "compare"):
        run_names = list(args.run_name or [])
        # Resolve --pattern to run names via WandB API
        if hasattr(args, "pattern") and args.pattern:
            matched = fetch_api_runs_by_pattern(args.entity, args.project, args.pattern)
            print(f"Pattern '{args.pattern}' matched {len(matched)} runs: {[r.name for r in matched]}")
            run_names.extend(r.name for r in matched)
        if not run_names:
            print("ERROR: provide --run-name or --pattern")
            sys.exit(1)

        if args.cmd == "report":
            path = build_report(run_names, Path(args.wandb_root), Path(args.out_dir), args.entity, args.project)
        else:
            path = compare_runs(run_names, Path(args.wandb_root), Path(args.out_dir), args.entity, args.project)
        print(path)
        return


if __name__ == "__main__":
    main()
