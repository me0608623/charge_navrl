#!/usr/bin/env python3
"""Convergence-driven, fail-closed SA stage advancement supervisor."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path("/home/aa/IsaacLab")
STATE_DIR = REPO / "logs" / "training_supervisor"
STATE_FILE = STATE_DIR / "auto_advance_state.json"
EXPECTED_RUN_FILE = STATE_DIR / "expected_run.txt"
STATUS_FILE = STATE_DIR / "status.txt"
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
CONDA_ENV = PYTHON.parent.parent
TRAINER = REPO / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"
VALIDATOR = REPO / "scripts/reinforcement_learning/skrl/rnn_car_wdclean/validate_checkpoint.sh"
CONFIG_DIR = REPO / "scripts/reinforcement_learning/skrl/rnn_car_modular/configs"

# Must stay aligned with validate_gates.py::STAGE_THRESH[*]["det_sr"].
DET_SR_THRESH = {1: 0.98, 2: 0.96, 3: 0.94, 4: 0.92, 5: 0.90, 6: 0.88, 7: 0.86, 8: 0.90}
TERMINAL_PHASES = {"HALTED_ALERT", "COMPLETE"}
DRY_RUN = False


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def isaaclab_subprocess_env() -> dict[str, str]:
    """Build a deterministic env_isaaclab environment for non-interactive jobs."""
    env = os.environ.copy()
    env["CONDA_PREFIX"] = str(CONDA_ENV)
    env["CONDA_DEFAULT_ENV"] = CONDA_ENV.name
    path_entries = [entry for entry in env.get("PATH", "").split(os.pathsep) if entry]
    conda_bin = str(CONDA_ENV / "bin")
    env["PATH"] = os.pathsep.join([conda_bin, *[entry for entry in path_entries if entry != conda_bin]])
    return env


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value.rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _set_phase(state: dict[str, Any], phase: str, detail: str) -> None:
    state["phase"] = phase
    state["detail"] = detail
    state["updated_at"] = _now()
    if not DRY_RUN:
        _write_text_atomic(
            STATUS_FILE,
            f"{phase} SA{state['stage']} {state['run_name']}: {detail}",
        )


def _proc_cmdline(pid: int) -> str:
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""


def find_training_pid(run_name: str) -> int | None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        cmd = _proc_cmdline(int(entry.name))
        if "train_rnn_car_wdclip.py" in cmd and re.search(
            rf"--run_name(?:=|\s+){re.escape(run_name)}(?:\s|$)", cmd
        ):
            return int(entry.name)
    return None


def pid_alive(pid: int | None) -> bool:
    return bool(pid and Path(f"/proc/{pid}").exists())


def read_metrics(path: Path) -> list[dict[str, float]]:
    """Read JSONL and retain the newest record for each iteration."""
    by_iteration: dict[int, dict[str, float]] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except FileNotFoundError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
            iteration = int(row["iteration"])
            by_iteration[iteration] = row
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return [by_iteration[key] for key in sorted(by_iteration)]


def _window_means(rows: list[dict[str, float]], key: str, window_size: int, count: int) -> list[float]:
    tail = rows[-window_size * count :]
    means = []
    for start in range(0, len(tail), window_size):
        values = [float(row[key]) for row in tail[start : start + window_size]]
        means.append(sum(values) / len(values))
    return means


def evaluate_convergence(
    rows: list[dict[str, float]],
    *,
    stage: int,
    min_iterations: int,
    window_size: int = 10,
    window_count: int = 5,
) -> dict[str, Any]:
    required = ("sr", "vf", "entropy", "kl", "clip_fraction", "encoder_grad")
    result: dict[str, Any] = {
        "converged": False,
        "iteration": int(rows[-1]["iteration"]) if rows else 0,
        "checks": {},
    }
    if not rows or result["iteration"] < min_iterations:
        result["reason"] = f"warmup {result['iteration']}/{min_iterations} iterations"
        return result
    needed = window_size * window_count
    if len(rows) < needed:
        result["reason"] = f"need {needed} metric rows, have {len(rows)}"
        return result
    tail = rows[-needed:]
    if any(key not in row for row in tail for key in required):
        result["reason"] = "missing required metric"
        return result
    if any(not math.isfinite(float(row[key])) for row in tail for key in required):
        result["reason"] = "non-finite metric"
        return result

    windows = {key: _window_means(rows, key, window_size, window_count) for key in required}
    sr_span = max(windows["sr"]) - min(windows["sr"])
    vf_mid = max(abs(sorted(windows["vf"])[len(windows["vf"]) // 2]), 1e-6)
    vf_span = (max(windows["vf"]) - min(windows["vf"])) / vf_mid
    ent_mid = max(abs(sorted(windows["entropy"])[len(windows["entropy"]) // 2]), 1e-6)
    ent_span = (max(windows["entropy"]) - min(windows["entropy"])) / ent_mid
    kl_values = [float(row["kl"]) for row in tail]
    clip_values = [float(row["clip_fraction"]) for row in tail]
    kl_median = sorted(kl_values)[len(kl_values) // 2]
    clip_mean = sum(clip_values) / len(clip_values)
    enc = windows["encoder_grad"]
    enc_mid = max(abs(sorted(enc)[len(enc) // 2]), 1e-6)
    enc_change = abs(enc[-1] - enc[0]) / enc_mid
    enc_last_ratio = enc[-1] / enc_mid

    checks = {
        "sr_high": min(windows["sr"]) >= DET_SR_THRESH[stage],
        "sr_plateau": sr_span < 0.01,
        "vf_plateau": vf_span < 0.15,
        "entropy_stable": ent_span < 0.05,
        "kl_small": kl_median < 0.01,
        "clip_small": clip_mean < 0.10,
        "encoder_active": min(enc) > 1e-3,
        "encoder_stable": enc_change < 0.35 and 0.50 <= enc_last_ratio <= 2.0,
    }
    result.update(
        converged=all(checks.values()),
        checks=checks,
        windows=windows,
        summary={
            "sr_span": sr_span,
            "vf_relative_span": vf_span,
            "entropy_relative_span": ent_span,
            "kl_median": kl_median,
            "clip_mean": clip_mean,
            "encoder_relative_change": enc_change,
        },
    )
    result["reason"] = "all convergence checks passed" if result["converged"] else "checks pending"
    return result


def latest_checkpoint(run_name: str) -> Path | None:
    candidates = []
    for path in (REPO / "logs" / "rnn_car" / run_name).glob("checkpoint_*.pt"):
        match = re.fullmatch(r"checkpoint_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def parse_wandb_id(log_path: Path) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""
    matches = re.findall(r"wandb: setting up run ([A-Za-z0-9]+)", text)
    return matches[-1] if matches else ""


def training_finished(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return False
    return "Training complete:" in text or "Training stopped cooperatively:" in text


def _gate_job_paths(state: dict[str, Any], checkpoint: Path) -> tuple[Path, Path]:
    gate_dir = STATE_DIR / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    stem = f"sa{state['stage']}_{state['run_name']}_{checkpoint.stem}"
    return gate_dir / f"{stem}.log", gate_dir / f"{stem}.exitcode"


def launch_gate(state: dict[str, Any], *, dry_run: bool = False) -> None:
    checkpoint = latest_checkpoint(state["run_name"])
    if checkpoint is None:
        _set_phase(state, "HALTED_ALERT", "training ended without a checkpoint")
        return
    if not state.get("wandb_id"):
        state["wandb_id"] = parse_wandb_id(Path(f"/tmp/{state['run_name']}.log"))
    if not state.get("wandb_id"):
        _set_phase(state, "HALTED_ALERT", "cannot run Gate1 without W&B run id")
        return
    gate_log, exit_file = _gate_job_paths(state, checkpoint)
    if exit_file.exists():
        exit_file.unlink()
    cmd = ["bash", str(VALIDATOR), str(checkpoint), str(state["stage"]), state["wandb_id"]]
    if dry_run:
        state["gate_command"] = cmd
        _set_phase(state, "GATING", f"dry-run gate for {checkpoint.name}")
        return
    with gate_log.open("w", encoding="utf-8") as output:
        wrapper = (
            '"$1" "$2" "$3" "$4" "$5"; rc=$?; '
            'printf "%s\\n" "$rc" > "$6.tmp"; mv "$6.tmp" "$6"; exit "$rc"'
        )
        proc = subprocess.Popen(
            ["bash", "-c", wrapper, "gate-job", *cmd, str(exit_file)],
            cwd=REPO,
            env=isaaclab_subprocess_env(),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state.update(
        gate_pid=proc.pid,
        gate_checkpoint=str(checkpoint),
        gate_log=str(gate_log),
        gate_exit_file=str(exit_file),
    )
    _set_phase(state, "GATING", f"validator PID {proc.pid} on {checkpoint.name}")


def launch_next_stage(state: dict[str, Any], *, dry_run: bool = False) -> None:
    stage = int(state["stage"])
    if stage >= 8:
        _set_phase(state, "COMPLETE", "SA8 passed all hard gates")
        _write_text_atomic(EXPECTED_RUN_FILE, "")
        return
    next_stage = stage + 1
    config = f"e2e_sa{next_stage}_k8_obb"
    config_path = CONFIG_DIR / f"{config}.py"
    if not config_path.is_file():
        _set_phase(state, "HALTED_ALERT", f"missing frozen next-stage config {config_path}")
        return
    checkpoint = Path(state["gate_checkpoint"])
    if not checkpoint.is_file():
        _set_phase(state, "HALTED_ALERT", f"accepted checkpoint disappeared: {checkpoint}")
        return
    if any(find_training_pid(name) for name in (state["run_name"], f"sa{next_stage}_e2e_k8_obb_acthist_future_s42")):
        _set_phase(state, "HALTED_ALERT", "refusing to launch while a training process is active")
        return

    run_name = f"sa{next_stage}_e2e_k8_obb_acthist_future_s42"
    command = [
        str(PYTHON), str(TRAINER),
        "--experiment_config", config,
        "--checkpoint", str(checkpoint),
        "--run_name", run_name,
        "--headless",
    ]
    if dry_run:
        state["next_train_command"] = command
        _set_phase(state, "ADVANCING", f"dry-run launch SA{next_stage}")
        return
    log_path = Path(f"/tmp/{run_name}.log")
    run_dir = REPO / "logs" / "rnn_car" / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        _set_phase(state, "HALTED_ALERT", f"next-stage run directory already exists: {run_dir}")
        return
    env = isaaclab_subprocess_env()
    env["PYTHONUNBUFFERED"] = "1"
    env["CHARGE_SUPERVISOR_STOP_FILE"] = str(run_dir / "supervisor_stop.request")
    with log_path.open("w", encoding="utf-8") as output:
        proc = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state.clear()
    state.update(
        enabled=True,
        phase="TRAINING",
        stage=next_stage,
        run_name=run_name,
        config=config,
        wandb_id="",
        train_pid=proc.pid,
        parent_checkpoint=str(checkpoint),
        min_iterations=420,
        window_size=10,
        window_count=5,
        created_at=_now(),
        updated_at=_now(),
        detail=f"launched from accepted SA{stage} {checkpoint.name}",
    )
    _write_text_atomic(EXPECTED_RUN_FILE, run_name)
    _write_text_atomic(STATUS_FILE, f"TRAINING SA{next_stage} {run_name}: PID {proc.pid}")


def tick(state: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    if not state.get("enabled", False) or state.get("phase") in TERMINAL_PHASES:
        return state
    run_name = state["run_name"]
    log_path = Path(f"/tmp/{run_name}.log")
    if not state.get("wandb_id"):
        state["wandb_id"] = parse_wandb_id(log_path)
    pid = find_training_pid(run_name)
    state["train_pid"] = pid

    if state["phase"] == "TRAINING":
        metrics_path = REPO / "logs" / "rnn_car" / run_name / "supervisor_metrics.jsonl"
        rows = read_metrics(metrics_path)
        convergence = evaluate_convergence(
            rows,
            stage=int(state["stage"]),
            min_iterations=int(state.get("min_iterations", 420)),
            window_size=int(state.get("window_size", 10)),
            window_count=int(state.get("window_count", 5)),
        )
        state["convergence"] = convergence
        if pid is not None and convergence["converged"]:
            stop_file = REPO / "logs" / "rnn_car" / run_name / "supervisor_stop.request"
            if not dry_run:
                _write_text_atomic(stop_file, f"converged at iteration {convergence['iteration']} {_now()}")
            _set_phase(state, "STOP_REQUESTED", f"converged at iteration {convergence['iteration']}")
        elif pid is None:
            if training_finished(log_path):
                launch_gate(state, dry_run=dry_run)
            else:
                _set_phase(state, "HALTED_ALERT", "training process disappeared without a clean completion marker")
        else:
            _set_phase(state, "TRAINING", convergence["reason"])

    elif state["phase"] == "STOP_REQUESTED":
        if pid is None:
            if training_finished(log_path):
                launch_gate(state, dry_run=dry_run)
            else:
                _set_phase(state, "HALTED_ALERT", "process exited before cooperative stop completed")
        else:
            _set_phase(state, "STOP_REQUESTED", "waiting for rollout-boundary checkpoint and clean exit")

    elif state["phase"] == "GATING":
        exit_file = Path(state.get("gate_exit_file", ""))
        if exit_file.is_file():
            try:
                rc = int(exit_file.read_text(encoding="utf-8").strip())
            except ValueError:
                rc = 255
            state["gate_exit_code"] = rc
            if rc == 0:
                _set_phase(state, "ADVANCING", "all applicable hard gates passed")
                launch_next_stage(state, dry_run=dry_run)
            else:
                _set_phase(state, "HALTED_ALERT", f"hard gate failed with exit code {rc}; no advancement")
        elif not pid_alive(state.get("gate_pid")):
            _set_phase(state, "HALTED_ALERT", "gate process disappeared without an exit-code artifact")
        else:
            _set_phase(state, "GATING", f"validator PID {state.get('gate_pid')} still running")

    state["updated_at"] = _now()
    return state


def command_init(args: argparse.Namespace) -> int:
    if args.stage not in DET_SR_THRESH:
        raise SystemExit("stage must be in 1..8")
    state = {
        "enabled": True,
        "phase": "TRAINING",
        "stage": args.stage,
        "run_name": args.run_name,
        "config": args.config or f"e2e_sa{args.stage}_k8_obb",
        "wandb_id": args.wandb_id,
        "train_pid": find_training_pid(args.run_name),
        "min_iterations": args.min_iterations,
        "window_size": args.window_size,
        "window_count": args.window_count,
        "created_at": _now(),
        "updated_at": _now(),
        "detail": "auto-advance supervision initialized",
    }
    _write_json_atomic(STATE_FILE, state)
    _write_text_atomic(EXPECTED_RUN_FILE, args.run_name)
    _write_text_atomic(STATUS_FILE, f"TRAINING SA{args.stage} {args.run_name}: auto-advance enabled")
    print(json.dumps(state, indent=2))
    return 0


def main() -> int:
    global DRY_RUN
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--stage", type=int, required=True)
    init.add_argument("--run-name", required=True)
    init.add_argument("--config", default="")
    init.add_argument("--wandb-id", default="")
    init.add_argument("--min-iterations", type=int, default=420)
    init.add_argument("--window-size", type=int, default=10)
    init.add_argument("--window-count", type=int, default=5)
    tick_parser = sub.add_parser("tick")
    tick_parser.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "init":
        return command_init(args)
    if not STATE_FILE.is_file():
        print("auto-advance state not initialized")
        return 0
    state = _load_json(STATE_FILE)
    if args.command == "tick":
        DRY_RUN = args.dry_run
        state = tick(state, dry_run=args.dry_run)
        if not args.dry_run:
            _write_json_atomic(STATE_FILE, state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
