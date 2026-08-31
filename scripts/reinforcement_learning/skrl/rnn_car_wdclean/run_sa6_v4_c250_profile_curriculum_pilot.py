#!/usr/bin/env python3
"""Fail-closed launcher for the SA6 c250 matched profile-curriculum pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


REPO = Path(__file__).resolve().parents[4]
AUTHORIZATION = (
    REPO
    / "docs/freeze/"
    "sa6_v4_c250_profile_curriculum_pilot_launch_authorization_20260829.json"
)
AUTHORIZATION_SHA256 = (
    "34f19379cb81cdc607628bbf3e13e06e9891cd1bac6d95e1b5e0c314489bf5a3"
)
ISAACLAB = REPO / "isaaclab.sh"
TRAINER = REPO / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"
CONDA_PREFIX = Path("/home/aa/miniconda3/envs/env_isaaclab")
EXPECTED_RUN = REPO / "logs/training_supervisor/expected_run.txt"
SUPERVISOR_STATUS = REPO / "logs/training_supervisor/status.txt"
OUTPUT_ROOT = (
    REPO
    / "logs/gates/sa6_v4_c250_profile_curriculum_pilot/launch_20260829_r3"
)
PREFLIGHT_FAILURE = (
    REPO
    / "logs/gates/sa6_v4_c250_profile_curriculum_pilot/"
    "preflight_failure_20260829.json"
)
LOCK_PATH = Path("/tmp/sa6_v4_c250_profile_curriculum_pilot.lock")
STRICT_ERROR_RE = re.compile(
    r"Traceback|CUDA out of memory|\bOOM\b|RuntimeError|"
    r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])",
    re.IGNORECASE,
)
CORE_METRICS = ("sr", "cr", "timeout", "entropy", "kl", "clip_fraction", "vf")


@dataclass(frozen=True)
class Arm:
    label: str
    config_module: str
    run_name: str
    iterations: int
    checkpoints: tuple[str, ...]
    smoke: bool


ARMS = (
    Arm(
        label="control_smoke",
        config_module="e2e_sa6_v4_c250_profile_curriculum_control_p50_smoke",
        run_name="sa6_v4_c250_profile_curriculum_control_smoke_ne64_s42_p1_r2",
        iterations=1,
        checkpoints=("checkpoint_128.pt",),
        smoke=True,
    ),
    Arm(
        label="rung1_smoke",
        config_module="e2e_sa6_v4_c250_profile_curriculum_rung1_p50_smoke",
        run_name="sa6_v4_c250_profile_curriculum_rung1_smoke_ne64_s42_p1_r1",
        iterations=1,
        checkpoints=("checkpoint_128.pt",),
        smoke=True,
    ),
    Arm(
        label="control",
        config_module="e2e_sa6_v4_c250_profile_curriculum_control_p50",
        run_name="sa6_v4_c250_profile_curriculum_control_ne1024_s42_p50_r1",
        iterations=50,
        checkpoints=("checkpoint_3200.pt", "checkpoint_6400.pt"),
        smoke=False,
    ),
    Arm(
        label="rung1",
        config_module="e2e_sa6_v4_c250_profile_curriculum_rung1_p50",
        run_name="sa6_v4_c250_profile_curriculum_rung1_ne1024_s42_p50_r1",
        iterations=50,
        checkpoints=("checkpoint_3200.pt", "checkpoint_6400.pt"),
        smoke=False,
    ),
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _verify_authorization() -> dict[str, Any]:
    if _sha256(AUTHORIZATION) != AUTHORIZATION_SHA256:
        raise RuntimeError("launch authorization hash drifted")
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    automatic = authorization.get("automatic_actions") or {}
    execution = authorization.get("execution") or {}
    if (
        authorization.get("schema")
        != "sa6_v4_c250_profile_curriculum_pilot_launch_authorization/v1"
        or authorization.get("decision")
        != "HUMAN_AUTHORIZED_MATCHED_CONTROL_THEN_RUNG1_LAUNCH"
        or execution.get("order") != [arm.label for arm in ARMS]
        or execution.get("fail_closed") is not True
        or automatic.get("gpu_smoke_authorized") is not True
        or automatic.get("formal_control_launch_authorized") is not True
        or automatic.get("formal_rung1_launch_authorized") is not True
        or automatic.get("run_arms_sequentially") is not True
        or automatic.get("fixed_gate_auto_start") is not False
        or automatic.get("extend_beyond_50_iterations") is not False
        or automatic.get("start_sa7") is not False
        or automatic.get("enable_teacher_distillation_or_override") is not False
    ):
        raise RuntimeError("authorization does not permit this exact queue")

    parent = authorization.get("parent") or {}
    parent_path = REPO / str(parent.get("checkpoint"))
    if (
        _sha256(parent_path) != parent.get("sha256")
        or parent.get("conceptual_iteration") != 250
        or parent.get("resume_rl_optimizer") is not True
        or parent.get("rl_optimizer_state_entries") != 38
    ):
        raise RuntimeError("c250 parent contract drifted")

    for item in (authorization.get("source_locks") or {}).values():
        path = REPO / str(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise RuntimeError(f"source lock drifted: {path}")
    return authorization


def _active_trainers() -> list[str]:
    matches: list[str] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if "train_rnn_car_wdclip.py" in command:
            matches.append(f"{entry.name} {command.strip()}")
    return matches


def _gpu_compute_apps() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _preflight() -> dict[str, Any]:
    authorization = _verify_authorization()
    trainers = _active_trainers()
    if trainers:
        raise RuntimeError(f"training process already active: {trainers}")
    expected = EXPECTED_RUN.read_text(encoding="utf-8").strip() if EXPECTED_RUN.exists() else ""
    if expected:
        raise RuntimeError(f"expected_run is not empty: {expected}")
    compute_apps = _gpu_compute_apps()
    disallowed = [line for line in compute_apps if "gnome-remote-desktop-daemon" not in line]
    if disallowed:
        raise RuntimeError(f"GPU has non-desktop compute apps: {disallowed}")
    collisions = []
    for arm in ARMS:
        run_dir = REPO / "logs/rnn_car" / arm.run_name
        console = REPO / "logs/rnn_car" / f"{arm.run_name}.console.log"
        if run_dir.exists() or console.exists():
            collisions.append(str(run_dir if run_dir.exists() else console))
    if collisions:
        raise RuntimeError(f"refusing to overwrite prior outputs: {collisions}")
    return {"authorization": authorization, "gpu_compute_apps": compute_apps}


def _strict_errors(console: Path) -> list[str]:
    matches = []
    for line in console.read_text(encoding="utf-8", errors="replace").splitlines():
        if STRICT_ERROR_RE.search(line):
            matches.append(line[-500:])
    return matches


def _verify_arm(arm: Arm, console: Path) -> dict[str, Any]:
    run_dir = REPO / "logs/rnn_car" / arm.run_name
    metrics_path = run_dir / "supervisor_metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError(f"{arm.label} produced no supervisor metrics")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(rows) != arm.iterations:
        raise RuntimeError(
            f"{arm.label} metrics rows {len(rows)} != target {arm.iterations}"
        )
    final = rows[-1]
    if (
        final.get("iteration") != arm.iterations
        or final.get("iterations_target") != arm.iterations
    ):
        raise RuntimeError(f"{arm.label} did not reach its target iteration")
    for row in rows:
        for key in CORE_METRICS:
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"{arm.label} has invalid {key}: {value!r}")
        if row.get("corridor_family/accounting/reconciliation_ok") != 1.0:
            raise RuntimeError(f"{arm.label} family accounting failed")
        if row.get("corridor_profile_family/accounting/reconciliation_ok") != 1.0:
            raise RuntimeError(f"{arm.label} profile-family accounting failed")
    errors = _strict_errors(console)
    if errors:
        raise RuntimeError(f"{arm.label} strict error scan failed: {errors[-3:]}")
    text = console.read_text(encoding="utf-8", errors="replace")
    if "[INFO] Loaded optimizer state: charge_opt_rl" not in text:
        raise RuntimeError(f"{arm.label} did not load the RL optimizer")
    if "[INFO] Loaded checkpoint:" not in text:
        raise RuntimeError(f"{arm.label} did not load its checkpoint")

    checkpoints = []
    for filename in arm.checkpoints:
        checkpoint = run_dir / filename
        if not checkpoint.is_file():
            raise RuntimeError(f"{arm.label} missing checkpoint: {filename}")
        checkpoints.append({"filename": filename, "sha256": _sha256(checkpoint)})
    return {
        "label": arm.label,
        "run_name": arm.run_name,
        "smoke": arm.smoke,
        "completed_at": _now(),
        "metrics_rows": len(rows),
        "final_metric": {key: final[key] for key in CORE_METRICS},
        "family_reconciliation_ok": True,
        "profile_family_reconciliation_ok": True,
        "strict_error_scan_count": 0,
        "checkpoints": checkpoints,
    }


def _run_arm(arm: Arm) -> dict[str, Any]:
    _verify_authorization()
    if _active_trainers():
        raise RuntimeError(f"another trainer appeared before {arm.label}")
    run_dir = REPO / "logs/rnn_car" / arm.run_name
    console = REPO / "logs/rnn_car" / f"{arm.run_name}.console.log"
    if run_dir.exists() or console.exists():
        raise RuntimeError(f"refusing to overwrite output for {arm.run_name}")

    _atomic_text(EXPECTED_RUN, arm.run_name + "\n")
    _atomic_text(SUPERVISOR_STATUS, f"TRAINING {arm.run_name}\n")
    command = [
        str(ISAACLAB),
        "-p",
        str(TRAINER.relative_to(REPO)),
        "--experiment_config",
        arm.config_module,
        "--run_name",
        arm.run_name,
        "--headless",
    ]
    print(f"[{_now()}] START {arm.label}: {' '.join(command)}", flush=True)
    environment = os.environ.copy()
    environment["CONDA_PREFIX"] = str(CONDA_PREFIX)
    environment["PATH"] = f"{CONDA_PREFIX / 'bin'}:{environment.get('PATH', '')}"
    environment["PYTHONUNBUFFERED"] = "1"
    with console.open("wb") as stream:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{arm.label} exited with code {completed.returncode}")
    result = _verify_arm(arm, console)
    print(f"[{_now()}] COMPLETE {arm.label}", flush=True)
    time.sleep(3)
    return result


def _clear_expected_run() -> None:
    _atomic_text(EXPECTED_RUN, "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another SA6-v4 pilot queue holds the lock", file=sys.stderr)
            return 2

        try:
            preflight = _preflight()
            if args.dry_run:
                print(
                    json.dumps(
                        {
                            "status": "DRY_RUN_OK",
                            "authorization_sha256": AUTHORIZATION_SHA256,
                            "order": [arm.label for arm in ARMS],
                            "runs": [arm.run_name for arm in ARMS],
                            "gpu_compute_apps": preflight["gpu_compute_apps"],
                        },
                        indent=2,
                    )
                )
                return 0

            OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
            state: dict[str, Any] = {
                "schema": "sa6_v4_c250_profile_curriculum_pilot_launch/v1",
                "status": "RUNNING",
                "started_at": _now(),
                "authorization_sha256": AUTHORIZATION_SHA256,
                "arms": [],
                "fixed_gate_auto_start": False,
                "auto_extend": False,
                "sa7_started": False,
            }
            _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)
            for arm in ARMS:
                state["current_arm"] = arm.label
                _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)
                state["arms"].append(_run_arm(arm))
                _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)
            state["status"] = "COMPLETE_AWAITING_FIXED_SCREEN"
            state["completed_at"] = _now()
            state.pop("current_arm", None)
            _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)
            _clear_expected_run()
            _atomic_text(
                SUPERVISOR_STATUS,
                "IDLE SA6-v4 c250 matched pilot complete; fixed screen not started\n",
            )
            return 0
        except Exception as error:
            failure = {
                "schema": "sa6_v4_c250_profile_curriculum_pilot_failure/v1",
                "status": "INCOMPLETE_NO_VERDICT",
                "failed_at": _now(),
                "error": f"{type(error).__name__}: {error}",
                "fixed_gate_started": False,
                "sa7_started": False,
            }
            if args.dry_run:
                _atomic_json(PREFLIGHT_FAILURE, failure)
            else:
                OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
                _atomic_json(OUTPUT_ROOT / "INCOMPLETE_NO_VERDICT.json", failure)
            _clear_expected_run()
            _atomic_text(SUPERVISOR_STATUS, f"HALTED {failure['error']}\n")
            print(failure["error"], file=sys.stderr, flush=True)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
