#!/usr/bin/env python3
"""Fail-closed smoke and training queue for the paper realism factorial."""

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
FREEZE = REPO / "docs/freeze/paper_realism_factorial_v1_ready_not_run_20260830.json"
FREEZE_SHA256 = "e36949071245b271ee010cd06be2e9c1e3a1e2ba5d073bcb908fd4fad0297b6d"
AUTHORIZATION = REPO / "docs/freeze/paper_realism_factorial_v1_launch_authorization_20260830.json"
AUTHORIZATION_SHA256 = "bb7ce644737079d1c9a08a9f959c1fb1d2ec3e3b7d8ed29b3bc07db086474719"
ISAACLAB = REPO / "isaaclab.sh"
TRAINER = REPO / "scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py"
CONDA_PREFIX = Path("/home/aa/miniconda3/envs/env_isaaclab")
EXPECTED_RUN = REPO / "logs/training_supervisor/expected_run.txt"
SUPERVISOR_STATUS = REPO / "logs/training_supervisor/status.txt"
OUTPUT_ROOT = REPO / "logs/gates/paper_realism_factorial_v1/launch_20260830_r1"
LOCK_PATH = Path("/tmp/paper_realism_factorial_v1.lock")
GPU_POLL_SECONDS = 60
GPU_STABLE_SECONDS = 30
STRICT_ERROR_RE = re.compile(
    r"Traceback|CUDA out of memory|\bOOM\b|RuntimeError|"
    r"(?<![A-Za-z])(?:nan|inf)(?![A-Za-z])",
    re.IGNORECASE,
)
CORE_METRICS = ("sr", "cr", "timeout", "entropy", "kl", "clip_fraction", "vf")


@dataclass(frozen=True)
class Arm:
    key: str
    cell: str
    config_module: str
    run_name: str
    seed: int
    iterations: int
    checkpoints: tuple[str, ...]
    smoke: bool
    lidar_mode: str
    delay_enabled: bool


def _arm(
    cell: str,
    *,
    seed: int,
    smoke: bool,
    lidar_mode: str,
    delay_enabled: bool,
) -> Arm:
    factor = {
        "A": "a_ideal_d0",
        "B": "b_full_d0",
        "C": "c_ideal_u012",
        "D": "d_full_u012",
    }[cell]
    suffix = "_smoke" if smoke else ""
    config_module = f"e2e_sa1_paper_factorial_{factor}_p600{suffix}"
    if smoke:
        run_name = f"paper_realism_v1_{factor}_smoke_ne64_s{seed}_p1_r1"
        return Arm(
            key=f"smoke_{cell}",
            cell=cell,
            config_module=config_module,
            run_name=run_name,
            seed=seed,
            iterations=1,
            checkpoints=("checkpoint_128.pt",),
            smoke=True,
            lidar_mode=lidar_mode,
            delay_enabled=delay_enabled,
        )
    run_name = f"paper_realism_v1_{factor}_ne1024_s{seed}_p600_r1"
    return Arm(
        key=f"formal_s{seed}_{cell}",
        cell=cell,
        config_module=config_module,
        run_name=run_name,
        seed=seed,
        iterations=600,
        checkpoints=tuple(f"checkpoint_{iteration * 128}.pt" for iteration in range(100, 601, 100)),
        smoke=False,
        lidar_mode=lidar_mode,
        delay_enabled=delay_enabled,
    )


_CELL_RUNTIME = {
    "A": ("ideal", False),
    "B": ("full", False),
    "C": ("ideal", True),
    "D": ("full", True),
}


def _make_arm(cell: str, seed: int, smoke: bool) -> Arm:
    lidar_mode, delay_enabled = _CELL_RUNTIME[cell]
    return _arm(
        cell,
        seed=seed,
        smoke=smoke,
        lidar_mode=lidar_mode,
        delay_enabled=delay_enabled,
    )


SMOKE_ARMS = tuple(_make_arm(cell, 42, True) for cell in "ABCD")
# Pre-registered blocked randomization generated with random.Random(20260830).
FORMAL_ORDER = ((42, "CBDA"), (43, "BDCA"), (44, "ABDC"))
FORMAL_ARMS = tuple(
    _make_arm(cell, seed, False)
    for seed, order in FORMAL_ORDER
    for cell in order
)
ARMS = SMOKE_ARMS + FORMAL_ARMS


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


def _verify_file_locks(files: dict[str, str]) -> None:
    for relative, expected in files.items():
        path = REPO / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"source lock drifted: {relative}")


def _verify_authorization() -> dict[str, Any]:
    if _sha256(FREEZE) != FREEZE_SHA256:
        raise RuntimeError("frozen paper protocol drifted")
    if _sha256(AUTHORIZATION) != AUTHORIZATION_SHA256:
        raise RuntimeError("launch authorization hash drifted")

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    execution = authorization.get("execution") or {}
    automatic = authorization.get("automatic_actions") or {}
    if (
        authorization.get("schema") != "paper_realism_factorial_launch_authorization/v1"
        or authorization.get("decision") != "HUMAN_AUTHORIZED_SMOKE_THEN_12_FORMAL_RUNS"
        or authorization.get("frozen_protocol_sha256") != FREEZE_SHA256
        or execution.get("order") != [arm.key for arm in ARMS]
        or execution.get("fail_closed") is not True
        or execution.get("wait_for_exclusive_gpu") is not True
        or automatic.get("gpu_smoke_authorized") is not True
        or automatic.get("formal_training_authorized") is not True
        or automatic.get("run_sequentially") is not True
        or automatic.get("fixed_gate_launch_authorized") is not False
        or automatic.get("auto_advance_authorized") is not False
    ):
        raise RuntimeError("authorization does not permit this exact queue")
    if freeze["training_contract"]["training_seeds"] != [42, 43, 44]:
        raise RuntimeError("frozen training seed contract drifted")
    _verify_file_locks(freeze["source_locks"]["files"])
    _verify_file_locks(authorization["source_locks"]["files"])
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


def _gpu_blockers() -> list[str]:
    return [
        line
        for line in _gpu_compute_apps()
        if "gnome-remote-desktop-daemon" not in line
    ]


def _expected_run() -> str:
    return EXPECTED_RUN.read_text(encoding="utf-8").strip() if EXPECTED_RUN.exists() else ""


def _run_collisions() -> list[str]:
    collisions = []
    for arm in ARMS:
        run_dir = REPO / "logs/rnn_car" / arm.run_name
        console = REPO / "logs/rnn_car" / f"{arm.run_name}.console.log"
        if run_dir.exists():
            collisions.append(str(run_dir))
        if console.exists():
            collisions.append(str(console))
    return collisions


def _static_preflight() -> dict[str, Any]:
    authorization = _verify_authorization()
    collisions = _run_collisions()
    if collisions:
        raise RuntimeError(f"refusing to overwrite prior outputs: {collisions}")
    return {
        "authorization": authorization,
        "active_trainers": _active_trainers(),
        "expected_run": _expected_run(),
        "gpu_compute_apps": _gpu_compute_apps(),
        "gpu_blockers": _gpu_blockers(),
    }


def _write_waiting_state(state: dict[str, Any], blockers: list[str]) -> None:
    state["status"] = "WAITING_FOR_EXCLUSIVE_GPU"
    state["last_checked_at"] = _now()
    state["gpu_blockers"] = blockers
    state["active_trainers"] = _active_trainers()
    state["expected_run"] = _expected_run()
    _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)


def _wait_for_exclusive_gpu(state: dict[str, Any]) -> None:
    while True:
        _verify_authorization()
        blockers = _gpu_blockers()
        trainers = _active_trainers()
        expected = _expected_run()
        if not blockers and not trainers and not expected:
            state["status"] = "VERIFYING_GPU_STABLE"
            state["last_checked_at"] = _now()
            state["gpu_blockers"] = []
            state["active_trainers"] = []
            state["expected_run"] = ""
            _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)
            time.sleep(GPU_STABLE_SECONDS)
            if not _gpu_blockers() and not _active_trainers() and not _expected_run():
                return
            continue
        _write_waiting_state(state, blockers)
        print(
            f"[{_now()}] WAIT GPU blockers={blockers} "
            f"trainers={len(trainers)} expected_run={expected!r}",
            flush=True,
        )
        time.sleep(GPU_POLL_SECONDS)


def _strict_errors(console: Path) -> list[str]:
    matches = []
    for line in console.read_text(encoding="utf-8", errors="replace").splitlines():
        if STRICT_ERROR_RE.search(line):
            matches.append(line[-500:])
    return matches


def _required_runtime_markers(arm: Arm) -> tuple[str, ...]:
    markers = [
        f"[MODEL-INIT-SEED] requested=-1 resolved={arm.seed} run_seed={arm.seed}",
        "[NO_DR] domain_randomization event: physics/sensor_noise/external_force = False",
        f"[SIM2REAL][VLP16-ablation] mode={arm.lidar_mode}",
        "[SIM2REAL][mixed-pixel-eligibility] mode=valid_return_only",
        "[SCENE-MIX] native=0.780 sa5_general=0.000 narrow=0.120 long_corridor=0.100 classes_disjoint=True",
        "Training complete:",
    ]
    if arm.delay_enabled:
        markers.append(
            "[SIM2REAL] Actuator DR: delay=(0, 2) steps, "
            "vel_scale=(1.0, 1.0), motor_lag alpha=1.0, "
            "pipeline=decode->delay->scale->lag, history=issued_command_queue"
        )
    return tuple(markers)


def _verify_arm(arm: Arm, console: Path) -> dict[str, Any]:
    run_dir = REPO / "logs/rnn_car" / arm.run_name
    metrics_path = run_dir / "supervisor_metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError(f"{arm.key} produced no supervisor metrics")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if len(rows) != arm.iterations:
        raise RuntimeError(f"{arm.key} metrics rows {len(rows)} != {arm.iterations}")
    final = rows[-1]
    if final.get("iteration") != arm.iterations or final.get("iterations_target") != arm.iterations:
        raise RuntimeError(f"{arm.key} did not reach its target iteration")
    for row in rows:
        for key in CORE_METRICS:
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"{arm.key} has invalid {key}: {value!r}")

    errors = _strict_errors(console)
    if errors:
        raise RuntimeError(f"{arm.key} strict error scan failed: {errors[-3:]}")
    console_text = console.read_text(encoding="utf-8", errors="replace")
    missing_markers = [marker for marker in _required_runtime_markers(arm) if marker not in console_text]
    if missing_markers:
        raise RuntimeError(f"{arm.key} missing runtime markers: {missing_markers}")
    if "[INFO] Loaded checkpoint:" in console_text or "[INFO] Loaded optimizer state:" in console_text:
        raise RuntimeError(f"{arm.key} unexpectedly resumed a checkpoint or optimizer")
    actuator_marker = "[SIM2REAL] Actuator DR:"
    if arm.delay_enabled != (actuator_marker in console_text):
        raise RuntimeError(f"{arm.key} actuator runtime marker disagrees with its cell")

    checkpoints = []
    for filename in arm.checkpoints:
        checkpoint = run_dir / filename
        if not checkpoint.is_file():
            raise RuntimeError(f"{arm.key} missing checkpoint: {filename}")
        checkpoints.append({"filename": filename, "sha256": _sha256(checkpoint)})
    return {
        "key": arm.key,
        "cell": arm.cell,
        "run_name": arm.run_name,
        "training_seed": arm.seed,
        "smoke": arm.smoke,
        "completed_at": _now(),
        "metrics_rows": len(rows),
        "final_metric": {key: final[key] for key in CORE_METRICS},
        "runtime_markers_verified": True,
        "strict_error_scan_count": 0,
        "checkpoints": checkpoints,
    }


def _run_arm(arm: Arm, state: dict[str, Any]) -> dict[str, Any]:
    _verify_authorization()
    if _active_trainers() or _gpu_blockers() or _expected_run():
        raise RuntimeError(f"exclusive runtime precondition changed before {arm.key}")
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
        "--seed",
        str(arm.seed),
        "--headless",
    ]
    state["status"] = "RUNNING_SMOKE" if arm.smoke else "RUNNING_FORMAL"
    state["current_arm"] = arm.key
    state["current_run_name"] = arm.run_name
    state["current_started_at"] = _now()
    state["gpu_blockers"] = []
    _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)
    print(f"[{_now()}] START {arm.key}: {' '.join(command)}", flush=True)

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
        raise RuntimeError(f"{arm.key} exited with code {completed.returncode}")
    result = _verify_arm(arm, console)
    print(f"[{_now()}] COMPLETE {arm.key}", flush=True)
    _atomic_text(EXPECTED_RUN, "")
    time.sleep(10)
    return result


def _failure(error: BaseException, state: dict[str, Any]) -> int:
    failure = {
        "schema": "paper_realism_factorial_launch_failure/v1",
        "status": "INCOMPLETE_NO_VERDICT",
        "failed_at": _now(),
        "error": f"{type(error).__name__}: {error}",
        "completed_arms": state.get("arms", []),
        "fixed_gate_started": False,
        "auto_advance_started": False,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_ROOT / "INCOMPLETE_NO_VERDICT.json", failure)
    _atomic_text(EXPECTED_RUN, "")
    _atomic_text(SUPERVISOR_STATUS, f"HALTED paper factorial: {failure['error']}\n")
    print(failure["error"], file=sys.stderr, flush=True)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-gpu", action="store_true")
    args = parser.parse_args(argv)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another paper factorial queue holds the lock", file=sys.stderr)
            return 2

        state: dict[str, Any] = {
            "schema": "paper_realism_factorial_launch/v1",
            "status": "PREFLIGHT",
            "started_at": _now(),
            "authorization_sha256": AUTHORIZATION_SHA256,
            "frozen_protocol_sha256": FREEZE_SHA256,
            "order": [arm.key for arm in ARMS],
            "arms": [],
            "fixed_gate_started": False,
            "auto_advance_started": False,
        }
        try:
            preflight = _static_preflight()
            if args.dry_run:
                status = "DRY_RUN_GPU_BUSY" if preflight["gpu_blockers"] else "DRY_RUN_OK"
                print(
                    json.dumps(
                        {
                            "status": status,
                            "authorization_sha256": AUTHORIZATION_SHA256,
                            "order": state["order"],
                            "gpu_compute_apps": preflight["gpu_compute_apps"],
                            "active_trainers": preflight["active_trainers"],
                            "expected_run": preflight["expected_run"],
                        },
                        indent=2,
                    )
                )
                return 0

            if OUTPUT_ROOT.exists():
                raise RuntimeError(f"launch output already exists: {OUTPUT_ROOT}")
            OUTPUT_ROOT.mkdir(parents=True)
            _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)
            if preflight["gpu_blockers"] or preflight["active_trainers"] or preflight["expected_run"]:
                if not args.wait_for_gpu:
                    raise RuntimeError("GPU or training supervisor is busy; use --wait-for-gpu")
                _wait_for_exclusive_gpu(state)

            for index, arm in enumerate(ARMS):
                if index:
                    _wait_for_exclusive_gpu(state)
                state["arms"].append(_run_arm(arm, state))
                state.pop("current_arm", None)
                state.pop("current_run_name", None)
                state.pop("current_started_at", None)
                _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)

            state["status"] = "TRAINING_COMPLETE_GATE_BLOCKED_ON_INSTRUMENTATION"
            state["completed_at"] = _now()
            _atomic_json(OUTPUT_ROOT / "LAUNCH_STATE.json", state)
            _atomic_text(EXPECTED_RUN, "")
            _atomic_text(
                SUPERVISOR_STATUS,
                "IDLE paper realism factorial training complete; fixed Gate not started\n",
            )
            return 0
        except BaseException as error:
            if args.dry_run:
                print(f"{type(error).__name__}: {error}", file=sys.stderr)
                return 1
            return _failure(error, state)


if __name__ == "__main__":
    raise SystemExit(main())
