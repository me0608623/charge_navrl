"""Fail-closed acceptance queue for the SA1 sim2real_v1 R1 run.

Frozen specification: ``docs/freeze/sa1_r1_acceptance_matrix_20260729.md``.

The queue does not launch GPU work until the training service is inactive, the
completion marker is present, the exact final checkpoint is readable, the
training process is gone, and the GPU is free. Each runner report and runtime
log is converted into a canonical per-cell JSON record before the next cell is
allowed to start.

Nothing here modifies a file in the R1 SHA-256 freeze list. The existing gate
runners are invoked only through their public CLI.
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import json
import math
from numbers import Real
from pathlib import Path
import subprocess
from typing import Iterable, Sequence


REPO = Path(__file__).resolve().parents[4]
RUNNER_DIR = Path(__file__).resolve().parent
ISAAC_PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")

SERVICE = "sa1-sim2real-v1-r1.service"
RUN_NAME = "sa1_sim2real_v1_ne1024_s42_r1"
RUN_DIR = REPO / "logs" / "rnn_car" / RUN_NAME
TRAIN_LOG = REPO / "logs" / f"{RUN_NAME}.log"
TRAIN_PROCESS_PATTERN = "train_rnn_car_wdclip.py"

SCHEMA = "sa1_r1_acceptance/v2"

# Frozen R1 completion contract.
FINAL_ITERATIONS = 2109
FINAL_TOTAL_STEPS = 269_952
REQUESTED_TOTAL_STEPS = 270_000
STEPS_PER_ITERATION = 128
COMPLETION_MARKER = "Training complete: 270,000 steps"
EXPECTED_FINAL_CHECKPOINT = RUN_DIR / f"checkpoint_{FINAL_TOTAL_STEPS}.pt"
EXPECTED_CHECKPOINT_ARGS = {
    "experiment_config": "e2e_sa1_k8_obb_sim2real_v1",
    "run_name": RUN_NAME,
    "num_envs": 1024,
    "seed": 42,
    "timesteps": REQUESTED_TOTAL_STEPS,
    "save_interval": 50,
    "checkpoint": None,
    "no_resume_optimizer": True,
}

# Frozen evaluation parameters.
NUM_ENVS = 64
STEPS = 1200
SEEDS: tuple[int, ...] = (515, 616, 717)
SCREEN_SEED = 515
SCREEN_DELAY = "d1"
OFF_REGRESSION_SEED = 515
ACTUATOR_PROFILE = "sa1_delay_only"
EXPECTED_VLP16_MODE = "full"
NARROW_MODE = "sealed"
NARROW_STAGE = 5
NARROW_ARENA = 10.0
CORRIDOR_KINEMATICS = "wander"

# Deployment acceptance is exactly d=0/1/2. Actuator-off is a separate,
# single-seed diagnostic because enabling the d=0 action term consumes RNG and
# therefore is not a trajectory-paired identity arm.
DELAY_CONDITIONS: tuple[str, ...] = ("d0", "d1", "d2")
OFF_CONDITION = "off"
_DELAY_STEPS: dict[str, int | None] = {
    OFF_CONDITION: None,
    "d0": 0,
    "d1": 1,
    "d2": 2,
}

CORRIDOR_MODES: tuple[str, ...] = (
    "lateral",
    "longitudinal",
    "random_2d",
    "mixed_iid",
)
SCENARIOS: tuple[str, ...] = ("gate2", "narrow_sealed") + tuple(
    f"corridor_{mode}" for mode in CORRIDOR_MODES
)
SCREEN_CHECKPOINT_ITERATIONS: tuple[int, ...] = (1500, 1700, 1900, 2109)

_REPORT_NAMES = {
    "gate2": "gate2_suite.json",
    "narrow_sealed": "narrow_path_suite.json",
}


class PreconditionError(RuntimeError):
    """Raised when fail-closed execution must stop."""


@dataclasses.dataclass(frozen=True)
class Cell:
    """One rollout for one checkpoint, scenario, delay, and seed."""

    checkpoint: Path
    scenario: str
    delay_condition: str
    seed: int
    stage: str  # A=screening, B=fixed-delay acceptance, O=actuator-off diagnostic

    @property
    def cell_id(self) -> str:
        return (
            f"{self.checkpoint.stem}__{self.scenario}"
            f"__{self.delay_condition}__seed{self.seed}"
        )


# Command and matrix construction

def _actuator_args(delay_condition: str) -> list[str]:
    if delay_condition not in _DELAY_STEPS:
        raise ValueError(f"unknown delay condition {delay_condition!r}")
    args = ["--actuator-profile", ACTUATOR_PROFILE]
    delay_steps = _DELAY_STEPS[delay_condition]
    if delay_steps is not None:
        args += ["--actuator-delay-steps", str(delay_steps)]
    return args


def runner_name(scenario: str) -> str:
    if scenario == "gate2":
        return "run_gate2_suite.py"
    if scenario == "narrow_sealed":
        return "run_narrow_path_suite.py"
    if scenario.startswith("corridor_"):
        mode = scenario.removeprefix("corridor_")
        if mode in CORRIDOR_MODES:
            return "run_corridor_motion_suite.py"
    raise ValueError(f"unknown scenario {scenario!r}")


def report_name(scenario: str) -> str:
    if scenario in _REPORT_NAMES:
        return _REPORT_NAMES[scenario]
    if scenario.startswith("corridor_"):
        mode = scenario.removeprefix("corridor_")
        if mode in CORRIDOR_MODES:
            return "corridor_motion_suite.json"
    raise ValueError(f"unknown scenario {scenario!r}")


def build_command(cell: Cell, output_dir: Path) -> list[str]:
    """Build one runner command without performing I/O."""
    common = [
        str(cell.checkpoint),
        "--output-dir",
        str(output_dir),
        "--num-envs",
        str(NUM_ENVS),
        "--steps",
        str(STEPS),
        "--seeds",
        str(cell.seed),
        *_actuator_args(cell.delay_condition),
    ]

    if cell.scenario == "gate2":
        extra: list[str] = []
    elif cell.scenario == "narrow_sealed":
        extra = [
            "--mode",
            NARROW_MODE,
            "--stage",
            str(NARROW_STAGE),
            "--arena-size",
            str(NARROW_ARENA),
        ]
    elif cell.scenario.startswith("corridor_"):
        mode = cell.scenario.removeprefix("corridor_")
        if mode not in CORRIDOR_MODES:
            raise ValueError(f"unknown corridor mode {mode!r}")
        extra = [
            "--modes",
            mode,
            "--random-2d-kinematics",
            CORRIDOR_KINEMATICS,
        ]
    else:
        raise ValueError(f"unknown scenario {cell.scenario!r}")

    return [str(ISAAC_PYTHON), str(RUNNER_DIR / runner_name(cell.scenario)), *common, *extra]


def screening_matrix(checkpoints: Sequence[Path]) -> list[Cell]:
    return [
        Cell(checkpoint, scenario, SCREEN_DELAY, SCREEN_SEED, stage="A")
        for checkpoint in checkpoints
        for scenario in SCENARIOS
    ]


def acceptance_matrix(checkpoints: Sequence[Path]) -> list[Cell]:
    """Official fixed-delay matrix: 6 scenarios x 3 delays x 3 seeds."""
    return [
        Cell(checkpoint, scenario, delay, seed, stage="B")
        for checkpoint in checkpoints
        for scenario in SCENARIOS
        for delay in DELAY_CONDITIONS
        for seed in SEEDS
    ]


def off_regression_matrix(checkpoints: Sequence[Path]) -> list[Cell]:
    """True-clean diagnostic, reported separately from fixed-delay acceptance."""
    return [
        Cell(checkpoint, scenario, OFF_CONDITION, OFF_REGRESSION_SEED, stage="O")
        for checkpoint in checkpoints
        for scenario in SCENARIOS
    ]


def matrix_size(n_candidates: int, stage: str) -> int:
    if stage == "A":
        return n_candidates * len(SCENARIOS)
    if stage == "B":
        return (
            n_candidates
            * len(SCENARIOS)
            * len(DELAY_CONDITIONS)
            * len(SEEDS)
        )
    if stage == "O":
        return n_candidates * len(SCENARIOS)
    raise ValueError(f"unknown stage {stage!r}")


# Canonical result construction and validation

REQUIRED_RESULT_KEYS = (
    "schema",
    "cell_id",
    "stage",
    "checkpoint",
    "scenario",
    "delay_condition",
    "actuator_profile",
    "seed",
    "num_envs",
    "steps",
    "episodes_completed",
    "sr",
    "cr",
    "timeout",
    "gate_pass",
    "structural_pass",
    "thresholds",
    "action_stability",
    "actuator_eval",
    "runner",
    "runner_argv",
    "source_report",
    "runtime_markers",
    "exit_code",
    "started_at",
    "finished_at",
)

_ACTION_STABILITY_NONNEGATIVE = (
    "omega_rms_rad_s",
    "worst_seed_omega_abs_std_p95_rad_s",
    "worst_seed_dominant_frequency_p95_hz",
    "worst_seed_max_same_sign_turn_p95_s",
)
_ACTION_STABILITY_FRACTIONS = (
    "full_steer_fraction",
    "ratio_flip_rate_mean",
    "worst_seed_ratio_flip_rate_p95",
)


def _expected_actuator_metadata(delay_condition: str) -> dict:
    delay_steps = _DELAY_STEPS[delay_condition]
    if delay_steps is None:
        return {
            "enabled": False,
            "profile": ACTUATOR_PROFILE,
            "delay_steps": None,
            "delay_ms": None,
            "velocity_scale_range": None,
            "motor_lag_alpha": None,
        }
    return {
        "enabled": True,
        "profile": ACTUATOR_PROFILE,
        "delay_steps": delay_steps,
        "delay_ms": delay_steps * 200,
        "velocity_scale_range": [1.0, 1.0],
        "motor_lag_alpha": 1.0,
    }


def _same_checkpoint(left: object, right: Path) -> bool:
    try:
        return Path(str(left)).expanduser().resolve() == right.expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return False


def _source_report_data(cell: Cell, report: dict) -> dict:
    """Validate runner-owned identity fields and extract scenario metrics."""
    problems: list[str] = []
    if not _same_checkpoint(report.get("checkpoint"), cell.checkpoint):
        problems.append(
            f"source checkpoint {report.get('checkpoint')!r} does not match "
            f"{str(cell.checkpoint)!r}"
        )
    if report.get("seeds") != [cell.seed]:
        problems.append(
            f"source seeds {report.get('seeds')!r} do not match [{cell.seed}]"
        )

    actuator_eval = report.get("actuator_eval")
    if not isinstance(actuator_eval, dict):
        problems.append("source actuator_eval is missing or not an object")
        actuator_eval = {}
    else:
        for key, expected in _expected_actuator_metadata(
            cell.delay_condition
        ).items():
            if actuator_eval.get(key) != expected:
                problems.append(
                    f"source actuator_eval.{key}={actuator_eval.get(key)!r}, "
                    f"expected {expected!r}"
                )

    try:
        if cell.scenario == "gate2":
            aggregate = report["aggregate"]
            extracted = {
                "episodes_completed": aggregate["n"],
                "sr": aggregate["sr"],
                "cr": aggregate["cr"],
                "timeout": aggregate["to"],
                "gate_pass": report["pass"],
                "structural_pass": True,
                "thresholds": report["thresholds"],
                "action_stability": report["action_stability"],
            }
        elif cell.scenario == "narrow_sealed":
            aggregate = report["aggregate"]
            extracted = {
                "episodes_completed": aggregate["episodes"],
                "sr": aggregate["sr"],
                "cr": aggregate["cr"],
                "timeout": aggregate["to"],
                "gate_pass": report["pass"],
                "structural_pass": True,
                "thresholds": report["thresholds"],
                "action_stability": None,
            }
        else:
            mode = cell.scenario.removeprefix("corridor_")
            modes = report["modes"]
            if set(modes) != {mode}:
                problems.append(
                    f"source corridor modes {sorted(modes)!r}, expected [{mode!r}]"
                )
            mode_report = modes[mode]
            aggregate = mode_report["aggregate"]
            extracted = {
                "episodes_completed": aggregate["episodes"],
                "sr": aggregate["success_rate"],
                "cr": aggregate["collision_rate"],
                "timeout": aggregate["timeout_rate"],
                "gate_pass": mode_report["pass"],
                "structural_pass": mode_report["structural_pass"],
                "thresholds": report["thresholds"],
                "action_stability": None,
            }
    except (KeyError, TypeError) as exc:
        problems.append(f"source report schema incomplete: {exc}")
        extracted = {
            "episodes_completed": None,
            "sr": None,
            "cr": None,
            "timeout": None,
            "gate_pass": None,
            "structural_pass": None,
            "thresholds": None,
            "action_stability": None,
        }

    if problems:
        raise PreconditionError("; ".join(problems))
    extracted["actuator_eval"] = actuator_eval
    return extracted


def _log_paths(cell: Cell, output_dir: Path, report: dict) -> list[Path]:
    if cell.scenario == "gate2":
        raw_paths = report.get("logs")
        if not isinstance(raw_paths, list) or len(raw_paths) != 1:
            raise PreconditionError(
                f"{cell.cell_id}: gate2 source report must contain one log"
            )
        paths = [Path(str(item)) for item in raw_paths]
    elif cell.scenario == "narrow_sealed":
        paths = [output_dir / f"narrow_s{cell.seed}.log"]
    else:
        mode = cell.scenario.removeprefix("corridor_")
        paths = [output_dir / f"{mode}_s{cell.seed}.log"]

    root = output_dir.resolve()
    resolved: list[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise PreconditionError(
                f"{cell.cell_id}: runtime log escapes cell directory: {candidate}"
            ) from exc
        if not candidate.is_file():
            raise PreconditionError(
                f"{cell.cell_id}: runtime log is missing: {candidate}"
            )
        resolved.append(candidate)
    return resolved


def _runtime_markers(log_paths: Sequence[Path]) -> dict[str, str | None]:
    actuator_lines: list[str] = []
    vlp_lines: list[str] = []
    for path in log_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        actuator_lines.extend(
            line.strip()
            for line in text.splitlines()
            if "[SIM2REAL] Actuator DR:" in line
        )
        vlp_lines.extend(
            line.strip()
            for line in text.splitlines()
            if "[SIM2REAL][VLP16-ablation]" in line
        )
    return {
        "actuator_dr": "\n".join(dict.fromkeys(actuator_lines)) or None,
        "vlp16_noise": "\n".join(dict.fromkeys(vlp_lines)) or None,
    }


def canonicalize_runner_result(
    cell: Cell,
    output_dir: Path,
    runner_argv: Sequence[str],
    exit_code: int,
    started_at: str,
    finished_at: str,
) -> dict:
    source_path = output_dir / report_name(cell.scenario)
    if not source_path.is_file():
        raise PreconditionError(
            f"{cell.cell_id}: runner report is missing: {source_path}"
        )
    try:
        report = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreconditionError(
            f"{cell.cell_id}: runner report is unreadable: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise PreconditionError(f"{cell.cell_id}: runner report is not an object")

    extracted = _source_report_data(cell, report)
    markers = _runtime_markers(_log_paths(cell, output_dir, report))
    return {
        "schema": SCHEMA,
        "cell_id": cell.cell_id,
        "stage": cell.stage,
        "checkpoint": str(cell.checkpoint.expanduser().resolve()),
        "scenario": cell.scenario,
        "delay_condition": cell.delay_condition,
        "actuator_profile": ACTUATOR_PROFILE,
        "seed": cell.seed,
        "num_envs": NUM_ENVS,
        "steps": STEPS,
        **extracted,
        "runner": runner_name(cell.scenario),
        "runner_argv": list(runner_argv),
        "source_report": str(source_path.resolve()),
        "runtime_markers": markers,
        "exit_code": exit_code,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def validate_result(payload: dict, cell: Cell | None = None) -> list[str]:
    """Return validation problems; a valid policy gate FAIL still returns []."""
    problems: list[str] = []
    for key in REQUIRED_RESULT_KEYS:
        if key not in payload:
            problems.append(f"missing key: {key}")

    if payload.get("schema") != SCHEMA:
        problems.append(f"schema mismatch: {payload.get('schema')!r}")
    if payload.get("actuator_profile") != ACTUATOR_PROFILE:
        problems.append(
            f"actuator_profile must be {ACTUATOR_PROFILE!r}, "
            f"got {payload.get('actuator_profile')!r}"
        )
    if payload.get("num_envs") != NUM_ENVS:
        problems.append(f"num_envs={payload.get('num_envs')!r}, expected {NUM_ENVS}")
    if payload.get("steps") != STEPS:
        problems.append(f"steps={payload.get('steps')!r}, expected {STEPS}")

    episodes = payload.get("episodes_completed")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        problems.append("episodes_completed must be a positive integer")

    for metric in ("sr", "cr", "timeout"):
        value = payload.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            problems.append(f"{metric} must be finite and within [0, 1], got {value!r}")

    gate_pass = payload.get("gate_pass")
    structural_pass = payload.get("structural_pass")
    if not isinstance(gate_pass, bool):
        problems.append(f"gate_pass must be bool, got {gate_pass!r}")
    if structural_pass is not True:
        problems.append(
            f"structural_pass must be true, got {structural_pass!r}"
        )

    exit_code = payload.get("exit_code")
    if isinstance(gate_pass, bool):
        expected_exit = 0 if gate_pass else 1
        if exit_code != expected_exit:
            problems.append(
                f"exit_code={exit_code!r}, expected {expected_exit} "
                f"for gate_pass={gate_pass}"
            )
    elif exit_code not in (0, 1):
        problems.append(f"exit_code={exit_code!r}")

    delay_condition = payload.get("delay_condition")
    if delay_condition not in _DELAY_STEPS:
        problems.append(f"unknown delay_condition={delay_condition!r}")
    else:
        actuator_eval = payload.get("actuator_eval")
        if not isinstance(actuator_eval, dict):
            problems.append("actuator_eval is missing or not an object")
        else:
            for key, expected in _expected_actuator_metadata(
                str(delay_condition)
            ).items():
                if actuator_eval.get(key) != expected:
                    problems.append(
                        f"actuator_eval.{key}={actuator_eval.get(key)!r}, "
                        f"expected {expected!r}"
                    )

        markers = payload.get("runtime_markers")
        if not isinstance(markers, dict):
            problems.append("runtime_markers is missing or not an object")
        else:
            actuator_marker = markers.get("actuator_dr")
            vlp_marker = markers.get("vlp16_noise")
            if delay_condition == OFF_CONDITION:
                if actuator_marker is not None:
                    problems.append(
                        "actuator-off diagnostic unexpectedly emitted actuator marker"
                    )
            else:
                delay_steps = _DELAY_STEPS[str(delay_condition)]
                required = (
                    "[SIM2REAL] Actuator DR:",
                    f"delay=({delay_steps}, {delay_steps}) steps",
                    "vel_scale=(1.0, 1.0)",
                    "motor_lag alpha=1.0",
                    "pipeline=decode->delay->scale->lag",
                    "history=issued_command_queue",
                )
                if not isinstance(actuator_marker, str):
                    problems.append("runtime_markers.actuator_dr is absent")
                else:
                    for marker in required:
                        if marker not in actuator_marker:
                            problems.append(
                                f"runtime actuator marker missing {marker!r}"
                            )
            expected_vlp = (
                f"[SIM2REAL][VLP16-ablation] mode={EXPECTED_VLP16_MODE}"
            )
            if not isinstance(vlp_marker, str) or expected_vlp not in vlp_marker:
                problems.append(
                    f"runtime VLP16 marker missing {expected_vlp!r}"
                )

    if not isinstance(payload.get("runner_argv"), list):
        problems.append("runner_argv must be a list")
    if not payload.get("started_at") or not payload.get("finished_at"):
        problems.append("started_at/finished_at must be populated")

    action_stability = payload.get("action_stability")
    if payload.get("scenario") == "gate2":
        if not isinstance(action_stability, dict):
            problems.append("Gate2 action_stability is missing or not an object")
        else:
            aggregate = action_stability.get("aggregate")
            if not isinstance(aggregate, dict):
                problems.append("Gate2 action_stability.aggregate is missing")
            else:
                samples = aggregate.get("samples")
                if (
                    isinstance(samples, bool)
                    or not isinstance(samples, int)
                    or samples <= 0
                ):
                    problems.append(
                        "action_stability.aggregate.samples must be positive"
                    )
                for key in _ACTION_STABILITY_NONNEGATIVE:
                    value = aggregate.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(float(value))
                        or float(value) < 0.0
                    ):
                        problems.append(
                            f"action_stability.aggregate.{key} must be "
                            f"finite and nonnegative, got {value!r}"
                        )
                for key in _ACTION_STABILITY_FRACTIONS:
                    value = aggregate.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, Real)
                        or not math.isfinite(float(value))
                        or not 0.0 <= float(value) <= 1.0
                    ):
                        problems.append(
                            f"action_stability.aggregate.{key} must be "
                            f"within [0, 1], got {value!r}"
                        )
    elif action_stability is not None:
        problems.append("action_stability must be null outside Gate2")

    if cell is not None:
        expected = {
            "cell_id": cell.cell_id,
            "stage": cell.stage,
            "scenario": cell.scenario,
            "delay_condition": cell.delay_condition,
            "seed": cell.seed,
            "runner": runner_name(cell.scenario),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                problems.append(
                    f"{key}={payload.get(key)!r}, expected {value!r}"
                )
        if not _same_checkpoint(payload.get("checkpoint"), cell.checkpoint):
            problems.append("checkpoint does not match the requested cell")
    return problems


def assert_no_cross_delay_average(payloads: Iterable[dict]) -> None:
    """Reject duplicate checkpoint/scenario/delay/seed records."""
    seen: set[tuple[str, str, str, int]] = set()
    for payload in payloads:
        key = (
            str(payload.get("checkpoint")),
            str(payload.get("scenario")),
            str(payload.get("delay_condition")),
            int(payload.get("seed", -1)),
        )
        if key in seen:
            raise PreconditionError(f"duplicate cell would be pooled: {key}")
        seen.add(key)


# Preconditions

def _run(cmd: Sequence[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _read_checkpoint_metadata(path: Path) -> dict:
    probe = (
        "import json,sys,torch\n"
        f"keys={tuple(EXPECTED_CHECKPOINT_ARGS)!r}\n"
        "x=torch.load(sys.argv[1],map_location='cpu',weights_only=False)\n"
        "a=x.get('args',{})\n"
        "print(json.dumps({'iteration':x.get('iteration'),"
        "'total_steps':x.get('total_steps'),"
        "'args':{k:a.get(k) for k in keys}}))\n"
    )
    proc = _run([str(ISAAC_PYTHON), "-c", probe, str(path)])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("checkpoint probe produced no JSON")
    return json.loads(lines[-1])


def final_checkpoint() -> Path | None:
    candidates = sorted(
        RUN_DIR.glob("checkpoint_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    return candidates[-1] if candidates else None


def check_preconditions(*, checkpoint_reader=None) -> list[str]:
    """Return every unmet precondition; empty means GPU work may start."""
    unmet: list[str] = []
    reader = checkpoint_reader or _read_checkpoint_metadata

    service = _run(["systemctl", "--user", "is-active", SERVICE])
    state = service.stdout.strip()
    # A transient unit may be garbage-collected after clean exit and report
    # "unknown". Completion marker + exact checkpoint + no process remain the
    # positive proof in that case. A failed unit is never accepted.
    if state not in ("inactive", "unknown"):
        unmet.append(
            f"service {SERVICE} is {state!r}, expected inactive/not-found"
        )

    procs = _run(["pgrep", "-f", TRAIN_PROCESS_PATTERN])
    if procs.returncode not in (0, 1):
        unmet.append(f"pgrep failed: {procs.stderr.strip()!r}")
    elif procs.stdout.strip():
        unmet.append(
            f"training process still alive: {procs.stdout.splitlines()}"
        )

    smi = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ]
    )
    if smi.returncode != 0:
        unmet.append(f"nvidia-smi failed: {smi.stderr.strip()!r}")
    else:
        busy = [
            line
            for line in smi.stdout.splitlines()
            if line.strip() and "gnome-remote-desktop" not in line
        ]
        if busy:
            unmet.append(f"GPU still has compute processes: {busy}")

    try:
        train_text = TRAIN_LOG.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        unmet.append(f"training console log unreadable: {exc}")
    else:
        if COMPLETION_MARKER not in train_text:
            unmet.append(
                f"training console lacks completion marker {COMPLETION_MARKER!r}"
            )

    latest = final_checkpoint()
    if not EXPECTED_FINAL_CHECKPOINT.is_file():
        unmet.append(
            f"exact final checkpoint missing: {EXPECTED_FINAL_CHECKPOINT} "
            f"(latest={latest})"
        )
    elif latest != EXPECTED_FINAL_CHECKPOINT:
        unmet.append(
            f"latest checkpoint {latest} is not frozen final "
            f"{EXPECTED_FINAL_CHECKPOINT}"
        )
    else:
        try:
            metadata = reader(EXPECTED_FINAL_CHECKPOINT)
        except Exception as exc:  # noqa: BLE001 - report exact probe failure
            unmet.append(f"final checkpoint unreadable: {exc}")
        else:
            try:
                iteration = int(metadata.get("iteration"))
                total_steps = int(metadata.get("total_steps"))
            except (TypeError, ValueError):
                unmet.append(
                    f"final checkpoint metadata malformed: {metadata!r}"
                )
            else:
                if iteration + 1 != FINAL_ITERATIONS:
                    unmet.append(
                        f"final checkpoint completed iterations={iteration + 1}, "
                        f"expected {FINAL_ITERATIONS}"
                    )
                if total_steps != FINAL_TOTAL_STEPS:
                    unmet.append(
                        f"final checkpoint total_steps={total_steps}, "
                        f"expected {FINAL_TOTAL_STEPS}"
                    )
            args = metadata.get("args")
            if not isinstance(args, dict):
                unmet.append("final checkpoint args metadata missing")
            else:
                for key, expected in EXPECTED_CHECKPOINT_ARGS.items():
                    if args.get(key) != expected:
                        unmet.append(
                            f"final checkpoint args.{key}={args.get(key)!r}, "
                            f"expected {expected!r}"
                        )
    return unmet


def checkpoints_for_iterations(iterations: Sequence[int]) -> list[Path]:
    """Resolve exact frozen checkpoints; nearest-neighbour substitution is banned."""
    checkpoints: list[Path] = []
    missing: list[Path] = []
    for iteration in iterations:
        path = RUN_DIR / f"checkpoint_{iteration * STEPS_PER_ITERATION}.pt"
        if path.is_file():
            checkpoints.append(path)
        else:
            missing.append(path)
    if missing:
        raise PreconditionError(
            "frozen screening checkpoints are missing: "
            + ", ".join(str(path) for path in missing)
        )
    return checkpoints


# Execution

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_runner(
    command: Sequence[str],
    console_path: Path,
) -> subprocess.CompletedProcess:
    with console_path.open("w", encoding="utf-8") as stream:
        return subprocess.run(
            command,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )


def _resume_payload(cell: Cell, output_dir: Path) -> dict | None:
    if not output_dir.exists():
        return None
    canonical_path = output_dir / "cell.json"
    if not canonical_path.is_file():
        if any(output_dir.iterdir()):
            raise PreconditionError(
                f"{cell.cell_id}: non-empty incomplete cell directory; "
                "inspect it before retrying"
            )
        return None
    try:
        payload = json.loads(canonical_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreconditionError(
            f"{cell.cell_id}: saved canonical result unreadable: {exc}"
        ) from exc
    problems = validate_result(payload, cell)
    if problems:
        raise PreconditionError(
            f"{cell.cell_id}: saved canonical result invalid: {'; '.join(problems)}"
        )
    return payload


def _execute_cell(
    cell: Cell,
    output_dir: Path,
    *,
    runner=None,
) -> dict:
    resumed = _resume_payload(cell, output_dir)
    if resumed is not None:
        print(f"[resume] {cell.cell_id}")
        return resumed

    output_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(cell, output_dir)
    started_at = _utc_now()
    process = (runner or _run_runner)(command, output_dir / "runner_console.log")
    finished_at = _utc_now()
    if process.returncode not in (0, 1):
        raise PreconditionError(
            f"{cell.cell_id}: runner exited {process.returncode}; "
            "only rc=0 PASS and rc=1 valid gate FAIL are admissible"
        )

    payload = canonicalize_runner_result(
        cell,
        output_dir,
        command,
        process.returncode,
        started_at,
        finished_at,
    )
    problems = validate_result(payload, cell)
    if problems:
        _write_json(output_dir / "cell.invalid.json", payload)
        raise PreconditionError(
            f"{cell.cell_id}: canonical validation failed: {'; '.join(problems)}"
        )
    _write_json(output_dir / "cell.json", payload)
    verdict = "PASS" if payload["gate_pass"] else "FAIL"
    print(f"[recorded] {cell.cell_id} gate={verdict}")
    return payload


def summarize_results(payloads: Sequence[dict], stage: str) -> dict:
    """Create a non-deployment summary without averaging delay conditions."""
    by_checkpoint: dict[str, list[dict]] = {}
    for payload in payloads:
        by_checkpoint.setdefault(str(payload["checkpoint"]), []).append(payload)

    candidates: list[dict] = []
    for checkpoint, records in by_checkpoint.items():
        candidate = {
            "checkpoint": checkpoint,
            "cells": len(records),
            "gate_passes": sum(bool(item["gate_pass"]) for item in records),
            "all_recorded_gates_pass": all(
                bool(item["gate_pass"]) for item in records
            ),
        }
        if stage == "A":
            candidate["worst_scenario_sr"] = min(
                float(item["sr"]) for item in records
            )
        candidates.append(candidate)

    if stage == "A":
        candidates.sort(
            key=lambda item: (
                int(item["gate_passes"]),
                float(item["worst_scenario_sr"]),
                int(Path(item["checkpoint"]).stem.split("_")[-1]),
            ),
            reverse=True,
        )
        for rank, candidate in enumerate(candidates, start=1):
            candidate["advisory_rank"] = rank
        selection = (
            "Stage B candidates must still be supplied explicitly; this ranking "
            "does not auto-launch Stage B."
        )
    else:
        candidates.sort(key=lambda item: item["checkpoint"])
        selection = None

    return {
        "schema": SCHEMA,
        "stage": stage,
        "cells": len(payloads),
        "candidates": candidates,
        "selection_note": selection,
        "python_torchscript_parity": "required_external_83d_k8_gate",
        "deployable": False,
    }


def execute_queue(
    cells: Sequence[Cell],
    output_root: Path,
    *,
    execute: bool,
    precondition_checker=None,
    runner=None,
) -> int:
    """Print or execute one stage. A valid model FAIL is recorded and continues."""
    if not execute:
        for cell in cells:
            print(" ".join(build_command(cell, output_root / cell.cell_id)))
        return 0

    unmet = (precondition_checker or check_preconditions)()
    if unmet:
        raise PreconditionError("; ".join(unmet))

    payloads: list[dict] = []
    for index, cell in enumerate(cells, start=1):
        print(f"[{index}/{len(cells)}] {cell.cell_id}")
        payloads.append(
            _execute_cell(
                cell,
                output_root / cell.cell_id,
                runner=runner,
            )
        )
    assert_no_cross_delay_average(payloads)
    stage = cells[0].stage if cells else "unknown"
    _write_json(output_root / "stage_summary.json", summarize_results(payloads, stage))
    return 0


def _validated_candidates(paths: Sequence[Path]) -> list[Path]:
    if not 1 <= len(paths) <= 2:
        raise PreconditionError("Stage B/O requires one or two candidate checkpoints")
    allowed = {
        (RUN_DIR / f"checkpoint_{iteration * STEPS_PER_ITERATION}.pt").resolve()
        for iteration in SCREEN_CHECKPOINT_ITERATIONS
    }
    resolved: list[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        if candidate not in allowed:
            raise PreconditionError(
                f"candidate is outside the frozen Stage A set: {candidate}"
            )
        if not candidate.is_file():
            raise PreconditionError(f"candidate checkpoint is missing: {candidate}")
        if candidate in resolved:
            raise PreconditionError(f"duplicate candidate: {candidate}")
        resolved.append(candidate)
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("A", "B", "O"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--candidates",
        type=Path,
        nargs="*",
        default=None,
        help="One or two Stage A finalists for Stage B/O.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Launch GPU rollouts. Without it, print the exact plan only.",
    )
    parser.add_argument("--check-preconditions", action="store_true")
    args = parser.parse_args(argv)

    if args.check_preconditions:
        unmet = check_preconditions()
        if unmet:
            print("PRECONDITIONS UNMET:")
            for item in unmet:
                print(f"  - {item}")
            return 1
        print("preconditions met")
        return 0

    if args.stage is None or args.output_root is None:
        parser.error("--stage and --output-root are required unless checking preconditions")

    if args.stage == "A":
        checkpoints = checkpoints_for_iterations(SCREEN_CHECKPOINT_ITERATIONS)
        cells = screening_matrix(checkpoints)
    else:
        if not args.candidates:
            parser.error("--candidates is required for Stage B/O")
        checkpoints = _validated_candidates(args.candidates)
        cells = (
            acceptance_matrix(checkpoints)
            if args.stage == "B"
            else off_regression_matrix(checkpoints)
        )

    print(f"stage {args.stage}: {len(cells)} rollouts")
    return execute_queue(
        cells,
        args.output_root.expanduser().resolve(),
        execute=args.execute,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
