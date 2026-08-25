#!/usr/bin/env python3
"""Fail-closed COMPLETE -> IDLE transition for the training supervisor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
COMPLETION_RE = re.compile(
    r"^Training complete:\s*([0-9][0-9,]*)\s+steps\b",
    re.MULTILINE,
)
ANOMALY_RE = re.compile(
    r"traceback|cuda out of memory|runtimeerror"
    r"|(?:^|[^A-Za-z])oom(?:[^A-Za-z]|$)"
    r"|(?:^|[^A-Za-z])nan(?:[^A-Za-z]|$)",
    re.IGNORECASE | re.MULTILINE,
)


class CompletionNotReady(RuntimeError):
    """Raised when completion evidence is absent, inconsistent, or unsafe."""


@dataclass(frozen=True)
class CompletionEvidence:
    run_name: str
    completed_at: str
    iteration: int
    iterations_target: int
    total_steps: int
    metrics_file: str
    console_log: str
    checkpoint: str
    checkpoint_sha256: str


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        text=True,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o664)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CompletionNotReady(f"{field} must be a positive integer")
    return value


def read_expected_run(path: Path) -> str | None:
    if not path.is_file():
        return None
    nonempty = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not nonempty:
        return None
    if len(nonempty) != 1:
        raise CompletionNotReady("expected_run.txt must contain exactly one run")
    run_name = nonempty[0]
    if not RUN_NAME_RE.fullmatch(run_name):
        raise CompletionNotReady("expected run name failed the safe-name contract")
    return run_name


def _last_metrics(path: Path) -> dict:
    if not path.is_file():
        raise CompletionNotReady(f"metrics file is missing: {path}")
    last_line = ""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        raise CompletionNotReady("metrics file has no non-empty rows")
    try:
        row = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise CompletionNotReady("last metrics row is invalid JSON") from exc
    if not isinstance(row, dict):
        raise CompletionNotReady("last metrics row must be a JSON object")
    return row


def _resolve_console_log(repo: Path, run_name: str) -> Path:
    canonical = repo / "logs" / "rnn_car" / f"{run_name}.console.log"
    legacy = Path("/tmp") / f"{run_name}.log"
    if canonical.is_file():
        return canonical
    if legacy.is_file():
        return legacy
    raise CompletionNotReady(f"console log is missing: {canonical}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_completion_evidence(repo: Path, run_name: str) -> CompletionEvidence:
    if not RUN_NAME_RE.fullmatch(run_name):
        raise CompletionNotReady("run name failed the safe-name contract")

    run_dir = repo / "logs" / "rnn_car" / run_name
    metrics_file = run_dir / "supervisor_metrics.jsonl"
    row = _last_metrics(metrics_file)
    iteration = _positive_int(row.get("iteration"), "iteration")
    target = _positive_int(row.get("iterations_target"), "iterations_target")
    total_steps = _positive_int(row.get("total_steps"), "total_steps")
    if iteration != target:
        raise CompletionNotReady(
            f"training is incomplete: iteration={iteration}, target={target}"
        )

    checkpoint = run_dir / f"checkpoint_{total_steps}.pt"
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise CompletionNotReady(f"final checkpoint is missing or empty: {checkpoint}")
    if not os.access(checkpoint, os.R_OK):
        raise CompletionNotReady(f"final checkpoint is not readable: {checkpoint}")

    console_log = _resolve_console_log(repo, run_name)
    console_text = console_log.read_text(encoding="utf-8", errors="replace")
    if ANOMALY_RE.search(console_text):
        raise CompletionNotReady("strict console anomaly scan matched")
    completion_steps = {
        int(match.replace(",", ""))
        for match in COMPLETION_RE.findall(console_text)
    }
    if total_steps not in completion_steps:
        raise CompletionNotReady(
            "console completion marker is absent or does not match total_steps"
        )

    return CompletionEvidence(
        run_name=run_name,
        completed_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        iteration=iteration,
        iterations_target=target,
        total_steps=total_steps,
        metrics_file=str(metrics_file),
        console_log=str(console_log),
        checkpoint=str(checkpoint),
        checkpoint_sha256=_sha256(checkpoint),
    )


def _ledger_contains(ledger: Path, evidence: CompletionEvidence) -> bool:
    if not ledger.is_file():
        return False
    with ledger.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("run_name") == evidence.run_name
                and row.get("checkpoint_sha256") == evidence.checkpoint_sha256
            ):
                return True
    return False


def _append_ledger_once(ledger: Path, evidence: CompletionEvidence) -> None:
    if _ledger_contains(ledger, evidence):
        return
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(evidence), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def apply_transition(
    evidence: CompletionEvidence,
    *,
    expected_run_file: Path,
    status_file: Path,
    ledger_file: Path,
) -> None:
    status = (
        "phase=IDLE\n"
        "transition=COMPLETE_TO_IDLE\n"
        f"completed_at={evidence.completed_at}\n"
        f"run_name={evidence.run_name}\n"
        f"iteration={evidence.iteration}\n"
        f"iterations_target={evidence.iterations_target}\n"
        f"total_steps={evidence.total_steps}\n"
        f"checkpoint={evidence.checkpoint}\n"
        f"checkpoint_sha256={evidence.checkpoint_sha256}\n"
        f"console_log={evidence.console_log}\n"
        "expected_run=<none>\n"
        "auto_advance=HALTED\n"
    )
    _atomic_write(status_file, status)
    _append_ledger_once(ledger_file, evidence)
    # Clear last so an interrupted transition retries instead of losing evidence.
    _atomic_write(expected_run_file, "")


def transition_if_complete(
    *,
    repo: Path,
    expected_run_file: Path,
    status_file: Path,
    ledger_file: Path,
) -> CompletionEvidence | None:
    run_name = read_expected_run(expected_run_file)
    if run_name is None:
        return None
    evidence = collect_completion_evidence(repo, run_name)
    apply_transition(
        evidence,
        expected_run_file=expected_run_file,
        status_file=status_file,
        ledger_file=ledger_file,
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--expected-run-file", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    parser.add_argument("--ledger-file", type=Path, required=True)
    args = parser.parse_args()

    try:
        evidence = transition_if_complete(
            repo=args.repo,
            expected_run_file=args.expected_run_file,
            status_file=args.status_file,
            ledger_file=args.ledger_file,
        )
    except CompletionNotReady as exc:
        reason = " ".join(str(exc).splitlines())
        print(f"completion_transition=NOT_READY reason={reason}")
        return 1

    if evidence is None:
        print("completion_transition=NO_EXPECTED_RUN")
        return 0
    print(
        "completion_transition=COMPLETE_TO_IDLE "
        f"run={evidence.run_name} "
        f"iteration={evidence.iteration}/{evidence.iterations_target} "
        f"checkpoint={evidence.checkpoint} "
        f"sha256={evidence.checkpoint_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
