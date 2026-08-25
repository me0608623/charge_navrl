"""Run the frozen SA4-D9 current-vs-valid-return-only closed-loop A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import d9_noise_closed_loop_ab as d9  # noqa: E402
import run_sa4_d3_baseline as baseline  # noqa: E402
import run_sa4_d4_geometry_suite as d4_suite  # noqa: E402
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


OBS_FUNCTIONS = (
    REPO
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
    / "config/charge_skrl/mdp/observations/obs_functions.py"
)
ENV_OVERRIDES = REPO / "scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py"


def sha256_of(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_fingerprint() -> dict[str, str]:
    files = {
        "runner": Path(__file__).resolve(),
        "comparison": Path(d9.__file__).resolve(),
        "baseline_runner": Path(baseline.__file__).resolve(),
        "play": baseline.PLAY,
        "action_term": baseline.ACTION_TERM,
        "lidar_observation": OBS_FUNCTIONS,
        "env_overrides": ENV_OVERRIDES,
    }
    return {name: sha256_of(path) for name, path in files.items()}


def _require_new_targets(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "D9 closed-loop A/B refuses to overwrite evidence: "
            + ", ".join(existing)
        )


def gpu_is_busy() -> str | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()!r}")
    heavy = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3 and int(fields[2]) > 2000:
            heavy.append(line.strip())
    return "; ".join(heavy) if heavy else None


def _validate_runtime_log(log_path: Path, eligibility: str) -> None:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    marker = f"[SIM2REAL][mixed-pixel-eligibility] mode={eligibility}"
    problems = []
    if marker not in text:
        problems.append(f"missing runtime eligibility marker {marker!r}")
    for label, pattern in (
        ("traceback", r"Traceback \(most recent call last\)"),
        ("CUDA OOM", r"CUDA out of memory"),
        ("CUDA error", r"CUDA error:"),
        ("NaN", r"(?<![A-Za-z])nan(?![A-Za-z])"),
    ):
        if re.search(pattern, text, flags=re.IGNORECASE):
            problems.append(label)
    if problems:
        raise RuntimeError(
            f"invalid D9 runtime log {log_path}: " + "; ".join(problems)
        )


def _validate_metrics(metrics: dict) -> None:
    if int(metrics["n"]) < 1000:
        raise RuntimeError(f"D9 arm has too few episodes: {metrics['n']}")
    for key in ("sr", "cr", "to", "obstacle_cr", "wall_cr"):
        value = float(metrics[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise RuntimeError(f"invalid D9 metric {key}={value!r}")
    if abs(float(metrics["sr"]) + float(metrics["cr"]) + float(metrics["to"]) - 1.0) > 1.0e-6:
        raise RuntimeError("D9 SR/CR/TO do not reconcile")


def _render(comparison: dict) -> str:
    current = comparison["arms"][d9.CURRENT_ARM]
    corrected = comparison["arms"][d9.CORRECTED_ARM]
    delta = comparison["delta_b_minus_a"]
    lines = [
        "# SA4-D9 valid-return-only noise closed-loop A/B",
        "",
        "> Fixed SA4 c6400 lateral, seed818, d1=200 ms, 64 env x 2500 steps per arm. B - A is descriptive single-seed evidence.",
        "",
        "| outcome | A current_full | B valid_return_only | B - A |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("SR", "sr"),
        ("CR", "cr"),
        ("obstacle CR", "obstacle_cr"),
        ("wall CR", "wall_cr"),
        ("TO", "to"),
    ):
        lines.append(
            f"| {label} | {current[key]:.4%} | {corrected[key]:.4%} | "
            f"{delta[key] * 100:+.2f} pp |"
        )
    lines.extend(
        [
            "",
            f"- episodes A/B: {current['n']:,} / {corrected['n']:,}",
            "- directional closed-loop improvement observed: "
            f"{comparison['directional_closed_loop_improvement_observed']}",
            "- material (>=0.5 pp) improvement observed: "
            f"{comparison['material_closed_loop_improvement_observed']}",
            "- corrected arm passes SA4 SR/CR/TO thresholds: "
            f"{comparison['corrected_arm_sa4_threshold_pass']}",
            "- parent or SA5 authorized: False",
            "",
            "This A/B does not validate the replacement rule as a faithful VLP-16 model and does not generalize beyond this checkpoint/family/seed.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = sha256_of(checkpoint)
    if checkpoint_sha256 != args.expect_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch: expected {args.expect_checkpoint_sha256}, "
            f"got {checkpoint_sha256}"
        )
    if checkpoint_sha256 != d9.CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint does not match the frozen D9 protocol")
    busy = gpu_is_busy()
    if busy:
        raise RuntimeError(f"D9 refuses to share a busy GPU: {busy}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common_paths = {
        "protocol": output / "PROTOCOL.json",
        "summary": output / "SUMMARY.md",
        "manifest": output / "suite_manifest.json",
        "incomplete": output / "D9_INCOMPLETE.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_between": output / "source_fingerprint_between.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    arm_paths = {
        arm: {
            "log": output / f"{arm}_lateral_g4_d1_s818.log",
            "corridor": output / f"{arm}_lateral_g4_d1_s818_corridor.json",
            "cell": output / f"{arm}_lateral_g4_d1_s818_cell.json",
        }
        for arm in d9.ARMS
    }
    _require_new_targets(
        list(common_paths.values())
        + [path for paths in arm_paths.values() for path in paths.values()]
    )

    protocol = d9.closed_loop_protocol()
    planned_fingerprint = source_fingerprint()
    preregistration = {
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "protocol": protocol,
        "source_fingerprint": planned_fingerprint,
    }
    common_paths["protocol"].write_text(
        json.dumps(preregistration, indent=2, sort_keys=True), encoding="utf-8"
    )
    common_paths["source_before"].write_text(
        json.dumps(planned_fingerprint, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[SA4-D9] protocol frozen before rollout: {protocol['sha256']}",
        flush=True,
    )

    completed: dict[str, dict] = {}
    try:
        values = baseline.sa4.scene_values()
        actuator_args = baseline.sa4.base.fixed_actuator_cli_args(
            baseline.DELAY_STEPS, baseline.ACTUATOR_PROFILE
        )
        for index, arm in enumerate(d9.ARMS):
            eligibility = d9.ELIGIBILITY_BY_ARM[arm]
            paths = arm_paths[arm]
            scene_args = baseline.sa4.build_scene_args(
                baseline.SCENARIO,
                seed=baseline.SEED,
                actuator_args=actuator_args,
                corridor_json=paths["corridor"],
            )
            print(
                f"[SA4-D9] arm={arm} eligibility={eligibility} starting",
                flush=True,
            )
            _run_play(
                checkpoint,
                paths["log"],
                num_envs=baseline.NUM_ENVS,
                steps=baseline.ROLLOUT_STEPS,
                extra=[
                    *scene_args,
                    "--lidar-distractor-eligibility",
                    eligibility,
                ],
            )
            observed_fingerprint = source_fingerprint()
            if observed_fingerprint != planned_fingerprint:
                raise RuntimeError(f"D9 source drift during arm {arm}")
            if index == 0:
                common_paths["source_between"].write_text(
                    json.dumps(observed_fingerprint, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            _validate_runtime_log(paths["log"], eligibility)
            baseline.sa4.base.verify_fixed_actuator_runtime(
                paths["log"], baseline.DELAY_STEPS, baseline.ACTUATOR_PROFILE
            )
            stage_banner = baseline.sa4.base.verify_common_runtime(
                paths["log"], values
            )
            console = parse_play_summary(str(paths["log"]))
            if not console or console.get("sr") is None:
                raise RuntimeError(f"no play outcome summary in {paths['log']}")
            corridor = json.loads(paths["corridor"].read_text(encoding="utf-8"))
            scene_contract = baseline.sa4.base.verify_corridor_runtime(
                paths["log"], corridor, "lateral", values
            )
            baseline.sa4.base.reconcile_corridor_metrics(console, corridor)
            metrics = d4_suite._metrics(corridor)
            _validate_metrics(metrics)
            cell = {
                "schema": "sa4_d9_noise_closed_loop_cell/v1",
                "arm": arm,
                "distractor_eligibility": eligibility,
                "counterfactual_fed_to_policy": arm == d9.CORRECTED_ARM,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "protocol_sha256": protocol["sha256"],
                "fixed_cell": protocol["fixed_cell"],
                "stage_banner": stage_banner,
                "scene_contract": scene_contract,
                "metrics": metrics,
                "source_fingerprint": observed_fingerprint,
                "files": {key: str(path) for key, path in paths.items()},
            }
            paths["cell"].write_text(
                json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
            )
            completed[arm] = metrics
            print(
                f"[SA4-D9] arm={arm} n={metrics['n']} "
                f"SR={metrics['sr']:.4f} CR={metrics['cr']:.4f} "
                f"TO={metrics['to']:.4f}",
                flush=True,
            )

        final_fingerprint = source_fingerprint()
        if final_fingerprint != planned_fingerprint:
            raise RuntimeError("D9 source drift after both arms")
        common_paths["source_after"].write_text(
            json.dumps(final_fingerprint, indent=2, sort_keys=True), encoding="utf-8"
        )
        comparison = d9.compare_closed_loop(completed)
        common_paths["summary"].write_text(
            _render(comparison), encoding="utf-8"
        )
        manifest = {
            "schema": "sa4_d9_noise_closed_loop_bundle/v1",
            "status": "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE",
            "not_a_graduation_gate": True,
            "training_started": False,
            "next_stage_started": False,
            "action_override_used": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "protocol": protocol,
            "comparison": comparison,
            "source_fingerprint": final_fingerprint,
            "files": {
                **{name: str(path) for name, path in common_paths.items() if name != "incomplete"},
                "arms": {
                    arm: {key: str(path) for key, path in paths.items()}
                    for arm, paths in arm_paths.items()
                },
            },
        }
        common_paths["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            f"[SA4-D9] COMPLETE valid diagnostic evidence: {common_paths['manifest']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        common_paths["incomplete"].write_text(
            json.dumps(
                {
                    "schema": "sa4_d9_noise_closed_loop_incomplete/v1",
                    "status": "INCOMPLETE_NO_VERDICT",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_arms": completed,
                    "protocol_sha256": protocol["sha256"],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
