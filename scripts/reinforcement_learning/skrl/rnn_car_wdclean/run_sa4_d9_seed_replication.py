"""Run the frozen SA4-D9 A/B on evaluator seeds 515 and 616."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))

import d9_noise_closed_loop_ab as d9  # noqa: E402
import d9_seed_replication as replication  # noqa: E402
import run_sa4_d3_baseline as baseline  # noqa: E402
import run_sa4_d4_geometry_suite as d4_suite  # noqa: E402
import run_sa4_d9_noise_closed_loop_ab as d9_runner  # noqa: E402
from run_sa5_joint_retention_gates import _run_play  # noqa: E402
from validate_gates import parse_play_summary  # noqa: E402


def source_fingerprint() -> dict[str, str]:
    fingerprints = {
        "replication_runner": d9_runner.sha256_of(Path(__file__).resolve()),
        "replication_protocol": d9_runner.sha256_of(Path(replication.__file__).resolve()),
    }
    fingerprints.update(
        {f"d9_{name}": digest for name, digest in d9_runner.source_fingerprint().items()}
    )
    return fingerprints


def _load_seed818_manifest(path: Path) -> dict:
    if d9_runner.sha256_of(path) != replication.ORIGINAL_SEED818_MANIFEST_SHA256:
        raise RuntimeError("seed818 D9 manifest SHA-256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE_VALID_DIAGNOSTIC_EVIDENCE":
        raise RuntimeError("seed818 D9 manifest is not complete valid evidence")
    if payload.get("checkpoint_sha256") != d9.CHECKPOINT_SHA256:
        raise RuntimeError("seed818 D9 checkpoint hash mismatch")
    if (payload.get("protocol") or {}).get("sha256") != d9.closed_loop_protocol()["sha256"]:
        raise RuntimeError("seed818 D9 protocol hash mismatch")
    return payload


def _render(result: dict) -> str:
    lines = [
        "# SA4-D9 cross-seed replication",
        "",
        "> Seeds 515 and 616 were fixed before rollout and are combined with the immutable seed818 D9 manifest.",
        "",
        "| seed | A CR | B CR | Delta CR | A wall CR | B wall CR | Delta wall | qualifies |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in replication.ALL_SEEDS:
        item = result["per_seed"][str(seed)]
        comparison = item["comparison"]
        a = comparison["arms"][d9.CURRENT_ARM]
        b = comparison["arms"][d9.CORRECTED_ARM]
        delta = comparison["delta_b_minus_a"]
        lines.append(
            f"| {seed} | {a['cr']:.4%} | {b['cr']:.4%} | "
            f"{delta['cr'] * 100:+.2f} pp | {a['wall_cr']:.4%} | "
            f"{b['wall_cr']:.4%} | {delta['wall_cr'] * 100:+.2f} pp | "
            f"{item['qualifies_for_majority']} |"
        )
    lines.extend(
        [
            "",
            f"- qualifying seeds: {result['qualifying_seed_count']}/{result['seed_count']}",
            f"- majority required: {result['majority_required']}",
            "- SA4-R3 short pilot authorized: "
            f"{result['sa4_r3_short_pilot_authorized']}",
            "- SA5 authorized: False",
            "",
            "These are evaluator-seed replications, not independent training runs or an inferential proof.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed818-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    seed818_manifest_path = args.seed818_manifest.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not seed818_manifest_path.is_file():
        raise FileNotFoundError(seed818_manifest_path)
    checkpoint_sha256 = d9_runner.sha256_of(checkpoint)
    if checkpoint_sha256 != args.expect_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint hash mismatch: expected {args.expect_checkpoint_sha256}, "
            f"got {checkpoint_sha256}"
        )
    if checkpoint_sha256 != d9.CHECKPOINT_SHA256:
        raise RuntimeError("checkpoint does not match the frozen D9 protocol")
    seed818_manifest = _load_seed818_manifest(seed818_manifest_path)
    busy = d9_runner.gpu_is_busy()
    if busy:
        raise RuntimeError(f"D9 replication refuses to share a busy GPU: {busy}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common_paths = {
        "protocol": output / "PROTOCOL.json",
        "summary": output / "SUMMARY.md",
        "manifest": output / "suite_manifest.json",
        "incomplete": output / "D9_REPLICATION_INCOMPLETE.json",
        "source_before": output / "source_fingerprint_before.json",
        "source_after": output / "source_fingerprint_after.json",
    }
    arm_paths = {
        seed: {
            arm: {
                "log": output / f"{arm}_lateral_g4_d1_s{seed}.log",
                "corridor": output / f"{arm}_lateral_g4_d1_s{seed}_corridor.json",
                "cell": output / f"{arm}_lateral_g4_d1_s{seed}_cell.json",
            }
            for arm in d9.ARMS
        }
        for seed in replication.NEW_SEEDS
    }
    d9_runner._require_new_targets(
        list(common_paths.values())
        + [
            path
            for seed_paths in arm_paths.values()
            for paths in seed_paths.values()
            for path in paths.values()
        ]
    )

    protocol = replication.replication_protocol()
    planned_fingerprint = source_fingerprint()
    preregistration = {
        "status": "PREREGISTERED_BEFORE_ROLLOUT",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "seed818_manifest": str(seed818_manifest_path),
        "seed818_manifest_sha256": replication.ORIGINAL_SEED818_MANIFEST_SHA256,
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
        f"[SA4-D9-REPLICATION] protocol frozen before rollout: {protocol['sha256']}",
        flush=True,
    )

    completed: dict[int, dict[str, dict]] = {}
    try:
        values = baseline.sa4.scene_values()
        actuator_args = baseline.sa4.base.fixed_actuator_cli_args(
            baseline.DELAY_STEPS, baseline.ACTUATOR_PROFILE
        )
        for seed in replication.NEW_SEEDS:
            completed[seed] = {}
            for arm in d9.ARMS:
                eligibility = d9.ELIGIBILITY_BY_ARM[arm]
                paths = arm_paths[seed][arm]
                scene_args = baseline.sa4.build_scene_args(
                    baseline.SCENARIO,
                    seed=seed,
                    actuator_args=actuator_args,
                    corridor_json=paths["corridor"],
                )
                print(
                    f"[SA4-D9-REPLICATION] seed={seed} arm={arm} "
                    f"eligibility={eligibility} starting",
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
                    raise RuntimeError(
                        f"D9 replication source drift at seed={seed} arm={arm}"
                    )
                d9_runner._validate_runtime_log(paths["log"], eligibility)
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
                d9_runner._validate_metrics(metrics)
                cell = {
                    "schema": "sa4_d9_seed_replication_cell/v1",
                    "seed": seed,
                    "arm": arm,
                    "distractor_eligibility": eligibility,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_sha256,
                    "protocol_sha256": protocol["sha256"],
                    "stage_banner": stage_banner,
                    "scene_contract": scene_contract,
                    "metrics": metrics,
                    "source_fingerprint": observed_fingerprint,
                    "files": {key: str(path) for key, path in paths.items()},
                }
                paths["cell"].write_text(
                    json.dumps(cell, indent=2, sort_keys=True), encoding="utf-8"
                )
                completed[seed][arm] = metrics
                print(
                    f"[SA4-D9-REPLICATION] seed={seed} arm={arm} "
                    f"n={metrics['n']} SR={metrics['sr']:.4f} "
                    f"CR={metrics['cr']:.4f} TO={metrics['to']:.4f}",
                    flush=True,
                )

        final_fingerprint = source_fingerprint()
        if final_fingerprint != planned_fingerprint:
            raise RuntimeError("D9 replication source drift after all cells")
        common_paths["source_after"].write_text(
            json.dumps(final_fingerprint, indent=2, sort_keys=True), encoding="utf-8"
        )
        comparisons = {
            seed: d9.compare_closed_loop(arms) for seed, arms in completed.items()
        }
        comparisons[818] = seed818_manifest["comparison"]
        result = replication.compare_replications(comparisons)
        common_paths["summary"].write_text(_render(result), encoding="utf-8")
        manifest = {
            "schema": "sa4_d9_seed_replication_bundle/v1",
            "status": "COMPLETE_VALID_REPLICATION_EVIDENCE",
            "training_started": False,
            "next_stage_started": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "seed818_manifest": str(seed818_manifest_path),
            "protocol": protocol,
            "result": result,
            "source_fingerprint": final_fingerprint,
            "files": {
                **{
                    name: str(path)
                    for name, path in common_paths.items()
                    if name != "incomplete"
                },
                "new_cells": {
                    str(seed): {
                        arm: {key: str(path) for key, path in paths.items()}
                        for arm, paths in seed_paths.items()
                    }
                    for seed, seed_paths in arm_paths.items()
                },
            },
        }
        common_paths["manifest"].write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(
            "[SA4-D9-REPLICATION] COMPLETE "
            f"qualifying={result['qualifying_seed_count']}/{result['seed_count']} "
            f"pilot_authorized={result['sa4_r3_short_pilot_authorized']}",
            flush=True,
        )
        return 0
    except BaseException as exc:
        common_paths["incomplete"].write_text(
            json.dumps(
                {
                    "schema": "sa4_d9_seed_replication_incomplete/v1",
                    "status": "INCOMPLETE_NO_VERDICT",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "completed_new_cells": completed,
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

