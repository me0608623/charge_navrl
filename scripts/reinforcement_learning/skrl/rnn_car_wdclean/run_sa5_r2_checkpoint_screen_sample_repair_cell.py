"""Run one longer SA5-R2 cell under the frozen sample-size addendum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_sa5_r2_checkpoint_screen_cell as base_runner
import sa5_r2_checkpoint_screen as base
import sa5_r2_checkpoint_screen_sample_repair as repair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--scenario", choices=base.ALL_SCENARIOS, required=True)
    parser.add_argument("--expect-checkpoint-sha256", required=True)
    parser.add_argument("--expect-addendum-sha256", required=True)
    parser.add_argument("--base-n", type=int, required=True)
    args = parser.parse_args(argv)

    addendum = repair.repair_addendum()
    if args.expect_addendum_sha256 != addendum["sha256"]:
        raise ValueError("CLI addendum hash mismatch")
    base_steps = base.STEPS_BY_SCENARIO[args.scenario]
    repair_steps = repair.repair_steps(base_steps, args.base_n)
    if repair_steps is None:
        raise ValueError("sample-size repair requested for a sufficient cell")
    execution = repair.execution_protocol(args.scenario, repair_steps)

    # The frozen base runner remains the sole implementation of scene setup,
    # runtime checks, metric reconciliation, and cell serialization. This
    # process-local override changes only the allowed rollout duration.
    original_steps = base.STEPS_BY_SCENARIO[args.scenario]
    base.STEPS_BY_SCENARIO[args.scenario] = repair_steps
    try:
        returncode = base_runner.main(
            [
                str(args.checkpoint),
                "--output-dir",
                str(args.output_dir),
                "--checkpoint-name",
                args.checkpoint_name,
                "--scenario",
                args.scenario,
                "--expect-checkpoint-sha256",
                args.expect_checkpoint_sha256,
                "--expect-protocol-sha256",
                execution["sha256"],
                "--steps",
                str(repair_steps),
            ]
        )
    finally:
        base.STEPS_BY_SCENARIO[args.scenario] = original_steps

    stem = f"{args.scenario}_g5_d{base.DELAY_STEPS}_s{base.SEED}"
    cell_path = args.output_dir.expanduser().resolve() / f"{stem}_cell.json"
    if not cell_path.is_file():
        raise RuntimeError("base runner returned without a repair cell JSON")
    payload = json.loads(cell_path.read_text(encoding="utf-8"))
    payload["sample_size_repair"] = {
        "schema": "sa5_r2_checkpoint_screen_cell_sample_repair/v1",
        "addendum_sha256": addendum["sha256"],
        "base_protocol_sha256": base.screen_protocol()["sha256"],
        "execution_protocol_sha256": execution["sha256"],
        "base_n": args.base_n,
        "base_steps": base_steps,
        "repair_steps": repair_steps,
        "target_episodes": repair.TARGET_EPISODES,
        "trigger_uses_outcome_metrics": False,
        "from_scratch_same_seed": True,
        "replaces_base_cell": True,
        "pooled_with_base": False,
    }
    cell_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return returncode


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

