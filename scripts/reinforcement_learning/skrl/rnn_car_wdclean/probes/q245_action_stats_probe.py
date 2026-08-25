"""回答車端 Q2 / Q4 / Q5 的訓練端動作統計探針。

刻意重用 `run_sa4_checkpoint_screen` 的場景參數與 `_run_play`，讓條件與已凍結的
SA4 checkpoint screen 完全一致 —— 只多傳一個 `--action_stats_output`。
這樣量到的分布可以直接和 screen 的 SR/CR 對照，不會有「另一套設定」的分歧。

Q2  slew 飽和率：policy 想要的 |Δω| 有多常觸及 `α_max × dt`
Q4  act_hist 分布：obs[79:83] 的 p50 / p95 / 夾到 ±2 的比例
Q5  倒車頻率：實際下達 `next_v < 0` 的比例
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WDCLEAN = HERE.parent
sys.path.insert(0, str(WDCLEAN))

import run_sa4_checkpoint_screen as base  # noqa: E402
import sa4_r3_checkpoint_screen as protocol  # noqa: E402
from run_sa5_joint_retention_gates import _run_play  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--scenario", choices=base.base.SCENARIOS, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--num-envs", type=int, default=64)
    args = ap.parse_args(argv)

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.scenario}_g4_d{protocol.DELAY_STEPS}_s{protocol.SEED}"
    log_path = out / f"{stem}.log"
    corridor_path = out / f"{stem}_corridor.json"
    stats_path = out / f"{stem}_action_stats.json"

    actuator_args = base.base.fixed_actuator_cli_args(
        protocol.DELAY_STEPS, base.base.ACTUATOR_PROFILE
    )
    scene_args = base.build_scene_args(
        args.scenario,
        seed=protocol.SEED,
        actuator_args=actuator_args,
        corridor_json=corridor_path,
    )

    print(
        f"[Q245] checkpoint={ckpt.name} scenario={args.scenario} "
        f"steps={args.steps} envs={args.num_envs} "
        f"eligibility={protocol.ELIGIBILITY} delay_steps={protocol.DELAY_STEPS}",
        flush=True,
    )

    _run_play(
        ckpt,
        log_path,
        num_envs=args.num_envs,
        steps=args.steps,
        extra=[
            *scene_args,
            "--lidar-distractor-eligibility",
            protocol.ELIGIBILITY,
            "--action_stats_output",
            str(stats_path),
        ],
    )

    if not stats_path.is_file():
        raise RuntimeError(f"probe produced no stats file: {stats_path}")
    report = json.loads(stats_path.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
