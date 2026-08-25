"""量測訓練場景中 goal 觀測（obs[4:6]）的距離分布。

動機：車端 goal 由 SubgoalSelector 以 pure-pursuit carrot 產生，
`path_lookahead_m = 2.0`，實測分布極窄（p05→p95 僅 2.00→2.05 或 3.18→3.42）。
訓練端的 goal 則由場景直接指定，分布未知。

若兩者分布差異大，則 `obs[4:6]` 存在訓練／部署的分布落差 ——
這是 parity 抓不到、且不在現有部署契約清單裡的一項。

⚠️ 同時檢查 ±10 m 裁切（`functions.py` 的 clamp）的觸及比例。
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

GOAL_CLAMP_M = 10.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--scenario", choices=base.base.SCENARIOS, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--num-envs", type=int, default=32)
    ap.add_argument(
        "--geometry-stage",
        type=int,
        default=None,
        help=(
            "override the scene geometry stage (default: the frozen screen's "
            "stage 4). Use 1 to measure SA1, whose 20x20 m arena is the largest "
            "and therefore the most likely to reach the +/-10 m goal clamp."
        ),
    )
    args = ap.parse_args(argv)

    ckpt = args.checkpoint.expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage = args.geometry_stage
    stem = f"goaldist_{args.scenario}_g{stage or 4}_s{protocol.SEED}"
    log_path = out / f"{stem}.log"
    corridor_path = out / f"{stem}_corridor.json"
    stats_path = out / f"{stem}_goal_stats.json"

    actuator_args = base.base.fixed_actuator_cli_args(
        protocol.DELAY_STEPS, base.base.ACTUATOR_PROFILE
    )
    if stage is None:
        scene_args = base.build_scene_args(
            args.scenario,
            seed=protocol.SEED,
            actuator_args=actuator_args,
            corridor_json=corridor_path,
        )
    else:
        # 直接呼叫底層，繞過 screen wrapper 寫死的 stage 4
        scene_args = base.base.build_scene_args(
            args.scenario,
            geometry_stage=stage,
            seed=protocol.SEED,
            actuator_args=actuator_args,
            corridor_json=corridor_path,
        )

    print(
        f"[GOALDIST] checkpoint={ckpt.name} scenario={args.scenario} "
        f"stage={stage or 4} steps={args.steps} envs={args.num_envs}",
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
            "--goal_distance_stats_output",
            str(stats_path),
        ],
    )

    if not stats_path.is_file():
        raise RuntimeError(f"probe produced no stats: {stats_path}")
    print(json.dumps(json.loads(stats_path.read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
