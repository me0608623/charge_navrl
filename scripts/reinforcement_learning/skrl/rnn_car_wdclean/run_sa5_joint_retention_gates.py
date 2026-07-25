"""Run the fixed SA5 Gate2, Gate5 and deployment-corridor gate suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path("/home/aa/IsaacLab")
PYTHON = Path("/home/aa/miniconda3/envs/env_isaaclab/bin/python")
PLAY = REPO / "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py"
CURRICULUM = "warp_drive_e2e_final20_v1"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_gates import (  # noqa: E402
    NARROW_DEPLOY_THRESH,
    STAGE_THRESH,
    agg_seeds,
    narrow_gap_checks,
    parse_narrow_gap,
)


# Multi-expert routing：若 main() 填入 --multi_expert 相關旗標，_run_play 會把它
# 併入每個場景的 play 命令（additive；空 list = 單專家行為完全不變）。
_GATE_EXTRA_APPEND: list[str] = []


def _run_play(
    checkpoint: Path,
    log_path: Path,
    *,
    num_envs: int,
    steps: int,
    extra: list[str],
) -> None:
    command = [
        str(REPO / "isaaclab.sh"),
        "-p",
        str(PLAY),
        "--checkpoint",
        str(checkpoint),
        "--task",
        "Isaac-Navigation-Charge-VLP16-Curriculum-WD",
        "--curriculum_version",
        CURRICULUM,
        "--deterministic",
        "--num_envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--headless",
        "--num_goals_override",
        "1",
        "--no_goal_movement",
        *extra,
        *_GATE_EXTRA_APPEND,
    ]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["CONDA_PREFIX"] = str(PYTHON.parent.parent)
    env["CONDA_DEFAULT_ENV"] = "env_isaaclab"
    env["PATH"] = f"{PYTHON.parent}:{env['PATH']}"
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    if result.returncode != 0:
        raise RuntimeError(
            f"play exited {result.returncode}: {log_path}"
        )
    if "Traceback (most recent call last)" in text or "RuntimeError:" in text:
        raise RuntimeError(f"simulator traceback: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument(
        "--corridor-static",
        type=int,
        default=4,
        help="Corridor diagnostic static count; formal default remains 4.",
    )
    parser.add_argument(
        "--corridor-dynamic",
        type=int,
        default=2,
        help="Corridor diagnostic dynamic count; formal default remains 2.",
    )
    # Multi-expert routing（additive；不給 --corridor-ckpt/--narrow-ckpt 時 = 單專家行為不變）
    parser.add_argument("--corridor-ckpt", type=Path, default=None,
                        help="走廊專家 checkpoint；提供時啟用 multi-expert 路由。")
    parser.add_argument("--narrow-ckpt", type=Path, default=None,
                        help="窄縫專家 checkpoint；提供時啟用 multi-expert 路由。")
    parser.add_argument("--router-mode", type=str, default="rule",
                        choices=["rule", "always_corridor", "always_narrow"],
                        help="multi-expert 路由模式（parity 用 always_*）。")
    parser.add_argument("--router-side-clear-thresh", type=float, default=1.10)
    parser.add_argument("--router-front-min-clear", type=float, default=0.55)
    parser.add_argument("--router-front-cone-deg", type=float, default=30.0)
    parser.add_argument("--router-side-cone-deg", type=float, default=40.0)
    args = parser.parse_args()

    # 啟用 multi-expert：把旗標填入全域，_run_play 會併入每個場景 play 命令。
    if args.corridor_ckpt is not None or args.narrow_ckpt is not None:
        if args.corridor_ckpt is None or args.narrow_ckpt is None:
            raise ValueError("multi-expert 需同時提供 --corridor-ckpt 與 --narrow-ckpt")
        _GATE_EXTRA_APPEND.extend([
            "--multi_expert",
            "--corridor_ckpt", str(args.corridor_ckpt.expanduser().resolve()),
            "--narrow_ckpt", str(args.narrow_ckpt.expanduser().resolve()),
            "--router_mode", args.router_mode,
            "--router_side_clear_thresh", str(args.router_side_clear_thresh),
            "--router_front_min_clear", str(args.router_front_min_clear),
            "--router_front_cone_deg", str(args.router_front_cone_deg),
            "--router_side_cone_deg", str(args.router_side_cone_deg),
        ])
        print(f"[JOINT-GATE] multi-expert routing enabled: mode={args.router_mode} "
              f"corridor={args.corridor_ckpt} narrow={args.narrow_ckpt}", flush=True)

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    det_logs: list[str] = []
    for seed in (101, 202, 303):
        log = output / f"det_s{seed}.log"
        print(f"[JOINT-GATE] Gate2 seed={seed}", flush=True)
        _run_play(
            checkpoint,
            log,
            num_envs=args.num_envs,
            steps=args.steps,
            extra=[
                "--stage", "5",
                "--arena_size", "15",
                "--num_static_obs", "10",
                "--num_dynamic_obs", "3",
                "--obs_near_goal_count", "0",
                "--seed", str(seed),
            ],
        )
        det_logs.append(str(log))

    print("[JOINT-GATE] Gate5 1.2m", flush=True)
    narrow_log = output / "narrow_1p2.log"
    _run_play(
        checkpoint,
        narrow_log,
        num_envs=args.num_envs,
        steps=args.steps,
        extra=[
            "--stage", "5",
            "--arena_size", "10",
            "--num_static_obs", "0",
            "--num_dynamic_obs", "0",
            "--obs_near_goal_count", "0",
            "--narrow_gap_eval",
            "--narrow_gap_width", "1.2",
            "--narrow_gap_yaw_limit_deg", "10",
            "--seed", "404",
        ],
    )

    print(
        "[JOINT-GATE] corridor 4m x 10m "
        f"{args.corridor_static}S+{args.corridor_dynamic}D",
        flush=True,
    )
    corridor_log = output / "long_corridor.log"
    corridor_json = output / "long_corridor.json"
    _run_play(
        checkpoint,
        corridor_log,
        num_envs=args.num_envs,
        steps=args.steps,
        extra=[
            "--stage", "5",
            "--arena_size", "12",
            "--obs_near_goal_count", "0",
            "--long_corridor_eval",
            "--long_corridor_static_obstacles",
            str(args.corridor_static),
            "--long_corridor_dynamic_obstacles",
            str(args.corridor_dynamic),
            "--long_corridor_output", str(corridor_json),
            "--seed", "515",
        ],
    )

    det = agg_seeds(det_logs)
    narrow = parse_narrow_gap(str(narrow_log))
    corridor = json.loads(corridor_json.read_text(encoding="utf-8"))
    stage_threshold = STAGE_THRESH[5]
    gate2_pass = bool(
        det["sr"] is not None
        and det["sr"] >= stage_threshold["det_sr"]
        and det["cr"] <= stage_threshold["det_cr"]
        and det["to"] <= stage_threshold["det_to"]
    )
    narrow_checks = narrow_gap_checks(narrow or {})
    gate5_pass = bool(narrow_checks) and all(
        passed for passed, _ in narrow_checks.values()
    )
    corridor_pass = bool(corridor.get("gate_pass", False))
    report = {
        "checkpoint": str(checkpoint),
        "gate2": {
            **det,
            "thresholds": {
                "sr_min": stage_threshold["det_sr"],
                "cr_max": stage_threshold["det_cr"],
                "to_max": stage_threshold["det_to"],
            },
            "pass": gate2_pass,
        },
        "gate5_1p2m": {
            **(narrow or {}),
            "thresholds": NARROW_DEPLOY_THRESH,
            "pass": gate5_pass,
        },
        "long_corridor": corridor,
        "joint_pass": gate2_pass and gate5_pass and corridor_pass,
    }
    report_path = output / "joint_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "[JOINT-GATE] "
        f"Gate2={'PASS' if gate2_pass else 'FAIL'} "
        f"(SR={det['sr']:.3f} CR={det['cr']:.3f} TO={det['to']:.3f}) | "
        f"Gate5={'PASS' if gate5_pass else 'FAIL'} "
        f"(SR={(narrow or {}).get('sr', float('nan')):.3f} "
        f"CR={(narrow or {}).get('cr', float('nan')):.3f} "
        f"crossing={(narrow or {}).get('crossing_rate', float('nan')):.3f}) | "
        f"Corridor={'PASS' if corridor_pass else 'FAIL'} "
        f"(SR={corridor['success_rate']:.3f} "
        f"CR={corridor['collision_rate']:.3f} "
        f"TO={corridor['timeout_rate']:.3f}) | "
        f"JOINT={'PASS' if report['joint_pass'] else 'FAIL'}",
        flush=True,
    )
    print(f"[JOINT-GATE] report={report_path}", flush=True)
    return 0 if report["joint_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
