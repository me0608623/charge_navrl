#!/usr/bin/env python
"""det_eval_watcher.py — 訓練中「部署 SR」自動監看器（外部進程，零侵入訓練熱路徑）。

動機（2026-07-14）：訓練 SR 走 stochastic 政策 + ent_coeff floor 撐高熵 → 訓練 SR 被雜訊壓低
（實測 SA4 訓練 0.618 vs det 0.789，gap 17pp）。部署走 argmax，真正的「部署 SR」要 deterministic eval。
本監看器＝獨立進程，偵測到新 checkpoint 就自動跑 `play_rnn_car.py --deterministic`、解析 SR/CR/碰撞分解、
記到 wandb（獨立 eval run，同 project、同 x 軸=total_steps，wandb workspace 可與訓練曲線疊圖）＋ CSV。

★為什麼是外部 watcher 而非 in-loop callback（2026-07-14 用戶選）：
  - 訓練靠 env + RNN 跨 iteration 連續性；in-loop eval 要 env.reset 會擾動訓練連續性 + 需注入 5000 行熱路徑。
  - 外部 watcher 對訓練程式碼「一行不動」＝零風險。代價：eval 頻率 = checkpoint 頻率（save_interval，現 100 iter）。
    想更密 → 調訓練的 --save_interval。

用法：
  # 監看某訓練 run 的 checkpoint 目錄，在 stage 4 密度下跑 det eval
  /home/aa/miniconda3/envs/env_isaaclab/bin/python \
    scripts/reinforcement_learning/skrl/play_eval/det_eval_watcher.py \
    --ckpt_dir logs/rnn_car/sa4_deploy_dense_lr2e4_popartfix_ne1024_s42 \
    --stage 4 --wandb_run_name sa4_deploy_dense_lr2e4_popartfix_ne1024_s42

  # 只解析既有 ckpt 一次、不常駐（--once）；--num_envs/--steps 控 GPU 用量
  ... --once --num_envs 128 --steps 1200

注意：
  - 每次 eval 會啟一個 Isaac Sim（~2-3min startup）跑 play_rnn_car，會與訓練競爭 GPU。ckpt 頻率下（~每 100 iter）
    影響可接受；用小 --num_envs（預設 128）降 GPU 壓力避免 OOM 害訓練。
  - 需 conda env_isaaclab。stage 要對齊訓練 config 的 initial_stage。
  - 冪等：已 eval 過的 ckpt（記在 --state_file / CSV）跳過。
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import time

# ── SR/CR 解析（對齊 play_rnn_car.py 輸出格式）──────────────────────────────
# 目標行例：
#   [累積統計 n=2458]  [step 1000]  SR=78.9%  CR=21.1%  TO=0.0%
#   牆壁碰撞率:    126 (5.1%)
#     ├─ 靜態障礙碰撞:  295 (12.0% of episodes, 73% of 障礙碰撞)
#     └─ 動態障礙碰撞:  110 (4.5% of episodes, 27% of 障礙碰撞)
_RE_SRCR = re.compile(r"SR=([\d.]+)%\s+CR=([\d.]+)%\s+TO=([\d.]+)%")
_RE_WALL = re.compile(r"牆壁碰撞率:\s*\d+\s*\(([\d.]+)%\)")
_RE_STATIC = re.compile(r"靜態障礙碰撞:\s*\d+\s*\(([\d.]+)%")
_RE_DYNAMIC = re.compile(r"動態障礙碰撞:\s*\d+\s*\(([\d.]+)%")
_RE_N = re.compile(r"累積統計 n=(\d+)")


def _last(regex: re.Pattern, text: str, group: int = 1):
    """回傳最後一個 match 的指定 group（float）；無則 None。全程彙總在最後 → 取最完整。"""
    matches = regex.findall(text)
    if not matches:
        return None
    m = matches[-1]
    val = m[group - 1] if isinstance(m, tuple) else m
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def parse_eval_output(text: str) -> dict | None:
    """從 play_rnn_car 輸出解析 det 指標。缺 SR/CR 視為失敗回 None。"""
    sr = _last(_RE_SRCR, text, group=1)
    cr = _last(_RE_SRCR, text, group=2)
    to = _last(_RE_SRCR, text, group=3)
    if sr is None or cr is None:
        return None
    return {
        "det_success_rate": sr / 100.0,
        "det_collision_rate": cr / 100.0,
        "det_timeout_rate": (to / 100.0) if to is not None else None,
        "det_wall_cr": (_last(_RE_WALL, text) or 0.0) / 100.0,
        "det_static_cr": (_last(_RE_STATIC, text) or 0.0) / 100.0,
        "det_dynamic_cr": (_last(_RE_DYNAMIC, text) or 0.0) / 100.0,
        "det_n_episodes": _last(_RE_N, text),
    }


# ── checkpoint 掃描 ─────────────────────────────────────────────────────────
_RE_CKPT = re.compile(r"checkpoint_(\d+)\.pt$")


def scan_checkpoints(ckpt_dir: str) -> list[tuple[int, str]]:
    """回傳 [(total_steps, path), ...] 按 step 排序。"""
    out = []
    if not os.path.isdir(ckpt_dir):
        return out
    for fn in os.listdir(ckpt_dir):
        m = _RE_CKPT.search(fn)
        if m:
            out.append((int(m.group(1)), os.path.join(ckpt_dir, fn)))
    out.sort(key=lambda x: x[0])
    return out


def is_stable(path: str, wait: float = 3.0) -> bool:
    """確認 ckpt 不在寫入中（size 兩次讀取一致）。"""
    try:
        s1 = os.path.getsize(path)
        time.sleep(wait)
        s2 = os.path.getsize(path)
        return s1 == s2 and s1 > 0
    except OSError:
        return False


# ── det eval 執行 ───────────────────────────────────────────────────────────
def run_det_evaluation(args, ckpt_path: str) -> dict | None:
    """跑 play_rnn_car --deterministic，回傳解析後的指標 dict（失敗 None）。"""
    repo = args.repo_root
    play_args = [
        "./isaaclab.sh", "-p",
        "scripts/reinforcement_learning/skrl/play_eval/play_rnn_car.py",
        "--checkpoint", os.path.relpath(ckpt_path, repo),
        "--task", args.task,
        "--curriculum_version", args.curriculum_version,
        "--stage", str(args.stage),
        "--obs_near_goal_count", str(args.obs_near_goal_count),
        "--deterministic", "--headless",
        "--num_envs", str(args.num_envs),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
    ]
    if args.feat_norm:
        play_args.append("--feat_norm")
    if args.extra_args:
        play_args.extend(args.extra_args.split())

    # ★isaaclab.sh 需要 conda env_isaaclab 已啟動才找得到 isaac python；
    #   subprocess 不繼承 shell 的 conda，故明確 source+activate（比照手動 det 腳本）。
    inner = " ".join(play_args)
    bash_cmd = (f"source {args.conda_profile} && conda activate {args.conda_env} && "
                f"PYTHONUNBUFFERED=1 CHARGE_USE_ACT_HIST={args.charge_use_act_hist} {inner}")
    cmd = ["bash", "-c", bash_cmd]

    env = dict(os.environ)

    print(f"[watcher] ▶ det eval: {os.path.basename(ckpt_path)} (stage {args.stage}, "
          f"{args.num_envs}env×{args.steps}步)", flush=True)
    try:
        proc = subprocess.run(
            cmd, cwd=repo, env=env, capture_output=True, text=True,
            timeout=args.eval_timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"[watcher] ✗ eval timeout ({args.eval_timeout}s)", flush=True)
        return None
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    metrics = parse_eval_output(out)
    if metrics is None:
        print(f"[watcher] ✗ 解析失敗（play_rnn_car 可能沒跑完）。retcode={proc.returncode}\n"
              f"  尾 15 行:\n" + "\n".join(out.splitlines()[-15:]), flush=True)
    return metrics


# ── 輸出（CSV + wandb）──────────────────────────────────────────────────────
def append_csv(csv_path: str, step: int, m: dict):
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["total_steps", "det_success_rate", "det_collision_rate",
                        "det_timeout_rate", "det_wall_cr", "det_static_cr",
                        "det_dynamic_cr", "det_n_episodes"])
        w.writerow([step, m["det_success_rate"], m["det_collision_rate"],
                    m.get("det_timeout_rate"), m["det_wall_cr"], m["det_static_cr"],
                    m["det_dynamic_cr"], m.get("det_n_episodes")])


def _wandb_id(run_name: str) -> str:
    """把 run_name 轉成合法 wandb id（≤64, alnum/dash/underscore）。"""
    slug = re.sub(r"[^A-Za-z0-9_-]", "-", run_name) + "-deteval"
    return slug[:64]


def main():
    p = argparse.ArgumentParser(description="訓練中部署 SR 自動監看（外部 det eval）")
    p.add_argument("--ckpt_dir", required=True, help="訓練 run 的 checkpoint 目錄")
    p.add_argument("--stage", type=int, required=True, help="curriculum stage（對齊訓練 initial_stage）")
    p.add_argument("--task", default="Isaac-Navigation-Charge-VLP16-Curriculum-WD")
    p.add_argument("--curriculum_version", default="warp_drive_single_agent_v3e_deploy_dense")
    p.add_argument("--obs_near_goal_count", type=int, default=2)
    p.add_argument("--num_envs", type=int, default=128, help="eval env 數（小=省 GPU 不害訓練）")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feat_norm", action="store_true", default=True)
    p.add_argument("--no_feat_norm", dest="feat_norm", action="store_false")
    p.add_argument("--charge_use_act_hist", default="0", help="CHARGE_USE_ACT_HIST env（v3f=0）")
    p.add_argument("--conda_profile", default="/home/aa/miniconda3/etc/profile.d/conda.sh")
    p.add_argument("--conda_env", default="env_isaaclab")
    p.add_argument("--extra_args", default="", help="額外傳給 play_rnn_car 的參數（空白分隔）")
    p.add_argument("--repo_root", default="/home/aa/IsaacLab")
    p.add_argument("--poll_interval", type=float, default=60.0, help="輪詢 ckpt 目錄秒數")
    p.add_argument("--eval_timeout", type=float, default=900.0, help="單次 eval 上限秒")
    p.add_argument("--once", action="store_true", help="只掃一次既有 ckpt 就退出，不常駐")
    p.add_argument("--wandb_project", default="charge_skrl")
    p.add_argument("--wandb_run_name", default=None, help="預設從 ckpt_dir basename 推")
    p.add_argument("--no_wandb", action="store_true", help="只寫 CSV，不記 wandb")
    p.add_argument("--csv", default=None, help="CSV 輸出路徑（預設 ckpt_dir/det_eval.csv）")
    p.add_argument("--state_file", default=None,
                   help="已 eval ckpt 記錄檔（預設 ckpt_dir/.det_eval_seen）")
    args = p.parse_args()

    run_name = args.wandb_run_name or os.path.basename(os.path.normpath(args.ckpt_dir))
    csv_path = args.csv or os.path.join(args.ckpt_dir, "det_eval.csv")
    state_file = args.state_file or os.path.join(args.ckpt_dir, ".det_eval_seen")

    # 已處理 ckpt（冪等）
    seen: set[int] = set()
    if os.path.isfile(state_file):
        with open(state_file) as f:
            seen = {int(x) for x in f.read().split() if x.strip().isdigit()}

    # wandb（獨立 eval run，同 project；resume 讓 watcher 重啟續記）
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{run_name}_deteval",
                id=_wandb_id(run_name),
                resume="allow",
                config={"source_run": run_name, "eval_stage": args.stage,
                        "eval_num_envs": args.num_envs, "eval_steps": args.steps,
                        "eval_type": "deterministic_argmax"},
                job_type="det_eval",
            )
            print(f"[watcher] wandb eval run: {wandb_run.url}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[watcher] ⚠ wandb init 失敗（改只寫 CSV）: {e}", flush=True)
            wandb_run = None

    print(f"[watcher] 監看 {args.ckpt_dir} | stage {args.stage} | CSV {csv_path} | "
          f"poll {args.poll_interval}s | {'ONCE' if args.once else '常駐'}", flush=True)

    def process_new():
        for step, path in scan_checkpoints(args.ckpt_dir):
            if step in seen:
                continue
            if not is_stable(path):
                print(f"[watcher] ckpt_{step} 仍在寫入，稍後再試", flush=True)
                continue
            m = run_det_evaluation(args, path)
            if m is None:
                # 不標記 seen → 下輪重試
                continue
            append_csv(csv_path, step, m)
            if wandb_run is not None:
                log = {f"eval/{k}": v for k, v in m.items() if v is not None}
                wandb_run.log(log, step=step)
            print(f"[watcher] ✓ ckpt_{step}: det_SR={m['det_success_rate']:.1%} "
                  f"det_CR={m['det_collision_rate']:.1%} "
                  f"(靜{m['det_static_cr']:.1%}/動{m['det_dynamic_cr']:.1%}/牆{m['det_wall_cr']:.1%}) "
                  f"n={m.get('det_n_episodes')}", flush=True)
            seen.add(step)
            with open(state_file, "w") as f:
                f.write(" ".join(str(s) for s in sorted(seen)))

    process_new()
    while not args.once:
        time.sleep(args.poll_interval)
        process_new()

    if wandb_run is not None:
        wandb_run.finish()
    print("[watcher] done.", flush=True)


if __name__ == "__main__":
    main()
