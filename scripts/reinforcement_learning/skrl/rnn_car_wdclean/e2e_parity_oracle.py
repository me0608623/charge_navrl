"""Golden parity oracle for the clean end-to-end frame-stack PPO policy.

部署到車上前的「推論契約」黃金測試。此腳本在 **訓練機（IsaacLab repo）** 端，
用與 play_rnn_car.py 完全一致的 e2e forward，把一段固定 obs 序列跑成 logits/action，
存成 golden npz。車端 export_policy.py 產生的 TorchScript bundle 必須逐 bit 重現這份
golden，才允許上車跑實體。

e2e 資訊流（stateless，4 幀 LiDAR 疊幀，175D head）：

    obs_raw[79]  ── obs_normalizer(79D) -> clamp[-5,5] ─→ obs_normed[79]
    ext_in = cat(obs_normed[79], lidar_hist[(K-1)*72])            # 295D (K=4)
    feat   = LidarStateExtractor(ext_in)                          # 96D
    lidar_hist <- cat(obs_normed[6:78], lidar_hist[:-72])         # prepend 當前幀,丟最舊
    rl_in  = cat(obs_normed[79], feat[96])                        # 175D
    logits = PolicyHead(rl_in)                                    # 38 = [19 linear | 19 angular]
    action = [argmax(logits[:19]), argmax(logits[19:])]          # deterministic

lidar_hist 在 episode reset 時歸零（車端 PolicyRunner.reset()）。

用法:
    python e2e_parity_oracle.py \
        --checkpoint logs/rnn_car/sa4_e2e_fs4_cleanppo_from_sa3c12800_acthist0_ne1024_s42/checkpoint_89600.pt \
        --out /tmp/e2e_golden_sa4c89600.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_SKRL = Path(__file__).resolve().parents[1]  # scripts/reinforcement_learning/skrl
sys.path.insert(0, str(_SKRL / "models"))
from modular_rnn_models import (  # noqa: E402
    ACT_HIST_DIM,
    ACT_HIST_END,
    ACT_HIST_START,
    TIME_END,
    TIME_START,
    LidarStateExtractor,
    PolicyHead,
)

LIDAR_START, LIDAR_END = 6, 78
LIDAR_LEN = 72


def build_obs_sequence(steps: int, num_envs: int, obs_dim: int, seed: int) -> torch.Tensor:
    """產生固定、可重現、涵蓋合理數值範圍的 raw obs 序列 ``[T, N, obs_dim]``。

    數值範圍模擬部署 obs：ego 小值、goal ±數 m、LiDAR 正距離、time in [0,1]。
    序列本身不需真實物理，只需確定性且能完整驅動 forward + 疊幀 buffer。

    ``obs_dim=83`` 時尾端 4 維是 act_hist（過去 2 步 issued
    ``[linear_accel/0.5, omega/1.2]``，車端最終 clip 到 ``[-2, 2]``），
    所以取樣範圍用 ``[-2, 2]`` 覆蓋該契約的完整值域。
    """
    g = torch.Generator().manual_seed(seed)
    obs = torch.zeros(steps, num_envs, obs_dim)
    # ego [0:4] = accel, speed, omega, radius
    obs[..., 0:4] = torch.rand(steps, num_envs, 4, generator=g) * 0.6 - 0.3
    obs[..., 3] = 0.35  # radius 固定
    # goal [4:6] body frame，±4 m
    obs[..., 4:6] = torch.rand(steps, num_envs, 2, generator=g) * 8.0 - 4.0
    # LiDAR [6:78] 正距離，模擬 scaled 值 [0, 1.2]，隨步變化以驅動疊幀
    obs[..., LIDAR_START:LIDAR_END] = torch.rand(steps, num_envs, LIDAR_LEN, generator=g) * 1.2
    # time [78] 遞減比例
    ramp = torch.linspace(1.0, 0.0, steps).reshape(steps, 1)
    obs[..., TIME_START] = ramp.expand(steps, num_envs)
    if obs_dim == ACT_HIST_END:
        obs[..., ACT_HIST_START:ACT_HIST_END] = (
            torch.rand(steps, num_envs, ACT_HIST_DIM, generator=g) * 4.0 - 2.0
        )
    return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--num_envs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260716)
    args = ap.parse_args()

    device = "cpu"
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckargs = ck["args"]
    assert bool(ckargs.get("end_to_end_frame_stack")), "not an e2e checkpoint"
    K = int(ckargs.get("lidar_frame_stack", 4))
    assert K >= 2, "e2e requires lidar_frame_stack >= 2"
    assert not bool(ckargs.get("feat_norm", False)), "oracle assumes feat_norm=false"
    assert "feat_normalizer" not in ck, "oracle assumes no feat_normalizer"

    # --- obs_normalizer (79D) ---
    on = ck["obs_normalizer"]
    mean = on["mean"].to(device).reshape(-1).float()
    var = on["var"].to(device).reshape(-1).float()
    obs_dim = int(mean.shape[-1])
    # 79D = v3f lineage (no action history). 83D = current sim2real lineage,
    # which appends act_hist [79:83]; the frame-stack history still goes last,
    # after act_hist, because the extractor slices it with obs[:, -(K-1)*72:].
    assert obs_dim in (TIME_END, ACT_HIST_END), (
        f"expected {TIME_END}D or {ACT_HIST_END}D normalizer, got {obs_dim}"
    )
    include_act_hist = obs_dim == ACT_HIST_END

    def normalize(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - mean) / (var.sqrt() + 1e-8), -5.0, 5.0)

    # --- models (identical class the car reuses) ---
    # act_hist_dropout is deliberately left at 0: the oracle runs in eval mode
    # where dropout is a passthrough, so a nonzero training value must not
    # change the golden.
    extractor = LidarStateExtractor(
        legacy=False, include_act_hist=include_act_hist, frame_stack=K
    ).to(device)
    extractor.load_state_dict(ck["extractor"])
    extractor.train(False)

    rl_in_dim = obs_dim + extractor.output_dim  # 79+96=175 or 83+96=179
    policy = PolicyHead(input_dim=rl_in_dim).to(device)
    policy.load_state_dict(ck["policy_head"])
    policy.train(False)
    # Fail closed if the checkpoint's head disagrees with the derived input dim;
    # a silent mismatch here would ship a golden the car can never reproduce.
    _first = policy.state_dict()[next(iter(policy.state_dict()))]
    assert _first.shape[-1] == rl_in_dim, (
        f"policy head expects {_first.shape[-1]}D input, derived {rl_in_dim}D"
    )

    # --- deterministic obs sequence ---
    obs_seq = build_obs_sequence(args.steps, args.num_envs, obs_dim, args.seed).to(device)

    hist_dim = (K - 1) * LIDAR_LEN
    lidar_hist = torch.zeros(args.num_envs, hist_dim, device=device)

    all_logits, all_actions = [], []
    with torch.no_grad():
        for t in range(args.steps):
            obs_normed = normalize(obs_seq[t])                       # [N, 79]
            ext_in = torch.cat([obs_normed, lidar_hist], dim=-1)     # [N, 295]
            feat = extractor(ext_in)                                 # [N, 96]
            # 更新滾動 buffer：prepend 當前 normed lidar，丟最舊 72
            lidar_hist = torch.cat(
                [obs_normed[:, LIDAR_START:LIDAR_END], lidar_hist[:, :-LIDAR_LEN]], dim=-1
            )
            rl_in = torch.cat([obs_normed, feat], dim=-1)            # [N, 175]
            logits = policy(rl_in)                                   # [N, 38]
            act_a = logits[:, :19].argmax(dim=-1)
            act_w = logits[:, 19:].argmax(dim=-1)
            all_logits.append(logits.cpu().numpy())
            all_actions.append(torch.stack([act_a, act_w], dim=-1).cpu().numpy())

    logits_arr = np.stack(all_logits, axis=0)   # [T, N, 38]
    actions_arr = np.stack(all_actions, axis=0)  # [T, N, 2]

    np.savez(
        args.out,
        obs_seq=obs_seq.cpu().numpy().astype(np.float32),  # [T, N, 79] RAW obs (pre-normalize)
        logits=logits_arr.astype(np.float32),
        actions=actions_arr.astype(np.int64),
        mean=mean.cpu().numpy().astype(np.float32),
        var=var.cpu().numpy().astype(np.float32),
        frame_stack=np.int64(K),
        rl_in_dim=np.int64(rl_in_dim),
        checkpoint=np.array(str(args.checkpoint)),
    )
    print(f"[oracle] wrote {args.out}")
    print(f"[oracle] steps={args.steps} envs={args.num_envs} K={K} rl_in={rl_in_dim}")
    print(f"[oracle] logits[0,0,:5]={logits_arr[0,0,:5]}")
    print(f"[oracle] actions[:,0]=\n{actions_arr[:,0]}")


if __name__ == "__main__":
    main()
