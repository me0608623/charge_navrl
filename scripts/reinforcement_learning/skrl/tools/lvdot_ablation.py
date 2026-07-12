#!/usr/bin/env python
"""LV-DOT channel 因果 ablation — RL policy 是否真的利用障礙 channel 的驗收關卡。

背景 (2026-07-08 資料流裁決)：LV-DOT 障礙 channel 接在 obs[79:109]，架構上繞過
extractor/RNN(它們只 slice LiDAR/ego/goal/time)，只經 PolicyHead 直連 rl_input[79:109]。
因此可做「乾淨因果 ablation」：固定 preprocess_feat，只 toggle 障礙 channel(ON/OFF)，
輸出分布變化 100% 歸因於 channel。

用途 = 每個 stage checkpoint 的驗收關卡：
  - 無稅 baseline (SA1/SA2)：速度頭 KL ~ 0.0001 (policy 不用 channel 改速度) = 已知基準。
  - 加距離稅後 (SA3+)：期望速度頭 KL 竄升 (> 0.01) = value-function 傳播讓 policy 真的
    拿障礙速度做提早減速。若仍 ~ 0 → 稅需升級成顯式吃速度 (TTC / threat x (1+beta*v_closing))。

用法:
  ENVPY scripts/reinforcement_learning/skrl/tools/lvdot_ablation.py \
    --checkpoint logs/rnn_car/sa2_lvdot_ne1024_s42/checkpoint_XXXXX.pt
"""
import argparse
import torch
import torch.nn as nn

TOTAL_LOGITS = 38
OBST_START, OBST_END = 79, 109
NORM_CLIP = 5.0


class PolicyHead(nn.Module):
    """對齊 models/modular_rnn_models.py::PolicyHead (無 privileged 分支)。"""

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, TOTAL_LOGITS),
        )

    def forward(self, x):
        return self.net(x)


def _kl(p, q):
    return (p * (torch.log(p + 1e-9) - torch.log(q + 1e-9))).sum(-1).mean().item()


def run_ablation(checkpoint, batch=4096, seed=0):
    torch.manual_seed(seed)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "policy_head" not in ck or "obs_normalizer" not in ck:
        raise SystemExit("[ERR] checkpoint 缺 policy_head/obs_normalizer: " + checkpoint)

    w0 = ck["policy_head"]["net.0.weight"]
    in_dim = w0.shape[1]
    ph = PolicyHead(in_dim)
    ph.load_state_dict(ck["policy_head"])
    ph.eval()

    nm = ck["obs_normalizer"]
    mean = nm["mean"].float().reshape(-1)
    var = nm["var"].float().reshape(-1)
    obs_dim = mean.numel()
    pre_dim = in_dim - obs_dim
    if obs_dim < OBST_END:
        raise SystemExit(
            "[ERR] normalizer 維度 %d < %d: 此 checkpoint 非 LV-DOT 佈局。" % (obs_dim, OBST_END))

    it = int(ck.get("iteration", -1))
    print("checkpoint : " + checkpoint)
    print("iteration  : %d   policy_head in_dim = %d (obs %d + preprocess %d)" % (it, in_dim, obs_dim, pre_dim))

    def normalize(raw):
        return torch.clamp((raw - mean) / torch.sqrt(var + 1e-8), -NORM_CLIP, NORM_CLIP)

    base_raw = mean[:79] + torch.randn(batch, 79) * torch.sqrt(var[:79] + 1e-8)
    pre = torch.randn(batch, pre_dim) * 4.0

    def dists(obst_raw, base=None):
        braw = base_raw if base is None else base
        raw = torch.cat([braw, obst_raw], dim=1)
        rl = torch.cat([normalize(raw), pre], dim=1)
        with torch.no_grad():
            lg = ph(rl)
        return torch.softmax(lg[:, :19], -1), torch.softmax(lg[:, 19:], -1)

    off = torch.zeros(batch, obs_dim - 79)
    on = torch.zeros(batch, obs_dim - 79)
    px = (0.6 + 2.4 * torch.rand(batch)) / 8.0
    py = (torch.rand(batch) - 0.5) * 1.0 / 8.0
    vx = -(0.3 + 0.7 * torch.rand(batch)) / 1.5
    vy = (torch.rand(batch) - 0.5) * 0.6 / 1.5
    r = 0.3 * torch.ones(batch)
    val = torch.ones(batch)
    on[:, 0:6] = torch.stack([px, py, vx, vy, r, val], dim=1)

    pa_off, po_off = dists(off)
    pa_on, po_on = dists(on)
    kl_a = _kl(pa_on, pa_off)
    kl_o = _kl(po_on, po_off)
    arg_a = (pa_on.argmax(-1) != pa_off.argmax(-1)).float().mean().item()
    arg_o = (po_on.argmax(-1) != po_off.argmax(-1)).float().mean().item()

    base_near = base_raw.clone()
    base_near[:, 30:43] = 0.05 + 0.05 * torch.rand(batch, 13)
    pa_far, po_far = dists(off, base=base_raw)
    pa_nr, po_nr = dists(off, base=base_near)
    kl_la = _kl(pa_nr, pa_far)
    kl_lo = _kl(po_nr, po_far)

    print("\n=== 因果 ablation (障礙 channel ON vs OFF, preprocess 固定) ===")
    print("  速度頭 KL   障礙=%.5f   參考(LiDAR前錐近)=%.5f   比值=%.2fx" % (kl_a, kl_la, kl_a / max(kl_la, 1e-9)))
    print("  轉向頭 KL   障礙=%.5f   參考(LiDAR前錐近)=%.5f   比值=%.2fx" % (kl_o, kl_lo, kl_o / max(kl_lo, 1e-9)))
    print("  argmax 改變 速度=%.1f%%  轉向=%.1f%%" % (arg_a * 100, arg_o * 100))

    verdict = "使用(速度)" if kl_a > 0.01 else ("弱用(僅轉向)" if kl_o > 0.005 else "未使用")
    print("\n判定: policy 對 LV-DOT channel = 【%s】 (速度 KL 門檻 0.01;baseline 無稅 ~0.0001)" % verdict)
    return {"kl_speed": kl_a, "kl_steer": kl_o, "kl_lidar_speed": kl_la, "kl_lidar_steer": kl_lo, "iteration": it}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LV-DOT channel 因果 ablation 驗收關卡")
    ap.add_argument("--checkpoint", required=True, help="LV-DOT 佈局 checkpoint (.pt)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run_ablation(a.checkpoint, a.batch, a.seed)
