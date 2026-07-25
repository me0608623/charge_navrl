"""CPU parity self-check for multi-expert e2e routing infrastructure.

不佔 GPU、不啟動 sim。目的:證明 play_rnn_car.py 內新增的 multi_expert 前向路徑
(per-expert normalize → per-expert lidar_hist stack → extractor → policy_head → router
gather)與「忠實單專家 e2e 參考前向」在數值上完全一致(atol=1e-5),從而排除
normalizer / stack / 載入邏輯的 bug。

驗證方式
--------
1. 建立兩個隨機初始化的 e2e 專家(corridor / narrow),各自 extractor + policy_head +
   obs_normalizer 統計(mean/var)。用與 play_rnn_car.py 相同的 checkpoint 結構存成 .pt。
2. 參考路徑:對「corridor 專家」用 7 步 e2e 前向(見 play_rnn_car.py 精確路徑)算 logits_ref。
3. 多專家路徑:用 play_rnn_car.py 的 multi_expert forward 邏輯(此檔以獨立函式複刻,
   與 play 內程式碼逐行對齊)+ router_mode=always_corridor 算 logits_me。
4. assert torch.allclose(logits_ref, logits_me, atol=1e-5)。narrow 專家同理
   (router_mode=always_narrow)。
5. 連跑數步讓 lidar_hist 歷史累積,確認 stack 邏輯跨步一致。

執行:
    python scripts/reinforcement_learning/skrl/play_eval/multi_expert_parity_check.py
成功時 exit 0 並印 "PARITY OK"。
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SKRL_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_SKRL_ROOT, "models"))
sys.path.insert(0, _HERE)

from modular_rnn_models import (  # noqa: E402
    ACT_HIST_DIM,
    LIDAR_CONV_CH,
    STATE_DIM,
    LidarStateExtractor,
    PolicyHead,
)
from multi_expert_router import (  # noqa: E402
    RouterConfig,
    route as router_route,
    EXPERT_CORRIDOR,
    EXPERT_NARROW,
)

# --- 與 play_rnn_car.py 一致的常數 ---
OBS_DIM = 79                       # v3f 部署 obs 維度
POLICY_OBS_INDICES = list(range(0, OBS_DIM))  # e2e 非 wd_exact → 全用 79D
LIDAR_FRAME_STACK = 8
HIST_DIM = (LIDAR_FRAME_STACK - 1) * 72


def build_expert(seed: int, label: str, device: torch.device) -> dict:
    """建立一個隨機初始化的 e2e 專家 bundle(結構同 play_rnn_car.py load_expert)。"""
    torch.manual_seed(seed)
    extractor = LidarStateExtractor(
        legacy=False, include_act_hist=False, frame_stack=LIDAR_FRAME_STACK
    ).to(device)
    extractor.train(False)
    for p in extractor.parameters():
        p.requires_grad_(False)
    policy_head = PolicyHead(input_dim=len(POLICY_OBS_INDICES) + extractor.output_dim).to(device)
    policy_head.train(False)
    for p in policy_head.parameters():
        p.requires_grad_(False)
    # 隨機但穩定的 obs_normalizer 統計(每專家不同,測試隔離)
    mean = torch.randn(OBS_DIM, device=device) * 0.3
    var = torch.rand(OBS_DIM, device=device) * 2.0 + 0.5  # (0.5, 2.5)
    return {
        "label": label,
        "extractor": extractor,
        "policy_head": policy_head,
        "mean": mean,
        "var": var,
        "lidar_hist": torch.zeros(0, HIST_DIM, device=device),  # 後面依 B 重建
        "hist_dim": HIST_DIM,
    }


def save_mock_checkpoint(expert: dict, path: str) -> None:
    """存成與 play_rnn_car.py load_expert 相容的 checkpoint 結構。"""
    ckpt = {
        "extractor": expert["extractor"].state_dict(),
        "policy_head": expert["policy_head"].state_dict(),
        "obs_normalizer": {"mean": expert["mean"], "var": expert["var"]},
        "args": {
            "end_to_end_frame_stack": True,
            "lidar_frame_stack": LIDAR_FRAME_STACK,
            "charge_encoder_mode": "extractor_rnn",
        },
    }
    torch.save(ckpt, path)


def normalize(x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """對齊 play_rnn_car.py normalize():clamp((x-mean)/(sqrt(var)+1e-8), -5, 5)。"""
    return torch.clamp((x - mean) / (var.sqrt() + 1e-8), -5.0, 5.0)


def reference_single_expert_forward(expert: dict, obs_tensor: torch.Tensor,
                                    lh_state: torch.Tensor):
    """忠實複刻 play_rnn_car.py 單專家 e2e 前向 7 步(K8 frame-stack)。

    Args:
        expert: bundle。
        obs_tensor: [B, 79] 原始(未正規化)obs。
        lh_state: [B, HIST_DIM] 該步進入時的 lidar 歷史。
    Returns:
        (logits [B,38], new_lh [B, HIST_DIM])
    """
    # step 2: normalize
    obs_normed = normalize(obs_tensor, expert["mean"], expert["var"])
    # step 3: charge_obs_for_rl = obs_normed[:, POLICY_OBS_INDICES]
    p_obs = obs_normed[:, POLICY_OBS_INDICES]
    # step 4: charge_features_for_rnn(lidar_frame_stack=8)
    #   _ext_in = cat(obs_normed, _lh); _cur = obs_normed[:,6:78];
    #   更新 _lh = cat(_cur, _lh[:,:-72]); features = extractor(_ext_in)
    ext_in = torch.cat([obs_normed, lh_state], dim=-1)
    cur = obs_normed[:, 6:78]
    new_lh = torch.cat([cur, lh_state[:, :-72]], dim=-1) if HIST_DIM > 0 else lh_state
    features = expert["extractor"](ext_in)
    # step 5: rl_in = cat(p_obs, features)
    rl_in = torch.cat([p_obs, features], dim=-1)
    # step 6: logits = policy_head(rl_in)
    logits = expert["policy_head"](rl_in)
    return logits, new_lh


def multi_expert_forward(experts: list, obs_tensor: torch.Tensor,
                         router_mode: str, router_cfg: RouterConfig):
    """複刻 play_rnn_car.py multi_expert forward 覆寫邏輯(逐行對齊)。

    Returns:
        (selected_logits [B,38], choice [B], per_expert_logits list)
    每個 expert 的 lidar_hist 會就地(原地)更新。
    """
    me_logits = []
    for exp in experts:
        exp_normed = torch.clamp(
            (obs_tensor - exp["mean"]) / (exp["var"].sqrt() + 1e-8), -5.0, 5.0
        )
        exp_p_obs = exp_normed[:, POLICY_OBS_INDICES]
        exp_ext_in = torch.cat([exp_normed, exp["lidar_hist"]], dim=-1)
        exp_feat = exp["extractor"](exp_ext_in)
        exp_rl_in = torch.cat([exp_p_obs, exp_feat], dim=-1)
        me_logits.append(exp["policy_head"](exp_rl_in))
        exp_cur_lidar = exp_normed[:, 6:78]
        if exp["hist_dim"] > 0:
            exp["lidar_hist"] = torch.cat(
                [exp_cur_lidar, exp["lidar_hist"][:, :-72]], dim=-1
            )
    router_lidar_m = obs_tensor[:, 6:78]
    choice = router_route(router_lidar_m, mode=router_mode, cfg=router_cfg)
    stacked = torch.stack(me_logits, dim=0)  # [n_expert, B, 38]
    idx = choice.view(1, -1, 1).expand(1, stacked.shape[1], stacked.shape[2])
    sel = stacked.gather(0, idx).squeeze(0)
    return sel, choice, me_logits


def make_obs_batch(B: int, seed: int, device: torch.device) -> torch.Tensor:
    """產生隨機 obs batch。LiDAR 欄(6:78)以公尺尺度(0.2~6m)填,其餘標準常態。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = torch.randn(B, OBS_DIM, generator=g).to(device)
    lidar = torch.rand(B, 72, generator=g).to(device) * 5.8 + 0.2  # 0.2~6.0 m
    obs[:, 6:78] = lidar
    return obs


def main() -> int:
    device = torch.device("cpu")
    B = 6
    n_steps = 5

    corridor = build_expert(seed=101, label="corridor", device=device)
    narrow = build_expert(seed=202, label="narrow", device=device)

    # 存/讀 checkpoint 一次,驗證載入結構(額外覆蓋 load_expert 路徑)
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "corridor.pt")
        npth = os.path.join(td, "narrow.pt")
        save_mock_checkpoint(corridor, cp)
        save_mock_checkpoint(narrow, npth)
        # 讀回,確認 obs_normalizer / args 結構可被 load_expert 讀取
        ck = torch.load(cp, map_location=device, weights_only=False)
        assert ck["args"]["end_to_end_frame_stack"] is True
        assert ck["args"]["lidar_frame_stack"] == LIDAR_FRAME_STACK
        assert ck["obs_normalizer"]["mean"].numel() == OBS_DIM
        print("[parity] checkpoint 結構自檢通過(obs_normalizer + args)")

    router_cfg = RouterConfig()

    # ---- always_corridor parity ----
    # 參考路徑:單獨維護 corridor 的 lidar 歷史
    ref_lh = torch.zeros(B, HIST_DIM, device=device)
    # 多專家路徑:每專家自己的 lidar_hist(初始化為 B×HIST_DIM)
    corridor["lidar_hist"] = torch.zeros(B, HIST_DIM, device=device)
    narrow["lidar_hist"] = torch.zeros(B, HIST_DIM, device=device)
    experts = [corridor, narrow]

    max_abs_diff_c = 0.0
    for t in range(n_steps):
        obs = make_obs_batch(B, seed=1000 + t, device=device)
        logits_ref, ref_lh = reference_single_expert_forward(corridor, obs, ref_lh)
        sel_logits, choice, _ = multi_expert_forward(
            experts, obs, "always_corridor", router_cfg
        )
        assert torch.all(choice == EXPERT_CORRIDOR), "always_corridor 路由應全 0"
        diff = (logits_ref - sel_logits).abs().max().item()
        max_abs_diff_c = max(max_abs_diff_c, diff)
        assert torch.allclose(logits_ref, sel_logits, atol=1e-5), (
            f"always_corridor step {t} logits mismatch, max_abs_diff={diff:.2e}"
        )
    print(f"[parity] always_corridor: {n_steps} 步 logits 全一致 "
          f"(max_abs_diff={max_abs_diff_c:.2e})")

    # ---- always_narrow parity ----
    ref_lh_n = torch.zeros(B, HIST_DIM, device=device)
    corridor["lidar_hist"] = torch.zeros(B, HIST_DIM, device=device)
    narrow["lidar_hist"] = torch.zeros(B, HIST_DIM, device=device)
    experts = [corridor, narrow]
    max_abs_diff_n = 0.0
    for t in range(n_steps):
        obs = make_obs_batch(B, seed=2000 + t, device=device)
        logits_ref, ref_lh_n = reference_single_expert_forward(narrow, obs, ref_lh_n)
        sel_logits, choice, _ = multi_expert_forward(
            experts, obs, "always_narrow", router_cfg
        )
        assert torch.all(choice == EXPERT_NARROW), "always_narrow 路由應全 1"
        diff = (logits_ref - sel_logits).abs().max().item()
        max_abs_diff_n = max(max_abs_diff_n, diff)
        assert torch.allclose(logits_ref, sel_logits, atol=1e-5), (
            f"always_narrow step {t} logits mismatch, max_abs_diff={diff:.2e}"
        )
    print(f"[parity] always_narrow: {n_steps} 步 logits 全一致 "
          f"(max_abs_diff={max_abs_diff_n:.2e})")

    # ---- rule 模式:驗證 per-env gather 正確(混合路由)----
    #   構造一批 obs,一半是「窄縫」(左右近+前方開),一半是「開闊」,
    #   確認 route() 分類 + gather 取到正確專家 logits。
    corridor["lidar_hist"] = torch.zeros(B, HIST_DIM, device=device)
    narrow["lidar_hist"] = torch.zeros(B, HIST_DIM, device=device)
    experts = [corridor, narrow]
    obs = make_obs_batch(B, seed=42, device=device)
    # 前 3 env 造成窄縫:左右錐(±90°)近牆 0.5m,前錐(0°)開闊 4m
    #   bin 對應:0°→bin0 附近;90°→bin18;270°→bin54(5°/bin)
    lidar = torch.full((B, 72), 5.0, device=device)
    # 窄縫 env(0,1,2)
    lidar[0:3, :] = 5.0
    lidar[0:3, 14:22] = 0.5      # 左側(~70~110°)近
    lidar[0:3, 50:58] = 0.5      # 右側(~250~290°)近
    lidar[0:3, 0:7] = 4.0        # 前方開闊
    lidar[0:3, 65:72] = 4.0
    # 開闊 env(3,4,5):四周都遠
    lidar[3:6, :] = 5.0
    obs[:, 6:78] = lidar
    sel_logits, choice, per_logits = multi_expert_forward(experts, obs, "rule", router_cfg)
    print(f"[parity] rule 路由結果: choice={choice.tolist()} "
          f"(前 3 應為 narrow=1, 後 3 應為 corridor=0)")
    assert torch.all(choice[0:3] == EXPERT_NARROW), "窄縫 env 應路由到 narrow"
    assert torch.all(choice[3:6] == EXPERT_CORRIDOR), "開闊 env 應路由到 corridor"
    # gather 正確性:每個 env 的 sel_logits 必須等於其被選專家的 logits
    for b in range(B):
        exp_i = int(choice[b].item())
        assert torch.allclose(sel_logits[b], per_logits[exp_i][b], atol=1e-6), (
            f"env {b} gather 取錯專家 logits"
        )
    print("[parity] rule 模式 per-env gather 正確(窄縫→narrow, 開闊→corridor)")

    print("PARITY OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
