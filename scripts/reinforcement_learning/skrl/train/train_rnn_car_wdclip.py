#!/usr/bin/env python3
"""
train_rnn_car.py — Multi-Agent Modular RNN 訓練 (Isaac Lab 版) v5

保留 Warp Drive modular principle，搭配 IsaacLab 環境適配:
  - Charge (=Spot): ModularRNN-A2CK (vanilla RNN, per-head entropy, loss clamping)
  - Obstacle: FC-PPO (learnable, parameter shared, 取代腳本移動)
  - Alternating training: 2:1 charge:obstacle (Warp Drive train_goal_rate=3)

v5 WD-principle changes:
  - γ = 0.984 constant (WD: 1-(1-0.92)/fps, fps=5)
  - PolicyHead [256,256,256,512], ValueHead [256,256,256,512,512]
  - Vanilla RNN (not GRU), memory_dim=30, module_connect_dim=12
  - vf_coeff=0.025 (WD: spot_vf_loss_coeff)
  - A2C mode (--use_a2c): single epoch, full batch + PPO clipping for stability
  - Per-phase obstacle_speed_rate: 0.8→0.85→1.15
  - LR: rl_head=0.0002, rnn=0.0005

v7 WD-principle RNN training (論文§4.2 / custom_trainer.py):
  - Two-optimizer separation: RL head vs aux module (no shared gradient path)
  - RL optimizer: policy_head + value_head only (lr=0.0002)
  - Aux optimizer: RNN cell (lr=0.0005) + preprocess FC (lr=0, frozen) + extractor (lr=0, frozen)
  - Aux target: WD-style 7D privileged geometry (2 nearest obstacles × body-frame (x,y,d) + timestep)
  - Aux loss: WD module loss — log(clamp(L1, 0.01)) × per-dim weight (see wd_aux_targets.py)
  - Detach boundary: preprocess_rnn output .detach()'d before RL head input

Usage:
  PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_rnn_car.py \\
    --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \\
    --num_envs 4096 --headless --seed 1 --use_a2c \\
    --run_name wd_aligned_v5_s1
"""

# ============================================================================
# 1. CLI + AppLauncher (must be before any Isaac imports)
# ============================================================================

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Charge + Obstacle with Modular RNN (Warp Drive port)")

# --- Isaac Lab 標準 ---
parser.add_argument("--task", type=str, default="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=1)

# --- Charge 訓練超參 ---
# --timesteps
# - 用意：總訓練步數（所有 env 累計的 frame 數）。
# - 正常範圍：50K（快速測試）～ 4M+（正式 run）。
# - 更改影響：越多越有機會收斂，但也更花時間；太少可能剛進入穩定就停了。
parser.add_argument("--timesteps", type=int, default=262_144,
                    help="Total timesteps. 262144 = 1024 updates x 256 rollout")
# --rollout_length
# - 用意：每次 rollout 收集的步數（用於 GAE 計算與 PPO batch）。
# - 正常範圍：64～1024（太短 GAE 估計偏差大、太長記憶體需求高且 policy 老化）。
# - 更改影響：改動會連帶影響 effective batch size = rollout_length × num_envs / mini_batches。
parser.add_argument("--rollout_length", type=int, default=256, help="Steps per rollout")
# --ppo_epochs
# - 用意：每次 rollout 對收集的經驗做幾遍 PPO 更新。
# - 正常範圍：3～10（太大 = 對同一批資料反覆學，容易 overfit/diverge）。
# - 更改影響：增大可提升 sample efficiency 但增加 PPO clip 被觸發風險；與 clip_eps 相互制約。
parser.add_argument("--ppo_epochs", type=int, default=4, help="PPO learning epochs")
# --mini_batches
# - 用意：每個 epoch 把 rollout buffer 分幾個 mini-batch。
# - 正常範圍：4～64（需整除 num_envs × rollout_length）。
# - 更改影響：太多 → 每 batch 太小，梯度估計雜訊大；太少 → 更新次數少。
parser.add_argument("--mini_batches", type=int, default=32, help="Mini-batches per epoch")
# --lr
# - 用意：RL head（policy + value）的學習率。
# - 正常範圍：5e-5～1e-3（此處 2e-4 對齊 WD spot_lr）。
# - 更改影響：太大容易不穩/oscillate（尤其 value head）；太小學得慢。
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Charge RL head LR. WD: spot_lr=0.0002")
# --rnn_lr
# - 用意：RNN 模組的學習率（記憶/狀態更新部分）。
# - 正常範圍：1e-5～1e-3（RNN 常需要比 head 稍大或相近；此處 5e-4 偏積極）。
# - 更改影響：太大容易導致 hidden state 發散、梯度爆；太小則「記憶」學不動，表現像無 RNN。
parser.add_argument("--rnn_lr", type=float, default=5e-4,
                    help="Charge RNN module LR. WD: spot_rnn_model_lr=0.0005")
# --aux_lr / --aux_lr_*
# - 用意：Aux optimizer（輔助分支，例如預測頭/特徵抽取器/前後 FC）學習率。預設 0 表示「凍結」。
# - 正常範圍：凍結=0；小幅解凍實驗=1e-6～1e-4（通常從最末端 head 開始）。
# - 更改影響：
#   - 解凍前端（fc_front/extractor）：會改變 RNN 的輸入分佈，常造成訓練劇烈不穩。
#   - 只解凍末端（predict_head/fc_middle）：風險較低，用於修正「輸出映射」。
parser.add_argument("--aux_lr", type=float, default=0.0,
                    help="(Legacy fallback) Preprocess FC + extractor LR in aux optimizer. "
                         "Overridden by fine-grained --aux_lr_* args when they differ from default. "
                         "WD: spot_preprocess_model_lr=0 (frozen). RNN cell uses --rnn_lr.")
parser.add_argument("--aux_lr_predict_head", type=float, default=0.0,
                    help="Aux optimizer LR for predict_head. WD baseline: 0 (frozen). "
                         "Unfreeze experiment: 1e-4.")
parser.add_argument("--aux_lr_fc_middle", type=float, default=0.0,
                    help="Aux optimizer LR for fc_middle (post-RNN FC). WD baseline: 0 (frozen). "
                         "Unfreeze experiment: 5e-5.")
parser.add_argument("--aux_lr_fc_front", type=float, default=0.0,
                    help="Aux optimizer LR for fc_front (pre-RNN FC). Frozen: changing RNN "
                         "input distribution risks destabilizing RNN learning.")
parser.add_argument("--aux_lr_extractor", type=float, default=0.0,
                    help="Aux optimizer LR for extractor (Conv1d+MLP). Frozen: isolate "
                         "'output mapping' fix from 'input feature' changes.")
# --gamma
# - 用意：折扣因子 γ，越接近 1 越重視長期回報。
# - 正常範圍：0.95～0.999（此處 0.984 = WD 把 fps/時間常數換算後的值）。
# - 更改影響：太小偏短視；太大 value 估計方差增大、訓練更不穩。
parser.add_argument("--gamma", type=float, default=0.984,
                    help="Discount factor. WD: 1-(1-0.92)/fps = 0.984 (fps=5)")
# --gae_lambda
# - 用意：GAE 的 λ（bias-variance tradeoff）。
# - 正常範圍：0.90～0.97（常見 0.95）。
# - 更改影響：越大方差越大但偏差更小；越小更穩但可能學不到長期 credit。
parser.add_argument("--gae_lambda", type=float, default=0.95)
# --clip_eps
# - 用意：PPO clipping 範圍（限制 policy 更新幅度）。
# - 正常範圍：0.1～0.3（常見 0.2）。
# - 更改影響：太大容易不穩（更新過猛）；太小學得慢（被 clip 住）。
parser.add_argument("--clip_eps", type=float, default=0.2)
# --target_kl (2026-07-14 fix#2): KL early-stop。approx_kl > 1.5×target_kl 時停止本 iteration 剩餘 minibatch 更新,
#   防單次/多次更新 policy 跑太遠(a2c 高變異 advantage 下 overshoot)。0=關閉。配 mini_batches>1 才有意義。
parser.add_argument("--target_kl", type=float, default=0.0,
                    help="KL early-stop target. Stop remaining minibatch updates when approx_kl > 1.5*target_kl. "
                         "0=off. Needs mini_batches>1. Recommend 0.015.")
parser.add_argument("--value_clip_eps", type=float, default=0.0,
                    help="PPO clipped value-loss epsilon. 0 disables value clipping; standard baseline uses 0.2.")
# --vf_coeff
# - 用意：value loss 權重（critic 對 joint loss 的貢獻比例）。
# - 正常範圍：0.1～1.0 常見，但此專案用 0.025（偏小）代表偏向 policy 更新、critic 只做輔助。
# - 更改影響：太大 critic 主導（壓制 policy）；太小 critic 學不好，advantage 噪聲大。
parser.add_argument("--vf_coeff", type=float, default=0.025,
                    help="Value loss coefficient. WD: spot_vf_loss_coeff=0.025")
# --normalize_return
# - 用意：每 rollout 對 critic value target 做標準化（mean=0, std=1），抑制 return 原始量級造成的 gradient spike。
# - 正常範圍：布林旗標（預設 False = 不正規化）。
# - 更改影響：啟用後 vf_coeff 不需調整，但 value head 的 bias 初始化需配合（見 --value_init_bias）。
parser.add_argument("--normalize_return", "--normalize_returns", dest="normalize_return",
                    action="store_true", default=False,
                    help="Normalize critic value targets per rollout before value MSE. "
                         "This keeps vf_coeff unchanged but tests whether raw return scale "
                         "is driving critic loss/gradient spikes.")
# --popart (2026-07-14): 修 critic/actor 尺度不一致 bug。
# 問題:normalize_return 把 critic target 正規化成 std≈1,但 GAE 用 critic 原始輸出跟 raw reward
#   (get_goal=40級)算 delta=r+γV−V → critic baseline 小 10-20× = 壞 baseline(見 finding_critic_actor_scale_mismatch)。
# PopArt 正解:critic 輸出留正規化空間(訓練穩定),但用 running(EMA)統計把它**反正規化**再餵 GAE
#   (raw-reward 尺度一致=正確 baseline),value target 用同 running 統計正規化。--popart 開時取代 normalize_return。
parser.add_argument("--popart", action="store_true", default=False,
                    help="PopArt: critic outputs normalized values (stable training) but denormalize "
                         "via running EMA return stats before GAE (raw-reward scale baseline). "
                         "Fixes critic/actor scale mismatch that normalize_return causes. Overrides normalize_return.")
parser.add_argument("--popart_beta", type=float, default=0.99,
                    help="PopArt running-stat EMA decay (slow=stable). 0.99 default.")
# --adv_norm_mode
# - 用意：控制 advantage normalization 方式，影響 actor gradient 量級。
# - mean_only：A = A - mean（SA4 預設，raw std≈11，actor gradient 大）
# - partial：A = (A - mean) / sqrt(std)（折中，ME≈0.5 目標）
# - full：A = (A - mean) / (std + 1e-8)（標準 PPO，std=1，gradient 被壓縮）
parser.add_argument("--adv_norm_mode", type=str, default="mean_only",
                    choices=["mean_only", "partial", "full"],
                    help="Advantage normalization mode. "
                         "mean_only: A-mean (SA4 default). "
                         "partial: A/sqrt(std) (balanced gradient). "
                         "full: (A-mean)/(std+eps) (standard PPO).")
# --value_init_bias
# - 用意：覆寫 value head 最後一層 bias 的初始值（只對 fresh run 有效）。
# - 正常範圍：None（保持模型預設）或 0.0（搭配 --normalize_return，因為目標均值≈0）。
# - 更改影響：若 bias 和 target range 不匹配，早期 value 預測偏差大 → gradient spike。
parser.add_argument("--value_init_bias", type=float, default=None,
                    help="Optional override for the value head final-layer bias on fresh runs. "
                         "Use 0.0 with --normalize_return because normalized value targets "
                         "have mean near zero. None keeps the model default.")
# --ent_coeff / --ent_coeff_linear / --ent_coeff_angular
# - 用意：熵正則（鼓勵探索）。0 表示由 curriculum 自動給 WD 風格的每 head coeff。
# - 正常範圍：0～0.5（過大一直亂試、學不收斂）。
# - 更改影響：增加探索但更難收斂；手動改動會讓不同階段的探索行為不可比。
parser.add_argument("--ent_coeff", type=float, default=0.0,
                    help="Legacy single entropy coeff. 0=use per-head WD coeffs from curriculum")
parser.add_argument("--ent_coeff_linear", type=float, default=0.0,
                    help="Head1 (linear accel) entropy coeff. 0=auto from curriculum. "
                         "WD: 0.10 (Phase 1), 0.30 (Phase 2+)")
parser.add_argument("--ent_coeff_angular", type=float, default=0.0,
                    help="Head2 (angular vel) entropy coeff. 0=auto from curriculum. "
                         "WD: 0.375 (all phases)")
# --max_grad_norm
# - 用意：梯度裁切上限（防止爆梯度，對 RNN 特別重要）。
# - 正常範圍：0.5～5.0（常見 0.5 或 1.0）。
# - 更改影響：太小會「學不動」（梯度都被砍掉）；太大裁切效果不足、訓練可能發散。
parser.add_argument("--max_grad_norm", type=float, default=1.0)
# --aux_grad_clip
# - 用意：專給 aux/RNN 更新的獨立 gradient clip（與 RL 的 max_grad_norm 分開）。
# - 正常範圍：None（共用 max_grad_norm）或 0.3～1.0（壓抑 RNN spike 但不影響 RL 更新）。
# - 更改影響：比 max_grad_norm 小可以讓 RNN 更穩，但過小會壓制 aux 學習效率。
parser.add_argument("--aux_grad_clip", type=float, default=None,
                    help="Optional gradient clip just for aux/RNN updates. None uses --max_grad_norm. "
                         "Use values like 0.5 to suppress RNN spikes without changing RL clipping.")
# --wd_update_clip / --no_wd_update_clip
# - 用意：啟用 WD 風格的 actor/critic 分離 gradient cap（避免 critic spike 透過 merged clip 壓抑 actor）。
# - 正常範圍：布林旗標（預設 False = 不分離 clip）。
# - 更改影響：啟用後 actor/critic 各自 clip，可能改善 actor 學習信號；但增加超參數維度。
parser.add_argument("--wd_update_clip", dest="wd_update_clip", action="store_true",
                    help="Enable WD-style actor/critic separated gradient caps. "
                         "When enabled, actor and critic grads are clipped independently "
                         "to --wd_actor_update_clip / --wd_critic_update_clip, avoiding "
                         "critic spikes suppressing actor updates through merged grad clipping.")
parser.add_argument("--no_wd_update_clip", dest="wd_update_clip", action="store_false",
                    help="Disable WD-style actor/critic gradient scaling while keeping the experiment entrypoint.")
parser.set_defaults(wd_update_clip=True)
# --wd_update_monitor_only
# - 用意：(已棄用) 只 log WD-style metrics 不做實際 clip，等同預設行為。保留用於 backward compat。
parser.add_argument("--wd_update_monitor_only", action="store_true", default=False,
                    help="Deprecated alias for the default behavior: log WD-style update metrics but do not scale gradients.")
# --wd_actor_update_clip
# - 用意：WD 風格 actor gradient cap k（只在 --wd_update_clip 啟用時生效）。
# - 正常範圍：2.0～20.0（WD 常用 8.0）。
# - 更改影響：太小壓制 policy 學習；太大 cap 無效。
parser.add_argument("--wd_actor_update_clip", type=float, default=8.0,
                    help="WD-style policy gradient cap k. Applied directly to actor grad norm.")
# --wd_critic_update_clip
# - 用意：WD 風格 critic gradient cap q（只在 --wd_update_clip 啟用時生效）。
# - 正常範圍：10.0～100.0（WD 常用 30.0）。
# - 更改影響：太小壓制 value 學習；太大 cap 無效。
parser.add_argument("--wd_critic_update_clip", type=float, default=30.0,
                    help="WD-style critic gradient cap q. Applied directly to critic grad norm.")
# --wd_module_entropy_eps
# - 用意：計算 module_entropy = log10(actor_update) - log10(critic_update) 時的數值安全 epsilon。
# - 正常範圍：1e-15～1e-8（只防止 log(0)，不影響結果）。
# - 更改影響：通常不需要動。
parser.add_argument("--wd_module_entropy_eps", type=float, default=1e-12,
                    help="Numerical epsilon for module_entropy = log10(actor_update) - log10(critic_update).")
# --policy_loss_clamp
# - 用意：WD A2CK 動態 loss 夾緊（policy loss 項）。
# - 當 |policy_loss| 超過此閾值，等比例縮放以保留梯度方向但限制量級。
# - 正常範圍：5.0～50.0（WD 原版 20.0）。
parser.add_argument("--policy_loss_clamp", type=float, default=20.0,
                    help="WD A2CK: clamp policy loss magnitude. Scale down if |pl| > threshold.")
# --vf_term_clamp
# - 用意：WD A2CK 動態 loss 夾緊（vf_coeff × vf_loss 項）。
# - 當 |vf_coeff × vf_loss| 超過此閾值，動態縮放 vf_loss。
# - 正常範圍：3.0～30.0。注意：與 vf_coeff 相乘後判定。
# - vf_coeff=0.5 + vf_loss≈10 → vf_term≈5.0。設 8.0 可在 spike 時觸發。
parser.add_argument("--vf_term_clamp", type=float, default=8.0,
                    help="WD A2CK: clamp vf_coeff*vf_loss magnitude. Scale down if |vf_term| > threshold.")
# --use_a2c / --use_ppo
# - 用意：選擇 RL 演算法。A2C=WD 原版（無 PPO clip），PPO=標準 clip surrogate。
# - 正常範圍：布林旗標（預設 True = A2C，對齊 WD）。
# - 更改影響：PPO 有 clip 保護更穩但與 WD 不可直接比較；A2C 更新更直接但需 LR 搭配。
parser.add_argument("--use_a2c", "--a2c", action="store_true", default=True,
                    help="Use A2C (single epoch, full batch) + PPO clipping. DEFAULT.")
parser.add_argument("--use_ppo", "--ppo", dest="use_a2c", action="store_false",
                    help="Use PPO (multi-epoch mini-batch) with clipping.")
# --tbptt_len
# - 用意：對「主 RL 訓練」使用 truncated BPTT 的序列長度（>0 才啟用）。
# - 正常範圍：0（逐步更新）或 8～64（WD 常用 15）。
# - 更改影響：啟用可學更長依賴，但顯存增加且梯度可能不穩；維持 0 最省資源但只學短期。
parser.add_argument("--tbptt_len", type=int, default=0,
                    help="TBPTT sequence length. 0=disabled (step-by-step). WD: 15")
# --lr_decay
# - 用意：線性學習率衰減係數（每 iteration 逐步降低 LR）。
# - 正常範圍：0（不衰減）或 0.1～1.0（依實作定義）。
# - 更改影響：適度衰減可讓後期更穩；衰減太快則早早學停。
parser.add_argument("--lr_decay", type=float, default=0.0,
                    help="Linear LR decay factor per iteration. 0=no decay. "
                         "WD: uses ParamScheduler")
# --- N1 直穿模仿（07-27 裁決）---
# --narrow_imitation_weight (λ)
# - 用意：只在窄縫 replay 幀，對 scripted 直穿 teacher 的動作加 CE 模仿損失。
#   total = PPO loss + λ × [CE(linear) + CE(angular)]
# - 正常範圍：0（關閉）；>0 由 shadow rollout 的梯度比校準（目標 ≈ PPO actor 梯度 50%）
# - 更改影響：λ 太大會壓過 PPO 而在非窄縫能力上退步；太小則直穿學不起來。
# - 注意：teacher 只貼標籤、絕不代開車（無 rollout override），PPO 資料保持乾淨。
parser.add_argument("--narrow_imitation_weight", type=float, default=0.0,
                    help="λ for scripted direct-crossing CE on narrow-replay "
                         "frames only. 0=off. Calibrate from shadow rollout.")
# --narrow_imitation_shadow
# - 用意：只量測不學習——照常算 teacher 標籤與 CE、記錄兩邊梯度範數，但 λ 視為 0。
# - 用途：N1 步驟 3/4 的 λ 校準（不動參數地量 CE 梯度 vs PPO actor 梯度）。
parser.add_argument("--narrow_imitation_shadow", action="store_true", default=False,
                    help="Measure scripted-teacher CE and its gradient norm "
                         "against the PPO actor gradient without applying it.")

# --- Modular RNN ---
# --charge_encoder_mode
# - 用意：決定 Charge 分支的「觀測→特徵→RNN」路徑架構。
# - 正常範圍：固定選項
#   - extractor_rnn：先用 Conv1d+MLP 做特徵抽取（更強表示力、但更多參數/更難穩）
#   - raw_fc_rnn：更接近 WD 原始設計（更簡單、通常更穩，特徵更直接）
#   - wd_exact_rnn：完全對齊 WD 架構（113D obs, FC64, 雙層 fc_middle）
# - 更改影響：網路結構與特徵分佈會變，舊 checkpoint 不相容；也會改變學習難度與收斂速度。
parser.add_argument("--charge_encoder_mode", type=str, default="extractor_rnn",
                    choices=["extractor_rnn", "raw_fc_rnn", "wd_exact_rnn"],
                    help="Charge encoder mode. "
                         "extractor_rnn: 79D → Conv1d+MLP extractor(96D) → FC → RNN → FC → 12D (legacy). "
                         "raw_fc_rnn: 79D → FC → RNN → FC → 12D (legacy simplified WD-style). "
                         "wd_exact_rnn: 113D WD-like obs projection → FC64 → RNN30 → concat → FC32 → FC12.")
# --hidden_dim
# - 用意：RNN hidden state 維度（記憶容量）。
# - 正常範圍：16～256（太小記不住、太大容易過擬合/不穩）。
# - 更改影響：變大表達力↑但更難訓練；變小更穩但記憶上限受限。wd_exact_rnn 會強制 30。
parser.add_argument("--hidden_dim", type=int, default=30,
                    help="RNN hidden state dim. WD: memory_dim=30")
# --preprocess_dim
# - 用意：模組間連接的 bottleneck 特徵維度（送給 RL head 的壓縮表徵）。
# - 正常範圍：8～64（此處 12 是 WD 風格）。
# - 更改影響：變小資訊被壓縮（可能學不到關鍵訊號）；變大參數量與學習難度上升。wd_exact_rnn 強制 12。
parser.add_argument("--preprocess_dim", type=int, default=12,
                    help="Preprocess feature dim. WD: module_connect_dim=12")
# --fc_dim
# - 用意：RNN 前的 FC 中間層寬度（把輸入投影到適合 RNN 的表徵）。
# - 正常範圍：32～256（依觀測維度與任務）。
# - 更改影響：變大表達力↑但更慢/更易過擬合；變小可能學不動。wd_exact_rnn 強制 64。
parser.add_argument("--fc_dim", type=int, default=48, help="FC front dim before RNN. wd_exact_rnn forces 64.")
# --wd_middle_dim
# - 用意：wd_exact_rnn 專用，fc_middle 的中間層維度（雙層 FC 的瓶頸）。
# - 正常範圍：16～64（WD 用 32）。只在 wd_exact_rnn mode 生效。
# - 更改影響：變大增加 middle FC capacity；變小則資訊壓縮更強。非 wd_exact 模式不使用此參數。
parser.add_argument("--wd_middle_dim", type=int, default=32,
                    help="WD exact middle FC dim. Used only by wd_exact_rnn: concat(30+64)=94 → FC32 → FC12.")
# --rnn_type
# - 用意：RNN cell 類型（vanilla RNN vs GRU）。
# - 正常範圍：固定選項（RNN 或 GRU）。
# - 更改影響：GRU 通常更穩/記長期，但參數更多；切換後 checkpoint 不相容。wd_exact_rnn 強制 RNN。
parser.add_argument("--rnn_type", type=str, default="RNN", choices=["RNN", "GRU"],
                    help="RNN type. wd_exact_rnn forces vanilla RNN to match WD.")
# --predict_dim
parser.add_argument("--predict_dim", type=int, default=7,
                    help="Aux predict head output dim. 7=WD original, 13=+velocity (top-3 body-frame vx/vy).")
# --aux_velocity_topk
parser.add_argument("--aux_velocity_topk", type=int, default=0,
                    help="Append body-frame velocity of nearest K dynamic obstacles to aux target. 0=off, 3=13D total.")
# --hybrid_predict_to_policy (RNN 當顯式 MOT:把 predict_head 輸出的障礙動態預測接進 policy 輸入)
parser.add_argument("--hybrid_predict_to_policy", action="store_true", default=False,
                    help="Hybrid: feed predict_head output (障礙動態預測) into the policy/value input "
                         "(rl_input += predict_dim). RNN 成為顯式 MOT,其輸出餵 RL,而非只當訓練鷹架. "
                         "predict_head 仍由 aux 監督(位移 label);policy 每步吃其輸出. 預設關=現有隱式架構.")
# --oracle_obstacles_to_policy (oracle 上限測試:把【真值】障礙特權狀態 50D 餵進 policy)
# 診斷「動態碰撞卡 20% = 感知問題 vs 策略問題」。PolicyHead 殘差 priv_branch(init 0)→
# 暖啟動時 policy 與 baseline 相同,可 partial-load 舊 checkpoint。若動態碰撞大降=策略會用運動
# 資訊、缺的只是感知;若仍 ~20%=策略/reward 問題。非部署配置(真值僅 sim 有),純診斷用。
parser.add_argument("--oracle_obstacles_to_policy", action="store_true", default=False,
                    help="Oracle diagnostic: feed TRUE privileged obstacle state (50D, from sim) into "
                         "the policy via a zero-init residual branch. Tests whether the policy CAN use "
                         "obstacle motion (perception gap) vs cannot (policy gap). Not deployable.")
# --lidar_frame_stack (多幀 LiDAR:給 RNN 做 MOT 必需的連續幀;obs 尾端附 (K-1)×72 幀歷史)
parser.add_argument("--lidar_frame_stack", type=int, default=1,
                    help="LiDAR frame stacking K. K=1=現狀(單幀). K>1: extractor Conv1d 吃 K 幀,"
                         "rollout 維護滾動歷史 buffer 並把前 (K-1) 幀附在 obs 尾端,讓 Conv1d 在原始 LiDAR "
                         "層算跨幀運動(類光流). ⚠ 改 extractor 形狀→與舊 checkpoint 不相容,從頭訓.")
parser.add_argument("--end_to_end_frame_stack", action="store_true", default=False,
                    help="Feed frame-stacked LiDAR features directly to policy/value and recompute them "
                         "inside PPO minibatches so RL gradients update the encoder.")
# --penalty_speed_near_obs (reactive:clearance-gated 減速懲罰;覆寫 curriculum,可在任何 stage 套用)
parser.add_argument("--penalty_speed_near_obs", type=float, default=-1.0,
                    help="Reactive clearance-gated speed penalty weight w. <0=用 curriculum 值(預設,不影響). "
                         ">=0=CLI 覆寫(SA4-reactive 用 0.8):近障礙[d_stop 0.45m,d_react 1.2m]區內罰 -(w/fps)·p²·v_fwd, "
                         "p=clip((d_react-d)/(d_react-d_stop),0,1). 對症動態障礙晚反應,不靠 RNN 預測.")
# --penalty_hit (碰撞懲罰 CLI 覆寫; 0=用 curriculum 值(-5), !=0 時壓過, 例 -15 讓膨脹圈咬得痛)
parser.add_argument("--penalty_hit", type=float, default=0.0,
                    help="Collision penalty override. 0=curriculum (-5). e.g. -15: 讓 obs_collision_base 膨脹圈"
                         "有足夠期望損失強迫繞行 (診斷: -5×32%%碰撞率≈-1.6 與繞路成本同級, policy 吃罰不繞).")
# --near_obs_teardrop (水滴形速度稅: per-bin p(d)²·max(cosθ,0)·v_fwd 取max; 前伸側窄+速度門控)
parser.add_argument("--near_obs_teardrop", action="store_true", default=False,
                    help="水滴稅(07-06): react 減速罰改用 per-bin p²·max(cosθ,0) gate(前方伸遠側向收窄=水滴形) "
                         "取代全向 min 標量。並排障礙可過+正對提早反應+慢下免罰(不凍結)。配 --penalty_speed_near_obs>0。")
# --near_obs_d_react (水滴/react 反應區外緣, m; 表面距離, obs=/20 已修單位)
parser.add_argument("--near_obs_d_react", type=float, default=2.0,
                    help="減速反應區外緣(m表面距). 水滴稅預設 2.0。要 2m 實質咬建議 d_react=3.0(讓2m落區內)。d_stop 0.45。")
# --near_obs_penalty_shape (距離因子形狀: sq後載/linear/log前載最強)
parser.add_argument("--near_obs_penalty_shape", type=str, default="sq", choices=["sq", "linear", "log"],
                    help="水滴稅距離因子: sq=p²(後載,~1m才咬,舊) / linear=p / log=log(dr/d)(前載最強,2m 就實質罰,越近成長最陡)。")
# --teardrop_min_stage (只在 curriculum stage >= N 施稅; 用戶建議, 完整課程血緣用: stage1-2 純導航 stage3+ 才施稅)
parser.add_argument("--teardrop_min_stage", type=int, default=0,
                    help="水滴稅只在 curriculum stage >= 此值時施加(此前 gate=0)。0=不用 stage 閘。完整課程建議 3(障礙變多後才教躲)。")
# --teardrop_warmup_start/end (稅權重線性 ramp 的 iteration 區間; 防 from-scratch 早期凍結陷阱)
parser.add_argument("--teardrop_warmup_start", type=int, default=0,
                    help="水滴稅 warmup 起始 iteration (此前 w=0)。0=無 warmup。防政策先學凍結。")
parser.add_argument("--teardrop_warmup_end", type=int, default=0,
                    help="水滴稅 warmup 結束 iteration (此後全額)。建議 from-scratch: start~150/end~350 (SR 建立後才施稅)。")
# --- r_arc: action-conditioned swept-arc「預測軌跡安全度」reward (2026-07-14 用戶提案) ---
#   拿 policy 看到的 72-beam LiDAR + 實際 (v,ω) 預測未來弧軌，算最小間距 c_arc，
#   penalty=-w·[max(0,c_safe-c_arc)/c_safe]²。給「直走撞/左轉撞/右轉清」的方向性梯度(教往哪邊避)。
#   與 teardrop/global 距離稅不同：非距離稅背景稅，只在「此動作會撞」時扣。修正單位後 audit PASS(方向性 46.3%/全撞 0.8%)。
parser.add_argument("--use_arc_reward", action="store_true", default=False,
                    help="啟用 swept-arc r_arc（action-conditioned 預測碰撞成本）。預設關=對現有 run 零影響。")
parser.add_argument("--arc_w", type=float, default=0.08, help="r_arc 權重（危險一步量級 -0.05~-0.10）")
parser.add_argument("--arc_c_safe", type=float, default=0.5, help="車體邊緣安全間距門檻(m)，c_arc<此才開始扣")
parser.add_argument("--arc_cap", type=float, default=0.10, help="r_arc 單步 penalty 上限(絕對值)")
parser.add_argument("--arc_horizon", type=float, default=2.7, help="軌跡預測秒數(v≈0.75→2m/0.75≈2.67s)")
parser.add_argument("--arc_body_radius", type=float, default=0.35, help="車體半徑(m)")
# --ttc_tax_weight (★LV-DOT 密集場景: TTC 門檻稅, 從 channel obs[79:109] 算 per-obstacle
#   碰撞剩餘時間 TTC=d/v_closing, <門檻扣分。closing speed 只有 channel 有→強逼 policy 用速度預判。
#   需 CHARGE_USE_LVDOT_OBS=1 (obs>=109D); 0=off。penalty = w · gate · v_fwd (gate=max slot 危險度)。)
parser.add_argument("--ttc_tax_weight", type=float, default=0.0,
                    help="TTC 門檻稅權重 (LV-DOT 密集). >0: 從 channel 算 per-obstacle TTC=d/v_closing, "
                         "<ttc_thresh 秒扣分 gate=(1-TTC/thresh),逐障礙取 max。penalty=w·gate·v_fwd。"
                         "稅從 channel 速度算→policy 非用 channel 不可。需 CHARGE_USE_LVDOT_OBS=1。0=off。")
parser.add_argument("--ttc_thresh", type=float, default=2.0,
                    help="TTC 門檻(秒)。TTC<此值才扣分,越接近碰撞 gate 越大。預設 2.0s。")
parser.add_argument("--ttc_warmup_start", type=int, default=0,
                    help="TTC 稅 warmup 起始 iteration(此前 w=0)。0=無。防 from-scratch 早期凍結。")
parser.add_argument("--ttc_warmup_end", type=int, default=0,
                    help="TTC 稅 warmup 結束 iteration(此後全額)。建議 from-scratch start~150/end~350。")
# --gap_heading_weight (reactive 轉彎閃避:近障礙時獎勵 heading 朝最大可通行間隙,往側邊空隙轉)
parser.add_argument("--gap_heading_weight", type=float, default=0.0,
                    help="Gap-heading reward weight (轉彎閃避 head-on). 0=off. >0: 障礙近(d_safe<2m)時找最大可通行"
                         "弧段(>0.9m)中心角,獎勵 +(w·dt)·cos(gap_angle) → heading 朝 gap=往側邊空隙轉(reactive on "
                         "LiDAR,bin36=前,不靠速度預測). 配 --penalty_speed_near_obs 成完整 head-on dodge.")
# ★ Term 1: Predictive Early Deceleration + Receding Penalty (07-11 逼 policy 用 LV-DOT 速度)
#   核心對稱設計:同一距離下,障礙「逼近」vs「遠離」看 LiDAR 一樣,只有 channel 速度 vx,vy 能區分。
#   (a) 逼近中(v_closing>0.1)且預測會很近 → 減速給正 reward;
#   (b) 附近有遠離/靜止障礙(v_closing<0.05)但無逼近威脅時 → 減速扣分(沒必要卻減速)。
#   → 要拿滿 reward 必須「只在該減速時減速」= 非讀 channel 速度不可,堵住「LiDAR 一律減速」逃生口。
parser.add_argument("--predictive_decel_weight", type=float, default=0.0,
                    help="Term1 (a) 預測性減速獎勵權重. >0: 障礙逼近(channel v_closing>0.1)且預測2s後距離<1.5m 時,"
                         "若本步在減速→ +w·risk. risk=(1.5-pred_d)clamp. 需 CHARGE_USE_LVDOT_OBS=1(obs>=109D). 0=off.")
parser.add_argument("--receding_penalty_weight", type=float, default=0.0,
                    help="Term1 (b) 遠離時過度減速懲罰(escape-hatch fix). >0: 附近(d<2.5m)有遠離/靜止障礙且"
                         "『無任何逼近威脅』時仍減速 → -w·decel. 逼 policy 用速度區分該不該減速,而非一律看距離減速. 0=off.")
parser.add_argument("--predictive_pred_horizon", type=float, default=2.0,
                    help="Term1 預測前瞻時間(秒),pred_d = d - v_closing·horizon. 預設 2.0s.")
# ★ Term A: Speed-aware Turning Reward (07-11 Option A,ablation證policy靠轉向非減速→獎勵打steering維度).
#   A1 方向正確性 + A2 提早轉向 + 弱化decel(最後手段). 逼 policy 用速度做「提早往對的方向轉」而非走走停停.
parser.add_argument("--turning_direction_weight", type=float, default=0.0,
                    help="Term A1 轉向方向正確性. >0: 障礙逼近(v_closing>0.08)時,若 omega 符號=cross(px·vy−py·vx)"
                         "建議方向 → +w·strength(strength=v_closing/1.5 clamp). max over K=5. 需 obs>=109D. 0=off.")
parser.add_argument("--early_turning_weight", type=float, default=0.0,
                    help="Term A2 提早轉向時機. >0: 同方向且 TTC∈[lo,hi] → +w·time_factor((hi−TTC)/(hi−lo),越早越多). 0=off.")
parser.add_argument("--early_turning_ttc_lo", type=float, default=1.6, help="Term A2 TTC 下限(秒). 預設 1.6.")
parser.add_argument("--early_turning_ttc_hi", type=float, default=3.8, help="Term A2 TTC 上限(秒). 預設 3.8.")
parser.add_argument("--weakened_decel_weight", type=float, default=0.0,
                    help="弱化版 predictive decel(最後手段). >0: 僅極危 TTC<1.2 且 v_closing>0.35 且本步減速 → +w·(1.2−TTC). "
                         "避免走走停停,只在來不及轉向時才獎勵減速. 建議 0.03~0.05. 0=off.")
# --- LV-DOT channel encoder (2026-07-12): raw 30D → learned encoder,測速度使用性能否提升 ---
parser.add_argument("--use_lvdot_encoder", action="store_true",
                    help="LV-DOT 30D channel 不再 raw concat,先過小型 learned encoder(2層MLP+LayerNorm)→ "
                         "encoded 餵 policy/value head。需 CHARGE_USE_LVDOT_OBS=1(obs≥109D)。預設 off=baseline raw concat。")
parser.add_argument("--lvdot_encoder_dim", type=int, default=24,
                    help="LV-DOT encoder 輸出維度(取代 raw 30D)。預設 24(資訊保留 vs 穩定性平衡點)。")
parser.add_argument("--lvdot_encoder_hidden", type=int, default=48,
                    help="LV-DOT encoder 隱藏層維度。預設 48。")
parser.add_argument("--lvdot_encoder_wd", type=float, default=2e-5,
                    help="LV-DOT encoder 權重衰減(正則化)。預設 2e-5。")
# --aux_loss_type
parser.add_argument("--aux_loss_type", type=str, default="log", choices=["log", "huber"],
                    help="Aux loss form for dims 0-5. log=WD original (grad ∝ 1/|e|, 鼓勵常數陷阱); "
                         "huber=smooth-L1 (大誤差大梯度, 修正常數陷阱).")
parser.add_argument("--aux_huber_delta", type=float, default=1.0,
                    help="Huber 轉折點 δ（target 量級 ~1）。")
parser.add_argument("--aux_reinit_frozen", action="store_true", default=False,
                    help="載入 checkpoint 時跳過 fc_middle/fc_front/predict_head(保持隨機初始化),配合凍結=WD 隨機 readout")
parser.add_argument("--aux_skip_input", action="store_true", default=False,
                    help="predict_head 直接 concat extractor 輸入(繞過 RNN 洗位置;extractor 已保留障礙位置74%)。")

# --- Obstacle Policy ---
# --obs_lr
# - 用意：Obstacle policy 的學習率（控制動態障礙物行為的分支）。
# - 正常範圍：1e-5～1e-3。
# - 更改影響：太大障礙物行為不穩/策略震盪；太小障礙物幾乎不學新動作。
parser.add_argument("--obs_lr", type=float, default=3e-4, help="Obstacle policy LR")
# --obs_ent_coeff
# - 用意：障礙物 policy 的熵正則（鼓勵障礙物多元行為）。
# - 正常範圍：0.01～0.3（太大障礙物亂動、太小障礙物行為固定）。
# - 更改影響：增大讓障礙物更隨機（增加 charge 訓練難度多樣性）。
parser.add_argument("--obs_ent_coeff", type=float, default=0.1,
                    help="Obstacle entropy coeff (high = explore more)")
# --obs_speed_limit
# - 用意：障礙物最大移動速度（m/s）。
# - 正常範圍：0.3～1.5（charge 的 v_max=1.0，太快超越主體會不公平）。
# - 更改影響：越快障礙物越難避；越慢接近靜態。
parser.add_argument("--obs_speed_limit", type=float, default=0.8,
                    help="Obstacle max speed (m/s), relative to charge max 1.0")
# --train_goal_rate
# - 用意：每 N 次 iteration 中，1 次訓練障礙物（其餘都訓練 charge）。WD 風格交替訓練。
# - 正常範圍：2～10（N 越大 charge 獲得更多訓練機會，障礙物進步慢）。
# - 更改影響：太小障礙物學太快會壓制 charge 學習；太大障礙物幾乎不學。
parser.add_argument("--train_goal_rate", type=int, default=3,
                    help="Warp Drive: every N iters, 1 trains obstacle (rest train charge)")
# --obs_reward_mode
# - 用意：障礙物的 reward 設計（zero = 不給 reward，讓障礙物自由移動；approach = 弱對抗）。
# - 正常範圍：固定選項。
# - 更改影響：approach 模式下障礙物會主動靠近 charge，增加避障難度。
parser.add_argument("--obs_reward_mode", type=str, default="zero",
                    choices=["zero", "approach", "intercept"],
                    help="zero=Warp Drive style, approach=weak adversarial, intercept=path prediction")
# --obstacle_mode
# - 用意：障礙物控制模式。
#   learned: 用 learned FC policy 控制 (原有行為, 需要 PPO 訓練 obstacle)
#   rule_based: 用 BehaviorScheduler 控制 (deterministic, 無 neural network)
#   scripted: 用原有 move_obstacles_vectorized scripted 邏輯
# - 更改影響：rule_based 時完全跳過 obstacle policy 初始化/訓練，只跑 charge policy + RNN。
parser.add_argument("--obstacle_mode", type=str, default="learned",
                    choices=["learned", "rule_based", "scripted"],
                    help="learned=FC policy, rule_based=BehaviorScheduler, scripted=goal-directed walk")
# --max_active_obstacles
# - 用意：場景中同時活動的最大障礙物數量。
# - 正常範圍：1～20（越多越擁擠/越難，也越耗 GPU）。
# - 更改影響：影響觀測 obs60 的填充量與碰撞機率；curriculum 也會依階段控制實際數量。
parser.add_argument("--max_active_obstacles", type=int, default=10)

# --- Randomization (Warp Drive: obs_size_rand + floor_width_bias) ---
# --obs_size_rand
# - 用意：障礙物碰撞半徑的隨機化範圍（m）。0 表示由 curriculum 自動控制。
# - 正常範圍：0～0.5（WD 從 Phase 1 的 0.1 遞增到 Phase 8 的 0.4）。
# - 更改影響：增大讓 policy 學習適應不同大小障礙；太大可能超出 LiDAR 偵測合理性。
parser.add_argument("--obs_size_rand", type=float, default=0.0,
                    help="Obstacle collision radius randomization range (m). "
                         "0=auto from curriculum. WD: 0.1→0.4 across phases")
# --obs_collision_base
# - 用意：障礙物碰撞距離的基準值（center-to-center 判定門檻）。
# - 正常範圍：0.5～1.5（此處 0.9 = charge_radius + obs_radius + margin）。
# - 更改影響：太小碰撞判定寬鬆（穿越）；太大 policy 被頻繁終止而學不到。
parser.add_argument("--obs_collision_base", type=float, default=0.9,
                    help="Base collision distance (m). Randomized ± obs_size_rand/2")
# --scene_bound_rand
# - 用意：場景邊界的隨機化範圍（m）。0 由 curriculum 自動。
# - 正常範圍：0～3.0（讓 policy 適應不同大小活動空間）。
# - 更改影響：越大場景尺寸變化越大，policy 需泛化到大小不同的環境。
parser.add_argument("--scene_bound_rand", type=float, default=0.0,
                    help="Scene bound randomization range (m). "
                         "0=auto from curriculum. Effective bound = base ± rand/2")
# --scene_bound_base
# - 用意：障礙物活動範圍的基準半徑（m）。
# - 正常範圍：5.0～10.0（實際空間 = ±base，所以活動區域 = 2×base × 2×base）。
# - 更改影響：太小場景擁擠碰撞率高；太大場景空曠避障壓力小。
parser.add_argument("--scene_bound_base", type=float, default=7.0,
                    help="Base obstacle movement boundary (m)")
# --room_size
# - 用意：覆蓋場景物理邊界（外牆位置）。預設 None = 不覆蓋（使用 env_cfg 的 room_size=10.0 → 20×20m）。
# - 正常範圍：3.0～10.0（實際場景 = ±room_size → 2×room_size × 2×room_size m²）。
# - 更改影響：影響外牆位置 + LiDAR boundary 查詢 + wall/obstacle randomization boundary。
#   也會自動調整 scene_bound_base 和 randomize_walls boundary。
parser.add_argument("--room_size", type=float, default=None,
                    help="Override scene physical boundary (m). None=use env_cfg default (10.0=20x20m). "
                         "E.g. --room_size 5 → 10x10m scene.")
# --scene_layout
# - 用意：切換場景佈局。arena = 預設 20×20m 方形場景，t_corridor = T 字型走廊。
# - 設為 t_corridor 時自動切換 --task 為 *-Play-TCorridor 並設定 --play。
parser.add_argument("--scene_layout", type=str, default="arena",
                    choices=["arena", "t_corridor"],
                    help="Scene layout: arena (default 20x20) or t_corridor (T-shaped corridor)")

# --- Logging ---
# --run_name
# - 用意：WandB run 名稱（用於辨識實驗）。None 則由腳本自動生成。
parser.add_argument("--run_name", type=str, default=None)
# --wandb_notes
# - 用意：WandB run 的筆記說明（顯示在 run overview 頁面）。
parser.add_argument("--wandb_notes", type=str, default=None)
# --log_interval
# - 用意：每 N iteration 記錄一次 metrics 到 WandB/console。
# - 正常範圍：1～50（太小 I/O overhead 大；太大看不到中間趨勢）。
parser.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
# --save_interval
# - 用意：每 N iteration 存一次 checkpoint。
# - 正常範圍：50～500（太頻繁佔磁碟；太稀疏一旦崩潰丟失大量進度）。
parser.add_argument("--save_interval", type=int, default=100, help="Save checkpoint every N iterations")
# --checkpoint
# - 用意：載入先前的 checkpoint 繼續訓練或做 inference。
# - 正常範圍：None（fresh start）或有效 .pt 路徑。
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint path")
# --no_resume_optimizer
# - 用意：載入 checkpoint 時只恢復模型權重，不恢復 optimizer state（momentum/exp_avg）。
# - 正常範圍：布林旗標。
# - 更改影響：跳過 optimizer state 相當於 LR warm-restart，可能短暫不穩但避免繼承陳舊 momentum。
parser.add_argument("--no_resume_optimizer", action="store_true", default=False,
                    help="When loading --checkpoint, load model weights only and skip optimizer states. "
                         "Default is full resume if optimizer states are present.")
parser.add_argument("--resume_optimizer", action="store_true", default=False,
                    help="Force optimizer-state restore for an interrupted same-stage continuation.")
# --action_table_sample_size
# - 用意：每 iteration 記錄到 WandB 的 action table 最大抽樣行數。0=停用。
# - 正常範圍：0～4096（太大 WandB upload 變慢）。
parser.add_argument("--action_table_sample_size", type=int, default=2048,
                    help="Max sampled charge action rows logged as a WandB table per iteration. "
                         "0 disables charge/action_speed_accel_table.")
parser.add_argument(
    "--scene_probe_output",
    type=str,
    default=None,
    help=(
        "Optional .pt path for one rollout of normalized policy inputs and "
        "native/SA5/narrow/corridor labels. Training behavior is unchanged."
    ),
)
parser.add_argument(
    "--scene_probe_stride",
    type=int,
    default=4,
    help="Store every Nth rollout step in --scene_probe_output.",
)

# --- Env config overrides ---
# --reward_mode
# - 用意：選擇 reward shaping 方案（對應 navrl_rewards.py 中不同版本）。
# - 正常範圍：固定字串（"current" = 最新版本）。
# - 更改影響：切換 reward 結構會根本改變 policy gradient signal，不同 mode 結果不可比。
parser.add_argument("--reward_mode", type=str, default="current")
# --curriculum_version
# - 用意：選擇 curriculum phase config 檔案；預設使用 single-agent WD 設計。
# - 搭配 --initial_stage 決定目前從哪個 phase/stage 開始；--fixed_stage 決定是否固定不升降。
# - 正常範圍：固定字串。
# - 更改影響：不同 curriculum 學習路徑完全不同，影響所有下游指標。
parser.add_argument("--curriculum_version", type=str, default="warp_drive_single_agent_v1",
                    help="Curriculum phase config key. Default: warp_drive_single_agent_v1. "
                         "Use --initial_stage to choose the starting phase; --fixed_stage to disable transitions.")
# --lidar_no_noise
# - 用意：關閉 LiDAR 噪聲（消除 distractor 和 Uniform noise 干擾）。
# - 正常範圍：布林旗標（預設 False = 有噪聲）。
# - 更改影響：啟用後 LiDAR 資訊更乾淨，policy 更容易信任 LiDAR，但 sim-to-real gap 可能增大。
parser.add_argument("--lidar_no_noise", action="store_true", default=False)
# --vlp16_noise_mode
# - 用意：VLP-16 實測經驗雜訊 ablation 開關（README §5）。設定後覆蓋 lidar_* 細項參數，
#   以「實測固定值」注入（無 DR）。None = 用 YAML 的細項參數。
# - 選項：ideal(乾淨) / sigma(只 8.67mm σ) / bias(只 per-ring 系統偏差) /
#   dropout(只 19.5% 丟點+mixed-pixel) / full(σ+bias+dropout, 部署/sim2real)。
parser.add_argument("--vlp16_noise_mode", type=str, default=None,
                    choices=["ideal", "sigma", "bias", "dropout", "full", "full_material"],
                    help="VLP-16 empirical-noise ablation preset (overrides fine-grained lidar_* params). "
                         "full_material = full + measured human dropout(d) on dynamic-obstacle rays.")
# --no_domain_randomization
# - 用意：關閉所有 domain randomization（物理參數、摩擦、質量等隨機化）。
# - 正常範圍：布林旗標。
# - 更改影響：關閉後訓練更穩/更快收斂，但泛化能力下降。
parser.add_argument("--no_domain_randomization", action="store_true", default=False)
# --reward_speed_v05
# - 用意：啟用 v0.5 版 speed reward（較早期的 reward 變體）。
# - 正常範圍：布林旗標。
# - 更改影響：改變 speed 相關 reward 的計算方式，與其他 reward_mode 可能衝突。
parser.add_argument("--reward_speed_v05", action="store_true", default=False)
# --play
# - 用意：推理模式（載入 checkpoint 只跑 rollout，不做任何訓練）。
# - 正常範圍：布林旗標。
# - 更改影響：啟用後跳過所有 optimizer step，只做 forward pass + 可視化。
parser.add_argument("--play", action="store_true", default=False,
                    help="Inference only — no PPO training, just run rollout with loaded checkpoint")
# --initial_stage
# - 用意：用 CLI 決定目前調用哪個 phase/stage 參數。
# - 正常範圍：1～該 curriculum 的 stage 數（warp_drive_single_agent_v1 目前 1～5）。
# - 更改影響：跳過前面階段可能讓 policy 面臨太難的環境而學不動；但可節省早期時間。
parser.add_argument("--initial_stage", type=int, default=1,
                    help="Force curriculum to start at this phase/stage. "
                         "Range depends on --curriculum_version (warp_drive_single_agent_v1 currently 1-5).")
# --fixed_stage
# - 用意：固定在 initial_stage 不升降（禁用 curriculum 晉升/降級）。
# - 正常範圍：布林旗標。
# - 更改影響：啟用後環境難度固定，用於 WD 風格的定階段實驗或消融測試。
parser.add_argument("--fixed_stage", action="store_true", default=True,
                    help="Keep curriculum at initial_stage and disable promote/demote transitions. "
                         "Useful for WD-style fixed phase experiments. (default: True)")
# --zero_preprocess_feature_for_rl
# - 用意：消融測試——把送給 RL head 的 12D preprocess feature 替換為全零。
# - 正常範圍：布林旗標。
# - 更改影響：aux path 照常訓練但 RL 看不到 RNN 記憶；用於驗證 policy 是否真正利用 RNN feature。
parser.add_argument("--zero_preprocess_feature_for_rl", action="store_true", default=False,
                    help="Ablation: replace 12D preprocess feature with zeros in rl_in. "
                         "Aux path trains normally. Tests whether policy uses RNN features.")
# --disable_aux_training
# - 用意：凍結 aux/RNN 更新（保留模型結構，但不 step aux optimizer）。
# - 正常範圍：布林旗標。
# - 更改影響：preprocess feature 維持 checkpoint 初始值不再變化；用於 pure-RL resume 實驗防止 RNN drift。
parser.add_argument("--disable_aux_training", action="store_true", default=False,
                    help="Freeze/disable auxiliary RNN/preprocess updates during training. "
                         "RL still uses the checkpointed 12D preprocess features, but "
                         "charge_opt_aux is not stepped. Use for pure-RL resume experiments "
                         "to prevent RNN feature drift.")
# --rnn_rl_grad (un-detach 驗證)
# - 用意:讓 RL(PPO/A2C) loss 的梯度流進 RNN(rnn_cell+fc_front+fc_middle),拿掉 WD 的 detach。
# - 機制:PPO update 的 minibatch 內「重算」preprocess_feat(過 RNN,帶梯度),取代 cached rl_in。
# - 由獨立 optimizer charge_opt_rnn_rl(lr=--rnn_rl_lr)更新,extractor/predict_head 不動。
# - 用法:配 --disable_aux_training 做乾淨隔離(RNN 只吃 RL 梯度),短訓幾十 iter 後 probe RNN hidden→位置 rel err。
parser.add_argument("--rnn_rl_grad", action="store_true", default=False,
                    help="Un-detach validation: let RL loss gradient flow into the RNN "
                         "(rnn_cell+fc_front+fc_middle) via a separate optimizer. Recomputes "
                         "preprocess_feat with grad in the PPO update. Pair with "
                         "--disable_aux_training to isolate the RL gradient's effect on the RNN.")
parser.add_argument("--rnn_rl_lr", type=float, default=5e-4,
                    help="Learning rate for the RNN-via-RL optimizer (--rnn_rl_grad).")
# --aux_target_pos_scale (WD-diff #2:尺度錯配修正)
# - 用意:把 aux 位置 target(原始公尺 std~3)縮到 ~unit std,與正規化 obs 一致,
#   配合 huber 讓誤差進二次梯度區,對抗 constant collapse。0.33 ≈ 1/3(std 3m→1)。
# - probe/live metric scale-invariant,不受影響。
parser.add_argument("--aux_target_pos_scale", type=float, default=1.0,
                    help="Scale factor for aux position target (WD-diff #2 scale-match fix). "
                         "0.33 brings ~3m std targets to ~unit, matching normalized obs.")
# --aux_cpc (CPC contrastive aux — 文獻最 robust 的反常數塌縮)
# - 用意:regression aux 易塌成常數;CPC InfoNCE 讓 RNN hidden 對比「自己的障礙位置 target vs 其他樣本」,
#   常數 hidden 對所有樣本 sim 相同 → 無法對上正樣本 → 高 loss → 逼 RNN 編碼位置(discriminative)。
# - 機制:q=proj_q(rnn_out_t), k=proj_k(pos_target_t);logits=q@k.T/τ;CE(logits, 對角 label)。
# - 訓 fc_front+rnn+extractor+proj heads(獨立 cpc_opt)。可與 regression aux 並存(--aux_cpc_weight)。
parser.add_argument("--aux_cpc", action="store_true", default=False,
                    help="Enable CPC/InfoNCE contrastive aux: force RNN hidden to discriminatively "
                         "encode obstacle position (constant hidden fails contrastive task).")
parser.add_argument("--aux_cpc_dim", type=int, default=64,
                    help="CPC projection dim for q/k heads.")
parser.add_argument("--aux_cpc_temp", type=float, default=0.1,
                    help="CPC InfoNCE temperature τ.")
parser.add_argument("--aux_cpc_lr", type=float, default=5e-4,
                    help="CPC optimizer lr (trains fc_front+rnn+extractor+proj heads).")
parser.add_argument("--aux_cpc_max_samples", type=int, default=2048,
                    help="Max (L*B) samples used per CPC step (subsample for the logits matrix).")
# --reinit_rnn:載入 checkpoint 但 RNN cell 重新隨機初始化(extractor/policy 照載)
# - 用意:隔離「全新 RNN 能否從已訓練好的 extractor(有位置特徵)學會編碼」,避開 from-scratch
#   extractor 沒位置的 confound,也避開續訓 RNN 已陷常數陷阱。
parser.add_argument("--reinit_rnn", action="store_true", default=False,
                    help="Load checkpoint but reinitialize the RNN cell (fresh random). "
                         "Isolates whether a fresh RNN can learn to encode from a trained extractor.")
# --aux_epochs:每 iter 做 N 次 aux 更新(複製離線多 epoch,RL 1 update/iter 梯度步數不足是 RNN 學不會的關鍵)
parser.add_argument("--aux_epochs", type=int, default=1,
                    help="Number of aux (regression) updates per training iteration. "
                         ">1 replicates offline multi-epoch density (RL's 1 update/iter is too few "
                         "gradient steps for the RNN to learn to encode position).")
# --feat_norm:extractor 輸出進 RNN 前做 per-dim running 正規化(離線診斷出的關鍵缺件)
parser.add_argument("--feat_norm", action="store_true", default=False,
                    help="Per-dim running normalization on extractor features before the RNN. "
                         "Offline diagnostic: raw features 78%% vs per-dim normalized 31%% rel_err — "
                         "the missing piece preventing the RNN aux from learning to encode position.")
# --aux_zero_h0:aux 用 h0=0(對標離線 fresh hidden),逼 GRU 從 window 特徵萃取位置
# - 用意:RL aux 用 rollout 存的舊 hidden 當 h0(stale),GRU 可靠 h0 帶位置而非從特徵萃取→hidden 編碼差。
#   離線 h0=0 達 30%,RL stale h0 卡 92%。h0=0 + burn_in warm-up 對標離線。
parser.add_argument("--aux_zero_h0", action="store_true", default=False,
                    help="Use h0=0 in the aux RNN unroll (instead of stale rollout hidden), "
                         "forcing the RNN to extract position from window features (matches offline).")
# --- Aux TBPTT ---
# --aux_seq_len
# - 用意：Aux TBPTT 的序列長度（每次展開多少步計算 loss）。
# - 正常範圍：8～32（WD 常用 15）。
# - 更改影響：越長 RNN 可學更長依賴，但梯度消失/爆炸風險增加，且記憶體需求更大。
parser.add_argument("--aux_seq_len", type=int, default=15,
                    help="TBPTT sequence length for aux training. WD commonly uses 15")
# --aux_burn_in
# - 用意：每段序列開頭跳過前 N 步的 loss（只用於更新 hidden state，不計入梯度）。
# - 正常範圍：0～5（0=不 burn-in）。
# - 更改影響：增加 burn-in 讓 hidden state 更準確再計算 loss，但浪費部分序列。
parser.add_argument("--aux_burn_in", type=int, default=0,
                    help="Burn-in steps at start of each sequence (update hidden, skip loss)")
# --aux_seq_batch_size
# - 用意：每次 aux mini-batch 取幾條序列。
# - 正常範圍：64～512（太小梯度估計雜訊大；太大 VRAM 不夠）。
# - 更改影響：影響 aux 梯度品質和記憶體消耗。
parser.add_argument("--aux_seq_batch_size", type=int, default=256,
                    help="Number of sequences per aux mini-batch")

# --- Modular Profiles (Phase 0: metadata only, does not change behavior) ---
parser.add_argument("--reward_profile", type=str, default="wd_sparse",
                    choices=["wd_sparse", "clean_progress", "navrl_dense_v8", "hybrid_progress", "ttc_risk"],
                    help="Reward module profile. Phase 0: only wd_sparse executes.")
parser.add_argument("--anti_spin_weight", type=float, default=0.0,
                    help="A-only: max per-step penalty for sustained near-hazard high-omega low-progress drift")
parser.add_argument("--anti_spin_hazard_distance", type=float, default=1.5)
parser.add_argument("--anti_spin_omega_threshold", type=float, default=0.8)
parser.add_argument("--anti_spin_progress_threshold", type=float, default=0.02)
parser.add_argument("--anti_spin_grace_steps", type=int, default=5,
                    help="Consecutive candidate steps allowed before A starts penalizing (5 steps=1s)")
parser.add_argument("--anti_spin_ramp_steps", type=int, default=5,
                    help="Steps from zero to full A penalty after grace")
parser.add_argument("--anti_spin_yaw_grace_deg", type=float, default=180.0,
                    help="A-only: same-sign yaw allowed before penalty starts")
parser.add_argument("--anti_spin_yaw_ramp_deg", type=float, default=180.0,
                    help="A-only: additional same-sign yaw from zero to full penalty")
parser.add_argument("--anti_spin_dt", type=float, default=0.2,
                    help="Control timestep used to integrate same-sign yaw")
parser.add_argument("--future_occupancy_weight", type=float, default=0.0,
                    help="Max per-step penalty for action-conditioned dynamic future occupancy risk")
parser.add_argument("--future_occupancy_horizon_s", type=float, default=1.5)
parser.add_argument("--future_occupancy_samples", type=int, default=8)
parser.add_argument("--future_occupancy_safe_distance_m", type=float, default=1.0)
parser.add_argument("--future_occupancy_near_distance_m", type=float, default=3.0)
parser.add_argument("--future_occupancy_move_threshold_mps", type=float, default=0.1)
parser.add_argument("--scene_profile", type=str, default=None,
                    help="Scene/curriculum profile. Default: inferred from --curriculum_version.")
parser.add_argument("--algorithm_profile", type=str, default=None,
                    choices=["a2c_wd", "ppo_clip"],
                    help="Algorithm profile. Default: inferred from --use_a2c / --use_ppo.")
parser.add_argument("--aux_profile", type=str, default=None,
                    choices=["wd_7d_geometry", "none", "future_collision_risk"],
                    help="Aux training profile. Default: inferred from --disable_aux_training.")
parser.add_argument("--encoder_profile", type=str, default=None,
                    help="Encoder profile. Default: inferred from --charge_encoder_mode.")
parser.add_argument("--critic_profile", type=str, default="symmetric",
                    choices=["symmetric", "asymmetric"],
                    help="Critic profile: symmetric (same obs as policy) or asymmetric (+ privileged obs).")
parser.add_argument(
    "--critic_detach_encoder",
    action="store_true",
    default=False,
    help="Do not backpropagate value loss into the shared policy encoder.",
)

# --- Experiment Config (LEGO-style run composition) ---
parser.add_argument("--experiment_config", type=str, default=None,
                    help="Experiment config name (e.g. wd_sa2_a2c_aux_lowent) or .py file path. "
                         "Provides defaults for all training params; CLI flags override.")
parser.add_argument("--print_experiment_config", action="store_true", default=False,
                    help="Print resolved experiment config and exit.")
parser.add_argument(
    "--teacher_retention_checkpoint",
    type=str,
    default=None,
    help="Frozen teacher checkpoint for narrow-passage policy retention.",
)
parser.add_argument(
    "--teacher_retention_weight",
    type=float,
    default=0.0,
    help="Beta for narrow-only KL(teacher || current). Zero disables retention.",
)
parser.add_argument(
    "--teacher_retention_margin_weight",
    type=float,
    default=0.0,
    help="Weight for the narrow-only teacher deterministic-action margin loss.",
)
parser.add_argument(
    "--teacher_retention_action_ce_weight",
    type=float,
    default=0.0,
    help="Weight for narrow-only teacher-argmax cross-entropy.",
)
parser.add_argument(
    "--teacher_retention_argmax_margin",
    type=float,
    default=0.2,
    help="Required student logit lead for the teacher argmax action.",
)
parser.add_argument(
    "--teacher_retention_post_kl_epochs",
    type=int,
    default=0,
    help="Post-PPO narrow replay KL projection epochs; zero disables it.",
)
parser.add_argument(
    "--teacher_retention_post_kl_lr",
    type=float,
    default=1e-3,
    help="Stateless SGD learning rate for post-PPO KL projection.",
)
parser.add_argument(
    "--teacher_retention_post_kl_batch_size",
    type=int,
    default=4096,
    help="Narrow-frame batch size for post-PPO KL projection.",
)
parser.add_argument(
    "--teacher_retention_post_kl_max_grad_norm",
    type=float,
    default=0.5,
    help="Actor gradient clip for post-PPO KL projection.",
)
parser.add_argument(
    "--teacher_retention_post_margin_weight",
    type=float,
    default=0.0,
    help="Teacher-argmax margin weight in the post-PPO narrow projection.",
)
parser.add_argument(
    "--teacher_retention_post_action_ce_weight",
    type=float,
    default=0.0,
    help="Teacher-argmax cross-entropy weight in the post-PPO narrow projection.",
)
parser.add_argument(
    "--teacher_retention_post_policy_head_only",
    action="store_true",
    default=False,
    help="Restrict the post-PPO narrow projection to policy-head parameters.",
)
parser.add_argument(
    "--teacher_retention_post_anchor_weight",
    type=float,
    default=0.0,
    help=(
        "Weight for a second frozen teacher during post-update projection. "
        "Zero preserves single-teacher behavior."
    ),
)
parser.add_argument(
    "--teacher_retention_rollout_override",
    action="store_true",
    default=False,
    help=(
        "Execute the frozen narrow teacher's deterministic action on narrow "
        "replay envs while collecting projection data. Requires ppo_epochs=0."
    ),
)
parser.add_argument(
    "--previous_stage_teacher_checkpoint",
    type=str,
    default=None,
    help="Frozen teacher checkpoint for SA5-general replay retention.",
)
parser.add_argument(
    "--previous_stage_teacher_retention_weight",
    type=float,
    default=0.0,
    help="Beta for KL(SA5 teacher || current) on previous-stage replay only.",
)
parser.add_argument(
    "--previous_stage_teacher_scope",
    choices=("previous_stage", "non_narrow", "corridor", "all"),
    default="previous_stage",
    help="Frame mask used by the second frozen teacher.",
)
parser.add_argument(
    "--corridor_teacher_distill_epochs",
    type=int,
    default=0,
    help="Post-PPO privileged corridor action projection epochs; zero disables it.",
)
parser.add_argument(
    "--corridor_teacher_distill_lr",
    type=float,
    default=5e-4,
    help="Stateless SGD learning rate for corridor teacher projection.",
)
parser.add_argument(
    "--corridor_teacher_distill_batch_size",
    type=int,
    default=4096,
    help="Corridor-frame batch size for privileged teacher projection.",
)
parser.add_argument(
    "--corridor_teacher_distill_max_grad_norm",
    type=float,
    default=0.5,
    help="Policy-head gradient clip for corridor teacher projection.",
)
parser.add_argument(
    "--corridor_teacher_distill_neighbor_mass",
    type=float,
    default=0.20,
    help="Soft-label mass assigned to adjacent teacher action bins.",
)
parser.add_argument(
    "--corridor_teacher_distill_stride",
    type=int,
    default=2,
    help="Generate privileged labels every N rollout steps.",
)
parser.add_argument(
    "--corridor_teacher_distill_chunk_size",
    type=int,
    default=32,
    help="Maximum corridor envs per privileged teacher geometry batch.",
)
parser.add_argument(
    "--corridor_teacher_intervention_only",
    action="store_true",
    default=False,
    help=(
        "Distill only corridor states where the deterministic policy has a "
        "predicted collision/low-clearance trajectory and the teacher differs."
    ),
)
parser.add_argument(
    "--corridor_teacher_intervention_clearance_m",
    type=float,
    default=0.20,
    help="Minimum policy swept-path clearance before intervention distillation.",
)
parser.add_argument(
    "--corridor_adapter_enabled",
    action="store_true",
    default=False,
    help="Enable the deployable observation-gated corridor residual policy.",
)
parser.add_argument(
    "--corridor_adapter_hidden_dim",
    type=int,
    default=64,
    help="Hidden width of the corridor gate and residual branches.",
)
parser.add_argument(
    "--corridor_adapter_gate_loss_weight",
    type=float,
    default=0.05,
    help="Weight of class-balanced corridor gate BCE.",
)
parser.add_argument(
    "--corridor_adapter_gate_init_probability",
    type=float,
    default=0.01,
    help="Initial corridor prior when no pretrained gate is supplied.",
)
parser.add_argument(
    "--corridor_adapter_max_logit_delta",
    type=float,
    default=2.0,
    help="Absolute bound on each residual action logit before gate scaling.",
)
parser.add_argument(
    "--corridor_adapter_freeze_base",
    action="store_true",
    default=False,
    help="Freeze the base policy and CNN so PPO updates only the adapter actor.",
)
parser.add_argument(
    "--corridor_adapter_gate_checkpoint",
    type=str,
    default=None,
    help="Optional pretrained 83D corridor gate state_dict.",
)
parser.add_argument(
    "--corridor_adapter_residual_features",
    type=str,
    choices=("current_obs", "policy_features"),
    default="current_obs",
    help=(
        "Residual branch input: current 83D observation or the full deployable "
        "policy feature vector including the K8 CNN embedding."
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
_original_argv = list(sys.argv)  # Save before hydra strips args (for experiment_config CLI detection)

# Some observation terms are selected at isaaclab_tasks import time. Bootstrap
# only that config value now; the complete config is still applied below.
_bootstrap_experiment_cfg = None
if args_cli.experiment_config is not None:
    _bootstrap_skrl_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_bootstrap_skrl_root))
    from rnn_car_modular.configs.registry import get_experiment_config as _get_bootstrap_config

    _bootstrap_experiment_cfg = _get_bootstrap_config(args_cli.experiment_config)
    if _bootstrap_experiment_cfg.use_action_history is not None:
        os.environ["CHARGE_USE_ACT_HIST"] = "1" if _bootstrap_experiment_cfg.use_action_history else "0"
        print(
            "[EXPERIMENT_CONFIG] pre-import observation layout: "
            f"use_action_history={_bootstrap_experiment_cfg.use_action_history}"
        )

# --scene_layout: 自動切換 task 為對應的場景佈局
if args_cli.scene_layout == "t_corridor":
    args_cli.task = "Isaac-Navigation-Charge-VLP16-Curriculum-NavRL-Play-TCorridor"
    args_cli.play = True
    print(f"[INFO] scene_layout=t_corridor → task={args_cli.task}, play=True")

headless_mode = getattr(args_cli, "headless", False) or "--headless" in sys.argv
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ============================================================================
# 2. Post-launcher imports
# ============================================================================
# 注意：所有 Isaac Sim / Omniverse 相關套件必須在 AppLauncher 啟動後才能 import。
# 在此之前 import 會造成 Omniverse Kit 尚未初始化而崩潰。

import gymnasium as gym          # OpenAI Gym 相容介面 (IsaacLab 環境的 gym wrapper)
import numpy as np               # NumPy：CPU 端數值計算、metrics 統計
import torch                     # PyTorch：GPU 張量運算、模型訓練
import torch.nn as nn            # 神經網路模組 (Linear、RNN、Conv1d…)
import torch.nn.functional as F  # 無狀態 NN 函數 (mse_loss…)
from torch.distributions import Categorical  # 離散動作分佈 (policy logits → 動作採樣)

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # 把 IsaacLab env 包成 skrl 相容介面

import isaaclab_tasks  # noqa: register tasks  ← side-effect import，觸發所有 gym.register
from isaaclab_tasks.utils.hydra import hydra_task_config  # Hydra config decorator

# 將訓練腳本目錄加到 sys.path，使下方的相對 import 能找到同目錄與 parent 的模組
sys.path.insert(0, str(Path(__file__).parent))
_skrl_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_skrl_root))          # skrl/ root
sys.path.insert(0, str(_skrl_root / "models"))  # skrl/models/
sys.path.insert(0, str(_skrl_root / "utils"))   # skrl/utils/
from rnn_car_wdclean import swept_arc  # ★r_arc: action-conditioned swept-arc 碰撞預測 reward（--use_arc_reward 閘）
from rnn_car_wdclean.teacher_retention import (
    apply_masked_deterministic_teacher_actions,
    masked_two_head_retention_loss,
    post_update_dual_teacher_projection,
    post_update_kl_projection,
)
from rnn_car_wdclean.corridor_teacher_distillation import (
    post_update_corridor_action_projection,
    select_corridor_interventions,
)
from rnn_car_wdclean.privileged_corridor_teacher import (
    CorridorTeacherSpec,
    corridor_teacher_action_grid,
    predict_patrol_obstacle_paths,
)
from rnn_car_wdclean.scripted_narrow_teacher import (
    ScriptedNarrowTeacherSpec,
    narrow_bridge_teacher_geometry,
    scripted_narrow_gap_action_indices,
)
from rnn_car_wdclean.narrow_imitation_loss import scripted_action_ce_loss
from rnn_car_wdclean.reward_diagnostics import (
    add_long_corridor_reward_diagnostics,
)

# Charge 側網路模組（從 modular_rnn_models.py 匯入）
from modular_rnn_models import (
    LidarStateExtractor,
    adapt_lidar_frame_stack_state_dict,
    PreprocessRNN,
    PolicyHead,
    CorridorResidualAdapter,
    balanced_binary_gate_loss,
    ValueHead,
    LVDOTEncoder,
    RNNStateManager,
    ObstaclePolicyFC,
    ObstacleValueFC,
    OBS_POLICY_OBS_DIM,
    LIDAR_START,
    LIDAR_END,
    NUM_BINS,
)
from wd_aux_targets import (
    build_wd_preprocess_targets, compute_wd_module_loss,
    WD_DEFAULT_WEIGHT, WD_DEFAULT_WEIGHT_13D,
)
from privileged_obs import extract_privileged_obs, PRIVILEGED_OBS_DIM

from rnn_car_modular.profiles import resolve_profiles, validate_profiles, profiles_to_dict
from rnn_car_modular.experiment_config import (
    ExperimentConfig, apply_experiment_config, experiment_config_to_dict,
)

# --- Apply experiment config (if provided) before profile resolution ---
_experiment_cfg: ExperimentConfig | None = None
_experiment_applied_fields: list[str] = []
if args_cli.experiment_config is not None:
    from rnn_car_modular.configs.registry import get_experiment_config
    _experiment_cfg = _bootstrap_experiment_cfg or get_experiment_config(args_cli.experiment_config)
    _experiment_applied_fields = apply_experiment_config(args_cli, _experiment_cfg, _original_argv)
    if args_cli.resume_optimizer:
        args_cli.no_resume_optimizer = False
    print(f"[EXPERIMENT_CONFIG] name={_experiment_cfg.name} description={_experiment_cfg.description}")
    if _experiment_applied_fields:
        _applied_summary = " ".join(
            f"{k}={getattr(args_cli, k, getattr(_experiment_cfg, k, None))}"
            for k in _experiment_applied_fields[:15]
        )
        print(f"[EXPERIMENT_CONFIG] applied fields ({len(_experiment_applied_fields)}): {_applied_summary}"
              + ("..." if len(_experiment_applied_fields) > 15 else ""))

if args_cli.print_experiment_config:
    if _experiment_cfg is not None:
        import json
        print(json.dumps(experiment_config_to_dict(_experiment_cfg), indent=2, default=str))
    else:
        print("[EXPERIMENT_CONFIG] No --experiment_config provided.")
    simulation_app.close()
    sys.exit(0)

# --- Resolve & validate trainer profiles (Phase 0: metadata only) ---
_trainer_profiles = resolve_profiles(args_cli)
validate_profiles(_trainer_profiles, args_cli)
print(f"[PROFILES] reward={_trainer_profiles.reward_profile} "
      f"scene={_trainer_profiles.scene_profile} "
      f"algorithm={_trainer_profiles.algorithm_profile} "
      f"aux={_trainer_profiles.aux_profile} "
      f"encoder={_trainer_profiles.encoder_profile}")

# --- Create reward module from profile (Phase 2: runtime dispatch) ---
from rnn_car_modular.rewards.factory import create_reward_module
_reward_module = create_reward_module(
    _trainer_profiles.reward_profile,
    anti_spin_weight=args_cli.anti_spin_weight,
    anti_spin_hazard_distance=args_cli.anti_spin_hazard_distance,
    anti_spin_omega_threshold=args_cli.anti_spin_omega_threshold,
    anti_spin_progress_threshold=args_cli.anti_spin_progress_threshold,
    anti_spin_grace_steps=args_cli.anti_spin_grace_steps,
    anti_spin_ramp_steps=args_cli.anti_spin_ramp_steps,
    anti_spin_yaw_grace_deg=args_cli.anti_spin_yaw_grace_deg,
    anti_spin_yaw_ramp_deg=args_cli.anti_spin_yaw_ramp_deg,
    anti_spin_dt=args_cli.anti_spin_dt,
    future_occupancy_weight=args_cli.future_occupancy_weight,
    future_occupancy_horizon_s=args_cli.future_occupancy_horizon_s,
    future_occupancy_samples=args_cli.future_occupancy_samples,
    future_occupancy_safe_distance_m=args_cli.future_occupancy_safe_distance_m,
    future_occupancy_near_distance_m=args_cli.future_occupancy_near_distance_m,
    future_occupancy_move_threshold_mps=args_cli.future_occupancy_move_threshold_mps,
)
print(f"[REWARD] module={_reward_module.name} (runtime dispatch active)")

print("[INFO] Multi-Agent Modular RNN Training v5 — WD-Principle (A2CK + vanilla RNN)")


# ============================================================================
# Monitoring helpers (gradient norm, param delta, param norm)
# ============================================================================
# 這些函數提供 WD-style 訓練診斷，用來監測以下三個維度:
#   1. param_l2_norm  → 參數「量級」(模型是否發散/縮水)
#   2. grad_l2_norm   → 梯度「壓力」(更新是否被 clip 壓制或爆炸)
#   3. param_delta_norm → 實際更新「步長」(優化器一次 step 的位移量)

# WD module loss dim → human-readable name mapping
# wd_aux_targets.py 輸出 7D loss，前 6 維對應最近兩個障礙物的 body-frame 相對座標
_AUX_DIM_NAMES = {
    "module_feture_0_loss": "aux/t0_bx_loss",
    "module_feture_1_loss": "aux/t0_by_loss",
    "module_feture_2_loss": "aux/next_bx_loss",
    "module_feture_3_loss": "aux/next_by_loss",
    "module_feture_4_loss": "aux/hist_bx_loss",
    "module_feture_5_loss": "aux/hist_by_loss",
    "module_feture_6_loss": "aux/hist_dis_loss",
    "module_feture_7_loss": "aux/obs1_vbx_loss",
    "module_feture_8_loss": "aux/obs1_vby_loss",
    "module_feture_9_loss": "aux/obs2_vbx_loss",
    "module_feture_10_loss": "aux/obs2_vby_loss",
    "module_feture_11_loss": "aux/obs3_vbx_loss",
    "module_feture_12_loss": "aux/obs3_vby_loss",
}


def _param_l2_norm(params) -> float:
    """計算參數向量的 L2 norm（監控模型量級用）。"""
    total = 0.0
    for p in params:
        total += p.data.norm(2).item() ** 2
    return total ** 0.5


def _grad_l2_norm(params) -> float:
    """計算梯度向量的 L2 norm（監控梯度壓力用）。無梯度時返回 0。"""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


def _scale_grads(params, scale: float):
    """對一個參數組的梯度做 in-place 縮放（WD-style actor/critic 分離 clip 用）。
    scale >= 1.0 時直接跳過（不放大梯度）。"""
    if scale >= 1.0:
        return
    for p in params:
        if p.grad is not None:
            p.grad.data.mul_(scale)


def _snapshot_params(params) -> torch.Tensor:
    """把所有參數 flatten 並 clone 為單一向量，用來計算 param_delta_norm。"""
    return torch.cat([p.data.reshape(-1).clone() for p in params])


def _param_delta_norm(before: torch.Tensor, after_params) -> float:
    """計算 optimizer.step() 前後的參數位移量 (L2 norm)。
    公式: ||theta_after - theta_before||_2
    用途: 比較 actor/critic 更新幅度，診斷 WD module_entropy。
    """
    after = torch.cat([p.data.reshape(-1) for p in after_params])
    return (after - before).norm(2).item()


# ============================================================================
# Running observation normalizer
# ============================================================================
# 使用 Welford's online algorithm 對觀測值做增量式均值/方差估計，避免儲存所有歷史資料。
# 歸一化後再 clip 到 [-clip, clip]，防止離群值破壞網路輸入。

class RunningNormalizer:
    """Welford's online algorithm for running mean/var normalization.

    每個 rollout step 呼叫 update(obs) 更新統計量，
    呼叫 normalize(obs) 做標準化：(x - mean) / (std + 1e-8)，並 clip 到 ±clip。

    注意：此 normalizer 只對 Charge 的 139D 觀測做標準化。
    Obstacle obs (9D) 目前不另外 normalize，因為已在 env 端計算相對距離。
    """

    def __init__(self, shape, device, clip=5.0):
        self.mean = torch.zeros(shape, device=device)   # 每 dim 的 running mean
        self.var = torch.ones(shape, device=device)     # 每 dim 的 running variance
        self.count = 1e-4  # 初始化為小正值避免除以零（而非 0）
        self.clip = clip   # 標準化後的 clamp 範圍（對應 RunningStandardScaler 的 clip）

    @torch.no_grad()
    def update(self, x):
        """Welford parallel update：用新的 mini-batch x 更新 mean/var。
        x: [N, dim]，N = num_envs（每 step 全部 env 的觀測）
        """
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        # 更新 mean（加權平均）
        self.mean = self.mean + delta * batch_count / total
        # 更新 var（parallel Welford — 合併兩組統計量）
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x):
        """標準化 x 並 clip 到 [-clip, clip]。"""
        return torch.clamp((x - self.mean) / (self.var.sqrt() + 1e-8), -self.clip, self.clip)


# ============================================================================
# Charge Rollout Buffer (PPO + aux loss)
# ============================================================================
# Rollout buffer 同時儲存兩種訓練所需的資料：
#   1. RL 訓練（A2C/PPO）：rl_inputs, actions, log_probs, rewards, values, dones
#   2. Aux/RNN 訓練（TBPTT）：raw_obs, hiddens, aux_targets
#
# 設計重點：
#   - raw_obs 儲存未 normalize 的原始觀測，aux TBPTT 時會即時 normalize（避免 stale stats）
#   - hiddens 儲存每步的 RNN hidden state，TBPTT 用 h0 初始化序列而非 zero-init
#   - ptr 指向下一個寫入位置，每次 rollout 開始時 reset() 歸零

class ChargeRolloutBuffer:
    def __init__(self, num_steps, num_envs, rl_input_dim, obs_dim, hidden_dim, device,
                 privileged_dim: int = 0, predict_dim: int = 7,
                 encoder_input_dim: int = 0, teacher_logits_dim: int = 0,
                 previous_teacher_logits_dim: int = 0,
                 store_corridor_mask: bool = False,
                 store_narrow_teacher: bool = False):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        # RL 訓練所需欄位（形狀：[T, E, *]）
        self.rl_inputs = torch.zeros(num_steps, num_envs, rl_input_dim, device=device)  # [T,E, obs+rnn_feat]
        self.actions = torch.zeros(num_steps, num_envs, 2, dtype=torch.long, device=device)  # [T,E,2] discrete(linear, angular)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)   # [T,E] joint log prob
        self.rewards = torch.zeros(num_steps, num_envs, device=device)     # [T,E] WD sparse reward
        self.values = torch.zeros(num_steps, num_envs, device=device)      # [T,E] critic prediction
        self.dones = torch.zeros(num_steps, num_envs, device=device)       # [T,E] episode end (term|trunc, aux/reset 用)
        self.terminateds = torch.zeros(num_steps, num_envs, device=device) # ★fix#1: 真terminal(撞/到達),GAE bootstrap 用(truncation 不砍)
        # Aux/RNN 訓練所需欄位
        self.raw_obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)       # [T,E,139] 原始觀測
        self.hiddens = torch.zeros(num_steps, num_envs, hidden_dim, device=device)    # [T,E,H] RNN hidden
        self.aux_targets = torch.zeros(num_steps, num_envs, predict_dim, device=device)
        self._encoder_input_dim = encoder_input_dim
        if encoder_input_dim > 0:
            # Exact normalized frame stack seen during rollout. PPO recomputes
            # the encoder from this tensor so policy loss reaches the CNN.
            self.encoder_inputs = torch.zeros(
                num_steps, num_envs, encoder_input_dim, device=device
            )
        self._teacher_logits_dim = teacher_logits_dim
        if teacher_logits_dim > 0:
            self.teacher_logits = torch.zeros(
                num_steps, num_envs, teacher_logits_dim, device=device
            )
            self.retention_mask = torch.zeros(
                num_steps, num_envs, dtype=torch.bool, device=device
            )
        self._previous_teacher_logits_dim = previous_teacher_logits_dim
        if previous_teacher_logits_dim > 0:
            self.previous_teacher_logits = torch.zeros(
                num_steps, num_envs, previous_teacher_logits_dim, device=device
            )
            self.previous_retention_mask = torch.zeros(
                num_steps, num_envs, dtype=torch.bool, device=device
            )
        self._store_corridor_mask = bool(store_corridor_mask)
        if self._store_corridor_mask:
            self.corridor_mask = torch.zeros(
                num_steps, num_envs, dtype=torch.bool, device=device
            )
        # N1: scripted 直穿 teacher 的硬標籤動作（只在窄縫 replay 幀有效）。
        self._store_narrow_teacher = bool(store_narrow_teacher)
        if self._store_narrow_teacher:
            self.narrow_teacher_actions = torch.zeros(
                num_steps, num_envs, 2, dtype=torch.long, device=device
            )
            self.narrow_imitation_mask = torch.zeros(
                num_steps, num_envs, dtype=torch.bool, device=device
            )
        # Asymmetric critic privileged obs
        self._privileged_dim = privileged_dim
        if privileged_dim > 0:
            self.privileged_obs = torch.zeros(num_steps, num_envs, privileged_dim, device=device)
        self.ptr = 0  # 下一個寫入步數的 pointer

    def add(self, rl_input, action, log_prob, reward, value, done, raw_ob, hidden,
            aux_target=None, privileged=None, terminated=None, encoder_input=None,
            teacher_logits=None, retention_mask=None,
            previous_teacher_logits=None, previous_retention_mask=None,
            corridor_mask=None,
            narrow_teacher_actions=None, narrow_imitation_mask=None):
        """儲存一個 rollout step 的所有資料。每次 env.step() 後呼叫。"""
        i = self.ptr
        self.rl_inputs[i] = rl_input
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        # ★fix#1: 真terminal(撞/到達)存 terminateds;未傳則退回=done(舊行為,truncation 仍當 terminal)
        self.terminateds[i] = terminated if terminated is not None else done
        self.raw_obs[i] = raw_ob
        self.hiddens[i] = hidden.squeeze(0)  # 去掉 RNN 的 [1, E, H] 前導維度
        if aux_target is not None:
            self.aux_targets[i] = aux_target
        if privileged is not None and self._privileged_dim > 0:
            self.privileged_obs[i] = privileged
        if encoder_input is not None and self._encoder_input_dim > 0:
            self.encoder_inputs[i] = encoder_input
        if self._teacher_logits_dim > 0:
            if teacher_logits is None or retention_mask is None:
                raise RuntimeError(
                    "teacher retention buffer requires logits and scene mask at every step"
                )
            self.teacher_logits[i] = teacher_logits
            self.retention_mask[i] = retention_mask
        if self._previous_teacher_logits_dim > 0:
            if (
                previous_teacher_logits is None
                or previous_retention_mask is None
            ):
                raise RuntimeError(
                    "previous-stage teacher buffer requires logits and scene "
                    "mask at every step"
                )
            self.previous_teacher_logits[i] = previous_teacher_logits
            self.previous_retention_mask[i] = previous_retention_mask
        if self._store_corridor_mask:
            if corridor_mask is None:
                raise RuntimeError(
                    "corridor adapter buffer requires a scene mask at every step"
                )
            self.corridor_mask[i] = corridor_mask
        if self._store_narrow_teacher:
            if narrow_teacher_actions is None or narrow_imitation_mask is None:
                raise RuntimeError(
                    "narrow imitation buffer requires teacher actions and a "
                    "scene mask at every step"
                )
            self.narrow_teacher_actions[i] = narrow_teacher_actions
            self.narrow_imitation_mask[i] = narrow_imitation_mask
        self.ptr += 1

    def reset(self):
        """每次 rollout 開始前呼叫，重置 ptr（不清除 tensor，直接覆蓋）。"""
        self.ptr = 0

    def sample_aux_sequences(
        self, seq_len: int, batch_size: int, burn_in: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int] | None:
        """為 TBPTT aux 訓練採樣不跨 episode 的連續序列。

        演算法:
          1. 建立有效起始位置集合: (t0, env)，條件是:
               - t0 + seq_len <= T_filled (序列不超過 buffer 已填充量)
               - [t0, t0+seq_len-2] 之間沒有 done（不跨 episode 邊界；done at t0+seq_len-1 允許，
                 因為最後一步仍是同一 episode 的觀測）
          2. 隨機採樣 batch_size 個起始位置（若不足則全取）
          3. 收集每條序列的 obs / aux_target / h0（用儲存的 hidden state 初始化）

        Args:
            seq_len:    序列總長（含 burn_in）
            batch_size: 目標序列數
            burn_in:    序列開頭只更新 hidden、不計入 loss 的步數

        Returns:
            (obs_seq, target_seq, h0, valid_count) 或 None（無合法序列時）
            obs_seq:     [B, L, obs_dim]
            target_seq:  [B, L, 7]
            h0:          [1, B, hidden_dim]
            valid_count: 合法起始位置總數（用於監控）
        """
        T_filled = self.ptr  # actual steps stored this rollout
        if T_filled < seq_len:
            return None

        # --- 1. Build valid start mask [T_filled - seq_len + 1, E] ---
        # For start t0, sequence covers [t0, t0+seq_len).
        # Invalid if any done[t] == 1 for t in [t0, t0+seq_len-2].
        # done at last step t0+seq_len-1 is ALLOWED — the terminal obs is still
        # from the same episode; there's no t0+seq_len that would cross the boundary.
        max_t0 = T_filled - seq_len  # inclusive
        E = self.num_envs

        dones_slice = self.dones[:T_filled]  # [T_filled, E]

        if seq_len <= 1:
            # Single step — always valid (no mid-sequence done possible)
            valid_mask = torch.ones(max_t0 + 1, E, dtype=torch.bool, device=self.device)
        elif seq_len == 2:
            # 2-step: only check done at t0 (not at t0+1 = last step)
            valid_mask = dones_slice[:max_t0 + 1] < 0.5  # [max_t0+1, E]
        else:
            # seq_len >= 3: check dones in [t0, t0+seq_len-2] via cumsum
            cum = torch.cumsum(dones_slice, dim=0)  # [T_filled, E]
            # end of check window: t0 + seq_len - 2 (NOT t0 + seq_len - 1)
            end_idx = torch.arange(seq_len - 2, seq_len - 2 + max_t0 + 1, device=self.device)
            cum_end = cum[end_idx]  # [max_t0+1, E]
            # cum_start: cum[t0-1] for t0 in [0..max_t0]; cum[-1] defined as 0
            cum_start = torch.zeros(1, E, device=self.device)
            if max_t0 > 0:
                start_idx = torch.arange(0, max_t0, device=self.device)
                cum_start = torch.cat([cum_start, cum[start_idx]], dim=0)  # [max_t0+1, E]
            dones_in_window = cum_end - cum_start  # [max_t0+1, E]
            valid_mask = dones_in_window < 0.5  # no mid-sequence dones

        # --- 2. Get valid (t0, env) pairs ---
        valid_positions = valid_mask.nonzero(as_tuple=False)  # [N_valid, 2] → (t0_offset, env_id)
        n_valid = valid_positions.shape[0]

        if n_valid == 0:
            return None

        # Sample
        actual_batch = min(batch_size, n_valid)
        chosen_idx = torch.randperm(n_valid, device=self.device)[:actual_batch]
        chosen = valid_positions[chosen_idx]  # [B, 2]
        t0s = chosen[:, 0]  # [B] — these are offsets from 0
        envs = chosen[:, 1]  # [B]

        # --- 3. Gather sequences ---
        B = actual_batch
        L = seq_len
        obs_dim = self.raw_obs.shape[-1]
        hidden_dim = self.hiddens.shape[-1]

        # Time indices: [B, L] where each row is [t0, t0+1, ..., t0+L-1]
        time_offsets = torch.arange(L, device=self.device).unsqueeze(0)  # [1, L]
        t_indices = t0s.unsqueeze(1) + time_offsets  # [B, L]
        e_indices = envs.unsqueeze(1).expand(B, L)  # [B, L]

        obs_seq = self.raw_obs[t_indices, e_indices]        # [B, L, obs_dim]
        target_seq = self.aux_targets[t_indices, e_indices]  # [B, L, 7]
        h0 = self.hiddens[t0s, envs].unsqueeze(0)           # [1, B, hidden_dim]

        return obs_seq, target_seq, h0, n_valid


# ============================================================================
# Obstacle Rollout Buffer
# ============================================================================
# 障礙物 rollout buffer 與 Charge 的主要差異：
#   - 每個障礙物視為「獨立 agent」，buffer 的 batch 維度 = num_envs × max_obstacles
#   - 不需要 RNN hidden state（障礙物 policy 是 stateless FC network）
#   - 不需要 raw_obs 或 aux_targets（障礙物沒有 aux loss）
#   - done = charge 的 done，廣播到所有障礙物（同一 env 的所有障礙物同步重置）

class ObstacleRolloutBuffer:
    """Flat buffer: 每個障礙物視為獨立樣本，B = num_envs × max_obstacles。

    obs_dim = OBS_POLICY_OBS_DIM = 9D per obstacle:
      own_xy(2) + own_vel(2) + robot_rel_xy(2) + robot_vel(2) + d_wall(1)
    act_dim = 2: continuous velocity command (vx, vy) in [-1, 1]
    """

    def __init__(self, num_steps, num_envs, max_obstacles, obs_dim, act_dim, device):
        self.num_steps = num_steps
        self.B = num_envs * max_obstacles  # 展平的 batch 維度（所有 env × 所有障礙物）
        self.device = device
        self.obs = torch.zeros(num_steps, self.B, obs_dim, device=device)    # [T, E*N, 9]
        self.actions = torch.zeros(num_steps, self.B, act_dim, device=device)# [T, E*N, 2]
        self.log_probs = torch.zeros(num_steps, self.B, device=device)
        self.rewards = torch.zeros(num_steps, self.B, device=device)
        self.values = torch.zeros(num_steps, self.B, device=device)
        self.dones = torch.zeros(num_steps, self.B, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        """儲存一個 rollout step 的障礙物資料。"""
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def reset(self):
        """每次 rollout 開始前重置 ptr。"""
        self.ptr = 0


# ============================================================================
# Obstacle environment interface
# ============================================================================
# 這組函數是 Charge 環境與 obstacle policy 之間的橋接層。
# Isaac Lab 沒有內建 multi-agent obstacle policy，因此需要手動:
#   1. build_obstacle_obs:  從 scene entities 讀取障礙物位置/速度，組成 policy 輸入
#   2. apply_obstacle_actions: 把 policy 輸出的速度命令寫回 sim（更新障礙物位置）
#   3. compute_obstacle_reward: 計算障礙物 policy 的 reward（WD: zero，或 weak adversarial）
# 注意：障礙物位置以 local frame（相對 env_origins）計算，再還原到 world frame 寫回 sim。

def build_obstacle_obs(env_unwrapped, max_obstacles: int, device) -> torch.Tensor:
    """讀取障礙物 + 機器人狀態，組裝 [num_envs, N, 9] obs tensor。

    Per-obstacle obs (以 env local frame 為基準):
      own_local_xy(2) + own_vel(2) + robot_rel_xy(2) + robot_vel(2) + d_wall(1) = 9D
    其中:
      own_local_xy: 障礙物在 env 內的 (x,y) 座標
      own_vel: 來自 _obstacle_velocities cache（上一步的 apply_obstacle_actions 更新）
      robot_rel_xy: 機器人位置 - 障礙物位置（相對方向）
      robot_vel: 機器人的世界系速度 (vx, vy)
      d_wall: 障礙物到最近邊界的距離
    非活躍障礙物（Z ≤ 0）填全零。
    """
    num_envs = env_unwrapped.num_envs
    env_origins = env_unwrapped.scene.env_origins  # [num_envs, 3]

    # Robot state
    robot_pos_w = env_unwrapped.scene["robot"].data.root_pos_w[:, :3]  # [E, 3]
    robot_local = robot_pos_w - env_origins  # [E, 3]
    robot_vel_w = env_unwrapped.scene["robot"].data.root_lin_vel_w[:, :2]  # [E, 2]

    # Ensure velocity cache exists
    if not hasattr(env_unwrapped, "_obstacle_velocities"):
        env_unwrapped._obstacle_velocities = torch.zeros(num_envs, max_obstacles, 2, device=device)

    obs_all = torch.zeros(num_envs, max_obstacles, OBS_POLICY_OBS_DIM, device=device)

    # Build obstacle cache if needed
    if not hasattr(env_unwrapped, "_obs_policy_cache"):
        env_unwrapped._obs_policy_cache = []
        for i in range(max_obstacles):
            name = f"obstacle_{i}"
            if name in env_unwrapped.scene.keys():
                env_unwrapped._obs_policy_cache.append(env_unwrapped.scene[name])
            else:
                env_unwrapped._obs_policy_cache.append(None)

    # Per-env scene bounds (randomized per episode reset)
    if hasattr(env_unwrapped, "_scene_bounds"):
        bound_limits = env_unwrapped._scene_bounds  # [E]
    else:
        bound_limits = torch.full((num_envs,), 7.0, device=device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue
        obs_pos_w = obstacle.data.root_pos_w[:, :3]  # [E, 3]
        obs_local = obs_pos_w - env_origins  # [E, 3]

        # Check if active (Z > 0)
        active = obs_pos_w[:, 2] > 0.0  # [E]

        obs_xy = obs_local[:, :2]  # [E, 2]
        obs_vel = env_unwrapped._obstacle_velocities[:, i, :]  # [E, 2]
        robot_rel = robot_local[:, :2] - obs_xy  # [E, 2] relative robot pos

        # d_wall: min distance to any boundary (per-env bound)
        d_left = obs_xy[:, 0] + bound_limits
        d_right = bound_limits - obs_xy[:, 0]
        d_bottom = obs_xy[:, 1] + bound_limits
        d_top = bound_limits - obs_xy[:, 1]
        d_wall = torch.stack([d_left, d_right, d_bottom, d_top], dim=-1).min(dim=-1).values  # [E]
        d_wall = torch.max(torch.min(d_wall, bound_limits), torch.zeros_like(d_wall))

        obs_all[:, i, 0:2] = obs_xy
        obs_all[:, i, 2:4] = obs_vel
        obs_all[:, i, 4:6] = robot_rel
        obs_all[:, i, 6:8] = robot_vel_w
        obs_all[:, i, 8] = d_wall

        # Zero out inactive obstacles
        inactive = ~active
        if inactive.any():
            obs_all[inactive, i, :] = 0.0

    return obs_all  # [num_envs, N, 9]


def apply_obstacle_actions(env_unwrapped, actions: torch.Tensor, max_obstacles: int,
                           dt: float = 0.2, speed_limit: float = 0.8, bound_limit: float = 7.0):
    """把障礙物 policy 輸出的速度命令套用到場景：位置更新 + 邊界反彈 + 寫回 sim。

    只移動 ACTIVE 障礙物（Z > 0），隱藏障礙物（Z=-10）跳過。
    使用 per-env _scene_bounds（支援 scene size randomization）。

    流程:
      1. actions * speed_limit → velocity（[-1,1] → 真實速度，m/s）
      2. L2 norm clamp：確保實際速度不超過 speed_limit
      3. local_xy += velocity * dt（Euler 積分）
      4. 邊界反彈（soft bounce）：越界後對稱反射 + final clamp
      5. write_root_pose_to_sim：把 new_pos 寫回 Isaac Sim（只更新 active 障礙物）
      6. 更新 _obstacle_velocities cache（供下一步 build_obstacle_obs 使用）

    Args:
        actions: [num_envs, N, 2] 速度命令，範圍 [-1, 1]，會乘以 speed_limit
    """
    env_origins = env_unwrapped.scene.env_origins  # [E, 3]

    velocity = actions * speed_limit  # [E, N, 2], per-axis scale
    # Vector norm clamp: limit actual speed (L2 norm) to speed_limit
    speed_norm = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    excess = speed_norm > speed_limit
    velocity = torch.where(excess, velocity * (speed_limit / speed_norm), velocity)

    # Per-env scene bounds (randomized per episode reset)
    if hasattr(env_unwrapped, "_scene_bounds"):
        bound_limits = env_unwrapped._scene_bounds  # [E]
    else:
        bound_limits = torch.full((env_unwrapped.num_envs,), bound_limit, device=actions.device)

    # Update velocity cache for obs_functions.py
    if not hasattr(env_unwrapped, "_obstacle_velocities"):
        env_unwrapped._obstacle_velocities = torch.zeros(
            env_unwrapped.num_envs, max_obstacles, 2, device=actions.device)

    for i, obstacle in enumerate(env_unwrapped._obs_policy_cache):
        if obstacle is None:
            continue

        pos_w = obstacle.data.root_pos_w.clone()  # [E, 3]
        active = pos_w[:, 2] > 0.0  # only move visible obstacles

        if not active.any():
            # Zero velocity for inactive obstacles
            env_unwrapped._obstacle_velocities[:, i, :] = 0.0
            continue

        # Local frame
        local_xy = pos_w[:, :2] - env_origins[:, :2]

        # Apply velocity only to active envs
        vel_i = velocity[:, i, :] * active.unsqueeze(-1).float()  # zero for inactive
        local_xy = local_xy + vel_i * dt

        # Geofence with soft bounce: reflect when hitting per-env boundary
        for dim in range(2):
            over_max = local_xy[:, dim] > bound_limits
            under_min = local_xy[:, dim] < -bound_limits
            local_xy[over_max, dim] = 2 * bound_limits[over_max] - local_xy[over_max, dim]
            local_xy[under_min, dim] = -2 * bound_limits[under_min] - local_xy[under_min, dim]
            # Final safety clamp (element-wise with per-env bounds)
            local_xy[:, dim] = torch.max(torch.min(local_xy[:, dim], bound_limits), -bound_limits)

        # Write back to world frame (only active)
        new_pos = pos_w.clone()
        new_pos[:, :2] = local_xy + env_origins[:, :2]
        # Preserve Z for inactive (keep at -10)
        new_pos[:, :2] = torch.where(active.unsqueeze(-1), new_pos[:, :2], pos_w[:, :2])

        quat = torch.zeros(env_unwrapped.num_envs, 4, device=actions.device)
        quat[:, 0] = 1.0  # identity quaternion
        pose = torch.cat([new_pos, quat], dim=-1)  # [E, 7]

        obstacle.write_root_pose_to_sim(pose)

        # Update velocity cache (only for active)
        env_unwrapped._obstacle_velocities[:, i, :] = vel_i


def compute_obstacle_reward(env_unwrapped, obs_obs: torch.Tensor, max_obstacles: int,
                            mode: str = "zero") -> torch.Tensor:
    """計算每個障礙物的 reward。

    WD 原版 (mode="zero")：障礙物沒有 reward 信號，純粹被 learned policy 驅動
    approach 模式 (mode="approach")：障礙物靠近機器人時得到小正 reward（弱對抗）

    Args:
        obs_obs: [num_envs, N, 9] 障礙物觀測（由 build_obstacle_obs 產生）
        mode: "zero"（WD 風格，障礙物無 reward）或 "approach"（弱對抗）
    Returns:
        [num_envs * N] 展平的 per-obstacle reward
    """
    num_envs = obs_obs.shape[0]
    N = max_obstacles

    if mode == "zero":
        return torch.zeros(num_envs * N, device=obs_obs.device)

    # "approach" mode: small reward for being close to robot
    robot_rel = obs_obs[:, :, 4:6]  # [E, N, 2]
    dist_to_robot = robot_rel.norm(dim=-1)  # [E, N]

    # Reward: closer → higher, max 0.1 at distance 0
    reward = (0.1 * (2.0 - dist_to_robot).clamp(0.0, 2.0) / 2.0)  # [E, N]

    # Speed penalty
    obs_vel = obs_obs[:, :, 2:4]
    speed = obs_vel.norm(dim=-1)
    reward = reward - 0.01 * speed

    return reward.reshape(-1)  # [E*N]


@torch.no_grad()
def compute_charge_action_diagnostics(env_unwrapped, actions: torch.Tensor) -> dict[str, torch.Tensor] | None:
    """把離散動作索引映射為物理動作（速度/加速度/角速度），供診斷用。

    Policy 輸出 [linear_idx, angular_idx] 兩個離散索引（各 19 種選擇）。
    這個函數從 action_manager 讀取 processed_actions（實際執行的速度/角速度）
    與 applied_accelerations（實際套用的加速度），方便觀察 policy 行為。

    Returns:
        dict 包含: linear_idx, angular_idx, v_x, a_x, omega
        或 None（無法取得 action term 時）
    """
    try:
        terms = getattr(env_unwrapped.action_manager, "_terms", {})
        action_term = terms.get("diff_drive", None)
        if action_term is None:
            action_term = next((t for t in terms.values() if hasattr(t, "processed_actions")), None)
        if action_term is None:
            return None

        processed = action_term.processed_actions.detach()
        applied = action_term.applied_accelerations.detach()
        if processed.shape[0] != actions.shape[0] or applied.shape[0] != actions.shape[0]:
            return None

        return {
            "linear_idx": actions[:, 0].detach().float(),
            "angular_idx": actions[:, 1].detach().float(),
            "v_x": processed[:, 0].float(),
            "a_x": applied[:, 0].float(),
            "omega": processed[:, 1].float(),
        }
    except (AttributeError, KeyError, RuntimeError, StopIteration):
        return None


# ============================================================================
# Warp Drive Charge Reward — 保留原作 per-phase sparse reward 結構
#
# WD spot reward 結構 (flat terrain, car_mode):
#   1. spot_reward_get_goal: +40 on goal reached
#   2. spot_penalty_hit: -5 ~ -200 on collision (wall/obstacle/any)
#   3. spot_cost_operate: per-step action cost (Phase 1 only, ~0.03/fps)
#   4. spot_reward_on_floor: 0 (all phases, flat terrain)
#
# 相比 Isaac Lab NavRL dense rewards，WD 更 sparse：
#   - 沒有 velocity_to_goal (方向速度獎勵)
#   - 沒有 safe_progress (安全進度 PBRS)
#   - 沒有 safety_log_distance (LiDAR 距離獎勵)
# ============================================================================

def compute_wd_charge_reward(
    env_unwrapped,
    actions: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    penalty_hit: float,
    reward_get_goal: float,
    cost_operate: float,
    penalty_timeout: float = 0.0,
    rl_fps: float = 5.0,
    cost_turn_rate: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """計算 Warp Drive 風格的 sparse reward（用於 charge agent 的 RL 訓練）。

    注意：Isaac Lab 環境本身也提供 dense reward（velocity_to_goal + safety 等），
    但這個函數完全獨立重新計算 WD 原版的 sparse reward，取代 env reward 供 PPO 使用。
    env 的 dense reward 只用於 MetricsCollector 裡的指標記錄（不參與 RL loss）。

    WD Reward 結構:
      + goal_reward:    到達目標 → +reward_get_goal（通常 +40）
      + wall_hit:       撞牆 → penalty_hit（-5 ~ -200，依 phase）
      + obs_hit:        撞障礙物 → penalty_hit（同上）
      + action_reward:  Phase 1 有 action cost（鼓勵省油/省力）

    從 termination_manager 讀取每個 termination term 的 buffer，
    區分 goal_reached / wall_collision / obstacle_collision / other_death。

    Returns:
        (reward, breakdown) where breakdown contains per-component tensors:
          - goal_reward: [N] goal reaching reward (WD: car goal reward)
          - wall_hit_reward: [N] wall/static collision penalty (WD: car static obstacle reward)
          - obs_hit_reward: [N] dynamic obstacle collision penalty (WD: car dynamic obstacle reward)
          - floor_reward: [N] always 0 for flat terrain (WD: car floor reward)
          - action_reward: [N] action cost (WD: car dynamic reward / cost_operate)
          - goal_reached: [N] bool
          - wall_collision: [N] bool
          - obs_collision: [N] bool
          - other_death: [N] bool (tipped/explosion)
    """
    N = terminated.shape[0]
    device = terminated.device
    reward = torch.zeros(N, device=device)

    terminated_flat = terminated.squeeze(-1).bool() if terminated.dim() > 1 else terminated.bool()
    truncated_flat = truncated.squeeze(-1).bool() if truncated.dim() > 1 else truncated.bool()

    # --- Detect goal_reached vs collision type from termination_manager ---
    goal_reached = torch.zeros(N, dtype=torch.bool, device=device)
    wall_collision = torch.zeros(N, dtype=torch.bool, device=device)
    obs_collision = torch.zeros(N, dtype=torch.bool, device=device)
    other_death = torch.zeros(N, dtype=torch.bool, device=device)

    try:
        tm = env_unwrapped.termination_manager
        for name in tm._term_names:
            buf = tm.get_term(name)
            if buf is None:
                continue
            if "goal_reached" in name or "reaching_goal" in name:
                goal_reached = goal_reached | buf.bool()
            elif "wall_collision" in name:
                wall_collision = wall_collision | buf.bool()
            elif "obstacle_collision" in name:
                obs_collision = obs_collision | buf.bool()
            elif "collision" in name:
                # Generic collision — attribute to obstacle if not wall
                obs_collision = obs_collision | buf.bool()
            elif "tipped" in name or "explosion" in name or "flying" in name:
                other_death = other_death | buf.bool()
    except (AttributeError, RuntimeError):
        # Fallback
        obs_collision = terminated_flat & ~truncated_flat

    any_collision = wall_collision | obs_collision | other_death

    # --- Static vs dynamic obstacle collision attribution ---
    static_obs_collision = getattr(
        env_unwrapped, "_obs_collision_static_mask",
        torch.zeros(N, dtype=torch.bool, device=device),
    )
    dynamic_obs_collision = getattr(
        env_unwrapped, "_obs_collision_dynamic_mask",
        torch.zeros(N, dtype=torch.bool, device=device),
    )

    # --- Per-component reward ---
    goal_reward = torch.zeros(N, device=device)
    wall_hit_reward = torch.zeros(N, device=device)
    obs_hit_reward = torch.zeros(N, device=device)
    action_reward = torch.zeros(N, device=device)

    # Goal reached: +reward_get_goal
    goal_reward[goal_reached] = reward_get_goal
    reward += goal_reward

    # Wall/static collision: penalty (WD: car static obstacle reward)
    wall_hit_reward[wall_collision] = penalty_hit
    reward += wall_hit_reward

    # Obstacle/dynamic collision: penalty (WD: car dynamic obstacle reward)
    obs_hit_reward[obs_collision & ~wall_collision] = penalty_hit
    reward += obs_hit_reward

    # Other death (tipped/explosion): penalty
    other_hit = other_death & ~wall_collision & ~obs_collision
    reward[other_hit] += penalty_hit

    # --- Action cost (WD: only meaningful when cost_operate > 0, typically Phase 1) ---
    if cost_operate > 0 and actions is not None:
        cost_per_step = cost_operate / rl_fps
        nomal_acc = (actions[:, 0].float() - 9.0).abs() / 9.0
        nomal_turn = (actions[:, 1].float() - 9.0).abs() / 9.0
        action_reward = cost_per_step * (1.0 - nomal_acc) ** 2
        action_reward = action_reward + cost_per_step * cost_turn_rate * (1.0 - nomal_turn) ** 2
        alive = ~terminated_flat
        action_reward = action_reward * alive.float()
        reward += action_reward

    # --- Timeout penalty (truncated but not terminated = episode 時間到但未碰撞/未到達目標) ---
    timeout_reward = torch.zeros(N, device=device)
    if penalty_timeout != 0.0:
        is_timeout = truncated_flat & ~terminated_flat
        timeout_reward[is_timeout] = penalty_timeout
        reward += timeout_reward

    breakdown = {
        "goal_reward": goal_reward,              # WD: car goal reward
        "wall_hit_reward": wall_hit_reward,      # WD: car static obstacle reward
        "obs_hit_reward": obs_hit_reward,        # WD: car dynamic obstacle reward
        "timeout_reward": timeout_reward,        # timeout penalty
        "floor_reward": torch.zeros(N, device=device),  # WD: car floor reward (0 for flat)
        "action_reward": action_reward,          # WD: car dynamic reward (action cost)
        "goal_reached": goal_reached,
        "wall_collision": wall_collision,
        "obs_collision": obs_collision,
        "static_obs_collision": static_obs_collision,
        "dynamic_obs_collision": dynamic_obs_collision,
        "other_death": other_death,
    }

    return reward, breakdown


# ============================================================================
# PPO Utilities
# ============================================================================
# 注意：compute_gae 和 sample_action / evaluate_actions 都是 Charge policy 用的。
# Obstacle policy 使用更通用的 ppo_update_continuous（continuous action）。

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda, terminateds=None):
    """計算 Generalized Advantage Estimation (GAE)。

    ★fix#1 truncation bootstrap 正確處理:
      - dones = episode 邊界(terminal|truncation) → 控 GAE 遞迴重置(episode 結束不跨界傳播)。
      - terminateds = 真 terminal(撞/到達) → 控 delta 是否砍 bootstrap。
      - truncation(timeout,dones=1 但 terminateds=0):不砍 bootstrap,用 V(s_t) 自身近似 V(s_final)
        (因狀態價值還在,只是時間到);但 GAE 遞迴仍重置(下一步是新 episode)。
      - terminateds=None → 退回舊行為(dones 全當 terminal;有 truncation-bootstrap bug,保留供對照)。
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        if terminateds is not None:
            term_t = terminateds[t]                                   # 1=真terminal
            trunc_t = torch.clamp(dones[t] - term_t, 0.0, 1.0)        # 1=truncation only
            # delta bootstrap:terminal→砍(next=0);truncation→用 V(s_t) 自身;normal→next_value
            _delta_next = torch.where(trunc_t.bool(), values[t], next_value)
            delta = rewards[t] + gamma * _delta_next * (1.0 - term_t) - values[t]
            # GAE 遞迴:episode 邊界(term 或 trunc)都重置 last_gae
            _recur_cont = 1.0 - dones[t]
        else:
            _non_terminal = 1.0 - dones[t]  # 舊行為:done=1 砍 bootstrap+遞迴
            delta = rewards[t] + gamma * next_value * _non_terminal - values[t]
            _recur_cont = _non_terminal
        last_gae = delta + gamma * gae_lambda * _recur_cont * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values  # (A, V_target)


def sample_action(logits):
    """從 policy logits 採樣離散動作，同時計算 log_prob 和 entropy。

    logits: [E, NUM_BINS*2]，前 NUM_BINS 是線性加速，後 NUM_BINS 是角速度
    Returns:
      actions:  [E, 2] long tensor (linear_idx, angular_idx)
      log_prob: [E] joint log prob = log π(a1) + log π(a2)
      entropy:  [E] joint entropy（用於 A2CK per-head entropy loss）
    """
    logits_a = logits[:, :NUM_BINS]   # 線性加速 head 的 logits（19 種選擇）
    logits_w = logits[:, NUM_BINS:]   # 角速度 head 的 logits（19 種選擇）
    dist_a = Categorical(logits=logits_a)
    dist_w = Categorical(logits=logits_w)
    action_a = dist_a.sample()
    action_w = dist_w.sample()
    log_prob = dist_a.log_prob(action_a) + dist_w.log_prob(action_w)
    entropy = dist_a.entropy() + dist_w.entropy()
    return torch.stack([action_a, action_w], dim=-1), log_prob, entropy


def evaluate_actions(logits, actions):
    """重新評估已採樣動作的 log_prob（PPO 用），同時返回 per-head entropy（A2CK 用）。

    Returns:
      log_prob:        [B] joint log prob（同 sample_action，但基於舊動作）
      entropy_linear:  [B] 線性加速 head 的 entropy（WD: ent_coeff_linear × H1）
      entropy_angular: [B] 角速度 head 的 entropy（WD: ent_coeff_angular × H2）
    """
    logits_a = logits[:, :NUM_BINS]
    logits_w = logits[:, NUM_BINS:]
    dist_a = Categorical(logits=logits_a)
    dist_w = Categorical(logits=logits_w)
    log_prob = dist_a.log_prob(actions[:, 0]) + dist_w.log_prob(actions[:, 1])
    return log_prob, dist_a.entropy(), dist_w.entropy()


def ppo_update_continuous(policy, value_fn, buffer, optimizer, epochs, mini_batches,
                          clip_eps, vf_coeff, ent_coeff, max_grad_norm, gamma, gae_lambda):
    """通用 PPO 更新（用於連續動作的 Obstacle policy）。

    與 Charge 的 A2C/PPO 更新不同之處：
      - 使用標準 clipped surrogate loss（PPO）
      - 不分 actor/critic 兩個 optimizer，單一 optimizer 更新所有參數
      - 不做 WD-style gradient cap（障礙物不是主要訓練目標）
      - 單一 entropy coeff（不分 per-head）
    """
    T = buffer.ptr
    B = buffer.B

    # Bootstrap value for last step
    with torch.no_grad():
        last_obs = buffer.obs[T - 1]
        last_value = value_fn(last_obs).squeeze(-1)

    advantages, returns = compute_gae(
        buffer.rewards[:T], buffer.values[:T], buffer.dones[:T],
        last_value, gamma, gae_lambda)
    advantages = advantages - advantages.mean()  # mean-only normalization（不除 std）

    flat_obs = buffer.obs[:T].reshape(-1, buffer.obs.shape[-1])
    flat_actions = buffer.actions[:T].reshape(-1, buffer.actions.shape[-1])
    flat_log_probs = buffer.log_probs[:T].reshape(-1)
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)

    batch_size = T * B
    mini_batch_size = batch_size // mini_batches

    p_losses, v_losses, ent_vals, total_losses = [], [], [], []
    for _ in range(epochs):
        indices = torch.randperm(batch_size, device=flat_obs.device)
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            idx = indices[start:end]

            mb_obs = flat_obs[idx]
            mb_act = flat_actions[idx]
            mb_old_lp = flat_log_probs[idx]
            mb_adv = flat_adv[idx]
            mb_ret = flat_ret[idx]

            new_lp, entropy = policy.evaluate(mb_obs, mb_act)
            new_val = value_fn(mb_obs).squeeze(-1)

            ratio = (new_lp - mb_old_lp).exp()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = F.mse_loss(new_val, mb_ret)
            loss = p_loss + vf_coeff * v_loss - ent_coeff * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_fn.parameters()), max_grad_norm)
            optimizer.step()
            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
            ent_vals.append(entropy.mean().item())
            total_losses.append(loss.item())

    return {
        "policy_loss": np.mean(p_losses) if p_losses else 0.0,
        "value_loss": np.mean(v_losses) if v_losses else 0.0,
        "entropy": np.mean(ent_vals) if ent_vals else 0.0,
        "total_loss": np.mean(total_losses) if total_losses else 0.0,
    }


# ============================================================================
# WandB Metrics — Warp Drive 完整指標移植 (spot→charge)
# ============================================================================
# WandB Metrics
# ============================================================================


class MetricsCollector:
    """Warp Drive 風格完整指標收集器。

    指標命名對照 Warp Drive custom_trainer.py:
      spot → charge, car → charge
      goal → goal (unchanged)
      obstacle → obstacle

    架構：
      - Per-env accumulators: 追蹤每個 env 當前 episode 的累計值（tensor, 長期存活）
      - Completed lists: 每當 episode 結束時，把累計值放進 list（CPU float list）
      - collect(): 把 completed lists 統計後輸出 WandB metrics dict，同時清空 list
      - reset(): 每次 iteration 開始前呼叫，清空 completed lists 和計數器
        （注意：per-env accumulators 不在 reset() 清空，
        因為它們需要跨 rollout 保持連續性；只在 episode 結束時清零）
    """

    def __init__(self, num_envs: int, max_obstacles: int, device, action_table_sample_size: int = 2048):
        self.num_envs = num_envs
        self.max_obstacles = max_obstacles
        self.device = device
        self.action_table_sample_size = max(0, int(action_table_sample_size))

        # --- Per-env episode accumulators（跨 rollout 累積，只在 episode 結束時清零）---
        self._ep_reward = torch.zeros(num_envs, device=device)       # 累計 episode reward
        self._ep_length = torch.zeros(num_envs, device=device)       # episode 總步數
        self._ep_steps_alive = torch.zeros(num_envs, device=device)  # 存活步數（done=0 的步數）

        # --- Per-env WD reward decomposition accumulators ---
        # 每個 env 的 episode 內累加各 reward 分量，episode 結束時記錄
        self._ep_goal_reward = torch.zeros(num_envs, device=device)      # +goal reward 累計
        self._ep_wall_hit_reward = torch.zeros(num_envs, device=device)  # wall hit penalty 累計
        self._ep_obs_hit_reward = torch.zeros(num_envs, device=device)   # obs hit penalty 累計
        self._ep_floor_reward = torch.zeros(num_envs, device=device)     # floor reward 累計（恆 0）
        self._ep_action_reward = torch.zeros(num_envs, device=device)    # action cost 累計

        # --- Completed episode reward decomposition ---
        self._completed_goal_reward: list[float] = []
        self._completed_wall_hit_reward: list[float] = []
        self._completed_obs_hit_reward: list[float] = []
        self._completed_floor_reward: list[float] = []
        self._completed_action_reward: list[float] = []

        # --- Completed episode stats ---
        self._completed_rewards: list[float] = []
        self._completed_lengths: list[float] = []
        self._completed_alive: list[float] = []

        # --- Per-outcome expected value tracking (WD 期望值) ---
        # 每個 episode 完成時，根據結局（goal/collision/timeout）分別記錄 reward 和 length，
        # 用於計算條件期望值 E[reward | outcome]。
        self._ev_reward_goal: list[float] = []       # E[reward | goal_reached]
        self._ev_reward_collision: list[float] = []  # E[reward | collision]
        self._ev_reward_timeout: list[float] = []    # E[reward | timeout]
        self._ev_length_goal: list[float] = []       # E[length | goal_reached]
        self._ev_length_collision: list[float] = []  # E[length | collision]
        self._ev_length_timeout: list[float] = []    # E[length | timeout]
        self._ev_alive_ratio: list[float] = []       # alive_steps / total_steps per episode

        # --- Goal-directed behavior diagnostics (monitoring only, not reward) ---
        self._ep_goal_start_dist = torch.full((num_envs,), float("nan"), device=device)
        self._ep_goal_prev_dist = torch.full((num_envs,), float("nan"), device=device)
        self._ep_goal_progress_sum = torch.zeros(num_envs, device=device)
        self._ep_goal_velocity_sum = torch.zeros(num_envs, device=device)
        self._ep_goal_heading_abs_sum = torch.zeros(num_envs, device=device)
        self._ep_goal_diag_steps = torch.zeros(num_envs, device=device)
        self._ep_goal_prev_target_id = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._ep_goal_switch_count = torch.zeros(num_envs, device=device)
        self._completed_goal_start_dist: list[float] = []
        self._completed_goal_end_dist: list[float] = []
        self._completed_goal_distance_delta: list[float] = []
        self._completed_goal_progress_mean: list[float] = []
        self._completed_velocity_to_goal_mean: list[float] = []
        self._completed_heading_error_abs_mean: list[float] = []
        self._completed_target_switch_rate: list[float] = []
        self._completed_goal_reset_dist: list[float] = []
        self._completed_goal_reset_angle_abs: list[float] = []
        self._completed_steps_to_goal: list[float] = []
        self._completed_goal_blocked_by_wall: list[float] = []

        # --- Charge action diagnostics ---
        self._action_linear_idx: list[float] = []
        self._action_angular_idx: list[float] = []
        self._action_speed_x: list[float] = []
        self._action_accel_x: list[float] = []
        self._action_omega: list[float] = []
        self._action_speed_accel_rows: list[list[float]] = []

        # --- v3: Per-env jitter sampling (16 random envs, per-env trajectories) ---
        # 1024 envs 平均會掩蓋單 env 抽動 → 抽 16 個 env 個別記錄 omega + angular_idx 軌跡
        # → flush 時計算每個 env 的 std / flip_rate，再取 p50 / p95 作為抽動真相指標
        self._jitter_n_sample: int = 16
        self._jitter_sample_ids: torch.Tensor | None = None  # set on first action; fixed across rollouts
        self._jitter_omega_traj: list[list[float]] = []      # [n_sample][T] applied ω trajectory
        self._jitter_ang_ratio_traj: list[list[float]] = []  # [n_sample][T] angular ratio trajectory

        # --- LiDAR-noise impact diagnostics (post-noise, normalized to [0, 1]) ---
        # step-mean of per-env min(lidar) → 0 means right at obstacle, 1 = clear
        self._lidar_min_step: list[float] = []
        # step-mean number of near-zero rays per env (proxy for hole_rate effect)
        self._lidar_zero_count_step: list[float] = []
        # ★07-06 護欄: LiDAR 飽和率 (obs=1.0 的 bin 佔比; obs=(d_surf)/r_max, =1 表 >r_max 無資訊)
        self._lidar_sat_frac_step: list[float] = []
        # step-mean speed_x conditioned on min(lidar) < LIDAR_NEAR_THRESH
        # (only logged when at least one env is near an obstacle)
        self._action_speed_x_near_obs: list[float] = []
        self._lidar_near_obs_env_count: list[float] = []
        # ★07-06 前錐減速指標(密集階段技能監控): 前方±30°錐(bin36±6) min<1.2m 時的前進速度
        self._action_speed_x_front_blocked: list[float] = []
        self._front_blocked_env_count: list[float] = []
        # ★速度 vs 前方距離 曲線診斷: 前錐(±30°)最近障礙分段(m),各段平均前進速度
        #   → 直接看 policy「離前方障礙多近、速度掉多少」= 幾米開始反應/減速。key=bin 上緣(m)。
        self._front_dist_speed: dict[str, list[float]] = {
            k: [] for k in ("0.5", "1.0", "1.5", "2.0", "2.5", "3.0")
        }
        # ★|ω| vs 前方距離: policy 靠轉向避障→提早反應藏在角速度(遠處就加大轉向、線速度不降)。
        #   線速度曲線平≠沒反應;要看此 |ω| 曲線是否在遠距(2-3m)就抬高=轉向式提早避開。
        self._front_dist_omega: dict[str, list[float]] = {
            k: [] for k in ("0.5", "1.0", "1.5", "2.0", "2.5", "3.0")
        }
        # ★動態專屬反應曲線(LV-DOT channel, 按 closing speed 分組) — 前錐 LiDAR 曲線混了牆/靜態,
        #   無法乾淨看「對快速接近的動態物是否提早反應」。改讀 LV-DOT slot0(最近動態物):
        #   px=body 前向距離, vx=body 接近速度(-=接近)。分快接近(closing>0.4)/慢或遠離兩組。
        #   驗證: 若 policy 用 LV-DOT 速度提早避,快接近組應在較遠距離(2-3m)就降速/抬|ω|
        #   (vs 慢組)。stage3 加距離稅前後對比 = 稅有沒有教會「用速度提早避」的行為層鐵證。
        #   需 CHARGE_USE_LVDOT_OBS=1(obs≥85D);無 channel 時整段跳過不記。
        _dyn_bk = ("0.5", "1.0", "1.5", "2.0", "2.5", "3.0")
        self._dyn_fast_speed: dict[str, list[float]] = {k: [] for k in _dyn_bk}
        self._dyn_fast_omega: dict[str, list[float]] = {k: [] for k in _dyn_bk}
        self._dyn_slow_speed: dict[str, list[float]] = {k: [] for k in _dyn_bk}
        self._dyn_slow_omega: dict[str, list[float]] = {k: [] for k in _dyn_bk}

        # --- Termination counters ---
        self._goal_reached = 0
        self._collision = 0           # total: wall + obstacle + geometric
        self._wall_collision = 0
        self._obstacle_collision = 0
        self._static_obstacle_collision = 0
        self._dynamic_obstacle_collision = 0
        self._timeout = 0
        self._tipped_over = 0
        self._total_eps = 0
        self._first_step_deaths = 0   # dies_at_birth: done at step <= 2

        # --- Reward term accumulators (Isaac Lab Episode_Reward/) ---
        self._reward_terms: dict[str, list[float]] = {}

        # --- Curriculum info ---
        self._curriculum_info: dict[str, float] = {}

        # --- Obstacle policy per-step stats ---
        self._obs_speed_limit: float = 0.8  # updated per-iter from curriculum
        self._obs_speeds: list[float] = []
        self._obs_distances: list[float] = []  # distance to robot

    def compute_goal_diagnostics(self, env_unwrapped) -> dict[str, torch.Tensor] | None:
        """計算 per-env 目標導向診斷指標（在 env.step() 之前呼叫）。

        這些指標用來判斷 policy 是否真的在向目標移動，只做記錄，不影響 reward。
        包括：
          - distance:        機器人到目標的距離
          - velocity_to_goal: 機器人速度在目標方向上的投影（正值 = 靠近目標）
          - heading_error_abs: |yaw - goal_angle|（絕對 heading 誤差，rad）
          - target_id:       目前追蹤的目標 ID（用於偵測目標切換）
          - goal_blocked_by_wall: 目標被牆擋住（LOS check）
          - is_reset:        episode 第一步（用於記錄 reset 時的初始距離/角度）
        """
        try:
            robot = env_unwrapped.scene["robot"]
            robot_pos = robot.data.root_pos_w[:, :2]
            robot_vel = robot.data.root_lin_vel_w[:, :2]
            robot_quat = robot.data.root_quat_w

            if hasattr(env_unwrapped, "_local_goal_world") and env_unwrapped._local_goal_world is not None:
                goal_pos = env_unwrapped._local_goal_world[:, :2]
            else:
                goal_pos = env_unwrapped.command_manager.get_command("goal_command")[:, :2]

            target_id = None
            try:
                goal_term = env_unwrapped.command_manager.get_term("goal_command")
                if hasattr(goal_term, "nearest_goal_idx"):
                    target_id = goal_term.nearest_goal_idx.detach().long().to(robot_pos.device)
            except (AttributeError, KeyError, RuntimeError, ValueError):
                target_id = None
            if target_id is None or target_id.shape[0] != robot_pos.shape[0]:
                quantized_goal = torch.round(goal_pos * 100.0).long()
                target_id = quantized_goal[:, 0] * 1000003 + quantized_goal[:, 1]
            reset_buf = getattr(env_unwrapped, "episode_length_buf", None)
            is_reset = reset_buf.detach().to(robot_pos.device) == 0 if reset_buf is not None else None

            diff = torch.nan_to_num(goal_pos - robot_pos, nan=0.0, posinf=0.0, neginf=0.0)
            dist = torch.norm(diff, dim=1).clamp_min(1e-6)
            goal_dir = diff / dist.unsqueeze(-1)
            velocity_to_goal = (robot_vel * goal_dir).sum(dim=1)

            w, x, y, z = robot_quat[:, 0], robot_quat[:, 1], robot_quat[:, 2], robot_quat[:, 3]
            yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            goal_angle = torch.atan2(diff[:, 1], diff[:, 0])
            heading_error = torch.atan2(torch.sin(goal_angle - yaw), torch.cos(goal_angle - yaw))

            blocked_by_wall = torch.zeros_like(dist, dtype=torch.bool)
            try:
                from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.wall_layout import (
                    check_los_perenv,
                    get_combined_wall_data,
                )
                wall_c, wall_s, wall_mask = get_combined_wall_data(env_unwrapped)
                visible = check_los_perenv(robot_pos, goal_pos.unsqueeze(1), wall_c, wall_s, wall_mask)
                blocked_by_wall = ~visible[:, 0]
            except (ImportError, AttributeError, RuntimeError, ValueError):
                blocked_by_wall = torch.zeros_like(dist, dtype=torch.bool)

            return {
                "distance": dist,
                "velocity_to_goal": velocity_to_goal,
                "heading_error_abs": heading_error.abs(),
                "target_id": target_id,
                "goal_blocked_by_wall": blocked_by_wall.float(),
                "is_reset": is_reset if is_reset is not None else torch.zeros_like(dist, dtype=torch.bool),
            }
        except (AttributeError, KeyError, RuntimeError):
            return None

    def step(self, obs, reward, done, info,
             obs_obs: torch.Tensor | None = None,
             obs_actions: torch.Tensor | None = None,
             reward_breakdown: dict[str, torch.Tensor] | None = None,
             goal_diagnostics: dict[str, torch.Tensor] | None = None,
             charge_actions: dict[str, torch.Tensor] | None = None,
             rollout_step: int | None = None):
        """記錄一個 env step 的所有指標。在每個 rollout step 結束後呼叫。

        Args:
            obs:             [E, 139] 當前觀測（未使用，保留介面）
            reward:          [E] 當前 step reward（Isaac Lab env reward）
            done:            [E] 是否 episode 結束
            info:            env.step() 返回的 info dict（含 Episode_Reward/ 等）
            obs_obs:         [E, N, 9] 障礙物觀測（用於障礙物速度/距離 metrics）
            obs_actions:     [E, N, 2] 障礙物動作（用於障礙物速度 metrics）
            reward_breakdown: compute_wd_charge_reward 返回的 per-component reward dict
            goal_diagnostics: compute_goal_diagnostics 返回的診斷 dict
            charge_actions:  compute_charge_action_diagnostics 返回的物理動作診斷
            rollout_step:    當前 rollout 內的步數（用於 action table 的時間戳）
        """
        self._ep_reward += reward.reshape(-1)
        self._ep_length += 1.0
        self._ep_steps_alive += (1.0 - done.reshape(-1).float())

        # --- Accumulate WD reward decomposition ---
        if reward_breakdown is not None:
            self._ep_goal_reward += reward_breakdown["goal_reward"]
            self._ep_wall_hit_reward += reward_breakdown["wall_hit_reward"]
            self._ep_obs_hit_reward += reward_breakdown["obs_hit_reward"]
            self._ep_floor_reward += reward_breakdown["floor_reward"]
            self._ep_action_reward += reward_breakdown["action_reward"]
            # ★shaping 項(rollout-loop 加的)→ 記入 _reward_terms → wandb reward/term/shaping_*,
            #   供監控 policy 是否響應 Term1(predictive_decel 隨訓練上升=學會該減速時減速)。
            for _sk in ("ttc_tax", "gap_heading", "predictive_decel", "receding_penalty",
                        "turning_direction", "early_turning", "weakened_decel", "anti_spin",
                        "anti_spin_active", "anti_spin_run_steps", "anti_spin_same_sign_yaw_deg",
                        "future_occupancy", "future_occupancy_active", "future_occupancy_risk",
                        "future_occupancy_min_distance_m",
                        "future_occupancy_long_corridor",
                        "future_occupancy_active_long_corridor",
                        "future_occupancy_risk_long_corridor",
                        "future_occupancy_risk_active_long_corridor",
                        "future_occupancy_min_distance_m_active_long_corridor",
                        "progress_reward_long_corridor",
                        "wall_collision_long_corridor",
                        "static_obs_collision_long_corridor",
                        "dynamic_obs_collision_long_corridor",
                        "future_occupancy_risk_on_dynamic_collision_long_corridor",
                        "future_occupancy_active_on_dynamic_collision_long_corridor",
                        "future_occupancy_min_distance_m_on_dynamic_collision_long_corridor"):
                _sv = reward_breakdown.get(_sk, None)
                if _sv is not None:
                    self._reward_terms.setdefault(f"shaping_{_sk}", []).append(
                        float(_sv.mean().item()) if hasattr(_sv, "mean") else float(_sv))

        # --- Goal-directed diagnostics ---
        if goal_diagnostics is not None:
            goal_dist = goal_diagnostics["distance"].detach()
            first_valid = ~torch.isfinite(self._ep_goal_start_dist)
            self._ep_goal_start_dist[first_valid] = goal_dist[first_valid]
            reset_valid = goal_diagnostics.get("is_reset")
            if reset_valid is None:
                reset_valid = first_valid
            else:
                reset_valid = reset_valid.detach().bool()
            if reset_valid.any():
                self._completed_goal_reset_dist.extend(goal_dist[reset_valid].detach().cpu().tolist())
                self._completed_goal_reset_angle_abs.extend(
                    (goal_diagnostics["heading_error_abs"][reset_valid].detach().cpu() * 180.0 / math.pi).tolist()
                )
                if "goal_blocked_by_wall" in goal_diagnostics:
                    self._completed_goal_blocked_by_wall.extend(
                        goal_diagnostics["goal_blocked_by_wall"][reset_valid].detach().cpu().tolist()
                    )
            prev_valid = torch.isfinite(self._ep_goal_prev_dist)
            step_progress = torch.zeros_like(goal_dist)
            step_progress[prev_valid] = self._ep_goal_prev_dist[prev_valid] - goal_dist[prev_valid]
            self._ep_goal_prev_dist = goal_dist
            self._ep_goal_progress_sum += step_progress
            self._ep_goal_velocity_sum += goal_diagnostics["velocity_to_goal"].detach()
            self._ep_goal_heading_abs_sum += goal_diagnostics["heading_error_abs"].detach()
            target_id = goal_diagnostics.get("target_id")
            if target_id is not None:
                target_id = target_id.detach().long()
                prev_target_valid = self._ep_goal_prev_target_id >= 0
                target_switched = prev_target_valid & (self._ep_goal_prev_target_id != target_id)
                self._ep_goal_switch_count += target_switched.float()
                self._ep_goal_prev_target_id = target_id
            self._ep_goal_diag_steps += 1.0

        # --- Charge action diagnostics ---
        if charge_actions is not None:
            v_x = charge_actions["v_x"].detach()
            a_x = charge_actions["a_x"].detach()
            omega = charge_actions["omega"].detach()
            lin_idx = charge_actions["linear_idx"].detach()
            ang_idx = charge_actions["angular_idx"].detach()
            self._action_linear_idx.append(lin_idx.float().mean().item())
            self._action_angular_idx.append(ang_idx.float().mean().item())
            self._action_speed_x.append(v_x.float().mean().item())
            self._action_accel_x.append(a_x.float().mean().item())
            self._action_omega.append(omega.float().mean().item())

            # --- v3: Per-env jitter sampling ---
            # 從 envs 隨機抽 jitter_n_sample 個 env（首次呼叫時固定 ids）
            # 記錄個別 env 的 applied_omega 與 angular_ratio 軌跡，flush 時算 std / flip_rate
            num_envs_total = omega.shape[0]
            if self._jitter_sample_ids is None and num_envs_total > 0:
                n_sample = min(self._jitter_n_sample, num_envs_total)
                # deterministic by training seed: 用 torch.randperm 在 omega 的 device 上
                self._jitter_sample_ids = torch.randperm(
                    num_envs_total, device=omega.device
                )[:n_sample]
                self._jitter_omega_traj = [[] for _ in range(n_sample)]
                self._jitter_ang_ratio_traj = [[] for _ in range(n_sample)]
            if self._jitter_sample_ids is not None:
                sampled_omega = omega[self._jitter_sample_ids].float().cpu().tolist()
                # angular ratio: (ang_idx - 9) / 9 ∈ [-1, 1]
                sampled_ratio = ((ang_idx[self._jitter_sample_ids].float() - 9.0) / 9.0).cpu().tolist()
                for i, (o_val, r_val) in enumerate(zip(sampled_omega, sampled_ratio)):
                    self._jitter_omega_traj[i].append(o_val)
                    self._jitter_ang_ratio_traj[i].append(r_val)

        # --- LiDAR-noise impact diagnostics ---
        # obs is 139D: [ego(4) + goal(2) + lidar(72) + obstacles(60) + time(1)]
        # lidar slice [6:78] is normalized to [0, 1] where 0=at-obstacle, 1=at-max-range.
        # See observations/functions.py:75 — normalized = distances / max_range.
        #
        # FIX (2026-06-06): raw min(lidar) is dominated by hole_rate/distractor noise
        # (post-noise rays ≈ 0 are not real obstacles). Mask holes first, then min.
        # See feedback_lidar_min_metric_fix.md.
        HOLE_MASK_THRESH = 0.02          # normalized ≈ 2% of max_range → treat as hole/distractor
        LIDAR_NEAR_THRESH = 0.15         # normalized ≈ 15% of max_range → robust slowdown region
        if obs is not None and obs.dim() >= 2 and obs.shape[-1] >= 78:
            lidar = obs[..., 6:78].detach()                                          # [E, 72]
            # Hole counter (cheap, pre-mask) — proxy for hole_rate visible to policy
            zero_count_per_env = (lidar < HOLE_MASK_THRESH).float().sum(dim=-1)      # [E]
            self._lidar_zero_count_step.append(zero_count_per_env.mean().item())
            # 飽和率: obs >= 0.999 的 bin 佔比 (健康血緣應遠 <0.5; 若 >0.7 = 視野被 clip 鍘)
            self._lidar_sat_frac_step.append((lidar >= 0.999).float().mean().item())
            # Hole-masked min: replace hole/distractor rays with +inf so they lose the min vote
            lidar_clean = torch.where(
                lidar < HOLE_MASK_THRESH,
                torch.full_like(lidar, float("inf")),
                lidar,
            )
            lidar_min_clean = lidar_clean.min(dim=-1).values                         # [E]
            # If ALL rays are holes (degenerate), fall back to 1.0 (max-range, "no info")
            all_holes_mask = torch.isinf(lidar_min_clean)
            lidar_min_per_env = torch.where(
                all_holes_mask,
                torch.ones_like(lidar_min_clean),
                lidar_min_clean,
            )
            self._lidar_min_step.append(lidar_min_per_env.mean().item())
            # Speed when close to obstacle (using masked min)
            near_mask = lidar_min_per_env < LIDAR_NEAR_THRESH                        # [E] bool
            n_near = int(near_mask.sum().item())
            if n_near > 0 and charge_actions is not None:
                v_near = v_x[near_mask].float().mean().item()
                self._action_speed_x_near_obs.append(v_near)
                self._lidar_near_obs_env_count.append(float(n_near))
            # ★前錐減速: bin36=正前, ±6 bins=±30°; 門檻 0.10(norm)×20=2.0m (對齊水滴稅 d_react;07-06 修 obs=/20)
            front_min = lidar_clean[:, 30:43].min(dim=-1).values                     # [E]
            front_blocked = (~torch.isinf(front_min)) & (front_min < 0.10)
            n_fb = int(front_blocked.sum().item())
            if n_fb > 0 and charge_actions is not None:
                self._action_speed_x_front_blocked.append(v_x[front_blocked].float().mean().item())
                self._front_blocked_env_count.append(float(n_fb))
            # ★速度 vs 前方距離 曲線: 前錐最近障礙(m)分段,記各段平均前進速度(bin 上緣為 key)
            #   _LIDAR_MAX_DISTANCE_M 為訓練函數域區域變數(此處不可及),obs=/20 故硬編 20.0。
            if charge_actions is not None:
                _front_m = front_min * 20.0
                for _lo, _hi, _bk in (
                    (0.0, 0.5, "0.5"), (0.5, 1.0, "1.0"), (1.0, 1.5, "1.5"),
                    (1.5, 2.0, "2.0"), (2.0, 2.5, "2.5"), (2.5, 3.0, "3.0"),
                ):
                    _bm = (~torch.isinf(front_min)) & (_front_m >= _lo) & (_front_m < _hi)
                    if int(_bm.sum().item()) > 0:
                        self._front_dist_speed[_bk].append(v_x[_bm].float().mean().item())
                        self._front_dist_omega[_bk].append(omega[_bm].abs().float().mean().item())

            # ★動態專屬反應曲線: 讀 LV-DOT channel slot0(最近動態物)按 closing speed 分組。
            #   channel 在 obs[79:109], slot0=obs[79:85]=[px,py,vx,vy,r,valid]
            #   (已正規化: px/8, vx/1.5)。僅 CHARGE_USE_LVDOT_OBS=1(obs≥85D)時記錄。
            if charge_actions is not None and obs is not None and obs.dim() >= 2 and obs.shape[-1] >= 85:
                _ch0 = obs[..., 79:85].detach()               # [E,6] slot0(最近動態物)
                _dpx = _ch0[:, 0] * 8.0                        # body 前向距離 m (+前)
                _dpy = _ch0[:, 1] * 8.0                        # body 側向 m
                _dvx = _ch0[:, 2] * 1.5                        # body 接近速度 m/s (-=接近)
                _dvalid = _ch0[:, 5] > 0.5
                _dfront = _dvalid & (_dpx > 0.0)              # 在前方的有效動態物
                _closing = -_dvx                              # +=接近
                _ddist = torch.sqrt(_dpx ** 2 + _dpy ** 2 + 1e-8)  # euclidean 距離
                _fast = _dfront & (_closing > 0.4)           # 快接近(>0.4 m/s)
                _slow = _dfront & (_closing <= 0.4)          # 慢/遠離
                for _lo, _hi, _bk in (
                    (0.0, 0.5, "0.5"), (0.5, 1.0, "1.0"), (1.0, 1.5, "1.5"),
                    (1.5, 2.0, "2.0"), (2.0, 2.5, "2.5"), (2.5, 3.0, "3.0"),
                ):
                    _fm = _fast & (_ddist >= _lo) & (_ddist < _hi)
                    if int(_fm.sum().item()) > 0:
                        self._dyn_fast_speed[_bk].append(v_x[_fm].float().mean().item())
                        self._dyn_fast_omega[_bk].append(omega[_fm].abs().float().mean().item())
                    _sm = _slow & (_ddist >= _lo) & (_ddist < _hi)
                    if int(_sm.sum().item()) > 0:
                        self._dyn_slow_speed[_bk].append(v_x[_sm].float().mean().item())
                        self._dyn_slow_omega[_bk].append(omega[_sm].abs().float().mean().item())

            remaining = self.action_table_sample_size - len(self._action_speed_accel_rows)
            if remaining > 0:
                take = min(remaining, max(1, self.num_envs // 32), self.num_envs)
                idx = torch.randperm(self.num_envs, device=self.device)[:take]
                heading_deg = None
                velocity_to_goal = None
                if goal_diagnostics is not None:
                    heading_deg = goal_diagnostics["heading_error_abs"].detach() * 180.0 / math.pi
                    velocity_to_goal = goal_diagnostics["velocity_to_goal"].detach()
                terminal_code = torch.zeros(self.num_envs, device=self.device)
                if reward_breakdown is not None:
                    terminal_code = torch.where(
                        reward_breakdown["goal_reward"] > 0,
                        torch.ones_like(terminal_code),
                        terminal_code,
                    )
                    terminal_code = torch.where(
                        (reward_breakdown["wall_hit_reward"] < 0) | (reward_breakdown["obs_hit_reward"] < 0),
                        -torch.ones_like(terminal_code),
                        terminal_code,
                    )
                for j in idx.detach().cpu().tolist():
                    self._action_speed_accel_rows.append([
                        float(rollout_step if rollout_step is not None else -1),
                        float(v_x[j].item()),
                        float(a_x[j].item()),
                        float(omega[j].item()),
                        float(lin_idx[j].item()),
                        float(ang_idx[j].item()),
                        float(heading_deg[j].item()) if heading_deg is not None else float("nan"),
                        float(velocity_to_goal[j].item()) if velocity_to_goal is not None else float("nan"),
                        float(terminal_code[j].item()),
                    ])

        # --- Parse info["log"] ---
        if "log" in info:
            for key, val in info["log"].items():
                if key.startswith("Curriculum/"):
                    # IsaacLab CurriculumManager format: "Curriculum/{term_name}/{key}"
                    # Extract the leaf key after the last '/'
                    parts = key.split("/")
                    k = parts[-1] if len(parts) >= 3 else key.replace("Curriculum/", "")
                    try:
                        self._curriculum_info[k] = val.item() if isinstance(val, torch.Tensor) else float(val)
                    except (ValueError, TypeError):
                        pass

                if key.startswith("Episode_Reward/"):
                    k = key.replace("Episode_Reward/", "")
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    self._reward_terms.setdefault(k, []).append(v)

                if key.startswith("Episode_Termination/"):
                    k = key.replace("Episode_Termination/", "")
                    v = val.item() if isinstance(val, torch.Tensor) else float(val)
                    num_done = done.reshape(-1).bool().sum().item()
                    count = max(0, round(v * num_done))
                    if "goal_reached" in k:
                        self._goal_reached += count
                    elif "obstacle_collision" in k:
                        self._obstacle_collision += count
                        self._collision += count
                    elif "wall_collision" in k:
                        self._wall_collision += count
                        self._collision += count
                    elif "collision" in k:  # generic collision
                        self._collision += count
                    elif "time_out" in k:
                        self._timeout += count
                    elif "tipped_over" in k or "physics_explosion" in k:
                        self._tipped_over += count

        # --- Obstacle policy metrics ---
        if obs_actions is not None:
            # Record actual velocity (action * speed_limit), not raw action norm
            actual_vel = obs_actions.reshape(-1, 2) * self._obs_speed_limit
            speed = actual_vel.norm(dim=-1).mean().item()
            self._obs_speeds.append(speed)
        if obs_obs is not None:
            robot_rel = obs_obs[:, :, 4:6]  # [E, N, 2]
            dist = robot_rel.norm(dim=-1).mean().item()
            self._obs_distances.append(dist)

        # --- Episode completion ---
        done_mask = done.reshape(-1).bool()
        if done_mask.any():
            ids = done_mask.nonzero(as_tuple=False).reshape(-1)
            for idx in ids:
                self._completed_rewards.append(self._ep_reward[idx].item())
                self._completed_lengths.append(self._ep_length[idx].item())
                self._completed_alive.append(self._ep_steps_alive[idx].item())
                # WD reward decomposition per completed episode
                self._completed_goal_reward.append(self._ep_goal_reward[idx].item())
                self._completed_wall_hit_reward.append(self._ep_wall_hit_reward[idx].item())
                self._completed_obs_hit_reward.append(self._ep_obs_hit_reward[idx].item())
                self._completed_floor_reward.append(self._ep_floor_reward[idx].item())
                self._completed_action_reward.append(self._ep_action_reward[idx].item())
                if torch.isfinite(self._ep_goal_start_dist[idx]) and torch.isfinite(self._ep_goal_prev_dist[idx]):
                    steps = max(self._ep_goal_diag_steps[idx].item(), 1.0)
                    start_dist = self._ep_goal_start_dist[idx].item()
                    end_dist = self._ep_goal_prev_dist[idx].item()
                    self._completed_goal_start_dist.append(start_dist)
                    self._completed_goal_end_dist.append(end_dist)
                    self._completed_goal_distance_delta.append(start_dist - end_dist)
                    self._completed_goal_progress_mean.append(self._ep_goal_progress_sum[idx].item() / steps)
                    self._completed_velocity_to_goal_mean.append(self._ep_goal_velocity_sum[idx].item() / steps)
                    self._completed_heading_error_abs_mean.append(self._ep_goal_heading_abs_sum[idx].item() / steps)
                    switch_steps = max(steps - 1.0, 1.0)
                    self._completed_target_switch_rate.append(self._ep_goal_switch_count[idx].item() / switch_steps)
                    if self._ep_goal_reward[idx].item() > 0:
                        self._completed_steps_to_goal.append(self._ep_length[idx].item())
                # --- Per-outcome expected value tracking ---
                ep_rwd = self._ep_reward[idx].item()
                ep_len = self._ep_length[idx].item()
                alive_ratio = self._ep_steps_alive[idx].item() / max(ep_len, 1.0)
                self._ev_alive_ratio.append(alive_ratio)
                if reward_breakdown is not None:
                    if reward_breakdown["goal_reached"][idx]:
                        self._ev_reward_goal.append(ep_rwd)
                        self._ev_length_goal.append(ep_len)
                    elif reward_breakdown["wall_collision"][idx] or reward_breakdown["obs_collision"][idx]:
                        self._ev_reward_collision.append(ep_rwd)
                        self._ev_length_collision.append(ep_len)
                    else:
                        self._ev_reward_timeout.append(ep_rwd)
                        self._ev_length_timeout.append(ep_len)

                # Static/dynamic obstacle collision attribution
                if reward_breakdown is not None:
                    if reward_breakdown["static_obs_collision"][idx]:
                        self._static_obstacle_collision += 1
                    if reward_breakdown["dynamic_obs_collision"][idx]:
                        self._dynamic_obstacle_collision += 1

                self._total_eps += 1
                # dies_at_birth: episode ended within 2 steps
                if self._ep_length[idx].item() <= 2:
                    self._first_step_deaths += 1
            self._ep_reward[ids] = 0.0
            self._ep_length[ids] = 0.0
            self._ep_steps_alive[ids] = 0.0
            self._ep_goal_reward[ids] = 0.0
            self._ep_wall_hit_reward[ids] = 0.0
            self._ep_obs_hit_reward[ids] = 0.0
            self._ep_floor_reward[ids] = 0.0
            self._ep_action_reward[ids] = 0.0
            self._ep_goal_start_dist[ids] = float("nan")
            self._ep_goal_prev_dist[ids] = float("nan")
            self._ep_goal_progress_sum[ids] = 0.0
            self._ep_goal_velocity_sum[ids] = 0.0
            self._ep_goal_heading_abs_sum[ids] = 0.0
            self._ep_goal_diag_steps[ids] = 0.0
            self._ep_goal_prev_target_id[ids] = -1
            self._ep_goal_switch_count[ids] = 0.0

    def get_gamma(self):
        """從 curriculum_info 取得當前的 gamma 值（若有的話）。"""
        return self._curriculum_info.get("gamma", None)

    def collect(self) -> dict[str, float]:
        """把本 iteration 收集的所有指標整理成 WandB log dict，並清空 completed lists。

        WandB 指標分組（/前的 prefix）:
          - charge/*:              episode 層級統計（reward, SR, CR, TO, 碰撞分解）
          - goal_diagnostics/*:    目標導向行為診斷（progress, velocity_to_goal, heading error）
          - obstacle/*:            障礙物 agent 指標（速度、距離）
          - charge/reward_term/*:  Isaac Lab 原始 reward term（從 info["log"] 讀取）
          - curriculum/*:          curriculum 狀態（stage）
        """
        m: dict[str, float] = {}
        eps = 1e-8
        te = self._total_eps + eps

        # ==============================================================
        # charge/ — canonical episode-level metrics
        # ==============================================================

        if self._completed_rewards:
            m["reward/episode_mean"] = np.mean(self._completed_rewards)
            m["reward/episode_max"] = np.max(self._completed_rewards)
            m["reward/episode_min"] = np.min(self._completed_rewards)
        if self._completed_lengths:
            m["charge/episode_length_mean"] = np.mean(self._completed_lengths)
        m["charge/total_episodes"] = self._total_eps
        m["charge/goal_reach_rate"] = self._goal_reached / te
        m["charge/hit_probability"] = self._collision / te
        m["charge/timeout_rate"] = self._timeout / te
        m["charge/survival_probability"] = 1.0 - self._collision / te
        m["charge/dies_at_birth_rate"] = self._first_step_deaths / te

        # 碰撞分解: wall vs obstacle (overall + static/dynamic)
        m["charge/wall_collision_rate"] = self._wall_collision / te
        m["charge/obstacle_collision_rate"] = self._obstacle_collision / te
        m["charge/static_obstacle_collision_rate"] = self._static_obstacle_collision / te
        m["charge/dynamic_obstacle_collision_rate"] = self._dynamic_obstacle_collision / te
        total_col = self._collision + eps
        m["charge/VR_wall"] = self._wall_collision / total_col
        m["charge/VR_obstacle"] = self._obstacle_collision / total_col

        # --- Goal-directed behavior diagnostics ---
        if self._completed_goal_start_dist:
            m["goal_diagnostics/start_distance_mean"] = np.mean(self._completed_goal_start_dist)
            m["goal_diagnostics/end_distance_mean"] = np.mean(self._completed_goal_end_dist)
            m["goal_diagnostics/distance_delta_mean"] = np.mean(self._completed_goal_distance_delta)
            m["goal_diagnostics/progress_per_step_mean"] = np.mean(self._completed_goal_progress_mean)
            m["goal_diagnostics/velocity_to_goal_mean"] = np.mean(self._completed_velocity_to_goal_mean)
            m["goal_diagnostics/heading_error_abs_mean_rad"] = np.mean(self._completed_heading_error_abs_mean)
            m["goal_diagnostics/heading_error_abs_mean_deg"] = (
                np.mean(self._completed_heading_error_abs_mean) * 180.0 / math.pi
            )
        if self._completed_target_switch_rate:
            m["goal_diagnostics/target_switch_rate"] = np.mean(self._completed_target_switch_rate)
        if self._completed_goal_reset_dist:
            m["goal_diagnostics/nearest_goal_distance_at_reset"] = np.mean(self._completed_goal_reset_dist)
            m["goal_diagnostics/goal_angle_at_reset_abs_deg"] = np.mean(self._completed_goal_reset_angle_abs)
        if self._completed_steps_to_goal:
            m["goal_diagnostics/steps_to_goal_mean"] = np.mean(self._completed_steps_to_goal)
        if self._completed_goal_blocked_by_wall:
            m["goal_diagnostics/goal_blocked_by_wall_rate"] = np.mean(self._completed_goal_blocked_by_wall)
        m["goal_diagnostics/successes_per_rollout"] = self._goal_reached

        # --- Goal agent metrics ---
        m["goal_diagnostics/reached_count"] = self._goal_reached

        # --- Obstacle agent metrics ---
        m["obstacle/collision_count"] = self._collision
        m["obstacle/collision_rate"] = self._collision / te
        if self._obs_speeds:
            m["obstacle/mean_speed"] = np.mean(self._obs_speeds)
        if self._obs_distances:
            m["obstacle/mean_distance_to_robot"] = np.mean(self._obs_distances)

        # --- Charge action distribution diagnostics ---
        if self._action_speed_x:
            m["charge/action_linear_idx_mean"] = np.mean(self._action_linear_idx)
            m["charge/action_angular_idx_mean"] = np.mean(self._action_angular_idx)
            m["charge/speed_x_mean"] = np.mean(self._action_speed_x)
            m["charge/accel_x_mean"] = np.mean(self._action_accel_x)
            m["charge/omega_mean"] = np.mean(self._action_omega)
            m["charge/speed_x_std_over_steps"] = np.std(self._action_speed_x)
            m["charge/accel_x_std_over_steps"] = np.std(self._action_accel_x)
            m["charge/omega_std_over_steps"] = np.std(self._action_omega)

        # --- v3: Per-env jitter p50/p95 (抽動真相) ---
        # 注意：全 env 平均的 omega_std 會因為「不同 env 抽動相位不同」互相抵消
        # 這裡用 sampled per-env trajectories 算 per-env std/flip-rate 再取 p50/p95
        # → p95 才是抽動嚴重程度的真實指標
        if self._jitter_omega_traj and len(self._jitter_omega_traj[0]) > 1:
            FLIP_THRESH = 0.5  # |Δratio_ang| > 0.5 視為「抽動」(idx 9→13 以上)
            per_env_omega_std = []
            per_env_flip_rate = []
            for omega_traj, ratio_traj in zip(self._jitter_omega_traj, self._jitter_ang_ratio_traj):
                if len(omega_traj) > 1:
                    per_env_omega_std.append(float(np.std(omega_traj)))
                    ratio_arr = np.asarray(ratio_traj)
                    delta = np.abs(np.diff(ratio_arr))
                    per_env_flip_rate.append(float((delta > FLIP_THRESH).mean()))
            if per_env_omega_std:
                m["jitter/per_env_omega_std_p50"] = float(np.percentile(per_env_omega_std, 50))
                m["jitter/per_env_omega_std_p95"] = float(np.percentile(per_env_omega_std, 95))
                m["jitter/per_env_ratio_flip_rate_p50"] = float(np.percentile(per_env_flip_rate, 50))
                m["jitter/per_env_ratio_flip_rate_p95"] = float(np.percentile(per_env_flip_rate, 95))
                m["jitter/n_sample"] = float(len(per_env_omega_std))

        # --- LiDAR-noise impact diagnostics ---
        # lidar_min_step_mean : mean of per-env min(lidar) over the rollout (post-noise).
        # lidar_zero_count_step_mean : avg # of near-zero rays per env per step (hole proxy).
        # speed_x_near_obs_mean : speed_x conditioned on at-least-one-env close to obstacle.
        # near_obs_env_count_mean : how many envs were "near obstacle" per step (denominator).
        if self._lidar_min_step:
            m["charge/lidar_min_step_mean"] = float(np.mean(self._lidar_min_step))
            m["charge/lidar_zero_count_step_mean"] = float(np.mean(self._lidar_zero_count_step))
        if self._lidar_sat_frac_step:
            m["charge/lidar_saturated_frac"] = float(np.mean(self._lidar_sat_frac_step))
        if self._action_speed_x_near_obs:
            m["charge/speed_x_near_obs_mean"] = float(np.mean(self._action_speed_x_near_obs))
            m["charge/near_obs_env_count_mean"] = float(np.mean(self._lidar_near_obs_env_count))
        if self._action_speed_x_front_blocked:
            m["charge/speed_fwd_front_blocked_mean"] = float(np.mean(self._action_speed_x_front_blocked))
            m["charge/front_blocked_env_count_mean"] = float(np.mean(self._front_blocked_env_count))
        # ★速度 vs 前方距離 曲線 (每項=該距離段的平均前進速度;段內無樣本則不記錄)
        for _bk, _bv in self._front_dist_speed.items():
            if _bv:
                m[f"charge/speed_vs_front_dist_{_bk}m"] = float(np.mean(_bv))
        for _bk, _bo in self._front_dist_omega.items():  # ★|ω| vs 前距: 轉向式提早避障訊號
            if _bo:
                m[f"charge/omega_vs_front_dist_{_bk}m"] = float(np.mean(_bo))
        # ★動態專屬反應曲線(按 closing speed): fast=快接近 / slow=慢或遠離。
        #   比對「fast 是否在較遠距離就降速/抬|ω|」= policy 有沒有用 LV-DOT 速度提早避的鐵證。
        for _bk, _bv in self._dyn_fast_speed.items():
            if _bv:
                m[f"charge/dyn_fast_speed_{_bk}m"] = float(np.mean(_bv))
        for _bk, _bv in self._dyn_fast_omega.items():
            if _bv:
                m[f"charge/dyn_fast_omega_{_bk}m"] = float(np.mean(_bv))
        for _bk, _bv in self._dyn_slow_speed.items():
            if _bv:
                m[f"charge/dyn_slow_speed_{_bk}m"] = float(np.mean(_bv))
        for _bk, _bv in self._dyn_slow_omega.items():
            if _bv:
                m[f"charge/dyn_slow_omega_{_bk}m"] = float(np.mean(_bv))

        # --- All Isaac Lab reward terms (raw) ---
        for k, vals in self._reward_terms.items():
            if vals:
                m[f"reward/term/{k}"] = np.mean(vals)

        # --- Curriculum info (only log stage — rest is constant under --fixed_stage) ---
        if "stage" in self._curriculum_info:
            m["curriculum/stage"] = self._curriculum_info["stage"]

        # ==============================================================
        # expect_value/ — WD 期望值 (Expected Value / Hit Probability)
        # ==============================================================
        # WD 論文用 V_Survive / V_Move / V_Spot-Goal 等期望值追蹤訓練進程。
        # 以下為 charge 版本，將 episode reward 按結局分類計算條件期望值。

        # total_expected_value: E[reward] (WD: V_Spot)
        if self._completed_rewards:
            m["expect_value/total_expected_value"] = np.mean(self._completed_rewards)

        # navigate_expected_value: E[reward | goal_reached] (WD: V_Move)
        if self._ev_reward_goal:
            m["expect_value/navigate_expected_value"] = np.mean(self._ev_reward_goal)

        # collision_expected_value: E[reward | collision]
        if self._ev_reward_collision:
            m["expect_value/collision_expected_value"] = np.mean(self._ev_reward_collision)

        # timeout_expected_value: E[reward | timeout]
        if self._ev_reward_timeout:
            m["expect_value/timeout_expected_value"] = np.mean(self._ev_reward_timeout)

        # survive_expected_value: E[alive_ratio] (WD: V_Survive)
        if self._ev_alive_ratio:
            m["expect_value/survive_expected_value"] = np.mean(self._ev_alive_ratio)

        # Hit Probability: 結局機率 (WD: p_Spot-Goal / p_Spot-Obs+Map)
        m["expect_value/goal_hit_probability"] = self._goal_reached / te
        m["expect_value/collision_hit_probability"] = self._collision / te
        m["expect_value/timeout_hit_probability"] = self._timeout / te

        # Collision attribution ratio (WD: VR_Spot-Map / VR_Spot-Obs)
        m["expect_value/wall_collision_ratio"] = self._wall_collision / total_col
        m["expect_value/obs_collision_ratio"] = self._obstacle_collision / total_col
        obs_col = self._obstacle_collision + eps
        m["expect_value/static_obs_collision_ratio"] = self._static_obstacle_collision / obs_col
        m["expect_value/dynamic_obs_collision_ratio"] = self._dynamic_obstacle_collision / obs_col

        # goal_length_expected_value / collision_length_expected_value
        if self._ev_length_goal:
            m["expect_value/goal_length_expected_value"] = np.mean(self._ev_length_goal)
        if self._ev_length_collision:
            m["expect_value/collision_length_expected_value"] = np.mean(self._ev_length_collision)

        # --- reward/ group: reward component decomposition ---
        if self._completed_goal_reward:
            m["reward/goal_reward_expected_value"] = np.mean(self._completed_goal_reward)

        if self._completed_wall_hit_reward and self._completed_obs_hit_reward:
            wall_arr = np.array(self._completed_wall_hit_reward)
            obs_arr = np.array(self._completed_obs_hit_reward)
            m["reward/penalty_expected_value"] = np.mean(wall_arr + obs_arr)

        if self._completed_action_reward:
            action_mean = np.mean(self._completed_action_reward)
            if abs(action_mean) > 1e-6:
                m["reward/action_cost_expected_value"] = action_mean

        return m

    def action_speed_accel_rows(self) -> list[list[float]]:
        return self._action_speed_accel_rows

    def reset(self):
        self._completed_rewards.clear()
        self._completed_lengths.clear()
        self._completed_alive.clear()
        self._completed_goal_reward.clear()
        self._completed_wall_hit_reward.clear()
        self._completed_obs_hit_reward.clear()
        self._completed_floor_reward.clear()
        self._completed_action_reward.clear()
        self._completed_goal_start_dist.clear()
        self._completed_goal_end_dist.clear()
        self._completed_goal_distance_delta.clear()
        self._completed_goal_progress_mean.clear()
        self._completed_velocity_to_goal_mean.clear()
        self._completed_heading_error_abs_mean.clear()
        self._completed_target_switch_rate.clear()
        self._completed_goal_reset_dist.clear()
        self._completed_goal_reset_angle_abs.clear()
        self._completed_steps_to_goal.clear()
        self._completed_goal_blocked_by_wall.clear()
        self._action_linear_idx.clear()
        self._action_angular_idx.clear()
        self._action_speed_x.clear()
        self._action_accel_x.clear()
        self._action_omega.clear()
        self._action_speed_accel_rows.clear()
        # v3: jitter sampling — keep _jitter_sample_ids (stable across rollouts)
        # but clear trajectories so each rollout's p50/p95 is over fresh data
        for traj in self._jitter_omega_traj:
            traj.clear()
        for traj in self._jitter_ang_ratio_traj:
            traj.clear()
        self._lidar_min_step.clear()
        self._lidar_zero_count_step.clear()
        self._lidar_sat_frac_step.clear()
        self._action_speed_x_near_obs.clear()
        self._lidar_near_obs_env_count.clear()
        for _bv in self._front_dist_speed.values():  # ★速度vs前方距離曲線 per-rollout 清空(反映當前 policy)
            _bv.clear()
        for _bo in self._front_dist_omega.values():
            _bo.clear()
        for _dyn_d in (self._dyn_fast_speed, self._dyn_fast_omega,
                       self._dyn_slow_speed, self._dyn_slow_omega):  # ★動態專屬曲線 per-rollout 清空
            for _dyn_v in _dyn_d.values():
                _dyn_v.clear()
        self._goal_reached = 0
        self._collision = 0
        self._wall_collision = 0
        self._obstacle_collision = 0
        self._static_obstacle_collision = 0
        self._dynamic_obstacle_collision = 0
        self._timeout = 0
        self._tipped_over = 0
        self._total_eps = 0
        self._first_step_deaths = 0
        self._reward_terms.clear()
        self._obs_speeds.clear()
        self._obs_distances.clear()
        self._ev_reward_goal.clear()
        self._ev_reward_collision.clear()
        self._ev_reward_timeout.clear()
        self._ev_length_goal.clear()
        self._ev_length_collision.clear()
        self._ev_length_timeout.clear()
        self._ev_alive_ratio.clear()


# ============================================================================
# Main Training
# ============================================================================
# main() 函數是整個訓練的入口，負責以下事項（依序）：
#   1. 設定 random seed（確保可重現性）
#   2. 套用 env config overrides（num_envs, LiDAR noise, curriculum version 等）
#   3. 建立 IsaacLab 環境（gym.make + SkrlVecEnvWrapper）
#   4. 建立模型（Charge: PreprocessRNN + PolicyHead + ValueHead；Obstacle: FC policy/value）
#   5. 設定 Two-Optimizer（charge_opt_rl + charge_opt_aux）
#   6. 建立 RolloutBuffer、MetricsCollector
#   7. 初始化 WandB
#   8. 載入 checkpoint（若有）
#   9. 進入主訓練迴圈（Rollout → RL update → Aux update → Obstacle update → Logging）

@hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    # @hydra_task_config 裝飾器會從 gym 已註冊的 task 取得 env_cfg 和 agent_cfg（yaml）。
    # env_cfg = ManagerBasedRLEnvCfg，包含 scene, observations, rewards, terminations 等。
    # agent_cfg = PPO YAML（skrl 用，但本腳本不使用 skrl 的 trainer，只借用 env wrapper）。

    # --- Seed：確保所有隨機源都固定 ---
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)  # -1 = 隨機 seed（每次都不同）
    torch.manual_seed(args_cli.seed)   # PyTorch GPU/CPU 隨機種子
    np.random.seed(args_cli.seed)      # NumPy 隨機種子
    random.seed(args_cli.seed)         # Python random 模組種子

    # --- Env config overrides：在 gym.make 之前修改 cfg ---
    # ManagerBasedEnv.__init__ 會用 cfg.seed 重設「全域」torch/numpy/random RNG
    # （manager_based_env.py: self.seed(self.cfg.seed)），發生在上面 manual_seed 之後。
    # 不傳遞的話 --seed 會被 env 預設 42 整個洗掉（seed 複驗全變同一條軌跡）。
    env_cfg.seed = args_cli.seed

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs  # 覆蓋 YAML 裡的 num_envs

    if getattr(args_cli, "end_to_end_frame_stack", False):
        # 1024 envs instantiate many hidden obstacle assets. The PhysX default
        # (2**21) reports missed broad-phase interactions, which can corrupt
        # collision labels even at SA1. 2**23 covers the observed 6.0M peak.
        env_cfg.sim.physx.gpu_found_lost_pairs_capacity = max(
            env_cfg.sim.physx.gpu_found_lost_pairs_capacity, 2**23
        )
        print("[E2E-PPO] PhysX gpu_found_lost_pairs_capacity=8388608")

    # --room_size: 覆蓋場景物理邊界（外牆位置 + LiDAR boundary 查詢）
    if args_cli.room_size is not None:
        _rs = args_cli.room_size
        _wt = 1.0  # wall_thickness
        _wl = _rs * 2 + _wt  # wall_length
        _wh = 3.0  # wall_height (match VLP16 visibility)

        # 1) 覆蓋 scene cfg 物理牆位置
        scene = env_cfg.scene
        for wall_attr, pos in [
            ("wall_north", (0.0, _rs, _wh / 2)),
            ("wall_south", (0.0, -_rs, _wh / 2)),
            ("wall_east", (_rs, 0.0, _wh / 2)),
            ("wall_west", (-_rs, 0.0, _wh / 2)),
        ]:
            wall = getattr(scene, wall_attr, None)
            if wall is not None:
                wall.init_state.pos = pos
                # 更新牆壁尺寸
                if "North" in wall_attr or "South" in wall_attr or "north" in wall_attr or "south" in wall_attr:
                    wall.spawn.size = (_wl, _wt, _wh)
                else:
                    wall.spawn.size = (_wt, _wl, _wh)

        # 2) Monkey-patch wall_layout module-level constants (用於 LiDAR/collision 查詢)
        import isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.wall_layout as _wl_mod
        _wl_mod.BOUNDARY_WALLS_20x20 = [
            (0.0,  _rs, _wl, _wt),   # North
            (0.0, -_rs, _wl, _wt),   # South
            (_rs,  0.0, _wt, _wl),   # East
            (-_rs, 0.0, _wt, _wl),   # West
        ]
        # 過濾超出新邊界的 maze walls
        _margin = _rs - 1.0
        _wl_mod.MAZE_WALLS_20x20 = [
            w for w in _wl_mod.MAZE_WALLS_20x20
            if abs(w[0]) + w[2] / 2 < _margin and abs(w[1]) + w[3] / 2 < _margin
        ]
        _wl_mod.ALL_WALLS_20x20 = _wl_mod.MAZE_WALLS_20x20 + _wl_mod.BOUNDARY_WALLS_20x20

        # 3) 自動調整 scene_bound_base（障礙物移動範圍 < 外牆）
        if not any(a.startswith("--scene_bound_base") for a in sys.argv):
            args_cli.scene_bound_base = max(2.0, _rs - 2.0)

        # 4) ★修 spawn 縮放 (07-09): room_size 原本只縮牆/LiDAR/障礙,漏縮
        #    「機器人 spawn 範圍(pose_range ±5)」+「目標放置邊界(wall_boundary 7.5-9.5)」
        #    → 縮小 arena 時 spawn/goal 跑到新牆(±rs)外 → dies_at_birth(出生即撞死)。
        #    修: robot pose_range ±(rs-2), goal wall_boundary ±(rs-1.5)(牆內安全邊距)。
        _spawn_lim = max(1.5, _rs - 2.0)
        _goal_bound = max(1.5, _rs - 1.5)
        try:
            _rb = env_cfg.events.reset_base.params["pose_range"]
            _rb["x"] = (-_spawn_lim, _spawn_lim)
            _rb["y"] = (-_spawn_lim, _spawn_lim)
        except Exception as _e:
            print(f"[WARN] room_size: 無法縮 robot pose_range: {_e}")
        try:
            env_cfg.commands.goal_command.wall_boundary = (_goal_bound, _goal_bound)
        except Exception as _e:
            print(f"[WARN] room_size: 無法縮 goal wall_boundary: {_e}")

        print(f"[INFO] room_size={_rs} → scene {_rs*2}×{_rs*2}m, "
              f"boundary_walls at ±{_rs}, scene_bound_base={args_cli.scene_bound_base}, "
              f"robot spawn ±{_spawn_lim:.1f}, goal_boundary ±{_goal_bound:.1f}")

    # --- Sim-to-Real DR：LiDAR 噪聲 + Physics/Disturbance/Actuator 域隨機化 ---
    # 統一由 charge_env_overrides 處理，YAML 設定經 ExperimentConfig → args_cli 傳入。
    # lidar_no_noise=True → 全部歸零（legacy）；False → 使用 YAML 中的 per-param 值。
    from charge_env_overrides import (
        _apply_obb_collision_config,
        _apply_lidar_noise_config,
        _apply_dr_param_overrides,
    )
    _apply_obb_collision_config(env_cfg, args_cli)
    _apply_lidar_noise_config(env_cfg, args_cli)
    _apply_dr_param_overrides(env_cfg, args_cli)

    # Fixed SA5 general-scene replay for SA6+. No extra assets are needed:
    # selected envs receive the accepted SA5 obstacle and internal-wall layout.
    _previous_stage_fraction = float(
        getattr(args_cli, "previous_stage_replay_fraction", 0.0)
    )
    if _previous_stage_fraction > 0.0:
        if not (0.0 < _previous_stage_fraction <= 0.60):
            raise ValueError(
                "previous_stage_replay_fraction must stay in (0, 0.60], got "
                f"{_previous_stage_fraction}"
            )
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.events.previous_stage_replay import (
            configure_previous_stage_replay,
        )

        configure_previous_stage_replay(
            env_cfg,
            fraction=_previous_stage_fraction,
            static_obstacles=int(
                getattr(
                    args_cli,
                    "previous_stage_replay_static_obstacles",
                    10,
                )
            ),
            dynamic_obstacles=int(
                getattr(
                    args_cli,
                    "previous_stage_replay_dynamic_obstacles",
                    3,
                )
            ),
            min_walls=int(
                getattr(args_cli, "previous_stage_replay_min_walls", 2)
            ),
            max_walls=int(
                getattr(args_cli, "previous_stage_replay_max_walls", 3)
            ),
            wall_length=float(
                getattr(
                    args_cli,
                    "previous_stage_replay_wall_length",
                    4.0,
                )
            ),
            obstacle_boundary=float(
                getattr(
                    args_cli,
                    "previous_stage_replay_obstacle_boundary",
                    5.5,
                )
            ),
        )

    # Deployment corridor replay: dedicated 10 m walls and a controlled 4S+2D
    # obstacle layout. This is independent of the legacy near-wall crossing event.
    _long_corridor_fraction = float(
        getattr(args_cli, "long_corridor_fraction", 0.0)
    )
    if _long_corridor_fraction > 0.0:
        if not (0.0 < _long_corridor_fraction <= 0.20):
            raise ValueError(
                "long_corridor_fraction must stay in (0, 0.20], got "
                f"{_long_corridor_fraction}"
            )
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.events.long_corridor_replay import (
            configure_long_corridor_assets,
        )

        configure_long_corridor_assets(
            env_cfg,
            fraction=_long_corridor_fraction,
            free_width=float(
                getattr(args_cli, "long_corridor_free_width", 4.0)
            ),
            length=float(getattr(args_cli, "long_corridor_length", 10.0)),
            static_obstacles=int(
                getattr(args_cli, "long_corridor_static_obstacles", 4)
            ),
            dynamic_obstacles=int(
                getattr(args_cli, "long_corridor_dynamic_obstacles", 2)
            ),
            dynamic_speed_range=tuple(
                getattr(
                    args_cli,
                    "long_corridor_dynamic_speed_range",
                    (0.30, 0.60),
                )
            ),
            dynamic_motion_mode=str(
                getattr(
                    args_cli,
                    "long_corridor_dynamic_motion_mode",
                    "lateral",
                )
            ),
            random_2d_kinematics=str(
                getattr(
                    args_cli,
                    "long_corridor_random_2d_kinematics",
                    "patrol",
                )
            ),
            dynamic_motion_weights=getattr(
                args_cli,
                "long_corridor_dynamic_motion_weights",
                None,
            ),
            obstacle_count_mix=getattr(
                args_cli,
                "long_corridor_obstacle_count_mix",
                None,
            ),
        )

    # SA5 narrow-passage bridge: add two dedicated wall assets before gym.make
    # and enable the last reset event. Reward, network, PPO and DR stay untouched.
    _narrow_fraction = float(getattr(args_cli, "narrow_passage_fraction", 0.0))
    if _narrow_fraction > 0.0:
        if not (0.10 <= _narrow_fraction <= 0.15):
            raise ValueError(
                "narrow_passage_fraction must stay in [0.10, 0.15] for the "
                f"accepted SA5 bridge, got {_narrow_fraction}"
            )
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.mdp.events.narrow_passage_bridge import (
            configure_narrow_passage_assets,
        )

        configure_narrow_passage_assets(
            env_cfg,
            fraction=_narrow_fraction,
            schedule_steps=int(getattr(args_cli, "narrow_passage_schedule_steps", 19200)),
            room_half_extent=float(args_cli.room_size),
            segment_length=float(getattr(args_cli, "narrow_passage_segment_length", 9.0)),
            final_stress_ratio=float(
                getattr(args_cli, "narrow_passage_final_stress_ratio", 0.25)
            ),
            fixed_width_range=getattr(
                args_cli, "narrow_passage_fixed_width_range", None
            ),
            fixed_yaw_limit_deg=getattr(
                args_cli, "narrow_passage_fixed_yaw_limit_deg", None
            ),
            exact_width=getattr(
                args_cli, "narrow_passage_exact_width", None
            ),
            exact_width_ratio=float(
                getattr(args_cli, "narrow_passage_exact_width_ratio", 0.0)
            ),
            goal_distance_range=getattr(
                args_cli, "narrow_passage_goal_distance_range", None
            ),
            goal_lateral_offset_range=getattr(
                args_cli, "narrow_passage_goal_lateral_offset_range", None
            ),
        )
    _replay_fraction_total = (
        _previous_stage_fraction
        + _narrow_fraction
        + _long_corridor_fraction
    )
    if _replay_fraction_total >= 1.0:
        raise ValueError(
            "scene replay fractions leave no current-stage native envs"
        )
    if _replay_fraction_total > 0.0:
        print(
            "[SCENE-MIX] "
            f"native={1.0 - _replay_fraction_total:.3f} "
            f"sa5_general={_previous_stage_fraction:.3f} "
            f"narrow={_narrow_fraction:.3f} "
            f"long_corridor={_long_corridor_fraction:.3f} "
            "classes_disjoint=True",
            flush=True,
        )

    # --- Curriculum version：設定 goal_obstacle_curriculum 的版本與起始階段 ---
    # curriculum_version 決定用哪組 phase config（warp_drive_single_agent_v1 等）
    # initial_stage 決定從哪個 phase 開始（1 = 最簡單，通常 5 = 最難）
    # fixed_stage = True 時不會自動升/降級，用於定階段消融實驗
    cv = args_cli.curriculum_version
    cur = getattr(env_cfg, 'curriculum', None)
    if cur is not None:
        term = getattr(cur, 'goal_obstacle_curriculum', None)
        if term is not None:
            term.params["curriculum_version"] = cv
            term.params["initial_stage"] = args_cli.initial_stage
            term.params["fixed_stage"] = args_cli.fixed_stage
            print(
                f"[INFO] Curriculum version: {cv}, initial_stage: {args_cli.initial_stage}, "
                f"fixed_stage={args_cli.fixed_stage}"
            )

    # --- Create env：建立 IsaacLab 環境 ---
    # gym.make → ManagerBasedRLEnv（多 env、並行 sim、Isaac Sim 後端）
    # SkrlVecEnvWrapper → 包成 skrl/PyTorch 相容介面（提供 env.num_envs, env.device 等屬性）
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    num_envs = env.num_envs   # 實際並行 env 數（可能與 cfg 設定略有不同）
    device = env.device        # GPU device（Isaac Sim 固定在 CUDA:0）

    # --room_size: 設定 _room_boundary 供 curriculum + BehaviorScheduler 使用
    if args_cli.room_size is not None:
        _env_unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env
        _env_unwrapped._room_boundary = args_cli.room_size - 1.5  # 牆內安全邊距

    # --- Obstacle mode setup：設定障礙物控制模式 ---
    # 優先順序: CLI 明確指定 > phase config (GLOBAL) > CLI default
    # learned: 使用 learned FC policy（ObstaclePolicyFC + ObstacleValueFC）
    # rule_based: 使用 BehaviorScheduler（deterministic 行為，無 neural network）
    # scripted: 使用環境內建的 move_obstacles_vectorized（目標導向移動）
    _cli_explicitly_set = any(a.startswith("--obstacle_mode") for a in sys.argv)
    _phase_obstacle_mode = None
    try:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.charge_skrl.curriculum.phases import PHASE_REGISTRY
        _phase_cfg = PHASE_REGISTRY.get(cv, {})
        _phase_obstacle_mode = _phase_cfg.get("obstacle_mode")
    except Exception:
        pass
    # CLI 明確指定時用 CLI；否則用 phase config；再 fallback CLI default
    if _cli_explicitly_set:
        _obstacle_mode = args_cli.obstacle_mode
        print(f"[INFO] Obstacle mode: {_obstacle_mode} (CLI override)")
    elif _phase_obstacle_mode is not None:
        _obstacle_mode = _phase_obstacle_mode
        print(f"[INFO] Obstacle mode: {_obstacle_mode} (from phase config GLOBAL)")
    else:
        _obstacle_mode = args_cli.obstacle_mode
        print(f"[INFO] Obstacle mode: {_obstacle_mode} (CLI default)")
    if _obstacle_mode == "learned":
        env.unwrapped._obstacle_policy_active = True   # 通知 env：外部 policy 控制障礙物
        print(f"[INFO] Obstacle mode: LEARNED — FC policy controls obstacles")
    elif _obstacle_mode == "rule_based":
        # BehaviorScheduler 由 curriculum _apply_stage 自動建立
        # 不設 _obstacle_policy_active → scripted motion 也不跑（scheduler guard 優先）
        env.unwrapped._obstacle_policy_active = False
        print(f"[INFO] Obstacle mode: RULE_BASED — BehaviorScheduler controls obstacles")
    else:  # scripted
        env.unwrapped._obstacle_policy_active = False
        print(f"[INFO] Obstacle mode: SCRIPTED — move_obstacles_vectorized controls obstacles")

    obs_dim = env.observation_space.shape[-1]  # 139D 觀測維度（Charge）
    N_obs = args_cli.max_active_obstacles        # 最大同時活動障礙物數量

    # --- Obstacle size + scene bound randomization（對應 WD: obs_size_rand, floor_width_bias）---
    # WD 原版在每個 episode 重置時，隨機化障礙物半徑（agent_size）和場景大小（floor_width）。
    # IsaacLab 版本：
    #   _obstacle_radii: [E, N] per-env per-obstacle collision radius（幾何碰撞判定距離）
    #   _scene_bounds:   [E] per-env 障礙物活動範圍半徑（越大場景越空曠）
    # 這兩個 tensor 被寫入 env.unwrapped，讓 obs_functions.py / 本腳本的 geofence 都能讀取。
    # 如果 CLI 設定 obs_size_rand=0，則從 curriculum 自動同步（per-phase 遞增）。
    _obs_size_rand = args_cli.obs_size_rand  # 0 = auto from curriculum
    _scene_bound_rand = args_cli.scene_bound_rand  # 0 = auto from curriculum
    _obs_collision_base = args_cli.obs_collision_base   # 碰撞半徑基準值（m）
    _scene_bound_base = args_cli.scene_bound_base       # 活動邊界基準值（m）

    # LV-DOT lineage 碰撞語義: 障礙「邊緣」離 LiDAR/車中心 < 0.5m 判碰撞 (對齊 r_min 盲區)。
    #   → 碰撞中心距門檻 = 物理半徑 + 0.5。物理半徑隨機化 (尺寸 DR)，碰撞跟著變。
    #   非 LV-DOT 血緣維持原語義 (_obstacle_radii = obs_collision_base ± rand)。
    _LVDOT_MODE = os.environ.get("CHARGE_USE_LVDOT_OBS", "0") != "0"
    _LVDOT_COLLISION_EDGE = 0.5   # 中心→障礙邊緣碰撞距 (= r_min 盲區，兩者對齊)
    _LVDOT_PHYS_BASE = 0.3        # 障礙物理半徑基準 (尺寸隨機化中心)

    # 初始化所有 env 的 radii/bounds 為基準值（後續每次 episode reset 重新隨機化）
    env.unwrapped._obstacle_phys_radii = torch.full(
        (num_envs, N_obs), _LVDOT_PHYS_BASE, device=device)   # per-env 物理半徑 (obs r 欄 + 碰撞用)
    if _LVDOT_MODE:
        env.unwrapped._obstacle_radii = env.unwrapped._obstacle_phys_radii + _LVDOT_COLLISION_EDGE
    else:
        env.unwrapped._obstacle_radii = torch.full(
            (num_envs, N_obs), _obs_collision_base, device=device)
    env.unwrapped._scene_bounds = torch.full(
        (num_envs,), _scene_bound_base, device=device)

    def _randomize_obstacle_sizes(env_ids: torch.Tensor, rand_range: float):
        """對指定 env_ids 重新隨機化障礙物尺寸/碰撞半徑。
        LV-DOT: 隨機化物理半徑 (base±rand/2)，碰撞門檻 = 物理半徑 + 0.5 (中心→邊緣)。
        非 LV-DOT: 直接隨機化碰撞距離 (WD: agent_size = rand·obs_size_rand + bias − obs_size_rand/2)。
        """
        if rand_range <= 0.0:
            return
        n = env_ids.shape[0]
        noise = (torch.rand(n, N_obs, device=device) - 0.5) * rand_range
        if _LVDOT_MODE:
            phys = (_LVDOT_PHYS_BASE + noise).clamp(min=0.15)
            env.unwrapped._obstacle_phys_radii[env_ids] = phys
            env.unwrapped._obstacle_radii[env_ids] = phys + _LVDOT_COLLISION_EDGE
        else:
            env.unwrapped._obstacle_radii[env_ids] = (_obs_collision_base + noise).clamp(min=0.3)

    def _randomize_scene_bounds(env_ids: torch.Tensor, rand_range: float):
        """對指定 env_ids 重新隨機化場景活動邊界。
        bound = base ± rand/2，最小 4.0m（避免場景過小）
        """
        if rand_range <= 0.0:
            return
        n = env_ids.shape[0]
        noise = (torch.rand(n, device=device) - 0.5) * rand_range
        env.unwrapped._scene_bounds[env_ids] = (_scene_bound_base + noise).clamp(min=4.0)

    # Initial randomization
    all_ids = torch.arange(num_envs, device=device)
    _randomize_obstacle_sizes(all_ids, _obs_size_rand)
    _randomize_scene_bounds(all_ids, _scene_bound_rand)

    print(f"[INFO] Randomization: obs_size_rand={_obs_size_rand} (base={_obs_collision_base}), "
          f"scene_bound_rand={_scene_bound_rand} (base={_scene_bound_base})")
    print(f"[INFO] Env: {args_cli.task}, {num_envs} envs, device={device}")
    print(f"[INFO] obs_dim={obs_dim}, max_active_obstacles={N_obs}")

    # --- Charge policy / WD-like obs path ---
    # IsaacLab 觀測（2026-05-27 起移除 60D 障礙物，env 觀測 = 79D）：
    #   [0:4]   ego state: accel, vel, omega, radius
    #   [4:6]   goal (x, y)
    #   [6:78]  72-bin LiDAR（5° 解析度）
    #   [78]    time (episode timestep / episode_length)
    #   （舊 139D 佈局的 [78:138] 60D TopK obstacles 已從 PolicyCfg/CriticCfg 移除）
    #
    # 三種 obs 路徑（由 --charge_encoder_mode 選擇）：
    #   extractor_rnn: 79D → Conv1d LiDAR extractor → FC → RNN → 12D → RL（SA1_v2 用此）
    #   raw_fc_rnn:    79D → FC → RNN → 12D → RL（更簡單）
    #   wd_exact_rnn:  ⚠️ 需要 60D 障礙物重組 113D，已不相容 79D env（v2 不使用）
    # obs_dim 由 env.observation_space 動態取得（79D）；POLICY_OBS_INDICES 選全部 79D。
    OBS_DIM_RUNTIME = env.observation_space.shape[-1]   # 79D（移除 60D 後）
    if OBS_DIM_RUNTIME >= 139:
        POLICY_OBS_INDICES = list(range(0, 78)) + [138]   # 舊 139D 佈局相容（fallback）
    else:
        POLICY_OBS_INDICES = list(range(0, OBS_DIM_RUNTIME))   # 79D：全選（time 在 78）
    _policy_obs_idx = torch.tensor(POLICY_OBS_INDICES, dtype=torch.long, device=device)
    wd_exact_mode = (args_cli.charge_encoder_mode == "wd_exact_rnn")   # 是否使用 WD 精確模式
    use_extractor = (args_cli.charge_encoder_mode == "extractor_rnn")  # 是否使用 Conv1d extractor
    _e2e_frame_stack = bool(getattr(args_cli, "end_to_end_frame_stack", False))
    if _e2e_frame_stack:
        if not use_extractor:
            raise ValueError("--end_to_end_frame_stack requires charge_encoder_mode=extractor_rnn")
        if args_cli.lidar_frame_stack < 2:
            raise ValueError("--end_to_end_frame_stack requires --lidar_frame_stack >= 2")
        if args_cli.use_a2c:
            raise ValueError("--end_to_end_frame_stack requires PPO, not A2C")
        if not args_cli.disable_aux_training:
            raise ValueError("--end_to_end_frame_stack requires aux_profile=none")
        if getattr(args_cli, "critic_profile", "symmetric") != "symmetric":
            raise ValueError("The clean end-to-end baseline requires a symmetric critic")
        if getattr(args_cli, "feat_norm", False):
            raise ValueError("The clean end-to-end baseline uses extractor LayerNorm; disable --feat_norm")
        print(f"[E2E-PPO] enabled: {args_cli.lidar_frame_stack}-frame LiDAR CNN receives policy/value gradients")

    if wd_exact_mode:
        _wd_overrides = []
        if args_cli.fc_dim != 64:
            _wd_overrides.append(f"fc_dim {args_cli.fc_dim}->64")
            args_cli.fc_dim = 64
        if args_cli.hidden_dim != 30:
            _wd_overrides.append(f"hidden_dim {args_cli.hidden_dim}->30")
            args_cli.hidden_dim = 30
        if args_cli.preprocess_dim != 12:
            _wd_overrides.append(f"preprocess_dim {args_cli.preprocess_dim}->12")
            args_cli.preprocess_dim = 12
        if args_cli.rnn_type != "RNN":
            _wd_overrides.append(f"rnn_type {args_cli.rnn_type}->RNN")
            args_cli.rnn_type = "RNN"
        if args_cli.wd_middle_dim <= 0:
            _wd_overrides.append(f"wd_middle_dim {args_cli.wd_middle_dim}->32")
            args_cli.wd_middle_dim = 32
        # WD: all preprocess params share rnn_lr (optimizers_module registers all with one LR)
        # Force aux_lr_* = rnn_lr to match WD behavior (no selective freezing)
        _wd_rnn_lr = args_cli.rnn_lr
        for _attr, _name in [("aux_lr_fc_front", "fc_front"),
                             ("aux_lr_fc_middle", "fc_middle"),
                             ("aux_lr_predict_head", "predict_head")]:
            _cur = getattr(args_cli, _attr)
            if _cur != _wd_rnn_lr:
                _wd_overrides.append(f"{_attr} {_cur}->{_wd_rnn_lr}")
                setattr(args_cli, _attr, _wd_rnn_lr)
        # wd_exact_rnn 只代表 encoder/obs-layout 對齊 WD，不應覆蓋 algorithm。
        # algorithm 應由 --experiment_config 的 algorithm 欄位或 CLI --use_a2c/--use_ppo 決定。
        # A2C 仍會使用 WD-order Aux→FreshForward→RL；PPO 則保留 PPO 所需的 RL→Aux order。
        if _wd_overrides:
            print(f"[INFO] wd_exact_rnn forced encoder params: {', '.join(_wd_overrides)}")

    def _legacy_policy_obs(obs_normed: torch.Tensor) -> torch.Tensor:
        return obs_normed.index_select(-1, _policy_obs_idx)

    def _wd_like_obs(obs_normed: torch.Tensor) -> torch.Tensor:
        """Project IsaacLab 139D obs to WD car obs layout (113D) with correct semantics.

        WD 113D layout (train_rnn_car.py):
          base[17D]:
            [0]     speed_x (= forward velocity / max_speed)
            [1]     speed_y (= 0, car mode 無側移)
            [2:4]   goal (x, y) body-frame relative
            [4:7]   goal_z, goal_dir, goal_speed (= 0, car mode)
            [7:15]  terrain heights (= 0, car mode)
            [15]    timestep / episode_length
            [16]    in_game (always 1.0)
          obstacles[60D]: 5 features × 10 neighbors + 10 agent_size
          lidar[36D]:     360°/10° = 36 angular bins

        IsaacLab 139D layout:
            [0]     accel (m/s²)          ← WD 沒有
            [1]     vel (m/s)             ← 對應 WD speed_x
            [2]     omega (rad/s)         ← WD 沒有
            [3]     radius (m)            ← WD 沒有
            [4:6]   goal (x, y)           ← 對應 WD base[2:4]
            [6:78]  LiDAR 72 bins (5°)    ← mean-pool → 36 bins
            [78:138] obstacles 60D        ← 對應 WD obstacles
            [138]   time                  ← 對應 WD base[15]

        語義差異 (相比 WD):
          - WD base[0] = speed_x only; IL 有 accel/omega/radius 但 WD 沒有
          - WD 的 60D obstacles 多為 placeholder (spot_state_obs_agent_rate=0);
            IL 的 60D 是 TopK 實際障礙物資料（更有效）
          - LiDAR: WD 10°/bin=36; IL 5°/bin=72 → mean-pool pairs
        """
        leading = obs_normed.shape[:-1]
        base = torch.zeros(*leading, 17, dtype=obs_normed.dtype, device=obs_normed.device)
        # WD base[0] = speed_x: IL obs[1] = vel (m/s), 語義最接近
        base[..., 0] = obs_normed[..., 1]
        # WD base[1] = speed_y = 0 (car mode, 已為 zeros)
        # WD base[2:4] = goal (x, y): IL obs[4:6]
        base[..., 2:4] = obs_normed[..., 4:6]
        # WD base[4:15] = zeros (goal_z/dir/speed + terrain, all 0 in car mode)
        # WD base[15] = timestep / episode_length: IL obs[138]
        base[..., 15] = obs_normed[..., 138]
        # WD base[16] = in_game = 1.0
        base[..., 16] = 1.0
        # obstacles 60D (IL TopK data, WD is mostly placeholder)
        obstacles = obs_normed[..., 78:138]
        # LiDAR: 72 bins (5°) → mean-pool → 36 bins (10°, WD resolution)
        lidar72 = obs_normed[..., 6:78]
        lidar36 = lidar72.reshape(*leading, 36, 2).mean(dim=-1)
        return torch.cat([base, obstacles, lidar36], dim=-1)

    def _select_79d(obs_normed: torch.Tensor) -> torch.Tensor:
        """從 139D env obs 中選取部署可用的 79D: ego(4) + goal(2) + LiDAR(72) + time(1)。
        跳過 obstacles 60D（index 78:138），因為部署不可用且 WD 也是 placeholder。"""
        return obs_normed.index_select(-1, _policy_obs_idx)

    def _charge_obs_for_rl(obs_normed: torch.Tensor) -> torch.Tensor:
        if wd_exact_mode:
            return _wd_like_obs(obs_normed)
        return _select_79d(obs_normed)

    def _build_extractor_input(
        obs_normed: torch.Tensor,
        lidar_hist: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build the exact normalized current+history tensor consumed by the CNN."""
        ext_in = _select_79d(obs_normed)
        if args_cli.lidar_frame_stack <= 1:
            return ext_in
        if lidar_hist is None:
            cur_lidar = ext_in[:, LIDAR_START:LIDAR_END]
            lidar_hist = cur_lidar.repeat(1, args_cli.lidar_frame_stack - 1)
        return torch.cat([ext_in, lidar_hist], dim=-1)

    def _charge_features_for_rnn(obs_normed: torch.Tensor, update_norm: bool = False,
                                 lidar_hist: torch.Tensor = None) -> torch.Tensor:
        if use_extractor:
            if args_cli.lidar_frame_stack > 1 and lidar_hist is None:
                # 多幀:把 (K-1) 幀歷史 LiDAR 附在 extractor 輸入尾端。重要路徑(rollout/recompute/aux)
                #   傳真歷史;未傳則 fallback 複製當前幀(無運動)防 crash。
                if lidar_hist is None:
                    # 2026-07-03: fallback 從 silent 變 loud — 假歷史(複製當前幀=無運動資訊)
                    # 曾讓 aux_epochs/CPC/rnn_rl_grad/use_ppo-aux 多條路徑靜默用假幀(審計 C1-C4)。
                    if not getattr(_charge_features_for_rnn, "_fake_hist_warned", False):
                        _charge_features_for_rnn._fake_hist_warned = True
                        import traceback
                        print("[WARN][frame_stack] _charge_features_for_rnn 未傳 lidar_hist → "
                              "fallback 複製當前幀(無運動資訊)。呼叫點:")
                        traceback.print_stack(limit=4)
            _ext_in = _build_extractor_input(obs_normed, lidar_hist)
            _f = extractor(_ext_in)
        elif wd_exact_mode:
            _f = _wd_like_obs(obs_normed)
        else:
            _f = _select_79d(obs_normed)
        # ★feat_norm:extractor 輸出 per-dim 正規化再進 RNN(各維尺度不一會讓 RNN 學不動;
        #   離線診斷:raw 78% vs per-dim 標準化 31% — 這是 RL aux 學不會編碼的真正缺件)。
        if getattr(args_cli, "feat_norm", False):
            if update_norm:
                feat_normalizer.update(_f.detach().reshape(-1, _f.shape[-1]))
            _f = feat_normalizer.normalize(_f)
        return _f

    policy_obs_dim = 113 if wd_exact_mode else len(POLICY_OBS_INDICES)

    # --- Build Charge models（根據 --charge_encoder_mode 建立對應架構）---
    # ModularRNN 架構（WD 原版 §4.2）：
    #   Input → Extractor(optional) → fc_front → RNN → fc_middle → predict_head(aux)
    #                                                 → preprocess_feat(12D)
    #   RL head: concat(policy_obs, preprocess_feat) → PolicyHead / ValueHead
    if use_extractor:
        # v3d: act_hist_dropout > 0 時，訓練端對 4D 動作歷史隨機 mask（斷 copy shortcut）
        _act_hist_dropout = float(getattr(args_cli, "act_hist_dropout", 0.0))
        # v3f: CHARGE_USE_ACT_HIST=0 → 移除 act_hist（含 action_error），obs 79D、state 7D。
        #      必須與 env cfg 同一 env var 一致（env 端也用 CHARGE_USE_ACT_HIST 控制 obs 維度）。
        _use_act_hist = os.environ.get("CHARGE_USE_ACT_HIST", "1") != "0"
        extractor = LidarStateExtractor(
            include_act_hist=_use_act_hist,
            act_hist_dropout=(_act_hist_dropout if _use_act_hist else 0.0),
            frame_stack=args_cli.lidar_frame_stack,   # 多幀 LiDAR (K=1=現狀)
        ).to(device)   # Conv1d LiDAR + MLP obs 特徵抽取器
        if args_cli.lidar_frame_stack > 1:
            print(f"[INFO] 多幀 LiDAR: frame_stack={args_cli.lidar_frame_stack} "
                  f"(extractor Conv1d 吃 {args_cli.lidar_frame_stack} 幀;obs 尾端附 "
                  f"{(args_cli.lidar_frame_stack-1)*72}D 歷史;載入較少幀 ckpt 時會保守遷移第一層)")
        if not _use_act_hist:
            print("[v3f] CHARGE_USE_ACT_HIST=0 → act_hist 移除，obs 79D / state 7D（含 action_error 速度落差移除）")
        elif _act_hist_dropout > 0.0:
            print(f"[v3d] act_hist_dropout={_act_hist_dropout} (train-only mask on 4D action history)")
        rnn_input_dim = extractor.output_dim  # 96D（extractor 輸出維度）
    else:
        extractor = None
        rnn_input_dim = policy_obs_dim  # 79D（raw_fc_rnn）或 113D（wd_exact_rnn）
    _predict_dim = args_cli.predict_dim
    _aux_vel_topk = args_cli.aux_velocity_topk
    if _aux_vel_topk > 0 and _predict_dim == 7:
        _predict_dim = 7 + _aux_vel_topk * 2
        print(f"[INFO] aux_velocity_topk={_aux_vel_topk} → predict_dim auto-set to {_predict_dim}")
    _aux_weight = WD_DEFAULT_WEIGHT_13D[:_predict_dim] if _predict_dim > 7 else None
    _middle_dim = args_cli.wd_middle_dim if (wd_exact_mode or args_cli.wd_middle_dim > 0) else None
    preprocess_rnn = PreprocessRNN(
        input_dim=rnn_input_dim, fc_dim=args_cli.fc_dim,
        hidden_dim=args_cli.hidden_dim, preprocess_dim=args_cli.preprocess_dim,
        predict_dim=_predict_dim,
        rnn_type=args_cli.rnn_type,
        middle_dim=_middle_dim,
        aux_skip_input=args_cli.aux_skip_input,
    ).to(device)
    # Hybrid: 把 predict_head 輸出的障礙動態預測接進 policy 輸入(RNN 當顯式 MOT,輸出餵 RL)
    _hybrid_pred_dim = _predict_dim if args_cli.hybrid_predict_to_policy else 0
    if _e2e_frame_stack:
        rl_input_dim = policy_obs_dim + extractor.output_dim
    else:
        rl_input_dim = policy_obs_dim + args_cli.preprocess_dim + _hybrid_pred_dim  # concat(obs, preprocess_feat[, prediction])
    if args_cli.hybrid_predict_to_policy:
        if getattr(args_cli, "rnn_rl_grad", False):
            raise NotImplementedError(
                "--hybrid_predict_to_policy 暫不支援與 --rnn_rl_grad 並用"
                "(PPO recompute 路徑未含 prediction 會造成維度不符);請擇一。")
        print(f"[INFO] Hybrid predict→policy: rl_input += predict_dim({_predict_dim}) → {rl_input_dim}D "
              f"(predict_head 輸出當 policy 顯式輸入;仍由 aux 監督)")
    _use_asymmetric_critic = getattr(args_cli, 'critic_profile', 'symmetric') == 'asymmetric'
    _priv_dim = PRIVILEGED_OBS_DIM if _use_asymmetric_critic else 0
    _oracle_to_policy = getattr(args_cli, 'oracle_obstacles_to_policy', False)
    _policy_priv_dim = PRIVILEGED_OBS_DIM if _oracle_to_policy else 0
    # --- LV-DOT channel encoder (2026-07-12): raw 30D → learned encoded → head ---
    #   LVDOT 佔 rl_input[:, policy_obs_dim-30 : policy_obs_dim] (= obs[79:109] 的位置)。
    #   啟用時 head 輸入維度 rl_input_dim - 30 + encoder_dim(buffer 仍存 raw,encode 在 head 前套用)。
    _lvdot_enc_on = bool(args_cli.use_lvdot_encoder) and _LVDOT_MODE and policy_obs_dim >= 109
    lvdot_encoder = None
    _LVDOT_CH = 30  # K=5 × 6D
    if _lvdot_enc_on:
        lvdot_encoder = LVDOTEncoder(
            in_dim=_LVDOT_CH, hidden_dim=args_cli.lvdot_encoder_hidden,
            out_dim=args_cli.lvdot_encoder_dim, zero_init_out=True).to(device)
        _head_input_dim = rl_input_dim - _LVDOT_CH + args_cli.lvdot_encoder_dim
        print(f"[LVDOT-ENC] ★啟用 LV-DOT encoder: raw {_LVDOT_CH}D → {args_cli.lvdot_encoder_dim}D "
              f"(hidden {args_cli.lvdot_encoder_hidden}, 末層 zero-init, wd={args_cli.lvdot_encoder_wd}). "
              f"head 輸入 {rl_input_dim}→{_head_input_dim}D。")
    else:
        _head_input_dim = rl_input_dim

    def _encode_rl_input(ri: torch.Tensor) -> torch.Tensor:
        """把 rl_input 內的 raw LVDOT(30D)換成 encoder 輸出(encoder_dim);未啟用則原樣返回。"""
        if not _lvdot_enc_on:
            return ri
        L = policy_obs_dim
        base = ri[..., :L - _LVDOT_CH]
        lvdot = ri[..., L - _LVDOT_CH:L]
        rest = ri[..., L:]
        return torch.cat([base, lvdot_encoder(lvdot), rest], dim=-1)

    _corridor_adapter_enabled = bool(
        getattr(args_cli, "corridor_adapter_enabled", False)
    )
    _corridor_adapter_hidden_dim = int(
        getattr(args_cli, "corridor_adapter_hidden_dim", 64)
    )
    _corridor_adapter_gate_loss_weight = float(
        getattr(args_cli, "corridor_adapter_gate_loss_weight", 0.05)
    )
    _corridor_adapter_gate_init_probability = float(
        getattr(args_cli, "corridor_adapter_gate_init_probability", 0.01)
    )
    _corridor_adapter_max_logit_delta = float(
        getattr(args_cli, "corridor_adapter_max_logit_delta", 2.0)
    )
    _corridor_adapter_freeze_base = bool(
        getattr(args_cli, "corridor_adapter_freeze_base", False)
    )
    _corridor_adapter_gate_checkpoint = getattr(
        args_cli, "corridor_adapter_gate_checkpoint", None
    )
    _corridor_adapter_residual_features = str(
        getattr(
            args_cli,
            "corridor_adapter_residual_features",
            "current_obs",
        )
    )
    if _corridor_adapter_enabled:
        if (
            not _e2e_frame_stack
            or int(args_cli.lidar_frame_stack) != 8
        ):
            raise ValueError(
                "corridor adapter currently requires the K8 E2E lineage"
            )
        if policy_obs_dim != 83:
            raise ValueError(
                "corridor adapter requires the deployable current 83D policy obs"
            )
        if _oracle_to_policy or _lvdot_enc_on:
            raise ValueError(
                "corridor adapter does not support oracle/LV-DOT policy inputs"
            )
        if float(getattr(args_cli, "long_corridor_fraction", 0.0)) <= 0.0:
            raise ValueError(
                "corridor adapter requires long_corridor_fraction > 0"
            )
        if _corridor_adapter_hidden_dim <= 0:
            raise ValueError("corridor_adapter_hidden_dim must be positive")
        if _corridor_adapter_gate_loss_weight <= 0.0:
            raise ValueError(
                "corridor_adapter_gate_loss_weight must be positive"
            )
        if not 0.0 < _corridor_adapter_gate_init_probability < 1.0:
            raise ValueError(
                "corridor_adapter_gate_init_probability must be in (0, 1)"
            )
        if _corridor_adapter_max_logit_delta <= 0.0:
            raise ValueError(
                "corridor_adapter_max_logit_delta must be positive"
            )
        if _corridor_adapter_residual_features not in (
            "current_obs",
            "policy_features",
        ):
            raise ValueError(
                "corridor_adapter_residual_features must be current_obs or "
                "policy_features"
            )
        if (
            _corridor_adapter_gate_checkpoint
            and not os.path.isfile(_corridor_adapter_gate_checkpoint)
        ):
            raise FileNotFoundError(
                "corridor adapter gate checkpoint not found: "
                f"{_corridor_adapter_gate_checkpoint}"
            )

    policy_head = PolicyHead(input_dim=_head_input_dim, privileged_dim=_policy_priv_dim).to(device)   # 輸出 19×2=38 logits（雙頭離散）
    corridor_adapter = None
    if _corridor_adapter_enabled:
        _corridor_adapter_residual_input_dim = (
            _head_input_dim
            if _corridor_adapter_residual_features == "policy_features"
            else policy_obs_dim
        )
        corridor_adapter = CorridorResidualAdapter(
            input_dim=policy_obs_dim,
            residual_input_dim=_corridor_adapter_residual_input_dim,
            hidden_dim=_corridor_adapter_hidden_dim,
            gate_init_probability=_corridor_adapter_gate_init_probability,
            max_logit_delta=_corridor_adapter_max_logit_delta,
        ).to(device)
        if _corridor_adapter_gate_checkpoint:
            _gate_state = torch.load(
                _corridor_adapter_gate_checkpoint,
                map_location=device,
                weights_only=True,
            )
            corridor_adapter.gate_net.load_state_dict(_gate_state)
            print(
                "[CORRIDOR-ADAPTER] loaded pretrained observation gate: "
                f"{_corridor_adapter_gate_checkpoint}"
            )
        print(
            "[CORRIDOR-ADAPTER] enabled: "
            f"83D gate + {_corridor_adapter_residual_input_dim}D residual "
            f"-> {_corridor_adapter_hidden_dim} hidden "
            f"gate_bce={_corridor_adapter_gate_loss_weight:g} "
            f"|delta_logit|<={_corridor_adapter_max_logit_delta:g} "
            f"freeze_base={_corridor_adapter_freeze_base}; "
            "residual output is zero-init"
        )
    if _oracle_to_policy:
        if not _use_asymmetric_critic:
            raise ValueError("--oracle_obstacles_to_policy 需要 asymmetric critic "
                             "(共用 privileged_obs 緩衝/recompute 管線)。請用 critic_profile=asymmetric。")
        print(f"[ORACLE] obstacles→policy: PolicyHead 殘差 priv_branch dim={_policy_priv_dim} "
              f"(zero-init → 暖啟動 identity; 診斷用, 非部署)")
    value_head = ValueHead(input_dim=_head_input_dim, privileged_dim=_priv_dim).to(device)
    if _corridor_adapter_enabled and _corridor_adapter_freeze_base:
        policy_head.requires_grad_(False)
        extractor.requires_grad_(False)
        print(
            "[CORRIDOR-ADAPTER] base policy and K8 CNN frozen; "
            "actor updates are structurally limited to the adapter"
        )
    if _use_asymmetric_critic:
        print(f"[INFO] Asymmetric critic: rl_input={_head_input_dim} + privileged={_priv_dim} = {_head_input_dim + _priv_dim}D")
    if args_cli.value_init_bias is not None:
        nn.init.constant_(value_head.net[-1].bias, args_cli.value_init_bias)
        print(f"[INFO] Value head final bias override: {args_cli.value_init_bias}")
    rnn_state = RNNStateManager(num_envs, args_cli.hidden_dim, device)  # 管理每個 env 的 RNN hidden state
    # 多幀 LiDAR:每 env 滾動歷史 buffer [E,(K-1)*72](most-recent-first),與 hidden 同生命週期+同步 reset。
    _K_stack = args_cli.lidar_frame_stack
    _LL = LIDAR_END - LIDAR_START   # 72
    _lidar_hist = (torch.zeros(num_envs, (_K_stack - 1) * _LL, device=device)
                   if _K_stack > 1 else None)
    obs_normalizer = RunningNormalizer(obs_dim, device)                 # 線上均值/方差歸一化
    feat_normalizer = RunningNormalizer(rnn_input_dim, device)          # ★extractor 輸出 per-dim 正規化(feat_norm)

    # WD optimizer structure (custom_trainer.py lines 336-349):
    #   WD 原版: 兩個 optimizer 各包含 ALL params，用 lr=0 控制 freeze:
    #     optimizers[policy] (RL):     rl_params lr=spot_lr, preprocess lr=0, rnn lr=0
    #     optimizers_module[policy]:   rl_params lr=0, preprocess lr=spot_preprocess_model_lr, rnn lr=spot_rnn_model_lr
    #   WD detach (module_connected.py line 573): concat_input = rl_in_.detach()
    #     → RL loss.backward() 不流到 preprocess/RNN（即使 lr>0 也不會更新）
    #
    #   [IsaacLab adaptation — conservative output-side unfreeze]
    #     WD baseline: 只有 RNN cell 更新（preprocess_model_lr=0 for train_rnn_car.py）
    #     IsaacLab 差異: extractor 是 Conv1d+MLP（非 WD 的 FC），random init 品質較差
    #     解凍策略（由 aux target 端向 input 端逐層放開，先解決「輸出映射」再處理「輸入特徵」）:
    #       Phase 1: predict_head(1e-4) + fc_middle(5e-5) — 解凍 aux 輸出端
    #       Phase 2: (未來) fc_front — 若 Phase 1 有效，再放開 RNN 輸入端
    #       Phase 3: (未來) extractor — 最後才動 Conv1d 特徵提取
    #     理由:
    #       - predict_head 凍結在 random init → RNN 被迫用「隨機語言」輸出，aux loss 無法下降
    #       - fc_middle 凍結 → RNN→predict_head 的中間映射也是隨機的
    #       - fc_front 先不動 → 保持 RNN 輸入分佈穩定，避免同時改變兩端
    #       - extractor 先不動 → 隔離「輸入表徵」和「輸出映射」兩個問題
    #
    #   charge_opt_rl:  只含 policy_head + value_head
    #   charge_opt_aux: 5 groups（各自獨立 lr）

    # --- RL optimizer: 只含 policy_head + value_head（WD 兩 optimizer 架構）---
    # WD 原版在 RL optimizer 裡，preprocess/RNN 的 lr=0（等同 freeze）。
    # IsaacLab 版本直接把 RL optimizer 限縮到只包含 policy/value head 的 params，
    # 確保 RL loss.backward() 絕對不更新 preprocess_rnn 或 extractor。
    charge_params_actor = (
        []
        if (_corridor_adapter_enabled and _corridor_adapter_freeze_base)
        else list(policy_head.parameters())
    )
    if _e2e_frame_stack and not (
        _corridor_adapter_enabled and _corridor_adapter_freeze_base
    ):
        # Shared CNN baseline: both policy and value losses flow through this
        # encoder in one optimizer. It is classified with actor params only for
        # existing diagnostics, avoiding duplicate optimizer parameters.
        charge_params_actor += list(extractor.parameters())
    if _corridor_adapter_enabled:
        charge_params_actor += list(corridor_adapter.parameters())
    charge_params_critic = list(value_head.parameters())                  # ValueHead 的所有參數
    charge_params_rl = charge_params_actor + charge_params_critic
    if _lvdot_enc_on:
        # encoder 與 head 同 lr(RL 梯度流過它),獨立 param group 開 weight_decay 正則化
        charge_opt_rl = torch.optim.Adam([
            {"params": charge_params_rl, "lr": args_cli.lr},
            {"params": list(lvdot_encoder.parameters()), "lr": args_cli.lr,
             "weight_decay": args_cli.lvdot_encoder_wd},
        ], eps=1e-5)
        print(f"[LVDOT-ENC] encoder 加入 RL optimizer (lr={args_cli.lr}, wd={args_cli.lvdot_encoder_wd})")
    else:
        charge_opt_rl = torch.optim.Adam(charge_params_rl, lr=args_cli.lr, eps=1e-5)

    # --- Aux optimizer: 細粒度 param groups，各自設 lr ---
    # 5 個 group（依「output → input」方向排列，解凍策略也由 output 端向 input 端逐步放開）：
    #   group 0: rnn_cell      → 核心記憶更新，RNN 主力 lr（rnn_lr=5e-4）
    #   group 1: predict_head  → aux loss 輸出映射（預設 frozen，解凍實驗: 1e-4）
    #   group 2: fc_middle     → RNN → predict_head 中間 FC（預設 frozen，解凍: 5e-5）
    #   group 3: fc_front      → 輸入 → RNN 前 FC（預設 frozen，解凍風險高）
    #   group 4: extractor     → Conv1d+MLP 特徵抽取器（預設 frozen，最後才解凍）
    charge_params_rnn_cell = list(preprocess_rnn.rnn.parameters())
    charge_params_fc_front = list(preprocess_rnn.fc_front.parameters())
    charge_params_fc_middle = list(preprocess_rnn.fc_middle.parameters())
    charge_params_predict_head = list(preprocess_rnn.predict_head.parameters())
    charge_params_extractor = list(extractor.parameters()) if use_extractor else []
    # 所有 aux params（用於 clip_grad_norm_ 的便利合集）
    charge_params_aux = charge_params_extractor + list(preprocess_rnn.parameters())

    # Resolve per-group LR: use fine-grained --aux_lr_* if set, else fall back to --aux_lr
    _lr_predict_head = args_cli.aux_lr_predict_head
    _lr_fc_middle = args_cli.aux_lr_fc_middle
    _lr_fc_front = args_cli.aux_lr_fc_front
    _lr_extractor = args_cli.aux_lr_extractor

    _aux_param_groups = [
        {"params": charge_params_rnn_cell,     "lr": args_cli.rnn_lr},    # group 0: RNN cell
        {"params": charge_params_predict_head, "lr": _lr_predict_head},   # group 1: predict_head
        {"params": charge_params_fc_middle,    "lr": _lr_fc_middle},      # group 2: fc_middle
        {"params": charge_params_fc_front,     "lr": _lr_fc_front},       # group 3: fc_front
    ]
    if use_extractor:
        _aux_param_groups.append(
            {"params": charge_params_extractor, "lr": _lr_extractor},     # group 4: extractor
        )
    charge_opt_aux = torch.optim.Adam(_aux_param_groups, eps=1e-5)

    # --- RNN-via-RL optimizer (--rnn_rl_grad un-detach 驗證) ---
    # 拿掉 detach 後,RL loss 的梯度會流進 rnn_cell+fc_front+fc_middle(產生 12D preprocess_feat
    # 的路徑)。用獨立 optimizer 更新,不碰 extractor/predict_head,以隔離「RL 梯度是否能把
    # 障礙位置壓進 RNN hidden」這個假設。
    charge_opt_rnn_rl = None
    if args_cli.rnn_rl_grad:
        _rnn_rl_params = (charge_params_rnn_cell
                          + charge_params_fc_front
                          + charge_params_fc_middle)
        charge_opt_rnn_rl = torch.optim.Adam(_rnn_rl_params, lr=args_cli.rnn_rl_lr, eps=1e-5)
        print(f"[rnn_rl_grad] ✅ RL 梯度將流進 RNN(rnn_cell+fc_front+fc_middle), "
              f"lr={args_cli.rnn_rl_lr}, params={sum(p.numel() for p in _rnn_rl_params):,}")
        if not args_cli.disable_aux_training:
            print("[rnn_rl_grad] ⚠️ 未配 --disable_aux_training:aux 與 RL 會同時訓 RNN(非乾淨隔離)")

    # --- CPC contrastive aux(--aux_cpc):反常數塌縮 ---
    # proj_q: rnn_out(H) → d;proj_k: pos_target(2) → d。InfoNCE 逼 RNN hidden discriminative 編碼位置。
    # cpc_opt 訓 fc_front + rnn + extractor + proj heads(讓整條 input→hidden 學編碼)。
    cpc_proj_q = cpc_proj_k = charge_opt_cpc = None
    if args_cli.aux_cpc:
        _cd = args_cli.aux_cpc_dim
        cpc_proj_q = nn.Sequential(
            nn.Linear(args_cli.hidden_dim, _cd), nn.ReLU(), nn.Linear(_cd, _cd)).to(device)
        cpc_proj_k = nn.Sequential(
            nn.Linear(2, _cd), nn.ReLU(), nn.Linear(_cd, _cd)).to(device)
        # extractor 只在 aux_lr_extractor>0 時納入 CPC 訓練;=0 時凍結(用已訓練好的 extractor)
        _cpc_train_ext = _lr_extractor > 0 and use_extractor
        _cpc_params = (charge_params_fc_front + charge_params_rnn_cell
                       + (charge_params_extractor if _cpc_train_ext else [])
                       + list(cpc_proj_q.parameters()) + list(cpc_proj_k.parameters()))
        charge_opt_cpc = torch.optim.Adam(_cpc_params, lr=args_cli.aux_cpc_lr, eps=1e-5)
        print(f"[aux_cpc] ✅ CPC InfoNCE 啟用 dim={_cd} τ={args_cli.aux_cpc_temp} lr={args_cli.aux_cpc_lr} "
              f"params={sum(p.numel() for p in _cpc_params):,}(訓 fc_front+rnn+proj"
              f"{'+extractor' if _cpc_train_ext else ',extractor凍結'})")

    # Store initial LR for decay
    for pg in charge_opt_rl.param_groups:
        pg["initial_lr"] = pg["lr"]
    for pg in charge_opt_aux.param_groups:
        pg["initial_lr"] = pg["lr"]

    total_charge_params = sum(p.numel() for p in (
        charge_params_rl if _e2e_frame_stack else charge_params_rl + charge_params_aux
    ))
    _n_aux_groups = len(_aux_param_groups)
    print(f"[INFO] Encoder mode: {args_cli.charge_encoder_mode}")
    print(f"[INFO]   fc_front input dim: {rnn_input_dim}")
    print(f"[INFO]   Using extractor: {use_extractor}")
    if _e2e_frame_stack:
        print(f"[INFO] Charge E2E: {policy_obs_dim}D state/current LiDAR + {extractor.output_dim}D "
              f"frame-stack CNN = {rl_input_dim}D, {total_charge_params:,} trainable params")
        if _corridor_adapter_enabled and _corridor_adapter_freeze_base:
            print(
                f"[INFO] RL optimizer: corridor_adapter+value_head lr={args_cli.lr}; "
                "extractor/policy_head frozen"
            )
        else:
            print(
                f"[INFO] RL optimizer: extractor+policy_head"
                f"{'+corridor_adapter' if _corridor_adapter_enabled else ''}"
                f"+value_head lr={args_cli.lr}"
            )
    else:
        print(f"[INFO] Charge: {policy_obs_dim}D + {args_cli.rnn_type} {args_cli.preprocess_dim}D = "
              f"{rl_input_dim}D, {total_charge_params:,} params")
        print(f"[INFO] RL optimizer: policy_head+value_head lr={args_cli.lr}")
    print(f"[INFO] Grad clip: rl={args_cli.max_grad_norm}, aux={args_cli.aux_grad_clip if args_cli.aux_grad_clip is not None else args_cli.max_grad_norm}")
    print(f"[INFO] Aux optimizer ({_n_aux_groups} groups):")
    print(f"  rnn_cell:     lr={args_cli.rnn_lr}")
    print(f"  predict_head: lr={_lr_predict_head}")
    print(f"  fc_middle:    lr={_lr_fc_middle}")
    print(f"  fc_front:     lr={_lr_fc_front}  {'(frozen)' if _lr_fc_front == 0 else ''}")
    if use_extractor:
        print(f"  extractor:    lr={_lr_extractor}  {'(frozen)' if _lr_extractor == 0 else ''}")
    else:
        print(f"  extractor:    N/A (raw_fc_rnn mode, no extractor)")
    if _predict_dim > 7:
        print(f"[INFO] Velocity aux: predict_dim={_predict_dim}, topk={_aux_vel_topk}, weight={list(_aux_weight)}")

    # --- Build Obstacle models (skip if rule_based or scripted) ---
    total_obs_params = 0
    if _obstacle_mode == "learned":
        obs_policy = ObstaclePolicyFC().to(device)
        obs_value = ObstacleValueFC().to(device)
        obs_params = list(obs_policy.parameters()) + list(obs_value.parameters())
        obs_optimizer = torch.optim.Adam(obs_params, lr=args_cli.obs_lr, eps=1e-5)
        total_obs_params = sum(p.numel() for p in obs_params)
        print(f"[INFO] Obstacle: {OBS_POLICY_OBS_DIM}D obs, 2D act, {total_obs_params:,} params")
        print(f"[INFO] Alternating: train_goal_rate={args_cli.train_goal_rate} "
              f"(charge:{args_cli.train_goal_rate-1}, obstacle:1)")
    else:
        obs_policy = None
        obs_value = None
        obs_optimizer = None
        print(f"[INFO] Obstacle mode={_obstacle_mode}: no obstacle policy/optimizer initialized")

    # --- Training config：計算訓練迴圈的基本超參 ---
    RL = args_cli.rollout_length              # 每次 rollout 收集的步數（T）
    batch_size = num_envs * RL               # PPO batch size = E × T（每 iteration 的總樣本數）
    mini_batch_size = batch_size // args_cli.mini_batches  # 每個 mini-batch 的大小
    total_timesteps = args_cli.timesteps     # 總訓練 frame 數（所有 env 累計）
    num_iterations = total_timesteps // RL   # 總 iteration 數

    _teacher_retention_weight = float(
        getattr(args_cli, "teacher_retention_weight", 0.0)
    )
    _teacher_retention_checkpoint = getattr(
        args_cli, "teacher_retention_checkpoint", None
    )
    _teacher_retention_margin_weight = float(
        getattr(args_cli, "teacher_retention_margin_weight", 0.0)
    )
    _teacher_retention_action_ce_weight = float(
        getattr(args_cli, "teacher_retention_action_ce_weight", 0.0)
    )
    _teacher_retention_argmax_margin = float(
        getattr(args_cli, "teacher_retention_argmax_margin", 0.2)
    )
    _teacher_retention_post_kl_epochs = int(
        getattr(args_cli, "teacher_retention_post_kl_epochs", 0)
    )
    _teacher_retention_post_kl_lr = float(
        getattr(args_cli, "teacher_retention_post_kl_lr", 1e-3)
    )
    _teacher_retention_post_kl_batch_size = int(
        getattr(args_cli, "teacher_retention_post_kl_batch_size", 4096)
    )
    _teacher_retention_post_kl_max_grad_norm = float(
        getattr(
            args_cli,
            "teacher_retention_post_kl_max_grad_norm",
            0.5,
        )
    )
    _teacher_retention_post_margin_weight = float(
        getattr(args_cli, "teacher_retention_post_margin_weight", 0.0)
    )
    _teacher_retention_post_action_ce_weight = float(
        getattr(args_cli, "teacher_retention_post_action_ce_weight", 0.0)
    )
    _teacher_retention_post_policy_head_only = bool(
        getattr(args_cli, "teacher_retention_post_policy_head_only", False)
    )
    _teacher_retention_post_anchor_weight = float(
        getattr(args_cli, "teacher_retention_post_anchor_weight", 0.0)
    )
    _teacher_retention_rollout_override = bool(
        getattr(args_cli, "teacher_retention_rollout_override", False)
    )
    _previous_stage_teacher_checkpoint = getattr(
        args_cli, "previous_stage_teacher_checkpoint", None
    )
    _previous_stage_teacher_retention_weight = float(
        getattr(
            args_cli,
            "previous_stage_teacher_retention_weight",
            0.0,
        )
    )
    _previous_stage_teacher_scope = str(
        getattr(args_cli, "previous_stage_teacher_scope", "previous_stage")
    )
    _corridor_teacher_distill_epochs = int(
        getattr(args_cli, "corridor_teacher_distill_epochs", 0)
    )
    _corridor_teacher_distill_lr = float(
        getattr(args_cli, "corridor_teacher_distill_lr", 5e-4)
    )
    _corridor_teacher_distill_batch_size = int(
        getattr(args_cli, "corridor_teacher_distill_batch_size", 4096)
    )
    _corridor_teacher_distill_max_grad_norm = float(
        getattr(
            args_cli,
            "corridor_teacher_distill_max_grad_norm",
            0.5,
        )
    )
    _corridor_teacher_distill_neighbor_mass = float(
        getattr(
            args_cli,
            "corridor_teacher_distill_neighbor_mass",
            0.20,
        )
    )
    _corridor_teacher_distill_stride = int(
        getattr(args_cli, "corridor_teacher_distill_stride", 2)
    )
    _corridor_teacher_distill_chunk_size = int(
        getattr(args_cli, "corridor_teacher_distill_chunk_size", 32)
    )
    _corridor_teacher_intervention_only = bool(
        getattr(args_cli, "corridor_teacher_intervention_only", False)
    )
    _corridor_teacher_intervention_clearance_m = float(
        getattr(
            args_cli,
            "corridor_teacher_intervention_clearance_m",
            0.20,
        )
    )
    if _teacher_retention_weight < 0.0:
        raise ValueError("teacher_retention_weight must be non-negative")
    if _teacher_retention_margin_weight < 0.0:
        raise ValueError(
            "teacher_retention_margin_weight must be non-negative"
        )
    if _teacher_retention_action_ce_weight < 0.0:
        raise ValueError(
            "teacher_retention_action_ce_weight must be non-negative"
        )
    if _teacher_retention_argmax_margin < 0.0:
        raise ValueError(
            "teacher_retention_argmax_margin must be non-negative"
        )
    if _teacher_retention_post_kl_epochs < 0:
        raise ValueError(
            "teacher_retention_post_kl_epochs must be non-negative"
        )
    if _teacher_retention_post_kl_lr <= 0.0:
        raise ValueError("teacher_retention_post_kl_lr must be positive")
    if _teacher_retention_post_kl_batch_size <= 0:
        raise ValueError(
            "teacher_retention_post_kl_batch_size must be positive"
        )
    if _teacher_retention_post_kl_max_grad_norm <= 0.0:
        raise ValueError(
            "teacher_retention_post_kl_max_grad_norm must be positive"
        )
    if _teacher_retention_post_margin_weight < 0.0:
        raise ValueError(
            "teacher_retention_post_margin_weight must be non-negative"
        )
    if _teacher_retention_post_action_ce_weight < 0.0:
        raise ValueError(
            "teacher_retention_post_action_ce_weight must be non-negative"
        )
    if _teacher_retention_post_anchor_weight < 0.0:
        raise ValueError(
            "teacher_retention_post_anchor_weight must be non-negative"
        )
    if _previous_stage_teacher_retention_weight < 0.0:
        raise ValueError(
            "previous_stage_teacher_retention_weight must be non-negative"
        )
    if _previous_stage_teacher_scope not in {
        "previous_stage",
        "non_narrow",
        "corridor",
        "all",
    }:
        raise ValueError(
            "previous_stage_teacher_scope must be one of "
            "previous_stage/non_narrow/corridor/all"
        )
    if _corridor_teacher_distill_epochs < 0:
        raise ValueError(
            "corridor_teacher_distill_epochs must be non-negative"
        )
    if _corridor_teacher_distill_lr <= 0.0:
        raise ValueError("corridor_teacher_distill_lr must be positive")
    if _corridor_teacher_distill_batch_size <= 0:
        raise ValueError(
            "corridor_teacher_distill_batch_size must be positive"
        )
    if _corridor_teacher_distill_max_grad_norm <= 0.0:
        raise ValueError(
            "corridor_teacher_distill_max_grad_norm must be positive"
        )
    if not 0.0 <= _corridor_teacher_distill_neighbor_mass < 1.0:
        raise ValueError(
            "corridor_teacher_distill_neighbor_mass must be in [0, 1)"
        )
    if _corridor_teacher_distill_stride <= 0:
        raise ValueError(
            "corridor_teacher_distill_stride must be positive"
        )
    if _corridor_teacher_distill_chunk_size <= 0:
        raise ValueError(
            "corridor_teacher_distill_chunk_size must be positive"
        )
    if _corridor_teacher_intervention_clearance_m < 0.0:
        raise ValueError(
            "corridor_teacher_intervention_clearance_m must be non-negative"
        )
    _corridor_teacher_distill_enabled = (
        _corridor_teacher_distill_epochs > 0
    )
    if _corridor_adapter_enabled and _corridor_teacher_distill_enabled:
        raise ValueError(
            "corridor adapter and privileged corridor teacher projection "
            "must be ablated separately"
        )
    if (
        _corridor_adapter_enabled
        and _teacher_retention_post_kl_epochs > 0
    ):
        raise ValueError(
            "corridor adapter currently supports in-loss narrow retention only, "
            "not post-update KL projection"
        )
    _teacher_retention_enabled = (
        _teacher_retention_weight > 0.0
        or _teacher_retention_margin_weight > 0.0
        or _teacher_retention_action_ce_weight > 0.0
        or _teacher_retention_post_kl_epochs > 0
    )
    _previous_stage_teacher_enabled = (
        _previous_stage_teacher_retention_weight > 0.0
        or _teacher_retention_post_anchor_weight > 0.0
    )
    if (
        _teacher_retention_post_anchor_weight > 0.0
        and _teacher_retention_post_kl_epochs <= 0
    ):
        raise ValueError(
            "post anchor requires teacher_retention_post_kl_epochs > 0"
        )
    if _teacher_retention_enabled:
        if not _teacher_retention_checkpoint:
            raise ValueError(
                "teacher retention requires teacher_retention_checkpoint"
            )
        if not os.path.isfile(_teacher_retention_checkpoint):
            raise FileNotFoundError(
                f"teacher retention checkpoint not found: "
                f"{_teacher_retention_checkpoint}"
            )
        if not _e2e_frame_stack or _K_stack != 8:
            raise ValueError(
                "narrow teacher retention currently requires the K8 E2E lineage"
            )
        if _teacher_retention_post_kl_epochs > 0 and _lvdot_enc_on:
            raise ValueError(
                "post-update KL projection does not support LV-DOT encoder"
            )
        if float(getattr(args_cli, "narrow_passage_fraction", 0.0)) <= 0.0:
            raise ValueError(
                "teacher retention requires narrow_passage_fraction > 0"
            )
        if _lvdot_enc_on or _oracle_to_policy:
            raise ValueError(
                "teacher retention does not support LV-DOT/oracle policy inputs"
            )
    if _teacher_retention_rollout_override:
        if not _teacher_retention_enabled:
            raise ValueError(
                "teacher retention rollout override requires teacher retention"
            )
        if _teacher_retention_post_kl_epochs <= 0:
            raise ValueError(
                "teacher retention rollout override requires post-update "
                "projection epochs > 0"
            )
        if int(args_cli.ppo_epochs) != 0:
            raise ValueError(
                "teacher retention rollout override is projection-only and "
                "requires ppo_epochs=0"
            )
    if _previous_stage_teacher_enabled:
        if not _previous_stage_teacher_checkpoint:
            raise ValueError(
                "previous-stage teacher retention requires "
                "previous_stage_teacher_checkpoint"
            )
        if not os.path.isfile(_previous_stage_teacher_checkpoint):
            raise FileNotFoundError(
                "previous-stage teacher checkpoint not found: "
                f"{_previous_stage_teacher_checkpoint}"
            )
        if not _e2e_frame_stack or _K_stack != 8:
            raise ValueError(
                "previous-stage teacher retention requires the K8 E2E lineage"
            )
        if (
            _previous_stage_teacher_scope == "previous_stage"
            and float(
                getattr(args_cli, "previous_stage_replay_fraction", 0.0)
            ) <= 0.0
        ):
            raise ValueError(
                "previous-stage teacher retention requires "
                "previous_stage_replay_fraction > 0"
            )
        if (
            _previous_stage_teacher_scope == "non_narrow"
            and float(
                getattr(args_cli, "narrow_passage_fraction", 0.0)
            ) <= 0.0
        ):
            raise ValueError(
                "non-narrow teacher scope requires "
                "narrow_passage_fraction > 0"
            )
        if (
            _previous_stage_teacher_scope == "corridor"
            and float(
                getattr(args_cli, "long_corridor_fraction", 0.0)
            ) <= 0.0
        ):
            raise ValueError(
                "corridor teacher scope requires "
                "long_corridor_fraction > 0"
            )
        if _lvdot_enc_on or _oracle_to_policy:
            raise ValueError(
                "previous-stage teacher retention does not support "
                "LV-DOT/oracle policy inputs"
            )
    if _corridor_teacher_distill_enabled:
        if not _e2e_frame_stack or _K_stack != 8:
            raise ValueError(
                "corridor teacher distillation requires the K8 E2E lineage"
            )
        if _lvdot_enc_on or _oracle_to_policy:
            raise ValueError(
                "corridor teacher distillation does not support "
                "LV-DOT/oracle policy inputs"
            )
        if float(
            getattr(args_cli, "long_corridor_fraction", 0.0)
        ) <= 0.0:
            raise ValueError(
                "corridor teacher distillation requires "
                "long_corridor_fraction > 0"
            )
        if getattr(args_cli, "obstacle_mode", "") != "rule_based":
            raise ValueError(
                "corridor teacher distillation requires rule_based obstacles"
            )
        if bool(getattr(args_cli, "enable_actuator_dr", False)):
            raise ValueError(
                "corridor teacher does not model actuator DR"
            )
        if not bool(getattr(args_cli, "use_obb_collision", False)):
            raise ValueError(
                "corridor teacher distillation requires OBB collision lineage"
            )

    # WD: γ = 1 - (1 - 0.92) / rl_fps = 0.984（fps=5），跨所有 phase 固定不變
    # 注意：不同於 Isaac Lab NavRL 課程中 γ 會隨 stage 遞增（0.990→0.998）
    current_gamma = args_cli.gamma

    print(f"[INFO] {num_iterations} iterations, {RL} rollout, {batch_size:,} batch, "
          f"gamma={current_gamma} (WD constant)")
    print(f"[INFO] A2C mode: {'ON (no clipping)' if args_cli.use_a2c else 'OFF (PPO clip={})'.format(args_cli.clip_eps)}")
    print(
        f"[INFO] WD update monitor/clipping: "
        f"enabled={args_cli.wd_update_clip and not args_cli.wd_update_monitor_only} "
        f"monitor_only={args_cli.wd_update_monitor_only} "
        f"actor_grad_cap={args_cli.wd_actor_update_clip} critic_grad_cap={args_cli.wd_critic_update_clip}"
    )
    if args_cli.disable_aux_training:
        if _e2e_frame_stack:
            print("[INFO] Aux/RNN disabled: RNN is unused; frame-stack extractor is trained by PPO")
        else:
            print("[INFO] Aux/RNN training DISABLED: preprocess_rnn/extractor weights frozen")

    # --- Runtime-syncable trainer params (phase/task config may override these safely) ---
    _current_rl_lr = args_cli.lr
    _current_rnn_lr = args_cli.rnn_lr
    _current_vf_coeff = args_cli.vf_coeff
    _current_max_grad_norm = args_cli.max_grad_norm
    _current_aux_grad_clip = args_cli.aux_grad_clip if args_cli.aux_grad_clip is not None else args_cli.max_grad_norm
    _current_wd_actor_update_clip = args_cli.wd_actor_update_clip
    _current_wd_critic_update_clip = args_cli.wd_critic_update_clip

    # --- PopArt: running EMA return statistics (first two moments) for critic scale consistency ---
    # critic 輸出正規化空間(訓練穩定),用這些統計反正規化再餵 GAE(raw-reward 尺度=正確 baseline)。
    _popart_m1 = 0.0    # EMA of E[return]
    _popart_m2 = 1.0    # EMA of E[return^2]  (init → mean 0 / std 1)
    _popart_initialized = False  # ★fix#5a: 首rollout直接用其returns矩init(跳過EMA從0/1慢warmup的transient)
    if args_cli.popart:
        print(f"[POPART] enabled (beta={args_cli.popart_beta}): critic 輸出反正規化餵 GAE, "
              f"修 critic/actor 尺度不一致。overrides normalize_return.")

    def _set_optimizer_group_lr(group, lr: float):
        group["lr"] = lr
        group["initial_lr"] = lr

    def _sync_phase_trainer_params(cur_info: dict[str, float], stage_changed: bool = False):
        nonlocal _current_rl_lr, _current_rnn_lr, _current_vf_coeff
        nonlocal _current_max_grad_norm, _current_aux_grad_clip
        nonlocal _current_wd_actor_update_clip, _current_wd_critic_update_clip

        phase_rl_lr = float(cur_info.get("trainer_lr", 0.0) or 0.0)
        phase_rnn_lr = float(cur_info.get("trainer_rnn_lr", 0.0) or 0.0)
        phase_vf_coeff = float(cur_info.get("trainer_vf_coeff", 0.0) or 0.0)
        phase_max_grad_norm = float(cur_info.get("trainer_max_grad_norm", 0.0) or 0.0)
        phase_aux_grad_clip = float(cur_info.get("trainer_aux_grad_clip", 0.0) or 0.0)
        phase_actor_cap = float(cur_info.get("trainer_wd_actor_update_clip", 0.0) or 0.0)
        phase_critic_cap = float(cur_info.get("trainer_wd_critic_update_clip", 0.0) or 0.0)

        changed = []
        if phase_rl_lr > 0 and abs(phase_rl_lr - _current_rl_lr) > 1e-12:
            for pg in charge_opt_rl.param_groups:
                _set_optimizer_group_lr(pg, phase_rl_lr)
            _current_rl_lr = phase_rl_lr
            changed.append(f"lr={phase_rl_lr:g}")

        if phase_rnn_lr > 0 and abs(phase_rnn_lr - _current_rnn_lr) > 1e-12:
            for pg in charge_opt_aux.param_groups:
                base_lr = float(pg.get("initial_lr", pg["lr"]))
                if abs(base_lr - _current_rnn_lr) < 1e-12:
                    _set_optimizer_group_lr(pg, phase_rnn_lr)
            _current_rnn_lr = phase_rnn_lr
            changed.append(f"rnn_lr={phase_rnn_lr:g}")

        if phase_vf_coeff > 0 and abs(phase_vf_coeff - _current_vf_coeff) > 1e-12:
            _current_vf_coeff = phase_vf_coeff
            changed.append(f"vf_coeff={phase_vf_coeff:g}")

        if phase_max_grad_norm > 0 and abs(phase_max_grad_norm - _current_max_grad_norm) > 1e-12:
            _current_max_grad_norm = phase_max_grad_norm
            changed.append(f"max_grad_norm={phase_max_grad_norm:g}")

        if phase_aux_grad_clip > 0 and abs(phase_aux_grad_clip - _current_aux_grad_clip) > 1e-12:
            _current_aux_grad_clip = phase_aux_grad_clip
            changed.append(f"aux_grad_clip={phase_aux_grad_clip:g}")

        if phase_actor_cap > 0 and abs(phase_actor_cap - _current_wd_actor_update_clip) > 1e-12:
            _current_wd_actor_update_clip = phase_actor_cap
            changed.append(f"wd_actor_clip={phase_actor_cap:g}")

        if phase_critic_cap > 0 and abs(phase_critic_cap - _current_wd_critic_update_clip) > 1e-12:
            _current_wd_critic_update_clip = phase_critic_cap
            changed.append(f"wd_critic_clip={phase_critic_cap:g}")

        if changed or stage_changed:
            print(
                f"[INFO] Phase trainer sync: stage={int(cur_info.get('stage', -1))} "
                f"rl_lr={_current_rl_lr:g} rnn_lr={_current_rnn_lr:g} "
                f"vf={_current_vf_coeff:g} grad={_current_max_grad_norm:g} "
                f"aux_grad={_current_aux_grad_clip:g} wd_caps=({_current_wd_actor_update_clip:g}, {_current_wd_critic_update_clip:g})"
                f"{' | changed: ' + ', '.join(changed) if changed else ''}"
            )

    # --- Buffers ---
    _encoder_input_dim = (
        policy_obs_dim + (_K_stack - 1) * _LL if _e2e_frame_stack else 0
    )
    # === N1 直穿模仿（07-27 裁決）：scripted teacher 只貼標籤，不參與控制 ===
    # 舊 SA5 teacher KL 被取代——那顆 teacher 自己就繞路（3 seed n=2094,
    # crossing 95.9% 但 direct 0.000、pre_y_p95 3.86m），KL 0.30 一直在蒸餾繞行。
    _narrow_imitation_weight = float(
        getattr(args_cli, "narrow_imitation_weight", 0.0)
    )
    _narrow_imitation_shadow = bool(
        getattr(args_cli, "narrow_imitation_shadow", False)
    )
    if _narrow_imitation_weight < 0.0:
        raise ValueError("narrow_imitation_weight must be non-negative")
    _narrow_imitation_enabled = (
        _narrow_imitation_weight > 0.0 or _narrow_imitation_shadow
    )
    if _narrow_imitation_shadow:
        # Shadow is a measurement run, not a short PPO continuation. Keep all
        # optimizer-owned parameters fixed while still building both gradient
        # graphs below. This also prevents the auxiliary optimizer from moving
        # the shared encoder between the two calibration iterations.
        args_cli.disable_aux_training = True
    _narrow_imitation_action_term = None
    _narrow_imitation_spec = None
    if _narrow_imitation_enabled:
        if float(getattr(args_cli, "narrow_passage_fraction", 0.0)) <= 0.0:
            raise ValueError(
                "narrow imitation requires narrow_passage_fraction > 0"
            )
        if _teacher_retention_weight > 0.0:
            raise ValueError(
                "narrow imitation replaces the SA5 teacher KL (that teacher "
                "detours: 3-seed direct=0). Set teacher_retention_weight=0."
            )
        _narrow_imitation_action_term = next(
            (
                term
                for term in getattr(
                    env.unwrapped.action_manager, "_terms", {}
                ).values()
                if hasattr(term, "_current_velocity")
                and hasattr(term, "_current_omega")
                and hasattr(term, "_dt")
            ),
            None,
        )
        if _narrow_imitation_action_term is None:
            raise RuntimeError(
                "narrow imitation could not find the discrete drive action term"
            )
        _narrow_imitation_spec = ScriptedNarrowTeacherSpec()
        print(
            "[N1-IMITATION] enabled: "
            f"lambda={_narrow_imitation_weight:g} "
            f"shadow={_narrow_imitation_shadow} "
            "(scripted direct-crossing CE on narrow-replay frames only; "
            "teacher labels but never drives)"
        )

    charge_buf = ChargeRolloutBuffer(
        RL, num_envs, rl_input_dim, obs_dim, args_cli.hidden_dim, device,
        privileged_dim=_priv_dim, predict_dim=_predict_dim,
        encoder_input_dim=_encoder_input_dim,
        teacher_logits_dim=(2 * NUM_BINS if _teacher_retention_enabled else 0),
        previous_teacher_logits_dim=(
            2 * NUM_BINS if _previous_stage_teacher_enabled else 0
        ),
        store_corridor_mask=_corridor_adapter_enabled,
        store_narrow_teacher=_narrow_imitation_enabled,
    )
    obs_buf = ObstacleRolloutBuffer(RL, num_envs, N_obs, OBS_POLICY_OBS_DIM, 2, device) if _obstacle_mode == "learned" else None

    # --- Metrics ---
    metrics = MetricsCollector(num_envs, N_obs, device, args_cli.action_table_sample_size)

    # --- Log dir + WandB ---
    run_name = args_cli.run_name or f"marl_{datetime.now().strftime('%m%d_%H%M')}"
    log_dir = os.path.join("logs", "rnn_car", run_name)
    os.makedirs(log_dir, exist_ok=True)

    wandb_run = None
    if headless_mode:
        try:
            import wandb
            wandb.init(
                project="charge_skrl", name=run_name,
                notes=args_cli.wandb_notes if args_cli.wandb_notes else None,
                config={
                    "agent": "MARL-ModularRNN-A2CK-v5" if args_cli.use_a2c else "MARL-ModularRNN-PPO-v5",
                    "charge_encoder_mode": args_cli.charge_encoder_mode,
                    "num_envs": num_envs, "seed": args_cli.seed,
                    "charge_lr": args_cli.lr, "rnn_lr": args_cli.rnn_lr,
                    "aux_lr": args_cli.aux_lr,
                    "aux_lr_predict_head": _lr_predict_head,
                    "aux_lr_fc_middle": _lr_fc_middle,
                    "aux_lr_fc_front": _lr_fc_front,
                    "aux_lr_extractor": _lr_extractor if use_extractor else "N/A",
                    "max_grad_norm": args_cli.max_grad_norm,
                    "aux_grad_clip": args_cli.aux_grad_clip if args_cli.aux_grad_clip is not None else args_cli.max_grad_norm,
                    "wd_update_clip": args_cli.wd_update_clip and not args_cli.wd_update_monitor_only,
                    "wd_actor_grad_cap": args_cli.wd_actor_update_clip,
                    "wd_critic_grad_cap": args_cli.wd_critic_update_clip,
                    "policy_loss_clamp": args_cli.policy_loss_clamp,
                    "vf_term_clamp": args_cli.vf_term_clamp,
                    "vf_coeff": args_cli.vf_coeff,
                    "normalize_return": args_cli.normalize_return,
                    "value_init_bias": args_cli.value_init_bias,
                    "disable_aux_training": args_cli.disable_aux_training,
                    "resume_optimizer": (args_cli.checkpoint is not None and not args_cli.no_resume_optimizer),
                    "action_table_sample_size": args_cli.action_table_sample_size,
                    "teacher_retention_checkpoint": _teacher_retention_checkpoint,
                    "teacher_retention_weight": _teacher_retention_weight,
                    "teacher_retention_margin_weight": _teacher_retention_margin_weight,
                    "teacher_retention_action_ce_weight": _teacher_retention_action_ce_weight,
                    "teacher_retention_argmax_margin": _teacher_retention_argmax_margin,
                    "teacher_retention_post_kl_epochs": _teacher_retention_post_kl_epochs,
                    "teacher_retention_post_kl_lr": _teacher_retention_post_kl_lr,
                    "teacher_retention_post_kl_batch_size": _teacher_retention_post_kl_batch_size,
                    "teacher_retention_post_kl_max_grad_norm": _teacher_retention_post_kl_max_grad_norm,
                    "teacher_retention_post_margin_weight": _teacher_retention_post_margin_weight,
                    "teacher_retention_post_action_ce_weight": _teacher_retention_post_action_ce_weight,
                    "teacher_retention_post_policy_head_only": _teacher_retention_post_policy_head_only,
                    "teacher_retention_post_anchor_weight": _teacher_retention_post_anchor_weight,
                    "teacher_retention_rollout_override": _teacher_retention_rollout_override,
                    "previous_stage_teacher_checkpoint": _previous_stage_teacher_checkpoint,
                    "previous_stage_teacher_retention_weight": _previous_stage_teacher_retention_weight,
                    "previous_stage_teacher_scope": _previous_stage_teacher_scope,
                    "corridor_teacher_distill_epochs": _corridor_teacher_distill_epochs,
                    "corridor_teacher_distill_lr": _corridor_teacher_distill_lr,
                    "corridor_teacher_distill_batch_size": _corridor_teacher_distill_batch_size,
                    "corridor_teacher_distill_max_grad_norm": _corridor_teacher_distill_max_grad_norm,
                    "corridor_teacher_distill_neighbor_mass": _corridor_teacher_distill_neighbor_mass,
                    "corridor_teacher_distill_stride": _corridor_teacher_distill_stride,
                    "corridor_teacher_distill_chunk_size": _corridor_teacher_distill_chunk_size,
                    "corridor_teacher_intervention_only": _corridor_teacher_intervention_only,
                    "corridor_teacher_intervention_clearance_m": _corridor_teacher_intervention_clearance_m,
                    "critic_detach_encoder": args_cli.critic_detach_encoder,
                    "gamma": args_cli.gamma, "use_a2c": args_cli.use_a2c,
                    "rnn_type": args_cli.rnn_type,
                    "hidden_dim": args_cli.hidden_dim,
                    "preprocess_dim": args_cli.preprocess_dim,
                    "predict_dim": _predict_dim,
                    "aux_velocity_topk": _aux_vel_topk,
                    "obs_lr": args_cli.obs_lr, "obs_ent_coeff": args_cli.obs_ent_coeff,
                    "obs_speed_limit": args_cli.obs_speed_limit,
                    "train_goal_rate": args_cli.train_goal_rate,
                    "obs_reward_mode": args_cli.obs_reward_mode,
                    "rollout_length": RL, "ppo_epochs": args_cli.ppo_epochs,
                    "tbptt_len": args_cli.tbptt_len,
                    "lr_decay": args_cli.lr_decay,
                    "total_charge_params": total_charge_params,
                    "total_obs_params": total_obs_params,
                    "obs_size_rand": args_cli.obs_size_rand,
                    "obs_collision_base": args_cli.obs_collision_base,
                    "scene_bound_rand": args_cli.scene_bound_rand,
                    "scene_bound_base": args_cli.scene_bound_base,
                    "profiles": profiles_to_dict(_trainer_profiles),
                    "experiment_config": experiment_config_to_dict(_experiment_cfg) if _experiment_cfg else None,
                },
                tags=["marl", "obstacle-policy", "v5", "wd-principle",
                       _trainer_profiles.reward_profile,
                       _trainer_profiles.algorithm_profile]
                      + (list(_experiment_cfg.tags) if _experiment_cfg else []),
            )
            wandb_run = wandb.run
            print(f"[INFO] WandB: {wandb.run.name}")
        except Exception as e:
            print(f"[WARN] WandB init failed: {e}")

    # --- Load checkpoint ---
    if args_cli.checkpoint:
        ckpt = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        if use_extractor and "extractor" in ckpt:
            _extractor_state, _stack_migration = adapt_lidar_frame_stack_state_dict(
                extractor, ckpt["extractor"]
            )
            extractor.load_state_dict(_extractor_state)
            if _stack_migration is not None:
                print(
                    f"[E2E-PPO] LiDAR frame-stack warm migration: "
                    f"K={_stack_migration[0]} -> K={_stack_migration[1]}; "
                    "old Conv1d channels copied, added history channels zero-initialized"
                )
        # preprocess_rnn: 若 predict_head shape 不符（predict_dim 改變，例 13→7 position-only），
        # 載入 shape 相符的部分（RNN/fc_front/fc_middle），predict_head 留新初始化。
        _pp_ckpt = ckpt["preprocess_rnn"]
        _pp_model = preprocess_rnn.state_dict()
        # aux_reinit_frozen: 跳過 fc_middle/fc_front/predict_head(保持隨機初始化)=WD 隨機凍結 readout
        _reinit_prefixes = ("fc_middle", "fc_front", "predict_head") if getattr(args_cli, "aux_reinit_frozen", False) else ()
        # reinit_rnn: 額外跳過 RNN cell(全新隨機)— 隔離「fresh RNN + 好 extractor」
        if getattr(args_cli, "reinit_rnn", False):
            _reinit_prefixes = _reinit_prefixes + ("rnn",)
            print("[INFO] reinit_rnn: RNN cell 重新隨機初始化(extractor/policy 照載)")
        _pp_filtered = {k: v for k, v in _pp_ckpt.items()
                        if k in _pp_model and v.shape == _pp_model[k].shape
                        and not k.startswith(_reinit_prefixes)}
        _pp_skipped = [k for k in _pp_ckpt if k not in _pp_filtered]
        if _reinit_prefixes:
            print(f"[INFO] aux_reinit_frozen: 跳過載入(保持隨機) {_reinit_prefixes}")
        _pp_missing, _pp_unexpected = preprocess_rnn.load_state_dict(_pp_filtered, strict=False)
        if _pp_skipped:
            print(f"[INFO] preprocess_rnn: 跳過 shape 不符的 keys（重初始化）: {_pp_skipped}")
        if _pp_missing:
            print(f"[INFO] preprocess_rnn: 新初始化 params（checkpoint 缺）: {list(_pp_missing)}")
        if _lvdot_enc_on and "lvdot_encoder" not in ckpt:
            # ★暖啟遷移: 從「無 encoder」checkpoint(如 SA2)起訓。head 第一層輸入維度改變
            #   (加 encoder: 121→115),只重初始化第一層 + encoder(全新),深層照載(shape 相符)。
            _ph_first = ("net.0.weight", "net.0.bias")   # PolicyHead 第一層
            _ph_filtered = {k: v for k, v in ckpt["policy_head"].items() if k not in _ph_first}
            _ph_m, _ph_u = policy_head.load_state_dict(_ph_filtered, strict=False)
            print(f"[LVDOT-ENC] policy_head 暖啟遷移: 深層載入, 第一層(net.0)重初始化。new={list(_ph_m)}")
            # ValueHead: asymmetric 第一層=rl_proj(input→128); symmetric=net.0(input→256)
            _vh_first = ("rl_proj.weight", "rl_proj.bias") if _use_asymmetric_critic else ("net.0.weight", "net.0.bias")
            _vh_filtered = {k: v for k, v in ckpt["value_head"].items() if k not in _vh_first}
            _vh_m, _vh_u = value_head.load_state_dict(_vh_filtered, strict=False)
            print(f"[LVDOT-ENC] value_head 暖啟遷移: 深層載入, 第一層({_vh_first[0].split('.')[0]})重初始化。new={list(_vh_m)}")
            print("[LVDOT-ENC] lvdot_encoder 全新(末層 zero-init → 初始輸出≈0,漸進加入)")
        elif _lvdot_enc_on:
            # resume: checkpoint 已含 encoder → head 維度相符,正常載入 + 載 encoder
            policy_head.load_state_dict(ckpt["policy_head"], strict=(not _oracle_to_policy))
            value_head.load_state_dict(ckpt["value_head"], strict=(not _use_asymmetric_critic))
            if lvdot_encoder is not None and "lvdot_encoder" in ckpt:
                lvdot_encoder.load_state_dict(ckpt["lvdot_encoder"])
                print("[LVDOT-ENC] resume: lvdot_encoder 已從 checkpoint 載入")
        else:
            policy_head.load_state_dict(ckpt["policy_head"], strict=(not _oracle_to_policy))  # oracle: priv_branch 新建→strict=False
            _vh_strict = not _use_asymmetric_critic
            _vh_missing, _vh_unexpected = value_head.load_state_dict(ckpt["value_head"], strict=_vh_strict)
            if _vh_missing:
                print(f"[INFO] ValueHead: new params (cold start): {_vh_missing}")
        if _corridor_adapter_enabled:
            if "corridor_adapter" in ckpt:
                corridor_adapter.load_state_dict(ckpt["corridor_adapter"])
                print("[CORRIDOR-ADAPTER] resumed adapter state from checkpoint")
            else:
                if not args_cli.no_resume_optimizer:
                    raise ValueError(
                        "adding a corridor adapter to a legacy checkpoint "
                        "requires no_resume_optimizer=True"
                    )
                print(
                    "[CORRIDOR-ADAPTER] legacy checkpoint migration: base loaded, "
                    "adapter residual remains exactly zero"
                )
        if "obs_policy" in ckpt and obs_policy is not None:
            obs_policy.load_state_dict(ckpt["obs_policy"])
            obs_value.load_state_dict(ckpt["obs_value"])
        elif "obs_policy" in ckpt and obs_policy is None:
            print(f"[INFO] Checkpoint has obs_policy but obstacle_mode={_obstacle_mode} — skipping obs_policy load")
        if "obs_normalizer" in ckpt:
            obs_normalizer.mean = ckpt["obs_normalizer"]["mean"]
            obs_normalizer.var = ckpt["obs_normalizer"]["var"]
            obs_normalizer.count = ckpt["obs_normalizer"]["count"]
        if args_cli.feat_norm and "feat_normalizer" in ckpt:
            feat_normalizer.mean = ckpt["feat_normalizer"]["mean"].to(device)
            feat_normalizer.var = ckpt["feat_normalizer"]["var"].to(device)
            feat_normalizer.count = ckpt["feat_normalizer"]["count"]
            print("[INFO] feat_normalizer 統計已從 checkpoint 載入")
        if not args_cli.no_resume_optimizer and not args_cli.play:
            for key, opt in (
                ("charge_opt_rl", charge_opt_rl),
                ("charge_opt_aux", charge_opt_aux),
                ("obs_optimizer", obs_optimizer),
            ):
                if key in ckpt and opt is not None:
                    try:
                        opt.load_state_dict(ckpt[key])
                        for pg in opt.param_groups:
                            pg["initial_lr"] = pg["lr"]
                        print(f"[INFO] Loaded optimizer state: {key}")
                    except (ValueError, RuntimeError, KeyError) as e:
                        print(f"[WARN] Failed to load optimizer state {key}: {e}")
        print(f"[INFO] Loaded checkpoint: {args_cli.checkpoint}")

    _teacher_extractor = None
    _teacher_policy_head = None
    _teacher_obs_mean = None
    _teacher_obs_var = None
    _teacher_lidar_hist = None
    if _teacher_retention_enabled:
        teacher_ckpt = torch.load(
            _teacher_retention_checkpoint,
            map_location=device,
            weights_only=False,
        )
        teacher_args = teacher_ckpt.get("args", {})
        teacher_contract = {
            "end_to_end_frame_stack": True,
            "lidar_frame_stack": _K_stack,
            "use_obb_collision": bool(getattr(args_cli, "use_obb_collision", False)),
            "use_action_history": bool(getattr(args_cli, "use_action_history", False)),
        }
        mismatches = {
            key: (teacher_args.get(key), expected)
            for key, expected in teacher_contract.items()
            if teacher_args.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"teacher checkpoint is incompatible with current lineage: {mismatches}"
            )
        if "extractor" not in teacher_ckpt or "policy_head" not in teacher_ckpt:
            raise ValueError("teacher checkpoint lacks extractor/policy_head state")
        teacher_norm = teacher_ckpt.get("obs_normalizer")
        if teacher_norm is None:
            raise ValueError("teacher checkpoint lacks obs_normalizer")
        _teacher_obs_mean = teacher_norm["mean"].to(device).reshape(-1)
        _teacher_obs_var = teacher_norm["var"].to(device).reshape(-1)
        if _teacher_obs_mean.numel() != obs_dim or _teacher_obs_var.numel() != obs_dim:
            raise ValueError(
                "teacher observation normalizer shape mismatch: "
                f"teacher={_teacher_obs_mean.numel()} runtime={obs_dim}"
            )

        _teacher_extractor = copy.deepcopy(extractor)
        _teacher_policy_head = copy.deepcopy(policy_head)
        _teacher_extractor.load_state_dict(teacher_ckpt["extractor"], strict=True)
        _teacher_policy_head.load_state_dict(
            teacher_ckpt["policy_head"], strict=True
        )
        _teacher_extractor.eval()
        _teacher_policy_head.eval()
        for parameter in (
            list(_teacher_extractor.parameters())
            + list(_teacher_policy_head.parameters())
        ):
            parameter.requires_grad_(False)
        _teacher_lidar_hist = torch.zeros(
            num_envs, (_K_stack - 1) * _LL, device=device
        )
        print(
            "[TEACHER-RETENTION] enabled: "
            f"beta={_teacher_retention_weight:g} "
            f"margin_weight={_teacher_retention_margin_weight:g} "
            f"action_ce_weight={_teacher_retention_action_ce_weight:g} "
            f"argmax_margin={_teacher_retention_argmax_margin:g} "
            f"post_kl_epochs={_teacher_retention_post_kl_epochs} "
            f"post_kl_lr={_teacher_retention_post_kl_lr:g} "
            f"post_margin_weight={_teacher_retention_post_margin_weight:g} "
            f"post_action_ce_weight={_teacher_retention_post_action_ce_weight:g} "
            f"post_head_only={_teacher_retention_post_policy_head_only} "
            f"post_anchor_weight={_teacher_retention_post_anchor_weight:g} "
            f"rollout_override={_teacher_retention_rollout_override} "
            "scope=narrow_only "
            f"teacher={_teacher_retention_checkpoint} "
            "loss=KL(teacher_linear||student_linear)"
            "+KL(teacher_angular||student_angular)"
            "+teacher_argmax_margin+teacher_argmax_CE",
            flush=True,
        )

    _previous_teacher_extractor = None
    _previous_teacher_policy_head = None
    _previous_teacher_obs_mean = None
    _previous_teacher_obs_var = None
    _previous_teacher_lidar_hist = None
    if _previous_stage_teacher_enabled:
        previous_teacher_ckpt = torch.load(
            _previous_stage_teacher_checkpoint,
            map_location=device,
            weights_only=False,
        )
        previous_teacher_args = previous_teacher_ckpt.get("args", {})
        previous_teacher_contract = {
            "end_to_end_frame_stack": True,
            "lidar_frame_stack": _K_stack,
            "use_obb_collision": bool(
                getattr(args_cli, "use_obb_collision", False)
            ),
            "use_action_history": bool(
                getattr(args_cli, "use_action_history", False)
            ),
        }
        previous_teacher_mismatches = {
            key: (previous_teacher_args.get(key), expected)
            for key, expected in previous_teacher_contract.items()
            if previous_teacher_args.get(key) != expected
        }
        if previous_teacher_mismatches:
            raise ValueError(
                "previous-stage teacher checkpoint is incompatible with "
                f"current lineage: {previous_teacher_mismatches}"
            )
        if (
            "extractor" not in previous_teacher_ckpt
            or "policy_head" not in previous_teacher_ckpt
        ):
            raise ValueError(
                "previous-stage teacher checkpoint lacks "
                "extractor/policy_head state"
            )
        previous_teacher_norm = previous_teacher_ckpt.get("obs_normalizer")
        if previous_teacher_norm is None:
            raise ValueError(
                "previous-stage teacher checkpoint lacks obs_normalizer"
            )
        _previous_teacher_obs_mean = previous_teacher_norm["mean"].to(
            device
        ).reshape(-1)
        _previous_teacher_obs_var = previous_teacher_norm["var"].to(
            device
        ).reshape(-1)
        if (
            _previous_teacher_obs_mean.numel() != obs_dim
            or _previous_teacher_obs_var.numel() != obs_dim
        ):
            raise ValueError(
                "previous-stage teacher observation normalizer shape "
                f"mismatch: teacher={_previous_teacher_obs_mean.numel()} "
                f"runtime={obs_dim}"
            )

        _previous_teacher_extractor = copy.deepcopy(extractor)
        _previous_teacher_policy_head = copy.deepcopy(policy_head)
        _previous_teacher_extractor.load_state_dict(
            previous_teacher_ckpt["extractor"], strict=True
        )
        _previous_teacher_policy_head.load_state_dict(
            previous_teacher_ckpt["policy_head"], strict=True
        )
        _previous_teacher_extractor.eval()
        _previous_teacher_policy_head.eval()
        for parameter in (
            list(_previous_teacher_extractor.parameters())
            + list(_previous_teacher_policy_head.parameters())
        ):
            parameter.requires_grad_(False)
        _previous_teacher_lidar_hist = torch.zeros(
            num_envs, (_K_stack - 1) * _LL, device=device
        )
        print(
            "[PREVIOUS-STAGE-RETENTION] enabled: "
            f"beta={_previous_stage_teacher_retention_weight:g} "
            f"scope={_previous_stage_teacher_scope} "
            f"teacher={_previous_stage_teacher_checkpoint} "
            "loss=KL(teacher_linear||student_linear)"
            "+KL(teacher_angular||student_angular)",
            flush=True,
        )

    # --- Initial reset ---
    obs, info = env.reset()
    _long_corridor_goal_error_max = 0.0
    _long_corridor_local_goal_error_max = 0.0
    _long_corridor_goal_audit_frames = 0

    def _audit_long_corridor_goal() -> None:
        nonlocal _long_corridor_goal_error_max
        nonlocal _long_corridor_local_goal_error_max
        nonlocal _long_corridor_goal_audit_frames
        if _long_corridor_fraction <= 0.0:
            return
        raw = env.unwrapped
        active = getattr(raw, "_long_corridor_active", None)
        expected = getattr(raw, "_long_corridor_goal_w", None)
        if active is None or expected is None:
            raise RuntimeError(
                "long-corridor replay is enabled but goal ownership state is missing"
            )
        if not bool(active.any()):
            return

        goal_term = raw.command_manager.get_term("goal_command")
        command_error = torch.linalg.vector_norm(
            goal_term.command[active, :2] - expected[active, :2], dim=1
        )
        command_max = float(command_error.max().item())
        _long_corridor_goal_error_max = max(
            _long_corridor_goal_error_max, command_max
        )
        local_max = 0.0
        local_goal = getattr(raw, "_local_goal_world", None)
        if local_goal is not None:
            local_error = torch.linalg.vector_norm(
                local_goal[active, :2] - expected[active, :2], dim=1
            )
            local_max = float(local_error.max().item())
            _long_corridor_local_goal_error_max = max(
                _long_corridor_local_goal_error_max, local_max
            )
        _long_corridor_goal_audit_frames += int(active.sum().item())
        if command_max > 1e-5 or local_max > 1e-5:
            raise RuntimeError(
                "long-corridor goal ownership violated: "
                f"command_error={command_max:.3e}m "
                f"local_error={local_max:.3e}m"
            )

    _audit_long_corridor_goal()
    if _long_corridor_fraction > 0.0:
        print(
            "[LONG-CORRIDOR-GOAL] initial alignment PASS: "
            f"command_error={_long_corridor_goal_error_max:.2e}m "
            f"local_error={_long_corridor_local_goal_error_max:.2e}m",
            flush=True,
        )
    start_time = time.time()
    _supervisor_stop_file = os.environ.get(
        "CHARGE_SUPERVISOR_STOP_FILE",
        os.path.join(log_dir, "supervisor_stop.request"),
    )
    _supervisor_metrics_file = os.path.join(log_dir, "supervisor_metrics.jsonl")
    _supervisor_stopped = False

    # --- WD reward params (initial, Phase 1 defaults) ---
    _spot_penalty_hit = -5.0
    _spot_reward_get_goal = 40.0
    _spot_cost_operate = 0.03  # Phase 1 default (will update from curriculum)
    _spot_penalty_timeout = 0.0  # timeout penalty (0 = no penalty, SA6+)
    _spot_penalty_smoothness = 0.0  # v3: anti-jitter Δratio penalty (will update from curriculum)
    _spot_penalty_speed_near_obs = 0.0  # v3f-react: clearance-gated 減速 (will update from curriculum)
    # LiDAR max range (m) for normalizing obs[6:78] back to meters in clearance-gated penalty.
    # Matches curriculum env config (charge_env_cfg_vlp16_curriculum.py: max_distance=8.0).
    _TEARDROP_COS = None           # ★07-06 水滴稅 max(cosθ,0) per-bin 權重快取 (首次用時建)
    _LIDAR_MAX_DISTANCE_M = 20.0   # ★07-06 修正 8→20 對齊 wd_like_sweep_72 r_max (obs=(d_surf)/20)。
    # 舊 8.0 = v2 時代 obs=/8 殭屍常數; v3 起 obs=/20 → react/gap/monitor 距離門檻全錯讀 2.5 倍
    # (react d_react 1.2m 實際 3m 生效)。護欄: tools/test_obs_units.py 斷言此值 == r_max。
    _reward_module.update_params({
        "spot_penalty_hit": _spot_penalty_hit,
        "spot_reward_get_goal": _spot_reward_get_goal,
        "spot_cost_operate": _spot_cost_operate,
        "spot_penalty_smoothness": _spot_penalty_smoothness,
        "spot_penalty_speed_near_obs": _spot_penalty_speed_near_obs,
    })

    # --- v3: prev_actions buffer for smoothness penalty (anti-jitter) ---
    # Initialize as None; populated after first action in rollout loop
    _prev_actions = None  # type: torch.Tensor | None

    # --- WD entropy params (initial, safe defaults) ---
    # Conservative defaults (0.01/0.02) prevent entropy explosion on iter 0
    # before curriculum sync kicks in at iter 1. Old WD defaults (0.10/0.375)
    # were too large and caused catastrophic entropy bonus > policy loss.
    _ent_coeff_linear = args_cli.ent_coeff_linear if args_cli.ent_coeff_linear > 0 else 0.01
    _ent_coeff_angular = args_cli.ent_coeff_angular if args_cli.ent_coeff_angular > 0 else 0.02
    if args_cli.ent_coeff > 0:  # Legacy single coeff override
        _ent_coeff_linear = args_cli.ent_coeff
        _ent_coeff_angular = args_cli.ent_coeff

    if _reward_module.name == "clean_progress":
        print("[INFO] clean_progress reward: progress=1.0 step=-0.01 "
              "smooth=(-0.01 dv,-0.005 dw) goal=+10 collision=-15 timeout=-5")
        if args_cli.anti_spin_weight > 0.0:
            print(
                f"[REWARD] A anti-spin: max=-{args_cli.anti_spin_weight:.3f}/step "
                f"hazard<{args_cli.anti_spin_hazard_distance:.2f}m "
                f"|omega|>{args_cli.anti_spin_omega_threshold:.2f} "
                f"progress<={args_cli.anti_spin_progress_threshold:.3f} "
                f"same-sign-yaw>{args_cli.anti_spin_yaw_grace_deg:.0f}deg "
                f"yaw-ramp={args_cli.anti_spin_yaw_ramp_deg:.0f}deg "
                f"grace={args_cli.anti_spin_grace_steps} ramp={args_cli.anti_spin_ramp_steps} steps"
            )
        if args_cli.future_occupancy_weight > 0.0:
            print(
                f"[REWARD] future occupancy: max=-{args_cli.future_occupancy_weight:.3f}/step "
                f"horizon={args_cli.future_occupancy_horizon_s:.2f}s "
                f"samples={args_cli.future_occupancy_samples} "
                f"safe<{args_cli.future_occupancy_safe_distance_m:.2f}m "
                f"near<{args_cli.future_occupancy_near_distance_m:.2f}m "
                f"moving>{args_cli.future_occupancy_move_threshold_mps:.2f}m/s"
            )
    else:
        print(
            f"[INFO] Warp Drive reward: "
            f"penalty_hit={_spot_penalty_hit}, get_goal={_spot_reward_get_goal}, "
            f"cost_operate={_spot_cost_operate}"
        )
    print(
        f"[INFO] Warp Drive entropy (A2CK per-head): "
        f"linear={_ent_coeff_linear:.3f}, angular={_ent_coeff_angular:.3f} "
        f"(Phase 1 defaults, auto-sync from curriculum)"
    )

    # ========================================================================
    # Training Loop（主訓練迴圈）
    # ========================================================================
    # 每次 iteration 的完整流程：
    #   1. Rollout: 收集 RL 步數（Charge + Obstacle 交替）
    #   2. Charge RL Update: A2C/PPO loss + WD gradient caps
    #   3. Charge Aux Update: TBPTT RNN module loss（WD-order 或 legacy-order）
    #   4. Obstacle PPO Update: 每 train_goal_rate 次訓練 1 次 obstacle
    #   5. Logging: WandB metrics + console print
    #   6. Checkpoint: 每 save_interval 次儲存模型

    # A same-stage interrupted resume must preserve the restored Adam state. Curriculum
    # metrics are empty before the first rollout, so seed the stage from the frozen config
    # instead of briefly treating the run as stage 1 and triggering a false 1→N reset.
    _optimizer_was_resumed = (
        args_cli.checkpoint is not None
        and not args_cli.no_resume_optimizer
        and not args_cli.play
    )
    _prev_stage = int(args_cli.initial_stage) if _optimizer_was_resumed else -1
    _prev_obs_agent_active = True  # 追蹤 obs_agent 是否活躍（用於 logging）
    _prev_rnn_feature_mean = None  # 追蹤 RNN feature 分佈，偵測漂移
    _prev_unsolvable_scene_count = int(getattr(env.unwrapped, "_unsolvable_scene_count_total", 0))

    # --- Obs Delay DR：模擬真實感測器管線延遲（20-50ms → 0-2 steps）---
    # 真實世界 LiDAR→policy 有 ~30ms 延遲，sim 裡 policy 看到即時觀測。
    # 用 tensor ring buffer 人為延遲 policy 收到的觀測，讓 policy 學到更保守的安全邊距。
    _delay_cfg = getattr(args_cli, 'obs_delay_steps', (0, 0))
    if isinstance(_delay_cfg, (list, tuple)):
        _delay_lo, _delay_hi = int(_delay_cfg[0]), int(_delay_cfg[1])
    else:
        _delay_lo = _delay_hi = 0
    _use_obs_delay = _delay_hi > 0
    if _use_obs_delay:
        _obs_ring = torch.zeros(_delay_hi + 1, num_envs, obs_dim, device=device)
        _obs_ring_ptr = torch.zeros(num_envs, dtype=torch.long, device=device)
        _delay_per_env = torch.randint(
            _delay_lo, _delay_hi + 1, (num_envs,), device=device)
        _env_arange = torch.arange(num_envs, device=device)  # 快取索引避免每步重建
        print(f"[SIM2REAL] Obs delay DR: delay∈[{_delay_lo}, {_delay_hi}] steps "
              f"(real latency ~30ms, dt=200ms → 1 step ≈ 200ms)")

    # --- Heading Stability：懲罰角速度符號翻轉（抗震盪）---
    _heading_stability_weight = getattr(args_cli, 'heading_stability_weight', 0.0)
    _prev_omega = torch.zeros(num_envs, device=device)
    # ★ Term 1 (predictive decel): 追蹤前一步速度以偵測「本步是否減速」(_prev_speed - cur_speed > 0)
    _prev_speed = torch.zeros(num_envs, device=device)
    _pd_w_cfg = float(getattr(args_cli, "predictive_decel_weight", 0.0))
    _rec_w_cfg = float(getattr(args_cli, "receding_penalty_weight", 0.0))
    # ★ Term A (speed-aware turning) 權重快取 + prev_speed 是否需維護
    _td_w_cfg = float(getattr(args_cli, "turning_direction_weight", 0.0))
    _et_w_cfg = float(getattr(args_cli, "early_turning_weight", 0.0))
    _wd_w_cfg = float(getattr(args_cli, "weakened_decel_weight", 0.0))
    _track_prev_speed = (_pd_w_cfg > 0.0 or _rec_w_cfg > 0.0 or _wd_w_cfg > 0.0)  # decel 偵測需前一步速度
    if _pd_w_cfg > 0.0 or _rec_w_cfg > 0.0:
        print(f"[REWARD] ★Term1 predictive-decel w={_pd_w_cfg} + receding-penalty w={_rec_w_cfg} "
              f"horizon={float(getattr(args_cli,'predictive_pred_horizon',2.0))}s "
              f"(逼 policy 用 LV-DOT 速度區分該不該減速)")
    if _td_w_cfg > 0.0 or _et_w_cfg > 0.0 or _wd_w_cfg > 0.0:
        print(f"[REWARD] ★Term A speed-aware turning: A1 direction w={_td_w_cfg} + A2 early-turn w={_et_w_cfg} "
              f"(TTC {float(getattr(args_cli,'early_turning_ttc_lo',1.6))}~{float(getattr(args_cli,'early_turning_ttc_hi',3.8))}s) "
              f"+ weakened-decel w={_wd_w_cfg} (逼 policy 用速度提早往對的方向轉,非走走停停)")
    if _heading_stability_weight != 0:
        print(f"[REWARD] Heading stability: weight={_heading_stability_weight} "
              f"(penalize omega sign flips)")

    # --- DORAEMON：自動 DR 擴展（基於 SR 回饋）---
    _doraemon_enabled = getattr(args_cli, 'doraemon_enabled', False)
    _doraemon_ctrl = None
    if _doraemon_enabled:
        from auto_dr_controller import AutoDRController
        _doraemon_ctrl = AutoDRController(
            threshold=getattr(args_cli, 'doraemon_sr_threshold', 0.85),
            check_interval=getattr(args_cli, 'doraemon_check_interval', 50),
            expansion_rate=getattr(args_cli, 'doraemon_expansion_rate', 0.1),
        )
        print(f"[DORAEMON] Auto DR enabled: τ={_doraemon_ctrl.threshold}, "
              f"interval={_doraemon_ctrl.check_interval}, "
              f"rate={_doraemon_ctrl.expansion_rate}")

    # --- RGDR：Reward-Guided Loss Weighting（聚焦失敗環境）---
    # 低 reward 的 env → 高 weight → A2C loss 集中訓練困難場景
    _rgdr_enabled = getattr(args_cli, 'rgdr_enabled', False)
    if _rgdr_enabled:
        _rgdr_env_returns = torch.zeros(num_envs, device=device)
        _rgdr_episode_reward = torch.zeros(num_envs, device=device)
        _rgdr_alpha = 0.1  # EMA decay for episode return tracking
        _rgdr_clamp = getattr(args_cli, 'rgdr_weight_clamp', (0.5, 2.0))
        _rgdr_warmup = 10  # 前 N 個 iteration 不啟用（讓 EMA 穩定）
        print(f"[RGDR] Enabled: weight clamp={_rgdr_clamp}, EMA α={_rgdr_alpha}")

    # ★r_arc: 取一次 action term（含實際套用的 processed_actions=(v_x, ω)，紅線②）
    _arc_action_term = None
    if getattr(args_cli, "use_arc_reward", False):
        _arc_terms = getattr(env.unwrapped.action_manager, "_terms", {})
        _arc_action_term = next((t for t in _arc_terms.values() if hasattr(t, "processed_actions")), None)
        if _arc_action_term is None:
            print("[r_arc] ⚠ 找不到有 processed_actions 的 action term，r_arc 停用")
        else:
            print(f"[r_arc] ✅ 啟用 swept-arc reward: w={args_cli.arc_w} c_safe={args_cli.arc_c_safe} "
                  f"cap={args_cli.arc_cap} horizon={args_cli.arc_horizon}s body_r={args_cli.arc_body_radius}")
    # ★r_arc: per-env 每集累積 arc penalty（跨 iteration 持續、done 時記錄並歸零）
    _ep_arc_accum = torch.zeros(num_envs, device=device)

    _corridor_teacher_action_term = None
    _corridor_teacher_spec = None
    if _corridor_teacher_distill_enabled:
        _action_terms = getattr(
            env.unwrapped.action_manager, "_terms", {}
        )
        _corridor_teacher_action_term = next(
            (
                term for term in _action_terms.values()
                if hasattr(term, "_current_velocity")
                and hasattr(term, "_current_omega")
                and hasattr(term, "_dt")
            ),
            None,
        )
        if _corridor_teacher_action_term is None:
            raise RuntimeError(
                "corridor teacher could not find the discrete drive action term"
            )
        _corridor_teacher_spec = CorridorTeacherSpec()
        print(
            "[CORRIDOR-DISTILL] enabled: "
            f"epochs={_corridor_teacher_distill_epochs} "
            f"lr={_corridor_teacher_distill_lr:g} "
            f"neighbor_mass={_corridor_teacher_distill_neighbor_mass:g} "
            f"stride={_corridor_teacher_distill_stride} "
            f"chunk={_corridor_teacher_distill_chunk_size} "
            f"horizon={_corridor_teacher_spec.horizon_s:.1f}s "
            f"intervention_only={_corridor_teacher_intervention_only} "
            f"clearance<{_corridor_teacher_intervention_clearance_m:.2f}m "
            "scope=long_corridor_only order=PPO->corridor->narrow_KL",
            flush=True,
        )

    @torch.no_grad()
    def _generate_corridor_teacher_labels(
        env_ids: torch.Tensor,
        policy_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """Generate robust privileged labels for selected corridor envs."""

        label_stats = {
            "candidates": int(env_ids.numel()),
            "feasible": 0,
            "teacher_differs": 0,
            "policy_obstacle_collision": 0,
            "policy_wall_collision": 0,
            "policy_low_clearance": 0,
            "interventions": 0,
            "selected": 0,
        }
        if env_ids.numel() == 0:
            return (
                env_ids,
                torch.empty(0, 2, dtype=torch.long, device=device),
                label_stats,
            )
        raw_env = env.unwrapped
        scheduler = getattr(raw_env, "_behavior_scheduler", None)
        if scheduler is None:
            raise RuntimeError(
                "corridor teacher requires rule_based BehaviorScheduler"
            )
        required_scene_fields = (
            "_long_corridor_wall_centers",
            "_long_corridor_wall_sizes",
            "_long_corridor_wall_mask",
            "_long_corridor_obstacle_counts",
        )
        missing = [
            name for name in required_scene_fields
            if not hasattr(raw_env, name)
        ]
        if missing:
            raise RuntimeError(
                f"corridor replay state is incomplete: {missing}"
            )

        static_count, dynamic_count = (
            raw_env._long_corridor_obstacle_counts
        )
        obstacle_slots = list(range(int(static_count))) + list(
            range(4, 4 + int(dynamic_count))
        )
        slot_ids = torch.tensor(
            obstacle_slots, dtype=torch.long, device=device
        )
        robot = raw_env.scene["robot"].data
        robot_xy = (
            robot.root_pos_w[:, :2]
            - raw_env.scene.env_origins[:, :2]
        )
        quaternion = robot.root_quat_w
        robot_yaw = torch.atan2(
            2.0
            * (
                quaternion[:, 0] * quaternion[:, 3]
                + quaternion[:, 1] * quaternion[:, 2]
            ),
            1.0
            - 2.0
            * (
                quaternion[:, 2].square()
                + quaternion[:, 3].square()
            ),
        )
        goal_term = raw_env.command_manager.get_term("goal_command")
        goal_xy = (
            goal_term.goal_pos_w[:, :2]
            - raw_env.scene.env_origins[:, :2]
        )
        physical_radii = getattr(
            raw_env, "_obstacle_phys_radii", None
        )
        if physical_radii is None:
            physical_radii = torch.full(
                (
                    raw_env.num_envs,
                    scheduler.positions.shape[1],
                ),
                0.30,
                dtype=robot_xy.dtype,
                device=device,
            )

        feasible_ids: list[torch.Tensor] = []
        teacher_actions: list[torch.Tensor] = []
        cfg = _corridor_teacher_action_term.cfg
        for start in range(
            0, env_ids.numel(), _corridor_teacher_distill_chunk_size
        ):
            chunk_ids = env_ids[
                start:start + _corridor_teacher_distill_chunk_size
            ]

            def _select_slots(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.index_select(0, chunk_ids).index_select(
                    1, slot_ids
                )

            moving_paths, moving_valid = predict_patrol_obstacle_paths(
                _select_slots(scheduler.positions),
                _select_slots(scheduler.behavior_type),
                _select_slots(scheduler.patrol_waypoints),
                _select_slots(scheduler.patrol_wp_index),
                _select_slots(scheduler.patrol_num_waypoints),
                _select_slots(scheduler.patrol_speed),
                _select_slots(scheduler.patrol_pause_remaining),
                dt=float(_corridor_teacher_action_term._dt),
                samples=_corridor_teacher_spec.samples,
                new_waypoint_pause_steps=0,
            )
            paused_paths, paused_valid = predict_patrol_obstacle_paths(
                _select_slots(scheduler.positions),
                _select_slots(scheduler.behavior_type),
                _select_slots(scheduler.patrol_waypoints),
                _select_slots(scheduler.patrol_wp_index),
                _select_slots(scheduler.patrol_num_waypoints),
                _select_slots(scheduler.patrol_speed),
                _select_slots(scheduler.patrol_pause_remaining),
                dt=float(_corridor_teacher_action_term._dt),
                samples=_corridor_teacher_spec.samples,
                new_waypoint_pause_steps=5,
            )
            obstacle_paths = torch.cat(
                [moving_paths, paused_paths], dim=1
            )
            obstacle_valid = torch.cat(
                [moving_valid, paused_valid], dim=1
            )
            radii = _select_slots(physical_radii)
            radii = torch.cat([radii, radii], dim=1)
            result = corridor_teacher_action_grid(
                current_velocity=(
                    _corridor_teacher_action_term._current_velocity[
                        chunk_ids
                    ]
                ),
                current_omega=(
                    _corridor_teacher_action_term._current_omega[chunk_ids]
                ),
                robot_xy_m=robot_xy[chunk_ids],
                robot_yaw_rad=robot_yaw[chunk_ids],
                goal_xy_m=goal_xy[chunk_ids],
                obstacle_paths_m=obstacle_paths,
                obstacle_radii_m=radii,
                obstacle_valid=obstacle_valid,
                wall_centers_m=raw_env._long_corridor_wall_centers[
                    chunk_ids
                ],
                wall_sizes_m=raw_env._long_corridor_wall_sizes[chunk_ids],
                wall_valid=raw_env._long_corridor_wall_mask[chunk_ids],
                num_bins=int(cfg.num_bins),
                dt=float(_corridor_teacher_action_term._dt),
                max_linear_velocity=float(cfg.max_linear_velocity),
                reverse_velocity_scale=float(cfg.reverse_velocity_scale),
                max_linear_accel=float(cfg.max_linear_accel),
                max_angular_velocity=float(cfg.max_angular_vel),
                max_angular_accel=float(cfg.max_angular_accel),
                spec=_corridor_teacher_spec,
            )
            feasible = result["any_feasible"]
            intervention = select_corridor_interventions(
                policy_actions.index_select(0, chunk_ids),
                result["actions"],
                feasible,
                result["obstacle_collision_grid"],
                result["wall_collision_grid"],
                result["min_obstacle_clearance_grid"],
                clearance_threshold_m=(
                    _corridor_teacher_intervention_clearance_m
                ),
            )
            selected = (
                intervention["intervention"]
                if _corridor_teacher_intervention_only
                else feasible
            )
            label_stats["feasible"] += int(feasible.sum().item())
            for key in (
                "teacher_differs",
                "policy_obstacle_collision",
                "policy_wall_collision",
                "policy_low_clearance",
            ):
                label_stats[key] += int(intervention[key].sum().item())
            label_stats["interventions"] += int(
                intervention["intervention"].sum().item()
            )
            label_stats["selected"] += int(selected.sum().item())
            if selected.any():
                feasible_ids.append(chunk_ids[selected])
                teacher_actions.append(result["actions"][selected])
        if not feasible_ids:
            return (
                env_ids[:0],
                torch.empty(0, 2, dtype=torch.long, device=device),
                label_stats,
            )
        return (
            torch.cat(feasible_ids, dim=0),
            torch.cat(teacher_actions, dim=0).long(),
            label_stats,
        )

    if args_cli.scene_probe_stride <= 0:
        raise ValueError("scene_probe_stride must be positive")
    if args_cli.scene_probe_output and not _e2e_frame_stack:
        raise ValueError("scene probe currently requires the K8 E2E lineage")
    _scene_probe_saved = False
    _scene_probe_inputs: list[torch.Tensor] = []
    _scene_probe_labels: list[torch.Tensor] = []
    _scene_probe_env_ids: list[torch.Tensor] = []
    _scene_probe_steps: list[torch.Tensor] = []

    for iteration in range(num_iterations):
        iter_start = time.time()
        charge_buf.reset()         # 重置 Charge rollout buffer（ptr=0）
        if _lidar_hist is not None and not _e2e_frame_stack:
            _lidar_hist.zero_()    # 多幀:rollout 起始 LiDAR 歷史歸零(對齊 recompute 從零起,三路一致)
        if obs_buf is not None:
            obs_buf.reset()        # 重置 Obstacle rollout buffer
        metrics.reset()            # 清空 MetricsCollector（完成 episode 統計）
        _arc_stat_sum, _arc_stat_fire, _arc_stat_n = 0.0, 0.0, 0  # ★r_arc per-iter step 統計
        _arc_ep_sum, _arc_ep_n = 0.0, 0  # ★r_arc per-iter 完成 episode 的累積 penalty 統計
        _corridor_teacher_sample_indices: list[torch.Tensor] = []
        _corridor_teacher_actions: list[torch.Tensor] = []
        _corridor_teacher_label_stats = {
            "candidates": 0,
            "feasible": 0,
            "teacher_differs": 0,
            "policy_obstacle_collision": 0,
            "policy_wall_collision": 0,
            "policy_low_clearance": 0,
            "interventions": 0,
            "selected": 0,
        }
        _teacher_forced_count = 0

        # === Determine who trains this iteration（Warp Drive 交替訓練）===
        # WD: 每 train_goal_rate(=3) 次 iteration 中，1 次訓練 obstacle，其餘訓練 charge
        # iteration % 3 == 1 時訓練 obstacle（0, 2 訓練 charge）
        _n_dynamic = int(metrics._curriculum_info.get("num_obstacles_dynamic", N_obs))
        _obs_agent_active = (_n_dynamic > 0) and (_obstacle_mode == "learned")
        train_charge = (iteration % args_cli.train_goal_rate != 1) or not _obs_agent_active
        train_obstacle = (iteration % args_cli.train_goal_rate == 1) and _obs_agent_active
        if not _obs_agent_active:
            train_charge = True  # 沒有動態障礙物時，charge 每次都訓練

        # === WD: Reset optimizer momentum on phase change（對應 WD: reset_model_mentum）===
        # 在 phase 切換時，清空 Adam 的動量（exp_avg/exp_avg_sq），
        # 避免舊 phase 積累的動量方向影響新 phase 的梯度更新。
        _cur_stage_default = _prev_stage if _prev_stage > 0 else 1
        _cur_stage = int(metrics._curriculum_info.get("stage", _cur_stage_default))
        _stage_changed = (_cur_stage != _prev_stage)
        if _stage_changed and _prev_stage > 0:
            for opt in [charge_opt_rl, charge_opt_aux] + ([obs_optimizer] if obs_optimizer else []):
                for group in opt.param_groups:
                    for p in group["params"]:
                        state = opt.state.get(p)
                        if state:
                            if "exp_avg" in state:
                                state["exp_avg"].zero_()      # Adam 一階動量清零
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"].zero_()   # Adam 二階動量清零
            print(f"[INFO] Phase {_prev_stage}→{_cur_stage}: optimizer momentum reset (WD: reset_model_mentum)")
            if _obs_agent_active != _prev_obs_agent_active:
                print(f"[INFO] obs_agent {'ACTIVATED' if _obs_agent_active else 'DEACTIVATED'} "
                      f"(dynamic: {_n_dynamic})")
        _prev_stage = _cur_stage
        _prev_obs_agent_active = _obs_agent_active

        # === Sync params from curriculum (gamma is constant per WD) ===

        # Update obs_size_rand / scene_bound_rand from curriculum (if not CLI-overridden)
        if args_cli.obs_size_rand == 0.0:
            _obs_size_rand = metrics._curriculum_info.get("obs_size_rand", 0.0)
        if args_cli.scene_bound_rand == 0.0:
            _scene_bound_rand = metrics._curriculum_info.get("scene_bound_rand", 0.0)

        # Sync WD reward params from curriculum (per-phase) via reward module
        _spot_penalty_hit = metrics._curriculum_info.get("spot_penalty_hit", -5.0)
        # ★07-05 CLI 覆寫(同 penalty_speed_near_obs 模式): !=0 時壓過 curriculum。
        #   用途: deployv3 膨脹臂 penalty -5→-15 (期望損失 -5×32%≈-1.6 太便宜, policy 吃罰不繞)。
        if getattr(args_cli, "penalty_hit", 0.0) != 0.0:
            _spot_penalty_hit = args_cli.penalty_hit
            metrics._curriculum_info["spot_penalty_hit"] = _spot_penalty_hit
        _spot_reward_get_goal = metrics._curriculum_info.get("spot_reward_get_goal", 40.0)
        _spot_cost_operate = metrics._curriculum_info.get("spot_cost_operate", 0.0)
        _spot_penalty_timeout = metrics._curriculum_info.get("spot_penalty_timeout", 0.0)
        _spot_penalty_smoothness = metrics._curriculum_info.get("spot_penalty_smoothness", 0.0)  # v3
        _spot_penalty_speed_near_obs = metrics._curriculum_info.get(
            "spot_penalty_speed_near_obs", 0.0)  # v3f-react
        if getattr(args_cli, "penalty_speed_near_obs", -1.0) >= 0.0:
            # CLI 覆寫:reactive 減速懲罰不靠 curriculum,可在任何 stage 套用(SA4-reactive,A 方案)。
            #   注入 curriculum_info 讓 reward module update_params 拿到。
            _spot_penalty_speed_near_obs = args_cli.penalty_speed_near_obs
            metrics._curriculum_info["spot_penalty_speed_near_obs"] = _spot_penalty_speed_near_obs
        _reward_module.update_params(metrics._curriculum_info)

        # Sync WD entropy params from curriculum (per-phase, A2CK per-head)
        _ent_before_lin = _ent_coeff_linear
        _ent_before_ang = _ent_coeff_angular
        if args_cli.ent_coeff_linear == 0.0:
            _ent_coeff_linear = metrics._curriculum_info.get("ent_coeff_linear", 0.01)
        if args_cli.ent_coeff_angular == 0.0:
            _ent_coeff_angular = metrics._curriculum_info.get("ent_coeff_angular", 0.02)
        # Legacy fallback: if --ent_coeff is set, use it for both heads
        if args_cli.ent_coeff > 0:
            _ent_coeff_linear = args_cli.ent_coeff
            _ent_coeff_angular = args_cli.ent_coeff
        # DEBUG: log ent_coeff sync result (first 5 iters + every 50)
        if iteration <= 5 or iteration % 50 == 0:
            _ci_has = "ent_coeff_linear" in metrics._curriculum_info
            print(
                f"[DEBUG ent_coeff sync] iter={iteration} "
                f"before=({_ent_before_lin:.4f},{_ent_before_ang:.4f}) "
                f"after=({_ent_coeff_linear:.4f},{_ent_coeff_angular:.4f}) "
                f"ci_has_key={_ci_has} "
                f"ci_val={metrics._curriculum_info.get('ent_coeff_linear', 'MISSING')}"
            )

        # Sync obstacle speed rate from curriculum (per-phase)
        # WD: obs_speed_rate 0.8 → 0.85 → 1.15 across phases
        _obs_speed_limit = metrics._curriculum_info.get(
            "obstacle_speed_rate", args_cli.obs_speed_limit)
        metrics._obs_speed_limit = _obs_speed_limit  # sync for speed metric

        # Sync safe trainer hyperparameters from phase/task registry
        if not _e2e_frame_stack:
            _sync_phase_trainer_params(metrics._curriculum_info, stage_changed=_stage_changed)

        # === DORAEMON：檢查是否擴展 DR 範圍 ===
        if _doraemon_ctrl is not None and not _doraemon_ctrl.fully_expanded:
            _recent_sr = metrics.get_success_rate() if hasattr(metrics, 'get_success_rate') else 0.0
            if _recent_sr == 0.0:
                _sr_data = metrics._curriculum_info.get("success_rate", 0.0)
                if _sr_data > 0:
                    _recent_sr = _sr_data
            expanded = _doraemon_ctrl.maybe_expand(iteration, _recent_sr)
            if expanded:
                _doraemon_ctrl.apply_to_env(env.unwrapped)
                print(_doraemon_ctrl.summary())

        # === LR decay (WD: ParamScheduler) ===
        if args_cli.lr_decay > 0 and iteration > 0:
            decay = max(0.01, 1.0 - args_cli.lr_decay * iteration)
            for opt in [charge_opt_rl, charge_opt_aux]:
                for pg in opt.param_groups:
                    pg["lr"] = pg.get("initial_lr", pg["lr"]) * decay

        # === Rollout: 兩個 policy 都 act，交替決定誰 train ===
        # Rollout 期間所有模型設 eval()（不需要 batch norm/dropout 行為）
        if use_extractor: extractor.eval()
        preprocess_rnn.eval(); policy_head.eval(); value_head.eval()
        if corridor_adapter is not None:
            corridor_adapter.eval()
        if obs_policy is not None:
            obs_policy.eval(); obs_value.eval()

        for step in range(RL):
            # --- 0. Obs delay DR：policy 看到延遲觀測，模擬真實感測器管線延遲 ---
            if _use_obs_delay:
                _w = _obs_ring_ptr % (_delay_hi + 1)
                _obs_ring[_w, _env_arange] = obs              # 寫入當前真實觀測
                _r = (_obs_ring_ptr - _delay_per_env) % (_delay_hi + 1)
                policy_obs = _obs_ring[_r, _env_arange]       # 讀取延遲觀測
                _obs_ring_ptr += 1
            else:
                policy_obs = obs

            # --- 1. Charge forward（推論模式，torch.no_grad() 加速）---
            with torch.no_grad():
                obs_normalizer.update(policy_obs)   # 更新 running stats（用 policy 看到的觀測）
                obs_normed = obs_normalizer.normalize(policy_obs)  # 標準化觀測
                _encoder_input = (
                    _build_extractor_input(obs_normed, _lidar_hist)
                    if _e2e_frame_stack else None
                )
                features = _charge_features_for_rnn(obs_normed, update_norm=True, lidar_hist=_lidar_hist)  # 多幀:帶歷史
                if _lidar_hist is not None:
                    # prepend 當前 normed lidar、drop oldest → 供下一步用([t, t-1, ...])
                    _cur_lidar = _select_79d(obs_normed)[:, LIDAR_START:LIDAR_END]
                    _lidar_hist = torch.cat([_cur_lidar, _lidar_hist[:, :-_LL]], dim=-1)
                hidden = rnn_state.get()                          # 取得當前 hidden state [1, E, H]
                p_obs = _charge_obs_for_rl(obs_normed)    # 取得 RL head 的觀測部分（79D 或 113D）
                if _e2e_frame_stack:
                    new_hidden = hidden
                    rl_in = torch.cat([p_obs, features], dim=-1)
                else:
                    rnn_feat, _pred_rl, new_hidden = preprocess_rnn(
                        features, hidden, training=args_cli.hybrid_predict_to_policy)  # hybrid 時順便算 prediction
                    # Ablation: zero out RNN feature for RL（aux path 照常訓練，只是 RL 看不到）
                    _rnn_for_rl = (torch.zeros_like(rnn_feat)
                                   if args_cli.zero_preprocess_feature_for_rl else rnn_feat)
                    rl_in = (torch.cat([p_obs, _rnn_for_rl, _pred_rl], dim=-1)
                             if args_cli.hybrid_predict_to_policy
                             else torch.cat([p_obs, _rnn_for_rl], dim=-1))
                _priv_obs = extract_privileged_obs(env.unwrapped) if (_use_asymmetric_critic or _oracle_to_policy) else None
                _rl_in_enc = _encode_rl_input(rl_in)
                logits = policy_head(_rl_in_enc, _priv_obs if _oracle_to_policy else None)  # [E, 38] policy logits（雙頭各 19）
                if corridor_adapter is not None:
                    (
                        _corridor_residual_step,
                        _corridor_gate_logits_step,
                        _corridor_gate_probability_step,
                    ) = corridor_adapter(
                        p_obs,
                        (
                            _rl_in_enc
                            if _corridor_adapter_residual_features
                            == "policy_features"
                            else None
                        ),
                    )
                    logits = logits + _corridor_residual_step
                value = value_head(_rl_in_enc, _priv_obs if _use_asymmetric_critic else None).squeeze(-1)  # [E] critic value
                actions, log_prob, _ = sample_action(logits)     # 採樣動作 + joint log_prob
                _corridor_policy_actions_step = torch.stack(
                    [
                        logits[:, :NUM_BINS].argmax(dim=-1),
                        logits[:, NUM_BINS:].argmax(dim=-1),
                    ],
                    dim=-1,
                )
                goal_diagnostics = metrics.compute_goal_diagnostics(env.unwrapped)  # 目標診斷（不影響 reward）
                _teacher_logits_step = None
                _retention_mask_step = None
                _previous_teacher_logits_step = None
                _previous_retention_mask_step = None
                if _teacher_retention_enabled:
                    _teacher_obs_normed = torch.clamp(
                        (policy_obs - _teacher_obs_mean)
                        / (_teacher_obs_var.sqrt() + 1e-8),
                        -5.0,
                        5.0,
                    )
                    _teacher_encoder_input = _build_extractor_input(
                        _teacher_obs_normed, _teacher_lidar_hist
                    )
                    _teacher_features = _teacher_extractor(
                        _teacher_encoder_input
                    )
                    _teacher_policy_obs = _charge_obs_for_rl(
                        _teacher_obs_normed
                    )
                    _teacher_logits_step = _teacher_policy_head(
                        torch.cat(
                            [_teacher_policy_obs, _teacher_features], dim=-1
                        )
                    )
                    _teacher_cur_lidar = _teacher_policy_obs[
                        :, LIDAR_START:LIDAR_END
                    ]
                    _teacher_lidar_hist = torch.cat(
                        [
                            _teacher_cur_lidar,
                            _teacher_lidar_hist[:, :-_LL],
                        ],
                        dim=-1,
                    )
                    _narrow_active = getattr(
                        env.unwrapped, "_narrow_bridge_active", None
                    )
                    if _narrow_active is None:
                        raise RuntimeError(
                            "teacher retention is enabled but the narrow bridge "
                            "injector did not create _narrow_bridge_active"
                        )
                    _retention_mask_step = _narrow_active.clone()
                if _previous_stage_teacher_enabled:
                    _previous_teacher_obs_normed = torch.clamp(
                        (policy_obs - _previous_teacher_obs_mean)
                        / (_previous_teacher_obs_var.sqrt() + 1e-8),
                        -5.0,
                        5.0,
                    )
                    _previous_teacher_encoder_input = _build_extractor_input(
                        _previous_teacher_obs_normed,
                        _previous_teacher_lidar_hist,
                    )
                    _previous_teacher_features = _previous_teacher_extractor(
                        _previous_teacher_encoder_input
                    )
                    _previous_teacher_policy_obs = _charge_obs_for_rl(
                        _previous_teacher_obs_normed
                    )
                    _previous_teacher_logits_step = (
                        _previous_teacher_policy_head(
                            torch.cat(
                                [
                                    _previous_teacher_policy_obs,
                                    _previous_teacher_features,
                                ],
                                dim=-1,
                            )
                        )
                    )
                    _previous_teacher_cur_lidar = _previous_teacher_policy_obs[
                        :, LIDAR_START:LIDAR_END
                    ]
                    _previous_teacher_lidar_hist = torch.cat(
                        [
                            _previous_teacher_cur_lidar,
                            _previous_teacher_lidar_hist[:, :-_LL],
                        ],
                        dim=-1,
                    )
                    if _previous_stage_teacher_scope == "previous_stage":
                        _previous_active = getattr(
                            env.unwrapped,
                            "_previous_stage_replay_active",
                            None,
                        )
                        missing_name = "_previous_stage_replay_active"
                    elif _previous_stage_teacher_scope == "non_narrow":
                        _narrow_active = getattr(
                            env.unwrapped, "_narrow_bridge_active", None
                        )
                        _previous_active = (
                            None
                            if _narrow_active is None
                            else ~_narrow_active
                        )
                        missing_name = "_narrow_bridge_active"
                    elif _previous_stage_teacher_scope == "corridor":
                        _previous_active = getattr(
                            env.unwrapped,
                            "_long_corridor_active",
                            None,
                        )
                        missing_name = "_long_corridor_active"
                    else:
                        _previous_active = torch.ones(
                            num_envs, dtype=torch.bool, device=device
                        )
                        missing_name = ""
                    if _previous_active is None:
                        raise RuntimeError(
                            "second teacher retention is enabled but the "
                            f"environment did not create {missing_name}"
                        )
                    _previous_retention_mask_step = _previous_active.clone()
                if _teacher_retention_rollout_override:
                    if (
                        _teacher_logits_step is None
                        or _retention_mask_step is None
                    ):
                        raise RuntimeError(
                            "teacher rollout override requires teacher logits "
                            "and the narrow replay mask at every step"
                        )
                    actions = apply_masked_deterministic_teacher_actions(
                        actions,
                        _teacher_logits_step,
                        _retention_mask_step,
                        num_bins=NUM_BINS,
                    )
                    log_prob, _, _ = evaluate_actions(logits, actions)
                    _teacher_forced_count += int(
                        _retention_mask_step.sum().item()
                    )

            # Capture scene identity before env.step. Done environments are reset
            # inside env.step, so the live mask afterward may describe a new episode.
            _long_corridor_mask_step = getattr(
                env.unwrapped, "_long_corridor_active", None
            )
            if _long_corridor_mask_step is not None:
                _long_corridor_mask_step = _long_corridor_mask_step.clone()
            if (
                args_cli.scene_probe_output
                and not _scene_probe_saved
                and step % args_cli.scene_probe_stride == 0
            ):
                _probe_env = env.unwrapped
                _probe_empty = torch.zeros(
                    num_envs, dtype=torch.bool, device=device
                )
                _probe_previous = getattr(
                    _probe_env,
                    "_previous_stage_replay_active",
                    _probe_empty,
                ).clone()
                _probe_narrow = getattr(
                    _probe_env,
                    "_narrow_bridge_active",
                    _probe_empty,
                ).clone()
                _probe_corridor = (
                    _long_corridor_mask_step
                    if _long_corridor_mask_step is not None
                    else _probe_empty
                )
                _probe_overlap = (
                    (_probe_previous & _probe_narrow)
                    | (_probe_previous & _probe_corridor)
                    | (_probe_narrow & _probe_corridor)
                )
                if bool(_probe_overlap.any()):
                    raise RuntimeError(
                        "scene probe found overlapping replay labels"
                    )
                _probe_label = torch.zeros(
                    num_envs, dtype=torch.uint8, device=device
                )
                _probe_label[_probe_previous] = 1
                _probe_label[_probe_narrow] = 2
                _probe_label[_probe_corridor] = 3
                _scene_probe_inputs.append(
                    _encoder_input.detach().to(device="cpu")
                )
                _scene_probe_labels.append(_probe_label.cpu())
                _scene_probe_env_ids.append(
                    torch.arange(
                        num_envs, dtype=torch.int32, device="cpu"
                    )
                )
                _scene_probe_steps.append(
                    torch.full(
                        (num_envs,),
                        step,
                        dtype=torch.int16,
                        device="cpu",
                    )
                )
            if (
                _corridor_teacher_distill_enabled
                and step % _corridor_teacher_distill_stride == 0
            ):
                if _long_corridor_mask_step is None:
                    raise RuntimeError(
                        "corridor teacher is enabled but the replay injector "
                        "did not create _long_corridor_active"
                    )
                _corridor_env_ids = (
                    _long_corridor_mask_step.nonzero(
                        as_tuple=False
                    ).flatten()
                )
                (
                    _corridor_feasible_ids,
                    _corridor_actions_step,
                    _corridor_label_stats_step,
                ) = _generate_corridor_teacher_labels(
                    _corridor_env_ids,
                    _corridor_policy_actions_step,
                )
                for _label_key, _label_value in (
                    _corridor_label_stats_step.items()
                ):
                    _corridor_teacher_label_stats[_label_key] += (
                        _label_value
                    )
                if _corridor_feasible_ids.numel() > 0:
                    _corridor_teacher_sample_indices.append(
                        step * num_envs + _corridor_feasible_ids
                    )
                    _corridor_teacher_actions.append(
                        _corridor_actions_step
                    )

            # Capture dynamic obstacle state before env.step so future occupancy
            # compares the selected action against the same state seen by policy.
            _future_occupancy_ctx = {}
            if args_cli.future_occupancy_weight > 0.0:
                with torch.no_grad():
                    _scheduler = getattr(env.unwrapped, "_behavior_scheduler", None)
                    if _scheduler is None:
                        raise RuntimeError("future occupancy reward requires rule_based BehaviorScheduler")
                    _robot_data_future = env.unwrapped.scene["robot"].data
                    _robot_local_xy = (
                        _robot_data_future.root_pos_w[:, :2] - env.unwrapped.scene.env_origins[:, :2]
                    )
                    _delta_world = _scheduler.positions[:, :, :2] - _robot_local_xy[:, None, :]
                    _velocity_world = _scheduler.velocities[:, :, :2]
                    _q_future = _robot_data_future.root_quat_w
                    _yaw_future = torch.atan2(
                        2.0 * (_q_future[:, 0] * _q_future[:, 3] + _q_future[:, 1] * _q_future[:, 2]),
                        1.0 - 2.0 * (_q_future[:, 2].square() + _q_future[:, 3].square()),
                    )
                    _cos_future = torch.cos(_yaw_future)[:, None]
                    _sin_future = torch.sin(_yaw_future)[:, None]
                    _future_occupancy_ctx["dynamic_obstacle_positions_body_m"] = torch.stack(
                        [
                            _cos_future * _delta_world[:, :, 0] + _sin_future * _delta_world[:, :, 1],
                            -_sin_future * _delta_world[:, :, 0] + _cos_future * _delta_world[:, :, 1],
                        ],
                        dim=-1,
                    )
                    _future_occupancy_ctx["dynamic_obstacle_velocities_body_mps"] = torch.stack(
                        [
                            _cos_future * _velocity_world[:, :, 0] + _sin_future * _velocity_world[:, :, 1],
                            -_sin_future * _velocity_world[:, :, 0] + _cos_future * _velocity_world[:, :, 1],
                        ],
                        dim=-1,
                    )

            # --- N1: scripted 直穿 teacher 標籤（窄縫 replay 幀才有效）---
            # 必須在 env.step 之前算：標籤要對應 policy 當下看到的狀態，才與同一
            # buffer slot 的 rl_inputs/actions 對齊。step 之後讀會晚一步，且已終止
            # 的 env 早被 reset（位姿與 _narrow_bridge_active 都屬於下一回合）。
            # 只讀狀態、不改動作：PPO 仍用自己抽樣的 actions 與環境互動。
            _narrow_teacher_actions_step = None
            _narrow_imitation_mask_step = None
            if _narrow_imitation_enabled:
                with torch.no_grad():
                    _ni_geom = narrow_bridge_teacher_geometry(env.unwrapped)
                    _ni_robot = env.unwrapped.scene["robot"]
                    _ni_xy = (
                        _ni_robot.data.root_pos_w[:, :2]
                        - env.unwrapped.scene.env_origins[:, :2]
                    )
                    _ni_q = _ni_robot.data.root_quat_w
                    _ni_yaw = torch.atan2(
                        2.0 * (_ni_q[:, 0] * _ni_q[:, 3] + _ni_q[:, 1] * _ni_q[:, 2]),
                        1.0 - 2.0 * (_ni_q[:, 2] ** 2 + _ni_q[:, 3] ** 2),
                    )
                    _ni_cfg = _narrow_imitation_action_term.cfg
                    _narrow_teacher_actions_step = (
                        scripted_narrow_gap_action_indices(
                            _ni_xy,
                            _ni_yaw,
                            _narrow_imitation_action_term._current_velocity,
                            _narrow_imitation_action_term._current_omega,
                            barrier_x_m=_ni_geom["barrier_x_m"],
                            gap_center_y_m=_ni_geom["gap_center_y_m"],
                            goal_xy_m=_ni_geom["goal_xy_m"],
                            num_bins=int(_ni_cfg.num_bins),
                            dt=float(_narrow_imitation_action_term._dt),
                            max_linear_velocity=float(_ni_cfg.max_linear_velocity),
                            reverse_velocity_scale=float(
                                _ni_cfg.reverse_velocity_scale
                            ),
                            max_linear_accel=float(_ni_cfg.max_linear_accel),
                            max_angular_velocity=float(_ni_cfg.max_angular_vel),
                            max_angular_accel=float(_ni_cfg.max_angular_accel),
                            spec=_narrow_imitation_spec,
                        )
                    )
                    _narrow_imitation_mask_step = _ni_geom["active"].clone()

            # --- 2. Env step ---
            next_obs, reward, terminated, truncated, info = env.step(actions.float())
            _audit_long_corridor_goal()
            done = (terminated.squeeze(-1) | truncated.squeeze(-1)).float()  # [E] episode 結束(term|trunc, reset/stats/aux 用)
            _terminated_flat = terminated.squeeze(-1).float()                # ★fix#1: 真terminal(撞/到達),GAE bootstrap 用
            env_reward_flat = reward.squeeze(-1)  # Isaac Lab env reward（dense，只用於 logging）
            charge_action_diagnostics = compute_charge_action_diagnostics(env.unwrapped, actions)  # 物理動作診斷

            # Reward computation via modular dispatch (Phase 2)
            # v3: pass prev_actions via context for frame-to-frame smoothness penalty
            # v3f-react: pass per-env nearest-obstacle distance (m) + forward speed (m/s)
            #   for clearance-gated speed penalty. obs layout: [1]=v_x norm(/v_max=1.0 → m/s),
            #   [6:78]=lidar norm[0,1] (×max_distance=20.0 → m). Hole-mask <0.02 like metrics.
            _reward_ctx = {
                "prev_actions": _prev_actions,
                "obs": obs,
                "next_obs": next_obs,
            }
            if charge_action_diagnostics is not None:
                _reward_ctx["v_forward_actual_m"] = charge_action_diagnostics["v_x"]
                _reward_ctx["omega_actual_rad_s"] = charge_action_diagnostics["omega"]
            _reward_ctx.update(_future_occupancy_ctx)
            with torch.no_grad():
                _lidar_norm = obs[:, 6:78]                                       # [E,72] norm[0,1]
                _lidar_clean = torch.where(
                    _lidar_norm < 0.02, torch.full_like(_lidar_norm, float("inf")), _lidar_norm
                )
                _dmin_norm = _lidar_clean.min(dim=-1).values                     # [E]
                _dmin_norm = torch.where(
                    torch.isinf(_dmin_norm), torch.ones_like(_dmin_norm), _dmin_norm
                )
                _reward_ctx["near_obs_dist_m"] = _dmin_norm * _LIDAR_MAX_DISTANCE_M  # [E] meters
                _reward_ctx["v_forward_m"] = obs[:, 1]                           # [E] m/s (v_max=1.0)
                # ★07-06 水滴稅 gate: per-bin p(d)²·max(cosθ,0) 取 max。前伸側窄(cos)+速度門控(reward端×v_fwd)。
                #   啟用: --near_obs_teardrop; d_react/d_stop 由 curriculum/CLI(near_obs_d_react)。θ: bin36=正前。
                if getattr(args_cli, "near_obs_teardrop", False):
                    _d_m = _lidar_clean * _LIDAR_MAX_DISTANCE_M                  # [E,72] meters (inf=hole)
                    _dr = float(getattr(args_cli, "near_obs_d_react", 2.0))
                    _ds = 0.45
                    _shape = getattr(args_cli, "near_obs_penalty_shape", "sq")
                    # 距離因子: sq=p²(後載,舊) / linear=p / log=log(dr/d)(前載最強,2m 就實質咬)
                    if _shape == "log":
                        _dc = _d_m.clamp(min=_ds)                              # 夾 d_stop 防 log→∞
                        _cap = math.log(_dr / _ds)
                        _pf = (torch.log(_dr / _dc).clamp(min=0.0) / max(_cap, 1e-6))  # [0,1] 正規化
                        _pf = torch.where(_d_m >= _dr, torch.zeros_like(_pf), _pf)     # 反應區外=0
                    else:
                        _p = ((_dr - _d_m) / max(_dr - _ds, 1e-3)).clamp(0.0, 1.0)
                        _pf = _p if _shape == "linear" else _p.pow(2)
                    if _TEARDROP_COS is None:
                        _bang = (torch.arange(72, device=obs.device).float() * 5.0 - 180.0) * (math.pi / 180.0)
                        _TEARDROP_COS = torch.cos(_bang).clamp(min=0.0).unsqueeze(0)   # [1,72] max(cosθ,0), bin36=正前=1
                    _threat = _pf * _TEARDROP_COS                              # [E,72] 前伸側窄
                    _gate = _threat.max(dim=-1).values.clamp(0.0, 1.0)         # [E]
                    # ★07-07 stage 稅閘(用戶建議): 只在 curriculum stage >= N 施稅。比 step-warmup 更符課程
                    #   學習——stage1-2 純學導航、stage3+ 障礙變多才教躲。完整課程血緣用此(暖啟固定stage用 step-warmup)。
                    _min_stage = int(getattr(args_cli, "teardrop_min_stage", 0))
                    if _min_stage > 0 and _cur_stage < _min_stage:
                        _gate = _gate * 0.0
                    # ★07-07 warmup: 前期 w=0 讓政策先學到達,避免 dense 稅在 from-scratch/暖啟早期
                    #   誘發凍結(先學「別往前=不繳稅」卡局部最優)。線性 ramp over [start, end] iter。
                    _wu_s = int(getattr(args_cli, "teardrop_warmup_start", 0))
                    _wu_e = int(getattr(args_cli, "teardrop_warmup_end", 0))
                    if _wu_e > _wu_s:
                        _ramp = min(1.0, max(0.0, (iteration - _wu_s) / (_wu_e - _wu_s)))
                        _gate = _gate * _ramp
                    _reward_ctx["teardrop_gate"] = _gate
            reward_flat, reward_breakdown = _reward_module.compute(
                env.unwrapped, actions, terminated, truncated,
                context=_reward_ctx,
            )
            add_long_corridor_reward_diagnostics(
                reward_breakdown,
                _long_corridor_mask_step,
            )
            # Update prev_actions for next step (clone to detach from autograd graph)
            _prev_actions = actions.detach().clone()

            # --- 2c. r_arc: action-conditioned swept-arc 預測軌跡安全度（教「往哪邊避」，非背景距離稅）---
            #   紅線①只用 policy 看到的 72-beam LiDAR；紅線②用實際套用 (v_x,ω)=processed_actions。
            if _arc_action_term is not None:
                with torch.no_grad():
                    # Policy obs 是車身淨空：(sensor_range - body_radius) / r_max。
                    # Arc 幾何從車體中心算到 hit point，故先還原 sensor range；body radius 會在 arc_clearance 扣一次。
                    _lidar_m_arc = swept_arc.policy_lidar_to_sensor_range(
                        obs[:, 6:78],
                        max_range=_LIDAR_MAX_DISTANCE_M,
                        body_radius=args_cli.arc_body_radius,
                    )
                    _pa_arc = _arc_action_term.processed_actions                # [E,2] 實際 (v_x, ω)
                    _c_arc = swept_arc.arc_clearance(
                        _lidar_m_arc, _pa_arc[:, 0].float(), _pa_arc[:, 1].float(),
                        horizon=args_cli.arc_horizon, body_radius=args_cli.arc_body_radius,
                        max_range=_LIDAR_MAX_DISTANCE_M)
                    _r_arc = swept_arc.r_arc_from_clearance(
                        _c_arc, c_safe=args_cli.arc_c_safe, w_arc=args_cli.arc_w, cap=args_cli.arc_cap)
                reward_flat = reward_flat + _r_arc
                _ep_arc_accum = _ep_arc_accum + _r_arc     # 每集累積(done 時記錄歸零)
                _arc_stat_sum += float(_r_arc.mean().item())
                _arc_stat_fire += float((_r_arc < -1e-6).float().mean().item())  # arc_active_fraction
                _arc_stat_n += 1

            # --- 2b. Heading stability：懲罰角速度符號翻轉（抗震盪）---
            # obs layout: [0]=accel [1]=speed [2]=omega ...
            # 用真實 obs 的 omega（不是 delayed），因為要量測實際機器人行為
            if _heading_stability_weight != 0:
                _curr_omega = obs[:, 2]
                _sign_flip = (_prev_omega * _curr_omega < 0).float()
                _heading_pen = _heading_stability_weight * _sign_flip * _curr_omega.abs()
                reward_flat = reward_flat + _heading_pen
                reward_breakdown["heading_stability"] = _heading_pen
                _prev_omega = _curr_omega.clone()

            # --- 2c. ★TTC 門檻稅 (LV-DOT 密集場景: 逼 policy 用 channel 速度預判) ---
            # 從 channel obs[79:109] 逐障礙算碰撞剩餘時間 TTC=d/v_closing(<門檻扣分)。
            # closing speed 只有 channel 有 → policy 要減稅非讀 channel 不可 = 逼出速度預判。
            # channel slot k: obs[79+6k : 85+6k] = [px÷8, py÷8, vx÷1.5, vy÷1.5, r, valid]。
            _ttc_w = float(getattr(args_cli, "ttc_tax_weight", 0.0))
            if _ttc_w > 0.0 and obs.shape[-1] >= 109:
                with torch.no_grad():
                    _ttc_thresh = float(getattr(args_cli, "ttc_thresh", 2.0))
                    _ch = obs[:, 79:109].view(obs.shape[0], 5, 6)     # [E,5,6] K=5 slots
                    _px = _ch[:, :, 0] * 8.0                          # body 縱向 m (+前)
                    _py = _ch[:, :, 1] * 8.0                          # body 橫向 m
                    _vx = _ch[:, :, 2] * 1.5                          # body 相對速度 x m/s
                    _vy = _ch[:, :, 3] * 1.5
                    _valid = _ch[:, :, 5] > 0.5                       # [E,5]
                    _d = torch.sqrt(_px * _px + _py * _py + 1e-6)     # [E,5] 徑向距離
                    # 徑向接近速度 v_closing = -(px·vx+py·vy)/d ; >0=逼近
                    _v_close = -(_px * _vx + _py * _vy) / _d          # [E,5]
                    _approaching = _valid & (_v_close > 0.05)         # 有效且逼近中
                    _ttc = _d / _v_close.clamp(min=0.05)              # [E,5] 碰撞剩餘時間 s
                    # gate = (1 - TTC/thresh) clamp[0,1]; 只在逼近且 TTC<thresh
                    _slot_gate = ((1.0 - _ttc / _ttc_thresh).clamp(0.0, 1.0)) * _approaching.float()
                    _ttc_gate = _slot_gate.max(dim=1).values          # [E] 取最危障礙
                    # warmup ramp (防 from-scratch 早期凍結)
                    _tw_s = int(getattr(args_cli, "ttc_warmup_start", 0))
                    _tw_e = int(getattr(args_cli, "ttc_warmup_end", 0))
                    if _tw_e > _tw_s:
                        _ttc_gate = _ttc_gate * min(1.0, max(0.0, (iteration - _tw_s) / (_tw_e - _tw_s)))
                    _v_fwd = obs[:, 1].clamp(min=0.0)                 # 前進速度(逼近時越快罰越重)
                    _ttc_pen = -_ttc_w * _ttc_gate * _v_fwd          # [E] 負獎勵(扣分)
                reward_flat = reward_flat + _ttc_pen
                reward_breakdown["ttc_tax"] = _ttc_pen

            # --- 2d. ★Term 1: Predictive Early Deceleration + Receding Penalty ---
            #   對稱設計逼 policy 用 LV-DOT 速度判斷「該不該減速」:同距離下逼近 vs 遠離看 LiDAR 一樣,
            #   只有 channel vx,vy 能區分 → 堵住「LiDAR 一律減速」低智策略逃生口。
            #   (a) 逼近(v_closing>0.1)且預測 horizon 秒後很近 → 本步減速給 +reward;
            #   (b) 無逼近威脅但附近有遠離/靜止障礙,卻仍減速 → -penalty(過度保守)。
            _pd_w = float(getattr(args_cli, "predictive_decel_weight", 0.0))
            _rec_w = float(getattr(args_cli, "receding_penalty_weight", 0.0))
            if (_pd_w > 0.0 or _rec_w > 0.0) and obs.shape[-1] >= 109:
                with torch.no_grad():
                    _ph = float(getattr(args_cli, "predictive_pred_horizon", 2.0))
                    _pchn = obs[:, 79:109].view(obs.shape[0], 5, 6)        # [E,5,6]
                    _ppx = _pchn[:, :, 0] * 8.0
                    _ppy = _pchn[:, :, 1] * 8.0
                    _pvx = _pchn[:, :, 2] * 1.5
                    _pvy = _pchn[:, :, 3] * 1.5
                    _pvalid = _pchn[:, :, 5] > 0.5
                    _pdist = torch.sqrt(_ppx * _ppx + _ppy * _ppy + 1e-6)  # [E,5]
                    _pvclose = -(_ppx * _pvx + _ppy * _pvy) / _pdist       # [E,5] >0=逼近
                    _appr = _pvalid & (_pvclose > 0.1)                     # 逼近中
                    _recede = _pvalid & (_pvclose < 0.05) & (_pdist < 2.5)  # 遠離/靜止且在附近
                    # 本步減速量 (mirror _prev_omega pattern)
                    _cur_spd = obs[:, 1]
                    _decel = (_prev_speed - _cur_spd).clamp(min=0.0)       # [E] >0=減速量
                    _is_decel = (_decel > 0.02).float()                    # [E] 有明顯減速
                    # (a) 逼近預測風險 pred_d = d - v_closing·horizon (越小越危險)
                    _pred_d = _pdist - _pvclose.clamp(min=0.0) * _ph       # [E,5]
                    _appr_risk = ((1.5 - _pred_d).clamp(min=0.0)) * _appr.float()  # [E,5]
                    _appr_risk_max = _appr_risk.max(dim=1).values          # [E]
                    _has_appr = _appr.any(dim=1).float()                   # [E] 有逼近威脅
                    _pd_reward = _pd_w * _appr_risk_max * _is_decel        # [E] (a) 正
                    _rec_near = _recede.any(dim=1).float()                 # [E]
                    _rec_pen = -_rec_w * _decel * _rec_near * (1.0 - _has_appr)  # [E] (b) 負
                    _pd_term = _pd_reward + _rec_pen
                reward_flat = reward_flat + _pd_term
                reward_breakdown["predictive_decel"] = _pd_reward
                reward_breakdown["receding_penalty"] = _rec_pen

            # --- 2e. ★Term A: Speed-aware Turning Reward (07-11 Option A) ---
            #   ablation 證 policy 靠轉向避障非減速 → 獎勵打 steering 維度才對症。全部 max over K=5 障礙。
            #   A1 方向正確性:逼近時 omega 符號 = cross(px·vy−py·vx) 建議方向 → +w·strength(v_closing);
            #   A2 提早轉向:同方向且 TTC∈[lo,hi] → +w·time_factor(越早給越多);
            #   weakened decel:僅極危 TTC<1.2 且 v_closing>0.35 且減速才給(最後手段,免走走停停)。
            if (_td_w_cfg > 0.0 or _et_w_cfg > 0.0 or _wd_w_cfg > 0.0) and obs.shape[-1] >= 109:
                with torch.no_grad():
                    _ach = obs[:, 79:109].view(obs.shape[0], 5, 6)         # [E,5,6]
                    _apx = _ach[:, :, 0] * 8.0
                    _apy = _ach[:, :, 1] * 8.0
                    _avx = _ach[:, :, 2] * 1.5
                    _avy = _ach[:, :, 3] * 1.5
                    _aval = _ach[:, :, 5] > 0.5
                    _adist = torch.sqrt(_apx * _apx + _apy * _apy + 1e-6)   # [E,5]
                    _avclose = -(_apx * _avx + _apy * _avy) / _adist        # [E,5] >0=逼近
                    _aappr = _aval & (_avclose > 0.08)                      # [E,5]
                    # 建議轉向方向 = sign(cross=px·vy−py·vx);與當前 omega 符號比對
                    _across = _apx * _avy - _apy * _avx                     # [E,5]
                    # ★方向 = −sign(cross): body frame py=左+,omega>0=左轉。前障橫越 cross≈px·vy→sign(vy);
                    #   障礙往左移(vy>0)應右轉(繞到其後方)=omega<0 → desired=−sign(cross)。(Codex審查修正:原+sign反了)
                    _adesire = torch.where(_across > 0, -1.0, 1.0)          # [E,5]
                    _aomega_sign = torch.sign(obs[:, 2]).unsqueeze(1)       # [E,1] omega=obs[2]
                    _adir_ok = (_adesire == _aomega_sign) & _aappr         # [E,5] 方向對且逼近
                    _attc = _adist / _avclose.clamp(min=0.05)              # [E,5] TTC
                    # A1 方向正確性
                    _astrength = (_avclose / 1.5).clamp(max=1.0)          # [E,5]
                    _a1 = (_td_w_cfg * _astrength * _adir_ok.float()).max(dim=1).values          # [E]
                    # A2 提早轉向 (TTC∈[lo,hi])
                    _et_lo = float(getattr(args_cli, "early_turning_ttc_lo", 1.6))
                    _et_hi = float(getattr(args_cli, "early_turning_ttc_hi", 3.8))
                    _attc_ok = (_attc > _et_lo) & (_attc < _et_hi)
                    # ★越早(TTC大)給越多: (TTC−lo)/(hi−lo)。TTC=hi→1(最早最多),TTC=lo→0。(Codex審查修正:原(hi−TTC)反了)
                    _atf = ((_attc - _et_lo) / max(_et_hi - _et_lo, 1e-3)).clamp(0.0, 1.0)
                    _a2 = (_et_w_cfg * _atf * (_adir_ok & _attc_ok).float()).max(dim=1).values   # [E]
                    # weakened decel (最後手段)
                    _adecel = (_prev_speed - obs[:, 1]).clamp(min=0.0)
                    _ais_decel = (_adecel > 0.02).float()
                    _adanger = _aappr & (_attc < 1.2) & (_avclose > 0.35)
                    _awd_gate = ((1.2 - _attc).clamp(min=0.0) * _adanger.float()).max(dim=1).values
                    _awd = _wd_w_cfg * _awd_gate * _ais_decel              # [E]
                    _aterm = _a1 + _a2 + _awd
                reward_flat = reward_flat + _aterm
                reward_breakdown["turning_direction"] = _a1
                reward_breakdown["early_turning"] = _a2
                reward_breakdown["weakened_decel"] = _awd

            # 更新 prev_speed(每步,供下一步偵測減速;done 時歸零見下方),與 _prev_omega 同步機制
            if _track_prev_speed:
                _prev_speed = obs[:, 1].clone()

            # --- Gap-heading reward (轉彎閃避 head-on:獎勵 heading 朝最大可通行間隙) ---
            # 障礙近(d_safe<2m)時,找 LiDAR 最大連續可通行弧段(>0.9m)中心角,獎勵 cos(gap_angle):
            #   gap 在前=+1 → policy 轉頭對準 gap = 往側邊空隙閃。obs[6:78] bin36=正前方(已驗 atan2 body frame)。
            #   reactive、用當前位置(RNN 有編碼)、不靠速度預測 → 對症「直直來的 head-on 要轉彎避開」。
            if getattr(args_cli, "gap_heading_weight", 0.0) > 0:
                with torch.no_grad():
                    _lnorm = obs[:, 6:78]                                       # [E,72] norm[0,1]
                    _lm = torch.where(_lnorm < 0.02, torch.ones_like(_lnorm), _lnorm) * _LIDAR_MAX_DISTANCE_M  # hole→max(m)
                    _gactive = (_lm.min(dim=1).values < 2.0)                   # d_safe<2m 才啟用
                    # ★07-05 修正(用戶抓漏): gap 搜尋限前方半圓 ±90°(bins 18..54, bin36=正前)。
                    #   全圓搜尋時,單一前方障礙的「最大弧段」= 繞車尾一大圈 → 中心=正後方
                    #   → cos 梯度指向掉頭,病態。FGM 慣例=只在前進扇區找 gap;
                    #   前方無 gap → 本項 0(減速罰接手)。窗內 cos∈[0,1] 恆非負。
                    _fwd = _lm[:, 18:55]                                       # [E,37] −90°..+90°
                    _passable = _fwd > 0.9                                     # 深度門檻(m)
                    _cs = _passable.float().cumsum(dim=1)
                    _rb = (_cs * (~_passable).float()).cummax(dim=1).values
                    _runl = _cs - _rb                                          # 連續可通行長度
                    _mrl, _mre = _runl.max(dim=1)
                    _gci = _mre.float() - _mrl.float() / 2 + 18.0              # gap 中心 bin(全域座標)
                    _bin_ang = (_gci * 5.0 - 180.0) * (math.pi / 180.0)        # bin36→0=正前方
                    _gap_r = torch.cos(_bin_ang) * _gactive.float() * (_mrl > 0).float()
                    _gap_r = _gap_r * (obs[:, 1].abs() > 0.05).float()         # 速度門檻:靜止不給
                _gap_term = (args_cli.gap_heading_weight * 0.2) * _gap_r       # ×dt(=1/rl_fps5) per-step dense
                reward_flat = reward_flat + _gap_term
                reward_breakdown["gap_heading"] = _gap_term

            # --- 3. Obstacle forward + apply（若無動態障礙物則跳過）---
            if _obs_agent_active:
                with torch.no_grad():
                    obs_obs = build_obstacle_obs(env.unwrapped, N_obs, device)  # [E, N, 9]
                    obs_flat = obs_obs.reshape(-1, OBS_POLICY_OBS_DIM)           # [E*N, 9]
                    obs_act, obs_lp, obs_ent = obs_policy.sample(obs_flat)       # [E*N, 2]
                    obs_val = obs_value(obs_flat).squeeze(-1)                    # [E*N]

                apply_obstacle_actions(env.unwrapped, obs_act.reshape(num_envs, N_obs, 2),
                                       N_obs, dt=0.2, speed_limit=_obs_speed_limit)

                # Obstacle reward（WD: zero，approach: 弱對抗）
                obs_rew = compute_obstacle_reward(env.unwrapped, obs_obs, N_obs, args_cli.obs_reward_mode)

                # Obstacle done = charge done（同一 env 的所有障礙物同步 done）
                obs_done = done.unsqueeze(-1).expand(-1, N_obs).reshape(-1)
            else:
                # 無動態障礙物：填零 placeholder，不移動障礙物
                obs_obs = torch.zeros(num_envs, N_obs, OBS_POLICY_OBS_DIM, device=device)
                obs_flat = obs_obs.reshape(-1, OBS_POLICY_OBS_DIM)
                obs_act = torch.zeros(num_envs * N_obs, 2, dtype=torch.long, device=device)
                obs_lp = torch.zeros(num_envs * N_obs, device=device)
                obs_val = torch.zeros(num_envs * N_obs, device=device)
                obs_rew = torch.zeros(num_envs * N_obs, device=device)
                obs_done = done.unsqueeze(-1).expand(-1, N_obs).reshape(-1)

            # --- 4. Store transitions（儲存本步資料到 rollout buffer）---
            wd_aux_tgt = None
            if not _e2e_frame_stack:
                with torch.no_grad():
                    wd_aux_tgt = build_wd_preprocess_targets(
                        env.unwrapped, N_obs, device,
                        top_k_velocity=_aux_vel_topk,
                        pos_scale=args_cli.aux_target_pos_scale)  # [E, predict_dim]
            charge_buf.add(rl_in, actions, log_prob, reward_flat, value, done, obs, hidden,
                           aux_target=wd_aux_tgt, privileged=_priv_obs, terminated=_terminated_flat,
                           encoder_input=_encoder_input,
                           teacher_logits=_teacher_logits_step,
                           retention_mask=_retention_mask_step,
                           previous_teacher_logits=(
                               _previous_teacher_logits_step
                           ),
                           previous_retention_mask=(
                               _previous_retention_mask_step
                           ),
                           corridor_mask=(
                               _long_corridor_mask_step
                               if _corridor_adapter_enabled
                               else None
                           ),
                           narrow_teacher_actions=_narrow_teacher_actions_step,
                           narrow_imitation_mask=_narrow_imitation_mask_step)
            if obs_buf is not None:
                obs_buf.add(obs_flat, obs_act, obs_lp, obs_rew, obs_val, obs_done)

            # --- RGDR：累積 per-env episode reward ---
            if _rgdr_enabled:
                _rgdr_episode_reward += reward_flat.detach()

            # --- 5. Metrics：記錄本步指標 ---
            # 2026-07-03 fix: rule_based(_obs_agent_active=False)下 obs_obs/obs_act 是 zeros
            # placeholder → obstacle/mean_speed 等變「假數據恆0」；改傳 None = 缺席而非假值
            metrics.step(obs, reward_flat, done, info,
                         obs_obs=(obs_obs if _obs_agent_active else None),
                         obs_actions=(obs_act.reshape(num_envs, N_obs, 2) if _obs_agent_active else None),
                         reward_breakdown=reward_breakdown,
                         goal_diagnostics=goal_diagnostics,
                         charge_actions=charge_action_diagnostics,
                         rollout_step=step)

            # --- 6. Update states（更新 RNN hidden state + episode reset 處理）---
            rnn_state.update(new_hidden)    # 把新的 hidden state 存回 RNNStateManager
            done_mask = done.bool()
            if done_mask.any():
                done_ids = done_mask.nonzero(as_tuple=False).reshape(-1)
                rnn_state.reset(done_ids)  # episode 結束的 env 重置 hidden state 為 0
                if _arc_action_term is not None:  # ★r_arc: 記錄完成 episode 的累積 penalty 並歸零
                    _arc_ep_sum += float(_ep_arc_accum[done_ids].sum().item())
                    _arc_ep_n += int(done_ids.numel())
                    _ep_arc_accum[done_ids] = 0.0
                if _lidar_hist is not None:
                    _lidar_hist[done_ids] = 0.0   # 多幀:同步 reset LiDAR 歷史(新 episode 從零)
                if _teacher_lidar_hist is not None:
                    _teacher_lidar_hist[done_ids] = 0.0
                if _previous_teacher_lidar_hist is not None:
                    _previous_teacher_lidar_hist[done_ids] = 0.0
                # 重置障礙物速度 cache（避免舊 episode 的速度污染新 episode）
                if hasattr(env.unwrapped, "_obstacle_velocities"):
                    env.unwrapped._obstacle_velocities[done_ids] = 0.0
                # 重置 WD aux 歷史 position cache（t-1/t-2/t-3）
                if hasattr(env.unwrapped, "_wd_aux_obs_pos_t1"):
                    env.unwrapped._wd_aux_obs_pos_t1[done_ids] = 0.0
                    env.unwrapped._wd_aux_obs_pos_t2[done_ids] = 0.0
                    env.unwrapped._wd_aux_obs_pos_t3[done_ids] = 0.0
                # 每次 episode reset 重新隨機化障礙物大小和場景邊界（WD: obs_size_rand）
                _randomize_obstacle_sizes(done_ids, _obs_size_rand)
                _randomize_scene_bounds(done_ids, _scene_bound_rand)
                # Obs delay DR：重置 ring buffer + 重新抽 delay（per-episode DR）
                if _use_obs_delay:
                    _obs_ring[:, done_ids] = 0.0
                    _obs_ring_ptr[done_ids] = 0
                    _delay_per_env[done_ids] = torch.randint(
                        _delay_lo, _delay_hi + 1, (len(done_ids),), device=device)
                # Heading stability：重置 prev_omega（新 episode 無歷史）
                _prev_omega[done_ids] = 0.0
                # ★Term 1：重置 prev_speed（新 episode 無速度歷史，避免跨場景假減速）
                _prev_speed[done_ids] = 0.0
                if _reward_module.name == "clean_progress" and _prev_actions is not None:
                    # A reset starts from zero command; do not couple the next
                    # episode's smoothness cost to the previous terminal action.
                    _prev_actions[done_ids] = 9
                # RGDR：更新 per-env EMA episode return + 重置累積器
                if _rgdr_enabled:
                    _rgdr_env_returns[done_ids] = (
                        (1 - _rgdr_alpha) * _rgdr_env_returns[done_ids]
                        + _rgdr_alpha * _rgdr_episode_reward[done_ids])
                    _rgdr_episode_reward[done_ids] = 0.0
            obs = next_obs  # 更新當前觀測

        if args_cli.scene_probe_output and not _scene_probe_saved:
            probe_path = Path(args_cli.scene_probe_output).expanduser().resolve()
            probe_path.parent.mkdir(parents=True, exist_ok=True)
            probe_labels = torch.cat(_scene_probe_labels, dim=0)
            torch.save(
                {
                    "inputs": torch.cat(_scene_probe_inputs, dim=0),
                    "labels": probe_labels,
                    "env_ids": torch.cat(_scene_probe_env_ids, dim=0),
                    "rollout_steps": torch.cat(_scene_probe_steps, dim=0),
                    "metadata": {
                        "label_names": {
                            0: "native",
                            1: "sa5_general",
                            2: "narrow",
                            3: "corridor",
                        },
                        "policy_obs_dim": int(policy_obs_dim),
                        "encoder_input_dim": int(_encoder_input_dim),
                        "lidar_frame_stack": int(_K_stack),
                        "stride": int(args_cli.scene_probe_stride),
                        "num_envs": int(num_envs),
                    },
                },
                probe_path,
            )
            counts = torch.bincount(probe_labels.long(), minlength=4)
            print(
                "[SCENE-PROBE] saved "
                f"path={probe_path} samples={probe_labels.numel()} "
                f"counts={counts.tolist()}",
                flush=True,
            )
            _scene_probe_saved = True

        # === Charge PPO Update（play 模式跳過所有訓練）===
        charge_ppo_loss = 0.0
        charge_vf_loss = 0.0
        charge_entropy = 0.0
        aux_loss_val = 0.0
        _narrow_imitation_stats = None
        if args_cli.play:
            # --play: 推論模式，只跑 rollout，不做任何優化
            train_charge = False
            train_obstacle = False
        aux_per_feature = {}
        wd_update_monitor = {}

        if train_charge:
            _aux_already_done = False  # flag: True 表示 WD-order 已在 RL 之前完成 aux 更新

            # =============================================================
            # WD-order: Aux FIRST → Fresh forward → RL（A2C 模式自動啟用）
            # =============================================================
            # WD 原版流程 (custom_trainer.py):
            #   1. Rollout（收集 obs/actions/rewards）
            #   2. Aux update（用 module loss 更新 RNN 權重）
            #   3. Fresh RL forward（用更新後的 RNN 重新 forward obs → 取得新 features）
            #   4. RL update（用新 features 做 A2C 更新）
            # 這確保 RL head 永遠看到「最新 RNN feature」，不是延遲一個 iteration 的舊 feature。
            # A2C 無 importance sampling ratio，不需要 π_old/features 對應，可安全使用 WD-order。
            # PPO 模式仍走 legacy order（RL → Aux），因為需要舊 log_probs 和舊 features 對應。
            if args_cli.use_a2c and not args_cli.disable_aux_training:
                # --- Run aux update (same logic as the section below) ---
                if use_extractor: extractor.train()
                preprocess_rnn.train()

                _mon_modules_pre = {
                    "rnn": list(preprocess_rnn.rnn.parameters()),
                    "fc_front": list(preprocess_rnn.fc_front.parameters()),
                    "fc_middle": list(preprocess_rnn.fc_middle.parameters()),
                    "predict_head": list(preprocess_rnn.predict_head.parameters()),
                }
                if use_extractor:
                    _mon_modules_pre["extractor"] = list(extractor.parameters())

                # Snapshot for monitoring
                _snaps_pre = {k: _snapshot_params(ps) for k, ps in _mon_modules_pre.items()}

                _seq_len = args_cli.aux_seq_len
                _burn_in = args_cli.aux_burn_in
                _seq_bs = args_cli.aux_seq_batch_size
                # --aux_epochs:每 iter 多做 (N-1) 次 regression aux 更新(複製離線多 epoch 梯度密度)。
                # 離線 clean supervised 證 fresh GRU hidden 可達 30.9%,但需 ~1000 次更新;RL 1 update/iter
                # 太少。這裡每 iter 重採樣多更新幾次,快速逼近離線收斂。(僅 regression 路徑,非 CPC)
                if not args_cli.aux_cpc and args_cli.aux_epochs > 1:
                    for _xep in range(args_cli.aux_epochs - 1):
                        _smp = charge_buf.sample_aux_sequences(_seq_len, _seq_bs, _burn_in)
                        if _smp is None:
                            break
                        _os, _ts, _h0x, _ = _smp
                        if args_cli.aux_zero_h0:
                            _h0x = torch.zeros_like(_h0x)
                        _Bx, _Lx = _os.shape[0], _os.shape[1]
                        _of = obs_normalizer.normalize(_os.reshape(_Bx * _Lx, -1))
                        # 2026-07-03 fix(C3): 此迴圈原本沒建多幀歷史 → frame_stack>1 時
                        # fallback 複製當前幀 → aux_epochs 的前 N-1 次更新全用假歷史(silent)。
                        # 歷史 mf4 run2(aux_epochs 8 + frame_stack 4)因此 7/8 次更新被汙染。
                        _lhx = None
                        if args_cli.lidar_frame_stack > 1:
                            _on_x = _of.reshape(_Bx, _Lx, -1)
                            _ld_x = _on_x[..., LIDAR_START:LIDAR_END]
                            _hfx = []
                            for _jx in range(1, args_cli.lidar_frame_stack):
                                _shx = torch.zeros_like(_ld_x)
                                _shx[:, _jx:] = _ld_x[:, :-_jx]
                                _hfx.append(_shx)
                            _lhx = torch.cat(_hfx, dim=-1).reshape(_Bx * _Lx, -1)
                        _ff = _charge_features_for_rnn(_of, lidar_hist=_lhx)
                        _fs = _ff.reshape(_Bx, _Lx, -1).permute(1, 0, 2).contiguous()  # [L,B,*] 已修 reshape
                        _, _ps, _ = preprocess_rnn(_fs, _h0x, training=True)
                        _te = _ts.permute(1, 0, 2)[_burn_in:]
                        _pe = _ps[_burn_in:]
                        _tl = torch.tensor(0.0, device=device)
                        for _ti in range(_pe.shape[0]):
                            _l, _ = compute_wd_module_loss(
                                _pe[_ti], _te[_ti], weight=_aux_weight,
                                loss_type=args_cli.aux_loss_type, huber_delta=args_cli.aux_huber_delta)
                            _tl = _tl + _l
                        _tl = _tl / max(_pe.shape[0], 1)
                        charge_opt_aux.zero_grad()
                        _tl.backward()
                        nn.utils.clip_grad_norm_(charge_params_aux, _current_aux_grad_clip)
                        charge_opt_aux.step()
                sampled = charge_buf.sample_aux_sequences(_seq_len, _seq_bs, _burn_in)
                _wd_aux_valid_count = 0
                _wd_aux_batch = 0
                _wd_aux_grad_norms = {k: 0.0 for k in _mon_modules_pre}
                _wd_aux_delta_norms = {k: 0.0 for k in _mon_modules_pre}
                _wd_aux_pos = {"rel": float("nan"), "r2": float("nan"), "vf": float("nan")}
                if sampled is not None:
                    obs_seq, target_seq, h0, _wd_aux_valid_count = sampled
                    if args_cli.aux_zero_h0:
                        h0 = torch.zeros_like(h0)  # 對標離線 fresh hidden,逼 GRU 從特徵萃取
                    B_seq, L_seq = obs_seq.shape[0], obs_seq.shape[1]
                    _wd_aux_batch = B_seq
                    obs_flat = obs_seq.reshape(B_seq * L_seq, -1)
                    obs_normed_aux = obs_normalizer.normalize(obs_flat)
                    _lhist_aux = None
                    if args_cli.lidar_frame_stack > 1:
                        # 多幀:chunk 內 shift 建歷史(序列不跨 episode 由 sampling 保證);
                        #   前 (K-1) 幀 zero-pad,靠 aux_burn_in>=K-1 把它們排除出 loss(只當 hidden 暖機)。
                        _on_s = obs_normed_aux.reshape(B_seq, L_seq, -1)
                        _ld_s = _on_s[..., LIDAR_START:LIDAR_END]            # [B,L,72]
                        _hf = []
                        for _j in range(1, args_cli.lidar_frame_stack):
                            _shf = torch.zeros_like(_ld_s)
                            _shf[:, _j:] = _ld_s[:, :-_j]                    # j 步前的幀,前 j 步 zero
                            _hf.append(_shf)
                        _lhist_aux = torch.cat(_hf, dim=-1).reshape(B_seq * L_seq, -1)
                    feat_flat = _charge_features_for_rnn(obs_normed_aux, lidar_hist=_lhist_aux)
                    # ★BUG FIX:obs_seq 是 [B,L,obs](batch-major),reshape(L,B) 會把時間/batch 維度打亂
                    #   → 特徵與 permute 過的 target 不對應 → RNN 學不出映射 → constant collapse 真根因。
                    #   正解:先 reshape 回 [B,L] 再 permute → [L,B](與 target_seq.permute 一致)。
                    feat_seq = feat_flat.reshape(B_seq, L_seq, -1).permute(1, 0, 2).contiguous()
                    _, pred_seq, _ = preprocess_rnn(feat_seq, h0, training=True)
                    effective_start = _burn_in
                    effective_len = L_seq - effective_start
                    pred_eff = pred_seq[effective_start:]
                    tgt_eff = target_seq.permute(1, 0, 2)[effective_start:]
                    # 即時 tracking 指標：位置 rel err + R²（避開 aux log-loss 常數陷阱）
                    with torch.no_grad():
                        _pe = pred_eff.reshape(-1, pred_eff.shape[-1])
                        _te = tgt_eff.reshape(-1, tgt_eff.shape[-1])
                        _nd = min(6, _pe.shape[-1])
                        _pv = _te[:, 0:2].norm(dim=-1) < 9.0
                        _m = _pv if _pv.any() else torch.ones(_te.shape[0], dtype=torch.bool, device=_te.device)
                        _p = _pe[_m][:, 0:_nd]; _t = _te[_m][:, 0:_nd]
                        _wd_aux_pos["rel"] = ((_p - _t).norm(dim=-1).mean()
                                              / _t.norm(dim=-1).mean().clamp(min=1e-6)).item()
                        _ssr = ((_p - _t) ** 2).sum().item()
                        _sst = ((_t - _t.mean(dim=0, keepdim=True)) ** 2).sum().clamp(min=1e-6).item()
                        _wd_aux_pos["r2"] = 1.0 - _ssr / _sst
                        _wd_aux_pos["vf"] = _pv.float().mean().item()
                        # ★aux/disp_r2: 預測位移(next−t0) vs 真位移 R² — 直接量「移動編碼」。
                        #   塌成靜態(pred_next≈pred_t0)→pred_disp≈0→disp_r2≈0/負;學到移動→>0。
                        #   pos_r2 高分不出移動vs塌陷(next≈now 時絕對位置照樣對),故另記(監控協定第0層命脈)。
                        if _nd >= 4:
                            _pd = _p[:, 2:4] - _p[:, 0:2]
                            _td = _t[:, 2:4] - _t[:, 0:2]
                            _dssr = ((_td - _pd) ** 2).sum().item()
                            _dsst = ((_td - _td.mean(dim=0, keepdim=True)) ** 2).sum().clamp(min=1e-3).item()
                            _wd_aux_pos["disp_r2"] = max(-1.0, 1.0 - _dssr / _dsst)  # floor:未訓頭+小變異會爆,只關心爬向>0
                        # ★aux/velocity_r2: 預測速度(dims 7:13) vs 真速度 R² — predict_dim>=13(--aux_velocity_topk) 才有。
                        if _pe.shape[-1] >= 13:
                            _pvel = _pe[_m][:, 7:13]; _tvel = _te[_m][:, 7:13]
                            _vssr = ((_pvel - _tvel) ** 2).sum().item()
                            _vsst = ((_tvel - _tvel.mean(dim=0, keepdim=True)) ** 2).sum().clamp(min=1e-3).item()
                            _wd_aux_pos["velocity_r2"] = max(-1.0, 1.0 - _vssr / _vsst)  # floor:同 disp_r2

                    # ===== CPC contrastive aux(--aux_cpc):逼 RNN hidden discriminative 編碼位置 =====
                    # 獨立 fresh forward(不重用 regression graph)。q=proj_q(rnn_out), k=proj_k(pos_target)。
                    # InfoNCE:常數 hidden 對所有樣本 sim 相同→無法對上對角正樣本→高 loss→逼 RNN 編碼。
                    if args_cli.aux_cpc and charge_opt_cpc is not None:
                        # 2026-07-03 fix(C4): 補傳 _lhist_aux(原漏傳→frame_stack>1 時假歷史)
                        _feat_cpc = _charge_features_for_rnn(obs_normed_aux, lidar_hist=_lhist_aux)  # fresh extractor
                        _Lc, _Bc = L_seq, B_seq
                        # ★同 BUG FIX:_feat_cpc 是 [B*L] batch-major,要 reshape(B,L).permute→[L,B]
                        _fc_c = preprocess_rnn.fc_front(_feat_cpc).reshape(_Bc, _Lc, -1).permute(1, 0, 2).contiguous()
                        _rnn_out_c, _ = preprocess_rnn.rnn(_fc_c, h0)                   # [L,B,H] 每步 hidden
                        _ro = _rnn_out_c[effective_start:].reshape(-1, args_cli.hidden_dim)
                        _tg2 = target_seq.permute(1, 0, 2)[effective_start:].reshape(
                            -1, target_seq.shape[-1])[:, 0:2]                          # [N,2] pos(已含 scale)
                        _far_t = 9.0 * args_cli.aux_target_pos_scale
                        _vm = _tg2.norm(dim=-1) < _far_t
                        _ro = _ro[_vm]; _tg2 = _tg2[_vm]
                        _Ncpc = _ro.shape[0]
                        if _Ncpc > args_cli.aux_cpc_max_samples:
                            _perm = torch.randperm(_Ncpc, device=device)[:args_cli.aux_cpc_max_samples]
                            _ro = _ro[_perm]; _tg2 = _tg2[_perm]
                        if _ro.shape[0] >= 8:
                            _q = F.normalize(cpc_proj_q(_ro), dim=-1)
                            _k = F.normalize(cpc_proj_k(_tg2), dim=-1)
                            _logits = (_q @ _k.t()) / args_cli.aux_cpc_temp
                            _labels = torch.arange(_ro.shape[0], device=device)
                            _cpc_loss = F.cross_entropy(_logits, _labels)
                            with torch.no_grad():
                                _cpc_acc = (_logits.argmax(dim=-1) == _labels).float().mean().item()
                            charge_opt_cpc.zero_grad()
                            _cpc_loss.backward()
                            nn.utils.clip_grad_norm_(_cpc_params, _current_aux_grad_clip)
                            charge_opt_cpc.step()
                            _wd_aux_pos["cpc_loss"] = _cpc_loss.item()
                            _wd_aux_pos["cpc_acc"] = _cpc_acc
                            aux_loss_val = _cpc_loss.item()

                    total_loss = torch.tensor(0.0, device=device)
                    last_display_pre = {}
                    if not args_cli.aux_cpc:
                        for t_idx in range(effective_len):
                            l_t, last_display_pre = compute_wd_module_loss(pred_eff[t_idx], tgt_eff[t_idx], weight=_aux_weight, loss_type=args_cli.aux_loss_type, huber_delta=args_cli.aux_huber_delta)
                            total_loss = total_loss + l_t
                        total_loss = total_loss / max(effective_len, 1)
                        charge_opt_aux.zero_grad()
                        # WD: reset_model_mentum 只在訓練啟動時一次性 reset（預設 False）
                        # 之前誤解為每步 reset → Adam 退化為無 momentum SGD → RNN 無法收斂
                        # 修正：移除 unconditional momentum reset，保留 Adam 正常累積
                        total_loss.backward()
                        nn.utils.clip_grad_norm_(
                            charge_params_aux,
                            _current_aux_grad_clip,
                        )
                        # Capture grad norms before step
                        for k, ps in _mon_modules_pre.items():
                            _wd_aux_grad_norms[k] = _grad_l2_norm(ps)
                        charge_opt_aux.step()
                    # Capture param delta norms after step
                    for k, ps in _mon_modules_pre.items():
                        _wd_aux_delta_norms[k] = _param_delta_norm(_snaps_pre[k], ps)
                    aux_loss_val = total_loss.item()
                    # Carry per-feature display for monitoring
                    for k, v in last_display_pre.items():
                        if k != "preprcess_loss":
                            aux_per_feature[f"charge_{k}"] = v
                _aux_already_done = True

                # --- Fresh forward: recompute rl_inputs with UPDATED RNN (WD step 3) ---
                if use_extractor: extractor.eval()
                preprocess_rnn.eval()
                with torch.no_grad():
                    _fresh_h = charge_buf.hiddens[0].unsqueeze(0)  # [1, E, H] initial hidden
                    _rc_hist = (torch.zeros(num_envs, (_K_stack - 1) * _LL, device=device)
                                if _K_stack > 1 else None)  # 多幀:sequential 重放重建 rollout 真歷史
                    for t in range(RL):
                        _obs_t = charge_buf.raw_obs[t]
                        _obs_n_t = obs_normalizer.normalize(_obs_t)
                        _feat_t = _charge_features_for_rnn(_obs_n_t, lidar_hist=_rc_hist)
                        if _rc_hist is not None:
                            _cur_l = _select_79d(_obs_n_t)[:, LIDAR_START:LIDAR_END]
                            _rc_hist = torch.cat([_cur_l, _rc_hist[:, :-_LL]], dim=-1)
                        _rnn_feat_t, _pred_t, _fresh_h = preprocess_rnn(
                            _feat_t, _fresh_h, training=args_cli.hybrid_predict_to_policy)
                        _p_obs_t = _charge_obs_for_rl(_obs_n_t)
                        if args_cli.zero_preprocess_feature_for_rl:
                            _rnn_feat_t = torch.zeros_like(_rnn_feat_t)
                        charge_buf.rl_inputs[t] = (
                            torch.cat([_p_obs_t, _rnn_feat_t, _pred_t], dim=-1)  # hybrid: + 障礙動態預測
                            if args_cli.hybrid_predict_to_policy
                            else torch.cat([_p_obs_t, _rnn_feat_t], dim=-1))
                        _priv_t = charge_buf.privileged_obs[t] if _use_asymmetric_critic else None
                        charge_buf.values[t] = value_head(_encode_rl_input(charge_buf.rl_inputs[t]), _priv_t).squeeze(-1)
                        # Handle episode resets: zero hidden for envs that were done at step t
                        _done_mask = charge_buf.dones[t].unsqueeze(0).unsqueeze(-1)  # [1,E,1]
                        _fresh_h = _fresh_h * (1.0 - _done_mask)
                        if _rc_hist is not None:
                            _rc_hist = _rc_hist * (1.0 - charge_buf.dones[t].unsqueeze(-1))  # 多幀:同步 reset

            # Bootstrap：用 rollout 最後一個 obs 的 value 做 GAE 的 V_{T+1}
            # （WD-order 模式時，使用 fresh features；legacy 模式使用原始 features）
            with torch.no_grad():
                obs_normed = obs_normalizer.normalize(obs)
                features = _charge_features_for_rnn(obs_normed, lidar_hist=_lidar_hist)  # 多幀:用 rollout 末尾歷史
                hidden = rnn_state.get()
                p_obs = _charge_obs_for_rl(obs_normed)
                if _e2e_frame_stack:
                    rl_in = torch.cat([p_obs, features], dim=-1)
                else:
                    rnn_feat, _pred_bt, _ = preprocess_rnn(
                        features, hidden, training=args_cli.hybrid_predict_to_policy)
                    _rnn_for_rl = (torch.zeros_like(rnn_feat)
                                   if args_cli.zero_preprocess_feature_for_rl else rnn_feat)
                    rl_in = (torch.cat([p_obs, _rnn_for_rl, _pred_bt], dim=-1)
                             if args_cli.hybrid_predict_to_policy
                             else torch.cat([p_obs, _rnn_for_rl], dim=-1))
                _priv_bootstrap = extract_privileged_obs(env.unwrapped) if _use_asymmetric_critic else None
                last_value = value_head(_encode_rl_input(rl_in), _priv_bootstrap).squeeze(-1)  # [E] bootstrap value V_{T+1}

            # GAE 計算（使用 WD 固定 gamma=0.984 + gae_lambda）
            if args_cli.popart:
                # ★PopArt: critic 輸出在正規化空間(std≈1)。反正規化(×std+mean)成 raw-reward 尺度再餵 GAE,
                #   讓 delta=r+γV−V 的 V 與 raw reward 同尺度=正確 baseline(修尺度不一致 bug)。
                _pa_std = math.sqrt(max(_popart_m2 - _popart_m1 ** 2, 1e-4))
                _pa_mean = _popart_m1
                _values_denorm = charge_buf.values * _pa_std + _pa_mean
                _last_value_denorm = last_value * _pa_std + _pa_mean
                advantages, returns = compute_gae(
                    charge_buf.rewards, _values_denorm, charge_buf.dones,
                    _last_value_denorm, current_gamma, args_cli.gae_lambda,
                    terminateds=charge_buf.terminateds)  # ★fix#1
                # 更新 running EMA 統計(前二階矩) from raw returns
                _b = args_cli.popart_beta
                _rm1 = returns.mean().item()
                _rm2 = (returns ** 2).mean().item()
                if not _popart_initialized:
                    # ★fix#5a: 首rollout直接init(跳過從0/1慢warmup;denorm 從iter2就對raw尺度)
                    _popart_m1, _popart_m2 = _rm1, _rm2
                    _popart_initialized = True
                else:
                    _popart_m1 = _b * _popart_m1 + (1.0 - _b) * _rm1
                    _popart_m2 = _b * _popart_m2 + (1.0 - _b) * _rm2
                # critic target 用更新後 running 統計正規化(critic 續在正規化空間學=訓練穩定)
                _pa_std_new = math.sqrt(max(_popart_m2 - _popart_m1 ** 2, 1e-4))
                value_targets = (returns - _popart_m1) / (_pa_std_new + 1e-8)
                if iteration % args_cli.log_interval == 0:
                    print(f"[POPART] iter={iteration} ret_mean={_popart_m1:.2f} ret_std={_pa_std_new:.2f} "
                          f"(critic 反正規化尺度已對齊 raw reward)")
            else:
                advantages, returns = compute_gae(
                    charge_buf.rewards, charge_buf.values, charge_buf.dones,
                    last_value, current_gamma, args_cli.gae_lambda,
                    terminateds=charge_buf.terminateds)  # ★fix#1
                if args_cli.normalize_return:
                    # --normalize_return：只對 critic target 做歸一化（actor advantages 路徑不變）
                    # ⚠️此路有 critic/actor 尺度不一致 bug(見 --popart);保留供對照。
                    value_targets = (returns - returns.mean()) / (returns.std() + 1e-8)
                else:
                    value_targets = returns  # 不歸一化：使用 raw GAE returns 作為 critic target
            _raw_adv_std = advantages.std().item()  # 歸一化前的 advantage std（用於診斷）
            _adv_mean = advantages.mean()
            if args_cli.adv_norm_mode == "full":
                # 標準 PPO normalization: std=1, gradient 被壓縮
                advantages = (advantages - _adv_mean) / (advantages.std() + 1e-8)
            elif args_cli.adv_norm_mode == "partial":
                # 折中: 除 sqrt(std)，保留部分量級，目標 ME≈0.5
                advantages = (advantages - _adv_mean) / (math.sqrt(_raw_adv_std) + 1e-8)
            else:
                # mean_only (SA4 default): 保留 raw advantage 量級
                advantages = advantages - _adv_mean

            # --- RGDR：per-env advantage weighting（聚焦失敗環境）---
            if _rgdr_enabled and iteration >= _rgdr_warmup:
                _rgdr_floor = _rgdr_env_returns.min()
                _difficulty = 1.0 / (_rgdr_env_returns - _rgdr_floor + 1e-6)
                _env_weight = (_difficulty / _difficulty.mean()).clamp(
                    _rgdr_clamp[0], _rgdr_clamp[1])  # [E]
                advantages = advantages * _env_weight.unsqueeze(0)  # [T, E]
                if iteration % args_cli.log_interval == 0:
                    print(f"[RGDR] env_weight: min={_env_weight.min():.2f} "
                          f"max={_env_weight.max():.2f} "
                          f"std={_env_weight.std():.3f}")

            if _corridor_adapter_enabled and _corridor_adapter_freeze_base:
                policy_head.eval()
            else:
                policy_head.train()
            value_head.train()
            if corridor_adapter is not None:
                corridor_adapter.train()
            if _e2e_frame_stack and not (
                _corridor_adapter_enabled and _corridor_adapter_freeze_base
            ):
                extractor.train()
            elif _e2e_frame_stack:
                extractor.eval()
            # extractor/preprocess_rnn 維持 eval()：RL 只訓練 RL heads，不更新 aux module
            # WD 等價做法：concat_input = rl_in_.detach()（custom_trainer.py line 573）
            # IsaacLab 版本：rl_in 在 torch.no_grad() 下計算，效果相同（無梯度流到 RNN/extractor）

            flat_ri = charge_buf.rl_inputs.reshape(-1, rl_input_dim)
            flat_act = charge_buf.actions.reshape(-1, 2)
            flat_lp = charge_buf.log_probs.reshape(-1)
            flat_adv = advantages.reshape(-1)
            flat_ret_raw = returns.reshape(-1)
            flat_value_target = value_targets.reshape(-1)
            flat_old_value = charge_buf.values[:RL].reshape(-1)
            flat_priv = charge_buf.privileged_obs[:RL].reshape(-1, _priv_dim) if _use_asymmetric_critic else None
            flat_encoder_input = (
                charge_buf.encoder_inputs[:RL].reshape(-1, _encoder_input_dim)
                if _e2e_frame_stack else None
            )
            flat_teacher_logits = (
                charge_buf.teacher_logits[:RL].reshape(-1, 2 * NUM_BINS)
                if _teacher_retention_enabled else None
            )
            flat_retention_mask = (
                charge_buf.retention_mask[:RL].reshape(-1)
                if _teacher_retention_enabled else None
            )
            flat_previous_teacher_logits = (
                charge_buf.previous_teacher_logits[:RL].reshape(
                    -1, 2 * NUM_BINS
                )
                if _previous_stage_teacher_enabled else None
            )
            flat_previous_retention_mask = (
                charge_buf.previous_retention_mask[:RL].reshape(-1)
                if _previous_stage_teacher_enabled else None
            )
            flat_narrow_teacher_actions = (
                charge_buf.narrow_teacher_actions[:RL].reshape(-1, 2)
                if _narrow_imitation_enabled else None
            )
            flat_narrow_imitation_mask = (
                charge_buf.narrow_imitation_mask[:RL].reshape(-1)
                if _narrow_imitation_enabled else None
            )
            flat_corridor_mask = (
                charge_buf.corridor_mask[:RL].reshape(-1)
                if _corridor_adapter_enabled else None
            )

            # --rnn_rl_grad: 準備在 minibatch 內重算 preprocess_feat(過 RNN,帶梯度)所需的
            # raw_obs 與 input hidden(rollout 當步存的)。讓 RL 梯度可流進 RNN。
            if args_cli.rnn_rl_grad:
                _obs_dim_rr = charge_buf.raw_obs.shape[-1]
                flat_raw_rr = charge_buf.raw_obs[:RL].reshape(-1, _obs_dim_rr)         # [T*E, obs_dim]
                flat_hid_rr = charge_buf.hiddens[:RL].reshape(-1, args_cli.hidden_dim)  # [T*E, H]
                preprocess_rnn.train()  # 開啟(對無 dropout 的 cell 無副作用;確保梯度路徑啟用)

            # --- Patch 2: returns/advantage statistics for WandB ---
            _ret_mean = flat_ret_raw.mean().item()
            _ret_std = flat_ret_raw.std().item()
            _ret_min = flat_ret_raw.min().item()
            _ret_max = flat_ret_raw.max().item()
            _target_mean = flat_value_target.mean().item()
            _target_std = flat_value_target.std().item()
            _target_min = flat_value_target.min().item()
            _target_max = flat_value_target.max().item()
            flat_value_pred = charge_buf.values[:RL].reshape(-1)
            value_error = flat_value_target - flat_value_pred
            _value_pred_mean = flat_value_pred.mean().item()
            _value_pred_std = flat_value_pred.std().item()
            _value_pred_min = flat_value_pred.min().item()
            _value_pred_max = flat_value_pred.max().item()
            _value_error_mean = value_error.mean().item()
            _value_error_std = value_error.std().item()
            _value_error_abs_max = value_error.abs().max().item()
            _adv_mean = flat_adv.mean().item()
            _adv_std = flat_adv.std().item()
            _adv_min = flat_adv.min().item()
            _adv_max = flat_adv.max().item()

            # A2C: 單 epoch 全 batch 更新（WD: A2CK mode，不 mini-batch）
            # PPO: 多 epoch + mini-batch（標準 PPO）
            n_epochs = 1 if args_cli.use_a2c else args_cli.ppo_epochs

            ppo_l, vf_l, ent_l = [], [], []
            vf_raw_l, pl_clamped_l, vf_clamped_l = [], [], []  # Patch 1: raw vs clamped
            pl_clamp_triggered_l, vf_clamp_triggered_l = [], []  # Patch 1: clamp flags
            total_loss_l = []  # Patch 5: total RL loss
            approx_kl_l, ent_lin_l, ent_ang_l = [], [], []  # Patch 3: kl + per-head entropy
            ppo_clip_frac_l, ppo_ratio_l = [], []  # Patch 3: PPO-only metrics
            wd_actor_grad_l, wd_critic_grad_l = [], []
            encoder_grad_l = []
            wd_actor_update_l, wd_critic_update_l = [], []
            wd_actor_delta_l, wd_critic_delta_l = [], []
            wd_actor_clip_l, wd_critic_clip_l = [], []
            wd_module_entropy_l = []
            # Patch 4: before/after clip
            wd_actor_grad_post_l, wd_critic_grad_post_l = [], []
            wd_merged_grad_pre_l, wd_merged_grad_post_l = [], []
            # Patch 5: update ratio
            wd_actor_update_ratio_l, wd_critic_update_ratio_l = [], []
            wd_actor_param_norm_l, wd_critic_param_norm_l = [], []
            # Post-update trust-region diagnostics
            post_kl_l, post_ratio_mean_l, post_ratio_std_l = [], [], []
            post_ratio_min_l, post_ratio_max_l = [], []
            post_ratio_outside_20_l, post_ratio_outside_50_l = [], []
            retention_kl_linear_l, retention_kl_angular_l = [], []
            retention_margin_linear_l, retention_margin_angular_l = [], []
            retention_action_ce_linear_l, retention_action_ce_angular_l = [], []
            retention_agree_linear_l, retention_agree_angular_l = [], []
            retention_post_kl_linear_l, retention_post_kl_angular_l = [], []
            retention_post_margin_linear_l, retention_post_margin_angular_l = [], []
            retention_post_agree_linear_l, retention_post_agree_angular_l = [], []
            retention_loss_l, retention_active_counts = [], []
            previous_retention_kl_linear_l = []
            previous_retention_kl_angular_l = []
            previous_retention_agree_linear_l = []
            previous_retention_agree_angular_l = []
            previous_retention_loss_l = []
            previous_retention_active_counts = []
            adapter_gate_loss_l = []
            adapter_gate_corridor_prob_l = []
            adapter_gate_other_prob_l = []
            adapter_gate_recall_l = []
            adapter_gate_specificity_l = []
            adapter_residual_corridor_l2_l = []
            adapter_residual_other_l2_l = []
            _retention_projection_stats = None
            _retention_anchor_projection_stats = None
            _corridor_projection_stats = None
            _kl_early_stop = False
            _ppo_update_count = 0
            for _ in range(n_epochs):
                if args_cli.use_a2c:
                    # A2C: full batch, single pass (WD: A2CK mode)
                    batches = [(torch.arange(batch_size, device=device),)]
                else:
                    idx = torch.randperm(batch_size, device=device)
                    batches = [(idx[s:min(s + mini_batch_size, batch_size)],)
                               for s in range(0, batch_size, mini_batch_size)]

                for (mb,) in batches:
                    if _e2e_frame_stack:
                        _enc_mb = flat_encoder_input[mb]
                        _feat_mb = extractor(_enc_mb)
                        _pobs_mb = _enc_mb[:, :policy_obs_dim]
                        _ri_mb = torch.cat([_pobs_mb, _feat_mb], dim=-1)
                    # --rnn_rl_grad: 重算 preprocess_feat(過 RNN,帶梯度),取代 cached flat_ri[mb]，
                    # 讓 RL loss 的梯度經 policy/value head 流回 RNN。
                    elif args_cli.rnn_rl_grad:
                        with torch.no_grad():
                            _normed_mb = obs_normalizer.normalize(flat_raw_rr[mb])
                        _feat_mb = _charge_features_for_rnn(_normed_mb)            # extractor(凍結,grad 不 step)
                        _h_mb = flat_hid_rr[mb].unsqueeze(0)                       # [1, b, H] rollout 當步 input hidden
                        _rnnfeat_mb, _, _ = preprocess_rnn(_feat_mb, _h_mb)        # [b,12] 帶梯度
                        _pobs_mb = _charge_obs_for_rl(_normed_mb)                  # [b, policy_obs_dim]
                        _ri_mb = torch.cat([_pobs_mb, _rnnfeat_mb], dim=-1)        # [b, rl_input_dim]
                    else:
                        _ri_mb = flat_ri[mb]
                    _priv_mb = flat_priv[mb] if flat_priv is not None else None
                    _ri_mb_enc = _encode_rl_input(_ri_mb)   # ★LV-DOT encoder(帶梯度,RL loss 流回 encoder)
                    nl = policy_head(_ri_mb_enc, _priv_mb if _oracle_to_policy else None)
                    _adapter_gate_loss = torch.zeros((), device=device)
                    _adapter_residual_mb = None
                    _adapter_gate_logits_mb = None
                    _adapter_gate_probability_mb = None
                    if corridor_adapter is not None:
                        (
                            _adapter_residual_mb,
                            _adapter_gate_logits_mb,
                            _adapter_gate_probability_mb,
                        ) = corridor_adapter(
                            _pobs_mb,
                            (
                                _ri_mb_enc
                                if _corridor_adapter_residual_features
                                == "policy_features"
                                else None
                            ),
                        )
                        nl = nl + _adapter_residual_mb
                        _adapter_gate_loss = balanced_binary_gate_loss(
                            _adapter_gate_logits_mb,
                            flat_corridor_mask[mb],
                        )
                    nlp, ent_lin, ent_ang = evaluate_actions(nl, flat_act[mb])
                    _critic_input = (
                        _ri_mb_enc.detach()
                        if args_cli.critic_detach_encoder
                        else _ri_mb_enc
                    )
                    nv = value_head(_critic_input, _priv_mb).squeeze(-1)
                    _retention_result = None
                    if _teacher_retention_enabled:
                        _retention_result = masked_two_head_retention_loss(
                            flat_teacher_logits[mb],
                            nl,
                            flat_retention_mask[mb],
                            num_bins=NUM_BINS,
                            argmax_margin=_teacher_retention_argmax_margin,
                        )
                    _narrow_imitation_result = None
                    if _narrow_imitation_enabled:
                        _narrow_imitation_result = scripted_action_ce_loss(
                            nl,
                            flat_narrow_teacher_actions[mb],
                            flat_narrow_imitation_mask[mb],
                            num_bins=NUM_BINS,
                        )
                    _previous_retention_result = None
                    if _previous_stage_teacher_enabled:
                        _previous_retention_result = (
                            masked_two_head_retention_loss(
                                flat_previous_teacher_logits[mb],
                                nl,
                                flat_previous_retention_mask[mb],
                                num_bins=NUM_BINS,
                                argmax_margin=0.0,
                            )
                        )

                    _log_ratio = nlp - flat_lp[mb]
                    # Non-negative PPO approximate KL: E[(ratio-1)-log(ratio)].
                    with torch.no_grad():
                        _approx_kl = ((_log_ratio.exp() - 1.0) - _log_ratio).mean().item()
                    if (args_cli.target_kl > 0.0 and _ppo_update_count > 0
                            and _approx_kl > 1.5 * args_cli.target_kl):
                        _kl_early_stop = True
                        break

                    # PPO clipped surrogate（A2C/PPO 共用）：
                    # ratio = π_new(a|s) / π_old(a|s)
                    # loss = -min(ratio*A, clip(ratio, 1±ε)*A)
                    # A2C 模式：single-epoch full-batch + clipping 防護
                    # PPO 模式：multi-epoch mini-batch + clipping
                    ratio = _log_ratio.exp()
                    s1 = ratio * flat_adv[mb]
                    s2 = torch.clamp(ratio, 1 - args_cli.clip_eps, 1 + args_cli.clip_eps) * flat_adv[mb]
                    pl = -torch.min(s1, s2).mean()
                    with torch.no_grad():
                        _clip_frac = ((ratio - 1.0).abs() > args_cli.clip_eps).float().mean().item()
                        ppo_clip_frac_l.append(_clip_frac)
                        ppo_ratio_l.append(ratio.mean().item())

                    _value_error = (nv - flat_value_target[mb]).pow(2)
                    if args_cli.value_clip_eps > 0.0:
                        _value_clipped = flat_old_value[mb] + (
                            nv - flat_old_value[mb]
                        ).clamp(-args_cli.value_clip_eps, args_cli.value_clip_eps)
                        _value_error_clipped = (_value_clipped - flat_value_target[mb]).pow(2)
                        vl = torch.maximum(_value_error, _value_error_clipped).mean()
                    else:
                        vl = _value_error.mean()
                    vl_raw = vl.item()  # Patch 1: 記錄 clamp 前的原始 vf_loss（用於診斷）

                    # WD A2CK: per-head entropy（線性/角速度分別用不同的 entropy coeff）
                    # 對應 WD: spot_entropy_coeff=0.04 × 2.5 = 0.10（線性）, angular=0.375
                    entropy_loss = (_ent_coeff_linear * ent_lin.mean()
                                    + _ent_coeff_angular * ent_ang.mean())

                    # WD A2CK loss clamping: 防止 catastrophic gradient spikes（WD 論文 §4.2）
                    # 閾值由 --policy_loss_clamp / --vf_term_clamp 控制
                    _pl_clamp_threshold = args_cli.policy_loss_clamp
                    _vf_clamp_threshold = args_cli.vf_term_clamp
                    _pl_clamp_triggered = abs(pl.item()) > _pl_clamp_threshold
                    pl_clamped = pl
                    if _pl_clamp_triggered:
                        pl_clamped = pl * (_pl_clamp_threshold / abs(pl.item()))
                    # vf_coeff × vf_loss 超過閾值 → 縮放（按 policy loss 量級動態調整）
                    vf_term = _current_vf_coeff * vl
                    _vf_clamp_triggered = abs(vf_term.item()) > _vf_clamp_threshold
                    if _vf_clamp_triggered:
                        _vf_scale = abs(max(min(_vf_clamp_threshold, abs(pl.item())), _vf_clamp_threshold / 3.0) / vl.item())
                        vl = vl * _vf_scale

                    _retention_term = (
                        (
                            _teacher_retention_weight
                            * _retention_result.loss
                            + _teacher_retention_margin_weight
                            * _retention_result.margin_loss
                            + _teacher_retention_action_ce_weight
                            * _retention_result.action_ce_loss
                        )
                        if _retention_result is not None
                        else torch.zeros((), device=device)
                    )
                    _previous_retention_term = (
                        _previous_stage_teacher_retention_weight
                        * _previous_retention_result.loss
                        if _previous_retention_result is not None
                        else torch.zeros((), device=device)
                    )
                    # N1: scripted 直穿 teacher 的 CE，只作用在窄縫 replay 幀。
                    # shadow 模式下係數為 0——照常算 CE 與梯度比但不影響更新。
                    _narrow_imitation_term = (
                        (0.0 if _narrow_imitation_shadow
                         else _narrow_imitation_weight)
                        * _narrow_imitation_result.loss
                        if _narrow_imitation_result is not None
                        else torch.zeros((), device=device)
                    )
                    # Ordinary frames remain pure PPO. Only injected narrow
                    # frames are anchored to c20, and only previous-stage
                    # replay frames are independently anchored to c12.
                    loss = (
                        pl_clamped
                        + _current_vf_coeff * vl
                        - entropy_loss
                        + _retention_term
                        + _previous_retention_term
                        + _narrow_imitation_term
                        + (
                            _corridor_adapter_gate_loss_weight
                            * _adapter_gate_loss
                        )
                    )
                    if (
                        _narrow_imitation_result is not None
                        and _narrow_imitation_result.active_count > 0
                    ):
                        _narrow_imitation_stats = {
                            "ce": float(_narrow_imitation_result.loss.detach()),
                            "ce_linear": float(
                                _narrow_imitation_result.ce_linear.detach()
                            ),
                            "ce_angular": float(
                                _narrow_imitation_result.ce_angular.detach()
                            ),
                            "agreement_linear":
                                _narrow_imitation_result.agreement_linear,
                            "agreement_angular":
                                _narrow_imitation_result.agreement_angular,
                            "active_frames": _narrow_imitation_result.active_count,
                        }
                        # N1 步驟 3/4：量「PPO actor 梯度」與「teacher CE 梯度」的
                        # 範數比供 λ 校準（目標 CE ≈ actor 的 50%）。兩次額外反傳
                        # 很貴，只在 shadow 校準模式下做——正式訓練不付這個成本。
                        if _narrow_imitation_shadow:
                            _ni_actor_params = [
                                p for p in charge_params_actor if p.requires_grad
                            ]
                            _ni_g_ppo = torch.autograd.grad(
                                pl_clamped, _ni_actor_params,
                                retain_graph=True, allow_unused=True,
                            )
                            _ni_g_ce = torch.autograd.grad(
                                _narrow_imitation_result.loss, _ni_actor_params,
                                retain_graph=True, allow_unused=True,
                            )

                            def _gnorm(grads):
                                total = 0.0
                                for g in grads:
                                    if g is not None:
                                        total += float(g.detach().pow(2).sum())
                                return total ** 0.5

                            _ni_ratio = _gnorm(_ni_g_ce) / max(
                                _gnorm(_ni_g_ppo), 1e-12
                            )
                            _narrow_imitation_stats.update({
                                "grad_ppo_actor": _gnorm(_ni_g_ppo),
                                "grad_ce_unweighted": _gnorm(_ni_g_ce),
                                "grad_ratio_at_lambda1": _ni_ratio,
                                # 讓 CE 梯度 = PPO actor 梯度 50% 所需的 λ。
                                "lambda_for_half_ppo": 0.5 / max(_ni_ratio, 1e-12),
                            })
                    actor_before = _snapshot_params(charge_params_actor)
                    critic_before = _snapshot_params(charge_params_critic)

                    charge_opt_rl.zero_grad()
                    if charge_opt_rnn_rl is not None:
                        charge_opt_rnn_rl.zero_grad()
                    loss.backward()

                    # WD thesis-inspired update monitor（模組熵診斷）：
                    #   module_entropy = log10(|actor update|) - log10(|critic update|)
                    # 這裡用 lr × grad_norm 估計更新幅度（Adam 的 adaptive rescaling 之前）。
                    # 這是透明的診斷指標，不等同於 WD 的 custom optimizer，
                    # 但可以反映 actor/critic 更新壓力是否失衡。
                    actor_grad = _grad_l2_norm(charge_params_actor)   # actor 梯度 L2 norm
                    critic_grad = _grad_l2_norm(charge_params_critic)  # critic 梯度 L2 norm
                    encoder_grad_l.append(
                        _grad_l2_norm(extractor.parameters()) if _e2e_frame_stack else 0.0
                    )
                    actor_update_est = _current_rl_lr * actor_grad    # 估計 actor 更新量
                    critic_update_est = _current_rl_lr * critic_grad  # 估計 critic 更新量

                    actor_scale = 1.0
                    critic_scale = 1.0
                    if args_cli.wd_update_clip and not args_cli.wd_update_monitor_only:
                        # WD thesis eq. 4.9/4.10：分別對 actor/critic 梯度做 cap（k=8, q=30）。
                        # 防止 critic spike 透過 merged clip 壓制 actor 更新。
                        # 在 PyTorch port 中，直接對 per-module grad norm 做 cap：
                        #   actor_scale = k / grad_norm（若 norm > k）
                        #   critic_scale = q / grad_norm（若 norm > q）
                        if actor_grad > _current_wd_actor_update_clip:
                            actor_scale = _current_wd_actor_update_clip / (actor_grad + 1e-12)
                        if critic_grad > _current_wd_critic_update_clip:
                            critic_scale = _current_wd_critic_update_clip / (critic_grad + 1e-12)
                        _scale_grads(charge_params_actor, actor_scale)   # actor 梯度縮放
                        _scale_grads(charge_params_critic, critic_scale) # critic 梯度縮放

                    # Patch 4: 記錄 clip 前後的 merged grad norm（診斷用）
                    _merged_pre = _grad_l2_norm(charge_params_rl)  # clip 之前的合併 grad norm
                    if args_cli.wd_update_clip and not args_cli.wd_update_monitor_only:
                        # WD-style 模式：不做 merged global clip（已在上方做 per-module cap）。
                        # 原因：merged clip 在 critic 主導時會同時縮小 actor 梯度，
                        # 可能讓 actor 學不到任何東西。
                        _merged_post = _grad_l2_norm(charge_params_rl)
                    else:
                        # 標準模式：做 global clip_grad_norm_（返回裁切後的 norm）
                        _merged_post = nn.utils.clip_grad_norm_(charge_params_rl, _current_max_grad_norm)
                    # Patch 4: per-module grad norm after all clipping
                    _actor_grad_post = _grad_l2_norm(charge_params_actor)
                    _critic_grad_post = _grad_l2_norm(charge_params_critic)

                    if not _narrow_imitation_shadow:
                        charge_opt_rl.step()
                        _ppo_update_count += 1
                        if charge_opt_rnn_rl is not None:
                            # RNN 走 RL 梯度:clip 後 step(防 spike)
                            _rnn_rl_gn = nn.utils.clip_grad_norm_(
                                [p for g in charge_opt_rnn_rl.param_groups for p in g["params"]],
                                _current_max_grad_norm)
                            charge_opt_rnn_rl.step()
                    else:
                        # Leave no stale gradients that could be mistaken for
                        # an update by later diagnostics or future code.
                        charge_opt_rl.zero_grad(set_to_none=True)
                        if charge_opt_rnn_rl is not None:
                            charge_opt_rnn_rl.zero_grad(set_to_none=True)

                    # --- post-update policy diagnostics ---
                    with torch.no_grad():
                        if _e2e_frame_stack:
                            _post_feat = extractor(flat_encoder_input[mb])
                            _post_ri = torch.cat(
                                [flat_encoder_input[mb][:, :policy_obs_dim], _post_feat], dim=-1
                            )
                        else:
                            _post_ri = flat_ri[mb]
                        _post_ri_enc = _encode_rl_input(_post_ri)
                        _post_nl = policy_head(_post_ri_enc, (flat_priv[mb] if (_oracle_to_policy and flat_priv is not None) else None))
                        if corridor_adapter is not None:
                            _post_pobs = _post_ri[:, :policy_obs_dim]
                            (
                                _post_adapter_residual,
                                _post_adapter_gate_logits,
                                _post_adapter_gate_probability,
                            ) = corridor_adapter(
                                _post_pobs,
                                (
                                    _post_ri_enc
                                    if _corridor_adapter_residual_features
                                    == "policy_features"
                                    else None
                                ),
                            )
                            _post_nl = _post_nl + _post_adapter_residual
                            _post_corridor_mask = flat_corridor_mask[mb]
                            _post_other_mask = ~_post_corridor_mask
                            adapter_gate_loss_l.append(
                                balanced_binary_gate_loss(
                                    _post_adapter_gate_logits,
                                    _post_corridor_mask,
                                ).item()
                            )
                            if bool(_post_corridor_mask.any()):
                                _corridor_prob = _post_adapter_gate_probability[
                                    _post_corridor_mask
                                ]
                                adapter_gate_corridor_prob_l.append(
                                    _corridor_prob.mean().item()
                                )
                                adapter_gate_recall_l.append(
                                    (_corridor_prob >= 0.5).float().mean().item()
                                )
                                adapter_residual_corridor_l2_l.append(
                                    _post_adapter_residual[
                                        _post_corridor_mask
                                    ].norm(dim=-1).mean().item()
                                )
                            if bool(_post_other_mask.any()):
                                _other_prob = _post_adapter_gate_probability[
                                    _post_other_mask
                                ]
                                adapter_gate_other_prob_l.append(
                                    _other_prob.mean().item()
                                )
                                adapter_gate_specificity_l.append(
                                    (_other_prob < 0.5).float().mean().item()
                                )
                                adapter_residual_other_l2_l.append(
                                    _post_adapter_residual[
                                        _post_other_mask
                                    ].norm(dim=-1).mean().item()
                                )
                        _post_lp, _, _ = evaluate_actions(_post_nl, flat_act[mb])
                        _post_ratio = (_post_lp - flat_lp[mb]).exp()
                        post_kl_l.append((flat_lp[mb] - _post_lp).mean().item())
                        post_ratio_mean_l.append(_post_ratio.mean().item())
                        post_ratio_std_l.append(_post_ratio.std().item())
                        post_ratio_min_l.append(_post_ratio.min().item())
                        post_ratio_max_l.append(_post_ratio.max().item())
                        post_ratio_outside_20_l.append(((_post_ratio - 1.0).abs() > 0.2).float().mean().item())
                        post_ratio_outside_50_l.append(((_post_ratio - 1.0).abs() > 0.5).float().mean().item())
                        if _teacher_retention_enabled:
                            _post_retention_result = (
                                masked_two_head_retention_loss(
                                    flat_teacher_logits[mb],
                                    _post_nl,
                                    flat_retention_mask[mb],
                                    num_bins=NUM_BINS,
                                    argmax_margin=_teacher_retention_argmax_margin,
                                )
                            )
                            if _post_retention_result.active_count > 0:
                                retention_post_kl_linear_l.append(
                                    _post_retention_result.kl_linear.item()
                                )
                                retention_post_kl_angular_l.append(
                                    _post_retention_result.kl_angular.item()
                                )
                                retention_post_margin_linear_l.append(
                                    _post_retention_result.margin_linear.item()
                                )
                                retention_post_margin_angular_l.append(
                                    _post_retention_result.margin_angular.item()
                                )
                                retention_post_agree_linear_l.append(
                                    _post_retention_result.agreement_linear.item()
                                )
                                retention_post_agree_angular_l.append(
                                    _post_retention_result.agreement_angular.item()
                                )

                    actor_delta = _param_delta_norm(actor_before, charge_params_actor)
                    critic_delta = _param_delta_norm(critic_before, charge_params_critic)
                    module_entropy = math.log10(actor_update_est + args_cli.wd_module_entropy_eps) - math.log10(
                        critic_update_est + args_cli.wd_module_entropy_eps)

                    # Patch 5: param norms and update ratios
                    _actor_pnorm = _param_l2_norm(charge_params_actor)
                    _critic_pnorm = _param_l2_norm(charge_params_critic)

                    # --- Append all tracking lists ---
                    ppo_l.append(pl.item()); vf_l.append(vl.item())
                    ent_l.append((ent_lin.mean() + ent_ang.mean()).item())
                    # Patch 1: raw/clamped separation
                    vf_raw_l.append(vl_raw)
                    pl_clamped_l.append(pl_clamped.item())
                    vf_clamped_l.append(vl.item())
                    pl_clamp_triggered_l.append(float(_pl_clamp_triggered))
                    vf_clamp_triggered_l.append(float(_vf_clamp_triggered))
                    # Patch 3: approx_kl + per-head entropy
                    approx_kl_l.append(_approx_kl)
                    ent_lin_l.append(ent_lin.mean().item())
                    ent_ang_l.append(ent_ang.mean().item())
                    # Patch 5: total loss
                    total_loss_l.append(loss.item())
                    if (
                        _retention_result is not None
                        and _retention_result.active_count > 0
                    ):
                        retention_active_counts.append(
                            _retention_result.active_count
                        )
                        retention_kl_linear_l.append(
                            _retention_result.kl_linear.item()
                        )
                        retention_kl_angular_l.append(
                            _retention_result.kl_angular.item()
                        )
                        retention_margin_linear_l.append(
                            _retention_result.margin_linear.item()
                        )
                        retention_margin_angular_l.append(
                            _retention_result.margin_angular.item()
                        )
                        retention_action_ce_linear_l.append(
                            _retention_result.action_ce_linear.item()
                        )
                        retention_action_ce_angular_l.append(
                            _retention_result.action_ce_angular.item()
                        )
                        retention_agree_linear_l.append(
                            _retention_result.agreement_linear.item()
                        )
                        retention_agree_angular_l.append(
                            _retention_result.agreement_angular.item()
                        )
                        retention_loss_l.append(_retention_term.item())
                    if (
                        _previous_retention_result is not None
                        and _previous_retention_result.active_count > 0
                    ):
                        previous_retention_active_counts.append(
                            _previous_retention_result.active_count
                        )
                        previous_retention_kl_linear_l.append(
                            _previous_retention_result.kl_linear.item()
                        )
                        previous_retention_kl_angular_l.append(
                            _previous_retention_result.kl_angular.item()
                        )
                        previous_retention_agree_linear_l.append(
                            _previous_retention_result.agreement_linear.item()
                        )
                        previous_retention_agree_angular_l.append(
                            _previous_retention_result.agreement_angular.item()
                        )
                        previous_retention_loss_l.append(
                            _previous_retention_term.item()
                        )
                    # Existing wd_update tracking
                    wd_actor_grad_l.append(actor_grad)
                    wd_critic_grad_l.append(critic_grad)
                    wd_actor_update_l.append(actor_update_est)
                    wd_critic_update_l.append(critic_update_est)
                    wd_actor_delta_l.append(actor_delta)
                    wd_critic_delta_l.append(critic_delta)
                    wd_actor_clip_l.append(float(actor_scale < 1.0))
                    wd_critic_clip_l.append(float(critic_scale < 1.0))
                    wd_module_entropy_l.append(module_entropy)
                    # Patch 4: before/after clip
                    wd_actor_grad_post_l.append(_actor_grad_post)
                    wd_critic_grad_post_l.append(_critic_grad_post)
                    wd_merged_grad_pre_l.append(_merged_pre)
                    wd_merged_grad_post_l.append(_merged_post if isinstance(_merged_post, float) else _merged_post.item())
                    # Patch 5: update ratio
                    wd_actor_param_norm_l.append(_actor_pnorm)
                    wd_critic_param_norm_l.append(_critic_pnorm)
                    wd_actor_update_ratio_l.append(actor_delta / (_actor_pnorm + 1e-12))
                    wd_critic_update_ratio_l.append(critic_delta / (_critic_pnorm + 1e-12))

                if _kl_early_stop:
                    break

            if _corridor_teacher_distill_enabled:
                if _corridor_teacher_sample_indices:
                    _corridor_projection_indices = torch.cat(
                        _corridor_teacher_sample_indices, dim=0
                    )
                    _corridor_projection_actions = torch.cat(
                        _corridor_teacher_actions, dim=0
                    )
                else:
                    _corridor_projection_indices = torch.empty(
                        0, dtype=torch.long, device=device
                    )
                    _corridor_projection_actions = torch.empty(
                        0, 2, dtype=torch.long, device=device
                    )

                def _corridor_projection_student_forward(
                    sample_indices: torch.Tensor,
                ) -> torch.Tensor:
                    with torch.no_grad():
                        encoder_input = flat_encoder_input[
                            sample_indices.long()
                        ]
                        feature = extractor(encoder_input)
                        policy_obs_projection = encoder_input[
                            :, :policy_obs_dim
                        ]
                        policy_input = torch.cat(
                            [policy_obs_projection, feature], dim=-1
                        )
                    return policy_head(policy_input)

                _corridor_projection_stats = (
                    post_update_corridor_action_projection(
                        _corridor_projection_student_forward,
                        policy_head.parameters(),
                        _corridor_projection_indices,
                        _corridor_projection_actions,
                        num_bins=NUM_BINS,
                        epochs=_corridor_teacher_distill_epochs,
                        learning_rate=_corridor_teacher_distill_lr,
                        batch_size=(
                            _corridor_teacher_distill_batch_size
                        ),
                        max_grad_norm=(
                            _corridor_teacher_distill_max_grad_norm
                        ),
                        neighbor_mass=(
                            _corridor_teacher_distill_neighbor_mass
                        ),
                    )
                )
                charge_opt_rl.zero_grad(set_to_none=True)

            if _teacher_retention_post_kl_epochs > 0:
                _projection_inputs = torch.arange(
                    batch_size, device=device, dtype=torch.long
                )

                def _projection_student_forward(
                    sample_indices: torch.Tensor,
                ) -> torch.Tensor:
                    encoder_input = flat_encoder_input[sample_indices.long()]
                    feature = extractor(encoder_input)
                    policy_obs_projection = encoder_input[:, :policy_obs_dim]
                    policy_input = torch.cat(
                        [policy_obs_projection, feature], dim=-1
                    )
                    privileged = (
                        flat_priv[sample_indices.long()]
                        if (_oracle_to_policy and flat_priv is not None)
                        else None
                    )
                    return policy_head(
                        _encode_rl_input(policy_input), privileged
                    )

                _projection_parameters = (
                    policy_head.parameters()
                    if _teacher_retention_post_policy_head_only
                    else charge_params_actor
                )
                if _teacher_retention_post_anchor_weight > 0.0:
                    _dual_projection_stats = (
                        post_update_dual_teacher_projection(
                            _projection_student_forward,
                            _projection_parameters,
                            _projection_inputs,
                            flat_teacher_logits,
                            flat_retention_mask,
                            flat_previous_teacher_logits,
                            flat_previous_retention_mask,
                            anchor_weight=(
                                _teacher_retention_post_anchor_weight
                            ),
                            num_bins=NUM_BINS,
                            epochs=_teacher_retention_post_kl_epochs,
                            learning_rate=_teacher_retention_post_kl_lr,
                            batch_size=(
                                _teacher_retention_post_kl_batch_size
                            ),
                            max_grad_norm=(
                                _teacher_retention_post_kl_max_grad_norm
                            ),
                            argmax_margin=(
                                _teacher_retention_argmax_margin
                            ),
                            margin_weight=(
                                _teacher_retention_post_margin_weight
                            ),
                            action_ce_weight=(
                                _teacher_retention_post_action_ce_weight
                            ),
                        )
                    )
                    _retention_projection_stats = (
                        _dual_projection_stats.primary
                    )
                    _retention_anchor_projection_stats = (
                        _dual_projection_stats.anchor
                    )
                else:
                    _retention_projection_stats = post_update_kl_projection(
                        _projection_student_forward,
                        _projection_parameters,
                        _projection_inputs,
                        flat_teacher_logits,
                        flat_retention_mask,
                        num_bins=NUM_BINS,
                        epochs=_teacher_retention_post_kl_epochs,
                        learning_rate=_teacher_retention_post_kl_lr,
                        batch_size=_teacher_retention_post_kl_batch_size,
                        max_grad_norm=(
                            _teacher_retention_post_kl_max_grad_norm
                        ),
                        argmax_margin=_teacher_retention_argmax_margin,
                        margin_weight=(
                            _teacher_retention_post_margin_weight
                        ),
                        action_ce_weight=(
                            _teacher_retention_post_action_ce_weight
                        ),
                    )
                charge_opt_rl.zero_grad(set_to_none=True)

            charge_ppo_loss = np.mean(ppo_l)
            charge_vf_loss = np.mean(vf_l)
            charge_entropy = np.mean(ent_l)

            def _retention_weighted_mean(values):
                if not values or not retention_active_counts:
                    return 0.0
                total = sum(retention_active_counts)
                return float(
                    sum(
                        value * count
                        for value, count in zip(
                            values, retention_active_counts
                        )
                    )
                    / max(total, 1)
                )

            def _previous_retention_weighted_mean(values):
                if not values or not previous_retention_active_counts:
                    return 0.0
                total = sum(previous_retention_active_counts)
                return float(
                    sum(
                        value * count
                        for value, count in zip(
                            values, previous_retention_active_counts
                        )
                    )
                    / max(total, 1)
                )

            wd_update_monitor = {
                # --- WD update diagnostics split by module for WandB grouping ---
                "wd_update_actor/grad_norm": float(np.mean(wd_actor_grad_l)) if wd_actor_grad_l else 0.0,
                "rl_encoder/grad_norm_pre_clip": float(np.mean(encoder_grad_l)) if encoder_grad_l else 0.0,
                "wd_update_actor/update_est": float(np.mean(wd_actor_update_l)) if wd_actor_update_l else 0.0,
                "wd_update_actor/param_delta_norm": float(np.mean(wd_actor_delta_l)) if wd_actor_delta_l else 0.0,
                "wd_update_actor/clip_fraction": float(np.mean(wd_actor_clip_l)) if wd_actor_clip_l else 0.0,
                "wd_update_actor/grad_norm_post_clip": float(np.mean(wd_actor_grad_post_l)) if wd_actor_grad_post_l else 0.0,
                "wd_update_actor/param_norm": float(np.mean(wd_actor_param_norm_l)) if wd_actor_param_norm_l else 0.0,
                "wd_update_actor/update_ratio": float(np.mean(wd_actor_update_ratio_l)) if wd_actor_update_ratio_l else 0.0,
                "wd_update_critic/grad_norm": float(np.mean(wd_critic_grad_l)) if wd_critic_grad_l else 0.0,
                "wd_update_critic/update_est": float(np.mean(wd_critic_update_l)) if wd_critic_update_l else 0.0,
                "wd_update_critic/param_delta_norm": float(np.mean(wd_critic_delta_l)) if wd_critic_delta_l else 0.0,
                "wd_update_critic/clip_fraction": float(np.mean(wd_critic_clip_l)) if wd_critic_clip_l else 0.0,
                "wd_update_critic/grad_norm_post_clip": float(np.mean(wd_critic_grad_post_l)) if wd_critic_grad_post_l else 0.0,
                "wd_update_critic/param_norm": float(np.mean(wd_critic_param_norm_l)) if wd_critic_param_norm_l else 0.0,
                "wd_update_critic/update_ratio": float(np.mean(wd_critic_update_ratio_l)) if wd_critic_update_ratio_l else 0.0,
                # --- Cross-module / merged diagnostics stay under wd_update ---
                "wd_update/module_entropy": float(np.mean(wd_module_entropy_l)) if wd_module_entropy_l else 0.0,
                "wd_update/merged_grad_norm_pre_clip": float(np.mean(wd_merged_grad_pre_l)) if wd_merged_grad_pre_l else 0.0,
                "wd_update/merged_grad_norm_post_clip": float(np.mean(wd_merged_grad_post_l)) if wd_merged_grad_post_l else 0.0,
                # --- Patch 1: raw vs clamped loss ---
                "rl/raw_vf_loss": float(np.mean(vf_raw_l)) if vf_raw_l else 0.0,
                "rl/clamped_policy_loss": float(np.mean(pl_clamped_l)) if pl_clamped_l else 0.0,
                "rl/clamped_vf_loss": float(np.mean(vf_clamped_l)) if vf_clamped_l else 0.0,
                "rl/vf_coeff": _current_vf_coeff,
                "rl/vf_coeff_times_vf_loss": _current_vf_coeff * float(np.mean(vf_raw_l)) if vf_raw_l else 0.0,
                "rl/policy_clamp_triggered": float(np.mean(pl_clamp_triggered_l)) if pl_clamp_triggered_l else 0.0,
                "rl/vf_clamp_triggered": float(np.mean(vf_clamp_triggered_l)) if vf_clamp_triggered_l else 0.0,
                # --- Patch 3: approx_kl + per-head entropy ---
                "rl/approx_kl": float(np.mean(approx_kl_l)) if approx_kl_l else 0.0,
                "rl/kl_early_stop": float(_kl_early_stop),
                "rl/ppo_update_count": float(_ppo_update_count),
                "rl/entropy_linear": float(np.mean(ent_lin_l)) if ent_lin_l else 0.0,
                "rl/entropy_angular": float(np.mean(ent_ang_l)) if ent_ang_l else 0.0,
                # --- Patch 5: total loss ---
                "rl/total_loss": float(np.mean(total_loss_l)) if total_loss_l else 0.0,
                "corridor_adapter/enabled": float(
                    _corridor_adapter_enabled
                ),
                "corridor_adapter/gate_bce": (
                    float(np.mean(adapter_gate_loss_l))
                    if adapter_gate_loss_l else 0.0
                ),
                "corridor_adapter/gate_probability_corridor": (
                    float(np.mean(adapter_gate_corridor_prob_l))
                    if adapter_gate_corridor_prob_l else 0.0
                ),
                "corridor_adapter/gate_probability_other": (
                    float(np.mean(adapter_gate_other_prob_l))
                    if adapter_gate_other_prob_l else 0.0
                ),
                "corridor_adapter/gate_recall": (
                    float(np.mean(adapter_gate_recall_l))
                    if adapter_gate_recall_l else 0.0
                ),
                "corridor_adapter/gate_specificity": (
                    float(np.mean(adapter_gate_specificity_l))
                    if adapter_gate_specificity_l else 0.0
                ),
                "corridor_adapter/residual_l2_corridor": (
                    float(np.mean(adapter_residual_corridor_l2_l))
                    if adapter_residual_corridor_l2_l else 0.0
                ),
                "corridor_adapter/residual_l2_other": (
                    float(np.mean(adapter_residual_other_l2_l))
                    if adapter_residual_other_l2_l else 0.0
                ),
                "corridor_distill/enabled": float(
                    _corridor_teacher_distill_enabled
                ),
                "corridor_distill/intervention_only": float(
                    _corridor_teacher_intervention_only
                ),
                "corridor_distill/candidate_count": float(
                    _corridor_teacher_label_stats["candidates"]
                ),
                "corridor_distill/feasible_count": float(
                    _corridor_teacher_label_stats["feasible"]
                ),
                "corridor_distill/intervention_count": float(
                    _corridor_teacher_label_stats["interventions"]
                ),
                "corridor_distill/selected_fraction": (
                    _corridor_teacher_label_stats["selected"]
                    / max(
                        _corridor_teacher_label_stats["candidates"],
                        1,
                    )
                ),
                "corridor_distill/intervention_fraction": (
                    _corridor_teacher_label_stats["interventions"]
                    / max(
                        _corridor_teacher_label_stats["candidates"],
                        1,
                    )
                ),
                "corridor_distill/policy_obstacle_collision_fraction": (
                    _corridor_teacher_label_stats[
                        "policy_obstacle_collision"
                    ]
                    / max(
                        _corridor_teacher_label_stats["candidates"],
                        1,
                    )
                ),
                "corridor_distill/policy_wall_collision_fraction": (
                    _corridor_teacher_label_stats[
                        "policy_wall_collision"
                    ]
                    / max(
                        _corridor_teacher_label_stats["candidates"],
                        1,
                    )
                ),
                "corridor_distill/policy_low_clearance_fraction": (
                    _corridor_teacher_label_stats[
                        "policy_low_clearance"
                    ]
                    / max(
                        _corridor_teacher_label_stats["candidates"],
                        1,
                    )
                ),
                "corridor_distill/active_count": float(
                    _corridor_projection_stats.active_count
                    if _corridor_projection_stats is not None
                    else 0
                ),
                "corridor_distill/optimizer_steps": float(
                    _corridor_projection_stats.optimizer_steps
                    if _corridor_projection_stats is not None
                    else 0
                ),
                "corridor_distill/loss_before": float(
                    _corridor_projection_stats.loss_before
                    if _corridor_projection_stats is not None
                    else 0.0
                ),
                "corridor_distill/loss_after": float(
                    _corridor_projection_stats.loss_after
                    if _corridor_projection_stats is not None
                    else 0.0
                ),
                "corridor_distill/agreement_before_joint": float(
                    _corridor_projection_stats.agreement_before_joint
                    if _corridor_projection_stats is not None
                    else 0.0
                ),
                "corridor_distill/agreement_after_joint": float(
                    _corridor_projection_stats.agreement_after_joint
                    if _corridor_projection_stats is not None
                    else 0.0
                ),
                "corridor_distill/within_one_before_joint": float(
                    _corridor_projection_stats.within_one_before_joint
                    if _corridor_projection_stats is not None
                    else 0.0
                ),
                "corridor_distill/within_one_after_joint": float(
                    _corridor_projection_stats.within_one_after_joint
                    if _corridor_projection_stats is not None
                    else 0.0
                ),
                "corridor_distill/agreement_after_linear": float(
                    _corridor_projection_stats.agreement_after_linear
                    if _corridor_projection_stats is not None
                    else 0.0
                ),
                "corridor_distill/agreement_after_angular": float(
                    _corridor_projection_stats.agreement_after_angular
                    if _corridor_projection_stats is not None
                    else 0.0
                ),
                "retention/enabled": float(_teacher_retention_enabled),
                "retention/beta": _teacher_retention_weight,
                "retention/margin_weight": _teacher_retention_margin_weight,
                "retention/action_ce_weight": _teacher_retention_action_ce_weight,
                "retention/argmax_margin": _teacher_retention_argmax_margin,
                "previous_retention/enabled": float(
                    _previous_stage_teacher_enabled
                ),
                "previous_retention/beta": (
                    _previous_stage_teacher_retention_weight
                ),
                "retention/post_kl_projection_epochs": float(
                    _teacher_retention_post_kl_epochs
                ),
                "retention/post_kl_projection_lr": (
                    _teacher_retention_post_kl_lr
                ),
                "retention/post_kl_projection_margin_weight": (
                    _teacher_retention_post_margin_weight
                ),
                "retention/post_kl_projection_action_ce_weight": (
                    _teacher_retention_post_action_ce_weight
                ),
                "retention/post_kl_projection_policy_head_only": float(
                    _teacher_retention_post_policy_head_only
                ),
                "retention/post_anchor_weight": (
                    _teacher_retention_post_anchor_weight
                ),
                "retention/rollout_override": float(
                    _teacher_retention_rollout_override
                ),
                "retention/teacher_forced_fraction": (
                    _teacher_forced_count / max(RL * num_envs, 1)
                ),
                "retention/post_kl_projection_active_count": float(
                    _retention_projection_stats.active_count
                    if _retention_projection_stats is not None
                    else 0
                ),
                "retention/post_kl_projection_optimizer_steps": float(
                    _retention_projection_stats.optimizer_steps
                    if _retention_projection_stats is not None
                    else 0
                ),
                "retention/post_kl_projection_before_linear": float(
                    _retention_projection_stats.kl_before_linear
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_before_angular": float(
                    _retention_projection_stats.kl_before_angular
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_after_linear": float(
                    _retention_projection_stats.kl_after_linear
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_after_angular": float(
                    _retention_projection_stats.kl_after_angular
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_agree_after_linear": float(
                    _retention_projection_stats.agreement_after_linear
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_agree_after_angular": float(
                    _retention_projection_stats.agreement_after_angular
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_margin_before_linear": float(
                    _retention_projection_stats.margin_before_linear
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_margin_before_angular": float(
                    _retention_projection_stats.margin_before_angular
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_margin_after_linear": float(
                    _retention_projection_stats.margin_after_linear
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_kl_projection_margin_after_angular": float(
                    _retention_projection_stats.margin_after_angular
                    if _retention_projection_stats is not None
                    else 0.0
                ),
                "retention/post_anchor_active_count": float(
                    _retention_anchor_projection_stats.active_count
                    if _retention_anchor_projection_stats is not None
                    else 0
                ),
                "retention/post_anchor_before_linear": float(
                    _retention_anchor_projection_stats.kl_before_linear
                    if _retention_anchor_projection_stats is not None
                    else 0.0
                ),
                "retention/post_anchor_before_angular": float(
                    _retention_anchor_projection_stats.kl_before_angular
                    if _retention_anchor_projection_stats is not None
                    else 0.0
                ),
                "retention/post_anchor_after_linear": float(
                    _retention_anchor_projection_stats.kl_after_linear
                    if _retention_anchor_projection_stats is not None
                    else 0.0
                ),
                "retention/post_anchor_after_angular": float(
                    _retention_anchor_projection_stats.kl_after_angular
                    if _retention_anchor_projection_stats is not None
                    else 0.0
                ),
                "retention/post_anchor_agree_after_linear": float(
                    _retention_anchor_projection_stats.agreement_after_linear
                    if _retention_anchor_projection_stats is not None
                    else 0.0
                ),
                "retention/post_anchor_agree_after_angular": float(
                    _retention_anchor_projection_stats.agreement_after_angular
                    if _retention_anchor_projection_stats is not None
                    else 0.0
                ),
                "retention/active_fraction": (
                    float(flat_retention_mask.float().mean().item())
                    if flat_retention_mask is not None else 0.0
                ),
                "retention/kl_linear": _retention_weighted_mean(
                    retention_kl_linear_l
                ),
                "retention/kl_angular": _retention_weighted_mean(
                    retention_kl_angular_l
                ),
                "retention/margin_linear": _retention_weighted_mean(
                    retention_margin_linear_l
                ),
                "retention/margin_angular": _retention_weighted_mean(
                    retention_margin_angular_l
                ),
                "retention/action_ce_linear": _retention_weighted_mean(
                    retention_action_ce_linear_l
                ),
                "retention/action_ce_angular": _retention_weighted_mean(
                    retention_action_ce_angular_l
                ),
                "retention/action_agreement_linear": _retention_weighted_mean(
                    retention_agree_linear_l
                ),
                "retention/action_agreement_angular": _retention_weighted_mean(
                    retention_agree_angular_l
                ),
                "retention/post_kl_linear": _retention_weighted_mean(
                    retention_post_kl_linear_l
                ),
                "retention/post_kl_angular": _retention_weighted_mean(
                    retention_post_kl_angular_l
                ),
                "retention/post_margin_linear": _retention_weighted_mean(
                    retention_post_margin_linear_l
                ),
                "retention/post_margin_angular": _retention_weighted_mean(
                    retention_post_margin_angular_l
                ),
                "retention/post_action_agreement_linear": _retention_weighted_mean(
                    retention_post_agree_linear_l
                ),
                "retention/post_action_agreement_angular": _retention_weighted_mean(
                    retention_post_agree_angular_l
                ),
                "retention/weighted_loss": _retention_weighted_mean(
                    retention_loss_l
                ),
                "previous_retention/active_fraction": (
                    float(
                        flat_previous_retention_mask.float().mean().item()
                    )
                    if flat_previous_retention_mask is not None
                    else 0.0
                ),
                "previous_retention/kl_linear": (
                    _previous_retention_weighted_mean(
                        previous_retention_kl_linear_l
                    )
                ),
                "previous_retention/kl_angular": (
                    _previous_retention_weighted_mean(
                        previous_retention_kl_angular_l
                    )
                ),
                "previous_retention/action_agreement_linear": (
                    _previous_retention_weighted_mean(
                        previous_retention_agree_linear_l
                    )
                ),
                "previous_retention/action_agreement_angular": (
                    _previous_retention_weighted_mean(
                        previous_retention_agree_angular_l
                    )
                ),
                "previous_retention/weighted_loss": (
                    _previous_retention_weighted_mean(
                        previous_retention_loss_l
                    )
                ),
                # --- Patch 2: returns/advantage statistics ---
                "rl_critic/returns_mean": _ret_mean,
                "rl_critic/returns_std": _ret_std,
                "rl_critic/returns_min": _ret_min,
                "rl_critic/returns_max": _ret_max,
                # Clear aliases: critic target distribution vs current value prediction
                "rl_critic/value_target_mean": _target_mean,
                "rl_critic/value_target_std": _target_std,
                "rl_critic/value_target_min": _target_min,
                "rl_critic/value_target_max": _target_max,
                "rl_critic/value_pred_mean": _value_pred_mean,
                "rl_critic/value_pred_std": _value_pred_std,
                "rl_critic/value_pred_min": _value_pred_min,
                "rl_critic/value_pred_max": _value_pred_max,
                "rl_critic/value_error_mean": _value_error_mean,
                "rl_critic/value_error_std": _value_error_std,
                "rl_critic/value_error_abs_max": _value_error_abs_max,
                "rl_adv/mean": _adv_mean,
                "rl_adv/std": _adv_std,
                "rl_adv/min": _adv_min,
                "rl_adv/max": _adv_max,
                "rl_adv/raw_std": _raw_adv_std,
            }
            # Patch 3: PPO-only metrics (conditional)
            if ppo_clip_frac_l:
                wd_update_monitor["rl/clip_fraction"] = float(np.mean(ppo_clip_frac_l))
                wd_update_monitor["rl/ratio_mean"] = float(np.mean(ppo_ratio_l))

            # Post-update trust-region diagnostics
            if post_kl_l:
                wd_update_monitor["rl_trust/approx_kl"] = float(np.mean(post_kl_l))
                wd_update_monitor["rl_trust/ratio_mean"] = float(np.mean(post_ratio_mean_l))
                wd_update_monitor["rl_trust/ratio_std"] = float(np.mean(post_ratio_std_l))
                wd_update_monitor["rl_trust/ratio_min"] = float(np.mean(post_ratio_min_l))
                wd_update_monitor["rl_trust/ratio_max"] = float(np.mean(post_ratio_max_l))
                wd_update_monitor["rl_trust/ratio_outside_20pct"] = float(np.mean(post_ratio_outside_20_l))
                wd_update_monitor["rl_trust/ratio_outside_50pct"] = float(np.mean(post_ratio_outside_50_l))

            # ================================================================
            # Auxiliary loss (WD module loss) — 每 iteration 都跑 (WD: 論文§4.2)
            # Skip if WD-order already ran it before RL (see _aux_already_done).
            # ================================================================
            if _aux_already_done:
                # WD-order: aux已在 RL 之前完成，此處只做 monitoring
                aux_per_feature = {}
                aux_monitor = {}
                _mon_modules = {
                    "rnn": list(preprocess_rnn.rnn.parameters()),
                    "fc_front": list(preprocess_rnn.fc_front.parameters()),
                    "fc_middle": list(preprocess_rnn.fc_middle.parameters()),
                    "predict_head": list(preprocess_rnn.predict_head.parameters()),
                }
                if use_extractor:
                    _mon_modules["extractor"] = list(extractor.parameters())
                _grad_norms = {k: 0.0 for k in _mon_modules}
                _delta_norms = {k: 0.0 for k in _mon_modules}
                _aux_valid_seq_count = 0
                _aux_actual_batch = 0
                # aux_loss_val set to 0 (will be overwritten by monitoring section)
                aux_monitor["aux/loss_per_step"] = 0.0

            if not _aux_already_done:
                # Legacy order (PPO mode): aux runs AFTER RL update.
                # Param delta 判讀 (conservative output-side unfreeze):
                #   rnn_param_delta_norm > 0       → 正常 (RNN cell 主力學習)
                #   predict_head_param_delta > 0   → 正常 (lr>0, 解凍 aux 輸出映射)
                #   fc_middle_param_delta > 0      → 正常 (lr>0, 輕微解凍)
                #   fc_front_param_delta ≈ 0       → 正常 (lr=0, 維持 RNN 輸入穩定)
                #   extractor_param_delta ≈ 0      → 正常 (lr=0, 隔離輸入特徵問題)
                aux_per_feature = {}
                aux_monitor = {}
                if use_extractor: extractor.train()
                preprocess_rnn.train()

            # Per-module param lists for monitoring
            _mon_modules = {
                "rnn": list(preprocess_rnn.rnn.parameters()),
                "fc_front": list(preprocess_rnn.fc_front.parameters()),
                "fc_middle": list(preprocess_rnn.fc_middle.parameters()),
                "predict_head": list(preprocess_rnn.predict_head.parameters()),
            }
            if use_extractor:
                _mon_modules["extractor"] = list(extractor.parameters())
            _grad_norms = {k: 0.0 for k in _mon_modules}
            _delta_norms = {k: 0.0 for k in _mon_modules}
            _aux_valid_seq_count = 0
            _aux_actual_batch = 0

            if _aux_already_done:
                # WD-order: aux 已在 RL 之前完成，carry forward monitoring data
                _aux_valid_seq_count = _wd_aux_valid_count
                _aux_actual_batch = _wd_aux_batch
                _grad_norms = _wd_aux_grad_norms
                _delta_norms = _wd_aux_delta_norms
                aux_monitor["aux/loss_per_step"] = aux_loss_val / max(args_cli.aux_seq_len - args_cli.aux_burn_in, 1)
                # carry forward 即時 tracking 指標（block A 算的）
                aux_monitor["aux/pos_rel_err"] = _wd_aux_pos["rel"]
                aux_monitor["aux/pos_r2"] = _wd_aux_pos["r2"]
                aux_monitor["aux/pos_valid_frac"] = _wd_aux_pos["vf"]
                # ★移動編碼命脈指標(監控協定第0層 live 版):disp_r2/velocity_r2
                aux_monitor["aux/disp_r2"] = _wd_aux_pos.get("disp_r2", float("nan"))
                aux_monitor["aux/velocity_r2"] = _wd_aux_pos.get("velocity_r2", float("nan"))
                # CPC contrastive 監控:cpc_acc 爬高(>對角 random 1/N)= hidden 編碼位置成功
                if "cpc_acc" in _wd_aux_pos:
                    aux_monitor["aux/cpc_loss"] = _wd_aux_pos["cpc_loss"]
                    aux_monitor["aux/cpc_acc"] = _wd_aux_pos["cpc_acc"]
            elif args_cli.disable_aux_training:
                # Pure-RL resume: keep checkpointed RNN/preprocess representation fixed.
                # Still log feature distribution/VE below to detect any unexpected drift.
                if use_extractor:
                    extractor.eval()
                preprocess_rnn.eval()
                aux_loss_val = 0.0
                aux_monitor["aux/loss_per_step"] = 0.0
            else:
                # --- TBPTT mode: sample contiguous sequences, unroll RNN ---
                _seq_len = args_cli.aux_seq_len
                _burn_in = args_cli.aux_burn_in
                _seq_bs = args_cli.aux_seq_batch_size

                sampled = charge_buf.sample_aux_sequences(_seq_len, _seq_bs, _burn_in)
                if sampled is not None:
                    obs_seq, target_seq, h0, _aux_valid_seq_count = sampled
                    # obs_seq: [B, L, obs_dim], target_seq: [B, L, 7], h0: [1, B, H]
                    B_seq, L_seq = obs_seq.shape[0], obs_seq.shape[1]
                    _aux_actual_batch = B_seq

                    # Normalize obs (no target normalization — WD convention)
                    obs_flat = obs_seq.reshape(B_seq * L_seq, -1)
                    obs_normed = obs_normalizer.normalize(obs_flat)

                    # Features: extractor, legacy 79D, or wd_exact 113D path
                    feat_flat = _charge_features_for_rnn(obs_normed)   # [B*L, D] batch-major
                    # ★BUG FIX:reshape(L,B) 打亂時間/batch → 改 reshape(B,L).permute→[L,B](對齊 target permute)
                    feat_seq = feat_flat.reshape(B_seq, L_seq, -1).permute(1, 0, 2).contiguous()  # [L, B, D]

                    # RNN unroll: sequence mode
                    _, pred_seq, _ = preprocess_rnn(
                        feat_seq, h0, training=True)               # pred_seq: [L, B, 7]

                    # Burn-in: only compute loss on t >= burn_in
                    effective_start = _burn_in
                    effective_len = L_seq - effective_start
                    pred_eff = pred_seq[effective_start:]           # [L_eff, B, 7]
                    tgt_eff = target_seq.permute(1, 0, 2)[effective_start:]  # [L_eff, B, 7]

                    # Compute loss: average over time steps and batch
                    total_loss = torch.tensor(0.0, device=device)
                    n_loss_steps = 0
                    last_display = {}
                    for t_idx in range(effective_len):
                        l_t, disp_t = compute_wd_module_loss(
                            pred_eff[t_idx], tgt_eff[t_idx], weight=_aux_weight,
                            loss_type=args_cli.aux_loss_type, huber_delta=args_cli.aux_huber_delta)
                        total_loss = total_loss + l_t
                        n_loss_steps += 1
                        last_display = disp_t
                    total_loss = total_loss / max(n_loss_steps, 1)

                    # --- 即時 tracking 指標：位置 rel err + R²（避開 aux log-loss 的常數陷阱）---
                    # rel err = mean‖pred−tgt‖/mean‖tgt‖（100%≈猜均值）；R²≤0=沒追蹤（常數），>0=有追蹤。
                    # 注意：訓練端 TBPTT/burn-in 與 decode 連續 rollout 略有差異，但「有無追蹤」結論一致。
                    with torch.no_grad():
                        _pe = pred_eff.reshape(-1, pred_eff.shape[-1])      # [L*B, D]
                        _te = tgt_eff.reshape(-1, tgt_eff.shape[-1])
                        _nd = min(6, _pe.shape[-1])                          # 位置維數（最多 6）
                        _pos_valid = _te[:, 0:2].norm(dim=-1) < 9.0          # 障礙 active（非 FAR_DEFAULT）
                        # 永遠記錄（valid 子集；若無 valid 則全集），另記 valid 佔比診斷
                        _m = _pos_valid if _pos_valid.any() else torch.ones(
                            _te.shape[0], dtype=torch.bool, device=_te.device)
                        _p = _pe[_m][:, 0:_nd]
                        _t = _te[_m][:, 0:_nd]
                        _rel = ((_p - _t).norm(dim=-1).mean()
                                / _t.norm(dim=-1).mean().clamp(min=1e-6)).item()
                        _ss_res = ((_p - _t) ** 2).sum().item()
                        _ss_tot = ((_t - _t.mean(dim=0, keepdim=True)) ** 2).sum().clamp(min=1e-6).item()
                        aux_monitor["aux/pos_rel_err"] = _rel
                        aux_monitor["aux/pos_r2"] = 1.0 - _ss_res / _ss_tot
                        aux_monitor["aux/pos_valid_frac"] = _pos_valid.float().mean().item()
                        # aux/disp_r2: 預測位移(next-t0) vs 真位移 的 R² — 直接量測 RNN 有沒有學到「移動」。
                        #   塌成靜態(pred_next≈pred_t0)→ pred_disp≈0 → disp_r2≈0/負;學到移動 → disp_r2>0。
                        #   pos_r2 高分不出「移動 vs 塌陷」(next≈now 時絕對位置照樣對),故另記此項。
                        #   見 finding_sa4_vaux_low_sr_is_headon / project_sa4_headon_fix_directions。
                        if _nd >= 4:
                            _pd = (_p[:, 2:4] - _p[:, 0:2]).detach()      # 預測位移 next-t0
                            _td = (_t[:, 2:4] - _t[:, 0:2]).detach()      # 真位移 next-t0
                            _dssr = ((_td - _pd) ** 2).sum().item()
                            _dsst = ((_td - _td.mean(dim=0, keepdim=True)) ** 2).sum().clamp(min=1e-6).item()
                            aux_monitor["aux/disp_r2"] = 1.0 - _dssr / _dsst

                    # Backward + step with monitoring
                    _snaps = {k: _snapshot_params(ps) for k, ps in _mon_modules.items()}
                    charge_opt_aux.zero_grad()
                    # WD: reset_model_mentum 只在訓練啟動時一次性 reset（預設 False）
                    # 之前誤解為每步 reset → Adam 退化為無 momentum SGD → RNN 無法收斂
                    # 修正：移除 unconditional momentum reset，保留 Adam 正常累積
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(
                        charge_params_aux,
                        _current_aux_grad_clip,
                    )
                    for k, ps in _mon_modules.items():
                        _grad_norms[k] = _grad_l2_norm(ps)
                    charge_opt_aux.step()
                    for k, ps in _mon_modules.items():
                        _delta_norms[k] = _param_delta_norm(_snaps[k], ps)

                    aux_loss_val = total_loss.item()
                    for k, v in last_display.items():
                        if k != "preprcess_loss":
                            aux_per_feature[f"charge_{k}"] = v
                    aux_monitor["aux/loss_per_step"] = aux_loss_val / max(effective_len, 1)


            # --- Common monitoring (both modes) ---
            for k in _mon_modules:
                aux_monitor[f"aux/{k}_grad_norm"] = _grad_norms[k]
                aux_monitor[f"aux/{k}_param_delta_norm"] = _delta_norms[k]
            aux_monitor["aux/rnn_param_norm"] = _param_l2_norm(_mon_modules["rnn"])
            aux_monitor["aux/predict_head_param_norm"] = _param_l2_norm(_mon_modules["predict_head"])

            for raw_key, readable_key in _AUX_DIM_NAMES.items():
                charge_key = f"charge_{raw_key}"
                if charge_key in aux_per_feature:
                    aux_monitor[readable_key] = aux_per_feature[charge_key]
            aux_monitor["aux/preprocess_loss"] = aux_loss_val
            aux_monitor["aux/training_enabled"] = 0.0 if args_cli.disable_aux_training else 1.0
            aux_monitor["aux/mode"] = 1.0  # always TBPTT
            aux_monitor["aux/seq_len"] = float(args_cli.aux_seq_len)
            aux_monitor["aux/burn_in"] = float(args_cli.aux_burn_in)
            aux_monitor["aux/valid_seq_count"] = float(_aux_valid_seq_count)
            aux_monitor["aux/seq_batch_size_actual"] = float(_aux_actual_batch)

            # RNN/preprocess feature distribution seen by the RL heads this rollout.
            # This diagnoses whether critic spikes are preceded by input-feature drift.
            with torch.no_grad():
                rnn_features = charge_buf.rl_inputs[:RL, :, policy_obs_dim:policy_obs_dim + args_cli.preprocess_dim].reshape(-1, args_cli.preprocess_dim)
                rnn_feature_mean_vec = rnn_features.mean(dim=0)
                aux_monitor["aux/rnn_feature_mean"] = rnn_features.mean().item()
                aux_monitor["aux/rnn_feature_std"] = rnn_features.std().item()
                aux_monitor["aux/rnn_feature_abs_mean"] = rnn_features.abs().mean().item()
                aux_monitor["aux/rnn_feature_norm_mean"] = rnn_features.norm(dim=1).mean().item()
                if _prev_rnn_feature_mean is None:
                    aux_monitor["aux/rnn_feature_delta_norm"] = 0.0
                else:
                    aux_monitor["aux/rnn_feature_delta_norm"] = (
                        rnn_feature_mean_vec - _prev_rnn_feature_mean).norm().item()
                _prev_rnn_feature_mean = rnn_feature_mean_vec.detach().clone()

            # Variance explained on the same target scale used by the critic.
            # With --normalize_return, V(s) predicts normalized returns, so VE
            # must use value_targets rather than raw returns.
            _value_residual = value_targets - charge_buf.values[:RL]
            _var_expl = max(-1.0, 1.0 - (_value_residual.var() / (value_targets.var() + 1e-8)).item())
            aux_monitor["rl_critic/variance_explained"] = _var_expl

            # One-time gradient verification + aux mode info (iteration 0)
            if iteration == 0:
                _rnn_grad = (not args_cli.disable_aux_training) and any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in preprocess_rnn.rnn.parameters())
                _head_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in policy_head.parameters())
                _adapter_grad = (
                    corridor_adapter is not None
                    and any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in corridor_adapter.parameters()
                    )
                )
                _encoder_grad = _e2e_frame_stack and any(
                    p.grad is not None and p.grad.abs().sum() > 0
                    for p in extractor.parameters())
                _ph_lr = charge_opt_aux.param_groups[1]["lr"]
                _fm_lr = charge_opt_aux.param_groups[2]["lr"]
                _ff_lr = charge_opt_aux.param_groups[3]["lr"]
                print(f"[梯度驗證] RL head={'✓' if _head_grad else '✗'} (PPO), "
                      f"corridor_adapter={'✓' if _adapter_grad else ('N/A' if corridor_adapter is None else '✗')} (PPO+BCE), "
                      f"encoder={'✓' if _encoder_grad else ('N/A' if not _e2e_frame_stack else '✗')} (RL), "
                      f"RNN cell={'✓' if _rnn_grad else '✗'} (aux)")
                _ext_info = ""
                if use_extractor:
                    _ext_lr = charge_opt_aux.param_groups[4]["lr"]
                    _ext_info = (f" | extractor aux_lr={_ext_lr} (PPO lr={args_cli.lr})"
                                 if _e2e_frame_stack else
                                 f" | extractor lr={_ext_lr} {'(frozen)' if _ext_lr == 0 else ''}")
                else:
                    _ext_info = " | extractor: N/A (raw_fc_rnn)"
                print(f"  predict_head lr={_ph_lr} | fc_middle lr={_fm_lr} | "
                      f"fc_front lr={_ff_lr} {'(frozen)' if _ff_lr == 0 else ''}"
                      f"{_ext_info}")
                if args_cli.zero_preprocess_feature_for_rl:
                    print("[ABLATION] --zero_preprocess_feature_for_rl ACTIVE: "
                          "rl_in uses zero instead of 12D preprocess feature")
                print(f"[AUX] training={'disabled' if args_cli.disable_aux_training else 'enabled'}, "
                      f"mode=tbptt, seq_len={args_cli.aux_seq_len}, "
                      f"burn_in={args_cli.aux_burn_in}, seq_batch={args_cli.aux_seq_batch_size}")

        # === Obstacle PPO Update（每 train_goal_rate 次訓練 1 次）===
        obs_metrics = None  # None 表示本 iteration 沒有訓練 obstacle
        if train_obstacle:
            obs_policy.train(); obs_value.train()
            # 使用通用 PPO update（continuous action，單 optimizer，無 WD cap）
            obs_metrics = ppo_update_continuous(
                obs_policy, obs_value, obs_buf, obs_optimizer,
                epochs=args_cli.ppo_epochs, mini_batches=args_cli.mini_batches,
                clip_eps=args_cli.clip_eps, vf_coeff=args_cli.vf_coeff,
                ent_coeff=args_cli.obs_ent_coeff, max_grad_norm=args_cli.max_grad_norm,
                gamma=current_gamma, gae_lambda=args_cli.gae_lambda)

        # === Logging：計算本 iteration 的性能指標 ===
        elapsed = time.time() - iter_start
        total_steps = (iteration + 1) * RL     # 累計 env-steps（所有 env 的總 frame 數）
        fps = num_envs * RL / elapsed           # 訓練速率（env-frames/sec）
        wd = metrics.collect()                  # 取得本 iteration 的 WandB metrics dict
        _unsolvable_total = int(getattr(env.unwrapped, "_unsolvable_scene_count_total", 0))
        _unsolvable_rollout = _unsolvable_total - _prev_unsolvable_scene_count
        _prev_unsolvable_scene_count = _unsolvable_total
        wd["scene/unsolvable_scene_count_total"] = float(_unsolvable_total)
        wd["scene/unsolvable_scene_count_rollout"] = float(_unsolvable_rollout)
        wd["scene/unsolvable_scene_rate"] = float(
            _unsolvable_rollout / max(wd.get("charge/total_episodes", 0.0), 1.0)
        )

        # Timeout rate from metrics
        _timeout_rate = wd.get("charge/timeout_rate", 0)

        if (iteration + 1) % args_cli.log_interval == 0 or iteration == 0:
            stage = wd.get("curriculum/stage", 0)
            sr = wd.get("charge/goal_reach_rate", 0)
            cr = wd.get("charge/hit_probability", 0)
            rwd = wd.get("reward/episode_mean", 0)  # 2026-07-03 fix: key 改名後 consumer 沒跟上(恆0)
            goal_v = wd.get("goal_diagnostics/velocity_to_goal_mean", 0)
            goal_d = wd.get("goal_diagnostics/distance_delta_mean", 0)
            goal_h = wd.get("goal_diagnostics/heading_error_abs_mean_deg", 0)
            goal_sw = wd.get("goal_diagnostics/target_switch_rate", 0)
            who = "CHARGE" if train_charge else "OBS"
            obs_tag = f" obs_agent={'ON' if _obs_agent_active else 'OFF'}" if not _obs_agent_active else ""
            # --- Line 1: RL status ---
            if train_charge:
                print(
                    f"[{iteration+1}/{num_iterations}] {who} "
                    f"S{int(stage)} | fps={fps:.0f} | "
                    f"R={rwd:.1f} SR={sr:.1%} CR={cr:.1%} TO={_timeout_rate:.1%} | "
                    f"ppo={charge_ppo_loss:.4f} vf={charge_vf_loss:.4f} "
                    f"ent={charge_entropy:.3f} enc_g={wd_update_monitor.get('rl_encoder/grad_norm_pre_clip', 0.0):.3f} | "
                    f"gV={goal_v:+.3f} gΔ={goal_d:+.2f} h={goal_h:.0f}° "
                    f"sw={goal_sw:.3f}{obs_tag}")
                _replay_env = env.unwrapped
                _mask_template = torch.zeros(
                    num_envs, dtype=torch.bool, device=device
                )
                _previous_mask = getattr(
                    _replay_env,
                    "_previous_stage_replay_active",
                    _mask_template,
                )
                _corridor_mask = getattr(
                    _replay_env, "_long_corridor_active", _mask_template
                )
                _narrow_mask = getattr(
                    _replay_env, "_narrow_bridge_active", _mask_template
                )
                _replay_overlap = (
                    (_previous_mask & _corridor_mask)
                    | (_previous_mask & _narrow_mask)
                    | (_corridor_mask & _narrow_mask)
                )
                if bool(_replay_overlap.any()):
                    raise RuntimeError(
                        "scene replay masks overlap for "
                        f"{int(_replay_overlap.sum().item())} envs"
                    )
                if bool(
                    _previous_mask.any()
                    or _corridor_mask.any()
                    or _narrow_mask.any()
                ):
                    _native_mask = ~(
                        _previous_mask | _corridor_mask | _narrow_mask
                    )
                    print(
                        "  REPLAY-MIX: "
                        f"native={_native_mask.float().mean().item():.1%} "
                        f"sa5_general={_previous_mask.float().mean().item():.1%} "
                        f"narrow={_narrow_mask.float().mean().item():.1%} "
                        f"corridor={_corridor_mask.float().mean().item():.1%} "
                        "overlap=0"
                    )
                if _corridor_adapter_enabled:
                    print(
                        "  CORRIDOR-ADAPTER: "
                        f"gate_bce={wd_update_monitor.get('corridor_adapter/gate_bce', 0.0):.4f} "
                        f"p(corr)={wd_update_monitor.get('corridor_adapter/gate_probability_corridor', 0.0):.3f} "
                        f"p(other)={wd_update_monitor.get('corridor_adapter/gate_probability_other', 0.0):.3f} "
                        f"recall={wd_update_monitor.get('corridor_adapter/gate_recall', 0.0):.1%} "
                        f"specificity={wd_update_monitor.get('corridor_adapter/gate_specificity', 0.0):.1%} "
                        f"residual_l2=({wd_update_monitor.get('corridor_adapter/residual_l2_corridor', 0.0):.3f},"
                        f"{wd_update_monitor.get('corridor_adapter/residual_l2_other', 0.0):.3f})"
                    )
                if _corridor_teacher_distill_enabled:
                    print(
                        "  CORRIDOR-DISTILL: "
                        f"mode={'intervention' if _corridor_teacher_intervention_only else 'all'} "
                        f"selected={wd_update_monitor.get('corridor_distill/active_count', 0.0):.0f}/"
                        f"{wd_update_monitor.get('corridor_distill/candidate_count', 0.0):.0f} "
                        f"intervene={wd_update_monitor.get('corridor_distill/intervention_fraction', 0.0):.1%} "
                        f"steps={wd_update_monitor.get('corridor_distill/optimizer_steps', 0.0):.0f} "
                        f"loss={wd_update_monitor.get('corridor_distill/loss_before', 0.0):.4f}"
                        "->"
                        f"{wd_update_monitor.get('corridor_distill/loss_after', 0.0):.4f} "
                        f"agree_joint={wd_update_monitor.get('corridor_distill/agreement_before_joint', 0.0):.1%}"
                        "->"
                        f"{wd_update_monitor.get('corridor_distill/agreement_after_joint', 0.0):.1%} "
                        f"within1_joint={wd_update_monitor.get('corridor_distill/within_one_before_joint', 0.0):.1%}"
                        "->"
                        f"{wd_update_monitor.get('corridor_distill/within_one_after_joint', 0.0):.1%}"
                    )
                if _teacher_retention_enabled:
                    print(
                        "  RETENTION: "
                        f"active={wd_update_monitor.get('retention/active_fraction', 0.0):.1%} "
                        f"teacher_forced={wd_update_monitor.get('retention/teacher_forced_fraction', 0.0):.1%} "
                        f"beta={_teacher_retention_weight:g} "
                        f"margin_w={_teacher_retention_margin_weight:g} "
                        f"ce_w={_teacher_retention_action_ce_weight:g} "
                        f"KL=({wd_update_monitor.get('retention/kl_linear', 0.0):.4f},"
                        f"{wd_update_monitor.get('retention/kl_angular', 0.0):.4f}) "
                        f"margin=({wd_update_monitor.get('retention/margin_linear', 0.0):.4f},"
                        f"{wd_update_monitor.get('retention/margin_angular', 0.0):.4f}) "
                        f"CE=({wd_update_monitor.get('retention/action_ce_linear', 0.0):.4f},"
                        f"{wd_update_monitor.get('retention/action_ce_angular', 0.0):.4f}) "
                        f"agree=({wd_update_monitor.get('retention/action_agreement_linear', 0.0):.1%},"
                        f"{wd_update_monitor.get('retention/action_agreement_angular', 0.0):.1%}) "
                        f"post_agree=({wd_update_monitor.get('retention/post_action_agreement_linear', 0.0):.1%},"
                        f"{wd_update_monitor.get('retention/post_action_agreement_angular', 0.0):.1%}) "
                        f"weighted_loss={wd_update_monitor.get('retention/weighted_loss', 0.0):.5f}"
                    )
                    if _teacher_retention_post_kl_epochs > 0:
                        print(
                            "  RETENTION-PROJECT: "
                            f"epochs={_teacher_retention_post_kl_epochs} "
                            f"steps={wd_update_monitor.get('retention/post_kl_projection_optimizer_steps', 0.0):.0f} "
                            f"KL=({wd_update_monitor.get('retention/post_kl_projection_before_linear', 0.0):.4f},"
                            f"{wd_update_monitor.get('retention/post_kl_projection_before_angular', 0.0):.4f})"
                            "->"
                            f"({wd_update_monitor.get('retention/post_kl_projection_after_linear', 0.0):.4f},"
                            f"{wd_update_monitor.get('retention/post_kl_projection_after_angular', 0.0):.4f}) "
                            f"margin=({wd_update_monitor.get('retention/post_kl_projection_margin_before_linear', 0.0):.4f},"
                            f"{wd_update_monitor.get('retention/post_kl_projection_margin_before_angular', 0.0):.4f})"
                            "->"
                            f"({wd_update_monitor.get('retention/post_kl_projection_margin_after_linear', 0.0):.4f},"
                            f"{wd_update_monitor.get('retention/post_kl_projection_margin_after_angular', 0.0):.4f}) "
                            f"agree=({wd_update_monitor.get('retention/post_kl_projection_agree_after_linear', 0.0):.1%},"
                            f"{wd_update_monitor.get('retention/post_kl_projection_agree_after_angular', 0.0):.1%})"
                        )
                        if _retention_anchor_projection_stats is not None:
                            print(
                                "  RETENTION-ANCHOR: "
                                f"scope={_previous_stage_teacher_scope} "
                                f"weight={_teacher_retention_post_anchor_weight:g} "
                                f"active={wd_update_monitor.get('retention/post_anchor_active_count', 0.0):.0f} "
                                f"KL=({wd_update_monitor.get('retention/post_anchor_before_linear', 0.0):.4f},"
                                f"{wd_update_monitor.get('retention/post_anchor_before_angular', 0.0):.4f})"
                                "->"
                                f"({wd_update_monitor.get('retention/post_anchor_after_linear', 0.0):.4f},"
                                f"{wd_update_monitor.get('retention/post_anchor_after_angular', 0.0):.4f}) "
                                f"agree=({wd_update_monitor.get('retention/post_anchor_agree_after_linear', 0.0):.1%},"
                                f"{wd_update_monitor.get('retention/post_anchor_agree_after_angular', 0.0):.1%})"
                            )
                if _previous_stage_teacher_enabled:
                    print(
                        "  PREVIOUS-RETENTION: "
                        f"active={wd_update_monitor.get('previous_retention/active_fraction', 0.0):.1%} "
                        f"beta={_previous_stage_teacher_retention_weight:g} "
                        f"KL=({wd_update_monitor.get('previous_retention/kl_linear', 0.0):.4f},"
                        f"{wd_update_monitor.get('previous_retention/kl_angular', 0.0):.4f}) "
                        f"agree=({wd_update_monitor.get('previous_retention/action_agreement_linear', 0.0):.1%},"
                        f"{wd_update_monitor.get('previous_retention/action_agreement_angular', 0.0):.1%}) "
                        f"weighted_loss={wd_update_monitor.get('previous_retention/weighted_loss', 0.0):.5f}"
                    )
            else:
                _obs_ent = obs_metrics["entropy"] if obs_metrics else 0.0
                _obs_pl = obs_metrics["policy_loss"] if obs_metrics else 0.0
                print(
                    f"[{iteration+1}/{num_iterations}] {who} "
                    f"S{int(stage)} | fps={fps:.0f} | "
                    f"R={rwd:.1f} SR={sr:.1%} CR={cr:.1%} TO={_timeout_rate:.1%} | "
                    f"obs_ppo={_obs_pl:.4f} obs_ent={_obs_ent:.3f} | "
                    f"gV={goal_v:+.3f} gΔ={goal_d:+.2f} h={goal_h:.0f}° "
                    f"sw={goal_sw:.3f}{obs_tag}")
            # --- Line 2: AUX status (only when charge trained) ---
            if train_charge:
                _rnn_gn = aux_monitor.get("aux/rnn_grad_norm", 0)
                _rnn_dn = aux_monitor.get("aux/rnn_param_delta_norm", 0)
                _ph_dn = aux_monitor.get("aux/predict_head_param_delta_norm", 0)
                _fm_dn = aux_monitor.get("aux/fc_middle_param_delta_norm", 0)
                _n1d = aux_monitor.get("aux/near1_d_loss", 0)
                _n2d = aux_monitor.get("aux/near2_d_loss", 0)
                _vsc = int(aux_monitor.get("aux/valid_seq_count", 0))
                _asl = int(aux_monitor.get("aux/seq_len", 1))
                _ve = aux_monitor.get("rl_critic/variance_explained", 0)
                _aux_label = "disabled" if args_cli.disable_aux_training else "tbptt"
                if args_cli.disable_aux_training:
                    print(f"  AUX(disabled): no loss/no updates | VE={_ve:.3f}")
                else:
                    print(
                        f"  AUX({_aux_label} L={_asl}): "
                        f"loss={aux_loss_val:.4f} "
                        f"n1d={_n1d:.3f} n2d={_n2d:.3f} "
                        f"valid={_vsc} | "
                        f"rnn_grad={_rnn_gn:.4f} rnn_delta={_rnn_dn:.6f} "
                        f"ph_delta={_ph_dn:.6f} fm_delta={_fm_dn:.6f} | "
                        f"VE={_ve:.3f}")

        if wandb_run is not None:
            # ----------------------------------------------------------------
            # WandB Logging 策略：
            #
            # rl/* 只在 charge 訓練的 iteration 寫入（避免 OBS iteration 寫 0 造成假性下降）。
            # 問題：如果每次 iteration 都寫 rl/entropy，OBS iteration 時 charge_entropy=0，
            # WandB 圖表會出現 0 ↔ ~5.8 的規律振盪，看起來像 entropy collapse。
            #
            # 修正：rl/policy_loss, rl/value_loss, rl/entropy 只在 train_charge=True 時寫入。
            # 同樣，obstacle/* 只在 train_obstacle=True 時寫入。
            # rl/success_rate, rl/collision_rate 等 episode-level 指標每次都寫（無論誰訓練）。
            # ----------------------------------------------------------------

            log_data = {}

            # --- rl/* = canonical RL metrics (single source of truth) ---
            log_data.update({
                "rl/return_mean": wd.get("reward/episode_mean", 0),  # 2026-07-03 fix: key 改名(恆0)
                "rl/success_rate": wd.get("charge/goal_reach_rate", 0),
                "rl/collision_rate": wd.get("charge/hit_probability", 0),
                "rl/timeout_rate": _timeout_rate,
                "rl/stage_idx": float(wd.get("curriculum/stage", 0)),
            })
            if train_charge:
                log_data.update({
                    "rl/policy_loss": charge_ppo_loss,
                    "rl/value_loss": charge_vf_loss,
                    "rl/entropy": charge_entropy,
                })

            # --- obstacle/* = obstacle agent metrics ---
            if obs_metrics is not None:
                log_data.update({
                    "obstacle/policy_loss": obs_metrics["policy_loss"],
                    "obstacle/value_loss": obs_metrics["value_loss"],
                    "obstacle/entropy": obs_metrics["entropy"],
                    "obstacle/total_loss": obs_metrics["total_loss"],
                })

            # --- aux/* + wd_update_actor/* + wd_update_critic/* + wd_update/* = training diagnostics ---
            if train_charge:
                log_data.update(aux_monitor)
                log_data.update(wd_update_monitor)

            # --- train/* = session info ---
            log_data.update({
                "train/fps": fps,
                "train/gamma": current_gamma,
                "train/active_agent": "charge" if train_charge else "obstacle",
                "train/obs_agent_active": float(_obs_agent_active),
                "train/zero_preprocess_for_rl": float(args_cli.zero_preprocess_feature_for_rl),
                # Runtime entropy coefficients (may differ from curriculum config)
                "train/ent_coeff_linear": _ent_coeff_linear,
                "train/ent_coeff_angular": _ent_coeff_angular,
            })

            # --- phase_parameter/* = 當前 phase 的訓練進度 + 實際生效超參數 ---
            # 目的：讓 W&B 的 phase_parameter 分組直接看到「這一個 phase / iteration
            # 實際用到的訓練參數」，不只看 CLI 初始 config。
            # 注意：有些參數會被 curriculum runtime sync 改掉（例如 entropy、reward、lr、clip caps），
            # 因此這裡記錄 runtime effective value，而不是只記 args_cli。
            _phase_param_log = {
                # Progress / budget
                "phase_parameter/stage": float(wd.get("curriculum/stage", metrics._curriculum_info.get("stage", 0))),
                "phase_parameter/initial_stage": float(args_cli.initial_stage),
                "phase_parameter/fixed_stage": float(args_cli.fixed_stage),
                "phase_parameter/timesteps": float(total_steps),  # legacy alias: W&B step axis used by old runs
                "phase_parameter/timesteps_current": float(total_steps),
                "phase_parameter/timesteps_target": float(total_timesteps),
                "phase_parameter/updates": float(iteration + 1),  # legacy alias
                "phase_parameter/updates_current": float(iteration + 1),
                "phase_parameter/updates_target": float(num_iterations),
                "phase_parameter/interactions": float(total_steps * num_envs),  # true env interactions = E × T × updates
                "phase_parameter/interactions_current": float(total_steps * num_envs),
                "phase_parameter/interactions_per_update": float(batch_size),
                "phase_parameter/interactions_target": float(batch_size * num_iterations),
                "phase_parameter/batch_size": float(batch_size),
                "phase_parameter/mini_batch_size": float(mini_batch_size),
                "phase_parameter/rollout_length": float(RL),
                "phase_parameter/num_envs": float(num_envs),
                "phase_parameter/ppo_epochs": float(args_cli.ppo_epochs),
                "phase_parameter/mini_batches": float(args_cli.mini_batches),
                "phase_parameter/train_goal_rate": float(args_cli.train_goal_rate),
                "phase_parameter/teacher_retention_weight": _teacher_retention_weight,
                "phase_parameter/teacher_retention_margin_weight": _teacher_retention_margin_weight,
                "phase_parameter/teacher_retention_action_ce_weight": _teacher_retention_action_ce_weight,
                "phase_parameter/teacher_retention_argmax_margin": _teacher_retention_argmax_margin,
                "phase_parameter/teacher_retention_post_kl_epochs": float(
                    _teacher_retention_post_kl_epochs
                ),
                "phase_parameter/teacher_retention_post_kl_lr": (
                    _teacher_retention_post_kl_lr
                ),
                "phase_parameter/teacher_retention_post_kl_batch_size": float(
                    _teacher_retention_post_kl_batch_size
                ),
                "phase_parameter/teacher_retention_post_kl_max_grad_norm": (
                    _teacher_retention_post_kl_max_grad_norm
                ),
                "phase_parameter/teacher_retention_rollout_override": float(
                    _teacher_retention_rollout_override
                ),
                "phase_parameter/previous_stage_teacher_retention_weight": (
                    _previous_stage_teacher_retention_weight
                ),
                "phase_parameter/corridor_teacher_distill_epochs": float(
                    _corridor_teacher_distill_epochs
                ),
                "phase_parameter/corridor_teacher_distill_lr": (
                    _corridor_teacher_distill_lr
                ),
                "phase_parameter/corridor_teacher_distill_batch_size": float(
                    _corridor_teacher_distill_batch_size
                ),
                "phase_parameter/corridor_teacher_distill_max_grad_norm": (
                    _corridor_teacher_distill_max_grad_norm
                ),
                "phase_parameter/corridor_teacher_distill_neighbor_mass": (
                    _corridor_teacher_distill_neighbor_mass
                ),
                "phase_parameter/corridor_teacher_distill_stride": float(
                    _corridor_teacher_distill_stride
                ),
                "phase_parameter/corridor_teacher_intervention_only": float(
                    _corridor_teacher_intervention_only
                ),
                "phase_parameter/corridor_teacher_intervention_clearance_m": (
                    _corridor_teacher_intervention_clearance_m
                ),
                "phase_parameter/corridor_adapter_enabled": float(
                    _corridor_adapter_enabled
                ),
                "phase_parameter/corridor_adapter_hidden_dim": float(
                    _corridor_adapter_hidden_dim
                ),
                "phase_parameter/corridor_adapter_gate_loss_weight": (
                    _corridor_adapter_gate_loss_weight
                ),
                "phase_parameter/corridor_adapter_max_logit_delta": (
                    _corridor_adapter_max_logit_delta
                ),
                "phase_parameter/corridor_adapter_freeze_base": float(
                    _corridor_adapter_freeze_base
                ),
                "phase_parameter/corridor_adapter_residual_policy_features":
                    float(
                        _corridor_adapter_residual_features
                        == "policy_features"
                    ),
                "phase_parameter/critic_detach_encoder": float(
                    args_cli.critic_detach_encoder
                ),
                # RL / optimizer effective hyperparameters
                "phase_parameter/gamma": float(current_gamma),
                "phase_parameter/gae_lambda": float(args_cli.gae_lambda),
                "phase_parameter/use_a2c": float(args_cli.use_a2c),
                "phase_parameter/clip_eps": float(args_cli.clip_eps),
                "phase_parameter/lr": float(_current_rl_lr),
                "phase_parameter/rnn_lr": float(_current_rnn_lr),
                "phase_parameter/vf_coeff": float(_current_vf_coeff),
                "phase_parameter/normalize_return": float(args_cli.normalize_return),
                "phase_parameter/max_grad_norm": float(_current_max_grad_norm),
                "phase_parameter/aux_grad_clip": float(_current_aux_grad_clip),
                "phase_parameter/wd_update_clip": float(args_cli.wd_update_clip and not args_cli.wd_update_monitor_only),
                "phase_parameter/wd_actor_update_clip": float(_current_wd_actor_update_clip),
                "phase_parameter/wd_critic_update_clip": float(_current_wd_critic_update_clip),
                "phase_parameter/policy_loss_clamp": float(args_cli.policy_loss_clamp),
                "phase_parameter/vf_term_clamp": float(args_cli.vf_term_clamp),
                "phase_parameter/ent_coeff": float(args_cli.ent_coeff),
                "phase_parameter/ent_coeff_linear": float(_ent_coeff_linear),
                "phase_parameter/ent_coeff_angular": float(_ent_coeff_angular),
                "phase_parameter/cli_ent_coeff_linear": float(args_cli.ent_coeff_linear),
                "phase_parameter/cli_ent_coeff_angular": float(args_cli.ent_coeff_angular),
                # Aux / RNN module
                "phase_parameter/disable_aux_training": float(args_cli.disable_aux_training),
                "phase_parameter/aux_seq_len": float(args_cli.aux_seq_len),
                "phase_parameter/aux_seq_batch_size": float(args_cli.aux_seq_batch_size),
                "phase_parameter/aux_burn_in": float(args_cli.aux_burn_in),
                "phase_parameter/aux_lr": float(args_cli.aux_lr),
                "phase_parameter/aux_lr_predict_head": float(_lr_predict_head),
                "phase_parameter/aux_lr_fc_middle": float(_lr_fc_middle),
                "phase_parameter/aux_lr_fc_front": float(_lr_fc_front),
                "phase_parameter/aux_lr_extractor": float(_lr_extractor if use_extractor else 0.0),
                "phase_parameter/tbptt_len": float(args_cli.tbptt_len),
                # Model / action-space
                "phase_parameter/hidden_dim": float(args_cli.hidden_dim),
                "phase_parameter/preprocess_dim": float(args_cli.preprocess_dim),
                "phase_parameter/fc_dim": float(args_cli.fc_dim),
                "phase_parameter/policy_obs_dim": float(policy_obs_dim),
                "phase_parameter/rl_input_dim": float(rl_input_dim),
                "phase_parameter/action_linear_bins": 19.0,
                "phase_parameter/action_angular_bins": 19.0,
                # Reward / scene / obstacle effective phase params
                "phase_parameter/reward_get_goal": float(_spot_reward_get_goal),
                "phase_parameter/penalty_hit": float(_spot_penalty_hit),
                "phase_parameter/penalty_timeout": float(_spot_penalty_timeout),
                "phase_parameter/penalty_smoothness": float(_spot_penalty_smoothness),  # v3
                "phase_parameter/penalty_speed_near_obs": float(_spot_penalty_speed_near_obs),  # v3f-react
                "phase_parameter/anti_spin_weight": float(args_cli.anti_spin_weight),
                "phase_parameter/anti_spin_hazard_distance": float(args_cli.anti_spin_hazard_distance),
                "phase_parameter/anti_spin_omega_threshold": float(args_cli.anti_spin_omega_threshold),
                "phase_parameter/anti_spin_progress_threshold": float(args_cli.anti_spin_progress_threshold),
                "phase_parameter/anti_spin_grace_steps": float(args_cli.anti_spin_grace_steps),
                "phase_parameter/anti_spin_ramp_steps": float(args_cli.anti_spin_ramp_steps),
                "phase_parameter/anti_spin_yaw_grace_deg": float(args_cli.anti_spin_yaw_grace_deg),
                "phase_parameter/anti_spin_yaw_ramp_deg": float(args_cli.anti_spin_yaw_ramp_deg),
                "phase_parameter/anti_spin_dt": float(args_cli.anti_spin_dt),
                "phase_parameter/future_occupancy_weight": float(args_cli.future_occupancy_weight),
                "phase_parameter/future_occupancy_horizon_s": float(args_cli.future_occupancy_horizon_s),
                "phase_parameter/future_occupancy_samples": float(args_cli.future_occupancy_samples),
                "phase_parameter/future_occupancy_safe_distance_m": float(args_cli.future_occupancy_safe_distance_m),
                "phase_parameter/future_occupancy_near_distance_m": float(args_cli.future_occupancy_near_distance_m),
                "phase_parameter/future_occupancy_move_threshold_mps": float(args_cli.future_occupancy_move_threshold_mps),
                "phase_parameter/cost_operate": float(_spot_cost_operate),
                "phase_parameter/obstacle_speed_rate": float(_obs_speed_limit),
                "phase_parameter/obs_size_rand": float(_obs_size_rand),
                "phase_parameter/obs_collision_base": float(args_cli.obs_collision_base),
                "phase_parameter/scene_bound_rand": float(_scene_bound_rand),
                "phase_parameter/scene_bound_base": float(args_cli.scene_bound_base),
                "phase_parameter/obs_lr": float(args_cli.obs_lr),
                "phase_parameter/obs_ent_coeff": float(args_cli.obs_ent_coeff),
            }
            # Curriculum registry 裡的 numeric/bool phase fields 也全部攤平成 W&B scalar。
            # 這會捕捉 num_goals / obstacle counts / walls / trainer_* 等後續新增欄位。
            for _k, _v in metrics._curriculum_info.items():
                _wb_key = f"phase_parameter/curriculum/{_k}"
                if isinstance(_v, bool):
                    _phase_param_log[_wb_key] = float(_v)
                elif isinstance(_v, (int, float)):
                    _phase_param_log[_wb_key] = float(_v)
                elif isinstance(_v, (list, tuple)) and len(_v) == 2 and all(isinstance(x, (int, float)) for x in _v):
                    _phase_param_log[f"{_wb_key}_min"] = float(_v[0])
                    _phase_param_log[f"{_wb_key}_max"] = float(_v[1])
            log_data.update(_phase_param_log)

            # --- behavior/* from BehaviorScheduler (rule_based mode) ---
            if _obstacle_mode == "rule_based" and hasattr(env.unwrapped, '_behavior_scheduler'):
                bsched = env.unwrapped._behavior_scheduler
                bm = bsched.get_metrics()
                log_data.update(bm)

            # --- charge/* + goal_diagnostics/* + curriculum/* from MetricsCollector ---
            log_data.update(wd)

            # --- Fixed previous-stage scene replay audit ---
            _raw_env = env.unwrapped
            if hasattr(_raw_env, "_previous_stage_replay_reset_count"):
                _ps_resets = max(
                    int(_raw_env._previous_stage_replay_reset_count), 1
                )
                _ps_active = _raw_env._previous_stage_replay_active
                log_data.update({
                    "previous_stage_replay/injected_fraction_actual":
                        float(_raw_env._previous_stage_replay_injected_count)
                        / _ps_resets,
                    "previous_stage_replay/active_fraction":
                        float(_ps_active.float().mean().item()),
                    "previous_stage_replay/injected_episodes":
                        float(_raw_env._previous_stage_replay_injected_count),
                    "previous_stage_replay/installed_episodes":
                        float(_raw_env._previous_stage_replay_installed_count),
                    "previous_stage_replay/pending_envs":
                        float(
                            _raw_env._previous_stage_replay_pending.sum().item()
                        ),
                })

            # --- SA5 narrow-passage bridge reset/schedule audit ---
            if hasattr(_raw_env, "_narrow_bridge_reset_count"):
                _nb_resets = max(int(_raw_env._narrow_bridge_reset_count), 1)
                log_data.update({
                    "narrow_bridge/injected_fraction_actual":
                        float(_raw_env._narrow_bridge_injected_count) / _nb_resets,
                    "narrow_bridge/injected_episodes":
                        float(_raw_env._narrow_bridge_injected_count),
                    "narrow_bridge/unsolvable_count":
                        float(_raw_env._narrow_bridge_unsolvable_count),
                    "narrow_bridge/schedule_progress":
                        float(getattr(_raw_env, "_narrow_bridge_last_progress", 0.0)),
                    "narrow_bridge/gap_min_m":
                        float(getattr(_raw_env, "_narrow_bridge_last_width_min", 0.0)),
                    "narrow_bridge/gap_max_m":
                        float(getattr(_raw_env, "_narrow_bridge_last_width_max", 0.0)),
                    "narrow_bridge/yaw_limit_deg":
                        float(getattr(_raw_env, "_narrow_bridge_last_yaw_limit_deg", 0.0)),
                    "narrow_bridge/stress_ratio":
                        float(getattr(_raw_env, "_narrow_bridge_last_stress_ratio", 0.0)),
                    "narrow_bridge/goal_distance_min_m":
                        float(getattr(
                            _raw_env, "_narrow_bridge_last_goal_distance_min", 0.0
                        )),
                    "narrow_bridge/goal_distance_max_m":
                        float(getattr(
                            _raw_env, "_narrow_bridge_last_goal_distance_max", 0.0
                        )),
                    "narrow_bridge/goal_lateral_offset_min_m":
                        float(getattr(
                            _raw_env, "_narrow_bridge_last_goal_offset_min", 0.0
                        )),
                    "narrow_bridge/goal_lateral_offset_max_m":
                        float(getattr(
                            _raw_env, "_narrow_bridge_last_goal_offset_max", 0.0
                        )),
                })

            # --- Deployment-corridor motion-family cumulative audit ---
            if hasattr(_raw_env, "_long_corridor_motion_env_counts_total"):
                _mc = _raw_env._long_corridor_motion_env_counts_total
                _sc = _raw_env._long_corridor_motion_slot_counts_total
                _assign_total = int(_mc.sum().item())
                if _assign_total > 0:
                    _mf = (_mc.double() / _assign_total).tolist()
                    log_data.update({
                        "long_corridor/motion_lateral_fraction_actual": float(_mf[0]),
                        "long_corridor/motion_longitudinal_fraction_actual": float(_mf[1]),
                        "long_corridor/motion_random_2d_fraction_actual": float(_mf[2]),
                        "long_corridor/motion_assignments_total": _assign_total,
                        "long_corridor/motion_slot_assignments_total":
                            int(_sc.sum().item()),
                        "long_corridor/pure_env_fraction_actual":
                            float(_raw_env._long_corridor_pure_env_count_total)
                            / _assign_total,
                    })

            # --- Deployment-corridor goal ownership audit ---
            if hasattr(_raw_env, "_long_corridor_reset_count"):
                _lc_resets = max(int(_raw_env._long_corridor_reset_count), 1)
                log_data.update({
                    "long_corridor/injected_fraction_actual":
                        float(_raw_env._long_corridor_injected_count) / _lc_resets,
                    "long_corridor/active_fraction":
                        float(_raw_env._long_corridor_active.float().mean().item()),
                    "long_corridor/goal_command_max_error_m":
                        _long_corridor_goal_error_max,
                    "long_corridor/local_goal_max_error_m":
                        _long_corridor_local_goal_error_max,
                    "long_corridor/goal_audit_frames":
                        float(_long_corridor_goal_audit_frames),
                    "long_corridor/unsolvable_count":
                        float(_raw_env._long_corridor_unsolvable_count),
                })

            if args_cli.action_table_sample_size > 0:
                action_rows = metrics.action_speed_accel_rows()
                if action_rows:
                    log_data["charge/action_speed_accel_table"] = wandb.Table(
                        columns=[
                            "rollout_step",
                            "v_x",
                            "a_x",
                            "omega",
                            "linear_idx",
                            "angular_idx",
                            "heading_error_deg",
                            "velocity_to_goal",
                            "terminal_code",
                        ],
                        data=action_rows,
                    )

            # --- LV-DOT encoder metrics (2026-07-12): 監控 encoder 使用程度 ---
            if _lvdot_enc_on and lvdot_encoder is not None:
                with torch.no_grad():
                    _gate = lvdot_encoder.out_scale.detach()
                    log_data["lvdot/gate_abs_mean"] = float(_gate.abs().mean())   # gate 開啟程度(0=沒用,↑=用)
                    log_data["lvdot/gate_abs_max"] = float(_gate.abs().max())
                    _lv_raw = charge_buf.rl_inputs[:RL, :, policy_obs_dim - 30:policy_obs_dim].reshape(-1, 30)
                    if _lv_raw.shape[0] > 0:
                        _enc_out = lvdot_encoder(_lv_raw[:512])
                        log_data["lvdot/encoder_output_norm"] = float(_enc_out.norm(dim=-1).mean())

            # --- DORAEMON metrics ---
            if _doraemon_ctrl is not None:
                log_data.update(_doraemon_ctrl.to_wandb_dict())

            # --- RGDR metrics ---
            if _rgdr_enabled:
                _rgdr_floor = _rgdr_env_returns.min()
                _difficulty = 1.0 / (_rgdr_env_returns - _rgdr_floor + 1e-6)
                _ew = (_difficulty / (_difficulty.mean() + 1e-8)).clamp(_rgdr_clamp[0], _rgdr_clamp[1])
                log_data["rgdr/env_weight_min"] = float(_ew.min())
                log_data["rgdr/env_weight_max"] = float(_ew.max())
                log_data["rgdr/env_weight_std"] = float(_ew.std())
                log_data["rgdr/env_returns_mean"] = float(_rgdr_env_returns.mean())
                log_data["rgdr/env_returns_std"] = float(_rgdr_env_returns.std())

            # --- r_arc metrics（啟用時）---
            if _arc_action_term is not None and _arc_stat_n > 0:
                log_data["r_arc/mean"] = _arc_stat_sum / _arc_stat_n          # 每步平均 r_arc（負值）
                log_data["r_arc/active_fraction"] = _arc_stat_fire / _arc_stat_n  # 每步有扣分的 env 比例
                if _arc_ep_n > 0:
                    log_data["r_arc/ep_penalty_mean"] = _arc_ep_sum / _arc_ep_n   # 每集累積 arc penalty 平均

            # --- 真實 LR：直接讀 optimizer param_group（非 CONFIG.lr）。
            # lr_decay / phase override / resume 蓋回 都以這裡為準（W2 protocol）。
            log_data["train/lr_rl_actual"] = float(charge_opt_rl.param_groups[0]["lr"])
            log_data["train/lr_aux_rnn_actual"] = float(charge_opt_aux.param_groups[0]["lr"])

            # --- N1 直穿模仿：CE、動作一致率與 λ 校準用的梯度比 ---
            if _narrow_imitation_stats is not None:
                for _ni_key, _ni_val in _narrow_imitation_stats.items():
                    log_data[f"narrow_imitation/{_ni_key}"] = _ni_val

            wandb_run.log(log_data, step=total_steps)

        # Local, structured metrics let the cron supervisor make convergence
        # decisions without depending on W&B/network availability.
        _supervisor_metrics = {
            "iteration": iteration + 1,
            "iterations_target": num_iterations,
            "total_steps": total_steps,
            "stage": int(wd.get("curriculum/stage", 0)),
            "sr": float(wd.get("charge/goal_reach_rate", 0.0)),
            "cr": float(wd.get("charge/hit_probability", 0.0)),
            "timeout": float(_timeout_rate),
            "lr_rl_actual": float(charge_opt_rl.param_groups[0]["lr"]),
            "narrow_imitation": _narrow_imitation_stats,
            "narrow_imitation_shadow": _narrow_imitation_shadow,
            "ppo_update_count": float(
                wd_update_monitor.get("rl/ppo_update_count", 0.0)
            ),
            "actor_param_delta_norm": float(
                wd_update_monitor.get(
                    "wd_update_actor/param_delta_norm", 0.0
                )
            ),
            "critic_param_delta_norm": float(
                wd_update_monitor.get(
                    "wd_update_critic/param_delta_norm", 0.0
                )
            ),
            "policy_loss": float(charge_ppo_loss),
            "vf": float(charge_vf_loss),
            "entropy": float(charge_entropy),
            "kl": float(wd_update_monitor.get("rl/approx_kl", 0.0)),
            "clip_fraction": float(wd_update_monitor.get("rl/clip_fraction", 0.0)),
            "encoder_grad": float(wd_update_monitor.get("rl_encoder/grad_norm_pre_clip", 0.0)),
            "corridor_adapter_gate_recall": float(
                wd_update_monitor.get(
                    "corridor_adapter/gate_recall", 0.0
                )
            ),
            "corridor_adapter_gate_specificity": float(
                wd_update_monitor.get(
                    "corridor_adapter/gate_specificity", 0.0
                )
            ),
            "corridor_adapter_residual_l2_corridor": float(
                wd_update_monitor.get(
                    "corridor_adapter/residual_l2_corridor", 0.0
                )
            ),
            "corridor_adapter_residual_l2_other": float(
                wd_update_monitor.get(
                    "corridor_adapter/residual_l2_other", 0.0
                )
            ),
            "retention_post_agreement_linear": float(
                wd_update_monitor.get(
                    "retention/post_action_agreement_linear", 0.0
                )
            ),
            "retention_post_agreement_angular": float(
                wd_update_monitor.get(
                    "retention/post_action_agreement_angular", 0.0
                )
            ),
            "retention_projection_kl_after_linear": float(
                wd_update_monitor.get(
                    "retention/post_kl_projection_after_linear", 0.0
                )
            ),
            "retention_projection_kl_after_angular": float(
                wd_update_monitor.get(
                    "retention/post_kl_projection_after_angular", 0.0
                )
            ),
            "retention_projection_agreement_linear": float(
                wd_update_monitor.get(
                    "retention/post_kl_projection_agree_after_linear", 0.0
                )
            ),
            "retention_projection_agreement_angular": float(
                wd_update_monitor.get(
                    "retention/post_kl_projection_agree_after_angular", 0.0
                )
            ),
            "retention_projection_margin_after_linear": float(
                wd_update_monitor.get(
                    "retention/post_kl_projection_margin_after_linear", 0.0
                )
            ),
            "retention_projection_margin_after_angular": float(
                wd_update_monitor.get(
                    "retention/post_kl_projection_margin_after_angular", 0.0
                )
            ),
            "fps": float(fps),
        }
        with open(_supervisor_metrics_file, "a", encoding="utf-8") as _metrics_fp:
            _metrics_fp.write(json.dumps(_supervisor_metrics, separators=(",", ":")) + "\n")

        # === Save checkpoint：定期儲存模型、optimizer state、normalizer 統計 ===
        _supervisor_stop_requested = os.path.isfile(_supervisor_stop_file)
        if ((iteration + 1) % args_cli.save_interval == 0
                or iteration == num_iterations - 1
                or _supervisor_stop_requested):
            ckpt_path = os.path.join(log_dir, f"checkpoint_{total_steps}.pt")
            _ckpt_dict = {
                "preprocess_rnn": preprocess_rnn.state_dict(),   # RNN cell + fc_front/middle + predict_head
                "policy_head": policy_head.state_dict(),         # actor head
                "value_head": value_head.state_dict(),           # critic head
                "obs_policy": obs_policy.state_dict() if obs_policy else {},  # 障礙物 policy（learned mode）
                "obs_value": obs_value.state_dict() if obs_value else {},     # 障礙物 critic
                "charge_opt_rl": charge_opt_rl.state_dict(),     # RL optimizer state（Adam momentum）
                "charge_opt_aux": charge_opt_aux.state_dict(),   # Aux optimizer state
                "obs_optimizer": obs_optimizer.state_dict() if obs_optimizer else {},
                "obs_normalizer": {
                    "mean": obs_normalizer.mean,   # running mean（139D）
                    "var": obs_normalizer.var,     # running variance
                    "count": obs_normalizer.count, # sample count
                },
                "iteration": iteration,            # 已完成的 iteration 數（用於 resume）
                "total_steps": total_steps,        # 累計 env-steps
                "args": vars(args_cli),            # 完整 CLI args（用於實驗重現）
            }
            if use_extractor:
                _ckpt_dict["extractor"] = extractor.state_dict()  # Conv1d extractor（only in extractor_rnn mode）
            if corridor_adapter is not None:
                _ckpt_dict["corridor_adapter"] = corridor_adapter.state_dict()
            if _lvdot_enc_on and lvdot_encoder is not None:
                _ckpt_dict["lvdot_encoder"] = lvdot_encoder.state_dict()  # ★LV-DOT channel encoder
            if args_cli.feat_norm:
                # ★部署關鍵:feat_norm 統計必須存進 checkpoint,否則 play/車端餵錯尺度特徵→hidden 錯亂
                _ckpt_dict["feat_normalizer"] = {
                    "mean": feat_normalizer.mean,   # extractor 輸出 per-dim running mean
                    "var": feat_normalizer.var,     # per-dim running variance
                    "count": feat_normalizer.count,
                }
            torch.save(_ckpt_dict, ckpt_path)
            print(f"[SAVE] {ckpt_path}")

        if _supervisor_stop_requested:
            print(
                f"[SUPERVISOR] cooperative stop accepted after iteration {iteration + 1}; "
                f"checkpoint={ckpt_path}"
            )
            try:
                os.remove(_supervisor_stop_file)
            except FileNotFoundError:
                pass
            _supervisor_stopped = True
            break

    # === Finish：訓練結束，收尾並關閉資源 ===
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    if _supervisor_stopped:
        print(
            f"Training stopped cooperatively: {total_steps:,}/{total_timesteps:,} steps "
            f"in {total_time:.0f}s ({total_time/60:.1f}min)"
        )
    else:
        print(f"Training complete: {total_timesteps:,} steps in {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*60}")
    if wandb_run is not None:
        import wandb
        wandb.finish()  # 完成 WandB run（上傳 summary、標記 finished）
    env.close()         # 關閉 Isaac Sim 環境（釋放 GPU 資源）


if __name__ == "__main__":
    main()
