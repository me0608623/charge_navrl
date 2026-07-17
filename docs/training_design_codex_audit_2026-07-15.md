# Charge-Car 訓練設計 Codex 實查版（2026-07-15）

> 本文件是 `training_design_dossier_for_codex.md` 的獨立程式碼審查結果。
> 審查來源：實際 source、SA1-SA4 checkpoint `args/state_dict`、W&B runs
> `zh2tzk85 / ehvf2aq1 / 3j75osp0 / egwadwgi`。不以註解或舊實驗記憶代替 runtime ground truth。

## 0. 裁決摘要

目前問題不應再用「加一個 reward term」處理。至少有四個結構性問題先於 reward：

1. **Actor 的 LiDAR extractor/RNN 預設完全收不到 RL 梯度。** RL 只更新 policy/value heads；
   extractor/RNN 只靠 auxiliary prediction 學習。
2. **唯一訓練 RNN 的 auxiliary 任務已在難場景失效。** SA4 位置 R² 只剩 0.04，
   動態速度 R² < 0；因此 actor 在真正需要動態預判時，沒有可靠的時序表徵。
3. **SA1-SA3 的 return normalization 與 GAE value 尺度不一致。** critic 看似有高 VE，
   但 GAE 實際拿 normalized value 去減 raw reward，baseline 尺度是錯的。
4. **名為 A2C 的更新不是受 PPO trust region 保護的更新。** 每 rollout 只有一次 full-batch
   optimizer step；step 前 ratio≈1，所以 PPO clip 幾乎不會介入，step 後才看到 policy 已跳遠。

因此建議的新主線是：**固定簡單 reward + 4-frame LiDAR end-to-end PPO**。先不用 RNN、aux、
asymmetric critic、PopArt，也不再加入 clearance/r_arc/TTC shaping。等乾淨 baseline 成立後，
再用真正 sequence PPO/TBPTT 比較 GRU 是否有額外收益。

## 1. 原 dossier 必須更正的三點

1. **目前 deploy_dense checkpoint chain 沒有 GRU→RNN mismatch。**
   SA1、SA2、SA3、SA4 checkpoint 的 recurrent weights 都是 vanilla RNN：
   `weight_ih_l0=(64,64)`、`weight_hh_l0=(64,64)`。GRU 會是 `(192,64)`。
2. **SA1/SA2 無動態障礙時，7D aux target 不是全部常數 10。**
   `wd_aux_targets.py:391-397` 會 fallback 到最近 active 靜態障礙；位置 target 仍會變，
   但 future=now、top-3 velocity 全零，所以「動態 supervision」確實退化。
3. **不能只用 goal +40 / collision -15 斷言碰撞划算。**
   collision 是 terminal，會同時失去未來 goal return；GAE 在尺度正確時能把這件事傳回先前動作。
   真正問題是 credit path、dense 直行偏置與優化器，而非單看兩個 terminal 數字。

## 2. Ground-truth 路徑

| 元件 | 實際路徑 |
|---|---|
| 訓練入口 | `scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py` |
| 網路 | `scripts/reinforcement_learning/skrl/models/modular_rnn_models.py` |
| deploy configs | `scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa{1..4}_deploy_dense.yaml` |
| RL 使用的 reward | `scripts/reinforcement_learning/skrl/rnn_car_wdclean/rewards.py` |
| reward facade | `scripts/reinforcement_learning/skrl/rnn_car_modular/rewards/wd_sparse.py` |
| curriculum/reward 表 | `source/isaaclab_tasks/.../curriculum/phases/wd_single_agent_v3.py` |
| deploy 密度覆寫 | `source/isaaclab_tasks/.../curriculum/phases/wd_single_agent_v3e_deploy_dense.py` |
| LiDAR | `source/isaaclab_tasks/.../mdp/observations/obs_functions.py:360` |
| aux target | `scripts/reinforcement_learning/skrl/utils/wd_aux_targets.py` |
| critic oracle | `scripts/reinforcement_learning/skrl/utils/privileged_obs.py` |

注意：環境 RewardManager 的 dense reward **不是 RL reward**。`env.step()` 回傳的 reward 只送進
`MetricsCollector`；rollout buffer 寫入的是 trainer 另算的 `reward_flat`（train `:3789-3793`,
`:3846-3849`, `:4066`）。因此 W&B `reward/episode_mean` 與 Isaac `reward/term/*` 不是 actor
真正最佳化的總回報。只有 WD `reward_breakdown` 的分項與 RL reward 相符。

## 3. 現行資訊流與梯度

```text
79D obs
  └─ RunningNormalizer
      ├─ 79D current obs ─────────────────────────────┐
      └─ LiDARStateExtractor 96D                      │
           └─ feat normalizer                         │
               └─ fc_front → RNN64 → fc_middle → 12D │
                                                    concat 91D
                                                      ├─ PolicyHead → 19 accel + 19 omega logits
                                                      └─ ValueHead + privileged 50D → V
```

實際 optimizer 邊界：

- `charge_opt_rl`：只含 `policy_head + value_head`（train `:3090-3106`）。
- `charge_opt_aux`：RNN、predict_head、fc_middle、fc_front、extractor（`:3129-3139`）。
- rollout 全在 `torch.no_grad()`（`:3763`），RL update 預設直接吃 cached `flat_ri`（`:4549`）。

結論：79D 當前 LiDAR 可直接得到 RL 梯度，所以 static reactive navigation 還能學；但歷史資訊
只存在 detached 12D path。過去的 multi-frame 實驗也只把歷史餵給這條 detached path，
所以其失敗不能證明「frame stacking 無效」。

現有 `--rnn_rl_grad` 也不是 recurrent PPO：它把所有 `T×E` 展平，拿 rollout 時已 detach 的
hidden state，各樣本做一次單步 RNN forward（`:4472-4478`, `:4537-4547`）。梯度不會穿過時間，
也不會回到 extractor；若 aux 同開，同一 RNN 還會被兩個獨立 Adam optimizer 輪流 step。

### 3.1 `r_min=0.5` 的 hole mask 語意寫反

`wd_like_sweep_72` 先把 `< r_min` 或無效 ray 設為 `r_max=20`，最後輸出
`(sensor_range - body_radius) / 20`。因此：

- 第一個有效回波 `sensor_range=0.5m` → observation `(0.5-0.35)/20 = 0.0075`。
- hole / blind / no-hit → observation `(20-0.35)/20 = 0.9825`，不是 0。

但 trainer 的 near-obstacle、teardrop、gap 與部分 metrics 把 `obs < 0.02` 當 hole（例如
train `:3802-3810`, `:4013-4015`）。這會把 `sensor_range < 0.75m` 的**最近有效回波**丟掉，
反而不會遮掉真正 hole。故先前 near-obstacle/gap shaping 的負結果帶有此 confound。
新 reward 不使用 LiDAR 距離 shaping，但所有診斷工具仍需統一修正此語意。

r_arc range-fix 後直接用正確 converter 還原 sensor range，沒有沿用上述 `<0.02` mask；
因此 r_arc 的 A/B 失敗結論不因這條更改。

## 4. Auxiliary 實證

| run | stage | pos R² | disp R² | velocity R² | RNN feature std |
|---|---:|---:|---:|---:|---:|
| `zh2tzk85` | SA1 | 0.693 | -1 sentinel | -1 sentinel | 5.05 |
| `ehvf2aq1` | SA2 | 0.659 | -1 sentinel | -1 sentinel | 3.37 |
| `3j75osp0` | SA3 | 0.111 | -1 sentinel | -0.533 | 2.55 |
| `egwadwgi` | SA4 | 0.040 | -0.559 | -0.048 | 1.87 |

SA3/SA4 的 velocity R² < 0，代表比固定猜 target 平均值還差。這不是「可能退化」，而是目前
唯一 RNN supervision 已在需要動態預判的 stage 實際失效。

原因不是單一 bug，而是任務設計本身很差：

- `log(clamp(|e|, .01))` loss 對大誤差梯度為 `1/|e|`，優先修小錯、忽略大錯。
- target 是 oracle object state，但 actor 只有無 segmentation 的 72-beam LiDAR。
- 最近障礙與 top-K 排序會在物體交錯時換 identity，target 不連續。
- 12D bottleneck 要承載 12 個有權重的 geometry/velocity 維度，沒有多餘容量留給 policy。
- 最重要的是 RL 無權修正「哪些時序特徵才對成功避障有用」。

第一條新 baseline 應完全關閉 aux。未來若重加，只能當 end-to-end encoder 的小權重 regularizer，
不能再當 encoder/RNN 的唯一 optimizer。

## 5. Reward 實際公式與尺度

目前 `wd_sparse`：

```text
r_t = terminal_goal_or_collision
    + (0.03 / 5) · (1 - |accel_idx-9|/9)²
    + (0.03 / 5) · 0.5 · (1 - |omega_idx-9|/9)²
    - 0.005 · |Δ omega_ratio|
```

`cost_operate` 名稱錯誤。它不是 cost，而是每步最多 `+0.009` 的「零加速度 + 零角速度命令」
bonus。零加速度在此 action model 代表保持既有速度，所以它直接偏好**恆速直行**，不是停車。

逐 stage terminal effective reward：

| stage | goal | collision | timeout |
|---:|---:|---:|---:|
| SA1 | +40 | -5 | 0 |
| SA2 | +40 | -8 | 0 |
| SA3 | +40 | -12 | 0 |
| SA4-SA5 | +40 | -15 | 0 |
| SA6 | +40 | -25 | -12.5 |
| SA7 | +40 | -50 | -25 |
| SA8 | +40 | -100 | -50 |

reward objective 在 stage 間改變 20 倍，critic target distribution 也跟著改。既然部署目標已知，
reward 公式與權重應從 SA1 到 final 完全固定，只改場景分布。

`γ=0.984` 在 5 Hz 下：2 秒後權重 `γ¹⁰≈0.851`、5 秒 `γ²⁵≈0.668`、10 秒
`γ⁵⁰≈0.446`。terminal collision 本身可提供 1.5-2m 的 credit，但前提是 value/GAE 尺度與
encoder gradient 正確。沒有必要先加新的障礙距離稅。

## 6. Critic / advantage / trust region

### 6.1 SA1-SA3 return normalization 是確定的尺度 bug

例如 SA1 W&B：raw returns `mean=24.85, std=9.19`，critic prediction
`mean=-0.018, std=0.845`。critic 用 normalized target 訓練，但下一輪 GAE 直接把該 normalized
output 當 raw value 使用（train `:4425-4434`）。W&B VE=0.80 只代表 normalized 空間擬合好，
不代表 actor 的 GAE baseline 尺度正確。

### 6.2 現在的 PopArt 不是完整 PopArt

SA4 會用 running mean/std 反正規化 value，尺度方向正確；但統計變更時沒有做 PopArt 必要的
output-preserving transform：

```text
w_new = (σ_old / σ_new) · w_old
b_new = (σ_old · b_old + μ_old - μ_new) / σ_new
```

目前只更新 `μ/σ`，所以統計移動本身會改變 denormalized value。新 reward 尺度若控制在約
`[-20, +20]`，第一版不需要 return normalization 或 PopArt。

### 6.3 PPO clip 在 A2C 單步路徑幾乎不生效

每 rollout 只執行一次 full-batch forward/backward/step。optimizer step 前 policy 幾乎等於
rollout policy，ratio≈1，故 clipped surrogate 無法限制「即將發生的」那一步。`target_kl`
只有 parser，更新 loop 沒有 break。

實際 post-update W&B：

- SA1：KL 0.0196、ratio max 3.34、22.1% samples 超出 ±20%。
- SA4：KL 0.0111、ratio max 18.87、10.6% samples 超出 ±20%。

另外 `wd_update_clip=true` 會繞過 global `max_grad_norm=0.5`，改用 actor=8、critic=30；
W&B 多數 rollout 根本沒碰到這兩個 cap。`adv_norm=mean_only` 又保留 raw std 8-13。

## 7. Critic privileged channel 的 final-density 缺口

`privileged_obs.py` 固定只讀 `obstacle_0...obstacle_9`，不是「距離最近 10 個」。final 場景有
20 個障礙，因此 critic oracle 會永久忽略後 10 slots；雖然 critic 仍有 LiDAR，但額外 oracle
是不完整且依 slot index 的。乾淨 baseline 應先用 symmetric critic。若日後需 asymmetric，
應從全部 24 slots 做 top-K nearest 或 permutation-invariant pooling。

## 8. Curriculum 與真實密度

真實部署：`20 / (12×12) = 0.1389 obs/m²`。

| stage | 場地 | 靜+動 | density | 相對真實 |
|---:|---:|---:|---:|---:|
| SA1 | 20×20 | 2+0 | 0.0050 | 3.6% |
| SA2 | 18×18 | 4+0 | 0.0123 | 8.9% |
| SA3 | 17×17 | 6+1 | 0.0242 | 17.4% |
| SA4 | 16×16 | 8+2 | 0.0391 | 28.1% |
| SA5 | 15×15 | 10+3 | 0.0578 | 41.6% |
| SA6 | 14×14 | 12+4 | 0.0816 | 58.8% |
| SA7 | 13×13 | 14+6 | 0.1183 | 85.2% |
| SA8 | 12×12 | 15+(6..8) | 0.1458..0.1597 | 105..115% |

目前 repository 只有 SA1-SA4 deploy config，沒有 SA5-SA8；SA4 policy 在真實密度評估是
3.56× density OOD。這不會否定同場 A/B 結果，但不能把 SA4 的 final-density SR 當成完整
curriculum 上限。另 `arena_shape: mix` 目前只是 placeholder，實際仍是 square。

新 curriculum 的「平滑」應明確定義：

- reward、網路、optimizer 全程不變；只改場景 difficulty distribution。
- 相鄰 difficulty 的 median density 增幅不超過 25%。
- promotion 時 per-env `p(next)` 依 `0 → .25 → .5 → .75 → 1` 漸增，不一次切全部 1024 env。
- final distribution 必須包含 exact deployment `12×12 / 20`，不要只訓 21-23。
- 建議單一連續 run，不在每個 stage 重置 Adam/normalizer；stage checkpoint 只作 eval/rollback。

## 9. 建議的新 SA1 主線

### 9.1 網路：4-frame LiDAR，端到端 feed-forward PPO

```text
LiDAR [t, t-1, t-2, t-3] = [4,72]（5 Hz，共 0.6 s）
  → circular Conv1d(4→32→64→64)
  → flatten/projection 128D
  → concat ego(v,ω,a), goal(x,y), time
  → actor MLP 256→256→38 logits

critic：獨立同型 encoder + MLP → V（symmetric，無 oracle）
```

要求：

- LiDAR encoder 必須在 actor optimizer 中，policy loss backward 後 grad norm > 0。
- 關閉 `PreprocessRNN`、12D bottleneck、predict_head、aux optimizer。
- reset 時歷史幀用第一張 current frame 重複，不可填 0；0 在此 LiDAR 語意不是正常歷史。
- 第一版保留現有兩個 19-way action heads以減少變因；另記 sampled `(accel,omega)` 聯合分布。
- 若 frame-stack baseline 成立，再做真正 GRU-PPO：按 env 保留連續 sequence、initial hidden，
  BPTT 16-32 steps。不可使用目前的 flat `--rnn_rl_grad`。

### 9.2 固定 reward（effective per environment step）

```text
r_t = 1.0 · (d_prev - d_now)
    - 0.01
    - 0.01 · |Δv| / v_max
    - 0.005 · |Δω| / ω_max
    + 10 · I(goal)
    - 15 · I(collision)
    - 5  · I(timeout)
```

規則：

- goal/collision/timeout 互斥；terminal 優先級要有單元測試。
- progress 不得跨 episode/reset 計算；在 env RewardManager reset 前計算。
- 所有 stage 完全相同，不再把 collision 從 -5 漸升到 -100。
- 不保留 `cost_operate` 正 bonus；不加 clearance、near-obstacle、TTC、r_arc、gap reward。
- 先用單一 canonical reward path，buffer、episode return、W&B 分項都來自同一 tensor。

若使用 IsaacLab RewardManager（它自動乘 `dt=0.2`），對應 config weights 是：
progress `+5`、time `-0.05`、goal `+50`、collision `-75`、timeout `-25`；smoothness
權重依函數是否已正規化設定。驗收應檢查 **effective tensor 值**，不能只看 YAML weight。

預期成功 episode（初始距離 2-9m）：progress 約 `+1.5..+8.5`、goal `+10`、time/smooth
通常 `<1`，總 return 約 `+10.5..+17.8`。collision 即使先取得最多約 +8.5 progress，
加 -15 後仍通常為負；停住 300 steps 則 time+timeout 約 -8。

### 9.3 PPO 固定設定

| 參數 | 建議起點 |
|---|---:|
| rollout | 128 × 1024 env |
| PPO epochs / minibatches | 4 / 8 |
| actor+critic lr | 2e-4 |
| clip ε / value clip | 0.2 / 0.2 |
| γ / GAE λ | 0.99 / 0.95 |
| advantage | full `(A-mean)/(std+eps)` |
| value target | raw，無 normalize_return、無 PopArt |
| entropy | joint coeff 0.005，線性降到 0.001；無 0.04/0.08 floor |
| target KL | 0.015，必須真的停止後續 minibatch/epoch |
| gradient clip | 所有 trainable params global norm 0.5 |
| aux / privileged critic | off / symmetric |

現行 SA4 entropy bonus約 `0.04×2.09 + 0.08×2.13 ≈ 0.254`，而 logged policy loss 約
0.004；新係數最大 joint bonus約 `0.005×ln(19²)=0.029`，避免 entropy 在收斂後主導。

## 10. 開長訓前的最小驗證

先用 256 env、固定 seed、相同資料量做兩臂，不要直接燒完整 SA1-SA8：

1. **Main：FS4 end-to-end PPO + 固定 reward。**
2. **Negative control：同網路但 detach LiDAR encoder。**

必須先通過：

- policy loss backward 後 actor LiDAR encoder grad norm > 0；negative control = 0。
- 同一 step 的 `buffer.reward == logged_total_reward == Σ logged_terms`。
- goal/collision/timeout/reset progress 的 deterministic unit tests。
- PPO 第二 minibatch開始 ratio 不再恆1；KL>0.015 真的 break。
- post-update ratio tail、global grad norm、value target/value prediction同尺度。

之後在 exact target `12×12 / 20` 做固定大樣本 deterministic eval，主要裁決：

- SR、static/dynamic/wall collision 分解。
- goal 正前方且障礙落在預測路徑時，`|ω|` onset 是否移到 1.5-2.0m。
- counterfactual safe-direction agreement 是否顯著高於 50%。
- clear-scene speed、stop rate、路徑長度與 jerk 不惡化。
- 至少 3 seeds；同 seed 的單次 +1~2pp 不算架構成功。

## 11. 最終判定

**可以從 SA1 重訓，但不能沿用現有 SA1 config。** 沿用會同時重現 detached temporal encoder、
失效 aux、normalize_return/GAE 尺度錯配、無效 PPO clipping與過高 entropy floor。

優先順序：

1. 單一 reward source + 上述固定簡單 reward。
2. frame-stack encoder 端到端吃 RL gradient。
3. 標準 minibatch PPO / full advantage norm / 真 global clip / KL stop。
4. exact 12×12/20 的連續 density curriculum。
5. baseline 成立後，才比較 true GRU-PPO 或重新加入小權重 auxiliary task。
