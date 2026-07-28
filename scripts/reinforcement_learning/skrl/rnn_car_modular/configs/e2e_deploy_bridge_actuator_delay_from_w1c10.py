"""Pre-deployment actuator-delay bridge — **diagnostic, not a graduation arm**.

⚠️ 這**不是**通過驗收的 final policy。SA7 / SA7.1 尚未通過四走廊 gate
（random_2d 十組樣本全落在 64.4–72.5%，門檻 85%），所以沒有任何 SA7 血緣的
checkpoint 可以充當 final accepted policy。本 config 的用途是**把目前唯一乾淨、
可部署的 checkpoint 橋接到實車的致動延遲**，讓部署先有東西可用；走廊能力的攻堅
另案進行（碰撞時相分析），與本橋接互不阻擋。

## 為什麼 parent 是 W1 而不是 SA7

`e2e_sa6_w1_wander_from_d0` 是目前 SA6 血緣最乾淨的配方，其 c10
（`checkpoint_1280.pt`）在四個走廊模式與 Gate2 上都優於 D0。若改繼承 SA7 配方，
會同時動到 stage / scene mix **和** actuator DR —— 兩個因子綁在一起，
橋接失敗時無法歸因。**已知的教訓**：SA7.1 正是因為
`long_corridor_fraction` 一個欄位同時承載「走廊擴張」與「native 稀釋」兩個因子，
導致 −13.34pp 的結果無法歸因到任一原因。這裡不重蹈覆轍。

**不得使用 SA7 或 SA7.1 的 checkpoint**（兩者皆未通過閘）。

## 單一實驗因子：凍結的致動器 DR bundle

相對 W1 parent，非 metadata 欄位只差**四項**。唯一新增的行為因子是
`enable_actuator_dr` 所啟用的整包致動器 DR，而不是「只有 delay」的純消融：

| 欄位 | W1 parent | 本 bridge | 性質 |
|---|---|---|---|
| `enable_actuator_dr` | False | **True** | ← 啟用 delay + velocity scale + motor lag |
| `checkpoint` | D0 | **W1-c10** | 暖啟動點 |
| `timesteps` | 3840 (30 iter) | **1280 (10 iter)** | 訓練預算 |
| `save_interval` | 5 | **2** | 存檔頻率 |

reward / network / PPO / LR / scene mix / seed / issued-action history / obs delay
全部逐欄等同 parent，由 `test_sa6_sa8_replay_configs.py` 的 allowed-diff 測試
以 `dataclasses.fields` 全欄比對鎖住。

因此，這次結果只能歸因為「凍結的 actuator-DR bundle」是否有效。若日後要回答
「200 ms delay 單獨是否有效」，必須另開 delay-only 臂，把
`actuator_velocity_scale=(1.0,1.0)` 且停用 motor lag；不得拿本 bridge 直接宣稱。

## 致動器規格（歷史規格，不得更動）

```
enable_actuator_dr      = True
actuator_delay_range    = (0, 2)      每 episode / per-env 離散均勻抽 {0,1,2} 步
                                       control_dt = 0.2 s → 0 / 200 / 400 ms
actuator_velocity_scale = (0.9, 1.1)  per-episode、對 (v, ω) 各自抽
actuator_motor_lag      = 0.3         一階低通 α
obs_delay_steps         = (0, 0)      ← **觀測延遲永遠關閉**
use_action_history      = True        前兩步 issued-command history 4D，觀測 83D
```

⚠️ **不要把 delay 改成固定 `(1, 1)`。** U{0,1,2} 是歷史規格：policy 必須在
單一權重下同時容忍 0 / 200 / 400 ms，而不是只被調到某一個延遲。

⚠️ `obs_delay_steps` 必須維持 `(0, 0)`。觀測延遲曾被誤植為「馬達延遲」的實作，
並被單變因隔離證實是 stage3 deterministic 滿舵塌縮的兇手
（見 `finding_obsdelay_misimplemented_as_motordelay` /
`finding_bangbang_degenerate_penalty0`）。致動延遲（本檔）保留，觀測延遲永遠關。

## 驗收協定

三個延遲值**各自** deterministic 驗收：

| delay | 對應 | 角色 |
|---|---|---|
| actuator off | true clean | **clean regression 守門** —— 不得比 W1-c10 退步 |
| 0 步 | 0 ms | 零 dead-time bundle（scale / lag 仍生效） |
| 1 步 | **200 ms** | **主要部署指標** |
| 2 步 | 400 ms | robustness stress |

## Warm start

model **與** Adam optimizer 都續用（`no_resume_optimizer=False`，繼承 parent）。
橋接只有 10 個 iteration，重置動量會讓這麼短的預算幾乎全花在重新暖機。
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa6_w1_wander_from_d0 import CONFIG as _W1


#: W1 wander 臂的 c10 —— 目前唯一乾淨、可部署的 checkpoint。
#: W1 每 5 iter 存檔且每 iteration 128 步，故 c10 = checkpoint_1280。
W1_C10_CHECKPOINT = (
    "/home/aa/IsaacLab/logs/rnn_car/"
    "sa6_k8_obb_corridor_w1_wander_s42/checkpoint_1280.pt"
)

#: 10 iterations × 128 步/iteration。
BRIDGE_TIMESTEPS = 1280
BRIDGE_SAVE_INTERVAL = 2

#: 歷史規格，逐字凍結。改這幾個值等於換掉實驗。
ACTUATOR_DELAY_RANGE = (0, 2)
ACTUATOR_VELOCITY_SCALE = (0.9, 1.1)
ACTUATOR_MOTOR_LAG = 0.3


CONFIG = replace(
    _W1,
    name="e2e_deploy_bridge_actuator_delay_from_w1c10",
    description=(
        "PRE-DEPLOYMENT DIAGNOSTIC BRIDGE (not a graduation arm). Warm-starts "
        "the clean W1-c10 checkpoint and turns on actuator domain "
        "randomisation so the deployable policy tolerates the real vehicle's "
        "command latency. Delay is the frozen historical spec U{0,1,2} steps "
        "= 0/200/400 ms at control_dt 0.2 s, with per-episode velocity scale "
        "(0.9, 1.1) and first-order motor lag 0.3. Observation delay stays "
        "hard off. Ten iterations, checkpoint every two, model + Adam warm "
        "start. Reward, network, PPO, LR, scene mix, seed and the 4D issued-"
        "action history (83D observation) are inherited from the W1 parent "
        "unchanged, so the frozen actuator-DR bundle is the only behavioural "
        "factor."
    ),
    checkpoint=W1_C10_CHECKPOINT,
    timesteps=BRIDGE_TIMESTEPS,
    save_interval=BRIDGE_SAVE_INTERVAL,
    # ── 唯一新增的行為因子：整包 actuator DR ──────────────────────────
    enable_actuator_dr=True,
    # 以下三項與 parent 預設同值，明寫是為了讓規格出現在 config 本體，
    # 而不是散落在 dataclass 預設值裡（config-lock 測試會逐項比對）。
    actuator_delay_range=ACTUATOR_DELAY_RANGE,
    actuator_velocity_scale=ACTUATOR_VELOCITY_SCALE,
    actuator_motor_lag=ACTUATOR_MOTOR_LAG,
    # 觀測延遲永遠關 —— 與 parent 同值，明寫以防日後被誤開。
    obs_delay_steps=(0, 0),
    tags=_W1.tags + ("actuator_delay_bridge", "pre_deployment", "diagnostic"),
    notes=(
        "Diagnostic pre-deployment bridge; NOT a final accepted policy. SA7 "
        "and SA7.1 both fail the four corridor gates (random_2d 64.4-72.5% "
        "across ten samples against an 85% bar), so neither may be shipped "
        "and neither is used here - the warm start is the clean W1-c10. The "
        "parent is the W1 SA6 recipe rather than SA7 on purpose: inheriting "
        "SA7 would change stage and scene mix at the same time as actuator "
        "DR, and a two-factor change cannot be attributed (exactly the defect "
        "that made SA7.1's -13.34pp result unattributable). Allowed diff vs "
        "parent is four fields: enable_actuator_dr True, the W1-c10 "
        "checkpoint, 1280 timesteps (10 iterations) and save_interval 2. "
        "Delay stays the frozen historical U{0,1,2} steps - do NOT collapse "
        "it to a fixed (1,1); the policy must tolerate 0/200/400 ms under one "
        "set of weights. obs_delay_steps stays (0,0): observation delay was "
        "the confirmed cause of the stage-3 bang-bang collapse and is a "
        "different mechanism from actuator delay. Acceptance evaluates the "
        "three delays separately, plus an actuator-off true-clean baseline. "
        "Zero steps is still a scale/lag bundle and must not be mislabeled "
        "clean; 1 step (200 ms) is the primary deployment metric and 2 steps "
        "(400 ms) is the robustness stress. Model and Adam state both resume; "
        "a ten-iteration "
        "budget cannot afford a momentum reset."
    ),
)
