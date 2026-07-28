"""SA7.1：同時訓練「真實混合走廊」與「正式 Gate 題型走廊」。

2026-07-28 診斷結論：SA7（r3，300 iterations）窄縫 Gate5a 全數 100%，
但**四種走廊模式全部沒過**，且 300 iterations 沒有產出優於起點的 checkpoint。
曝光審計（n=60001）指出根因不是訓練量不足，而是**訓練題型與評測題型不同**：

    訓練走廊：每 env 逐一抽 D=1..5、三種 family 各 33.33% 混合、
              24.56% 有配對（lat+long 67.8% / long+long 32.2%），
              **random_2d 從不參與配對**
    正式 Gate：固定 4 靜態 + 2 動態、單一純模式（lateral / longitudinal /
              random_2d / mixed_iid 各跑一次）

也就是說 policy 從來沒有在「純單模式 + 固定 4S2D」這個分佈下被訓練過。
SA7.1 的唯一改動就是**把正式 Gate 題型直接加進訓練分佈**，其餘
reward、network、LR、seed、gate 門檻全部不動。

replay 比例（本檔唯一覆寫 `long_corridor_fraction`）：

    58% 當前階段一般場景        （原 68%，讓出 10% 給 Gate 題型）
    10% 前階段一般場景          previous_stage_replay_fraction = 0.10
    12% 窄縫（1.2-1.4 m）        narrow_passage_fraction        = 0.12
    20% 長走廊 4x10 m            long_corridor_fraction         = 0.20
         ├─ 10% 真實混合  ← count_mix（SA7_DENSITY_MIX）+ 互動分佈
         └─ 10% Gate 題型 ← 固定 4S+2D、四種純模式輪派

`long_corridor_gate_aligned_share = 0.5` 是**走廊那 20% 之內的比例**，
不是全域比例；0.5 x 0.20 = 0.10 才是 Gate 題型的全域曝光。這樣寫的好處是
之後調 `long_corridor_fraction` 時兩邊會等比縮放，不會一邊被悄悄壓扁。

Gate 題型那一半在事件層走 `_GATE_ALIGNED_MODES` 四模式 round-robin
（`order[k::4]`），每個 batch 內四模式數量差 <= 1，且刻意不套 count_mix ——
它就是要複製評測當下的固定 4S+2D 幾何。

晉級 SA8 的條件（全部要過，缺一不可）：

    走廊 lateral / longitudinal / random_2d / mixed_iid 四閘
    Gate2（一般導航）
    Gate5a（sealed 窄縫；Gate5b 10 m 死角壓測不參與判定）

短跑協議：30 iterations、每 5 iter 存檔、逐顆評四種模式。
**若仍沒過就停止加長訓練**，改研究碰撞時相與 pair 類型再決定 reward。
"""

from dataclasses import replace

from rnn_car_modular.configs.e2e_sa7_wander_from_w1c10 import (
    CONFIG as _SA7_BASE,
    SA7_DENSITY_MIX,
    _W1_C10_CHECKPOINT,
)


#: 走廊總量 20%，其中一半給 Gate 題型 -> 全域各 10%。
LONG_CORRIDOR_FRACTION = 0.20
GATE_ALIGNED_SHARE = 0.5


CONFIG = replace(
    _SA7_BASE,
    name="e2e_sa71_gate_aligned_from_w1c10",
    description=(
        "SA7.1 warm-started from W1-c10, training the realistic mixed corridor "
        "and the formal gate geometry side by side. The corridor slice grows "
        "from 10% to 20% (native scenes 68% -> 58%); half of it replays the "
        "frozen per-env density mix with pedestrian interactions, the other "
        "half replays the evaluation geometry itself - fixed 4 static + 2 "
        "dynamic, one pure motion mode per env, round-robin across the four "
        "gate modes. Reward, network, learning rate, seed and gate thresholds "
        "are untouched."
    ),
    checkpoint=_W1_C10_CHECKPOINT,
    no_resume_optimizer=True,
    long_corridor_fraction=LONG_CORRIDOR_FRACTION,
    long_corridor_obstacle_count_mix=SA7_DENSITY_MIX,
    long_corridor_gate_aligned_share=GATE_ALIGNED_SHARE,
    tags=_SA7_BASE.tags + ("gate_aligned_replay", "sa71"),
    notes=(
        "Motivated by the 2026-07-28 exposure audit (n=60001): SA7 passes "
        "narrow-gap Gate5a at 100% on every candidate but fails all four "
        "corridor gates, and 300 iterations produced nothing better than its "
        "own starting checkpoint. Training and evaluation are different "
        "problem types - training draws D=1..5 per env with all three motion "
        "families mixed at 33.33% each and 24.56% of envs carrying an "
        "interaction pair (lat+long 67.8% / long+long 32.2%, random_2d never "
        "paired), while the gate presents a fixed 4S+2D scene in a single "
        "pure mode. The policy has never seen the gate distribution during "
        "training. SA7.1's only change is to put that distribution into the "
        "training mix: long_corridor_fraction 0.10 -> 0.20 with "
        "gate_aligned_share 0.5, so realistic-mixed and gate-aligned replay "
        "each get 10% globally and native scenes drop 68% -> 58%. Everything "
        "else - reward, network, LR, replay ratios for narrow (12%) and "
        "previous-stage (10%), seed, gate thresholds - is inherited "
        "unchanged. Protocol: 30 iterations, checkpoint every 5, evaluate all "
        "four corridor modes per checkpoint. Promotion to SA8 requires the "
        "four corridor gates plus Gate2 plus Gate5a. If it still fails, stop "
        "extending training and study collision phase and pair type before "
        "touching reward."
    ),
)
