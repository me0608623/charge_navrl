"""warp_drive_single_agent_v1 — 單車 WD phase/task registry

這個檔案是 `warp_drive_single_agent_v1` curriculum version 的「參數設定部門」。

設計目標：
1. 每個 phase 不只是難度 stage，而是一個完整 task spec。
2. 每個 task 集中定義：
   - scene        場景與 episode / gamma
   - reward       WD sparse reward 相關設定
   - exploration  每 phase 探索強度
   - behavior     rule-based obstacle behavior / speed overrides
   - trainer      允許 phase 微調的 trainer 超參（只同步安全欄位）
   - transition   升降階條件
3. `train_rnn_car_wdclip.py` 保持主要訓練邏輯；
   本檔負責提供每個 phase/task 的單一真實來源 (single source of truth)。

相容性原則：
- 為了不破壞既有 IsaacLab / curriculum 架構，
  本檔最後仍輸出 legacy flat stage dict 給 `goal_obstacle_curriculum.py` 使用。
- 也就是說：人類主要讀/改的是巢狀 schema；
  系統實際吃的是 bridge 後的 flat schema。

閱讀方式：
- 若你想直接調 task 定義，優先修改 STAGES。
- 若你想知道訓練時實際傳出去的欄位，請看最下面 `_flatten_phase()`。
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# 全域設定（適用所有 phase）
# ═══════════════════════════════════════════════════════════════════════════

GLOBAL = {
    # 障礙物控制模式（適用所有 phase）。
    # "rule_based": BehaviorScheduler 控制（patrol/random_walk 等確定性行為，無 NN）
    # "learned":    使用 learned FC policy（ObstaclePolicyFC），每 train_goal_rate 次交替訓練
    # "scripted":   環境內建 move_obstacles_vectorized（目標導向移動）
    "obstacle_mode": "rule_based",

    # 升階需要連續滿足條件幾次。
    # 例如設 5 代表：不是單次 SR 達標就升，而是要穩定達標 5 個檢查窗口。
    "upgrade_pass_required": 5,

    # 升階後是否清空 outcome window。
    # True = 升到新 phase 後重新累積新階段的 SR/CR/TO 統計，避免舊階段資料污染。
    "clear_window_on_promote": True,

    # 只有這些 trainer 參數允許在 phase transition 時由 training loop 動態同步。
    # 這些都是「不改 tensor shape / 不改網路結構」的安全欄位。
    "trainer_sync_allowlist": [
        "lr",                     # RL optimizer learning rate
        "rnn_lr",                 # Aux/RNN optimizer learning rate
        "vf_coeff",               # critic loss 權重
        "max_grad_norm",          # RL grad clip 上限
        "aux_grad_clip",          # aux grad clip 上限
        "wd_actor_update_clip",   # WD-style actor grad cap
        "wd_critic_update_clip",  # WD-style critic grad cap
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Phase/task 定義
# ═══════════════════════════════════════════════════════════════════════════
#
# 每個 stage 都包含 6 個區塊：
# 1. name         : 人類可讀的 task 名稱
# 2. scene        : 場景/回合長度/折扣率
# 3. reward       : sparse reward 相關權重
# 4. exploration  : 探索係數（目前是 linear / angular entropy）
# 5. behavior     : rule-based obstacle 行為分佈與速度覆寫
# 6. trainer      : 允許 phase 微調的安全 trainer 超參
# 7. transition   : 升降階條件
#
# 注意：
# - 這裡故意不放會破壞訓練架構的欄位，例如 num_envs / rollout_length / hidden_dim。
# - 那些屬於 run-level / architecture-level 參數，應留在 CLI 或主訓練腳本。
# ═══════════════════════════════════════════════════════════════════════════

STAGES = [
    # ──────────────────────────────────────────────────────────────────────
    # Stage 1 / SA1
    # 【訓練能力】抵達目的地（導航）
    #   論文策略：高 goal density + 低 penalty → p(goal) 高、p(obs) 低
    #   短 episode + 多 goal = 成功事件頻率高，讓 agent 先學會「往 goal 走」
    #   少量動態障礙(obs=3)只為讓 RNN 提前接觸時序信號，不期待學會避障
    #
    # ── WD P1 參考 (train_rnn_car.py, 8 phases, fps=4, ent×2.5) ──
    # spots=6, goals=20, obs=3, penalty=-5, cost=0.03,
    # ent_linear=0.10, ent_angular=0.375, speed=0.80, 60s, batch=50K
    # wall=0
    #
    # ── 訓練預算建議 (run-level，不由本檔自動套用) ──
    # rollout_length=300 固定；timesteps = updates × rollout_length；
    # interactions = num_envs × rollout_length × updates。
    # Sanity: num_envs=512,  timesteps=30000,  updates=100,  interactions=15.4M
    # Formal: num_envs=512,  timesteps=270000, updates=900,  interactions=138.2M
    # Long:   num_envs=512,  timesteps=450000, updates=1500, interactions=230.4M
    # 說明：SA1 是 sparse goal bootstrap；優先保證 update 次數，不急著放大 num_envs。
    # ──────────────────────────────────────────────────────────────────────
    {
        # task 名稱，主要用於 console / WandB / curriculum debug。
        "name": "SA1_goal_wall_dyn",

        "scene": {
            # 場內同時存在的 goal 數量。
            # goals 越多，隨機走到某個 goal 的機率越高，早期成功事件頻率越高。
            "goals": 10,                       # WD P1=20

            # goal 與 robot 的重採樣距離範圍 (min_m, max_m)。
            # 前期用短距離讓任務更容易，減少 sparse reward 太稀的問題。
            "goal_distance": (2.0, 6.0),

            # 靜態障礙物數量。
            # SA1 先設 0，避免一開始同時學太多種幾何干擾。
            "static_obstacles": 0,             # WD P1: 無 static（spots 是 virtual）

            # 動態障礙物最大數量。
            # 這是 per-env randomization 的上限，不一定每個 env 都塞滿。
            "dynamic_obstacles": 2,            # WD P1=3

            # 動態障礙物最小數量。
            # 代表每個 env 會在 [1, 2] 之間抽樣動態障礙數量。
            "dynamic_obstacles_min": 1,

            # 內牆最少數量。
            "walls_min": 0,                    # WD P1: 無牆

            # 內牆最多數量。
            "walls_max": 1,                    # WD P1: 無牆

            # 內牆目標長度（公尺）。
            # 不是保證每一面牆都剛好等長，而是 wall randomization 的參考長度。
            "wall_length": 5.0,

            # episode 最長秒數。
            # 越短代表 timeout 更快，鼓勵 agent 不要拖太久。
            "episode_s": 60,                   # WD P1=60

            # 這個 phase 對應的折扣因子。
            # 雖然 train script 目前主要維持 WD constant gamma 概念，
            # 但 env/curriculum 仍保存這個欄位作為 phase 描述與 runtime info。
            "gamma": 0.984,
        },

        "reward": {
            # 碰撞懲罰。數值越負，policy 越傾向保守避障。
            "penalty_hit": -5.0,               # WD P1=-5

            # 到 goal 的獎勵。這裡維持 WD 常見 +40 設定。
            "reward_get_goal": 40.0,           # WD P1=40

            # 每步操作成本。Phase 1 保留少量 cost，避免 agent 高頻亂抖。
            "cost_operate": 0.03,              # WD P1=0.03

            # 若要對 env reward manager 的 dense reward term 做 per-phase 調整，
            # 可在這裡填 dict；目前 WD sparse task 先不使用，設 None。
            "reward_weights": None,
        },

        "exploration": {
            # 線速度 head 的 entropy coeff。
            # 實驗證實 0.01 可讓 ent 自然收斂到 ~4.0，不需人為 floor。
            "entropy_linear": 0.01,            # WD P1=0.10 → 實測調降

            # 角速度 head 的 entropy coeff。
            # 配合 linear head 等比例調降。
            "entropy_angular": 0.02,           # WD P1=0.375 → 實測調降
        },

        # ── behavior ──
        # WD 原版沒有 behavior_mix / speed_overrides 機制，只有全局 speed 參數。
        # SA 新增 BehaviorScheduler，可 per-behavior 指定類型比例與速度範圍。
        "behavior": {
            "obstacle_speed": 0.80,            # WD P1=0.80 | 全局動態障礙速度倍率，乘以 base_speed 決定最大移動速度
            "behavior_mix": {
                "patrol": 0.60,                # WD: 無此機制 | 沿固定路徑巡邏，提供可預測的週期性動態模式
                "random_walk": 0.40,           # WD: 無此機制 | 隨機方向移動，低速不規則時序干擾
            },
            "speed_overrides": {               # WD: 無此機制（只有全局 speed）| per-behavior [min, max] 速度範圍覆寫
                "patrol": {"speed_range": (0.15, 0.35)},       # 低速穩定巡邏，作為最早期動態樣本
                "random_walk": {"speed_range": (0.10, 0.30)},  # 更低速，避免 SA1 被動態干擾壓垮
            },
        },

        # ── trainer ──
        # WD 原版全 phase 使用固定超參，不做 per-phase 微調。
        # SA 允許高難 phase 降低 LR / 收緊 grad clip，穩定後期訓練。
        "trainer": {
            "lr": 2e-4,                        # WD 全 phase 固定 2e-4 (spot_lr) | RL policy/value head optimizer 學習率
            "rnn_lr": 5e-4,                    # WD 全 phase 固定 5e-4 (spot_rnn_model_lr) | RNN / aux optimizer 學習率
            "vf_coeff": 0.5,                   # WD 全 phase 固定 0.025 (spot_vf_loss_coeff) | value loss 在 joint loss 中的權重係數
            "max_grad_norm": 1.0,              # WD 全 phase 固定 1.0 | RL optimizer 全局梯度裁剪上限
            "aux_grad_clip": 0.5,              # WD: 無獨立 aux clip（沿用 max_grad_norm）| aux/RNN 更新專用梯度裁剪，隔離 RNN spike
            "wd_actor_update_clip": 8.0,       # WD 全 phase 固定 8.0 (k) | actor grad norm 超過此值時縮放，防止 policy 更新過猛
            "wd_critic_update_clip": 30.0,     # WD 全 phase 固定 30.0 (q) | critic grad norm 超過此值時縮放，防止 value spike 壓制 actor
        },

        # ── transition ──
        # WD 原版用固定 iteration budget 升階（每 phase 跑完指定 batch 數即升），不用 SR/CR/TO 門檻。
        # SA 改為 performance-based 升降階，需連續滿足條件 upgrade_pass_required 次。
        "transition": {
            "upgrade_sr": 0.72,                # WD: 無 SR 門檻 | 成功率（到達 goal 比例）≥ 此值才允許升階
            "upgrade_max_cr": 1.0,             # WD: 無 CR 門檻 | 碰撞率必須 ≤ 此值；1.0=不限制，SA1 先專注導航
            "upgrade_max_to": 0.30,            # WD: 無 TO 門檻 | 超時率必須 ≤ 此值，避免 agent 拖延不前進
            "upgrade_min_dyn_sr": 0.0,         # WD: 無此指標 | 僅在含動態障礙 env 上的 SR 下限；0.0=不限制
            "min_stage_updates": 50,           # WD: 用固定 batch budget | 至少停留多少 rollout 更新次數，防止過早升階
            "downgrade_sr": 0.0,               # WD: 無降階機制 | SR 低於此值觸發降階；0.0=SA1 幾乎不降
            "downgrade_min_cr": 1.0,           # WD: 無降階機制 | CR 高於此值觸發降階；1.0=不限制
            "downgrade_min_to": 1.0,           # WD: 無降階機制 | TO 高於此值觸發降階；1.0=不限制
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 2 / SA2
    # 【訓練能力】抵達目的地（強化）+ 初步避障意識
    #   論文策略：penalty 從 -5 → -8，開始讓 p(obs) 提升但仍以導航為主
    #   goal 仍多(16)、episode 仍短(60s) → 導航成功事件仍是主要學習信號
    #   entropy 提高(0.30) → 鼓勵探索更多路徑，為後續避障做準備
    #
    # ── WD P2 參考 ──
    # spots=3, goals=16, obs=3, penalty=-8, cost=0.0,
    # ent_linear=0.30, ent_angular=0.375, speed=0.85, 60s, batch=90K
    # wall=0
    #
    # ── 訓練預算建議 (run-level，不由本檔自動套用) ──
    # rollout_length=300 固定；timesteps = updates × rollout_length；
    # interactions = num_envs × rollout_length × updates。
    # Sanity: num_envs=1024, timesteps=30000,  updates=100,  interactions=30.7M
    # Formal: num_envs=1024, timesteps=180000, updates=600,  interactions=184.3M
    # Long:   num_envs=1024, timesteps=450000, updates=1500, interactions=460.8M
    # 說明：SA2 從 SA1 best checkpoint 開新 run；建議先 600 updates 判斷，再延長。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA2_goal_wall_2dyn",

        "scene": {
            "goals": 8,                         # WD P2=16
            "goal_distance": (2.0, 7.0),       # 稍微拉遠最大目標距離
            "static_obstacles": 0,             # WD P2: 無 static
            "dynamic_obstacles": 2,            # WD P2=3
            "dynamic_obstacles_min": 1,        # 每 env 仍在 [1, 2] 抽樣
            "walls_min": 0,                    # WD P2: 無牆
            "walls_max": 1,                    # WD P2: 無牆
            "wall_length": 5.0,                # WD P2: 無牆
            "episode_s": 60,                   # WD P2=60
            "gamma": 0.984,                    # 與 Phase 1 統一，WD constant gamma
        },

        "reward": {
            "penalty_hit": -8.0,               # WD P2=-8
            "reward_get_goal": 40.0,           # WD P2=40
            "cost_operate": 0.0,               # WD P2=0.0
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.01,            # WD P2=0.30 → 實測調降
            "entropy_angular": 0.02,           # WD P2=0.375 → 實測調降
        },

        # ── behavior ──
        # WD P2 只有全局 speed=0.85，無 behavior_mix / speed_overrides。
        "behavior": {
            "obstacle_speed": 0.80,            # WD P2=0.85 | 全局動態障礙速度倍率
            "behavior_mix": {
                "static": 0.40,                # WD: 無此機制 | 固定不動，讓 RNN 學習區分靜態 vs 動態回波
                "patrol": 0.40,                # WD: 無此機制 | 沿固定路徑巡邏，可預測週期動態
                "random_walk": 0.20,           # WD: 無此機制 | 隨機方向移動，不規則時序干擾
            },
            "speed_overrides": {               # WD: 無此機制 | per-behavior [min, max] 速度範圍覆寫
                "patrol": {"speed_range": (0.20, 0.40)},       # 稍提速，鞏固動態感知
                "random_walk": {"speed_range": (0.15, 0.40)},  # 上限提高，增加不確定性
            },
        },

        # ── trainer ──
        # WD 全 phase 固定超參，不做 per-phase 微調。
        "trainer": {
            "lr": 2e-4,                        # WD 全 phase 固定 2e-4 | RL policy/value head 學習率
            "rnn_lr": 5e-4,                    # WD 全 phase 固定 5e-4 | RNN / aux optimizer 學習率
            "vf_coeff": 0.5,                   # WD 全 phase 固定 0.025 | value loss 權重係數
            "max_grad_norm": 1.0,              # WD 全 phase 固定 1.0 | RL 全局梯度裁剪上限
            "aux_grad_clip": 0.5,              # WD: 無獨立 aux clip | aux/RNN 專用梯度裁剪
            "wd_actor_update_clip": 8.0,       # WD 全 phase 固定 8.0 | actor grad norm cap
            "wd_critic_update_clip": 30.0,     # WD 全 phase 固定 30.0 | critic grad norm cap
        },

        # ── transition ──
        # WD 用固定 iteration budget 升階，無 performance 門檻。
        "transition": {
            "upgrade_sr": 0.65,                # WD: 無 SR 門檻 | 成功率 ≥ 此值才升階；比 SA1 低因任務更難
            "upgrade_max_cr": 0.40,            # WD: 無 CR 門檻 | 碰撞率 ≤ 此值；開始限制碰撞
            "upgrade_max_to": 0.30,            # WD: 無 TO 門檻 | 超時率 ≤ 此值
            "upgrade_min_dyn_sr": 0.0,         # WD: 無此指標 | 動態 env SR 下限；0.0=不限制
            "min_stage_updates": 65,           # WD: 用固定 batch budget | 最少停留更新次數
            "downgrade_sr": 0.15,              # WD: 無降階機制 | SR < 此值觸發降階
            "downgrade_min_cr": 0.70,          # WD: 無降階機制 | CR > 此值觸發降階
            "downgrade_min_to": 0.65,          # WD: 無降階機制 | TO > 此值觸發降階
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 3 / SA3
    # 【訓練能力】導航精確度 + 避障過渡
    #   論文策略：goal 驟降到 1 + episode 延長到 90s → p(goal) 降低、p(obs) 上升
    #   penalty -12 大幅提高避障代價；episode 延長讓 agent 有更多時間遇到障礙
    #   論文指出：延長 episode = 提升避障能力的需求機率（agent 可等安全時機再走）
    #   entropy 再提高(0.50) → 高探索維持，避免過早收斂到保守策略
    #
    # ── WD P3 參考 ──
    # spots=3, goals=1, obs=2, penalty=-12, cost=0.0,
    # ent_linear=0.50, ent_angular=0.375, speed=0.85, 90s, batch=105K
    # wall=0
    #
    # ── 訓練預算建議 (run-level，不由本檔自動套用) ──
    # rollout_length=300 固定；timesteps = updates × rollout_length；
    # interactions = num_envs × rollout_length × updates。
    # Sanity: num_envs=1024, timesteps=30000,  updates=100,  interactions=30.7M
    # Formal: num_envs=1024, timesteps=270000, updates=900,  interactions=276.5M
    # Long:   num_envs=1024, timesteps=450000, updates=1500, interactions=460.8M
    # 說明：SA3 開始拉高避障/timing 需求；update 不足會看不到策略轉變。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA3_dyn_intro",

        "scene": {
            "goals": 3,                         # WD P3=1
            "goal_distance": (2.0, 8.0),       # 導航距離再拉遠
            "static_obstacles": 3,             # WD P3: 無 static
            "dynamic_obstacles": 5,            # WD P3=2
            "dynamic_obstacles_min": 2,        # 每 env 至少 2 個動態障礙
            "walls_min": 1,                    # WD P3: 無牆
            "walls_max": 2,                    # WD P3: 無牆
            "wall_length": 4.5,                # WD P3: 無牆
            "episode_s": 90,                   # WD P3=90
            "gamma": 0.984,                    # 與 P1/P2 統一，WD constant gamma
        },

        "reward": {
            "penalty_hit": -12.0,               # WD P3=-12
            "reward_get_goal": 40.0,           # WD P3=40
            "cost_operate": 0.0,               # WD P3=0.0
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.01,            # WD P3=0.50
            "entropy_angular": 0.02,          # WD P3=0.375
        },

        # ── behavior ──
        # WD P3 只有全局 speed=0.85，無 behavior_mix / speed_overrides。
        "behavior": {
            "obstacle_speed": 0.85,            # WD P3=0.85 | 全局動態障礙速度倍率
            "behavior_mix": {
                "static": 0.20,                # WD: 無此機制 | 固定不動，保留靜態幾何辨識
                "patrol": 0.35,                # WD: 無此機制 | 沿路徑巡邏，可預測動態仍是主體
                "random_walk": 0.20,           # WD: 無此機制 | 隨機方向移動，不規則干擾
                "horizontal_crossing": 0.15,   # WD: 無此機制 | 橫向穿越 robot 路徑，訓練 timing 判斷
                "path_crossing": 0.10,         # WD: 無此機制 | 沿 robot→goal 方向交叉穿越，訓練近距離反應
            },
            "speed_overrides": {               # WD: 無此機制 | per-behavior [min, max] 速度範圍覆寫
                "patrol": {"speed_range": (0.25, 0.55)},              # 中速巡邏
                "random_walk": {"speed_range": (0.20, 0.55)},         # 中速隨機
                "horizontal_crossing": {"speed_range": (0.35, 0.65)}, # 較快橫穿，測試反應
                "path_crossing": {"speed_range": (0.25, 0.60)},       # 路徑交叉速度
            },
        },

        # ── trainer ──
        # WD 全 phase 固定超參，不做 per-phase 微調。
        "trainer": {
            "lr": 2e-4,                        # WD 全 phase 固定 2e-4 | RL policy/value head 學習率
            "rnn_lr": 5e-4,                    # WD 全 phase 固定 5e-4 | RNN / aux optimizer 學習率
            "vf_coeff": 0.5,                   # WD 全 phase 固定 0.025 | value loss 權重係數
            "max_grad_norm": 1.0,              # WD 全 phase 固定 1.0 | RL 全局梯度裁剪上限
            "aux_grad_clip": 0.5,              # WD: 無獨立 aux clip | aux/RNN 專用梯度裁剪
            "wd_actor_update_clip": 8.0,       # WD 全 phase 固定 8.0 | actor grad norm cap
            "wd_critic_update_clip": 30.0,     # WD 全 phase 固定 30.0 | critic grad norm cap
        },

        # ── transition ──
        # WD 用固定 iteration budget 升階，無 performance 門檻。
        "transition": {
            "upgrade_sr": 0.65,                # WD: 無 SR 門檻 | 成功率 ≥ 此值才升階
            "upgrade_max_cr": 0.40,            # WD: 無 CR 門檻 | 碰撞率 ≤ 此值
            "upgrade_max_to": 0.30,            # WD: 無 TO 門檻 | 超時率 ≤ 此值
            "upgrade_min_dyn_sr": 0.35,        # WD: 無此指標 | 動態 env SR 下限；開始要求避障能力
            "min_stage_updates": 80,           # WD: 用固定 batch budget | 最少停留更新次數
            "downgrade_sr": 0.15,              # WD: 無降階機制 | SR < 此值觸發降階
            "downgrade_min_cr": 0.70,          # WD: 無降階機制 | CR > 此值觸發降階
            "downgrade_min_to": 0.65,          # WD: 無降階機制 | TO > 此值觸發降階
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 4 / SA4
    # 【訓練能力】空間規劃（牆壁引入）
    #   論文策略：引入牆壁 = 新環境參數，改變「空間規劃」能力的需求機率
    #   penalty 維持 -12（不再靠 reward 加壓，而是靠環境結構改變能力需求）
    #   牆壁迫使 agent 學會繞路、預判路徑，而非直線衝向 goal
    #   episode 縮回 60s → 在有牆環境下仍要求效率
    #
    # ── WD P4 參考 ──
    # spots=3, goals=1, obs=2, penalty=-12, cost=0.0,
    # ent_linear=0.50, ent_angular=0.375, speed=0.85, 60s, batch=105K
    # wall=1/4.5m
    #
    # ── 訓練預算建議 (run-level，不由本檔自動套用) ──
    # rollout_length=300 固定；timesteps = updates × rollout_length；
    # interactions = num_envs × rollout_length × updates。
    # Sanity: num_envs=1024, timesteps=30000,  updates=100,  interactions=30.7M
    # Formal: num_envs=1024, timesteps=270000, updates=900,  interactions=276.5M
    # Long:   num_envs=1024, timesteps=450000, updates=1500, interactions=460.8M
    # 說明：SA4 加牆與繞路；先保持 1024 env，避免 batch 過大讓路徑訊號被平均。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA4_nav_avoid",

        "scene": {
            "goals": 4,                         # WD P4=1
            "goal_distance": (2.0, 9.0),       # 更長距離導航
            "static_obstacles": 0,             # WD P4: 無 static
            "dynamic_obstacles": 4,            # WD P4=2
            "dynamic_obstacles_min": 3,        # 每 env 至少 3 個動態
            "walls_min": 1,                    # WD P4=1
            "walls_max": 2,                    # WD P4=1
            "wall_length": 4.0,                # WD P4=4.5m
            "episode_s": 60,                   # WD P4=60
            "gamma": 0.994,
        },

        "reward": {
            "penalty_hit": -12.0,              # WD P4=-12（一致）
            "reward_get_goal": 40.0,           # WD P4=40
            "cost_operate": 0.0,               # WD P4=0.0
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.30,            # WD P4=0.50
            "entropy_angular": 0.375,          # WD P4=0.375
        },

        # ── behavior ──
        # WD P4 只有全局 speed=0.85，無 behavior_mix / speed_overrides。
        "behavior": {
            "obstacle_speed": 0.85,            # WD P4=0.85 | 全局動態障礙速度倍率
            "behavior_mix": {
                "patrol": 0.25,                # WD: 無此機制 | 沿路徑巡邏，可預測動態
                "random_walk": 0.15,           # WD: 無此機制 | 隨機方向移動，不規則干擾
                "horizontal_crossing": 0.20,   # WD: 無此機制 | 橫向穿越 robot 路徑，訓練 timing
                "path_crossing": 0.15,         # WD: 無此機制 | 沿 robot→goal 交叉穿越，近距離反應
                "near_miss": 0.15,             # WD: 無此機制 | 高速擦身而過，訓練精確空間判斷
                "corridor_crossing": 0.10,     # WD: 無此機制 | 狹窄通道中穿越，受限空間避障
            },
            "speed_overrides": {               # WD: 無此機制 | per-behavior [min, max] 速度範圍覆寫
                "patrol": {"speed_range": (0.30, 0.65)},              # 中高速巡邏
                "random_walk": {"speed_range": (0.25, 0.60)},         # 中速隨機
                "horizontal_crossing": {"speed_range": (0.40, 0.80)}, # 較快橫穿
                "path_crossing": {"speed_range": (0.35, 0.75)},       # 中高速路徑交叉
                "near_miss": {"speed_range": (0.35, 0.75)},           # 中高速擦身
                "corridor_crossing": {"speed_range": (0.25, 0.55)},   # 低速窄道穿越
            },
        },

        # ── trainer ──
        # WD 全 phase 固定超參。SA4 開始微降 rnn_lr 以穩定 representation。
        "trainer": {
            "lr": 2e-4,                        # WD 全 phase 固定 2e-4 | RL policy/value head 學習率
            "rnn_lr": 4e-4,                    # WD 全 phase 固定 5e-4 | RNN 學習率；SA4 降至 4e-4 減少 representation drift
            "vf_coeff": 0.5,                   # WD 全 phase 固定 0.025 | value loss 權重係數
            "max_grad_norm": 1.0,              # WD 全 phase 固定 1.0 | RL 全局梯度裁剪上限
            "aux_grad_clip": 0.5,              # WD: 無獨立 aux clip | aux/RNN 專用梯度裁剪
            "wd_actor_update_clip": 8.0,       # WD 全 phase 固定 8.0 | actor grad norm cap
            "wd_critic_update_clip": 30.0,     # WD 全 phase 固定 30.0 | critic grad norm cap
        },

        # ── transition ──
        # WD 用固定 iteration budget 升階，無 performance 門檻。
        "transition": {
            "upgrade_sr": 0.65,                # WD: 無 SR 門檻 | 成功率 ≥ 此值才升階
            "upgrade_max_cr": 0.40,            # WD: 無 CR 門檻 | 碰撞率 ≤ 此值
            "upgrade_max_to": 0.30,            # WD: 無 TO 門檻 | 超時率 ≤ 此值
            "upgrade_min_dyn_sr": 0.35,        # WD: 無此指標 | 動態 env SR 下限
            "min_stage_updates": 95,           # WD: 用固定 batch budget | 最少停留更新次數
            "downgrade_sr": 0.15,              # WD: 無降階機制 | SR < 此值觸發降階
            "downgrade_min_cr": 0.70,          # WD: 無降階機制 | CR > 此值觸發降階
            "downgrade_min_to": 0.65,          # WD: 無降階機制 | TO > 此值觸發降階
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 5 / SA5
    # 【訓練能力】耐久導航 + 空間規劃強化
    #   論文策略：episode 延長到 100s + 牆更長(5.0m) → 結合長時間與複雜結構
    #   penalty 維持 -12 不變，但更長 episode = 更多碰撞機會 = p(obs) 自然上升
    #   spots 降到 2 → 被 virtual spots 干擾的機率下降，focus 在真實任務
    #
    # ── WD P5 參考 ──
    # spots=2, goals=1, obs=2, penalty=-12, cost=0.0,
    # ent_linear=0.50, ent_angular=0.375, speed=0.85, 100s, batch=168K
    # wall=1/5.0m
    #
    # ── 訓練預算建議 (run-level，不由本檔自動套用) ──
    # rollout_length=300 固定；timesteps = updates × rollout_length；
    # interactions = num_envs × rollout_length × updates。
    # Sanity: num_envs=1024, timesteps=30000,  updates=100,  interactions=30.7M
    # Formal: num_envs=1024, timesteps=270000, updates=900,  interactions=276.5M
    # Long:   num_envs=1024, timesteps=450000, updates=1500, interactions=460.8M
    # 說明：SA5 是中高難度整合；若 1024 env VRAM/throughput 正常，可另做 1536 env 對照。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA5_medium",

        "scene": {
            "goals": 2,                         # WD P5=1
            "goal_distance": (2.0, 10.0),      # 長距離導航
            "static_obstacles": 0,             # WD P5: 無 static
            "dynamic_obstacles": 6,            # WD P5=2（SA5 加更多）
            "dynamic_obstacles_min": 4,        # 每 env 至少 4 個動態
            "walls_min": 1,                    # WD P5=1
            "walls_max": 2,                    # WD P5=1
            "wall_length": 3.5,                # WD P5=5.0m
            "episode_s": 90,                   # WD P5=100
            "gamma": 0.995,
        },

        "reward": {
            "penalty_hit": -15.0,              # WD P5=-12
            "reward_get_goal": 40.0,           # WD P5=40
            "cost_operate": 0.0,               # WD P5=0.0
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.30,            # WD P5=0.50
            "entropy_angular": 0.375,          # WD P5=0.375
        },

        # ── behavior ──
        # WD P5 只有全局 speed=0.85，無 behavior_mix / speed_overrides。
        # SA5 引入完整 7-behavior 分佈，含 occlusion 模擬遮蔽後突現。
        "behavior": {
            "obstacle_speed": 0.85,            # WD P5=0.85 | 全局動態障礙速度倍率
            "behavior_mix": {
                "patrol": 0.15,                # WD: 無此機制 | 沿路徑巡邏，可預測動態
                "random_walk": 0.15,           # WD: 無此機制 | 隨機方向移動，不規則干擾
                "horizontal_crossing": 0.15,   # WD: 無此機制 | 橫向穿越 robot 路徑，訓練 timing
                "path_crossing": 0.15,         # WD: 無此機制 | 沿 robot→goal 交叉穿越，近距離反應
                "near_miss": 0.15,             # WD: 無此機制 | 高速擦身而過，訓練精確空間判斷
                "corridor_crossing": 0.10,     # WD: 無此機制 | 狹窄通道中穿越，受限空間避障
                "occlusion": 0.15,             # WD: 無此機制 | 從牆/障礙物後方突然出現，訓練遮蔽反應
            },
            "speed_overrides": {               # WD: 無此機制 | per-behavior [min, max] 速度範圍覆寫
                "patrol": {"speed_range": (0.35, 0.75)},              # 中高速巡邏
                "random_walk": {"speed_range": (0.30, 0.75)},         # 中高速隨機
                "horizontal_crossing": {"speed_range": (0.45, 0.85)}, # 快速橫穿
                "path_crossing": {"speed_range": (0.40, 0.80)},       # 快速路徑交叉
                "near_miss": {"speed_range": (0.40, 0.80)},           # 快速擦身
                "corridor_crossing": {"speed_range": (0.30, 0.65)},   # 中速窄道穿越
                "occlusion": {"speed_range": (0.25, 0.60)},           # 中低速突現（給 agent 些微反應時間）
            },
        },

        # ── trainer ──
        # WD 全 phase 固定超參。SA5 開始降低 LR + 收緊 grad clip，穩定高難 phase 訓練。
        "trainer": {
            "lr": 1.5e-4,                      # WD 全 phase 固定 2e-4 | RL 學習率；SA5 降至 1.5e-4 減少 policy 更新過猛
            "rnn_lr": 3e-4,                    # WD 全 phase 固定 5e-4 | RNN 學習率；SA5 降至 3e-4 降低 feature drift
            "vf_coeff": 0.5,                   # WD 全 phase 固定 0.025 | value loss 權重係數
            "max_grad_norm": 0.8,              # WD 全 phase 固定 1.0 | RL 梯度裁剪；SA5 收緊至 0.8
            "aux_grad_clip": 0.4,              # WD: 無獨立 aux clip | aux/RNN 梯度裁剪；SA5 收緊至 0.4
            "wd_actor_update_clip": 8.0,       # WD 全 phase 固定 8.0 | actor grad norm cap
            "wd_critic_update_clip": 30.0,     # WD 全 phase 固定 30.0 | critic grad norm cap
        },

        # ── transition ──
        # WD 用固定 iteration budget 升階，無 performance 門檻。
        "transition": {
            "upgrade_sr": 0.60,                # WD: 無 SR 門檻 | 成功率 ≥ 此值才升階
            "upgrade_max_cr": 0.40,            # WD: 無 CR 門檻 | 碰撞率 ≤ 此值
            "upgrade_max_to": 0.30,            # WD: 無 TO 門檻 | 超時率 ≤ 此值
            "upgrade_min_dyn_sr": 0.30,        # WD: 無此指標 | 動態 env SR 下限
            "min_stage_updates": 110,          # WD: 用固定 batch budget | 最少停留更新次數
            "downgrade_sr": 0.15,              # WD: 無降階機制 | SR < 此值觸發降階
            "downgrade_min_cr": 0.70,          # WD: 無降階機制 | CR > 此值觸發降階
            "downgrade_min_to": 0.65,          # WD: 無降階機制 | TO > 此值觸發降階
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 6 / SA6
    # 【訓練能力】避障（核心訓練階段）
    #   論文策略：obs 從 2 暴增到 10 + episode 大幅延長到 210s
    #   → p(obs) 劇增，避障成為主要學習信號
    #   論文指出「延長 episode 使得避障的重要性提高」：agent 可等待安全時機
    #   penalty -15 適度提高，但主要靠環境參數（obs 數量 × episode 長度）驅動
    #   牆壁暫時移除 → 讓 agent 專注學「動態避障」而非「空間規劃」
    #
    # ── WD P6 參考 ──
    # spots=3, goals=1, obs=10, penalty=-15, cost=0.0,
    # ent_linear=0.50, ent_angular=0.375, speed=0.85, 210s, batch=105K
    # wall=0
    #
    # ── 訓練預算建議 (run-level，不由本檔自動套用) ──
    # rollout_length=300 固定；timesteps = updates × rollout_length；
    # interactions = num_envs × rollout_length × updates。
    # Sanity: num_envs=1024, timesteps=30000,  updates=100,  interactions=30.7M
    # Formal: num_envs=1024, timesteps=450000, updates=1500, interactions=460.8M
    # Stretch:num_envs=1536, timesteps=450000, updates=1500, interactions=691.2M
    # 說明：SA6 是核心 dense avoidance；可提高 interaction，但仍需保留足夠 update 次數。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA6_dense_avoid",

        "scene": {
            "goals": 1,                         # WD P6=1
            "goal_distance": (3.0, 10.0),      # 長距離，靠導航能力存量
            "static_obstacles": 0,             # WD P6: 無 static
            "dynamic_obstacles": 10,           # WD P6=10（大量動態）
            "dynamic_obstacles_min": 8,        # 每 env 至少 8 個
            "walls_min": 0,                    # WD P6: 無牆（專注動態避障）
            "walls_max": 0,                    # WD P6: 無牆
            "wall_length": 0.0,
            "episode_s": 210,                  # WD P6=210（大幅延長）
            "gamma": 0.996,
        },

        "reward": {
            "penalty_hit": -15.0,              # WD P6=-15
            "reward_get_goal": 40.0,           # WD P6=40
            "cost_operate": 0.0,               # WD P6=0.0
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.50,            # WD P6=0.50
            "entropy_angular": 0.375,          # WD P6=0.375
        },

        # ── behavior ──
        # WD P6 只有全局 speed=0.85，無 behavior_mix / speed_overrides。
        # SA6 沿用 SA5 的 7-behavior 分佈，配合 10 個動態障礙的 dense 場景。
        "behavior": {
            "obstacle_speed": 0.85,            # WD P6=0.85 | 全局動態障礙速度倍率
            "behavior_mix": {
                "patrol": 0.15,                # WD: 無此機制 | 沿路徑巡邏，可預測動態
                "random_walk": 0.15,           # WD: 無此機制 | 隨機方向移動，不規則干擾
                "horizontal_crossing": 0.15,   # WD: 無此機制 | 橫向穿越 robot 路徑，訓練 timing
                "path_crossing": 0.15,         # WD: 無此機制 | 沿 robot→goal 交叉穿越，近距離反應
                "near_miss": 0.15,             # WD: 無此機制 | 高速擦身而過，精確空間判斷
                "corridor_crossing": 0.10,     # WD: 無此機制 | 狹窄通道穿越，受限空間避障
                "occlusion": 0.15,             # WD: 無此機制 | 遮蔽後突現，訓練對隱藏障礙的反應
            },
            "speed_overrides": {               # WD: 無此機制 | per-behavior [min, max] 速度範圍覆寫
                "patrol": {"speed_range": (0.35, 0.75)},              # 中高速巡邏
                "random_walk": {"speed_range": (0.30, 0.75)},         # 中高速隨機
                "horizontal_crossing": {"speed_range": (0.45, 0.85)}, # 快速橫穿
                "path_crossing": {"speed_range": (0.40, 0.80)},       # 快速路徑交叉
                "near_miss": {"speed_range": (0.40, 0.80)},           # 快速擦身
                "corridor_crossing": {"speed_range": (0.30, 0.65)},   # 中速窄道穿越
                "occlusion": {"speed_range": (0.25, 0.60)},           # 中低速突現
            },
        },

        # ── trainer ──
        # WD 全 phase 固定超參。SA6 沿用 SA5 的降低 LR + 收緊 grad clip。
        "trainer": {
            "lr": 1.5e-4,                      # WD 全 phase 固定 2e-4 | RL 學習率；SA6 降至 1.5e-4
            "rnn_lr": 3e-4,                    # WD 全 phase 固定 5e-4 | RNN 學習率；SA6 降至 3e-4
            "vf_coeff": 0.5,                   # WD 全 phase 固定 0.025 | value loss 權重係數
            "max_grad_norm": 0.8,              # WD 全 phase 固定 1.0 | RL 梯度裁剪；收緊至 0.8
            "aux_grad_clip": 0.4,              # WD: 無獨立 aux clip | aux/RNN 梯度裁剪；收緊至 0.4
            "wd_actor_update_clip": 8.0,       # WD 全 phase 固定 8.0 | actor grad norm cap
            "wd_critic_update_clip": 30.0,     # WD 全 phase 固定 30.0 | critic grad norm cap
        },

        # ── transition ──
        # WD 用固定 iteration budget 升階，無 performance 門檻。
        "transition": {
            "upgrade_sr": 0.55,                # WD: 無 SR 門檻 | 成功率 ≥ 此值才升階；dense 場景放寬至 0.55
            "upgrade_max_cr": 0.35,            # WD: 無 CR 門檻 | 碰撞率 ≤ 此值；收緊至 0.35
            "upgrade_max_to": 0.35,            # WD: 無 TO 門檻 | 超時率 ≤ 此值；210s episode 放寬至 0.35
            "upgrade_min_dyn_sr": 0.40,        # WD: 無此指標 | 動態 env SR 下限；SA6 核心是動態避障，提高至 0.40
            "min_stage_updates": 120,          # WD: 用固定 batch budget | 最少停留更新次數
            "downgrade_sr": 0.10,              # WD: 無降階機制 | SR < 此值觸發降階
            "downgrade_min_cr": 0.75,          # WD: 無降階機制 | CR > 此值觸發降階
            "downgrade_min_to": 0.70,          # WD: 無降階機制 | TO > 此值觸發降階
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 7 / SA7
    # 【訓練能力】高壓綜合（避障 + 空間規劃 + 高速動態）
    #   論文策略：penalty 暴增到 -85 + obs=10 + wall=2 + speed 提升到 1.10
    #   → 這是「最終 reward 評估」的前置階段
    #   所有環境參數同時加壓：obs 多、速度快、有牆、penalty 高
    #   論文：「只有在逐步提升引導的情境下，才適合引入最終的 Reward 評估」
    #   此 phase 就是正式引入高 penalty 的階段
    #
    # ── WD P7 參考 ──
    # spots=2, goals=1, obs=10, penalty=-85, cost=0.0,
    # ent_linear=0.50, ent_angular=0.375, speed=1.10, 210s, batch=168K
    # wall=2/3.5m
    #
    # ── 訓練預算建議 (run-level，不由本檔自動套用) ──
    # rollout_length=300 固定；timesteps = updates × rollout_length；
    # interactions = num_envs × rollout_length × updates。
    # Sanity: num_envs=512,  timesteps=30000,  updates=100,  interactions=15.4M
    # Formal: num_envs=1024, timesteps=450000, updates=1500, interactions=460.8M
    # Safe:   num_envs=512,  timesteps=450000, updates=1500, interactions=230.4M
    # 說明：SA7 高 penalty/高速/牆壁同時加壓；若 CR/RNN 不穩，先用 512 env 降低事件密度。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA7_high_pressure",

        "scene": {
            "goals": 1,                         # WD P7=1
            "goal_distance": (3.0, 10.0),      # 長距離
            "static_obstacles": 0,             # WD P7: 無 static
            "dynamic_obstacles": 10,           # WD P7=10
            "dynamic_obstacles_min": 8,        # 每 env 至少 8 個
            "walls_min": 2,                    # WD P7=2
            "walls_max": 2,                    # WD P7=2
            "wall_length": 3.5,                # WD P7=3.5m
            "episode_s": 210,                  # WD P7=210
            "gamma": 0.997,
        },

        "reward": {
            "penalty_hit": -85.0,              # WD P7=-85（高壓）
            "reward_get_goal": 40.0,           # WD P7=40
            "cost_operate": 0.0,               # WD P7=0.0
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.50,            # WD P7=0.50
            "entropy_angular": 0.375,          # WD P7=0.375
        },

        # ── behavior ──
        # WD P7 speed 從 0.85 跳增到 1.10，為高壓綜合測試。無 behavior_mix。
        # SA7 near_miss 比例提至 0.20，強化高速擦身能力。
        "behavior": {
            "obstacle_speed": 1.10,            # WD P7=1.10 | 全局動態障礙速度倍率；高速壓力測試
            "behavior_mix": {
                "patrol": 0.10,                # WD: 無此機制 | 沿路徑巡邏；高壓 phase 降低可預測比例
                "random_walk": 0.10,           # WD: 無此機制 | 隨機方向移動；降低低威脅比例
                "horizontal_crossing": 0.15,   # WD: 無此機制 | 橫向穿越，訓練高速 timing
                "path_crossing": 0.15,         # WD: 無此機制 | 路徑交叉穿越，近距離反應
                "near_miss": 0.20,             # WD: 無此機制 | 高速擦身；比例最高，核心高壓訓練
                "corridor_crossing": 0.15,     # WD: 無此機制 | 狹窄通道穿越，受限空間+牆壁
                "occlusion": 0.15,             # WD: 無此機制 | 遮蔽後突現
            },
            "speed_overrides": {               # WD: 無此機制 | per-behavior [min, max] 速度範圍覆寫
                "patrol": {"speed_range": (0.45, 0.90)},              # 高速巡邏
                "random_walk": {"speed_range": (0.40, 0.85)},         # 高速隨機
                "horizontal_crossing": {"speed_range": (0.55, 1.00)}, # 接近 v_max 橫穿
                "path_crossing": {"speed_range": (0.50, 0.95)},       # 接近 v_max 路徑交叉
                "near_miss": {"speed_range": (0.50, 0.95)},           # 高速擦身
                "corridor_crossing": {"speed_range": (0.40, 0.75)},   # 中高速窄道穿越
                "occlusion": {"speed_range": (0.35, 0.75)},           # 中速突現
            },
        },

        # ── trainer ──
        # WD 全 phase 固定超參。SA7 進一步降低 LR + 收緊 grad clip，適應高 penalty 環境。
        "trainer": {
            "lr": 1e-4,                        # WD 全 phase 固定 2e-4 | RL 學習率；SA7 降至 1e-4 穩定高壓訓練
            "rnn_lr": 2e-4,                    # WD 全 phase 固定 5e-4 | RNN 學習率；SA7 降至 2e-4 防止 drift
            "vf_coeff": 0.5,                   # WD 全 phase 固定 0.025 | value loss 權重係數
            "max_grad_norm": 0.6,              # WD 全 phase 固定 1.0 | RL 梯度裁剪；SA7 收緊至 0.6
            "aux_grad_clip": 0.3,              # WD: 無獨立 aux clip | aux/RNN 梯度裁剪；SA7 收緊至 0.3
            "wd_actor_update_clip": 8.0,       # WD 全 phase 固定 8.0 | actor grad norm cap
            "wd_critic_update_clip": 30.0,     # WD 全 phase 固定 30.0 | critic grad norm cap
        },

        # ── transition ──
        # WD 用固定 iteration budget 升階，無 performance 門檻。
        "transition": {
            "upgrade_sr": 0.50,                # WD: 無 SR 門檻 | 成功率 ≥ 此值才升階；高壓放寬至 0.50
            "upgrade_max_cr": 0.30,            # WD: 無 CR 門檻 | 碰撞率 ≤ 此值；penalty=-85 需更嚴格
            "upgrade_max_to": 0.35,            # WD: 無 TO 門檻 | 超時率 ≤ 此值
            "upgrade_min_dyn_sr": 0.35,        # WD: 無此指標 | 動態 env SR 下限
            "min_stage_updates": 130,          # WD: 用固定 batch budget | 最少停留更新次數
            "downgrade_sr": 0.10,              # WD: 無降階機制 | SR < 此值觸發降階
            "downgrade_min_cr": 0.80,          # WD: 無降階機制 | CR > 此值觸發降階
            "downgrade_min_to": 0.75,          # WD: 無降階機制 | TO > 此值觸發降階
        },
    },

    # ──────────────────────────────────────────────────────────────────────
    # Stage 8 / SA8
    # 【訓練能力】最終績效評估級（全能力整合）
    #   論文策略：penalty=-100 是最終評估標準
    #   「若一開始即以最終期望值為訓練標準，Agent 難以達到最佳效果」
    #   經過 P1-P7 逐步引導後，此處才引入最終 penalty
    #   obs 降到 6（從 10 降回）+ speed 再提升 1.15 → 少但更快更難預測
    #   wall=2 維持空間規劃需求，episode=210s 維持耐久需求
    #
    # ── WD P8 參考 ──
    # spots=2, goals=1, obs=6, penalty=-100, cost=0.0,
    # ent_linear=0.50, ent_angular=0.375, speed=1.15, 210s, batch=168K
    # wall=2/3.5m
    #
    # ── 訓練預算建議 (run-level，不由本檔自動套用) ──
    # rollout_length=300 固定；timesteps = updates × rollout_length；
    # interactions = num_envs × rollout_length × updates。
    # Sanity: num_envs=512,  timesteps=30000,  updates=100,  interactions=15.4M
    # Formal: num_envs=1024, timesteps=450000, updates=1500, interactions=460.8M
    # Eval:   訓練後用 deterministic play/eval 檢查 1-goal、CR、TO、heading、gV。
    # 說明：SA8 是最終評估級；不要用過大 num_envs 掩蓋單環境失敗模式。
    # ──────────────────────────────────────────────────────────────────────
    {
        "name": "SA8_final",

        "scene": {
            "goals": 1,                         # WD P8=1
            "goal_distance": (3.0, 10.0),      # 長距離
            "static_obstacles": 0,             # WD P8: 無 static
            "dynamic_obstacles": 6,            # WD P8=6（少但快）
            "dynamic_obstacles_min": 5,        # 每 env 至少 5 個
            "walls_min": 2,                    # WD P8=2
            "walls_max": 2,                    # WD P8=2
            "wall_length": 3.5,                # WD P8=3.5m
            "episode_s": 210,                  # WD P8=210
            "gamma": 0.998,
        },

        "reward": {
            "penalty_hit": -100.0,             # WD P8=-100（最終評估級）
            "reward_get_goal": 40.0,           # WD P8=40
            "cost_operate": 0.0,               # WD P8=0.0
            "reward_weights": None,
        },

        "exploration": {
            "entropy_linear": 0.50,            # WD P8=0.50
            "entropy_angular": 0.375,          # WD P8=0.375
        },

        # ── behavior ──
        # WD P8 speed 達最高 1.15，obstacle 數降到 6（少但快更難預測）。無 behavior_mix。
        # SA8 沿用 SA7 的 behavior 分佈，但速度範圍全面提升。
        "behavior": {
            "obstacle_speed": 1.15,            # WD P8=1.15 | 全局動態障礙速度倍率；最高速，最終評估級
            "behavior_mix": {
                "patrol": 0.10,                # WD: 無此機制 | 沿路徑巡邏；最終 phase 降低可預測比例
                "random_walk": 0.10,           # WD: 無此機制 | 隨機方向移動
                "horizontal_crossing": 0.15,   # WD: 無此機制 | 橫向穿越，高速 timing
                "path_crossing": 0.15,         # WD: 無此機制 | 路徑交叉穿越
                "near_miss": 0.20,             # WD: 無此機制 | 高速擦身；最高比例，核心挑戰
                "corridor_crossing": 0.15,     # WD: 無此機制 | 狹窄通道穿越
                "occlusion": 0.15,             # WD: 無此機制 | 遮蔽後突現
            },
            "speed_overrides": {               # WD: 無此機制 | per-behavior [min, max] 速度範圍覆寫
                "patrol": {"speed_range": (0.50, 1.00)},              # 高速巡邏
                "random_walk": {"speed_range": (0.45, 0.95)},         # 高速隨機
                "horizontal_crossing": {"speed_range": (0.60, 1.10)}, # 超過 v_max 橫穿（最高挑戰）
                "path_crossing": {"speed_range": (0.55, 1.00)},       # 接近 v_max 路徑交叉
                "near_miss": {"speed_range": (0.55, 1.00)},           # 高速擦身
                "corridor_crossing": {"speed_range": (0.45, 0.80)},   # 中高速窄道穿越
                "occlusion": {"speed_range": (0.40, 0.80)},           # 中速突現
            },
        },

        # ── trainer ──
        # WD 全 phase 固定超參。SA8 用最保守的 LR 和最緊 grad clip，穩定最終評估訓練。
        "trainer": {
            "lr": 1e-4,                        # WD 全 phase 固定 2e-4 | RL 學習率；SA8 降至 1e-4 微調
            "rnn_lr": 2e-4,                    # WD 全 phase 固定 5e-4 | RNN 學習率；SA8 降至 2e-4
            "vf_coeff": 0.5,                   # WD 全 phase 固定 0.025 | value loss 權重係數
            "max_grad_norm": 0.5,              # WD 全 phase 固定 1.0 | RL 梯度裁剪；SA8 最緊 0.5
            "aux_grad_clip": 0.3,              # WD: 無獨立 aux clip | aux/RNN 梯度裁剪；SA8 收緊至 0.3
            "wd_actor_update_clip": 8.0,       # WD 全 phase 固定 8.0 | actor grad norm cap
            "wd_critic_update_clip": 30.0,     # WD 全 phase 固定 30.0 | critic grad norm cap
        },

        # ── transition ──
        # WD 用固定 iteration budget 升階。SA8 是最終 phase，升階條件設為不可能達到。
        "transition": {
            "upgrade_sr": 1.0,                 # WD: 無 SR 門檻 | 成功率門檻；1.0=不可能升階（最終 phase）
            "upgrade_max_cr": 0.0,             # WD: 無 CR 門檻 | 碰撞率門檻；0.0=不可能升階
            "upgrade_max_to": 0.0,             # WD: 無 TO 門檻 | 超時率門檻；0.0=不可能升階
            "upgrade_min_dyn_sr": 1.0,         # WD: 無此指標 | 動態 env SR 下限；1.0=不可能升階
            "min_stage_updates": 9999,         # WD: 用固定 batch budget | 設極大值鎖定在最終 phase
            "downgrade_sr": 0.08,              # WD: 無降階機制 | SR < 此值觸發降階；仍允許降回 SA7
            "downgrade_min_cr": 0.85,          # WD: 無降階機制 | CR > 此值觸發降階
            "downgrade_min_to": 0.80,          # WD: 無降階機制 | TO > 此值觸發降階
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# Legacy bridge
# ═══════════════════════════════════════════════════════════════════════════
#
# 目的：
# - 人類在上面使用巢狀 schema 編寫 task。
# - curriculum / training 舊程式碼仍偏好 flat dict。
# - 所以這個 bridge 會把巢狀 phase 攤平成 legacy 欄位。
#
# 關鍵原則：
# - 只做 schema 轉換，不改變原本訓練架構。
# - 不在這裡做任何 IsaacLab core 行為修改。
# ═══════════════════════════════════════════════════════════════════════════


def _flatten_phase(phase: dict) -> dict:
    """把巢狀 phase/task schema 攤平成 curriculum 既有可用格式。

    回傳結果會被 `CONFIG["stages"]` 使用，
    供 `goal_obstacle_curriculum.py` 與 `train_rnn_car_wdclip.py` 讀取。
    """

    # 若輸入本來就是 flat/legacy schema，直接回傳副本，保留相容性。
    if "scene" not in phase:
        return dict(phase)

    # 取出各區塊，方便下方轉換。
    scene = phase.get("scene", {})
    reward = phase.get("reward", {})
    exploration = phase.get("exploration", {})
    behavior = phase.get("behavior", {})
    trainer = phase.get("trainer", {})
    transition = phase.get("transition", {})

    # 障礙物數量與 ratio 轉換。
    # 既有 curriculum event 需要的是 static_ratio / dynamic_ratio / empty_ratio，
    # 所以這裡從數量推回比例。
    n_s = int(scene.get("static_obstacles", 0))
    n_d = int(scene.get("dynamic_obstacles", 0))
    total = n_s + n_d

    # 若完全沒障礙物，empty_ratio = 1。
    empty = 1.0 if total == 0 else 0.0

    if total > 0:
        # static_ratio = static 佔非空障礙物的比例
        s_ratio = round((1.0 - empty) * n_s / total, 2)

        # dynamic_ratio = 剩餘比例
        d_ratio = round(1.0 - empty - s_ratio, 2)
    else:
        s_ratio = 0.0
        d_ratio = 0.0

    # 主要 flat schema。
    flat = {
        # 基本識別
        "name": phase["name"],

        # scene -> legacy scene fields
        "num_goals": int(scene["goals"]),
        "goal_distance": scene["goal_distance"],
        "num_obstacles_static": n_s,
        "num_obstacles_dynamic": n_d,
        "min_obstacles_dynamic": int(scene.get("dynamic_obstacles_min", n_d)),
        "empty_ratio": empty,
        "static_ratio": s_ratio,
        "dynamic_ratio": d_ratio,
        "gamma": float(scene["gamma"]),
        "episode_length_s": float(scene["episode_s"]),
        "min_walls": int(scene["walls_min"]),
        "max_walls": int(scene["walls_max"]),
        "target_wall_length": float(scene.get("wall_length", 0.0)),

        # reward -> WD sparse reward fields
        "spot_penalty_hit": float(reward.get("penalty_hit", -5.0)),
        "spot_reward_get_goal": float(reward.get("reward_get_goal", 40.0)),
        "spot_cost_operate": float(reward.get("cost_operate", 0.0)),

        # exploration -> entropy fields
        "ent_coeff_linear": float(exploration.get("entropy_linear", 0.30)),
        "ent_coeff_angular": float(exploration.get("entropy_angular", 0.375)),

        # behavior -> obstacle speed / behavior fields
        "obstacle_speed_rate": float(behavior.get("obstacle_speed", 0.8)),

        # 目前單車任務不主用，但保留 legacy 欄位相容性
        "num_virtual_spots": float(scene.get("virtual_spots", 0)),
        "goal_speed_rate": float(scene.get("goal_speed", 0.7)),

        # transition -> curriculum threshold fields
        "upgrade_sr": float(transition["upgrade_sr"]),
        "upgrade_max_cr": float(transition["upgrade_max_cr"]),
        "upgrade_max_to": float(transition["upgrade_max_to"]),
        "upgrade_min_dyn_sr": float(transition.get("upgrade_min_dyn_sr", 0.0)),
        "min_stage_updates": int(transition.get("min_stage_updates", 0)),
        "downgrade_sr": float(transition["downgrade_sr"]),
        "downgrade_min_cr": float(transition["downgrade_min_cr"]),
        "downgrade_min_to": float(transition["downgrade_min_to"]),

        # trainer runtime keys：
        # 這些欄位不直接改 IsaacLab core，只是讓 training loop 能安全讀取並同步。
        "trainer_lr": float(trainer.get("lr", 2e-4)),
        "trainer_rnn_lr": float(trainer.get("rnn_lr", 5e-4)),
        "trainer_vf_coeff": float(trainer.get("vf_coeff", 0.5)),
        "trainer_max_grad_norm": float(trainer.get("max_grad_norm", 1.0)),
        "trainer_aux_grad_clip": float(trainer.get("aux_grad_clip", trainer.get("max_grad_norm", 1.0))),
        "trainer_wd_actor_update_clip": float(trainer.get("wd_actor_update_clip", 8.0)),
        "trainer_wd_critic_update_clip": float(trainer.get("wd_critic_update_clip", 30.0)),
    }

    # 可選欄位：只有存在時才傳遞，避免污染沒有用到的 phase。
    optional_map = {
        # scene 額外隨機化欄位
        "obs_size_rand": scene.get("obs_size_rand"),
        "scene_bound_rand": scene.get("scene_bound_rand"),

        # reward manager 權重表
        "reward_weights": reward.get("reward_weights"),

        # behavior scheduler 相關設定
        "behavior_mix": behavior.get("behavior_mix"),
        "speed_overrides": behavior.get("speed_overrides"),
        "safety_overrides": behavior.get("safety_overrides"),

        # 若未來某 phase 想自定安全同步白名單，可放這裡
        "trainer_sync_allowlist": trainer.get("sync_allowlist"),
    }

    for key, value in optional_map.items():
        if value is not None:
            flat[key] = value

    return flat


# 向後相容：其他 phase 檔案（例如 wd_goal_first.py）仍可能 import 舊名稱 `_to_legacy_stage`。
# 這裡保留同義 alias，避免重構後打斷既有 import chain。
def _to_legacy_stage(phase: dict) -> dict:
    return _flatten_phase(phase)


# 供 phases registry 匯出的標準格式。
CONFIG = {
    # 障礙物控制模式（全局）
    "obstacle_mode": GLOBAL["obstacle_mode"],

    # 全局升階連續通過次數
    "upgrade_pass_required": GLOBAL["upgrade_pass_required"],

    # 升階後是否清空 window
    "clear_window_on_promote": GLOBAL["clear_window_on_promote"],

    # 可安全動態同步的 trainer 欄位白名單
    "trainer_sync_allowlist": GLOBAL["trainer_sync_allowlist"],

    # curriculum 真正使用的 stage 列表（已 flatten）
    "stages": [_flatten_phase(stage) for stage in STAGES],
}
