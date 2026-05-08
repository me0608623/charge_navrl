"""ExperimentConfig 範例模板 — 給 Claude Code / 人類參考的完整註解版。

═══════════════════════════════════════════════════════════════════════════
                    ExperimentConfig 撰寫規則
═══════════════════════════════════════════════════════════════════════════

1. 檔案格式
   - 每個 config 是一個獨立的 .py 檔，放在此目錄下。
   - 檔案必須定義一個名為 CONFIG 的 ExperimentConfig 實例。
   - import 路徑: from rnn_car_modular.experiment_config import ExperimentConfig

2. 命名慣例
   - 檔名: {curriculum}_{stage}_{algorithm}_{特徵}.py
   - 例如: wd_sa2_a2c_aux_lowent.py, wd_sa3_ppo_noaux.py
   - name 欄位必須與檔名一致（不含 .py）。

3. 使用方式
   方法 A — 已註冊（在 registry.py 中加入）:
     --experiment_config wd_sa2_a2c_aux_lowent

   方法 B — 用檔案路徑（不用註冊）:
     --experiment_config /path/to/my_config.py

   CLI 參數會覆蓋 config 值:
     --experiment_config wd_sa2_a2c_aux --num_envs 512 --timesteps 90000

4. 兩層 config 的分工
   ExperimentConfig 控制「整個 run 不變的設定」:
     - 演算法 (A2C / PPO)
     - 網路架構 (extractor_rnn / raw_fc_rnn)
     - Reward function (wd_sparse / 未來 navrl_dense)
     - Aux 訓練 (wd_7d_geometry / none)
     - 訓練預算 (num_envs, timesteps, rollout_length)
     - RL 超參 (lr, gamma, vf_coeff, grad clip)
     - 起始場景 (initial_stage, fixed_stage)

   Phase Config (wd_single_agent_v1.py) 控制「訓練中隨 stage 變化的設定」:
     - 場景難度 (goals, obstacles, walls, episode 長度)
     - 碰撞懲罰 (penalty_hit: -5 → -100)
     - Entropy 自動同步 (若 ExperimentConfig 的 ent_coeff 設 0.0)
     - 障礙物行為分佈 (behavior_mix, speed_overrides)
     - 升降階門檻 (SR/CR/TO thresholds)

5. Entropy 規則
   - ent_coeff_linear / ent_coeff_angular 設為 0.0:
     → Phase Config 自動控制（每個 stage 不同 entropy）
   - 設為非零值（例如 0.05）:
     → 固定值，不隨 stage 變化。用於消融實驗控制變因。

6. Aux 訓練規則
   - aux_profile="wd_7d_geometry" + disable_aux_training=False:
     → 正常 WD 7D geometry aux 訓練。rnn_lr 應 > 0。
   - aux_profile="none" + disable_aux_training=True:
     → 關閉 aux 訓練。rnn_lr 建議設 0.0（凍結 RNN）。
   - 兩者必須一致，否則 validate_profiles() 會報錯。

7. 演算法規則
   - algorithm_profile="a2c_wd" + use_a2c=True:
     → A2C (Warp Drive 預設)。ppo_epochs/clip_eps 不影響。
   - algorithm_profile="ppo_clip" + use_a2c=False:
     → PPO。需設定 ppo_epochs, mini_batches, clip_eps。
   - 兩者必須一致，否則 validate_profiles() 會報錯。

8. Reward 規則（Phase 2 runtime dispatch）
   - reward_profile="wd_sparse":
     → 唯一已實作的 reward。碰撞/goal 獎勵由 Phase Config 動態 sync。
   - reward_profile="navrl_dense" / "hybrid_progress" / "ttc_risk":
     → 尚未實作，會 raise NotImplementedError。
   - 未來新增 reward 只需:
     a. 建立 rnn_car_modular/rewards/<name>.py
     b. 在 rewards/factory.py 加分支
     c. 在此 config 設 reward_profile="<name>"

9. 註冊新 config
   若希望用名字而非路徑呼叫，在 configs/registry.py 加入:
     from rnn_car_modular.configs.my_config import CONFIG as my_config
     EXPERIMENT_CONFIGS["my_config"] = my_config

═══════════════════════════════════════════════════════════════════════════
"""

from rnn_car_modular.experiment_config import ExperimentConfig

CONFIG = ExperimentConfig(
    # ─────────────────────────────────────────────────────────
    # 基本資訊
    # ─────────────────────────────────────────────────────────

    # config 的唯一名稱，必須與檔名一致（不含 .py）。
    # 會出現在 WandB config / console print / run_metadata。
    name="EXAMPLE_template",

    # 人類可讀的描述，記錄這份 config 的實驗目的。
    # 會出現在 [EXPERIMENT_CONFIG] console 輸出。
    description="範例模板 — 請複製此檔並修改為你的實驗設定",

    # ─────────────────────────────────────────────────────────
    # IsaacLab 任務 / 場景
    # ─────────────────────────────────────────────────────────

    # IsaacLab gym 任務 ID。決定觀測空間 / 動作空間 / 物理場景。
    # 目前只用: "Isaac-Navigation-Charge-VLP16-Curriculum-WD"
    # 不要隨便改，除非你新建了 gym.register 的任務。
    task="Isaac-Navigation-Charge-VLP16-Curriculum-WD",

    # Curriculum phase 定義檔。指向 curriculum/phases/ 下的 Python module。
    # 控制 stage 1-8 的場景難度、reward 權重、entropy、升降階門檻。
    # 目前只有: "warp_drive_single_agent_v1"
    curriculum_version="warp_drive_single_agent_v1",

    # 訓練從哪個 stage 開始（1-indexed）。
    # Stage 1 = 最簡單（多 goal、少障礙）。Stage 8 = 最終評估級。
    # 從 checkpoint 續訓時，通常設為當時的 stage。
    initial_stage=2,

    # True = 鎖定在 initial_stage 不升降階。用於固定 stage 消融實驗。
    # False = 啟用 curriculum 自動升降階。
    fixed_stage=True,

    # 場景 profile 名稱（metadata 用途，目前不影響行為）。
    # 通常與 curriculum_version 相同。
    scene_profile="warp_drive_single_agent_v1",

    # 障礙物控制模式:
    #   "rule_based" = BehaviorScheduler 控制（patrol/random_walk 等確定性行為）
    #   "learned"    = 用 learned FC policy 訓練對抗式障礙物
    # 目前主線用 rule_based。
    obstacle_mode="rule_based",

    # ─────────────────────────────────────────────────────────
    # 模組 Profiles — 決定「用什麼」
    # ─────────────────────────────────────────────────────────

    # Reward function 選擇。Phase 2 已啟用 runtime dispatch。
    # ✅ "wd_sparse"        — Warp Drive 稀疏獎勵（goal +40 / collision -5~-100）
    # ❌ "navrl_dense"      — 尚未實作
    # ❌ "hybrid_progress"  — 尚未實作
    # ❌ "ttc_risk"         — 尚未實作
    reward_profile="wd_sparse",

    # 演算法選擇。必須與 use_a2c 一致。
    # "a2c_wd"   — A2C (Warp Drive 風格，無 clipping)
    # "ppo_clip" — PPO (clipped surrogate objective)
    algorithm_profile="a2c_wd",

    # Aux RNN 訓練選擇。必須與 disable_aux_training 一致。
    # "wd_7d_geometry"       — WD 7D 幾何目標（最近 2 障礙物 body-frame 座標）
    # "none"                 — 關閉 aux 訓練
    # "future_collision_risk" — 尚未實作
    aux_profile="wd_7d_geometry",

    # 編碼器架構選擇:
    # "wd_exact_rnn"   — raw_fc_rnn 模式（更接近 WD 原版）
    # "extractor_rnn"  — Conv1d+MLP extractor → FC → RNN（有特徵提取）
    encoder_profile="wd_exact_rnn",

    # Critic 架構（metadata，目前不影響行為）。
    # "symmetric" = policy 和 critic 看一樣的 139D 觀測。
    critic_profile="symmetric",

    # ─────────────────────────────────────────────────────────
    # 訓練預算 — 決定「跑多久」
    # ─────────────────────────────────────────────────────────

    # 並行環境數。影響 batch size = num_envs × rollout_length。
    # 512 = 省 VRAM, sanity test 用
    # 1024 = 主線訓練
    # 4096 = 大規模（注意 RTX 5090 安全上限 4096）
    num_envs=1024,

    # 每次 rollout 的步數。num_envs × rollout_length = 一次 update 的 transition 數。
    # WD 風格固定 300。不建議改動，除非有明確理由。
    rollout_length=300,

    # 總 timesteps（= 總 transition 數）。
    # updates = timesteps / rollout_length。
    # 例如 180000 / 300 = 600 updates。
    # interactions = num_envs × timesteps = 1024 × 180000 ≈ 184M。
    timesteps=180000,

    # 隨機種子。固定種子 + 固定 config = 可重現實驗。
    seed=42,

    # ─────────────────────────────────────────────────────────
    # RL 超參數
    # ─────────────────────────────────────────────────────────

    # RL policy/value head optimizer 學習率。
    # WD 全 phase 固定 2e-4。消融實驗可降至 1e-4。
    lr=2e-4,

    # RNN / aux optimizer 學習率。
    # 若 disable_aux_training=True，建議設 0.0（凍結 RNN 權重）。
    # WD 全 phase 固定 5e-4。
    rnn_lr=5e-4,

    # Value loss 在 joint loss 中的權重: loss = policy_loss + vf_coeff × value_loss。
    # WD 原版 0.025，但 SA 實驗用 0.5 效果較好。
    vf_coeff=0.5,

    # Discount factor γ。決定 agent 多重視未來 reward。
    # WD 公式: γ = 1 - (1 - 0.92) / fps, fps=5 → γ = 0.984
    # SA2 實測 0.991 較好（更長視野）。
    gamma=0.991,

    # GAE λ。控制 advantage 估計的 bias-variance tradeoff。
    # 0.95 是標準值，通常不需要改。
    gae_lambda=0.95,

    # 是否 normalize value targets（per-rollout standardization）。
    # True = 穩定 critic loss scale。推薦開啟。
    normalize_return=True,

    # Value head 最後一層 bias 的初始值。None = 不覆寫（用 PyTorch 預設）。
    # 設 0.0 = 讓 critic 從零開始預測。
    value_init_bias=0.0,

    # RL optimizer 全域梯度裁剪上限。
    # 1.0 = 標準值。高壓 phase 可降至 0.5-0.8。
    max_grad_norm=1.0,

    # ─────────────────────────────────────────────────────────
    # A2C / PPO 專屬設定
    # ─────────────────────────────────────────────────────────

    # True = A2C（無 clipping，WD 預設模式）。
    # False = PPO（clipped surrogate）。
    # 必須與 algorithm_profile 一致:
    #   use_a2c=True  ↔ algorithm_profile="a2c_wd"
    #   use_a2c=False ↔ algorithm_profile="ppo_clip"
    use_a2c=True,

    # PPO 每次 rollout 後的學習 epochs（僅 PPO 有效）。
    # A2C 模式下此值被忽略。PPO 建議 2-4。
    ppo_epochs=2,

    # PPO mini-batch 數量（僅 PPO 有效）。
    # batch_size = num_envs × rollout_length / mini_batches。
    mini_batches=16,

    # PPO clipping epsilon（僅 PPO 有效）。
    # 0.1-0.2 是常見範圍。
    clip_eps=0.1,

    # ─────────────────────────────────────────────────────────
    # Entropy 係數
    # ─────────────────────────────────────────────────────────

    # 線速度 head 的 entropy 係數。
    # 設 0.0 = 自動從 Phase Config 同步（每個 stage 不同值）。
    # 設非零值 = 固定值，不隨 stage 變化（消融實驗用）。
    # WD Phase 1: 0.10, Phase 2+: 0.30, Phase 6+: 0.50
    ent_coeff_linear=0.05,

    # 角速度 head 的 entropy 係數。
    # 設 0.0 = 自動從 Phase Config 同步。
    # WD 全 phase 固定 0.375。
    ent_coeff_angular=0.10,

    # ─────────────────────────────────────────────────────────
    # WD-style 梯度分離裁剪
    # ─────────────────────────────────────────────────────────

    # 是否啟用 WD-style actor/critic 分離梯度 cap。
    # True = actor 和 critic grad norm 分別裁剪，避免 critic spike 壓制 actor。
    wd_update_clip=True,

    # Actor grad norm 上限。超過時按比例縮放。WD 全 phase 固定 8.0。
    wd_actor_update_clip=8.0,

    # Critic grad norm 上限。WD 全 phase 固定 30.0。
    wd_critic_update_clip=30.0,

    # ─────────────────────────────────────────────────────────
    # Aux RNN 訓練
    # ─────────────────────────────────────────────────────────

    # True = 關閉 aux/RNN 訓練。RL 仍用 checkpointed 的 12D preprocess features。
    # 必須與 aux_profile 一致:
    #   disable_aux_training=False ↔ aux_profile="wd_7d_geometry"
    #   disable_aux_training=True  ↔ aux_profile="none"
    disable_aux_training=False,

    # TBPTT 序列長度（aux 訓練的展開步數）。WD 常用 15。
    aux_seq_len=15,

    # Burn-in 步數（序列開頭跳過 loss 的步數）。0 = 不 burn-in。
    aux_burn_in=0,

    # 每次 aux mini-batch 的序列數量。
    # 太小 → 梯度估計雜訊大。太大 → VRAM 不夠。
    aux_seq_batch_size=256,

    # Aux/RNN 專用梯度裁剪。None = 沿用 max_grad_norm。
    # 設 0.5 可隔離 RNN gradient spike。
    aux_grad_clip=0.5,

    # ─────────────────────────────────────────────────────────
    # 安全 / 日誌
    # ─────────────────────────────────────────────────────────

    # True = 關閉 LiDAR 雜訊（distractor + Unoise）。
    # 真實 LiDAR 校準後建議開啟（Bug B 修復）。
    lidar_no_noise=True,

    # Action table 抽樣大小（用於動作分佈診斷）。
    # 0 = 關閉。2048 = 開啟（會稍微影響 throughput）。
    action_table_sample_size=0,

    # 每 N 個 iteration 輸出一次 console log + WandB 上傳。
    log_interval=10,

    # 每 N 個 iteration 存一次 checkpoint。
    save_interval=100,

    # ─────────────────────────────────────────────────────────
    # Checkpoint / 續訓
    # ─────────────────────────────────────────────────────────

    # 載入 checkpoint 路徑。None = 從頭訓練。
    # 例如: "logs/rnn_car/my_run/checkpoint_300.pt"
    # CLI 通常會覆蓋此值: --checkpoint <path>
    checkpoint=None,

    # True = 不從 checkpoint 恢復 optimizer state（只載入模型權重）。
    # 適用於: 換 lr / 換 stage / 換 config 時續訓。
    # False = 連 optimizer momentum 一起恢復（同 config 續訓用）。
    no_resume_optimizer=True,

    # ─────────────────────────────────────────────────────────
    # Metadata
    # ─────────────────────────────────────────────────────────

    # WandB run 的 tags。方便在 WandB UI 中篩選。
    # 建議至少包含: curriculum 版本、algorithm、aux 狀態。
    tags=("wd", "sa2", "a2c", "aux", "example"),

    # WandB run 的 notes。記錄實驗動機或假設。
    notes="這是一份範例 config，請複製修改後使用。",
)
