# WandB 指標完整參考手冊

> **最後更新**: 2026-03-11
> **適用任務**: Isaac-Navigation-Charge-VLP16 (SKRL PPO/AAC)
> **規則**: 每次有指標變動都要更新此文件

---

## 指標總覽

| 類別 | 前綴 | 數量 | 說明 |
|------|------|------|------|
| SKRL 標準損失 | `Loss /` | 3 | PPO 核心損失函數 |
| 策略分佈 | `Policy /` | 1-2 | 策略分佈特性 |
| 學習率 | `Learning /` | 1 | 學習率排程 |
| 環境資訊 | `Info /` | ~5 | Isaac Lab 環境統計 |
| 導航表現 | `nav/` | 5 | 累計成功/碰撞/超時率 |
| 訓練元資料 | `train/` | 3 | FPS、時間、步數 |
| 獎勵統計 | `reward_breakdown/` | 3 | 即時總獎勵統計 |
| 獎勵分項 | `reward_terms/` | 7 | 各獎勵函數 per-step 值 |
| 回合統計 | `episode/` | 4 | 回合結束統計 |
| 終止原因 | `termination/` | ~4 | 各終止條件觸發率 |
| 動作統計 | `action/` | 4 | 原始/裁切動作統計 |
| 機器人狀態 | `robot/` | 8 | 速度、進度、障礙距離 |
| PPO 內部 | `ppo/` | 19 | 優勢、梯度、KL 等 |

---

## 1. SKRL 標準損失 (`Loss /`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `Loss / Policy loss` | 策略損失 | PPO clip surrogate loss 的平均值。衡量策略更新品質 | 訓練初期波動較大，中後期應穩定在 -0.01~0.0 之間。若 loss 持續為 0 或不變化，代表策略「凍結」 |
| `Loss / Value loss` | 價值損失 | MSE(V_pred, returns)。衡量 Critic 對回報預測的準確度 | 初期較高（~10-100），隨訓練下降至 <1.0。若長期不下降表示 Critic 學不到正確的價值函數 |
| `Loss / Entropy loss` | 熵損失 | -entropy_coeff × mean(entropy)。鼓勵策略探索 | 初期較負（探索充分），隨訓練趨近 0。若過早趨近 0 代表策略過早收斂（探索不足） |

---

## 2. 策略分佈 (`Policy /`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `Policy / Entropy` | 策略熵 (離散) | MultiCategorical 分佈的平均熵。衡量動作選擇的隨機程度 | 初期高（~3.0，21 bins → max ln(21)≈3.04），訓練中逐漸下降至 1.0~2.0 表示學到偏好。不應降至 <0.5（策略退化） |
| `Policy / Standard deviation` | 策略標準差 (連續) | Gaussian 策略的平均 σ（僅連續動作空間使用） | 初期 ~0.5-1.0，隨訓練下降至 0.1-0.3。過低 (<0.05) 表示探索停止 |

---

## 3. 學習率 (`Learning /`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `Learning / Learning rate` | 學習率 | 當前優化器學習率（若啟用排程） | 依排程設定。常見 linear decay 從 3e-4 → 0 |

---

## 4. 環境資訊 (`Info /`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `Info / Episode Reward` | 回合獎勵 | 單回合累計獎勵（Isaac Lab 環境回報） | 成功回合：~+20，碰撞：~-14，超時：~-1.5。應隨訓練逐漸提升 |
| `Info / Episode Length` | 回合長度 | 回合持續步數 | 初期短（碰撞頻繁 ~20步），中期增長（學會避障），最終穩定在合理值（50-100步完成導航） |
| `Info / Episode_Reward/*` | 各獎勵項回合累計 | RewardManager 每個 term 的回合總和 | 因 term 而異，見獎勵分項說明 |
| `Info / Episode_Termination/*` | 各終止原因比例 | 該回合批次中各終止原因的占比 | 見終止原因章節 |

---

## 5. 任務統計 (`nav/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `nav/ success_rate` | 成功率 | 累計到達目標的回合比例 | 0→1。良好訓練應從 ~0% 逐漸上升至 60%+。最終目標 >80% |
| `nav/ collision_rate` | 碰撞率 | 累計碰撞終止的回合比例 | 初期高（~50-80%），應隨訓練下降至 <10% |
| `nav/ timeout_rate` | 超時率 | 累計超時終止的回合比例 | 初期低（碰撞太快），中期可能升高（學會避障但導航能力不足），最終應下降 |
| `nav/ total_episodes` | 總回合數 | 從訓練開始到目前的回合總數 | 單調遞增 |
| `nav/ episode_length_mean` | 回合平均長度 | 累計回合長度的平均值 | 與 Episode Length 類似趨勢 |

---

## 6. 訓練元資料 (`train/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `train/fps` | 訓練 FPS | 每秒模擬步數 (timesteps/elapsed_time) | 穩定值取決於硬體。256 envs 典型 2000-5000 FPS |
| `train/elapsed_time` | 已用時間 | 從訓練開始的秒數 | 單調遞增 |
| `train/timestep` | 當前步數 | 全局 timestep 計數器 | 單調遞增 |

---

## 7. 即時獎勵統計 (`reward_breakdown/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `reward_breakdown/step_mean` | 即時獎勵均值 | 窗口內所有 env 所有步的平均即時獎勵 | 應隨訓練從負值逐漸上升至正值 |
| `reward_breakdown/step_min` | 即時獎勵最小值 | 窗口內每步的 min(rewards across envs) 的平均 | 碰撞懲罰觸發時會很低（~-15）。整體應上升 |
| `reward_breakdown/step_max` | 即時獎勵最大值 | 窗口內每步的 max(rewards across envs) 的平均 | 到達目標時會很高（~+20）。整體應上升 |

---

## 8. 獎勵統計分項 (`reward_terms/`)

每個 term 的 per-step 值 = `func() × weight × dt`

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `reward_terms/reaching_goal_mean` | 到達目標獎勵 | 到達目標時 +20（= 1 × 100 × 0.2），否則 0 | 初期 ~0（很少成功），隨訓練頻率增加 |
| `reward_terms/collision_terminal_mean` | 碰撞終止懲罰 | 碰撞時 -15（= 1 × -75 × 0.2），否則 0 | 初期頻繁觸發（負值多），隨訓練頻率下降 |
| `reward_terms/potential_progress_mean` | 勢能進度獎勵 | (d_prev - d_curr) × 10 × 0.2。正值=接近目標 | 初期 ~0（隨機移動），學到導航後應穩定正值（~0.06/step） |
| `reward_terms/time_penalty_mean` | 時間懲罰 | 每步固定 -0.01（= 1 × -0.05 × 0.2） | 恆定 -0.01。若偏離此值表示有 bug |
| `reward_terms/heading_to_goal_mean` | 朝向目標獎勵 | cos(angle_to_goal) × 1.5 × 0.2，正=朝向目標 | 初期 ~0（隨機朝向），學到後應穩定正值 |
| `reward_terms/acceleration_penalty_mean` | 加速度懲罰 | -\|a\|² × 0.25 × 0.2。懲罰劇烈加速 | 小負值（~-0.001）。若過大表示動作震盪 |
| `reward_terms/angular_velocity_penalty_mean` | 角速度懲罰 | -\|ω\|² × 0.25 × 0.2。懲罰不必要轉彎 | 小負值（~-0.005）。若過大表示原地打轉 |

---

## 9. 回合統計 (`episode/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `episode/done_rate_per_step` | 每步結束率 | 窗口內平均每步有多少 env 結束 | 初期高（碰撞快），中期下降，穩定後 ~0.01-0.05 |
| `episode/total_episodes_window` | 窗口回合數 | 當前統計窗口內完成的回合總數 | 取決於 rollout 長度和回合持續時間 |
| `episode/length_mean` | 回合平均長度 | 窗口內結束回合的平均步數 | 初期短（~20，頻繁碰撞），應逐漸增長至 50-100 |
| `episode/length_min` | 回合最短長度 | 窗口內最短的回合步數 | 碰撞回合可能很短（<10步）。整體應上升 |
| `episode/length_max` | 回合最長長度 | 窗口內最長的回合步數 | 超時回合 = max_episode_steps。出現長回合表示開始學會生存 |

---

## 10. 終止原因 (`termination/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `termination/goal_reached_count` | 到達目標計數 | 窗口內因到達目標而結束的回合數 | 應隨訓練逐漸增加 |
| `termination/goal_reached_rate` | 到達目標率 | 到達目標占總回合的比例 | 0→1。良好訓練應持續上升 |
| `termination/collision_count` | 碰撞計數 | 窗口內因碰撞而結束的回合數 | 應隨訓練逐漸減少 |
| `termination/collision_rate` | 碰撞率 | 碰撞占總回合的比例 | 應從高值（~0.5-0.8）下降至 <0.1 |
| `termination/time_out_count` | 超時計數 | 窗口內因超時而結束的回合數 | 中期可能暫時升高，最終應下降 |
| `termination/time_out_rate` | 超時率 | 超時占總回合的比例 | 中期可能升高（避障但不導航），最終下降 |
| `termination/robot_tipped_over_count` | 翻覆計數 | 窗口內因機器人翻覆而結束的回合數 | 應始終為 0 或極低。若頻繁出現表示物理設定有問題 |
| `termination/robot_tipped_over_rate` | 翻覆率 | 翻覆占總回合的比例 | 應始終 ~0 |

---

## 11. 動作管線 (`action/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `action/raw_mean` | 原始動作均值 | 策略網路輸出（NN raw output）的平均值 | 初期 ~0（隨機初始化），訓練後根據策略偏好偏移 |
| `action/raw_std` | 原始動作標準差 | 策略網路輸出的標準差 | 初期較大，隨策略確信度提高可能下降 |
| `action/raw_min` | 原始動作最小值 | 策略網路輸出的最小值 | 觀察是否接近動作空間邊界 |
| `action/raw_max` | 原始動作最大值 | 策略網路輸出的最大值 | 觀察是否接近動作空間邊界 |
| `action/applied_mean` | 裁切後動作均值 | 經速度飽和裁切後的 [applied_a, ω] 平均值 | 反映實際施加的控制指令 |
| `action/applied_std` | 裁切後動作標準差 | 裁切後動作的標準差 | 若比 raw_std 小很多，表示大量動作被裁切 |
| `action/applied_min` | 裁切後動作最小值 | 裁切後動作的最小值 | 觀察實際控制範圍 |
| `action/applied_max` | 裁切後動作最大值 | 裁切後動作的最大值 | 觀察實際控制範圍 |
| `action/clipping_rate` | 動作裁切率 | 被 clamp 裁切的動作比例 | 應 <10%。若過高表示策略輸出超出合理範圍 |
| `action/saturation_rate` | 速度飽和率 | 車速在上限(1.0 m/s)或下限(0)的 env 比例 | 初期高（隨機撞牆 → v=0），學到後穩定在 10-30% |
| `action/safety_override_rate` | 安全覆寫率 | 因速度飽和導致加速度被強制歸零的比例 | 應 <20%。過高表示策略持續要求不可能的加速度 |

---

## 12. 運動狀態 (`robot/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `robot/speed_mean` | 平均線速度 | 所有 env 的 2D 速度大小平均值 (m/s) | 初期波動，學到後穩定在 0.3-0.8 m/s |
| `robot/speed_max` | 最大線速度 | 所有 env 中最高速度 (m/s) | 應不超過 max_v=1.0 m/s。若超過表示物理穿透 |
| `robot/angular_vel_mean` | 平均角速度 | 所有 env 的 \|ω_z\| 平均值 (rad/s) | 初期高（隨機旋轉），學到後下降（走直線為主） |
| `robot/angular_vel_max` | 最大角速度 | 所有 env 中最高角速度 (rad/s) | 應在合理範圍 (<π/15 ≈ 0.21 rad/s) |
| `robot/progress_per_step` | 每步接近目標距離 | d_prev - d_curr 平均值 (m/step)。正=接近 | 初期 ~0，學到後穩定正值（~0.01-0.03 m/step） |
| `robot/progress_per_step_std` | 每步進度標準差 | 進度的標準差 | 高 std 表示行為不穩定（有的 env 前進有的後退） |
| `robot/min_obstacle_dist_mean` | 平均最近障礙距離 | 所有 env 的 LiDAR 最小讀數平均值 (m)（排除 max_range hit） | 應維持在安全範圍（>0.5m）。低於 0.3m 觸發碰撞 |
| `robot/min_obstacle_dist_min` | 最小障礙距離 | 所有 env 中 LiDAR 最小讀數的最小值 (m) | 碰撞閾值 = 0.3m。此值接近閾值時碰撞頻繁 |
| `robot/lidar_min_raw` | LiDAR 原始最小值 | 所有 env 所有 ray 中的最小距離 (m)，包含 max_range 替換 | 診斷用。若與 min_obstacle_dist 差異大表示 max_range hit 處理有問題 |
| `robot/lidar_max_range_hit_ratio` | LiDAR 最大距離命中率 | ray 未命中（NaN/Inf/超出 max_range）的比例 | 空曠環境接近 1.0。密集障礙物環境 <0.5。若持續 ~1.0 表示 LiDAR 根本沒偵測到障礙物 |

---

## 13. PPO 內部健康 (`ppo/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `ppo/advantage_raw_mean` | 原始優勢均值 | GAE 計算的優勢函數均值（正規化前） | 應接近 0（PPO 設計）。持續偏離 0 表示 Value 估計有偏差 |
| `ppo/advantage_raw_std` | 原始優勢標準差 | 優勢函數的分散程度 | 典型 0.5-5.0。過小（<0.01）表示訊號消失；過大（>100）表示獎勵爆炸 |
| `ppo/advantage_raw_min` | 原始優勢最小值 | 最差 trajectory 的優勢 | 負值正常。若絕對值極大表示獎勵設計有問題 |
| `ppo/advantage_raw_max` | 原始優勢最大值 | 最佳 trajectory 的優勢 | 正值正常。若絕對值極大表示獎勵設計有問題 |
| `ppo/advantage_norm_mean` | 正規化優勢均值 | 正規化後優勢均值（應≈0） | 應非常接近 0（因為做了 mean-std normalization） |
| `ppo/advantage_norm_std` | 正規化優勢標準差 | 正規化後優勢標準差（應≈1） | 應非常接近 1.0 |
| `ppo/value_pred_mean` | 價值預測均值 | Critic 網路 V(s) 的平均輸出 | 應逐漸趨近實際平均回報。成功訓練時 ~10-20 |
| `ppo/value_pred_std` | 價值預測標準差 | V(s) 的分散程度 | 隨 Critic 學習應穩定在合理範圍 |
| `ppo/value_pred_min` | 價值預測最小值 | V(s) 的最小值 | 負值正常（碰撞 state）。若 min≈max 表示 critic 輸出常數 |
| `ppo/value_pred_max` | 價值預測最大值 | V(s) 的最大值 | 正值正常（成功 state）。若 min≈max 表示 critic 輸出常數 |
| `ppo/returns_mean` | 回報均值 | GAE returns (V + A) 的平均值 | 應與 value_pred_mean 趨近（Critic 在學） |
| `ppo/returns_std` | 回報標準差 | Returns 的分散程度 | 反映環境結果的多樣性 |
| `ppo/returns_min` | 回報最小值 | 最差 trajectory 的 return | 負值正常 |
| `ppo/returns_max` | 回報最大值 | 最佳 trajectory 的 return | 成功 trajectory 應為正值 |
| `ppo/reward_mem_mean` | Memory 獎勵均值 | 記憶體中 per-step 獎勵的平均值 | 反映 RewardManager 實際寫入的獎勵。應非零 |
| `ppo/reward_mem_std` | Memory 獎勵標準差 | 獎勵分散程度 | >0 表示有多樣的獎勵信號。≈0 表示獎勵退化 |
| `ppo/reward_mem_min` | Memory 獎勵最小值 | 最低 per-step 獎勵 | 碰撞步 ~-15。若最小=最大表示獎勵常數 |
| `ppo/reward_mem_max` | Memory 獎勵最大值 | 最高 per-step 獎勵 | 成功步 ~+20 |
| `ppo/explained_variance` | 解釋方差 | 1 - Var(returns-V)/Var(returns)。衡量 Critic 準確度 | 應從 ~0 上升至 >0.5。>0.8 優秀。<0 表示 Critic 預測比隨機還差 |
| `ppo/kl_divergence` | KL 散度 | π_new 與 π_old 的近似 KL 散度 | 應 <0.03（PPO 設計目標）。>0.05 表示更新步幅過大 |
| `ppo/policy_ratio_mean` | 策略比率均值 | exp(log_π_new - log_π_old) 的均值 | 應接近 1.0。遠離 1 表示策略變化劇烈 |
| `ppo/clip_fraction` | 裁切比例 | 被 PPO clip 裁切的 ratio 占比 | 典型 0.05-0.20。>0.3 表示步幅過大；≈0 表示學習停滯 |
| `ppo/grad_norm_policy` | 策略梯度範數 | Policy 網路梯度的 L2 範數（裁切前） | 典型 0.1-10.0。>100 表示梯度爆炸；<0.001 表示梯度消失 |
| `ppo/grad_norm_value` | 價值梯度範數 | Value 網路梯度的 L2 範數（裁切前） | 典型 0.1-10.0。與 policy 類似觀察 |
| `ppo/param_max_diff` | 參數最大變化 | 更新前後策略第一層參數的 max\|Δw\| | 典型 1e-4~1e-2。≈0 表示學習完全停止；>0.1 表示更新過大 |
| `ppo/learning_rate` | PPO 學習率 | 優化器當前學習率 | 依排程設定 |
| `ppo/update_count` | PPO 更新次數 | 窗口內 PPO _update 被呼叫的次數 | 取決於 rollout 長度設定 |

---

## 14. 元資料 (`meta/`)

| 指標名稱 (EN) | 指標名稱 (中文) | 物理意義 | 數值走向 |
|---------------|----------------|---------|---------|
| `meta/steps_in_window` | 窗口步數 | 本次 get_and_reset() 統計的總步數 | 應等於 rollout 長度（128） |

---

## 診斷快速查找表

### 問題：訓練訊號被切斷（Loss 持平、Reward 崖壁）

| 檢查指標 | 正常 | 異常 |
|---------|------|------|
| `ppo/advantage_raw_std` | 0.5~5.0 | <0.01（訊號消失） |
| `ppo/grad_norm_policy` | 0.1~10 | <0.001（梯度消失） |
| `ppo/param_max_diff` | 1e-4~1e-2 | ≈0（學習停止） |
| `ppo/clip_fraction` | 0.05~0.20 | ≈0（無更新）或 >0.5（崩潰） |
| `ppo/explained_variance` | >0.3 | <0 或突然歸零 |
| `Policy / Entropy` | 1.0~3.0 | <0.5（過早收斂） |

### 問題：碰撞率過高

| 檢查指標 | 正常 | 異常 |
|---------|------|------|
| `robot/min_obstacle_dist_mean` | >0.5m | <0.3m |
| `reward_terms/collision_terminal_mean` | 接近 0 | 頻繁大負值 |
| `action/saturation_rate` | <30% | >60%（無法減速） |
| `reward_terms/smooth_collision_penalty_mean` | 小負值 | 持續接近 -1 |

### 問題：導航效率低

| 檢查指標 | 正常 | 異常 |
|---------|------|------|
| `robot/progress_per_step` | >0.01 | ≈0 或負值 |
| `reward_terms/heading_to_goal_mean` | 正值 | ≈0（隨機朝向） |
| `robot/speed_mean` | 0.3~0.8 m/s | <0.1（幾乎靜止） |
| `nav/ timeout_rate` | <20% | >50% |

---

## 指標來源對照

| 來源文件 | 負責的指標前綴 |
|---------|---------------|
| `train_charge.py` (aac_update) | `Loss /`, `Policy /`, `Learning /`, `ppo/` (via `_debug_ppo_metrics`) |
| `wandb_trainer.py` (_log_to_wandb) | `train/`, `nav/`, `Info /` |
| `wandb_trainer.py` (_capture_tracking_data) | `Loss /`, `Policy /`, `Learning /` (快照轉發) |
| `training_debug_logger.py` (step) | `reward_breakdown/`, `reward_terms/`, `episode/`, `termination/`, `action/`, `robot/` |
| `training_debug_logger.py` (on_ppo_update) | `ppo/` (由 train_charge 傳入) |
| `training_debug_logger.py` (get_and_reset) | `meta/` |

---

## 更新日誌

| 日期 | 變更 |
|------|------|
| 2026-03-05 | 初版：完整列出所有 WandB 同步指標，包含 SKRL 標準 + Debug 5 大類共 ~80 項 |
| 2026-03-05 | v2: 新增 value_pred_min/max, returns_min/max, reward_mem_*, lidar_min_raw, lidar_max_range_hit_ratio 共 12 項新指標 |
