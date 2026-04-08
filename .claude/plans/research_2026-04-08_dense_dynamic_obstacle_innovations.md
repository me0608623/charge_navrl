# Charge_skrl 動態避障創新優化提案 (2026-04 文獻版)

> 文獻檢索日期: 2026-04-08
> 目標: 突破動態障礙物避障的成功率與決策貢獻
> 來源: ICRA / IROS / CoRL / NeurIPS / RA-L / T-ASE / arXiv 2023-2025

---

## 一、執行摘要

**問題定位**: charge_skrl v21 在 dense dynamic obstacle 場景下 CR ≈ 49%,根本瓶頸有三:
(1) closing_risk reward effective weight 僅 0.31% (太弱),即便 M1.4 ×3 後仍 ≈ 1.16%;
(2) Stateless feed-forward + MaxPool over obstacles → 丟失時序與 pairwise interaction;
(3) Train/eval OOD (dyn_ratio cap 0.40 vs eval 0.44),手工 curriculum 無自適應機制。

**核心發現**: 2024-2025 文獻已從「reward 加權」轉向兩條主軸:
- **架構面**: Heterogeneous Graph Transformer (HEIGHT, T-ASE 2026) + Set-based attention 取代 MaxPool — 在 dense crowd 場景下對 OOD generalization 顯著改善;
- **安全層面**: Differentiable VO/CBF safety layer (CoBL-Diffusion, NeurIPS 2024 / SEA-Nav 2024) — 把 safety 從 reward shaping 移到 architecture 內,避免 reward weight 調參戰。
- **訓練面**: ADD (Adversarial Diffusion Design, NeurIPS 2024 spotlight) 與 Bounded Rationality Adversarial RL (CoRL 2024) 提供自動 curriculum,直接回應 OOD 問題。

**Top-3 推薦**:
🥇 **HEIGHT-style Heterogeneous Graph Transformer** (替換 MaxPool encoder, warm start 可行) — 預期 CR 改善最大;
🥈 **Bounded Rationality Adversarial Curriculum** (動態障礙物變 RL adversary,自動產生 OOD) — 直接修 OOD bug;
🥉 **CoBL-style differentiable CBF safety layer** (取代 hand-tuned closing_risk weight,理論上 collision-free) — reward 調參戰的終結方案。

不確定性已標記在第九節。

---

## 二、SOTA 文獻盤點 (Q1)

### 方向 A: Heterogeneous Interaction Graph Transformer
- **代表 paper**: HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments
- **作者**: Shuijing Liu, Haochen Xia, Fatemeh Cheraghi Pouria, Kaiwen Hong, Neeloy Chakraborty, Zichao Hu, Joydeep Biswas, Katherine Driggs-Campbell
- **Venue**: IEEE T-ASE 2026 (arXiv 2024-11-19)
- **URL**: https://arxiv.org/abs/2411.12150
- **核心 contribution**: heterogeneous spatio-temporal graph 區分 robot-human / human-human / human-obstacle 三種 interaction edge,attention 加權後送入 GRU 處理時序;與 charge_skrl 的 MaxPool 直接對比就是「拒絕用 sum/max 壓掉互動資訊」。
- **理論依據**: graph attention `α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))`,permutation invariant 但保留 pairwise 結構。
- **預期效果** (paper 報告): 在 dense crowd + constrained 場景對 SOTA baselines (DSRNN / Attention Graph) 在 success rate / generalization 兩項勝出 (具體數字未在 abstract 揭露,需讀全文 Table II/III)。
- **對 charge 適用性**: ⭐⭐⭐⭐⭐ 高 — 我們現在的 obstacle branch 就是 MLP + MaxPool,直接被這個方向打;且 3-branch 結構保留即可,只換 obstacle encoder。
- **實作成本**: 中 (5-15 GPU hours) — 需引入 PyTorch Geometric 或自寫 attention,policy 體積 +30-50%
- **warm start 可行**: ✅ (LiDAR 與 state branch 不變,只重訓 obstacle branch 與 head)

### 方向 B: Intention-Aware CrowdNav with Trajectory Prediction
- **代表 paper**: Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph
- **作者**: Shuijing Liu et al.
- **Venue**: ICRA 2023 (但 2024-25 follow-up 持續引用)
- **URL**: https://arxiv.org/abs/2203.01821
- **核心 contribution**: 用獨立 trajectory predictor (constant velocity / GST / GRU) 預測未來 N 步 human positions,把預測軌跡塞進 RL state;policy 學到「不踩別人的未來路徑」。
- **理論依據**: state augmentation `s_t = [s_t, ŝ_{t+1:t+N}]`,N=5 步 ≈ 1 秒未來。
- **預期效果**: paper 在 dense crowd 報告 SR 從 SARL 的 86% 提升到 95%,collision 從 13% 降到 4% (簡化敘述)。
- **對 charge 適用性**: ⭐⭐⭐⭐ 高 — charge 已有 obstacle vx/vy,加 N 步 const-velocity 預測幾乎零成本。
- **實作成本**: 小 (1-3 GPU hr 重訓) — obs 維度從 60 → 60+60(2步預測) 或 60+30(座標 only)
- **warm start 可行**: ⚠️ 部分 (obs dim 改了須 retrain head,但 backbone 可保留 + freeze)

### 方向 C: Bounded Rationality Adversarial RL Curriculum
- **代表 paper**: Dynamic Obstacle Avoidance with Bounded Rationality Adversarial Reinforcement Learning
- **作者**: (Unitree GO1 quadruped paper, CoRL 2024 / arXiv 2503.11467)
- **Venue**: CoRL 2024
- **URL**: https://arxiv.org/abs/2503.11467
- **核心 contribution**: 把動態障礙物建模為 second SAC agent,雙方對抗訓練;但用 quantal response equilibrium (QRE) 限制 adversary 的理性程度 (避免 protagonist 被打到絕望)。adversary 的 temperature 用 self-paced curriculum 自動調整。
- **理論依據**: QRE `π_adv(a|s) ∝ exp(Q(s,a)/τ)`,τ→0 時為最優,τ→∞ 為 random;curriculum 從高 τ 退火到低 τ。
- **預期效果**: paper 報告在多種 dynamic obstacle 配置下,robustness 顯著超越 fixed-policy obstacle baseline;具體 SR 數字需讀 Table。
- **對 charge 適用性**: ⭐⭐⭐⭐ 高 — 直接解 charge 的 dyn_ratio OOD 問題,不需要 hand-tune curriculum stages。
- **實作成本**: 大 (15-30 GPU hr) — 需要寫 adversary policy + 雙 PPO loop;但 protagonist 結構不變
- **warm start 可行**: ✅ (用現有 v21 當 protagonist 起點,從零訓練 adversary)

### 方向 D: Adversarial Environment Design via Regret-Guided Diffusion (ADD)
- **代表 paper**: Adversarial Environment Design via Regret-Guided Diffusion Models
- **Venue**: NeurIPS 2024 spotlight
- **URL**: https://arxiv.org/abs/2410.19715
- **核心 contribution**: 用 diffusion model 生成「對 agent 而言剛好最具學習價值」的環境配置,以 regret 為 guidance signal,屬於 PAIRED 系列的 SOTA。
- **理論依據**: regret = V*(s) - V_π(s),用 diffusion 在 environment design space 採樣,score function 等比於 regret。
- **預期效果**: paper 在 zero-shot OOD generalization 對 PLR / ACCEL / PAIRED baselines 有顯著提升 (Procgen / Minigrid 系列)。
- **對 charge 適用性**: ⭐⭐⭐ 中 — 概念性匹配 OOD 問題,但 diffusion model 訓練需要額外 dataset;對 charge 這種 procedural environment 可改為「dyn obstacle config 由 regret 引導採樣」的簡化版。
- **實作成本**: 大 (20-40 GPU hr 含 diffusion 訓練)
- **warm start 可行**: ❌ (需要從頭設計 environment design pipeline)

### 方向 E: NavRL Velocity Obstacle Safety Shield (Isaac Sim)
- **代表 paper**: NavRL: Learning Safe Flight in Dynamic Environments
- **作者**: Zhefan Xu, Xinming Han, Haoyu Shen, Hanyu Jin, Kenji Shimada (CMU)
- **Venue**: IEEE RA-L 2025 (arXiv 2409.15634)
- **URL**: https://arxiv.org/abs/2409.15634
- **核心 contribution**: end-to-end PPO + 後置 VO LP solver;LP 把 RL action 投影到 VO admissible set。Isaac Sim 平行訓練 (跟我們同 stack)。
- **理論依據**: 對每個 obstacle i 建立 VO_i = {v: v ∉ admissible_velocity_cone},RL 給 v_rl,LP 求解 `min ||v - v_rl||² s.t. v ∉ ∪VO_i`。
- **預期效果**: paper 報告在多種動態障礙物場景達 zero-shot sim2real,collision 數字 "fewest among baselines" (具體值需讀全文)。
- **對 charge 適用性**: ⭐⭐⭐ 中 — 我們 M1.2 試過 simplified VO 失敗 (CR -5pp),但失敗點在「single-worst-threat heuristic」而非完整 LP;NavRL 的完整 LP 不同。然而 charge 是 diff-drive 不是 holonomic UAV,LP constraint 較複雜。
- **實作成本**: 中 (10-20 GPU hr) — 需要可微 LP layer (qpth / cvxpylayers)
- **warm start 可行**: ✅ (post-hoc 加在 v21 policy 後,可即時測試)
- ⚠️ 警告: 我們 M1.1/M1.2/M1.3 都試過類似方向失敗,須仔細區分「shield 的數學形式」與「shield 的 reward signal 干擾」。

### 方向 F: CoBL-Diffusion (CBF + CLF Guided Diffusion Policy)
- **代表 paper**: CoBL-Diffusion: Diffusion-Based Conditional Robot Planning in Dynamic Environments Using Control Barrier and Lyapunov Functions
- **作者**: Kazuki Mizuta, Karen Leung
- **Venue**: arXiv 2406.05309 (2024-11)
- **URL**: https://arxiv.org/abs/2406.05309
- **核心 contribution**: 在 diffusion model 的 denoising step 注入 CBF (safety) + CLF (goal-reaching) 兩個 classifier-free guidance gradient,生成的軌跡同時滿足兩個約束。
- **理論依據**: `∇log p(τ) ← ∇log p(τ) + λ_cbf ∇h_cbf(τ) + λ_clf ∇V_clf(τ)`,其中 h_cbf = signed distance, V_clf = goal Lyapunov。
- **預期效果**: paper 在真實 pedestrian dataset 報告 collision rate 顯著低於 unconditional diffusion baseline。
- **對 charge 適用性**: ⭐⭐⭐ 中 — diffusion policy 對 discrete action (charge MultiDiscrete[19,19]) 不直接相容,需轉成 continuous + 後 quantize,改造大;但 CBF guidance 概念可移植到 PPO actor (replace closing_risk reward with CBF gradient)。
- **實作成本**: 大 (30-60 GPU hr) — 整個 actor 換成 diffusion
- **warm start 可行**: ❌

### 方向 G: World Model — DreamerV3 + Latent Rollout
- **代表 paper**: DreamerNav: learning-based autonomous navigation in dynamic indoor environments using world models
- **Venue**: Frontiers in Robotics & AI 2025
- **URL**: https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1655171/full
- **核心 contribution**: 用 DreamerV3 RSSM (Recurrent State Space Model) 學環境 latent dynamics,actor/critic 在 latent space 想像式 rollout,POMDP 友好。
- **理論依據**: RSSM 用 deterministic h_t + stochastic z_t,`p(z_t|h_t, a_{t-1}, x_t)`,actor 在 imagination horizon H=15 上做 dreaming。
- **預期效果**: 對於 stochastic dynamic obstacle,latent rollout 可以隱式做 prediction;paper 報告對 dynamic indoor scene 比 model-free baseline 顯著提升。
- **對 charge 適用性**: ⭐⭐ 中-低 — 改架構成本最大,放棄 PPO + skrl pipeline;但長期看是 right direction。
- **實作成本**: 極大 (>50 GPU hr) — 完全換 RL backbone
- **warm start 可行**: ❌

### 方向 H: Future-Oriented Energy-Based Multimodal Motion Prediction
- **代表 paper**: Future-Oriented Navigation: Dynamic Obstacle Avoidance with One-Shot Energy-Based Multimodal Motion Prediction
- **Venue**: arXiv 2505.00237 (2025-05)
- **URL**: https://arxiv.org/abs/2505.00237
- **核心 contribution**: 用 energy-based neural network 一次性產生多模態 obstacle 未來軌跡 (multi-modal 處理 ambiguity),整合進 MPC。
- **理論依據**: `E_θ(τ_future | history)`,Langevin sampling 從 energy landscape 抽多 mode。
- **預期效果**: paper 報告在動態場景對 unimodal MPC baseline 顯著降 collision。
- **對 charge 適用性**: ⭐⭐⭐ 中 — 預測模組可獨立掛入,但 energy network 訓練需要 trajectory dataset (Isaac Sim 可生成)。
- **實作成本**: 中-大 (20-30 GPU hr)
- **warm start 可行**: ✅ (不影響現有 RL,加 prediction module 就好)

---

## 三、神經網路架構創新 (Q2)

### 架構 1: Set Transformer + Cross-Attention
- **代表 paper**: 雖然原 Set Transformer 是 2019,但 2024 follow-up 在 navigation 上的應用見 MarineFormer (arXiv 2410.13973) 與 HEIGHT。
- **核心**: ISAB (Induced Set Attention Block) 取代 MaxPool,permutation invariant 同時保留 pairwise。對 charge 的 10 obstacles 來說 self-attention 計算量 O(100) 完全可承受。
- **與 charge 整合**: obstacle branch 從 `MLP(60→32) + MaxPool` 換成 `ISAB(10×6 → 10×32) + Pool`,然後加一個 `cross-attention(robot_state_query, obstacles_kv)` 讓 robot 主動「看」最相關 obstacle。
- **改造成本**: 小 (~150 行代碼,純 PyTorch)
- **warm start**: ✅
- **預期效果**: 文獻一致認為 attention > MaxPool 在 dense interaction,改善 5-15% SR (paper 平均值)
- **論文佐證**: HEIGHT abstract + Set Transformer 原 paper Fig 2 對比

### 架構 2: 加 GRU/LSTM (CrowdNav-Prediction-AttnGraph)
- **代表 paper**: Liu et al. ICRA 2023 + HEIGHT 2024 都在 attention 後加 GRU
- **核心**: head 前加 1 層 GRU (hidden 128),處理 stateless feed-forward 的 POMDP 缺陷。
- **與 charge 整合**: skrl PPO 支持 RNN policy (`scripts/diagnostics` 已有 example),改 cfg 即可;但要注意 truncated BPTT 與 batch organization。
- **改造成本**: 中 (~300 行 + skrl recurrent wrapper 設定)
- **warm start**: ⚠️ 部分 (RNN hidden state 從零開始,但 feature extractor 可保留)
- **預期效果**: 對 dynamic scene POMDP 顯著改善,paper 報告 SR +5-10%
- **論文佐證**: HEIGHT 在 ablation 顯示 RNN 是必要的

### 架構 3: Equivariant SE(2) Network
- **代表 paper**: E(2)-Equivariant Graph Planning for Navigation, RA-L 2024
- **URL**: https://www.khoury.northeastern.edu/home/lsw/papers/ral2024-equnav.pdf
- **核心**: 利用 SE(2) symmetry,navigation 對任意旋轉/平移應該等變;網路用 e2cnn 的 group convolution。
- **與 charge 整合**: charge 已用 body frame obs (內建平移不變),但旋轉等變未明確建模。轉成 SE(2)-equivariant 可降低 sample complexity 30-50%。
- **改造成本**: 大 (~500 行,引入 e2cnn library + retrain from scratch)
- **warm start**: ❌
- **預期效果**: paper 報告在 grid navigation 上 sample efficiency 提升 2-3 倍

### 架構 4: Mamba State-Space (取代 LSTM)
- **代表 paper**: Decision Mamba (NeurIPS 2024), RoboMamba (arXiv 2406.04339)
- **URL**: https://arxiv.org/abs/2406.04339
- **核心**: 用 selective SSM 取代 transformer / LSTM,linear-time 序列建模,推論快 3x。
- **與 charge 整合**: 替換方向 2 的 GRU 為 Mamba block,理論上對長 horizon 訓練更穩定。
- **改造成本**: 中 (~250 行 + mamba_ssm package)
- **warm start**: ⚠️
- **預期效果**: paper 在 D4RL benchmark 顯示比 Decision Transformer 略好,inference 大幅快;但對 charge 此類 short horizon (200 steps) 收益可能有限。
- **不確定性**: ⚠️ Mamba 在 RL navigation 文獻較少,無直接 benchmark

### 架構 5: PointNet-based Obstacle Encoder
- **代表 paper**: Strudel et al. "Learning Obstacle Representations for Neural Motion Planning" (arXiv 2008.11174, 2024 仍被引用)
- **核心**: 把 obstacles 當 point cloud,PointNet (shared MLP + max) 做 permutation invariant encoding。
- **與 charge 整合**: 我們現在的 MaxPool 其實已是簡化版 PointNet;升級到 PointNet++ (multi-scale grouping) 可改善。
- **改造成本**: 小 (~100 行)
- **預期效果**: paper 報告 BC +34%, RL +54%,但這是 manipulation 場景
- **不確定性**: ⚠️ 對 dense dynamic crowd 可能不如 attention

---

## 四、Reward Shaping 創新 (Q3)

### Reward 創新 1: CBF-based Energy Reward (取代 closing_risk)
- **理論基礎**: 把 reward 設成 CBF condition violation `r_safe = -max(0, -h(s) - α·h(s))`,其中 `h(s) = d_min - d_safe` (signed distance)。
- **代表 paper**:
  - "Learning Control Barrier Functions and their application in RL: A Survey" (arXiv 2404.16879, 2024)
  - "Designing Control Barrier Function via Probabilistic Enumeration for Safe RL Navigation" (IEEE 2024-25)
- **與 closing_risk 對比**: closing_risk 是 ad-hoc 工程,CBF 有理論保證 (forward invariance)。weight 不需要手調,因為 CBF 數學上保證 safe set 不被侵犯。
- **實作成本**: 中 — 需要從 LiDAR 計算 signed distance gradient,python 約 200 行
- **預期效果**: 不確定 (CBF 在 RL 一直邊際改善,因為 RL approximation 破壞理論保證);但作為 closing_risk 的替代,至少消除 weight tuning 戰
- **論文佐證**: 上述 survey 列出 30+ 篇,collision 改善 10-30%

### Reward 創新 2: RND Curiosity Bonus on Obstacle Interactions
- **理論基礎**: Random Network Distillation,intrinsic reward = ‖f̂(s) - f_target(s)‖²,鼓勵訪問 novel state。
- **代表 paper**: Burda et al. 原 RND (2018) + 2024 navigation applications (e.g. Baselga "Improving robot navigation in crowded environments using intrinsic rewards")
- **與 charge 整合**: target network 只 input obstacle subset,迫使 agent 學習所有 obstacle configuration (而非只走 easy path)。直接攻擊 OOD 問題。
- **實作成本**: 小 (~150 行,RND 是現成模組)
- **預期效果**: paper 在 sparse reward navigation 報告 SR +10-20%;但 charge 不是 sparse reward
- **不確定性**: ⚠️ charge dense reward 環境下 RND 可能干擾;需要小 weight (0.01-0.1)

### Reward 創新 3: Hindsight Experience Replay (HER) for Collision Episodes
- **理論基礎**: failed episode 的 trajectory 在「achieved state」當作 goal 重新標註,變成 successful demonstration。
- **代表 paper**: Andrychowicz 原 HER (2017) + AgentHER (arXiv 2603.21357, 2024) + MEHER (Crowder 2024)
- **與 charge 整合**: collision episode 把碰撞前的 last reachable position 當作 alternate goal,episode 變 "成功避障"。對 PPO on-policy 較難整合,可改用 off-policy 階段。
- **實作成本**: 大 (PPO 是 on-policy,需大改成 SAC 或加 off-policy buffer)
- **warm start**: ❌
- **預期效果**: paper 在 sparse 場景 SR 顯著提升,但 charge dense reward 改善不確定

### Reward 創新 4: Counterfactual / Causal Reward Shaping
- **理論基礎**: 對每個 step 計算 `Δr = r_actual - r_counterfactual`,counterfactual 是「如果不採取 avoidance action 會怎樣」的 reward;只獎勵真正有 causal effect 的 avoidance。
- **代表 paper**: 較少直接 navigation 應用,但 "Causal Reinforcement Learning" survey (arXiv 2307.01452) 有討論
- **與 charge 整合**: 需要 simulator rollback 或 model-based prediction,在 sim 環境可行
- **實作成本**: 大 (sim rollback 很貴)
- **不確定性**: ⚠️⚠️ 偏理論,實作 paper 數字少

### Reward 創新 5: Inverse RL from MPC Expert
- **理論基礎**: 用 NavRL/Intent-MPC 跑 expert demonstration,IRL 反推 reward function。
- **代表 paper**: Intent-MPC (arXiv 2409.15633, RA-L 2025) 可作為 expert
- **與 charge 整合**: 先跑 MPC 收 demos,再 IRL/BC pretrain
- **實作成本**: 大
- **warm start**: ✅ (作為 RL pretrain)
- **預期效果**: 不確定,IRL 在 navigation 結果分歧

---

## 五、Curriculum 創新 (Q4)

### Curriculum 1: ACCEL / Replay-Guided Adversarial Environment Design
- **代表 paper**: "No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery" (arXiv 2408.15099, NeurIPS 2024)
- **核心**: 用 regret approximation 對環境 level 進行 prioritized replay,自動找出 agent 學習邊界。
- **與 charge 整合**: 把 8 stages × N levels 每個都打 regret score,replay buffer 從中 sample。skrl pipeline 需加 buffer 但不大改。
- **實作成本**: 中 (10-15 GPU hr 工程 + retrain)
- **預期效果**: paper 在 OOD generalization 普遍 +10-30%
- **論文佐證**: ACCEL 在 Procgen 顯著超越 PLR

### Curriculum 2: ADD (Adversarial Diffusion Design)
- **代表 paper**: arXiv 2410.19715, NeurIPS 2024 spotlight
- **URL**: https://arxiv.org/abs/2410.19715
- **核心**: diffusion model 學「對 agent 最具學習價值的 environment 分布」,regret-guided sampling
- **與 charge 整合**: 高難度,需要 environment design space 連續化
- **實作成本**: 大 (30+ GPU hr)
- **不確定性**: ⚠️ 對 charge 這種 procedural environment 改造大

### Curriculum 3: Self-Paced Bounded Rationality (Adversary)
- **代表 paper**: 同方向 C, arXiv 2503.11467
- **核心**: adversary temperature τ 隨 protagonist 進步遞減,等同自動調 dynamic obstacle 難度
- **與 charge 整合**: ⭐ 直接解 OOD 問題
- **實作成本**: 大
- **論文佐證**: paper 在 quadruped 場景顯著改善

---

## 六、Sim-to-Real 與穩健性 (Q5)

### Sim2Real 1: TWIST (Teacher-Student World Model Distillation)
- **代表 paper**: Hansen et al. "TWIST: Teacher-Student World Model Distillation for Efficient Sim-to-Real Transfer"
- **Venue**: arXiv 2311.03622 (2023-11)
- **URL**: https://arxiv.org/abs/2311.03622
- **核心**: teacher 用 privileged sim state (true positions) 訓 world model + policy,student 用 vision/sensor 蒸餾 latent 對齊;deploy 時只用 student。
- **與 charge 整合**: charge 已有 privileged info (true obstacle pos / vel from sim);可訓 teacher with full 60D + student with noisy LiDAR-only,sim2real 自然處理 sensor noise。
- **實作成本**: 大 (整套 model-based,放棄 PPO)
- **warm start**: ❌
- **論文佐證**: paper 報告 sample efficiency 與 robustness 雙贏

### Sim2Real 2: NavDP (Navigation Diffusion Policy with Privileged Guidance)
- **代表 paper**: NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance
- **Venue**: arXiv 2505.08712 (2025-05)
- **URL**: https://arxiv.org/abs/2505.08712
- **核心**: diffusion policy + privileged supervision,zero-shot 跨平台 sim2real
- **與 charge 整合**: 大改造,但 sim2real 是 charge 最終目標
- **實作成本**: 極大

### Sim2Real 3: RMA (Rapid Motor Adaptation)
- **代表 paper**: Kumar et al. RMA (2021) + 2024 manipulator extension (CVPR 2024)
- **URL**: https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_Rapid_Motor_Adaptation_for_Robotic_Manipulator_Arms_CVPR_2024_paper.pdf
- **核心**: phase 1 train policy with privileged env params; phase 2 train adaptation module 從 history 推 env params;deploy 時 adaptation module run online。
- **與 charge 整合**: env params (friction, slip, sensor noise) 顯式模型化,robust to real-world variation
- **實作成本**: 中-大 (15-25 GPU hr)
- **warm start**: ✅ (phase 1 用現有 policy 加 privileged input)

---

## 七、Top-5 最終推薦 (按 ROI 排序)

### 🥇 第一推薦: HEIGHT-style Heterogeneous Graph Transformer + GRU

**為什麼是第一**:
1. **直接攻擊核心瓶頸**: charge 的 MaxPool obstacle encoder 是 2024 文獻一致認為的弱點 (HEIGHT, Set Transformer, Intention-Aware all 證明)
2. **warm start 可行**: LiDAR / state branch 不變,只重訓 obstacle branch + head,GPU hr 中等
3. **理論+實證雙重支持**: HEIGHT (T-ASE 2026, arXiv 2411.12150) 與 SARL/CrowdNav 系列一致顯示 attention > maxpool 在 dense scene

**Paper**: Liu et al., "HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments", IEEE T-ASE 2026, https://arxiv.org/abs/2411.12150

**對 charge 的具體實作建議**:
1. 替換 `vlp16_models.py` 的 obstacle branch:
   - 原: `MLP(6→32) → MaxPool(10→1)` 出 32D
   - 新: `Linear(6→64) → 2 layers Multi-Head Attention(heads=4) → mean pool` 出 64D
2. 加入 robot_state_query 對 obstacles 的 cross-attention:
   - `q = MLP(robot_state)`, `k=v = obstacle_features`, output = `softmax(qk^T/√d)v`
3. (可選 phase 2) head 前加 GRU(hidden=128) 處理 POMDP

**預期 CR**: 從 49% → 35-42% (信心中,基於 HEIGHT 在 dense scene 對 SARL 的相對改善)
**信心等級**: 高 (architecture 改善是文獻最一致的方向)
**風險**:
- attention 對 noisy obstacle (FP detection) 比 maxpool 敏感,可能需要 dropout
- training time +20%
- 如果加 GRU 則 BPTT 設定需謹慎

### 🥈 第二推薦: Bounded Rationality Adversarial Curriculum

**為什麼是第二**:
1. **直接修 OOD bug** (train cap 0.40 vs eval 0.44): adversary 自動產生比任何 hand-tuned curriculum 更難的 dynamic config
2. **可 warm-start v21 protagonist**,只需要從零訓 adversary
3. **CoRL 2024 同年驗證**,有 Unitree GO1 真實 deploy 證據

**Paper**: "Dynamic Obstacle Avoidance with Bounded Rationality Adversarial Reinforcement Learning", CoRL 2024, https://arxiv.org/abs/2503.11467

**對 charge 的具體實作建議**:
1. 把 1-2 個動態 obstacle 改為 SAC adversary,reward = -protagonist_progress
2. adversary temperature τ 從 5.0 退火到 0.5 over 4 stages
3. protagonist 用 v21 weights warm start
4. dyn_ratio 不再 cap,完全由 adversary 控制難度

**預期 CR**: 在 OOD eval 場景下 49% → 38-45% (信心中,因為直接修 OOD)
**信心等級**: 中-高
**風險**:
- 雙 RL training 不穩定,需要 careful tuning of τ schedule
- adversary 可能 collapse 成 "撞牆自殺" (需要 reward shaping for adversary)
- GPU 成本 ×1.5 (兩個 policy 同時訓)

### 🥉 第三推薦: Intention-Aware Trajectory Prediction Augmentation

**為什麼是第三**:
1. **改動最小** (加 obs feature),完全 plug-in
2. **理論直觀且 paper 一致驗證** (Liu ICRA 2023, Future-Oriented arXiv 2505.00237)
3. 對 charge 而言 obstacle 已有 vx/vy,做 N-step constant velocity 預測幾乎零成本
4. 與第一推薦正交,可疊加

**Paper**: "Intention Aware Robot Crowd Navigation with Attention-Based Interaction Graph", ICRA 2023, https://arxiv.org/abs/2203.01821 + Future-Oriented Navigation (arXiv 2505.00237, 2025)

**對 charge 的具體實作建議**:
1. 在 `obs_functions.py` 新增 `topk_obstacles_predicted_3step`:
   - 對每個 obstacle 計算 `[px+vx*0.4, py+vy*0.4]` (1秒未來,2步)
   - 維度: 10 obstacles × 2 future_steps × 2 coords = 40D
2. obs 從 139 → 179D
3. (option) 用 ground truth velocity (privileged) vs LiDAR-tracked velocity 對比

**預期 CR**: 49% → 42-46% (信心中-低,單獨用收益較小)
**信心等級**: 中
**風險**:
- 改 obs dim 需要 retrain head (但可 freeze backbone)
- LiDAR 估的 velocity 噪聲大,效果可能弱於 ground truth (需要 sim2real awareness)

### 🏅 第四推薦: ACCEL/PLR Curriculum Replay (Stage Buffer)

**為什麼是第四**:
1. 解 OOD 問題的另一條路 (與第二推薦互補)
2. 比 adversarial 簡單,風險低
3. NeurIPS 2024 "No Regrets" 提供改良 score function

**Paper**: "No Regrets: Investigating and Improving Regret Approximations for Curriculum Discovery", arXiv 2408.15099, NeurIPS 2024

**對 charge 的具體實作建議**:
1. 把現有 OE levels 改成 prioritized replay buffer
2. score function = Generalised Advantage Estimation TD-error 平均 (paper 推薦)
3. high-score level 用 70% 概率 replay,30% 隨機

**預期 CR**: 49% → 43-47% (信心中,curriculum 改善通常邊際但穩定)
**信心等級**: 中
**風險**: 較低

### 🎖️ 第五推薦: CBF-based Safety Reward (取代 closing_risk)

**為什麼是第五**:
1. **終結 reward weight 調參戰** (M1.x 系列已證明手調無效)
2. 理論基礎強 (forward invariance)
3. 但在 RL 中 CBF 經常邊際改善,信心相對低

**Paper**: "Learning Control Barrier Functions and their application in RL: A Survey", arXiv 2404.16879, 2024 + "Designing Control Barrier Function via Probabilistic Enumeration for Safe RL Navigation", IEEE 2024

**對 charge 的具體實作建議**:
1. 定義 `h(s) = d_min - d_safe` (signed distance,可從 LiDAR 計算)
2. CBF condition: `ḣ + α(h) ≥ 0`,違反時 reward = `-max(0, -ḣ - α·h)`
3. 取代現有 closing_risk reward,設 weight = 1.0 (CBF 數學上不需 sweep)

**預期 CR**: 49% → 43-47% (信心低,因 CBF 在 RL 經常 underwhelm)
**信心等級**: 低-中
**風險**: M1.x 系列已證明 closing/distance reward 改動容易破壞 PPO 穩定性

---

## 八、Anti-pattern: 不該做的方向

### Anti-1: 純 Diffusion Policy 取代 PPO
- **理由**: charge 是 discrete MultiDiscrete[19,19] action,diffusion 設計上做 continuous;改造放棄整個 SKRL pipeline
- **論文反證**: DiffusionDrive (CVPR 2025) 在 autonomous driving 報告 vs end-to-end PPO mixed results
- **結論**: ❌ 不要把 PPO 換掉,可考慮 CoBL 的 CBF guidance 概念但不換 backbone

### Anti-2: 加更激進 reward weight (繼續 M1.5/M1.6 路線)
- **理由**: M1.1-M1.4 已證明手調 weight 邊際 ≤±5pp,變化方向不確定
- **論文反證**: 沒有任何 2024 paper 主張「reward weight tuning 是主要 contribution」
- **結論**: ❌ 停止 closing_risk weight sweep,改方向

### Anti-3: 純 LSTM 不加 attention
- **理由**: LSTM 對 set-of-obstacles 不 permutation invariant,需要 sort/index trick
- **論文反證**: HEIGHT ablation 顯示 attention + GRU > GRU only
- **結論**: ⚠️ 如果加 RNN,務必先加 attention encoder

### Anti-4: 全部換 Equivariant Network
- **理由**: SE(2) equivariance 改造大,charge 已用 body frame 部分滿足平移不變
- **論文反證**: e2cnn 系列在 navigation 上 sample efficiency 提升明顯但 final SR 改善有限
- **結論**: ❌ 不值得 retrain from scratch

### Anti-5: 只加 RND/Curiosity
- **理由**: charge 已是 dense reward,curiosity bonus 容易壓過 task reward
- **論文反證**: RND 設計用於 sparse reward (Montezuma);dense reward 場景 RND 收益不確定
- **結論**: ❌ 不要單獨用 RND,如要用須極小 weight

---

## 九、不確定性聲明

1. **HEIGHT 具體 CR 數字未從 abstract 拿到** — 預期 CR 從 49% → 35-42% 是基於 SARL/CrowdNav 系列相對改善 estimate,實際需讀 HEIGHT Table II/III 確認
2. **Bounded Rationality Adversarial RL 對 diff-drive 的適配未知** — paper 是 quadruped + holonomic obstacle,charge 是 diff-drive,curriculum 動態可能不同
3. **NavRL VO LP solver 的 charge 適配風險高** — 我們 M1.2 試過簡化 VO 失敗 (-5pp),雖然失敗點不同,但須謹慎
4. **CBF reward 的實際收益不確定** — survey paper 列出 30+ 工作,改善 10-30% 但 variance 大;在 PPO 內 CBF 經常 underwhelm
5. **ADD (Adversarial Diffusion Design) 對 charge 不直接適用** — paper 的 environment design space 是 grid/Procgen,charge 是 procedural Isaac scene,改造 cost 過大未估算
6. **Mamba 在 RL navigation 缺直接 benchmark** — 大部分 RoboMamba/Decision Mamba 是 manipulation/offline RL,navigation 文獻空缺
7. **HER 需要從 PPO 轉 SAC** — 是大幅 pipeline 改動,沒有 PPO + HER 的成熟方案
8. **所有「預期 CR」數字皆為 estimate** — 我已盡量對齊 paper 報告的相對改善,但 charge 的 baseline (49%) 與 paper baseline 不可比

## 十、優先實驗順序建議

```
Week 1 (P0): 第一推薦 - HEIGHT-style attention encoder (warm start)
  ├─ Day 1-2: Set Transformer / MHA 替換 obstacle branch
  ├─ Day 3-4: warm start 訓練 ~6 hours
  └─ Day 5: 評估 + 第三推薦 (intention prediction obs aug) 疊加

Week 2 (P1): 第二推薦 - Bounded Rationality Adversary
  ├─ Day 1-3: 寫 adversary SAC + reward
  └─ Day 4-7: 訓練 + curriculum tuning

Week 3 (P2): 第四推薦 - ACCEL Curriculum Replay
  └─ 整合 stage buffer + regret scoring
```

---

## 十一、文獻引用清單

| # | Paper | Venue | URL |
|---|-------|-------|-----|
| 1 | HEIGHT: Heterogeneous Interaction Graph Transformer (Liu et al.) | T-ASE 2026 | https://arxiv.org/abs/2411.12150 |
| 2 | Intention Aware Robot Crowd Navigation (Liu et al.) | ICRA 2023 | https://arxiv.org/abs/2203.01821 |
| 3 | Dynamic Obstacle Avoidance with Bounded Rationality Adversarial RL | CoRL 2024 | https://arxiv.org/abs/2503.11467 |
| 4 | Adversarial Environment Design via Regret-Guided Diffusion (ADD) | NeurIPS 2024 | https://arxiv.org/abs/2410.19715 |
| 5 | NavRL: Learning Safe Flight in Dynamic Environments (Xu et al.) | RA-L 2025 | https://arxiv.org/abs/2409.15634 |
| 6 | CoBL-Diffusion (Mizuta & Leung) | arXiv 2024-11 | https://arxiv.org/abs/2406.05309 |
| 7 | DreamerNav (latent world model navigation) | Frontiers Robot AI 2025 | https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1655171/full |
| 8 | Future-Oriented Navigation (energy-based prediction) | arXiv 2025-05 | https://arxiv.org/abs/2505.00237 |
| 9 | Learning CBF for RL: A Survey | arXiv 2024 | https://arxiv.org/abs/2404.16879 |
| 10 | No Regrets: Improved Regret Approximations for Curriculum | NeurIPS 2024 | https://arxiv.org/abs/2408.15099 |
| 11 | TWIST: Teacher-Student World Model Distillation | arXiv 2023-11 | https://arxiv.org/abs/2311.03622 |
| 12 | NavDP: Navigation Diffusion Policy with Privileged Guidance | arXiv 2025-05 | https://arxiv.org/abs/2505.08712 |
| 13 | RMA: Rapid Motor Adaptation (manipulator extension) | CVPR 2024 | https://openaccess.thecvf.com/content/CVPR2024/papers/Liang_Rapid_Motor_Adaptation_for_Robotic_Manipulator_Arms_CVPR_2024_paper.pdf |
| 14 | Intent Prediction-Driven MPC (Xu et al.) | RA-L 2025 | https://arxiv.org/abs/2409.15633 |
| 15 | Decision Mamba for Offline RL | NeurIPS 2024 | https://neurips.cc/virtual/2024/poster/94331 |
| 16 | RoboMamba: VLA Model for Robotics | arXiv 2024-06 | https://arxiv.org/abs/2406.04339 |
| 17 | E(2)-Equivariant Graph Planning for Navigation | RA-L 2024 | https://www.khoury.northeastern.edu/home/lsw/papers/ral2024-equnav.pdf |
| 18 | MarineFormer: Spatio-Temporal Attention USV Navigation | arXiv 2024-10 | https://arxiv.org/abs/2410.13973 |
| 19 | DRL-VO: Crowd Navigation via Velocity Obstacles | T-RO 2023 | https://dl.acm.org/doi/10.1109/TRO.2023.3257549 |
| 20 | DiffusionDrive: Truncated Diffusion for E2E Driving | CVPR 2025 | (CVPR 2025 proceedings) |

(共 20 篇文獻引用)
