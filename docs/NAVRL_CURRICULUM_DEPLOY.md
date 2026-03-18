# NavRL Curriculum — 部署與訓練指南

本文件提供完整的檔案結構、訓練指令、以及交給另一個 Claude Code agent 部署此訓練代碼的標準流程。

---

## 1. 檔案結構

所有路徑相對於 IsaacLab 專案根目錄（`/home/aa/IsaacLab/`）。

```
IsaacLab/
├── scripts/
│   ├── train_curriculum_navrl.sh                          # 訓練啟動腳本
│   └── reinforcement_learning/skrl/
│       ├── train_charge_ac.py                             # 主訓練程式（所有 task 共用）
│       ├── wandb_trainer.py                               # WandB metric logger
│       ├── console_summary.py                             # 終端摘要
│       ├── training_debug_logger.py                       # GPU debug metrics
│       ├── training_params_logger.py                      # 超參數記錄
│       ├── vlp16_models.py                                # VLP16 神經網路架構
│       └── charge_models.py                               # 基礎模型定義
│
└── source/isaaclab_tasks/isaaclab_tasks/manager_based/
    └── locomotion/velocity/config/charge_skrl/
        ├── __init__.py                                    # Gymnasium task 註冊
        ├── cfg/
        │   ├── __init__.py                                # Config exports
        │   ├── charge_cfg.py                              # 機器人物理配置
        │   ├── charge_env_cfg_vlp16.py                    # VLP16 基礎環境
        │   └── charge_env_cfg_vlp16_curriculum.py         # ★ Curriculum + NavRL 配置
        ├── mdp/
        │   └── rewards/
        │       ├── __init__.py                            # Reward exports
        │       ├── navrl_rewards.py                       # ★ 3 個 NavRL dense reward
        │       ├── potential_based_rewards.py              # PBRS 獎勵
        │       ├── smoothness_rewards.py                  # 平滑懲罰
        │       ├── goal_rewards.py                        # 目標獎勵
        │       └── safety_rewards.py                      # 安全獎勵
        ├── curriculum/
        │   └── goal_obstacle_curriculum.py                # ★ 5-stage v10 curriculum
        ├── agents/
        │   └── skrl_ppo_cfg_vlp16.yaml                   # PPO 超參數
        └── ...（其他共用模組）
```

標 ★ 為本次新增/修改的核心檔案。

---

## 2. 訓練指令

```bash
# 正式訓練（6144 parallel envs, headless, WandB enabled）
./scripts/train_curriculum_navrl.sh

# 小規模快速驗證（6 envs）
./scripts/train_curriculum_navrl.sh --num_envs 6

# 等效的直接指令
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
    --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
    --num_envs 6144 --headless
```

---

## 3. 已註冊的 Gymnasium Task

| Task ID | 配置類別 | 說明 |
|---|---|---|
| `Isaac-Navigation-Charge-VLP16-Curriculum` | `ChargeNavigationEnvCfgVLP16Curriculum` | v9 原版（純死亡機制） |
| **`Isaac-Navigation-Charge-VLP16-Curriculum-NavRL`** | `ChargeNavigationEnvCfgVLP16CurriculumNavRL` | **v10 NavRL dense rewards** |

兩者共用相同的 scene / observations / actions / terminations / curriculum，唯一差異是 reward 配置。

---

## 4. Reward 結構（NavRL v10）

| # | Term | Weight | 範圍 | 函數 |
|---|------|--------|------|------|
| 1 | `reaching_goal` | +250 | {0, 1} | 繼承 — 到達目標 |
| 2 | `velocity_to_goal` | +15 | [-1, 1] | 雙向速度獎勵（背離懲罰） |
| 3 | `safety_log_distance` | +3 | [-3, 3] | 速度耦合 log 安全距離 |
| 4 | `safe_progress` | +30 | [-1, 1] | PBRS × sigmoid 安全 gate |
| 5 | `potential_progress` | 0 | — | 被 safe_progress 取代 |
| 6 | `acceleration_penalty` | -0.05 | — | deadzone 版 |
| 7 | `angular_velocity_penalty` | -0.05 | — | context-aware 版 |

繼承自父類的 collision_terminal / near_obstacle / time / velocity_too_low 全部 weight=0。

---

## 5. Curriculum 結構（v10：5 階段 + TO-aware）

| Phase | Stage | 靜態 | 動態 | Goal 數 | Goal 距離 | 升級條件 |
|---|---|---|---|---|---|---|
| 1 | 1 | 0 | 0 | 8 | 2-13m | SR>72% TO<30% |
| 2 | 2 | 3 | 2 | 3 | 2-13m | SR>65% CR<40% TO<35% |
| 3 | 3 | 5 | 3 | 2 | 2-13m | SR>60% CR<40% TO<40% |
| 4a | 4 | 5 | 5 | 1 | 2-13m | SR>55% CR<35% TO<40% |
| 4b | 5 | 5 | 8 | 1 | 2-13m | 最終階段 |

所有 stage 共用相同 reward，只改環境參數。降級條件包含 TO 過高。

---

## 6. WandB 指標

自動記錄（Isaac Lab RewardManager 內建）：

| WandB Key | 說明 |
|---|---|
| `Info / Episode_Reward/velocity_to_goal` | 每 episode 朝目標速度獎勵 |
| `Info / Episode_Reward/safety_log_distance` | 每 episode 安全距離獎勵 |
| `Info / Episode_Reward/safe_progress` | 每 episode 安全進度獎勵 |
| `Info / Episode_Reward/reaching_goal` | 每 episode 到達獎勵 |

自訂 debug metrics（TrainingDebugLogger）：

| WandB Key | 說明 |
|---|---|
| `task/success_rate` | 成功率 |
| `task/collision_rate` | 碰撞率 |
| `task/timeout_rate` | 超時率 |

Curriculum stage 指標：

| WandB Key | 說明 |
|---|---|
| `Info / stage` | 當前 curriculum stage |
| `Info / success_rate` | Curriculum 窗口內 SR |
| `Info / collision_rate` | Curriculum 窗口內 CR |
| `Info / timeout_rate` | Curriculum 窗口內 TO |

---

## 7. 交給另一個 Claude Code Agent 部署的流程

以下是一個結構化的 prompt，可以直接交給另一個 Claude Code agent 執行部署：

```
你需要在 IsaacLab 專案中部署 NavRL Curriculum 訓練系統。請按以下步驟執行：

### Step 1：環境確認
1. 確認 IsaacLab 根目錄位置（預設 /home/aa/IsaacLab/）
2. 確認 conda 環境：conda activate env_isaaclab
3. 確認 GPU 可用：nvidia-smi
4. 確認 Isaac Lab 可執行：./isaaclab.sh --help

### Step 2：切換到正確分支
git fetch --all
git checkout curriculum_navrl_reward
git pull

### Step 3：驗證檔案完整性
確認以下 6 個核心檔案存在：
- source/.../charge_skrl/mdp/rewards/navrl_rewards.py（3 個 reward 函數）
- source/.../charge_skrl/mdp/rewards/__init__.py（export 3 個新函數）
- source/.../charge_skrl/cfg/charge_env_cfg_vlp16_curriculum.py（NavRL config）
- source/.../charge_skrl/cfg/__init__.py（export 新 config）
- source/.../charge_skrl/__init__.py（註冊新 task）
- source/.../charge_skrl/curriculum/goal_obstacle_curriculum.py（v10 curriculum）

### Step 4：小規模驗證（6 envs，確認不 crash）
./scripts/train_curriculum_navrl.sh --num_envs 6
觀察：
- 環境是否成功初始化
- "[Curriculum v10] 初始化" 訊息是否出現
- reward 計算是否有 NaN 錯誤
- 跑 30 秒後 Ctrl+C 結束

### Step 5：正式訓練
./scripts/train_curriculum_navrl.sh
或等效：
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
    --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL \
    --num_envs 6144 --headless

### Step 6：監控 WandB
訓練開始後到 https://wandb.ai 確認：
1. 專案 charge_skrl 下出現新 run
2. 以下指標存在且有數據：
   - Info / Episode_Reward/velocity_to_goal
   - Info / Episode_Reward/safety_log_distance
   - Info / Episode_Reward/safe_progress
   - Info / stage（應從 1 開始）
   - task/success_rate
3. Stage 1 訓練穩定後 SR 應上升到 70%+

### Step 7：觀察 Curriculum 升級
- Stage 1→2：SR > 72% 且 TO < 30%
- Stage 2→3：SR > 65% 且 CR < 40% 且 TO < 35%
- 如果升級後 SR 急跌，觀察是否自動降級

### 故障排除
- NVIDIA segfault：加 --headless，確認非 SSH + display 問題
- ImportError：確認在 curriculum_navrl_reward 分支，非 main
- NaN reward：檢查 LiDAR sensor 是否正確初始化
- WandB 無資料：確認 headless 模式（GUI 模式下 WandB 被禁用）
```

---

## 8. 架構概覽

```
train_charge_ac.py
    │
    ├── gym.make("Isaac-Navigation-Charge-VLP16-Curriculum-NavRL")
    │       │
    │       ├── ChargeNavigationEnvCfgVLP16CurriculumNavRL
    │       │       ├── scene:     MySceneCfgVLP16_20x20  (20×20m, 4 內部牆, 10 障礙物)
    │       │       ├── obs:       139D (ego+goal+lidar+obstacles+time)
    │       │       ├── actions:   Discrete(361) = 19×19
    │       │       ├── rewards:   RewardsCfgVLP16NavRL  ← 本次核心修改
    │       │       ├── events:    EventCfgVLP16Curriculum
    │       │       └── curriculum: CurriculumCfgVLP16 → goal_obstacle_curriculum()
    │       │
    │       └── ManagerBasedRLEnv.step()
    │               ├── reward_manager.compute() → 自動累計 per-term episodic sum
    │               └── reward_manager.reset()   → extras["log"]["Episode_Reward/<name>"]
    │
    ├── SkrlVecEnvWrapper (Standard AC)
    │
    ├── SKRL PPO Agent (vlp16_models.py)
    │       ├── Policy:  3-branch CNN → 128D → Categorical(361)
    │       └── Critic:  3-branch CNN → 128D → V(s)
    │
    └── WandBSequentialTrainer
            ├── infos["log"] → "Info / Episode_Reward/<name>" → wandb.log()
            └── _update_task_metrics() → task/success_rate, collision_rate, timeout_rate
```
