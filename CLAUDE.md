# Charge-SKRL Navigation — 消融實驗設定指南

> 給 PC-B 的 Claude Code 讀取，確保開發環境與 PC-A 一致。

## 第二大腦 (Obsidian Vault)

`/home/aa/Documents/Obsidian Vault/` 是用戶的知識庫（第二大腦）。
- **訓練相關問題必須優先到此搜尋**，再回答或做決策
- 主要子目錄：`isaaclab/`（Isaac Lab 訓練筆記）、`rnn/`（RNN 相關）
- 包含歷次實驗分析、設計決策、debug 記錄、文獻筆記
- 搜尋方式：`grep -r "關鍵字" "/home/aa/Documents/Obsidian Vault/"` 或讀取特定檔案

## 環境設定

### 1. Python 環境

```bash
conda activate env_isaaclab
# Python 3.11, Isaac Sim 5.1, CUDA 13.0
# 路徑: /home/aa/miniconda3/envs/env_isaaclab/
```

### 2. 專案路徑

```
/home/aa/IsaacLab/                              ← 專案根目錄
├── isaaclab.sh                                  ← 統一啟動腳本
├── scripts/reinforcement_learning/skrl/         ← 訓練腳本
│   ├── train_charge_ac.py                       ← 主訓練入口
│   ├── wandb_trainer.py                         ← WandB 整合
│   ├── console_summary.py                       ← 終端摘要
│   ├── vlp16_models.py                          ← 3-branch 網路
│   └── diagnostics/
│       └── ablation_metrics.py                  ← 15 指標診斷模組
└── source/isaaclab_tasks/.../charge_skrl/       ← 任務配置
    ├── cfg/
    │   ├── charge_cfg.py                        ← 機器人 USD 配置
    │   ├── charge_env_cfg_vlp16.py              ← VLP16 基礎配置
    │   └── charge_env_cfg_vlp16_curriculum.py   ← 課程 + NavRL + 消融 configs
    ├── mdp/
    │   ├── rewards/navrl_rewards.py             ← NavRL dense rewards
    │   ├── rewards/gap_rewards.py               ← Gap-seeking rewards
    │   ├── actions/discrete_differential_drive.py
    │   ├── actions/safety_shield.py             ← Safety shield
    │   ├── observations/obs_functions.py        ← 139D 觀測
    │   ├── wall_layout.py                       ← 牆壁幾何 + per-env 查詢
    │   └── events/walls.py                      ← 牆壁隨機化
    ├── curriculum/goal_obstacle_curriculum.py    ← 8 階段課程
    ├── goal_command.py                          ← Multi-goal 命令
    └── agents/skrl_ppo_cfg_vlp16.yaml           ← PPO 超參數
```

### 3. 機器人 USD 路徑

USD 檔案已納入 repo：`assets/usd/charge/charge.usd` (12KB)

`charge_cfg.py` 自動搜尋順序：
1. 環境變數 `CHARGE_USD_PATH`（最高優先）
2. `{repo_root}/assets/usd/charge/charge.usd`（repo 內，跨機器免設定）
3. `/home/aa/usd/charge/charge.usd`（本機 fallback）

通常不需額外設定，checkout 後即可用。

### 4. Git Remote

```bash
# charge_skrl remote (自訂 training code)
git remote add charge_skrl git@github.com:me0608623/charge_skrl.git
git fetch charge_skrl

# 切換到消融實驗 branch
git checkout charge_skrl/abl
```

### 5. WandB 設定

```bash
export WANDB_PROJECT=charge_skrl
# WandB API key 需在 PC-B 單獨設定: wandb login
```

## 任務與觀測

| 項目 | 值 |
|------|-----|
| Task | `Isaac-Navigation-Charge-VLP16-Curriculum-NavRL` |
| 觀測 | 139D: ego(4) + goal(2) + LiDAR(72) + obstacles(60) + time(1) |
| 動作 | MultiDiscrete([19, 19]) — linear accel × angular vel |
| 物理 | v_max=1.0 m/s, a_max=0.5 m/s², ω_max=0.25π rad/s, dt=0.2s |
| 碰撞 | LiDAR ≤ 0.45m (body_radius=0.35 + buffer=0.10) |
| 場景 | 20×20m, 4 boundary walls (1.0m thick), 8 internal wall slots, 10 obstacles |

## 消融實驗指令

**所有實驗用同一個 --task，只靠 CLI 參數切換 variant。**

### Family 1: v_gate (goal attraction 衰減)

```bash
# baseline
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl1_vgate-baseline_s1_ne512 --v_gate_mode baseline

# floor (near obstacle: 保留 20% goal attraction)
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl1_vgate-floor_s1_ne512 --v_gate_mode floor

# softer (延後關閉: d_attenuate 1.0→0.6)
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl1_vgate-softer_s1_ne512 --v_gate_mode softer
```

### Family 2: progress_gate (danger zone 門檻)

```bash
# baseline
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl2_progress-baseline_s1_ne512 --progress_gate_mode baseline

# delayed_negative (d_danger 0.8→0.55: 縮小 danger zone)
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl2_progress-delayed_neg_s1_ne512 --progress_gate_mode delayed_negative

# weaken_negative (negative_scale 0.5→0.2: 降低負區強度)
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl2_progress-weaken_neg_s1_ne512 --progress_gate_mode weaken_negative
```

### Family 3: gap_reward (繞行指引)

```bash
# off (baseline, 不帶 --use_gap_reward)
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl3_gap-off_s1_ne512

# heading w=1.0
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl3_gap-heading_w1.0_s1_ne512 --use_gap_reward --gap_reward_type heading --gap_reward_weight 1.0

# heading w=2.0
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl3_gap-heading_w2.0_s1_ne512 --use_gap_reward --gap_reward_type heading --gap_reward_weight 2.0

# clearance w=1.0
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl3_gap-clearance_w1.0_s1_ne512 --use_gap_reward --gap_reward_type clearance --gap_reward_weight 1.0

# clearance w=2.0
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl3_gap-clearance_w2.0_s1_ne512 --use_gap_reward --gap_reward_type clearance --gap_reward_weight 2.0
```

### Family 4: shield (action-level safety)

```bash
# off (baseline, 不帶 --use_safety_shield)
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl4_shield-off_s1_ne512

# soft (d_safe<1.2 線性降速, d_safe<0.55 停止)
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl4_shield-soft_s1_ne512 --use_safety_shield --shield_mode soft

# hard (d_safe<0.55 強制 v=0)
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 512 --headless --seed 1 \
  --run_name abl4_shield-hard_s1_ne512 --use_safety_shield --shield_mode hard
```

## 診斷指標

所有 NavRL* 任務自動啟用 `AblationMetricsLogger`，以下指標記錄到 WandB:

| WandB Key | 說明 |
|-----------|------|
| `ablation/stuck_count` | 連續 ≥5 步 speed<0.02 的次數 |
| `ablation/freeze_ratio` | speed<0.02 的步數佔比 |
| `ablation/oscillation_score` | v_toward 符號翻轉頻率 |
| `ablation/retreat_ratio` | 近障礙時 v_toward<0 的比例 |
| `ablation/progress_near_obs` | 近障礙時的 progress 平均 |
| `ablation/d_safe_mean` | 全局平均安全距離 |
| `ablation/shield_rate` | shield 介入比例 (Family 4) |

## 課程階段表 (8 階段)

| Phase | Goals | Static | Dynamic | Walls | γ |
|-------|-------|--------|---------|-------|---|
| 1 | 8 | 0 | 0 | 0-2 | 0.990 |
| 2 | 7 | 1 | 1 | 0-3 | 0.992 |
| 3 | 6 | 2 | 2 | 1-4 | 0.994 |
| 4 | 5 | 3 | 3 | 1-5 | 0.995 |
| 5 | 4 | 4 | 4 | 2-6 | 0.996 |
| 6 | 3 | 5 | 5 | 2-7 | 0.996 |
| 7 | 2 | 6 | 6 | 3-8 | 0.997 |
| 8 | 1 | 7 | 7 | 4-8 | 0.998 |

## PC-B 首次設定 Checklist

1. Clone repo: `git clone git@github.com:me0608623/charge_skrl.git IsaacLab` 或 add remote
2. `git checkout charge_skrl/abl`
3. `conda activate env_isaaclab`
4. USD 已在 repo 內 (`assets/usd/charge/charge.usd`)，無需額外設定
5. `wandb login`
6. 跑一個快速測試: `./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py --task Isaac-Navigation-Charge-VLP16-Curriculum-NavRL --num_envs 4 --headless --timesteps 10`
7. 確認 WandB 出現 `ablation/*` metrics

## 注意事項

- 語言偏好: 中文(繁體) 用於註解/文件，英文用於代碼
- 訓練不要用 `conda run`（會 buffer stdout），直接用 `PYTHONUNBUFFERED=1 ./isaaclab.sh`
- 不要修改 `--task` 名稱，消融實驗全靠 CLI 參數切換
- 所有新 CLI 參數預設值 = baseline 行為

## v3 改動 (2026-06-08)

從 `wd_sa1_v3` 起的新版本，與 v2 不相容（obs distribution + reward 都不同）：

| 改動 | 數值 |
|------|------|
| LiDAR `r_min` | 0.9 → **0.25** (VLP-16 實測：表面→人物中心 0.2m + 物理半徑 0.0515m) |
| Reward `penalty_smoothness` | 0 → **0.005** (全 stage，抗單幀抽動，作 inductive bias) |
| Curriculum | 新建 `warp_drive_single_agent_v3` |
| 新監控指標 | `jitter/per_env_omega_std_p95`、`jitter/per_env_ratio_flip_rate_p95` (抽 16 envs) |

**啟動指令**:
```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa1_v3 --headless \
  --run_name sa1_v3_ne1024_s42
```

⚠️ v3 不能 resume v1/v2 ckpt — 必須從頭訓練。

## MARL RNN 備忘

- `scripts/reinforcement_learning/skrl/train_marl_rnn.py` 已支援 `--grad-clip-mode {merged,separate}`。
- 目前預設值是 `separate`，用來保持既有行為不變。
- `merged`：對 `policy_head + value_head` 做一次 joint `clip_grad_norm_`。
- `separate`：分別對 `policy_head`、`value_head` 各做一次 `clip_grad_norm_`。
- 本機已查證的參考實作：
  - SKRL PPO: `/home/aa/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/skrl/agents/torch/ppo/ppo.py` 使用 merged clip。
  - SKRL A2C: `/home/aa/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/skrl/agents/torch/a2c/a2c.py` 使用 merged clip。
  - SKRL MAPPO: `/home/aa/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/skrl/multi_agents/torch/mappo/mappo.py` 使用 merged clip。
  - Warp Drive trainer: `new_warp_drive/warp_drive/training/trainer.py` 對單一 model 做一次 clip，不是 actor/value 分開 clip。
- `train_marl_rnn.py` 的 RL optimizer 是單一 joint loss：`policy_loss + vf_coeff * value_loss - entropy_loss`，因此 `merged` 應視為較接近標準 PPO baseline；`separate` 應視為額外 heuristic，只有在做對照實驗時才宣稱其效果。
- `train_marl_rnn.py` 使用 GAE：
  - `compute_gae`: `delta = r + gamma * V_next - V`
  - `returns = advantages + values`
- 不要把 WD 的 `a2c_ken.py` critic clamp 搬回 PPO：
  - WD 的 return/advantage 寫法不是 GAE，而是 normalized returns 直接減 value。
  - WD critic clamp 在 PPO 小 policy loss 條件下會把 `vf_loss` 拉到固定量級，不適合這份 `train_marl_rnn.py`。
- 做 `merged` vs `separate` 比較時，除了 `--grad-clip-mode` 以外，其餘條件都要固定：
  - seed
  - timesteps / iterations
  - `vf_coeff`
  - `max_grad_norm`
  - rollout / batch / epochs
  - learning rate
  - curriculum / eval protocol
