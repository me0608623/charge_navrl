# Claude Code Prompt — IsaacLab WD/RNN Modular Trainer Refactor

你是 Claude Code，請在 `/home/aa/IsaacLab` repo 內協助實作 WD/RNN modular trainer 的第一階段重構。

## 0. 重要背景與目標

目前專案正在訓練 IsaacLab Charge navigation WD/RNN car。現有主訓練入口是：

```text
scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py
```

它目前約 3600 行，已支援：

- WD-style sparse reward
- A2C / PPO flag
- wd_exact_rnn / raw_fc_rnn
- TBPTT aux RNN 7D geometry target
- rule-based / learned obstacle agent
- curriculum phase sync
- WandB diagnostics / checkpoints

但是 reward / scene / algorithm / aux / model 邏輯逐漸混在同一支 script 中，未來需要能快速參考其他論文 reward 設計，例如 NavRL / HEIGHT / TTC risk / collision-risk aux，因此需要模組化。

請注意：

```text
這次不是要重寫整個 trainer，也不是要改訓練行為。
第一階段目標是「不破壞現有訓練」的 profile/facade + reward 抽取準備。
```

## 1. 必須先做的檢查

開始前請執行並閱讀：

```bash
cd /home/aa/IsaacLab
git log --oneline -20
git status --short
```

注意：worktree 很髒，另一個 AI agent 也在改。不要做 destructive git command：

```text
禁止 git reset / git checkout . / git clean
```

## 2. 目前已存在的半模組化結構

請先閱讀這些檔案，不要只靠記憶：

```text
scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py
scripts/reinforcement_learning/skrl/rnn_car_wdclean/train.py
scripts/reinforcement_learning/skrl/rnn_car_wdclean/trainer.py
scripts/reinforcement_learning/skrl/rnn_car_wdclean/args.py
scripts/reinforcement_learning/skrl/rnn_car_wdclean/rewards.py
scripts/reinforcement_learning/skrl/rnn_car_wdclean/buffers.py
scripts/reinforcement_learning/skrl/rnn_car_wdclean/models.py
scripts/reinforcement_learning/skrl/rnn_car_wdclean/metrics.py
scripts/reinforcement_learning/skrl/utils/wd_aux_targets.py
scripts/reinforcement_learning/skrl/utils/charge_env_overrides.py
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_skrl/curriculum/phases/wd_single_agent_v1.py
```

目前 repo 已有：

```text
scripts/reinforcement_learning/skrl/rnn_car_wdclean/
```

這是一個已經從 `train_rnn_car_wdclip.py` 抽出部分 module 的 prototype，但它可能落後目前 canonical train script。請不要盲目取代 canonical script。

## 3. IsaacLab 框架限制：不要破壞 AppLauncher / Hydra / env wrapper

IsaacLab entrypoint 有硬性順序：

```python
from isaaclab.app import AppLauncher
parser = create_parser()
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# 之後才 import gym / torch-heavy / isaaclab_tasks trainer code
```

不要把 IsaacLab / gym / env 建立移到 AppLauncher 前面。

不要破壞：

```text
hydra_task_config(args_cli.task, "skrl_cfg_entry_point")
gym.make(args_cli.task, cfg=env_cfg)
SkrlVecEnvWrapper(env, ml_framework="torch")
```

模組化應該包在 AppLauncher 後的 trainer loop 內，不應改 IsaacLab env framework。

## 4. 第一階段要做什麼

請做「低風險 Phase 0/1」，不要大重構：

### 4.1 新增 profile CLI，但暫時不改行為

在 canonical `train/train_rnn_car_wdclip.py` 加入 profile 參數：

```text
--reward_profile {wd_sparse,navrl_dense,hybrid_progress,ttc_risk}
--scene_profile TEXT
--algorithm_profile {a2c_wd,ppo_clip}
--aux_profile {wd_7d_geometry,none,future_collision_risk}
--encoder_profile TEXT
```

預設值必須等價目前行為：

```text
reward_profile = wd_sparse
scene_profile = curriculum_version value or wd_single_agent_v1
algorithm_profile = a2c_wd if use_a2c else ppo_clip
aux_profile = none if disable_aux_training else wd_7d_geometry
encoder_profile = charge_encoder_mode
```

這一階段 profile 主要用於：

- run_metadata.yaml
- WandB config
- console print
- 未來模組切換

不能因 profile 預設值改變現有訓練結果。

### 4.2 新增 profile resolver module

建立：

```text
scripts/reinforcement_learning/skrl/rnn_car_modular/__init__.py
scripts/reinforcement_learning/skrl/rnn_car_modular/profiles.py
```

`profiles.py` 建議提供：

```python
from dataclasses import dataclass

@dataclass
class TrainerProfiles:
    reward_profile: str
    scene_profile: str
    algorithm_profile: str
    aux_profile: str
    encoder_profile: str


def resolve_profiles(args_cli) -> TrainerProfiles:
    ...


def validate_profiles(p: TrainerProfiles) -> None:
    ...


def profiles_to_dict(p: TrainerProfiles) -> dict:
    ...
```

驗證規則：

```text
reward_profile 先只允許 wd_sparse 真正執行；其他 profile 可接受但應 raise NotImplementedError，除非只是 metadata。
algorithm_profile 必須與 --use_a2c / --use_ppo 一致；若不一致，請清楚 print warning 或直接根據 flags resolve。
aux_profile=none 必須對應 disable_aux_training=True；若使用 aux_profile none 但未 disable，應自動設 disable 或報錯，請選較安全的報錯。
```

### 4.3 Reward module facade

建立：

```text
scripts/reinforcement_learning/skrl/rnn_car_modular/rewards/__init__.py
scripts/reinforcement_learning/skrl/rnn_car_modular/rewards/base.py
scripts/reinforcement_learning/skrl/rnn_car_modular/rewards/wd_sparse.py
```

`base.py`：

```python
from typing import Protocol
import torch

class RewardModule(Protocol):
    name: str
    def compute(self, env_unwrapped, actions: torch.Tensor, terminated: torch.Tensor,
                truncated: torch.Tensor, context: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...
```

`wd_sparse.py`：

- 先包裝現有 `compute_wd_charge_reward`，不要複製兩份邏輯造成 drift。
- 可以從 `scripts/reinforcement_learning/skrl/rnn_car_wdclean/rewards.py` 或 canonical train script 現有 function import。
- 如果 import path 不穩，先建立 wrapper 但內部仍呼叫 canonical function。
- 必須保持 breakdown key 不變：

```text
goal_reward
wall_hit_reward
obs_hit_reward
floor_reward
action_reward
goal_reached
wall_collision
obs_collision
other_death
```

### 4.4 將 canonical train script 接上 profile metadata

在 `train/train_rnn_car_wdclip.py` 中：

1. import `resolve_profiles`, `profiles_to_dict`。
2. 在解析 args 後或 main loop 初期 resolve。
3. console print：

```text
[PROFILES] reward=wd_sparse scene=... algorithm=... aux=... encoder=...
```

4. WandB config / run_metadata.yaml 若已有 metadata 寫入，加入：

```python
"profiles": profiles_to_dict(profiles)
```

如果目前 run_metadata 寫入位置不明，先搜尋 `run_metadata` / `wandb.init`，不要隨便重寫 logging。

### 4.5 不要做的事

這次不要做：

```text
不要新增 SAC
不要新增 privileged critic
不要重寫 rollout buffer
不要改 WD reward 數值
不要改 curriculum phase 內容
不要讓 rnn_car_wdclean 取代 canonical train script
不要刪除舊檔案
不要改正在跑的 training command 行為
```

## 5. 驗證要求

請至少執行：

```bash
cd /home/aa/IsaacLab
/home/aa/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/profiles.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/rewards/base.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/rewards/wd_sparse.py
```

如果可行，再跑 help smoke：

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py --help >/tmp/wdclip_help.txt 2>&1
```

確認 help 中有：

```text
--reward_profile
--algorithm_profile
--aux_profile
```

不要啟動長訓練。

## 6. 完成後回報格式

請回報：

```text
1. 修改檔案清單
2. 新增 profile CLI 與預設值
3. 是否保證預設行為不變
4. 驗證命令與結果
5. 後續 Phase 2 建議：真正抽 algorithm / aux / scene module
```

## 7. 設計原則

這次重構的核心是：

```text
先讓實驗條件可被 profile 命名與記錄，再逐步讓 profile 控制實作。
```

也就是：

```text
metadata first, behavior later
```

這樣可以避免一次大改破壞 IsaacLab 訓練框架，也能讓之後參考其他論文 reward 設計時，有清楚插槽。
