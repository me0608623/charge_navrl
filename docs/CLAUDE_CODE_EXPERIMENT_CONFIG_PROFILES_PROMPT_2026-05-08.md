# Claude Code Prompt — Experiment Config Profiles / LEGO-style Run Composition

你是 Claude Code。請在 `/home/aa/IsaacLab` repo 內實作下一階段：用一份 experiment config 來組合本次訓練的 scene / phase / reward / algorithm / aux / encoder / budget，減少 CLI 參數長度，並為未來消融實驗提供 LEGO 式組裝。

## 0. 必讀規則

開始前先執行：

```bash
cd /home/aa/IsaacLab
git log --oneline -20
git status --short
```

注意：worktree 很髒，另一個 AI agent 也在改。禁止 destructive git command：

```text
不要 git reset
不要 git checkout .
不要 git clean
不要刪除舊檔
```

語言：程式碼英文，註解/文件可繁中。

## 1. 現況

上一階段已完成 profile facade：

```text
scripts/reinforcement_learning/skrl/rnn_car_modular/profiles.py
scripts/reinforcement_learning/skrl/rnn_car_modular/rewards/base.py
scripts/reinforcement_learning/skrl/rnn_car_modular/rewards/wd_sparse.py
```

Canonical training entrypoint 仍是：

```text
scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py
```

目前 profile CLI 已有：

```text
--reward_profile
--scene_profile
--algorithm_profile
--aux_profile
--encoder_profile
```

但目前仍需要輸入很多 CLI，例如 `--num_envs --rollout_length --timesteps --lr --rnn_lr --vf_coeff --ent_coeff...`。

這次目標是新增：

```text
--experiment_config <name_or_path>
```

讓使用者可以用一份 config 決定大部分設定，CLI 只用來 override 少數欄位。

## 2. 設計目標

我要能像 LEGO 一樣組合一次訓練：

```text
ExperimentConfig =
  scene_profile
+ phase_schedule
+ reward_profile
+ algorithm_profile
+ aux_profile
+ encoder_profile
+ critic_profile
+ budget_profile
+ obstacle_behavior_profile
```

短期只做 config resolution / metadata / CLI defaults，不改訓練演算法行為。

中期才讓 reward_profile / algorithm_profile 真正 runtime dispatch。

## 3. 不要破壞 IsaacLab AppLauncher / Hydra

不要改 AppLauncher 順序：

```python
from isaaclab.app import AppLauncher
parser = create_parser()
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# 之後才 import gym / isaaclab_tasks / trainer
```

不要破壞：

```text
hydra_task_config(...)
gym.make(args_cli.task, cfg=env_cfg)
SkrlVecEnvWrapper(env, ml_framework="torch")
```

## 4. 新增檔案

請新增：

```text
scripts/reinforcement_learning/skrl/rnn_car_modular/experiment_config.py
scripts/reinforcement_learning/skrl/rnn_car_modular/configs/__init__.py
scripts/reinforcement_learning/skrl/rnn_car_modular/configs/registry.py
scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa2_a2c_aux_lowent.py
scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa2_a2c_aux.py
scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa2_a2c_noaux.py
scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa2_ppo_noaux.py
```

## 5. ExperimentConfig schema

在 `experiment_config.py` 定義 dataclass。

建議 schema：

```python
from dataclasses import dataclass, field, asdict
from typing import Any

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str = ""

    # IsaacLab task / scene
    task: str = "Isaac-Navigation-Charge-VLP16-Curriculum-WD"
    curriculum_version: str = "warp_drive_single_agent_v1"
    initial_stage: int = 1
    fixed_stage: bool = True
    scene_profile: str = "warp_drive_single_agent_v1"
    obstacle_mode: str = "rule_based"

    # profiles
    reward_profile: str = "wd_sparse"
    algorithm_profile: str = "a2c_wd"
    aux_profile: str = "wd_7d_geometry"
    encoder_profile: str = "wd_exact_rnn"
    critic_profile: str = "symmetric"
    budget_profile: str = "custom"

    # training budget
    num_envs: int = 1024
    rollout_length: int = 300
    timesteps: int = 180000
    seed: int = 42

    # RL hyperparams
    lr: float = 2e-4
    rnn_lr: float = 5e-4
    vf_coeff: float = 0.5
    gamma: float = 0.991
    gae_lambda: float = 0.95
    normalize_return: bool = True
    value_init_bias: float = 0.0
    max_grad_norm: float = 1.0

    # A2C/PPO
    use_a2c: bool = True
    ppo_epochs: int = 2
    mini_batches: int = 16
    clip_eps: float = 0.1

    # entropy / WD update caps
    ent_coeff_linear: float = 0.0
    ent_coeff_angular: float = 0.0
    wd_update_clip: bool = True
    wd_actor_update_clip: float = 8.0
    wd_critic_update_clip: float = 30.0

    # aux
    disable_aux_training: bool = False
    aux_seq_len: int = 15
    aux_burn_in: int = 0
    aux_seq_batch_size: int = 256
    aux_grad_clip: float = 0.5

    # safety/logging
    lidar_no_noise: bool = True
    action_table_sample_size: int = 0
    log_interval: int = 10
    save_interval: int = 100

    # checkpoint / resume
    checkpoint: str | None = None
    no_resume_optimizer: bool = True

    # metadata
    tags: tuple[str, ...] = ()
    notes: str = ""
```

如果某些 CLI 欄位在目前 parser 還沒有，請先檢查再決定是否加。不要硬加不相容欄位。

## 6. Registry 設計

在 `configs/registry.py`：

```python
from .wd_sa2_a2c_aux import CONFIG as wd_sa2_a2c_aux
...

EXPERIMENT_CONFIGS = {
    "wd_sa2_a2c_aux": wd_sa2_a2c_aux,
    "wd_sa2_a2c_aux_lowent": wd_sa2_a2c_aux_lowent,
    "wd_sa2_a2c_noaux": wd_sa2_a2c_noaux,
    "wd_sa2_ppo_noaux": wd_sa2_ppo_noaux,
}

def get_experiment_config(name_or_path: str) -> ExperimentConfig:
    ...
```

`get_experiment_config` 支援：

```text
1. registry name，例如 wd_sa2_a2c_aux_lowent
2. Python file path，例如 /path/to/my_config.py，該檔要有 CONFIG
```

不需要支援 YAML，先用 Python config 就好，因為本專案 config 需要註解與型別。

## 7. CLI apply 規則

在 `args.py` 或 canonical parser 加：

```text
--experiment_config TEXT
--print_experiment_config
```

實作一個函式：

```python
apply_experiment_config(args_cli, cfg: ExperimentConfig) -> None
```

規則：

```text
config 提供預設值，CLI 明確輸入者優先。
```

但 argparse 很難知道哪些是使用者明確輸入、哪些是 default。請用保守策略：

第一階段採取「config 覆蓋 parser defaults，但若使用者傳了 override flag 則保留 CLI」。

可用 sys.argv 檢查 explicit flags，例如：

```python
def _was_cli_provided(flag: str, argv: list[str]) -> bool:
    return flag in argv or any(x.startswith(flag + "=") for x in argv)
```

對每個 config 欄位建立對應 flag，例如：

```text
num_envs -> --num_envs
rollout_length -> --rollout_length
use_a2c -> --use_a2c / --use_ppo
fixed_stage -> --fixed_stage
normalize_return -> --normalize_return
lidar_no_noise -> --lidar_no_noise
```

如果 CLI 沒提供，就用 config 值覆蓋 args_cli。

重要：boolean flag 要小心：

```text
如果 cfg.fixed_stage=True 且 CLI 沒寫 --fixed_stage，應設 args_cli.fixed_stage=True。
如果 cfg.disable_aux_training=True 且 CLI 沒寫 --disable_aux_training，應設 True。
如果 cfg.use_a2c=False，應等價於 --use_ppo。
```

## 8. 必須支援的內建 configs

### 8.1 wd_sa2_a2c_aux

用途：目前 WD SA2 A2C + aux 主線。

```python
CONFIG = ExperimentConfig(
    name="wd_sa2_a2c_aux",
    description="Fixed SA2, A2C-WD, WD sparse reward, WD 7D aux enabled",
    task="Isaac-Navigation-Charge-VLP16-Curriculum-WD",
    curriculum_version="warp_drive_single_agent_v1",
    initial_stage=2,
    fixed_stage=True,
    obstacle_mode="rule_based",
    reward_profile="wd_sparse",
    algorithm_profile="a2c_wd",
    aux_profile="wd_7d_geometry",
    encoder_profile="wd_exact_rnn",
    critic_profile="symmetric",
    num_envs=1024,
    rollout_length=300,
    timesteps=180000,
    lr=2e-4,
    rnn_lr=5e-4,
    vf_coeff=0.5,
    gamma=0.991,
    gae_lambda=0.95,
    normalize_return=True,
    value_init_bias=0.0,
    use_a2c=True,
    ent_coeff_linear=0.05,
    ent_coeff_angular=0.10,
    wd_update_clip=True,
    aux_seq_len=15,
    aux_seq_batch_size=256,
    aux_grad_clip=0.5,
    lidar_no_noise=True,
    action_table_sample_size=0,
    no_resume_optimizer=True,
    tags=("wd", "sa2", "a2c", "aux"),
)
```

### 8.2 wd_sa2_a2c_aux_lowent

同上，但：

```text
ent_coeff_linear=0.01
ent_coeff_angular=0.02
```

### 8.3 wd_sa2_a2c_noaux

同上，但：

```text
aux_profile="none"
disable_aux_training=True
rnn_lr=0.0
```

### 8.4 wd_sa2_ppo_noaux

```text
algorithm_profile="ppo_clip"
use_a2c=False
aux_profile="none"
disable_aux_training=True
rnn_lr=0.0
lr=1e-4
ppo_epochs=2
mini_batches=16
clip_eps=0.1
```

## 9. Integration with profiles.py

`resolve_profiles(args_cli)` 應該在 experiment_config 套用後再呼叫。

流程：

```text
parse args
if args_cli.experiment_config:
    cfg = get_experiment_config(args_cli.experiment_config)
    apply_experiment_config(args_cli, cfg, sys.argv)
resolve_profiles(args_cli)
validate_profiles(...)
```

如果目前 canonical script 裡 resolve_profiles 位置不同，請調整成上述順序。

## 10. Metadata / print

加入 console print：

```text
[EXPERIMENT_CONFIG] name=wd_sa2_a2c_aux_lowent description=...
[EXPERIMENT_CONFIG] applied fields: num_envs=1024 rollout=300 timesteps=180000 ...
[PROFILES] reward=wd_sparse scene=... algorithm=... aux=... encoder=...
```

WandB config / run_metadata 若已有加入 profiles，請加入：

```python
"experiment_config": experiment_config_to_dict(cfg) if cfg else None
```

## 11. 驗證

請執行：

```bash
cd /home/aa/IsaacLab
/home/aa/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/profiles.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/experiment_config.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/registry.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa2_a2c_aux.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa2_a2c_aux_lowent.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa2_a2c_noaux.py \
  scripts/reinforcement_learning/skrl/rnn_car_modular/configs/wd_sa2_ppo_noaux.py
```

Help smoke：

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py --help >/tmp/wdclip_help_expconfig.txt 2>&1
```

確認：

```bash
grep -E "experiment_config|print_experiment_config" /tmp/wdclip_help_expconfig.txt
```

Config print smoke，若不會啟動 Isaac Sim 最好；若目前架構必須走 AppLauncher，請不要長跑，只測 --help / py_compile 即可。

另外新增簡單 unit test script 或 Python one-liner，測：

```text
get_experiment_config("wd_sa2_a2c_aux_lowent") 回傳正確 ent coeff
apply_experiment_config 對空 CLI args 能設 initial_stage=2, fixed_stage=True, use_a2c=True
CLI explicit --num_envs 512 能 override config num_envs=1024
wd_sa2_ppo_noaux 會讓 use_a2c=False, disable_aux_training=True
```

## 12. 完成回報

請回報：

```text
1. 修改/新增檔案
2. 新增 config names
3. example command before/after
4. CLI override 規則
5. py_compile/help/unit test 結果
6. 是否保證沒有改變不使用 --experiment_config 時的預設行為
```

## 13. Example command 目標

未來希望可以從很長：

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --task Isaac-Navigation-Charge-VLP16-Curriculum-WD \
  --num_envs 1024 --headless --seed 42 --use_a2c \
  --charge_encoder_mode wd_exact_rnn \
  --curriculum_version warp_drive_single_agent_v1 \
  --initial_stage 2 --fixed_stage \
  --obstacle_mode rule_based \
  --rollout_length 300 --timesteps 180000 \
  --lr 2e-4 --rnn_lr 5e-4 --vf_coeff 0.5 \
  --normalize_return --gamma 0.991 --gae_lambda 0.95 \
  --aux_seq_len 15 --aux_seq_batch_size 256 \
  --ent_coeff_linear 0.01 --ent_coeff_angular 0.02 \
  --lidar_no_noise --action_table_sample_size 0 \
  --run_name my_run
```

變成：

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa2_a2c_aux_lowent \
  --checkpoint <path> \
  --run_name my_run \
  --headless
```

並允許 override：

```bash
--experiment_config wd_sa2_a2c_aux_lowent --num_envs 512 --timesteps 90000
```
