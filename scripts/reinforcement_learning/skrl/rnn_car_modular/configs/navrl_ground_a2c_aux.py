"""NavRL-Ground V8 A2C + aux config -- v21-style dense reward.

Reward 由 navrl_dense_v8.py 外部計算（8 dense components），
與 WD sparse 一樣不依賴 env 的 reward_manager。

Task 用 WD env（env reward 歸零），reward 完全由 module 控制:
    task = "Isaac-Navigation-Charge-VLP16-Curriculum-WD"
    reward_profile = "navrl_dense_v8"

權重/參數定義在: rnn_car_modular/rewards/navrl_dense_v8.py → REWARD_TERMS dict
要改權重 → 直接改那隻檔案，或複製一份做 v9。
"""

from rnn_car_modular.experiment_config import ExperimentConfig

CONFIG = ExperimentConfig(
    name="navrl_ground_a2c_aux",
    description="NavRL-Ground v8 dense reward, A2C, WD 7D aux, SA2 fixed",
    # --- Task: WD env（reward 歸零，由 module 外部計算）---
    task="Isaac-Navigation-Charge-VLP16-Curriculum-WD",
    curriculum_version="warp_drive_single_agent_v1",
    initial_stage=2,
    fixed_stage=True,
    obstacle_mode="rule_based",
    # --- Profiles ---
    reward_profile="navrl_dense_v8",    # 自包含 module（公式+權重+參數）
    algorithm_profile="a2c_wd",         # A2C with WD-style update
    aux_profile="wd_7d_geometry",       # WD auxiliary prediction task
    encoder_profile="wd_exact_rnn",     # WD RNN encoder
    critic_profile="symmetric",         # symmetric critic (same obs as policy)
    # --- Training budget ---
    num_envs=1024,
    rollout_length=300,
    timesteps=180000,
    # --- Optimizer ---
    lr=2e-4,
    rnn_lr=5e-4,
    vf_coeff=0.5,
    gamma=0.99,                         # v21 default (slightly lower than WD 0.991)
    gae_lambda=0.95,
    normalize_return=True,
    value_init_bias=0.0,
    # --- Algorithm ---
    use_a2c=True,
    ent_coeff_linear=0.05,
    ent_coeff_angular=0.10,
    wd_update_clip=True,
    # --- Auxiliary task ---
    aux_seq_len=15,
    aux_seq_batch_size=256,
    aux_grad_clip=0.5,
    # --- Sensor ---
    lidar_no_noise=True,
    action_table_sample_size=0,
    no_resume_optimizer=True,
    tags=("navrl-ground-v8", "sa2", "a2c", "aux", "dense-reward"),
)
