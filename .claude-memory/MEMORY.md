# IsaacLab Charge Navigation Project Memory

## 🌐 Global Integration
- [CLAUDE_MEM_INTEGRATION.md](CLAUDE_MEM_INTEGRATION.md) — Claude-Mem 全域記憶整合（自動捕捉工具調用、語義搜索）
- Global DB: `~/.claude-mem/claude-mem.db` (SQLite)
- Vector Store: `~/.claude-mem/chroma/` (Semantic search)
- Worker: `localhost:37777` (auto-launched)

## Project Structure
- Custom robot nav RL training in Isaac Lab with SKRL/SB3
- Main task config: `source/isaaclab_tasks/.../charge_skrl/` (SKRL version)
- Training scripts: `scripts/reinforcement_learning/skrl/`
- Key files: `train_charge.py`, `vlp16_models.py`, `aac_wrapper.py`, `wandb_trainer.py`

## Architecture (v2 — 2026-03-07)
- v2: Actor 79D obs → 91D rl_input, Critic 91D + 50D privileged (obstacles only) = 141D asymmetric
- Obs layout: ego(4) + goal(2) + static/LiDAR(72) + obs/Top10×6D(60) + time(1) = 139D
- obs state: topk_obstacles_6d → [x,y,vx,vy,r,m] per object, body-frame, LOS occlusion
- VLP16 discrete action: Discrete(361) = 19×19 flat index, center-symmetric (idx 9 = zero)
- Dynamic acceleration bounds: a ∈ [max(-a_max, (-v_max-v)/dt), min(a_max, (v_max-v)/dt)]
- Physical limits: v_max=1.0 m/s, a_max=0.5 m/s², ω_max=0.25π rad/s, dt=0.2s
- Allows reverse: velocity ∈ [-v_max, +v_max]
- Policy head: CategoricalMixin, 361 logits (hidden 128→128→361)
- No frame stacking (num_stack=1)
- 3-branch feature extractor: Conv1d(LiDAR→64D) + ObsMLP(60D→32D MaxPool) + StateMLP(7D→32D) = 128D
- Internal maze walls: 6 segments (reduced from 10)
- Mixed parallel: 50% empty / 30% static / 20% dynamic

## Active / Open (2026-07-13) — SA3 deploy_dense 崩塌
- [finding_sa3_deploy_dense_value_led_collapse_20260713.md](finding_sa3_deploy_dense_value_led_collapse_20260713.md) — **必讀**：tzq22v0w 二次崩（value-led）、①不通過、②暫停；主嫌 mean_only+A2C+進場 LR 非網路結構；Obsidian 全文見 vault `bug/2026-07-13_sa3_deploy_dense_二次崩塌_actor_critic訓練規則.md`
- 下一刀建議：單變因 `adv_norm=full` 或鎖 LR 2e-4；**勿**先加 `penalty_speed_near_obs` 查崩因
- 早避靜態 flag（②，底座穩後才用）：`--penalty_speed_near_obs 0.2`

## Verified Correct (不需重查)
- [rnn_reset_on_done_verified.md](rnn_reset_on_done_verified.md) — RNN hidden state 在 episode reset 正確歸零，不跨場景污染（含 obstacle/aux cache + TBPTT 邊界）

## Critical Bugs Fixed (2026-03-05)
See [debugging.md](debugging.md) for details.

1. **NaN in obstacle observations** - Scene entities can have NaN positions after resets/collisions
2. **IEEE 754: NaN * 0 = NaN** - Invalid slot zeroing via multiplication doesn't clear NaN
3. **Temporal mismatch** - shared_states from infos corresponded to t+1, not t
4. **RunningStandardScaler poisoning** - NaN in inputs permanently corrupts scaler stats

## SHOWSTOPPER Bugs Found 2026-04-07 (v18b/v19 discovery)
See [v18_discovery_critical_bugs.md](v18_discovery_critical_bugs.md) for full details.

5. **Bug A: Kinematic obstacles + contact threshold 0.1N** — All collisions undercounted, CR ≈ 0% 即使物理穿透。修復: 加 `obstacle_collision_geometric` 純幾何碰撞 termination。
6. **Bug B: LiDAR 內建 distractor (rate=0.002) + Unoise** — lidar.min 永遠 ≈ 0，policy 學到不信任 LiDAR。修復: 加 `--lidar_no_noise` flag + `r_min/z_filter` 參數。
7. **真實 LiDAR 校準: min_range = 0.9m**（用戶實測）— sim 必須匹配此盲區。已加 `r_min=0.9` 預設。
8. **所有 v17 之前的訓練都受 Bug A+B 影響** — 需要在修復後重訓 v20 baseline 才能作為對照組。

## Environment Details
- Python env: `/home/aa/miniconda3/envs/env_isaaclab/`
- Run training: `./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge.py --task ... --headless`
- `conda run` fully buffers stdout - use direct python + PYTHONUNBUFFERED=1

## User Preferences
- Language: Chinese (Traditional) for comments/docs, English for code
- Uses WandB for metrics tracking
- Prefers detailed diagnostic output during debugging
- See [naming_convention.md](naming_convention.md) for run_name 命名規範

## Design Decisions
- [v16_directional_gate.md](v16_directional_gate.md) — v16 方向性 gate：解決 v15 Stage 8 retreat-dominant (63.8%)，gate 改用 goal-direction cone LiDAR
- [design_obstacles_60d_unnecessary.md](design_obstacles_60d_unnecessary.md) — Obstacles 60D 在 WD 是 placeholder、部署不可用，應移除

## References
- [cli_versions.md](cli_versions.md) — reward_mode / curriculum_version / CLI flags 各版本差異與歷史 runs 成效
- [reference_ntut_thesis_format.md](reference_ntut_thesis_format.md) — 北科論文官方格式(J1 114.02.01)權威來源 + 114-2 離校時程（上傳 8/16、離校 8/24），論文 §36

## Feedback
- [feedback_decode_each_stage_rnn_health.md](feedback_decode_each_stage_rnn_health.md) — reluFix lineage 每階段(SA1→SA5)一定要跑 aux 解碼確認 RNN 健康正常,不可跳過(解碼是必要驗證關卡)
- [feedback_obs_index_layout.md](feedback_obs_index_layout.md) — 79D obs layout: [0]=accel [1]=speed [2]=omega [3]=radius [4:6]=goal [6:78]=lidar [78]=time
- [feedback_reference_train_car.md](feedback_reference_train_car.md) — WD 參考代碼是 train_rnn_car.py（car），不是 train_spot（Spot）
- [feedback_reward_analysis_rigor.md](feedback_reward_analysis_rigor.md) — Reward 分析必須考慮完整 return 結構，不能只看 weight ratio
- [feedback_read_obsidian.md](feedback_read_obsidian.md) — 每次回答前必須讀取 Obsidian vault 相關內容
- [feedback_analysis_sources.md](feedback_analysis_sources.md) — 訓練分析用四來源交叉驗證：CSV + console + checkpoint + WandB API
- [feedback_shield_vs_policy.md](feedback_shield_vs_policy.md) — Shield 不能 substitute reward shaping，必須先讓 policy reward gradient 對齊目標再加 shield
- [feedback_obsidian_notes_path.md](feedback_obsidian_notes_path.md) — 筆記必須寫到 `/home/aa/Documents/Obsidian Vault/isaaclab/`，不能只存 .claude memory
- [feedback_use_ai_delegate.md](feedback_use_ai_delegate.md) — 必須用 ai-delegate 把 research/notes/log-analysis 丟給 Gemini/Codex 省 Opus tokens
- [feedback_scan_timestamp.md](feedback_scan_timestamp.md) — /scan 報告必須用 `date` 取系統當下時間（台北 UTC+8）
- [feedback_grad_clip_conclusion_scope.md](feedback_grad_clip_conclusion_scope.md) — 實驗結論必須 scope 到具體 regime，不能推廣為一般性定理
- [feedback_training_analysis_workflow.md](feedback_training_analysis_workflow.md) — 訓練分析三層流程：live monitoring → post-run report → analyze_wandb_run.py
- [feedback_training_run_yaml.md](feedback_training_run_yaml.md) — 每次啟動/監控訓練必須產生 run_metadata.yaml（目的、參數、架構、預期效果）
- [feedback_monitor_interval.md](feedback_monitor_interval.md) — 訓練監控嚴格 15 分鐘一次，不可自行延長
- [feedback_monitor_protocol_sa2cont2.md](feedback_monitor_protocol_sa2cont2.md) — SA2 cont2 五類監控 protocol + GREEN/YELLOW/RED 閾值 + 立即停止條件
- [feedback_normalize_episode_metrics.md](feedback_normalize_episode_metrics.md) — SR/CR/TO 必須用已完成 episodes 正規化，不能報告 raw %
- [feedback_rnn_notes_path.md](feedback_rnn_notes_path.md) — RNN 相關筆記寫到 `/home/aa/Documents/Obsidian Vault/rnn/`
- [feedback_etime_format.md](feedback_etime_format.md) — ps etime 格式：MM:SS / HH:MM:SS / D-HH:MM:SS，不要誤讀
- [feedback_monitor_wandb_full.md](feedback_monitor_wandb_full.md) — 監控必須每次從 WandB 拉所有分組指標，不能只看 console log
- [feedback_arrow_spacing.md](feedback_arrow_spacing.md) — 箭頭後面接數字/文字時必須空一格（`→ 32` 而非 `→32`），避免渲染遮擋
- [feedback_ask_yaml_before_train.md](feedback_ask_yaml_before_train.md) — 每次啟動訓練前必須問用戶要用哪個 experiment_config YAML
- [feedback_monitor_chinese.md](feedback_monitor_chinese.md) — Cron 監控報告必須用繁體中文回答
- [feedback_cron_train_summary.md](feedback_cron_train_summary.md) — Cron 監控結束後自動執行 /train_summary 產生報告
- [feedback_entropy_intervention_framework.md](feedback_entropy_intervention_framework.md) — Entropy 過快下降時的觀察 vs 介入標準 + 4 方案優先序
- [feedback_per_env_sampling.md](feedback_per_env_sampling.md) — 評估 policy 抽動/震盪必須 per-env p95 取樣，全 env 平均會掩蓋單幀現象
- [finding_p95flip_entropy_confound.md](finding_p95flip_entropy_confound.md) — 訓練 p95_flip 被 entropy 探索污染（RED 可能是假象），抽動驗收必須用 deterministic rollout（play_rnn_car.py --jitter_eval）
- [feedback_let_runs_finish.md](feedback_let_runs_finish.md) — 不要提前建議 kill 訓練；讓 run 跑完整 budget，除非真壞掉（NaN/OOM/死）；中途 RED 但 SR 健康就繼續
- [feedback_stall_autorecover.md](feedback_stall_autorecover.md) — 確診 stall（log停滯+GPU 0%+py-spy卡同行）且用戶失聯時，授權自動 kill + 從最近 ckpt resume 跑完，不枯等
- [feedback_parallel_claude_sessions_git.md](feedback_parallel_claude_sessions_git.md) — 兩個 Claude session 平行動同一 repo git 會互踩；動前先 fetch 對齊、跨機用分支、別信同功能 auto-merge

## Project Plans
- [project_head_on_behavior.md](project_head_on_behavior.md) — 新增 head_on(直線迎面)障礙到 SA3-5,對症「晚反應撞動態」;現有 crossing/near_miss 都不正面來;激活瞬間瞄一次非 homing(防舞龍舞獅);已實作驗證待 SA3 runtime
- [project_sa5_v3f_react_design.md](project_sa5_v3f_react_design.md) — SA5 v3f-react 對症「動態障礙晚反應碰撞」：clearance-gated 減速 reward(penalty_speed_near_obs=0.8) + 壓低動態速度；7 處實作驗證完待啟動
- [project_sa4_v3f_completion_diagnosis.md](project_sa4_v3f_completion_diagnosis.md) — SA4_v3f 跑完待辦：先做碰撞 breakdown 診斷(SR ~82-83% 是 SA4 難度天花板 vs 可改善)再決定動不動 curriculum/reward；勿盲調。✅診斷已完成(det SR88.5%/CR11.5%,死因=動態晚反應)
- [sa1_sa7_baseline_results.md](sa1_sa7_baseline_results.md) — SA1~SA7 baseline 最終結果表（canonical run + WandB ID + SR/CR + checkpoint chain）；SA7 TC 89.4%/10.0% 已中止飽和
- [project_v2_roadmap.md](project_v2_roadmap.md) — **v2 方向：SA1~SA7 為 baseline，從 SA1_v2 重訓整合 Asymmetric Critic + VLP16 TLNI + Sim-to-Real DR + RNN64 + Vel Aux**
- [project_v3_launch.md](project_v3_launch.md) — **v3 從 SA1_v3 從頭訓：r_min 0.9→0.25 + penalty_smoothness=0.005 + 新 curriculum + per-env p95 監控（2026-06-08）**
- [project_v3d_launch.md](project_v3d_launch.md) — **v3d 抗 sin 波三件套：act_hist action_error 編碼 + dropout 0.2 + curriculum floor（ent_ang 0.02 / smoothness 0.015），ego 已含 odom 故只缺 action_error（2026-06-12）**
- [project_lidar_rmin_change.md](project_lidar_rmin_change.md) — 用戶實測 r_min=0.2m（非 0.9/0.5），連帶 normalize/shield/reward gate 同步改，留到 v3
- [project_goal_curriculum_plan.md](project_goal_curriculum_plan.md) — SA2+ 計劃：減少 goal 數、增加最遠距離、goal 隨機移動，解決回合太短問題
- [pigdreamer_borrow_analysis.md](pigdreamer_borrow_analysis.md) — PIGDreamer 借用分析：privileged critic 已完成，剩 cost critic + representation alignment；SA1 須 λ_cost=0
- [project_sim2real_ablation_experiment.md](project_sim2real_ablation_experiment.md) — **VLP-16 雜訊 sim→real 對照實驗(A no-noise/B spec-Gauss/C TLNI)+T2 固定路徑行人；SOP 在車端 ~/rover_rl/docs/；用 sim→real gap 非 SR；⚠️ wd_sa5_v2 r_min=0.9 不能當 treatment 臂**

## Warp Drive Project (new_warp_drive/)
- [Warp Drive Overview](warp_drive_overview.md) — 北科大論文 Spot 行為規劃 RL，WarpDrive 框架
- [Warp Drive Train Car](warp_drive_train_car.md) — train_rnn_car.py 入口/流程/car_mode/雙 loss
- [Warp Drive Train RNN](warp_drive_train_rnn.md) — 模組化 RNN 架構，嵌入 policy 的 auxiliary 訓練
- [Warp Drive → Isaac Lab](warp_drive_to_isaaclab_migration.md) — 遷移差異與三套融合方案
- [Warp Drive Open Q](warp_drive_open_questions.md) — 待驗證問題清單

## Deployment (rover_rl 車端 repo)
- [deploy_v3c_rover_rl.md](deploy_v3c_rover_rl.md) — rover_rl v3c 對接：83D obs + action stacking + α slew + r_min 0.25；三個 ω 別混；啟動用 *_v3c.yaml
- [deploy_v3e_rover_rl.md](deploy_v3e_rover_rl.md) — rover_rl v3e 部署 (sa1/sa2_v3e，同 v3c 架構)：新增 *_v3e.yaml (ω_max=1.2 修正) + deploy_rl_shell variant 選單；act_hist 語義 (action_error vs 原始) 待上路驗
- [deploy_v3f_rover_rl.md](deploy_v3f_rover_rl.md) — rover_rl v3f 部署 (sa4_v3f)：79D obs (act_hist 整段移除)；79D 車端自動關 act_stack；新增 deploy_rl_shell v3f 分支 + *_v3f.yaml (校準同 v3e)
- [finding_v3c_omega_max_decode_bug.md](finding_v3c_omega_max_decode_bug.md) — v3c 角速度 decode 不一致：訓練 1.2 但車端 policy_params_v3c.yaml:55 + charge_skrl_adapter.py:72 誤設 0.785(0.25π)，意圖縮成 65%；改回 1.2

## MARL Migration
- [MARL Migration 4 Phases](marl_migration_4phases.md) — DirectMARLEnv 遷移：N 台車 + RNN + parameter sharing + curriculum
- [deploy_charge_rviz_62.md](deploy_charge_rviz_62.md) — 62 用 demo.rviz 顯示 charge 車體：charge_description package(package://) + charge_ 前綴 overlay + static TF 橋接，不碰現役 campusrover /robot_description；真機 charge≠campusrover 是同車兩套模型

## Pedestrian Simulation (Isaac Sim 5.1)
- [pedestrian_ira_setup.md](pedestrian_ira_setup.md) — 3F 走廊 IRA 行人設定流程、已知問題、完整腳本路徑

## Obstacle Agent
- [obs_agent_behavior_config_v2.md](obs_agent_behavior_config_v2.md) — 8 種 rule-based behavior + BehaviorScheduler + 6-stage curriculum + RNN aux 關聯

## Project Findings
- [v21_vo_shield_dead_end.md](v21_vo_shield_dead_end.md) — v21 VO Shield 實驗失敗紀錄，simplified VO heuristic 在 dense scene 的 ping-pong 問題
- [v22_physx_oom_crash.md](v22_physx_oom_crash.md) — v22 在 6144 envs + 1h10min 觸發 PhysX CUDA 717 OOM；charge_skrl 安全 num_envs 上限為 4096
- [research_findings_dense_dynamic.md](research_findings_dense_dynamic.md) — Dense dynamic 避障 SOTA 文獻 + NavRL ORCA 真相 + Top-5 改善方向 (HEIGHT/Bounded Rationality/Trajectory Prediction)
- [v24_height_attention_dead_end.md](v24_height_attention_dead_end.md) — v24 只 attention 動態，LiDAR 仍 MaxPool(1)。碰撞 61% 來自靜態 → SR 卡 62.5%。
- [v25_heterogeneous_attention.md](v25_heterogeneous_attention.md) — v25 完整 HEIGHT：LiDAR 18 tokens + dynamic + type/angular emb 統一 cross-attention。
- [v27_dense_results.md](v27_dense_results.md) — v27 dense 實驗：HEIGHT TopK=10 在 22 total obs (12s+10d) 的 capacity wall，SR 61.6%
- [finding_policy_oscillation_stuck.md](finding_policy_oscillation_stuck.md) — SA4 ckpt 390k 在 obs~1.1m + 目標在後方時 v_body ±0.078 震盪卡住
- [finding_cmd_delay_limit_cycle.md](finding_cmd_delay_limit_cycle.md) — 實車舞龍舞獅 = 致動延遲(~200ms)極限環(delay/dt≈1.0)；推論端補償會惡化；須訓練端建模延遲+抗抽動（論文 §34）⚠️ How-to-apply 把藥方誤植成 obs_delay 見 [[finding_obsdelay_misimplemented_as_motordelay]]
- [finding_obsdelay_misimplemented_as_motordelay.md](finding_obsdelay_misimplemented_as_motordelay.md) — v3 把筆記要的『動作/馬達延遲』(actuator_dr,實車極限環藥方,保留)誤植成『觀測延遲』(obs_delay_steps,sim 害 policy 滿舵,移除)；兩者不同；正解=noobsdelay(馬達延遲ON+觀測延遲OFF)
- [finding_action_error_selfref_oscillation.md](finding_action_error_selfref_oscillation.md) — v3d action_error obs 是自我參照延遲殘差=內建振盪器，長 goal sin 波根因；p95_flip/smoothness 測不到低頻 weave；v2 是乾淨對照組
- [feedback_regression_over_symptom.md](feedback_regression_over_symptom.md) — 回歸性 bug 先 bisect「改了什麼」修根因，別用加 reward 懲罰壓症狀
- [finding_bangbang_degenerate_penalty0.md](finding_bangbang_degenerate_penalty0.md) — stage3 deterministic 滿舵塌縮(舞龍舞獅)兇手**定案=obs_delay_steps[0,1]觀測延遲**(感測端)；單變因隔離排除 actuator延遲/penalty/ent/特權資訊(noobsdelay 空曠飽和 0% vs nodelay/v3f 100%)；lidar regime(空曠也飽和=病態)判別法 + 部署用 deterministic
- [finding_shared_gpu_contention_fake_stall.md](finding_shared_gpu_contention_fake_stall.md) — 單卡共享 GPU，別人 job（laksh DINO）競爭會偽裝成 stall/慢段；診斷前先查 nvidia-smi 有沒有別人的 process，別誤 kill 自己的
- [finding_predict_head_dead_relu.md](finding_predict_head_dead_relu.md) — ⭐RNN velocity-aux 的 predict_head 末端 ReLU→dead-ReLU→輸出恆0→RNN 從未學會預測障礙運動;「晚反應撞動態」真根因(非reward);修=去ReLU重訓aux ★SA3 解碼裁決:結構修好但 velocity rel err 仍 100%(容量不足分支),見 [[finding_velocity_aux_bottleneck]]
- [finding_velocity_aux_bottleneck.md](finding_velocity_aux_bottleneck.md) — ⭐⭐整條調查總結(走到牆):probe 證障礙位置在 obs(oracle 75%)、過 extractor 仍在(96D→74%),但**進 RNN hidden 被洗成常數(64D→~100%)**;斷崖在 RNN。試遍 posonly/Huber/predict_head低lr/skip/GRU/WD-faithful凍結都沒讓 RNN 編碼位置。WD 配方=凍結 readout 只訓 RNN cell(預移植版讓 predict_head 可訓→擬常數餓死 RNN),但續訓100iter仍沒救→velocity-aux 此路在當前架構未證可行;三條路 A顯式障礙channel(推薦)/B接受反應式/C續摳WD差異
- [feedback_detailed_report_each_cycle.md](feedback_detailed_report_each_cycle.md) — 用戶要求每監控 cycle 詳細回報(完整指標+aux 狀態+解讀)並寫進度記錄,非一行簡短
- [project_reluFix_retrain_log.md](project_reluFix_retrain_log.md) — reluFix lineage(SA1→SA5)重訓時間序進度記錄,每 cycle append 詳細快照
