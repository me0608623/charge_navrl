# IsaacLab Charge Navigation Project Memory

## Active / Open (2026-08-24) — SA5-v3 c50 matched motion curriculum
- [project_sa5_v3_c50_matched_control_result_20260824.md](project_sa5_v3_c50_matched_control_result_20260824.md) — **目前權威三臂裁決**：等權 it25 相對 c50 只讓 longitudinal CR `81.48%→47.50%`；70% random-2D 加權相對等權 control 雖使 random-2D CR `-5.65pp`，卻使 lateral/longitudinal/mixed CR `+5.88/+21.79/+5.01pp`，三項 retention 全 FAIL。拒絕 `(0.15,0.15,0.70)`、無 parent、SA6 HOLD；allocator 校準候選為 `(0.30,0.30,0.40)`，待 paired A2/B2 驗證。

## Active / Open (2026-08-20) — SA5-R2 c250 speed-rate learnability
- [project_sa5_pedestrian_speed_screen_20260821.md](project_sa5_pedestrian_speed_screen_20260821.md) — **行人速度 Phase 1 COMPLETE**：固定 c250 / speed_rate=1.0 / sealed 4S2D lateral，行人 P035→P100 使 CR `38.82%→98.47%`，impact radial p90 `0.376→1.057 m/s`；增量幾乎全為行人碰撞。最後 1s 仍有 D5 dynamic-feasible action 只 `22.56–28.65%`，標籤限定為 `D5_MODEL_FEASIBILITY_LIMITED...`，不等於物理必撞。新 SA3→SA4 應用速度×密度 curriculum；凍結 config 前先在 0S1D/1S1D 做 speed_rate 1.0 vs 0.7 交叉 screen。未訓練、未啟動 SA6。
- [project_sa5_r2_speed0p7_checkpoint_screen_20260820.md](project_sa5_r2_speed0p7_checkpoint_screen_20260820.md) — **24-cell screen COMPLETE，證據標籤已更正**：`ADAPTATION_DEGRADED_NO_IMPROVEMENT_AT_ANY_CHECKPOINT`。it25 vs it50 只差 1.65 SE，不足以稱 late forgetting；停止來自 it50 顯著劣於 baseline（3.10 SE），且 it25 未曾顯示改善。c250 vs natural c300 control 只差 0.04 SE，支持將 it50 退化歸於 speed-contract 改變，不是單純多訓 50 輪。不續訓、不接受 parent、不啟動 SA6；重建 SA3→SA4 前先跑無訓練的行人速度 sweep，一次定義速度比契約。
- [project_sa5_r2_c250_speed_rate_screen_20260820.md](project_sa5_r2_c250_speed_rate_screen_20260820.md) — **速度裁決來源**：c250 固定 lateral screen 中，車端 `speed_rate=0.7` 相對 1.0 使實際車速 `-19.2%`，但 SR `61.18→41.23%`、CR `38.82→58.77%`、反應 delay `0.6→0.8s`。screen 無 gradient update，因此只支持下一步做有界可學性實驗。
- [project_sa5_r2_c250_speed0p7_adapt_launch_20260820.md](project_sa5_r2_c250_speed0p7_adapt_launch_20260820.md) — **已完訓、待 fixed screen**：exact SA5-R2 c250 + optimizer 固定 `speed_rate=0.7` 完成 50/50。前/末 10 輪 lateral CR `34.20→30.53%`，但 longitudinal `26.48→28.57%`、native `8.66→10.44%`，corridor overall `37.23→37.27%` 持平。只能判定 training-health PASS，必須再比 c250 baseline、natural c300、adapt-it25/it50 的同 protocol fixed Gate；SA6 未授權。

## 🌐 Global Integration
- [CLAUDE_MEM_INTEGRATION.md](CLAUDE_MEM_INTEGRATION.md) — Claude-Mem 全域記憶整合（自動捕捉工具調用、語義搜索）
- Global DB: `~/.claude-mem/claude-mem.db` (SQLite)
- Vector Store: `~/.claude-mem/chroma/` (Semantic search)
- Worker: `localhost:37777` (auto-launched)

## Active / Open (2026-08-19) — SA5 sealed screen + corrective lineage
- [project_sa5_checkpoint_screen_20260819.md](project_sa5_checkpoint_screen_20260819.md) — **目前執行點**：r2 被 GUI 發現長走廊側牆每端與外牆留有 2.0m 可通行缺口，已全批標記 `INVALID_GEOMETRY_BYPASS`。v2 實體側牆改為 15m，與南北外牆各重疊 0.5m；sealed r3 仍在 fixed screen。舊 policy 在 GUI 會朝原缺口走並撞新牆角，c150/lateral wall CR `0.84% → 24.55%`，支持 legacy bypass 已被學到。SA5-R2 矯正血緣已建立並通過 64-env GPU smoke；從相同 SA4-R3 it125 parent 重跑、不載入舊 SA5 optimizer/checkpoint。正式訓練前先做 exact parent 的密封四-family baseline；目前不訓練、不啟動 SA6。
- [project_sa5_from_sa4r3_it125_actdelay12_launch_20260818.md](project_sa5_from_sa4r3_it125_actdelay12_launch_20260818.md) — delay-only 修正版 SA5 已正常完成 `300/300`（241.3min），六顆 checkpoint 齊全；末 50 episode-weighted corridor CR `4.95%`，lateral/longitudinal/random2d/mixed CR 分別 `5.01/3.98/3.94/5.99%`，但仍屬 training-distribution evidence。supervisor 已 `COMPLETE_TO_IDLE`、expected_run 清空，auto-advance `HALTED_ALERT`。
- [finding_real_robot_angular_gain_aggregate_contamination_20260818.md](finding_real_robot_angular_gain_aggregate_contamination_20260818.md) — 原先 28%～44% 是含未起步／疑似 mux 接管且未對時的 session aggregate，不能當轉向穩態增益；五個乾淨窗口 gain `0.983～1.046`、lag `350～400ms`。目前只修延遲中心，不可把 scale 改成 0.28～0.44。
- [project_sa4_r3_it125_checkpoint_screen_20260818.md](project_sa4_r3_it125_checkpoint_screen_20260818.md) — SA4-R3 conceptual it125 三場景 screen 已完成：longitudinal/native PASS、lateral 差 `1.0465pp`，frozen machine verdict 保持 FAIL。使用者另給 human waiver 並明確授權 SA5；不得在論文改寫為 SA4 Gate PASS。
- [finding_corridor_static_geometry_two_layouts_20260817.md](finding_corridor_static_geometry_two_layouts_20260817.md) — ★★★ **使用者 GUI 目視觸發的實測**：走廊靜態骨架**只有 2 種**（固定格位+±0.10m抖動+左右鏡像），靜態中心可達面積僅 `0.55%`，**可通行性翻轉 0 次**＝每回合同一題。閉環分組顯示 policy 對鏡像顯著敏感（lateral 牆 CR `0.09% vs 3.68%`, z≈6.2），**但撞牆的鏡像在 longitudinal 反轉**且障礙 CR 也差 `+4.59pp`。→ 「曝光不足」不是量而是**題目只有一種拓樸**；堆輪次報酬低。單 seed。
- [finding_action_grid_degeneracy_20260817.md](finding_action_grid_degeneracy_20260817.md) — ★★★ 19×19=361 動作格在巡航時只有 **110 個不同 (v,ω)**（重複 69.5%，其餘速度 42.1%）；角速度軸因 `slew 0.6 < ω_max 1.2` **永久浪費 8 格**。**同一解碼器也用在 policy 輸出頭** → 索引 entropy/KL 高估行為多樣性。機制已驗證、後果未量化。⚠️ 不是 teacher 53.7% 天花板成因（僅 0.31% 標籤落在崩塌格）。
- [finding_teacher_goal_denominator_guard_20260817.md](finding_teacher_goal_denominator_guard_20260817.md) — ★★ `C_goal` 的 `max(D,1)` 原意只是**除零防護**，卻成為最常經過區間（1.5–3m）的主導行為參數。改固定 `R=10` 後危險帶 clearance `0.104→0.382m`、貼硬底線格子 `3/72→0/72`，且**速度不變**（同速繞更開，非變膽小）。R=2/R=4 無效。蒸餾前應先修 teacher。
- [project_sa4_d9_replication_r3_pilot_result_20260817.md](project_sa4_d9_replication_r3_pilot_result_20260817.md) — ★★★ **目前權威裁決**：D9 `valid_return_only` 在 evaluator seeds 515/616/818 的 total CR 均下降（`-0.98/-1.87/-1.13pp`），但後續 R3 50 輪固定 screen 的 it50 lateral/longitudinal CR 為 `18.87%/21.73%`，皆 FAIL；native PASS。`accepted_parent=null`，不延長、不啟動 SA5。
- [finding_sa4_d4r2_argmin_result_20260817.md](finding_sa4_d4r2_argmin_result_20260817.md) — ★★★ **目前權威 D4-r2 裁決**：actual-angle feasible `17.49%` vs center `17.89%`，paired net `-0.40pp/-98 frames`，未恢復候選；obstacle CR `16.45%→16.44%` 不變。排除 5°中心為大量 no-feasible 主因；c6400 拒絕、SA4 HOLD、SA5 未啟動。`0.002515` noise 語意另案處理。
- [finding_sa4_d7_residual_lidar_origin_20260802.md](finding_sa4_d7_residual_lidar_origin_20260802.md) — D6 residual points 的 `59.95%` 是 distractor winner，`37.63%` 是 1° raw ray 壓成 5°中心角後的 geometry/dynamic-attribution artifact，只有 `1.76%` 是 raw geometry unmatched。point share 不是碰撞因果；argmin azimuth 後續已由 D4-r2 排除為大量 no-feasible 主因，剩餘待辦是獨立 noise contract。
- [finding_sa4_d6_static_feasibility_20260801.md](finding_sa4_d6_static_feasibility_20260801.md) — D6 顯示已知牆面移除只救回 `0.96%` blocked frames，residual partition 移除救回 `97.65%`；其生成來源已由 D7 分解，不再標為未知。
- [finding_sa4_d5_shadow_frontier_20260801.md](finding_sa4_d5_shadow_frontier_20260801.md) — D5 baseline-only shadow：碰撞最後有解 p50 `2.4 s`、持續無解 onset p50 `2.2 s`，linear intersection 比 radial TTC 中位早 `1.2 s`；其 static bottleneck 後續已由 D6 收窄。
- [finding_sa4_d3_lateral_closed_loop_20260801.md](finding_sa4_d3_lateral_closed_loop_20260801.md) — ★★★ **目前權威裁決**：SA4-R2 c50 lateral CR `16.95%` 仍 FAIL；D3 sustained-brake / best-turn / combined 全部未通過總體 Gate。best-turn 將 dynamic CR `15.79% → 2.28%`，卻把 static/wall CR 推至 `57.53% / 25.39%`，證明 lateral 動態碰撞可轉向但必須加入 wall/static 幾何可行性。`accepted_parent=null`，禁止啟動 SA5。
- [finding_sa4_d4_geometry_selector_result_20260801.md](finding_sa4_d4_geometry_selector_result_20260801.md) — D4 固定比較未觀察到改善；81.9% 分母是 active frames，後續成因已由 D5/D6 收窄但 residual 生成原因未定案。
- [project_sa4_sim2real_v2_pilot_hold_20260731.md](project_sa4_sim2real_v2_pilot_hold_20260731.md) — SA4 pilot、D1-D7/D4-r2 累積裁決；argmin angle 已結案，下一步只限獨立 LiDAR distractor/mixed-pixel contract。

## Active / Open (2026-07-28) — Fresh SA1 sim-to-real v1
- [project_sa1_sim2real_v1_20260728.md](project_sa1_sim2real_v1_20260728.md) — **目前權威訓練主線**：W1-c10 實車舞龍舞獅後停止 warm-start，從 SA1 隨機初始化共同訓練 native 78% + 1.2–1.4m 窄縫 12% + 4×10m 走廊 10% + 實測 VLP16 full noise + actuator delay U{0,1,2}=0/200/400ms；83D/K8、RNN 不進 policy；R1 `sa1_sim2real_v1_ne1024_s42_r1` 正式長訓練中，W&B `gxtcgj3y`；`Linger=yes`。

## 🔬 走廊密度混合與行人互動 (07-27)
- [finding_sa7_corridor_ceiling_gate_results.md](finding_sa7_corridor_ceiling_gate_results.md) — ★★★SA7-r3四候選正式gate全FAIL(SA8未啟動);窄縫Gate5a 100%但走廊卡住;最弱是longitudinal非random_2d;訓練SR最低的c90走廊最強
- [project_sa7_corridor_density_mix.md](project_sa7_corridor_density_mix.md) — ★★SA7已啟動(1024env/s42):走廊改逐env密度混合(平均動態2.25,retention非壓測)+行人互動;commit 5a7acb2e600
- [finding_interaction_sampling_order.md](finding_interaction_sampling_order.md) — ⭐⭐互動先抽、family服從;random_2d不得配對(會被wander覆寫成假陽性);crossing量事件非相位;並排速差須實作為0
- [finding_ledger_desync_early_return.md](finding_ledger_desync_early_return.md) — ⭐⭐提早return跳過記帳→補償對著假帳本調節;診斷法=內部帳本與實際計數並列;測試範圍須涵蓋生產實際值
- [feedback_no_append_plus_replace_same_file.md](feedback_no_append_plus_replace_same_file.md) — ⭐同檔混用cat>>與replace造成9個重複定義,舊版靜默生效且測試全綠;replace須先assert;AST守門要含AnnAssign

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

## Active / Open (2026-07-26) — Corridor random_2d root-cause audit
- [finding_corridor_d1_random2d_pause_model_mismatch.md](finding_corridor_d1_random2d_pause_model_mismatch.md) — **目前權威裁決**：D1 `30/10/60` 永久封存，D0 `checkpoint_3840.pt` 保持主基準；c10 只作診斷。下一步先做 evaluator-only `pause=default vs zero` 三 seed A/B + motion-phase collision audit，未通過因果門檻前禁止改 reward、開 D2 或再掃 replay 權重。

## Superseded / Historical (2026-07-27) — N1 到部署的窄口路線
- [project_n1_nearfield_sidegap_actuator_roadmap.md](project_n1_nearfield_sidegap_actuator_roadmap.md) — 2026-07-27 的歷史順序，已於 2026-07-28 被 fresh SA1 sim-to-real v1 取代；N1/SA8/λ 維持 HOLD，不可再依「最後才加 actuator delay」啟動訓練。

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
7. **歷史舊假設：LiDAR min_range = 0.9m（已作廢）** — 2026-07-27 使用者確認 0.30m 仍有稀疏人體點雲；未來近場模型以 [project_n1_nearfield_sidegap_actuator_roadmap.md](project_n1_nearfield_sidegap_actuator_roadmap.md) 為準。
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
- [project_sa5_v3_b3_cont25_screen_20260824.md](project_sa5_v3_b3_cont25_screen_20260824.md) — B3 it25 精確續訓到 it50 健康完成；24 格 screen r1 於 1/24、r2 於 22/24 皆因來源漂移 fail-closed。r2 診斷上六顆 P060 1S1D 全 FAIL（CR 13.59–15.03%），停止堆 iterations；但無 formal 24-cell verdict/parent/Phase-B/SA6
- [finding_2s2d_launch_timing_teacher_gap_20260824.md](finding_2s2d_launch_timing_teacher_gap_20260824.md) — 固定 d1 2S2D teacher-only 已完成；WAIT→COMMIT_SIDE→PASS 的 FSM 承諾側與剛騰空側一致率 99.9%、U-turn/360° 為 0–0.19%，但 lateral/mixed CR 皆約 53%、SR 僅 18–22%、等待約 71–73% frames。教師 FAIL；不做 K8、蒸餾、SA6 或新訓練，先分析 FSM 狀態下的碰撞與多階段 passage planning
- [project_sa5_v3_c50_mild_p060_alignment_result_20260824.md](project_sa5_v3_c50_mild_p060_alignment_result_20260824.md) — 30/30/40 改善高密度但遺忘歷史 P060；matched P060 對齊使四格 CR 全降且 P080 無退化，但 P060 1S1D CR 仍 14.10%；左 layout 22.55% 且 scheduler 接觸事件 94.4% 為行人（非 episode 比例），Phase-A FAIL、無 parent/Phase-B/SA6
- [project_sa5_low_density_speed_ratio_screen_20260821.md](project_sa5_low_density_speed_ratio_screen_20260821.md) — P100 在 0S1D/1S1D 仍全 FAIL；frozen c250 由 rate 1.0 縮至 0.7 使 CR +35–36pp；單 checkpoint/seed 診斷，不證 P100 不可學
- [project_sim2real_speed_density_v3_20260821.md](project_sim2real_speed_density_v3_20260821.md) — **c300 已正式接受、SA4-v3 正式訓練中**：五重 SHA lock、36 tests 與 GPU smoke 通過；exact c300 parent 啟動 300 iter，首六輪 GREEN，SA5 auto-launch 關閉
- [project_sa5_r2_sealed_densitymix_launch_20260819.md](project_sa5_r2_sealed_densitymix_launch_20260819.md) — **歷史 SA5-R2 血緣**：從原 SA4-R3 it125、reset optimizer，聯合修正密封幾何與低密度走廊配方；其 c250 後續只作診斷，現已由 speed-density v3 重建決策取代
- [finding_sa4_d4r2_argmin_result_20260817.md](finding_sa4_d4r2_argmin_result_20260817.md) — actual-angle 未恢復 D4 候選（17.49% vs center 17.89%），obstacle CR 不變，SA4 HOLD；另保留 0.2515% mixed-pixel 語意缺口
- [finding_sa4_d8_noise_eligibility_result_20260817.md](finding_sa4_d8_noise_eligibility_result_20260817.md) — paired shadow 證實 no-return/dropped-ray 近距 ghost 是 D4 大量無解的重要來源：no-feasible 82.73%→57.78%，救回 30.33%；仍非完整 VLP-16 模型且未估 corrected-policy SR/CR
- [project_sa4_d9_valid_return_noise_closed_loop_ab_20260817.md](project_sa4_d9_valid_return_noise_closed_loop_ab_20260817.md) — closed-loop A/B 完成：valid-return-only 在固定 c6400 lateral/d1/s818 使 SR +1.13pp、CR -1.13pp（obstacle -1.39pp、wall +0.27pp）；是單 seed 點估計改善，仍未過 SA4 且不是 validated VLP-16 model
- [project_sa4_d9_replication_r3_pilot_result_20260817.md](project_sa4_d9_replication_r3_pilot_result_20260817.md) — D9 三 evaluator seeds 3/3 同方向後啟動 R3；R3 50/50 健康完成但 it25/it50 都沒有通過 lateral+longitudinal+native 聯合 Gate，維持 SA4 HOLD、SA5 未啟動
- [project_sa4_r3_cont50_result_20260817.md](project_sa4_r3_cont50_result_20260817.md) — 使用者明確授權從 R3 it50 載入 optimizer 精確續訓至 it100；lateral CR 18.87%→11.92%、longitudinal 21.73%→5.29%，但 lateral 仍 FAIL，無 accepted parent、SA5 未啟動
- [finding_corridor_d1_random2d_pause_model_mismatch.md](finding_corridor_d1_random2d_pause_model_mismatch.md) — D1 random_2d 加權曝光無效且傷害既有家族；mixed 非 IID 偏差不是主因；短軌跡、pause=0 velocity、future-occupancy move gate 與反向模型落差待有界 A/B 定因果
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
