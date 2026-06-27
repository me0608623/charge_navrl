---
name: project_sa5_v3f_react_design
description: SA5 v3f-react 改善「動態障礙晚反應碰撞」的 reward+場景設計與實作（clearance-gated 減速懲罰 + 壓低動態速度）；2026-06-22 已實作驗證待啟動
metadata:
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

**背景（2026-06-22，用戶指示「以獎勵設計與訓練場景參數設計在 SA5 改善此問題」）**：對症 SA4_v3f 診斷的死因（見 [[project_sa4_v3f_completion_diagnosis]]）= 動態障礙晚反應碰撞（89% 全速撞、97% ≤0.5m 才偵測、0% stuck/追goal）。根因：sparse reward 對障礙的唯一訊號是接觸瞬間 penalty_hit，**接觸前無梯度**叫 policy 放慢。

**兩軸設計（用戶選推薦案：單項 reward + 壓低動態速度；皆只作用 SA5，單變因）**：

**軸一 Reward — clearance-gated 減速懲罰 `penalty_speed_near_obs`（罰「快速」非罰「接近」）**：
```
d_react=1.2m, d_stop=0.45m
p = clip((d_react - d)/(d_react - d_stop), 0, 1)   # 遠=0, 碰撞邊界=1
逐步懲罰 = -(w/fps) * p^2 * max(v_forward, 0)        # SA5 w=0.8
```
- p² 聚焦內側危險區（區邊緣寬鬆，不在空曠過度煞車）；慢下來(v→0)即免罰（鼓勵減速通過非凍結）；dense 每步在區內都給→塑造中段全速行為。
- **不盲加 penalty_hit**（對齊 [[feedback_regression_over_symptom]]）：加重懲罰碰不到「接觸前無梯度」根因。
- 單元測試驗過：d≥1.2→0、d=0.8快→-0.041、d=0.45快→-0.144(最負)、d=0.8但v=0→0。✅

**軸二 場景 — SA5 動態速度壓回「可反應」範圍（反應訓練哲學）**：現況 SA5 動態比 SA4 更快會 confound、v_rel 超轉向能力則物理避不掉。改：patrol 0.30-0.65→0.25-0.50、crossing 同步降、corridor_crossing mix 0.25→0.15（先練開放避障，難度留 SA6）、obs_near_goal_radius 2.0→2.5（障礙別生反應區內，緩解 7% spawn-trapped）。

**實作觸點（7 處，全 py_compile + 邏輯驗過）**：
1. `rnn_car_wdclean/rewards.py` `compute_wd_charge_reward`：加 `penalty_speed_near_obs/near_obs_dist_m/v_forward_m/near_obs_d_react/d_stop` 參數 + 新 breakdown `speed_near_obs_reward`。（注意：train_rnn_car_wdclip.py:1206 那份是 dead copy，實際用 wdclean 這份）
2. `rnn_car_modular/rewards/wd_sparse.py` facade：加屬性 + sync `spot_penalty_speed_near_obs` + 從 context 取 d/v 傳入。
3. `train/train_rnn_car_wdclip.py`：compute() 呼叫點建 context（`obs[:,6:78]×8.0`=d公尺 hole-mask<0.02、`obs[:,1]`=v_forward m/s）；常數 `_LIDAR_MAX_DISTANCE_M=8.0`；sync `_spot_penalty_speed_near_obs`；WandB `phase_parameter/penalty_speed_near_obs`。
4. `curriculum/phases/wd_single_agent_v3.py` `_flatten_phase`：export `spot_penalty_speed_near_obs`。
5. 新 `curriculum/phases/wd_single_agent_v3f_react.py`：v3e patch 全 stage（smoothness=0+ent floor 0.02）+ SA5_endurance 疊 react patch。驗過只 SA5 非零。
6. `phases/__init__.py`：註冊 `warp_drive_single_agent_v3f_react`。
7. 新 config `configs/wd_sa5_v3f.yaml`：接 `sa4_v3f_ne1024_s42/checkpoint_420000.pt`、initial_stage 5、gamma 0.995、obs 79D 可 resume、CHARGE_USE_ACT_HIST=0。

**啟動指令（待用戶確認後跑）**：
```bash
PYTHONUNBUFFERED=1 CHARGE_USE_ACT_HIST=0 \
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
  --experiment_config wd_sa5_v3f --headless \
  --checkpoint logs/rnn_car/sa4_v3f_ne1024_s42/checkpoint_420000.pt \
  --run_name sa5_v3f_react_ne1024_s42
```

**驗收 KPI**：`charge/speed_x_near_obs_mean` 應從 ~0.67 下降（學會近障礙減速）；撞動態 CR↓ 但 **TO 須維持低**（過度保守反指標）；SR 站穩/超 88.5%(det)。**主調參旋鈕 = `SA5_PENALTY_SPEED_NEAR_OBS`（v3f_react.py，現 0.8）**；若 TO↑/過度煞車則降、若不夠減速則升。

**⛔ 兩個踩雷更正（2026-06-23，用戶抓出）：**

**(1) reward 從未生效的 bug（curriculum→info key 漏複製）**：新 reward key 在 `_flatten_phase` 正確，但 `goal_obstacle_curriculum.py` 的 `_curriculum_info` 是**顯式逐 key 複製 dict**（~L1741），只列舊 key → 新 `spot_penalty_speed_near_obs` 被丟掉 → trainer 讀 0 → reward 全程 OFF。**症狀=SR/CR 紋風不動 + WandB phase_parameter/penalty_speed_near_obs=0**。修法：在該 dict 補 `"spot_penalty_speed_near_obs": float(stage_cfg.get(...,0.0))`。**教訓：加任何新 curriculum-synced reward key，必須同步改 4 處**（_flatten_phase + goal_obstacle_curriculum info dict + trainer sync/log + reward facade），漏一處就靜默失效。驗收必須拉 WandB 確認 phase_parameter 非 0，別只看 console。

**(2) 場景設計 v1 自相矛盾（用戶洞察）**：要訓「近障礙減速」卻把場景**調簡單**（速度↓+corridor↓+障礙離 goal 遠）→ 光轉向就能避→不需減速→reward 沒東西可咬（speed_x_near_obs 不降證實）。**原則：減速行為只有在「光轉向避不掉」時才被學到** → 場景要用**密度+距離製造「非減速不可」的遭遇**，速度維持「需減速但減速有用」中段（太快=物理避不掉 reward 也救不了；太慢=轉向就夠不用減速）。**v2 遭遇密集版（已重訓）**：goals 4→2、goal_distance 下限 2→5、dynamic 5→7/min 3→5、obs_near_goal radius 2.5→2.0/count 1→2、速度回中段(patrol 0.30-0.60)、擋路行為加重(path_crossing 0.25/horizontal 0.20/corridor 0.20)。iter1 即 SR 90%→75%/CR 5.7%→17.5% 證實場景真難了。run=sa5_v3f_react_ne1024_s42(v2)，舊兩版歸檔 _NOOP_rewardOFF / _v1_easyScene。

相關：[[project_sa4_v3f_completion_diagnosis]] [[finding_obsdelay_misimplemented_as_motordelay]] [[feedback_shield_vs_policy]] [[feedback_regression_over_symptom]]
