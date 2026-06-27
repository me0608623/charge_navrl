---
name: rnn-reset-on-done-verified
description: 已驗證 RNN hidden state 在 episode reset 時正確歸零，不會把上一場景記憶帶入下一場景（train_rnn_car_wdclip pipeline）
metadata: 
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

已驗證（2026-05-27）：`train_rnn_car_wdclip.py` 的 RNN reset-on-done 機制**正確**，episode reset 不會跨場景污染記憶。經典 RNN-RL bug（hidden state 帶入下一 episode）在此 code **沒有發生**。

機制（檔案/行號為當時，需重查確認）：
1. **Hidden 歸零**：`models/modular_rnn_models.py` `RNNStateManager.reset(env_ids)` → `self.hidden[:, env_ids, :] = 0.0`
2. **Rollout 時序**（`train_rnn_car_wdclip.py` ~line 3110-3114）：`rnn_state.update(new_hidden)` → `done = terminated | truncated` → `rnn_state.reset(done_ids)`。第 t 步 done 後，第 t+1 步（新場景 obs）配零 hidden。terminated（碰撞）+ truncated（超時）都觸發 reset。
3. **連帶 cache 也清**（同段）：`_obstacle_velocities[done_ids]=0`、`_wd_aux_obs_pos_t1[done_ids]=0`（避免障礙速度 / WD aux 歷史 position 跨 episode 污染）。
4. **TBPTT 不跨邊界**：`sample_sequences` 的 `valid_mask`（`dones_in_window < 0.5`）排除任何中途有 done 的序列，RNN BPTT 不跨 episode。

**Why:** 用戶擔心場景 reset 時 RNN 記住上一場景帶入下一場景。查證後確認已正確處理 — 這也是 SA1_v2 能從 scratch 正常 bootstrap 的前提（若有此 bug 會學不起來或極不穩）。

**How to apply:** 不需再為「RNN 跨 episode 污染」重查或除錯。若未來改 rollout loop / RNNStateManager / TBPTT 取樣邏輯，需重新確認此三處仍維持 reset-on-done。相關：[[design_obstacles_60d_unnecessary]]（obs 79D）、[[project-v2-roadmap]]。
