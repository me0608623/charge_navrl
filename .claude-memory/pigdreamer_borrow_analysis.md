---
name: pigdreamer_borrow_analysis
description: PIGDreamer 可借用設計分析結論 — privileged critic 已完成，剩 cost critic + representation alignment
metadata: 
  node_type: memory
  type: project
  originSessionId: fa60bc3c-f536-48cb-b593-11b061b7fa2b
---

PIGDreamer（ICML 2025, arXiv:2508.02159）對現有 WD-style 導航訓練的可借用分析（2026-05-27，只分析未改碼）。筆記：`/home/aa/Documents/Obsidian Vault/isaaclab_v2/架構與演算法研究/PIGDreamer_對現有導航訓練的可借用設計分析.md`。

關鍵非顯然結論：
1. **Privileged critic 已實作完成** — `ValueHead(privileged_dim=50)` asymmetric，priv_proj zeros-init，rollout/update/bootstrap 都帶 50D 障礙物 privileged。不需重做。
2. **Representation alignment 已部分存在** — `PreprocessRNN.predict_head` + `compute_wd_module_loss` 已逼 RNN latent 回歸 privileged geometry（7D/13D），且 detach 後才給 RL heads（部署只用 actor obs）。
3. **真正要新增的只有 cost critic** — reward breakdown（`compute_wd_charge_reward`，`train_rnn_car_wdclip.py:1206`）已含 goal_reached/wall/obs/static/dynamic/other_death mask，是 cost_signal 現成來源。
4. SA1 必須維持：`λ_cost=0` 且 `align_weight=0` → `A_actor = A_reward`（不改 actor gradient）。WD reward 完全繞過 IsaacLab reward manager。
5. 最大實作風險：two-optimizer detach 邊界（V_cost loss 不可反傳進 extractor/RNN）；cost 是 positive sparse，normalize/clamp 必須與 reward 分離。

用戶已確認決策（2026-05-27）：
- cost_signal 只算「真正避障碰撞」`(wall|obs).float()` clamp≤1，**不含 other_death**。
- **不加任何 dense cost**（near-obstacle/near-wall/TTC dense）→ cost 全程 sparse，staged 只調 λ_cost。
- **兩個 critic（reward + cost）都吃 privileged**（asymmetric）。
- reward 權重本來就隨 SA 階段變（用戶課程設計）→ cost 是獨立額外通道，不取代/不為 cost 改 reward。
- alignment：**不動現有 LidarStateExtractor**，只外掛 priv_encoder + projector，部署時拔掉。
- TTC / min-dist / wall-clearance 當 critic 特徵：min-dist/TTC 文獻支持較強但與現有 50D(含 v) 冗餘、wall-clearance 文獻較弱 → 期待增益小且不確定，**不優先**；要驗證用 ablation（Run C vs D）一次定維度。先看 50D asymmetric(SA1_v2) 本身效果。

定位：PIGDreamer-inspired model-free asymmetric actor-critic，非 model-based reimplementation。相關 [[project_v2_roadmap]]、[[research_findings_dense_dynamic]]。
