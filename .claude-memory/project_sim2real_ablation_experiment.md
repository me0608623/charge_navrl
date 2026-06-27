---
name: project_sim2real_ablation_experiment
description: VLP-16 雜訊模型 sim→real 對照實驗設計（A/B/C 三臂 + T2 固定路徑行人）；SOP 在車端
metadata: 
  node_type: memory
  type: project
  originSessionId: 0d22bb8d-d51c-492a-b396-d5641476bd15
---

**狀態（2026-06-23）**：用戶問「把實測 VLP-16 雜訊模型 + 隨機化帶進訓練，真的對 sim→real 有用嗎？」→ 設計了真車對照實驗。可執行 SOP 已寫好並 scp 到車端：`~/rover_rl/docs/2026-06-23_SIM2REAL_雜訊對照實驗_SOP_T2.md`。

**核心 framing（最重要）**：結論用 **sim→real gap = SR_sim − SR_real**，不是「SR 越高越好」。因 TLNI 在 sim 內 SR 通常低於 no-noise（它在更難雜訊環境訓練）。主張 =「TLNI 最小化 gap」。

**三對照臂（固定 curriculum/scene/seed/timesteps，只動 LiDAR 區塊）**：
- A. no-noise（`lidar_no_noise: true`，全關）— sim-only strawman，最大 gap
- B. spec-Gaussian（`lidar_displacement_std: 0.03` per-ray ±3cm，其餘關）— datasheet 傳統做法，定理1 會被 min-pool 吞
- C. TLNI（用戶實際部署的 full data-driven config）— treatment，最小 gap
- 真車 MVP：A vs C × {對向/橫穿/同向} × 10 trial = 60 trial；B 在 sim 證即可

**⚠️ 兩個已驗證的關鍵事實**：
1. **`wd_sa5_v2.yaml` 不能當 treatment 臂**——`curriculum_version=v1` → r_min=0.9（舊值），與車端 0.25 不符。必須用 **v3 家族 config**（r_min=0.25）當 base。Arm C base = 用戶實際部署的 config（待確認是 wd_sa4_v3f 或其他）。
2. **車端 r_min=0.25 已正確**：`lidar_preprocess.py` default 是 0.9，但 v3c/v3e/v3f 的 `lidar_preprocessor_params_*.yaml` 覆寫成 0.25，與訓練一致。命脈沒斷。

**課程陷阱**：noise 原本隨 SA1→SA7 漸進開，所以 A/B/C 不能直接拿不同 SA 比（那比的是難度）。三臂必須在同一固定難度 stage 從頭訓。

**待用戶確認**：(1) Arm C base config 到底是哪個（需含 full TLNI：per_ring+dist_bias+distractor+block_dropout+L3 DR）；(2) A/B fresh run 的 timesteps（須 = C）；(3) 真車要不要也跑 B 臂（預算）。

**對應貢獻**：C1（資料驅動雜訊，MF 67×→1.5×，§5.5 已證）、C2（min-pool-aware TLNI，三定理，§3.6.3 已證）、C3（policy 層面實證）= 本實驗要補的缺口。相關：[[finding_v3c_omega_max_decode_bug]]（ω_max 須 1.2）、[[project_lidar_rmin_change]]、[[finding_predict_head_dead_relu]]。
