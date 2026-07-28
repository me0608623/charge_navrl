# SA7.1 成對對照 source freeze
凍結時間: 2026-07-28 09:30:38 CST（初版）
**補正時間: 2026-07-28 09:37:09 CST** — 初版凍在 0S0D 帳本 bug 修復之前，
Control 臂啟動於 09:35:57，用的是**補正後**的 source。以下為權威 hash。

git HEAD: 5a7acb2e600c4b8c70c37d6eb78f2b5327cb2790
git 未提交: 15 個檔案

## 行為相關 source 檔 SHA256（前 16 碼）
84447669f926c34c  corridor_density.py
4a6f99297fcf4552  long_corridor_replay.py
887ee815dc7cf187  long_corridor_replay_geometry.py
30f20e75731180ea  rule_behaviors.py
1911882af1b9bebd  behavior_scheduler.py

初版與補正的唯一差異: long_corridor_replay.py 6ab63d4a23b1ed8b -> 見上
差異內容: Gate-aligned env 寫入真實 (4,2) 密度、count-mix 審計母體改 mix_selected、
          審計行加印 gate_aligned 逐模式計數。**不影響 Control 臂**（share=0.0 時
          該分支整段不執行），故兩臂仍在同一套幾何與行為下。

## 對照設計
Control  : e2e_sa7_wander_from_w1c10        走廊 10% 真實混合
Treatment: e2e_sa71_gate_aligned_from_w1c10 走廊 10% 混合 + 10% Gate 題型
共用: W1-c10 暖啟動 / seed 42 / 30 iterations / 1024 env / 每 5 iter 存檔
crossing 幾何: **新版每回合隨機交點**（兩臂相同）
