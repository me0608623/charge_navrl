---
name: finding_shared_gpu_contention_fake_stall
description: 共享 GPU 上別的使用者 job 會偽裝成 stall/慢段（log 凍結 + GPU 高 util + py-spy 抓不到），別誤判成自己的 bug
metadata:
  node_type: memory
  type: project
  originSessionId: 3ce3483d-ce16-48c2-a4e6-11086ed2cc3e
---

機器是**單張 RTX 5090（32GB）共享**，別的使用者（如 laksh 的 DINO ReID `unsupervised_train.py market1501`）會同時跑。他們的 job 啟動/重運算時會搶 GPU，**偽裝成我們訓練的 stall 或慢段**：

**2026-06-15 實例**：SA1_v3e 在 iter 20（17:57）log 凍結 20 分，GPU 顯示 97%、py-spy 抓不到 stack。我誤判成 CUDA hang 並 kill。事後發現是 **laksh 的 job 在 17:58 啟動搶 GPU**。重啟後與 laksh 共存：GPU util 僅 36%（互搶 context-switch thrashing）、fps 從 5123 掉到 1736、SA1_v3e 爬行。laksh job 18:43 結束後 fps 立刻回 5123 全速。

**今天所有「fps 掉到 2000 的慢段」（SA2/SA3 屢見）很可能也是間歇 GPU 競爭，不是我們的 bug。**

**辨識：三種 GPU 異常 pattern**
| pattern | GPU util | log | py-spy | 真因 |
|---|---|---|---|---|
| CUDA wedge stall | **0%** | 凍結 | 卡 `_grad_l2_norm .item()` 同行 | 自己的 CUDA context 卡死 → kill+resume |
| **外部 GPU 競爭** | **高(我們沒在算)** 或忽高忽低 | 凍結或慢 | **抓不到 stack**（被 starve） | 別人 job 搶 → 非我們 bug |
| 正常慢段 | 中(30-40%) | **仍更新** | sensor/step 活躍 | rollout 較重，progressing |

**How to apply（診斷 stall 前必查）**：
1. log 凍結時，先 `nvidia-smi --query-compute-apps=pid,used_memory` + `ps -p <pid> -o user,cmd` 看 **GPU 上有沒有別人的 process**。
2. 若有別的 user job（非自己 train_rnn_car）佔大記憶體/算力 → 是**外部競爭，不是 stall，不要 kill 自己的**（kill 也沒用，重啟還是慢）。
3. GPU util 高但「我們的 py-spy 抓不到 stack」= 被 starve，不是 wedge（wedge 是 GPU 0% + 卡 .item()）。
4. 處置：(A) 接受慢速共存（記憶體放得下才行，單卡 32GB：IsaacLab ~14GB + 對方 job）；(B) 暫停等對方結束自動重啟；(C) 與對方協調。不可 kill 別人的 job。
5. 單卡無法分 GPU（CUDA_VISIBLE_DEVICES 無效，只有 GPU 0）。

相關：[[feedback_stall_autorecover]]（真 wedge 才 kill+resume）[[v22_physx_oom_crash]]
