---
name: Warp Drive Open Questions
description: new_warp_drive 專案尚未確認的技術問題清單
type: project
---

## 未確認事項

1. **preprocess_info_weight 維度含義**: 只知長度 7 (train_rnn_car.py:95)，各維度對應什麼物理量？
   - 驗證: 讀 spot_3dmodule.py 中 `network_feture` 定義

2. **spot_3dmodule vs spot_3d 差異**: 兩個環境各 1700+ 行，module 版本增加了什麼？
   - 驗證: `diff spot_3d/spot_3d.py spot_3dmodule/spot_3dmodule.py`

3. **models.py merge conflict**: Line 401-411 未解決的 `<<<<<<< HEAD` 標記
   - 影響 RNN output detach 時機（不影響 module_connected.py）
   - 驗證: 問原作者或比對 branch

4. **Auxiliary loss 實際效果**: PreProcess_Module_Loss 是否實際改善避障？
   - 驗證: 查 WandB project `Spot_3D_NEW` 歷史 runs

5. **PPO 是否被使用**: import 了但 train_rnn_car 用 A2C，其他腳本是否用 PPO？
   - 驗證: grep 所有 train 腳本的 algorithm 設定

6. **network_feture 定義**: preprocess back 輸出維度由此決定
   - 驗證: 搜尋 spot_3dmodule.py 中 `network_feture`

7. **論文 Ch.4 RNN 量化驗證**: PDF pages 60-80 中的 RNN 驗證實驗數據
   - 驗證: 繼續閱讀 PDF
