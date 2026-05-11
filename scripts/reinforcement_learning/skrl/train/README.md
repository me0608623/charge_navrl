# Train — 訓練入口腳本

各版本 Charge Navigation 的訓練 entry point。

| 檔案 | 功能 |
|------|------|
| `train_charge_ac.py` | 主訓練入口 (v2 139D obs)。支援消融實驗 CLI 參數 (v_gate, progress_gate, gap_reward, shield)，搭配 VLP16 Curriculum 8 階段課程 |
| `train_charge_aac.py` | Asymmetric Actor-Critic 訓練。Policy 用部分觀測、Critic 用完整觀測的非對稱架構 |
| `train_charge_NavRL.py` | NavRL dense reward 版本訓練。參考 NavRL 論文的 reward shaping 設計 |
| `train_rnn_car.py` | Multi-Agent Modular RNN 訓練 (v5/v7)。WD 架構移植: vanilla RNN + 雙 optimizer (RL head vs aux module) + A2CK |
| `train_rnn_car_wdclip.py` | train_rnn_car 的 WD-style grad clipping 變體。增加 merged/separate grad clip mode 對照實驗 |
