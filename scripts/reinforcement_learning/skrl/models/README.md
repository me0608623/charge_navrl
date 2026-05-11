# Models — 神經網路架構定義

Charge Navigation 各版本的 Policy / Value Network 定義。

| 檔案 | 功能 |
|------|------|
| `charge_models.py` | Multi-Branch CNN 特徵萃取器 (SKRL)。三分支: Conv1d(LiDAR) + ObsMLP(obstacles) + StateMLP(ego+goal)，輸出 128D 合併特徵 |
| `vlp16_models.py` | VLP-16 v3 三分支網路 (139D obs)。單幀觀測、無 frame stacking，支援 CategoricalMixin 離散動作 (361 logits = 19×19) |
| `navrl_models.py` | NavRL-style 網路 (參考 IEEE RA-L 2025)。MLP 架構，支援 NavRL dense reward 配置下的 policy/value 分離 |
| `modular_rnn_models.py` | Modular RNN 模型 — 移植自 Warp Drive CustomModuleConnected。PreprocessRNN: FC→RNN→concat→middle FC→predict head，支援 auxiliary loss 訓練 RNN |
