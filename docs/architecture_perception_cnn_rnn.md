# 感知網路架構 (CNN + RNN + 多幀 LiDAR)

> 圖解與白話原理見 Obsidian: `rnn/2026-07-17_CNN-RNN-多幀LiDAR架構圖解.md`
> 程式: `scripts/reinforcement_learning/skrl/models/modular_rnn_models.py` (`LidarStateExtractor`)

## 資料流一條線

**多幀 LiDAR → CNN(看空間) → 拼 State → RNN(記時間) → policy/value head**

```
LiDAR [K幀×72] ──Conv1d──► [B,64,18] ──flatten──► [B,1152] ──Linear──► 空間特徵 64D ┐
State (ego4+goal2+time1+act_hist4 = 11D) ──MLP──► 32D ─────────────────────────────┤► cat 96D ─► RNN ─► v, ω
```

## 三個元件

- **CNN (Conv1d)**：看單幀「空間」——障礙在哪個方位。角度軸用 `padding_mode='circular'`
  （雷射 355°↔0° 是鄰居）；新架構 flatten 保留方位（舊版 `AdaptiveMaxPool1d` 會丟掉方向）。
- **多幀 LiDAR (`lidar_frame_stack`, 現行 K=8)**：把 8 幀連續 LiDAR 疊成 Conv1d 的 8 個
  input channel，CNN 直接算跨幀差分（≈光流）→ 障礙在接近還是遠離。單幀看不出速度；
  K8 是 RNN 做 MOT 的必要條件。
- **RNN**：吃 CNN+State 每步 96D 特徵，靠 hidden state 串起時間 → 追蹤行人動態。
  episode reset 歸零，不跨場景污染。

## 為什麼 CNN 是「64D」（兩個 64 不要混）

1. `LIDAR_CONV_CH=64`（`modular_rnn_models.py:61`）：Conv1d 最終輸出 **64 個 channel**
   （≈64 種 pattern 偵測器）。角度軸 72 經兩次 stride=2 卷積壓成 18（`LIDAR_CONV_LEN`），
   所以中間張量是 `[64, 18]` = 1152 個數。
2. 最後 `nn.Linear(1152 → 64)`（`modular_rnn_models.py:129`）：把這 1152 壓成
   **64D 的 LiDAR 分支輸出**，再 LayerNorm。這個 64 是設計選的維度，配 State 分支 32D
   → 合成 96D 餵 RNN。

> 結論：「空間特徵 64D」= LiDAR 分支經 Linear 投影後的**輸出維度**（超參數），
> 與 conv channel 數剛好同為 64 但意義不同——一個是「幾種 filter」，一個是「壓成幾維交給 RNN」。
