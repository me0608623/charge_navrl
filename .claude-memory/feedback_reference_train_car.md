---
name: WD reference is train_rnn_car not train_spot
description: WarpDrive 參考代碼是 train_rnn_car.py（car 模式），不是 train_spot_3d_state_distance.py（Spot）。作者說 car 直接訓練就能成功。
type: feedback
---

WarpDrive 的參考代碼是 `new_warp_drive/` 裡的 **train_rnn_car.py**（car 模式），不是 `train_spot_3d_state_distance.py`（Spot 機器人）。

**Why:** 用戶明確指出作者說「直接拿 train_rnn_car 來訓練理論上就成功了」。Spot 版本是不同任務（多 agent + 樓梯），不是我們的對照基準。

**How to apply:** 所有 WarpDrive 環境/curriculum/reward 的對齊工作，都以 `train_rnn_car.py` 的 phase config 為準。不要參考 `train_spot_3d_state_distance.py`。
