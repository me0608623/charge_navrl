---
name: 從工具啟動 GUI 必須顯式帶 DISPLAY，否則視窗開在看不見的地方
description: 2026-08-17 用 Bash 啟動 play_rnn_car GUI，進程是 XDG_SESSION_TYPE=tty 且無 DISPLAY，Isaac Sim 視窗完全不出現在使用者桌面；正解是從使用者看得到的程式抄環境變數
type: feedback
date: 2026-08-17
---

從工具啟動的 shell **沒有 `DISPLAY`**（`XDG_SESSION_TYPE=tty`），所以
`./isaaclab.sh -p play_rnn_car.py`（不帶 `--headless`）雖然正常跑、正常吃 GPU、
場景也正常建立，但**視窗不會出現在使用者桌面上**，看起來像沒啟動。

使用者桌面是 **Wayland + Xwayland**，可用的 X display 有 `X0/X1/X1024/X1025/X2`，
猜不出是哪一個。

## 正解：從使用者看得到的 GUI 程式抄環境變數

```bash
P=$(pgrep -u aa -f terminator | head -1)
tr '\0' '\n' < /proc/$P/environ | grep -E "^DISPLAY|^XAUTHORITY|^WAYLAND_DISPLAY|^XDG_RUNTIME_DIR"
```

本機當時得到（**XAUTHORITY 檔名每次 session 會變，不可寫死**）：

```
DISPLAY=:1
XAUTHORITY=/run/user/1001/.mutter-Xwaylandauth.XXXXXX
WAYLAND_DISPLAY=wayland-0
XDG_RUNTIME_DIR=/run/user/1001
```

啟動時四個都帶上，並用 `wmctrl -l` 或 `xdotool search --name Isaac` 驗證視窗真的存在。

**Why:** 沒有 `DISPLAY` 時**不會報錯**，log 一切正常，只是使用者看不到 ——
浪費一整輪來回，而且會誤判成「GUI 壞了」。

**How to apply:**
- 啟動任何 GUI 前先抄環境變數，啟動後**驗證視窗存在**再回報使用者
- 另外兩個 play GUI 的既有坑：
  - `BEV_VIS = True` 寫死，**關掉 BEV 視窗會直接中止 play**
    （log 會出現 `[PLAY] BEV 視窗已關閉，停止 play。`）；`--bev_vis` 是
    `store_true` 且 default=True，**無法從 CLI 關掉**
  - `--steps` 預設 3000；目視觀察要記得加大，否則使用者還沒切過去就結束
- 相關：[[feedback_no_edit_while_eval_running]]
