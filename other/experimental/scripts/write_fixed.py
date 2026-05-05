import os

content = """---
title: RL 基礎：Returns、GAE 與 Normalize Return
date: 2026-04-29
tags:
  - isaaclab
  - rl
  - returns
  - gae
  - normalize-return
aliases:
  - raw GAE return
  - returns 是 raw GAE
---

# RL 基礎：Returns、GAE 與 Normalize Return

> [!summary]
> 這份筆記解釋 `return`、`raw GAE return`、$\\text{vf\\_loss}$ 與 `normalize_return` 的關係。核心重點：critic 的 loss 量級由 value target 的尺度決定；target 是 raw return 或 normalized return，會讓同一個 $\\text{vf\\_coeff}$ 代表完全不同的更新強度。

## 1. Return 是什麼

`Return` 是從現在這個 state 開始，未來能拿到多少 reward 的總和。

目前 WD-style sparse reward 大致是：

| 事件 | reward |
|---|---:|
| 到達 goal | `+40` |
| 碰撞 | `-5` |
| 每步操作成本 | 約 `-0.03` |

例子：

$$
\\text{好的 episode: } +40 - 0.03 \\times 100 \\text{ steps} \\approx +37 \\\\
\\text{壞的 episode: } -5 - 0.03 \\times 50 \\text{ steps} \\approx -6.5 \\\\
\\text{混合 episode: } +40 - 5 - 0.03 \\times 200 \\text{ steps} \\approx +29
$$

再加上 GAE / discount 累積，raw return 可能落在約 $-10 \\sim 200$ 的量級。

`raw` 的意思就是：

```text
原始數值，沒有做 mean/std 縮放。
```

## 2. Critic 的工作

Critic 的任務是預測：

```text
我在這個 state，未來大概能拿多少 return？
```

數學上：

$$
\\text{value\\_loss} = \\operatorname{MSE}(\\text{value\\_pred}, \\text{value\\_target}) \\\\
           = \\operatorname{mean}((\\text{value\\_pred} - \\text{value\\_target})^2)
$$

如果 value target 是 raw return：

| 情況 | 計算 | vf_loss |
|---|---|---:|
| 預測得好 | `(100 - 105)^2` | `25` |
| 預測普通 | `(100 - 120)^2` | `400` |
| 預測很差 | `(50 - 200)^2` | `22500` |

所以 raw return 尺度下，$\\text{vf\\_loss}$ 本來就可能很大。

## 3. Normalize Return 是什麼

`normalize_return` 會把 value target 改成：

$$
\\text{normalized\\_return} = (\\text{return} - \\operatorname{mean}(\\text{return})) / \\operatorname{std}(\\text{return})
$$

結果：

$$
raw \\text{ return}:        [37, -6.5, 29, 42, ...] \\\\
\\frac{mean}{std}:        mean \\approx 30, std \\approx 30 \\\\
\\text{normalized\\_return}: [0.23, -1.22, -0.03, 0.40, ...]
$$

normalize 後：

$$
\\text{value\\_target\\_mean} \\approx 0 \\\\
\\text{value\\_target\\_std} \\approx 1
$$

這會讓 critic 的學習目標從「預測幾十到幾百的 raw score」變成「預測相對於平均表現的標準化分數」。

## 4. 為什麼同一個 $\\text{vf\\_coeff}$ 會變成不同意義

RL loss 裡 critic 的項目是：

$$
\\text{total\\_loss} = \\text{policy\\_loss} + \\text{vf\\_coeff} \\times \\text{value\\_loss} - \\text{entropy\\_loss}
$$

如果 target 是 raw return：

$$
\\text{value\\_loss} \\text{ 可能是 } 50 \\sim 200 \\\\
\\text{vf\\_coeff} \\times \\text{value\\_loss} = 0.025 \\times 100 = 2.5
$$

如果 target 是 normalized return：

$$
\\text{value\\_loss} \\text{ 初始可能約 } 1 \\\\
\\text{vf\\_coeff} \\times \\text{value\\_loss} = 0.025 \\times 1 = 0.025
$$

所以同樣的 `vf_coeff=0.025`，在兩種 target 尺度下不是同一件事。

## 5. 為什麼 clamp 會失效

WD-style clamp 大意是：

$$
\\text{if } \\text{vf\\_coeff} \\times \\text{vf\\_loss} > 30: \\\\
    \\text{壓住 } \\text{vf\\_loss}
$$

在 raw return 情況：

$$
0.025 \\times 100  = 2.5 \\\\
0.025 \\times 200  = 5.0 \\\\
0.025 \\times 1000 = 25.0
$$

要到：

$$
\\text{vf\\_loss} > 1200
$$

才會觸發 `30` 的門檻。

這表示如果 value loss 正常在 $50 \\sim 200$ 區間震盪，這個 clamp 幾乎不會動。

## 6. WD 的重點不是只有 $\\text{vf\\_coeff}$

WD A2C code 支援 `normalize_return`。若該設定啟用，critic loss 是對 normalized return 做 MSE。

這時候要注意兩件事：

| 項目 | 含意 |
|---|---|
| target scale | value target 變成 mean≈0、std≈1 |
| critic init | value head 初始輸出也應該在 normalized scale 附近 |

我們第一輪只開 `--normalize_return`，但 value head bias 仍是 `-8`，因此起點錯誤：

$$
\\text{target\\_mean} \\approx 0 \\\\
\\text{value\\_pred\\_mean} \\approx -8 \\\\
\\text{vf\\_loss} \\approx (-8 - 0)^2 = 64
$$

第二輪加上：

$$
--\\text{value\\_init\\_bias} \\approx 0.0
$$

才讓 critic 起點和 normalized target 對齊。

## 7. 一句話總結

| 項目 | raw return | normalized return |
|---|---:|---:|
| target mean | 可能幾十 | 約 `0` |
| target std | 可能幾十 | 約 `1` |
| vf_loss 正常量級 | `50~200+` | 約 `1` 起跳 |
| `vf_coeff=0.025` 效果 | 中等 | 很弱 |
| value init 合理 bias | 可偏負 | 應接近 `0` |

> [!important]
> `normalize_return` 不是單獨開就結束。它會改變 critic target 的尺度，所以 value head 初始化、$\\text{vf\\_coeff}$、VE 判讀都要跟著用同一個尺度理解。
"""

with open('/home/aa/Documents/Obsidian Vault/消融前/02_RL基礎_Returns-GAE與NormalizeReturn.md', 'w', encoding='utf-8') as f:
    f.write(content)
