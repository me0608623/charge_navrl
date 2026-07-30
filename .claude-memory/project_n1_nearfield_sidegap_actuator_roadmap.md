---
name: project-n1-nearfield-sidegap-actuator-roadmap
description: N1直穿後依序修近場LiDAR、4m走廊側窄口、動態牆人窄口，最後才做actuator-delay bridge
metadata:
  node_type: memory
  type: project
  status: superseded
  date: 2026-07-27
---

# N1 → 近場 LiDAR → 走廊側窄口 → 動態牆人窄口 → actuator bridge

> **2026-07-28 狀態更正：此路線保留作歷史，不再是目前執行順序。**
> W1-c10 實車出現舞龍舞獅、突然大腳命令與撞牆後，主線改為
> [[project_sa1_sim2real_v1_20260728]]：從 SA1 隨機初始化，同時納入窄縫、
> 長走廊、實測 VLP-16 雜訊與 actuator delay U{0,1,2}。N1、SA8 與
> λ=0.067 維持 HOLD；禁止依下方「最後才做 actuator bridge」再開新 run。

這是 2026-07-27 用戶核准的未來主方向。權威完整規格：
`docs/n1_nearfield_sidegap_actuator_roadmap_for_codex.md`，Obsidian：
`isaaclab/課程學習/2026-07-27_N1直穿_近場LiDAR_走廊側窄口_部署路線.md`。

## 新增實車事實

- 使用者確認實車在距 LiDAR 約 0.30m 時仍能看到人體的稀疏點雲。
- 因此現役 hard `r_min=0.5`（<0.5 全改 r_max）不符合部署感測。
- 舊筆記中「0.9m 完全盲」與「0.5m 是真實最小距離」不得再當未來血緣依據。
- 初步 hard cutoff 建議 0.25m，但必須先確認量測座標並用 0.30--0.80m ROI
  回波率凍結；0.25--0.60m 要做 per-ray 機率漏點，不可硬切。

## 嚴格順序

1. **N1 先完成**：固定牆、1.2--1.4m、yaw±4°、12% replay、scripted direct
   teacher CE；N1 goal 距牆 `U[2.0,4.0]m`、相對缺口橫偏
   `U[-1.5,+1.5]m`，要求先穿縫再轉向 goal；LiDAR/reward/network/DR 不動。
2. **NF0**：實車 0.30/0.40/0.50/0.60/0.80m × 角度/材質，每格至少200幀，
   量 ROI ray/bin keep rate 與連續漏幀，修 hard0.5。
3. **N2 靜態4m走廊側窄口**：表面淨寬1.0--1.6m、左右鏡像、goal後方/
   左前/右前；保持總68/10/12/10，只把12% narrow拆8%舊牆縫+4%靜態側縫。
4. **N3 動態牆＋人體窄口**：先做K8 observability probe（pass/yield label
   held-out≥90%才訓）；12% narrow拆8%舊牆縫+2%靜態側縫+2%動態側縫。
5. **最後 actuator bridge**：`enable_actuator_dr=True`，delay U{0,1,2}
   steps、motor lag α=0.3、velocity scale0.9--1.1；`obs_delay_steps=(0,0)`。
   OBB血緣已有83D（過去兩步 applied actions 4D），不需再改輸入維度。

## 安全語意

- 固定硬物 1.0m 窄口：幾何上可通，屬「應直穿」壓力題。
- 牆＋移動人體 1.0m：不是必穿題。預測1.5s內 gap≤1.0m或縮小就停讓；
  gap≥1.2m且穩定/擴大才低速通過（pass速度≤0.35m/s）。
- goal 在人體後方或右前方也不能覆蓋 YIELD。
- 靜態 direct teacher 絕不可套進動態人體場景。

## 初始訓練值

- N1：1024 env、10 iter、rollout128、save2、LR2e-4、model+Adam warm start；
  CE λ 已由同分布 no-update shadow 校準並凍結為 `0.067`。兩個有效 iteration
  的 λ50=0.03332/0.10024，pooled gradient norm 給 0.06669；舊的會更新 PPO
  的 row 已排除。shadow 現在硬性跳過 PPO/RNN/aux step，並記 update count /
  actor/critic param delta，重跑前也會封存舊 JSONL，避免混入陳舊資料。
- D0/W1/W2 的窄縫 goal 仍固定 `3.0m / 0m`；只有 N1 config 覆寫新範圍。
  N1 前先跑 randomized-goal teacher 三 seed（direct≥99%、CR≤1%），再做
  shadow λ 校準；舊共線 Gate5 保留作 regression。另先量 D0 在新 gate 的
  zero-shot 基線；N1 因同時修分布與 teacher loss，屬 capability arm，不可宣稱
  純單變因因果，除非另跑 randomized-goal/no-CE control。
- 2026-07-27 teacher 新分布前置閘已過：seed404/505/606 合計6706回合，
  direct 6706/6706、碰撞0、first-cross p95=4.2s、pre-cross lateral
  p95=0.158/0.160/0.160m。shadow λ 已完成並凍結為 0.067；下一步只剩
  D0 zero-shot 新 gate，N1 尚未啟訓。
- N2/N3：各先跑有界10 iter、save2，先gate再決定延長，不用聚合train SR裁決。
- N2 side-gap寬度：25% U[1.0,1.2]、50% U[1.2,1.4]、
  25% U[1.4,1.6]；yaw依寬度±2/±4/±8°且OBB validator優先。
- N3人體速度U[0.2,0.6]m/s；50% closing/25% stable/25% opening；
  future horizon1.5s/8 samples沿用。
- actuator bridge也先10 iter/save2，固定delay 0/1/2 step分開驗；禁止舊400
  iter盲跑。

## 禁止事項

- N1尚未裁決前不可同時改r_min。
- 不把1.0m動態人體gap的crossing率當唯一成功指標。
- 不開obs delay。
- clean gates未過前不開actuator DR。
- 每階段保留已驗 parent checkpoint，所有能力用det fixed-seed per-ckpt gates。
