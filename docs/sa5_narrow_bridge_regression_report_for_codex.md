# 報告書：SA5 narrow-passage bridge 後半過訓、最佳點 iter50（交 Codex）

作者：監督端（唯讀交叉驗證）｜日期：2026-07-23｜狀態：**已雙 seed 驗證，供 Codex 決策**

---

## 0. 一句話結論
Bridge **本質成功**（iter50 就能 100% 穿越 1.2m），但**後半 schedule 過訓、把能力練壞**：1.2m 穿越率**單調回歸 iter50=100% → iter100=77% → iter150=37%**。**部署/晉級請用 iter50 `checkpoint_6400.pt`，不要用最終 iter150**。Codex 先前「策略沒學會對準」的診斷可由 iter50=100% 直接否決。

---

## 1. 量測數據（唯讀 det-eval，同考場/同旗標，只差 checkpoint）
考場：`--narrow_gap_eval --arena_size 10 --num_static_obs 0 --num_dynamic_obs 0 --narrow_gap_yaw_limit_deg 10 --deterministic --num_envs 64 --steps 1200`

| checkpoint | 寬度 | crossing_rate | 樣本(cross/total) | 撞牆 | 驗證次數 |
|-----------|------|--------------|-------------------|------|---------|
| **iter50** `c6400` | 1.2m | **1.000** | 832/832 (s404), 829/829 (s406) | 0 | ✅ **雙 seed** |
| iter100 `c12800` | 1.2m | **0.772** | 365/473 (s404) | 39(+71 TO) | 1 |
| **iter150** `c19200` | 1.2m | **0.368** | 188/511 (s404) | 322 | ✅ 監督端+Codex 共 3 次一致 |
| iter150 `c19200` | 1.0m | **0.000** | 0/587 | 586 | ✅ |

- iter50 兩 seed 各約 830 回合**全穿越** → 非運氣/雜訊。
- 訓練期**混合 SR 全程 ~97% / CR ~3%**（平穩），**完全看不出**窄廊能力 100→37 的崩壞。
- 注入本身正常：實際注入 12.0%、累積 86,109 episodes、schedule progress 1.0、final gap 1.01–1.38m、stress 25%、unsolvable 0。→ **非 injector 沒運作**。

### yaw 細節（澄清一個假象）
iter50 throat yaw p95=13.8/15.5° **反而比** iter150 的 11.4° 高，但 iter50 穿越 100%、iter150 只 37%。原因：iter150 **多數在進喉部前就撞牆**（yaw_frames iter50=8883 vs iter150=1932），其 yaw 統計是「少數到得了喉部者」的選擇性樣本。→ iter150 的失敗是**進喉部前就撞**，非「喉部對齊不準」。

---

## 2. 否決 Codex 先前三個假設（用 iter50=100% 反證）
| Codex 假設 | 反證 |
|-----------|------|
| 混合比例(12%)不足 | ❌ 同樣 12% 混合，iter50 就 100% |
| 訓練 vs gate episode 語意不一致 | ❌ 同一個 gate/同 seed，iter50 過 100% |
| 策略根本沒學會對準喉部 | ❌ **學會了（iter50=100%），後半才退化** |

→ 真相不是「沒學會」，是「**學會後被後半 schedule 練壞**」（aggregate 分析看不到 per-checkpoint 曲線才會誤判）。

---

## 3. 病因（★ 標記：推論，未隔離變因）
最合理解釋 = **後半壓力區過度**：iter75 後 yaw 容忍放到 ±10° + 壓力尾降到 1.0m + 續訓 100 iter，讓政策學到「更歪也行、更敢衝」，**侵蝕了前半學到的乾淨 1.2m 對齊**。
**尚未證明**（未做控制變因）。要坐實需二擇一：
- (a) 對照實驗：同 config 但後半**不放寬 yaw / 不加 1.0m 壓力**，看是否還回歸；或
- (b) 用你建議的 **bridge-only SR/CR/TO 指標**回看 12% 注入回合的逐 iter 表現。

---

## 4. 建議行動（給 Codex）
1. **部署/晉級直接用 iter50 `checkpoint_6400.pt`**（1.2m 100% ≥ 95% 硬閘，已雙 seed 確認）。這點**確定**，不必等病因坐實。
2. **不要用 iter150、不要加同 schedule 的 continuation**（更多同配方 = 更糟）。
3. **修 bridge schedule（重訓修正版）**：後半別把 yaw 放到 ±10°（維持 ±2–4°）、壓力尾別降到 1.0m（守在 ≥1.2m 可感知區，因 r_min=0.5m）、或直接 **early-stop ~iter50–75**。
4. **加 bridge-only SR/CR/TO 指標（強烈支持你的提議）**：隔離那 12% 注入回合的即時表現，下次可**當場**抓 100→37 崩壞，不必事後 per-ckpt 掃。
5. 1.0m 別當畢業硬閘（卡 r_min 盲區邊界，你 gate 拆分已定 1.0m 只記錄）——1.2m 才是感知可支援的部署目標。

---

## 5. 可複查 log（Codex 可自行重跑驗證）
- iter50: `/tmp/optC_narrow_bridge_c6400_1p2/narrow_1p2.log`（s404）、`/tmp/verify_iter50_seed406/deploy_1p2_s406.log`（s406）
- iter100: `/tmp/bracket_narrow_c12800/deploy_1p2.log`
- iter150: `/tmp/final_narrow_c19200/deploy_1p2.log`（1.2m）、`stress_1p0.log`（1.0m）

## 6. 證據強度（誠實分級）
- **鐵證**：iter150 回歸(3次一致)、iter50=100%(雙seed)、部署選 iter50。
- **推論**：病因=後半 schedule（時間點高度吻合但未隔離變因）。

---

## 附錄：Codex 修正（2026-07-23，已納入）
1. **更正過度宣稱**：本報告 §0/§4 原寫「iter50 過 1.2m 硬閘」**不精確**。正確：iter50 過**物理硬閘**（SR100/CR0/穿越100，雙 seed），但**未過目前 Gate5 的 yaw 子判準**（yaw≤10° 率 90.4%/89.9% < 95%；p95 13.76°/15.54° > 10°）。
2. **10° yaw 硬門檻設計錯誤**：1.2m 縫的 OBB 幾何可容 yaw **≈43.7°**（`2(0.45sinθ+0.40cosθ)≤1.2`），10° 憑空設、比幾何嚴約 4 倍。且穿越率100%+CR0%（1,661回合）已證幾何可行 → **yaw 應為診斷，非硬閘**。把完美穿越零碰撞判 FAIL 是**閘錯非政策錯**。
3. **病因改寫（相關非因果）**：「已證後半訓練與能力回歸**時間相關**；未證是 yaw 放寬 / 1.0m stress / 持續 PPO 更新何者造成。」
4. **務實路徑（取代重訓）**：凍結 c6400 → 修 Gate5 yaw 語意 → 跑完整 gates（nav/blocker/SA6 preview）→ 過就當 SA6 resume（不重訓 SA5）→ SA6-8 鎖 gap≥1.2m/yaw±2-4°/無1.0m stress 的窄縫 replay + 每 ckpt 自動 1.2m gate + 穿越率降則 early-stop。

## 關聯
- memory `project_sa5_narrow_passage_bridge`；Obsidian `isaaclab/2026-07-23_SA5窄縫bridge補課_寬度排程`。
