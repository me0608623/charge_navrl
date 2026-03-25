# Curriculum Learning Strategy: Why Adaptive First?
## 課程學習策略分析與未來規劃 (Curriculum Learning Analysis & Roadmap)

---

### Slide 1: 為什麼一定要用課程學習？ (Why Curriculum Learning?)

**標題：解決強化學習在稀疏獎勵環境下的「冷啟動」問題**

**1. 核心挑戰 (The Challenge)**
*   **稀疏獎勵 (Sparse Reward)**：我們的目標是「到達 15m 外的終點」。但在訓練初期，Agent 連走 1m 都很困難。如果只給終極目標的獎勵，Agent 在 99% 的時間裡都只能拿到 0 分。
*   **探索效率低 (Inefficient Exploration)**：在巨大的狀態空間中，Agent 隨機亂走能撞大運走到終點的機率幾乎為零。這導致訓練初期 **收斂極慢**，甚至無法收斂。

**2. 解決方案：課程學習 (The Solution)**
*   **原理**：模仿人類學習過程。先學走 (1m)，再學跑 (5m)，最後學長跑 (15m)。
*   **優勢**：
    *   **引導探索 (Guided Exploration)**：將困難的大任務分解為一系列由簡入繁的小任務。
    *   **平滑梯度 (Smoother Gradients)**：讓 Agent 在每個階段都能獲得正向反饋，保持學習動力，避免陷入局部最佳解 (Local Minima)。

**3. 學術依據 (Academic Justification)**
*   Bengio et al. (2009) 提出 "Curriculum Learning"，證明「由簡入繁」能顯著加速收斂並提升最終表現。
*   在我們的導航任務中，距離就是最天然的 **難度指標 (Difficulty Metric)**。

---

### Slide 2: 現況分析 - 為什麼現在選擇「自適應」？ (Why Adaptive Now?)

**標題：自適應課程 (Adaptive CL) vs 線性/指數課程 (Linear/Exp CL)**

**1. 我們目前的策略：自適應課程 (Adaptive Curriculum)**
*   **機制**：監控成功率 (Success Rate)。 `SR > 80% → 升級`；`SR < 50% → 降級`。
*   **為什麼現在用它？ (Rationale)**
    *   **容錯率高**：我們還在摸索環境參數（如獎勵權重、物理摩擦力）。自適應機制能 **自動找到模型當前的能力邊界 (Performance Boundary)**。
    *   **避免崩潰**：Agent 在學習受阻時如果不降級，很容易「崩潰」導致遺忘之前學到的技能。自適應機制能動態回調難度，保護模型。
    *   **適合研發期**：當我們還不確定最佳的訓練步數 (Steps) 時，自適應能保證模型永遠在學習，不會因為課程安排太快或太慢而浪費時間。

**2. 潛在缺點 (Limitations)**
*   **訓練時間不確定**：因為難度是動態調整的，我們無法預知需要多少 Steps 才能達到 Level 10。
*   **震盪風險**：若參數設定不當，模型可能在兩個難度等級間反覆跳動 (Oscillation)。

---

### Slide 3: 未來規劃 - 線性與指數課程的引入 (Future Work)

**標題：邁向更高效、可預測的訓練排程**

**1. 下一階段：線性課程 (Linear Curriculum)**
*   **定義**：難度隨訓練步數 (Steps) **線性增加**。例如：每 1M Steps 增加 1m 距離。
*   **適用時機**：當我們已經透過自適應課程摸清了模型的學習速度後。
*   **優勢**：
    *   **可預測性 (Predictability)**：我們可以精確計算訓練所需的總時間與資源。
    *   **強迫進步**：給予模型固定的「進度壓力」，迫使它在有限時間內優化策略，有時能激發出更強的潛能。

**2. 進階目標：指數課程 (Exponential Curriculum)**
*   **定義**：難度隨訓練步數 **指數增加**。例如：初期慢速增長 (2m->3m)，後期快速增長 (5m->10m->15m)。
*   **適用時機**：當模型建立了穩固的基底能力 (Base Policy) 後。
*   **邏輯**：
    *   **能力爆發 (Emergent Capabilities)**：深度強化學習模型通常在突破某個臨界點後，泛化能力會大幅提升。指數課程符合這種「頓悟」式的學習曲線。
    *   **高效收尾**：在後期快速拉升難度，能有效驗證模型的極限與魯棒性 (Robustness)。

**3. 總結 (Conclusion)**
*   目前使用 **Adaptive** 是為了 **穩健 (Robustness)** 與 **探索 (Exploration)**。
*   未來將轉向 **Linear/Exponential** 以追求 **效率 (Efficiency)** 與 **可控性 (Controllability)**。

---
