# Intelligent Autonomous Navigation: Sim-to-Real Project Overview
## 基於端到端深度強化學習的自主充電機器人 (End-to-End DRL for Autonomous Charging Robot)

---

### Slide 1: 專案背景與核心挑戰 (Project Context & The "Why")

**標題：解決自主移動機器人 (AMR) 的最後一哩路挑戰**

**1. 我們在解決什麼問題？ (Problem Statement)**
*   **現實痛點**：傳統導航演算法 (Slam + Path Planning) 在面對 **動態環境**（如忽然出現的人、複雜障礙）時，反應不夠即時，且建圖成本高昂。
*   **專案目標**：打造一個 **「不需要地圖、能看懂環境、會自己走」** 的智慧導航大腦。
    *   應用場景：自動充電機器人 (Charger Robot)，需要在複雜且充滿未知的停車場或室內環境中，精確找到目標。

**2. 為什麼選用 Sim-to-Real 技術？ (Why Simulation?)**
*   **數據成本**：在真實世界撞壞一台機器人成本太高。我們利用 NVIDIA Isaac Sim 創造 **「無限且零成本」** 的試錯環境。
*   **訓練效率**：模擬時間是現實的 1000 倍快。Agent 可以在幾分鐘內累積人類數年的駕駛經驗。

**3. 核心技術路線 (Technical Route)**
*   **端到端學習 (End-to-End Learning)**：
    *   輸入：雷達 (Lidar) + 自身狀態 (State)。
    *   輸出：直接控制馬達速度 (Velocity Command)。
    *   優勢：省略了繁瑣的建圖與規劃步驟，反應速度極快 (20ms)，具備人類般的直覺反應。

---

### Slide 2: 系統架構與訓練流程 (System Overview)

**標題：從模擬到現實的完整訓練管線 (Training Pipeline)**

**我們如何訓練這個「大腦」？**

```mermaid
graph LR
    subgraph "Simulation (Isaac Lab)"
        A[Physics Engine] -->|Lidar & State| B[Agent Brain (PPO Policy)]
        B -->|Velocity Action| A
        A -->|Reward/Penalty| C[Critic Network]
        C -.->|Update Gradients| B
    end
    
    subgraph "Curriculum Manager"
        D[Success Rate Monitor] -->|Adjust Difficulty| A
        note[Phase 1: Static -> Phase 2: Dynamic -> Phase 3: Long Range]
    end

    subgraph "Real World Deployment"
        B -->|Load Weights| E[Jetson Orin Nano]
        E -->|Control Signals| F[Real Robot]
    end
```

**關鍵模組說明：**
1.  **感知層 (Perception)**：使用 2D Lidar 掃描周圍 360 度環境，將障礙物資訊編碼為 131 維向量。
2.  **決策層 (Policy Network)**：一個深度神經網路 (MLP)，負責判斷「現在該加速還是轉彎」。
3.  **學習機制 (PPO Algorithm)**：
    *   **獎勵 (Reward)**：靠近目標 +分、走得快 +分、活著 +分。
    *   **懲罰 (Penalty)**：撞牆 -分、翻車 -分、原地打轉 -分。
    *   **結果**：Agent 會為了追求高分，自動學會「既快又穩」的導航策略。

---

### Slide 3: 訓練策略與當前進度 (Methodology & Status)

**標題：階段性訓練策略與自適應課程 (Curriculum Learning)**

**1. 為什麼不能一次練好？ (The Challenge of Complexity)**
*   直接把嬰兒丟進複雜迷宮，他學不會走路。同理，Agent 也需要 **「循序漸進」**。

**2. 我們的訓練三部曲 (Three-Phase Curriculum)**:
*   **Phase 1: 基礎學步 (Basic Locomotion)**
    *   *環境*：空曠無障礙。
    *   *目標*：學會控制輪子，不翻車，直走。
    *   *狀態*：✅ 已完成。Agent 能穩定移動。
*   **Phase 2: 靜態避障 (Static Avoidance)**
    *   *環境*：加入固定障礙物。
    *   *目標*：學會看懂 Lidar，繞過障礙。
    *   *狀態*：⚠️ 發現問題 (蛇行、恐懼)。正在進行優化。
*   **Phase 3: 長程導航 (Long-Range Navigation) - [目前階段]**
    *   *策略*：**「拆解學習」**。先移除障礙，專注訓練 15m 長距離的 **「絕對導航意志」**。
    *   *技術亮點*：引入 **自適應課程 (Adaptive Curriculum)**，自動根據勝率調整目標距離 (2m-15m)，確保訓練效率。

**3. 下一步計畫 (What's Next?)**
*   完成 Phase 3 長距離訓練，建立強健的 Base Policy。
*   進入 **Phase 4**：將動態障礙物加回環境，結合「導航意志」與「避障能力」，完成最終型態。

---
