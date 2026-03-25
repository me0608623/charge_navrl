# 地形導航 Curriculum (Terrain Navigation Curriculum)

## 總覽

本 Curriculum 設計用於訓練機器人在複雜靜態地形中的導航能力，採用 **Global + Local 分層架構**。

```
╔══════════════════════════════════════════════════════════════╗
║                    靜態幾何 → Global Planner                   ║
║  • 牆壁、房間結構、走廊、門口                                   ║
║  • AIT* 規劃全局路徑（1-5 Hz）                                 ║
║  • 幾百毫秒～幾秒算一次                                         ║
├──────────────────────────────────────────────────────────────────┤
║                    動態障礙 → Local Policy                      ║
║  • 行人、箱子、移動物體                                         ║
║  • LiDAR + RL 即時閃避（50-100 Hz）                            ║
║  • 不影響全局規劃                                               ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Phase 0: 車輛動力學校準 (Basic Kinematics)

**環境：** 16×16m 空房間，四面牆壁，無內部障礙物

**目標：** 學會開車（油門、方向盤、煞車）

| 特點 | 說明 |
|------|------|
| 地形 | 空房間，只有四面牆 |
| 障礙物 | **無** |
| 目標距離 | 3-8m |
| AIT* | 規劃直線路徑（測試座標轉換） |
| LiDAR | 開啟（檢測牆壁） |

**畢業標準：**
- 成功率 > 95%
- 無震盪（輪子轉向平滑）
- 路徑效率 > 0.9

**訓練命令：**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase0 \
    --num_envs 256
```

---

## Phase 1: 內牆結構導航 (Internal Wall Navigation)

**環境：** 16×16m 房間 + 內部牆壁結構

**目標：** 學會繞路策略，理解無法直線到達

### 地形設計

| 結構 | 描述 |
|------|------|
| **U 型牆** | 開口朝東，產生死胡同效果 |
| **隔間牆** | 帶門口的隔間結構 |
| **門口寬度** | 1.5m |

**AIT* 規劃：** 必須規劃繞過內牆的路徑，包含轉彎

**畢業標準：**
- 成功率 > 85%
- 能夠繞過 U 型牆
- 能夠找到並穿過門口

**訓練命令：**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase1 \
    --num_envs 256
```

---

## Phase 2: 走廊與窄通道導航 (Corridor & Narrow Passage)

**環境：** 16×16m 房間 + 走廊結構

**目標：** 學會精確控制，通過窄通道

### 地形設計

| 結構 | 描述 |
|------|------|
| **中央走廊** | 橫貫中央的主走廊 |
| **T 型路口** | 走廊交叉路口 |
| **窄門口** | 1.2-1.3m 寬（機器人寬度 ~1m） |

**AIT* 規劃：** 路徑沿著走廊前進，精確對準門口

**畢業標準：**
- 成功率 > 80%
- 能夠通過 1.2m 寬門口
- 能夠在走廊中保持方向

**訓練命令：**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase2 \
    --num_envs 256
```

---

## Phase 3: 複雜地形 + 動態障礙物 (Complex Terrain + Dynamic Obstacles)

**環境：** 16×16m 房間 + 複雜內牆 + 動態障礙物

**目標：** Global + Local 協作導航

### 地形設計

| 組成 | 描述 |
|------|------|
| **內牆** | 結合 Phase 1 和 2 的結構 |
| **動態障礙物** | 5 個移動的圓柱體（半徑 0.25-0.5m） |

### 架構說明

```
Global Planner (AIT*)
├── 輸入：靜態牆壁幾何
├── 輸出：全局路徑（繞過牆壁）
└── 頻率：1-5 Hz

Local Policy (RL)
├── 輸入：LiDAR + 局部目標
├── 輸出：速度命令
└── 頻率：50-100 Hz
```

**關鍵設計：** AIT* **不考慮**動態障礙物（避免 Planner Thrashing）

**畢業標準：**
- 成功率 > 75%
- 能夠同時處理靜態牆壁和動態障礙
- 能夠在障礙物擋路時等待或繞行

**訓練命令：**
```bash
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase3 \
    --num_envs 256
```

---

## Curriculum 進度

```
Phase 0 → Phase 1 → Phase 2 → Phase 3
   ↓         ↓         ↓         ↓
 基礎控制   繞路策略   精確控制   完整導航
 (空房間)  (內牆)   (走廊)   (+動態)
```

| Phase | 地形複雜度 | AIT* 路徑 | 標準成功率 |
|-------|-----------|----------|-----------|
| P0 | 空房間 | 直線 | > 95% |
| P1 | U型牆 | 繞路 | > 85% |
| P2 | 走廊 | 轉向 | > 80% |
| P3 | +動態障礙 | 複雜 | > 75% |

---

## 與舊 Curriculum 的對比

### 舊設計（❌ 不推薦）

```
Phase 1: 3 個障礙物
Phase 2: 10 個障礙物
Phase 3: 動態障礙物
```

**問題：** 直線路徑總是存在，不會強迫「繞路」

### 新設計（✅ 推薦）

```
Phase 0: 空房間（基礎運動學）
Phase 1: 內牆結構（U型牆、隔間）
Phase 2: 走廊與窄通道
Phase 3: 複雜地形 + 動態障礙物
```

**好處：** 強迫繞路，測試真實的導航能力

---

## 檔案結構

```
charge_sb3/cfg/
├── charge_env_cfg_phase0.py  # Phase 0 配置
├── charge_env_cfg_phase1.py  # Phase 1 配置
├── charge_env_cfg_phase2.py  # Phase 2 配置
├── charge_env_cfg_phase3.py  # Phase 3 配置
└── __init__.py               # 任務註冊
```

---

## 測試命令

```bash
# 測試 Phase 1 內牆結構
./isaaclab.sh -p scripts/demos/test_phase1_env.py \
    --task Isaac-Navigation-Charge-Phase1-Play

# 測試 Phase 2 走廊
./isaaclab.sh -p scripts/demos/test_phase2_env.py \
    --task Isaac-Navigation-Charge-Phase2-Play

# 測試 Phase 3 完整環境
./isaaclab.sh -p scripts/demos/test_phase3_env.py \
    --task Isaac-Navigation-Charge-Phase3-Play
```
