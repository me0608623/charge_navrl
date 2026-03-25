# USD → AIT* Pipeline 驗收測試指南

## 概述

完整的 USD → AIT* Pipeline 驗收測試系統，包含三個層級、八個測試，全面驗證牆壁幾何是否正確進入 AIT* 規劃器。

## 三個層級、八個測試

### A. 資料層驗收（牆真的進到 grid）

#### A1. 牆點數量統計
- **檢查項**：牆點數量是否「穩定且合理」
- **合格標準**：
  - 牆點數：數百～數千
  - grid cell 標記：不為 0，也不爆到幾十萬
- **失敗原因**：
  - 沒有檢測到牆壁 → 場景配置問題
  - 採樣點過少/過多 → 採樣間距問題
  - grid 標記為 0 → 座標轉換問題

#### A2. 範圍檢查
- **檢查項**：grid index 是否越界
- **合格標準**：
  - grid_x min/max 在 [0, grid_width) 內
  - grid_y min/max 在 [0, grid_depth) 內
  - local 範圍落在房間尺寸內（16×16 房間約 [-8, 8]）
- **失敗原因**：
  - 有點越界 → 座標轉換錯誤
  - 範圍超出邊界 → env_origin 錯誤

#### A3. Probe 測試
- **檢查項**：隨機抽 20 個牆點，查 occupancy_grid 是否為 1
- **合格標準**：所有 probe 點都在 grid 中被標記為 1
- **失敗原因**：
  - 有點未被標記 → add_wall_points_to_occupancy_grid 有問題

---

### B. 幾何層驗收（紫球與牆對齊，門口沒被封）

#### B1. Doorway Probe
- **檢查項**：門口位置的 occupancy_grid 必須為 0
- **合格標準**：
  - Phase 1: U-Wall 開口、Partition 門口可通行
  - Phase 2: 走廊門口、T 型路口、窄通道可通行
  - Phase 3: 主要門口可通行
- **失敗原因**：
  - 門口被封死 → bbox 方式把門口也當牆、膨脹半徑過大

#### B2. 膨脹策略驗收
- **檢查項**：inflation 半徑是否與車寬一致
- **合格標準**：
  - 車寬 0.6m、grid 0.1m → inflation 約 3 cells
  - 不應為 0（會擦牆）
  - 不應過大（會封死門口）
- **失敗原因**：
  - inflation 為 0 → 機器人會擦牆
  - inflation 過大 → 門都被封死

---

### C. 行為層驗收（AIT* 路徑真的改變）

#### C1. A/B 測試
- **檢查項**：清空 grid vs 有牆 grid 的路徑差異
- **合格標準**：
  - 清空 grid: 路徑幾乎直線
  - 有牆 grid: 路徑繞行，node 數變多
- **失敗原因**：
  - 路徑長度相近 → 牆壁沒有生效

#### C2. 牆兩側測試
- **檢查項**：start/goal 分在牆兩側，路徑必須繞行
- **合格標準**：
  - Phase 1: 繞過 U-Wall、穿過 Partition 門口
  - Phase 2: 穿過走廊門口、T 型路口
  - Phase 3: 繞過複雜地形
- **失敗原因**：
  - 路徑只有 2 點 → 沒有繞行，牆壁沒生效

#### C3. hit-wall 計數器
- **檢查項**：規劃過程中有無因牆壁而無效的狀態
- **合格標準**：num_invalid_due_to_wall > 0
- **失敗原因**：
  - 永遠是 0 → 牆壁沒有在規劃時被查到

#### C4. 失敗模式檢查
- **檢查項**：路徑是否有異常行為
- **合格標準**：
  - 無路徑震盪（相鄰點距離過小）
  - 無過度曲折（繞路比 < 3x）
- **失敗原因**：
  - 路徑震盪 → 牆太鋸齒或 grid 太粗
  - 找不到路 → 門口被封

---

## 使用方式

### 1. 運行驗收測試

```bash
# 測試 Phase 1（U型牆、隔間）
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase1 \
    --num_envs 4

# 測試 Phase 2（走廊、窄通道）
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase2 \
    --num_envs 4

# 測試 Phase 3（複雜地形 + 動態障礙）
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase3 \
    --num_envs 4
```

### 2. 關閉驗收測試（正式訓練時）

在配置文件中設置 `run_verification: False`：

```python
plan_aitstar_path = EventTerm(
    func=plan_aitstar_and_update_local_goal,
    mode="reset",
    params={
        ...
        "run_verification": False,  # 關閉驗收測試
    },
)
```

---

## 預期輸出

### 基礎輸出（USD → AIT* Pipeline）

```
======================================================================
USD → AIT* Pipeline 驗證
======================================================================
  env_0:
    USD 牆壁採樣點: 1234 個
    添加到 grid: 456 個網格
    採樣範圍 local: x=[-7.50, 6.80] y=[-7.20, 7.10]
  env_1:
    USD 牆壁採樣點: 1234 個
    添加到 grid: 456 個網格
    ...
======================================================================
總計: 4936 個牆壁採樣點 → AIT* occupancy grid
AIT* 現在「看到」牆壁拓撲結構！
======================================================================
```

### 驗收測試輸出（8 個測試）

```
======================================================================
🔍 USD → AIT* Pipeline 驗收測試
======================================================================
  env_0:
    [WallGeometrySampler] 檢測到牆壁: env_0/Wall_North
    [WallGeometrySampler] 檢測到牆壁: env_0/Wall_South
    [WallGeometrySampler] 檢測到牆壁: env_0/Wall_East
    [WallGeometrySampler] 檢測到牆壁: env_0/Wall_West
    [WallGeometrySampler] 檢測到牆壁: env_0/U_Wall_Top
    [WallGeometrySampler] 檢測到牆壁: env_0/U_Wall_Left
    [WallGeometrySampler] 檢測到牆壁: env_0/U_Wall_Bottom
    [WallGeometrySampler] 檢測到牆壁: env_0/Partition_Top
    [WallGeometrySampler] 檢測到牆壁: env_0/Partition_Bottom
    [WallGeometrySampler] 檢測到牆壁: env_0/Partition_Side

======================================================================
Pipeline 驗收測試結果 - PHASE1 (env_0)
======================================================================

✓ PASS A1. 牆點數量統計
    檢測到 9 面牆，採樣 1234 點，標記 456 個 grid cell
    num_walls: 9
    num_sampled: 1234
    num_grid_marked: 456

✓ PASS A2. 範圍檢查
    Grid index 範圍正常: x=[0, 159], y=[0, 149]
    grid_x_min: 0
    grid_x_max: 159
    grid_y_min: 0
    grid_y_max: 149
    local_x_range: (-7.5, 6.8)
    local_y_range: (-7.2, 7.1)

✓ PASS A3. Probe 測試
    所有 20 個隨機 probe 點都在 grid 中正確標記為 1
    num_probes: 20
    marked_count: 20
    unmarked_count: 0

✓ PASS B1. Doorway Probe
    所有 2 個門口通過 probe 測試，可通行
    num_doorways: 2
    tested_count: 10
    blocked_count: 0

✓ PASS B2. 膨脹策略驗收
    膨脹半徑正確: 3 cells (0.3m)
    robot_radius: 0.3
    obstacle_inflation: 0.3
    inflation_cells: 3

✓ PASS C1. A/B 測試
    路徑差異明顯: 清空 2 nodes vs 有牆 12 nodes
    empty_path_length: 2
    walls_path_length: 12

✓ PASS C2. 牆兩側測試
    所有 2 個牆兩側測試通過
    U-Wall 左→右: 8 nodes (PASS)
    Partition 左→右（需穿過門口）: 10 nodes (PASS)

✓ PASS C3. hit-wall 計數器
    檢測到 47 個因牆壁無效的狀態 (12.3%)
    num_invalid: 47
    num_valid: 334
    num_total: 381
    invalid_ratio: 12.3

✓ PASS C4. 失敗模式檢查
    路徑正常: 12 個節點
    path_length: 12

======================================================================
總計: 9 PASS, 0 WARN, 0 FAIL
======================================================================
```

---

## Isaac Sim 視口驗證

| 元素 | 顏色 | 說明 |
|------|------|------|
| 牆壁實體 | 灰色/棕色 | USD stage 中的牆壁 |
| **紫色小球** | 紫色 | **採樣的牆壁點**（驗證對齊） |
| 綠色線條 | 綠色 | AIT* 規劃的路徑 |
| 紅色箭頭 | 紅色 | 目標位置 |

### 驗證要點

1. **紫球貼牆**：紫色小球應該貼在牆壁表面
2. **路徑繞行**：綠色路徑應該繞過紫色小球區域
3. **門口無紫球**：門口位置不應該有紫色小球

### 異常診斷

| 症狀 | 可能原因 | 檢查項 |
|------|----------|--------|
| 紫球沒有對齊牆壁 | USD 採樣位置錯誤 | A2 範圍檢查 |
| 綠色路徑穿過紫球 | 座標轉換錯誤 | A3 Probe 測試 |
| 綠色路徑直線穿牆 | 牆壁沒有進 grid | C1 A/B 測試 |
| 找不到路徑 | 門口被封死 | B1 Doorway Probe |
| 路徑貼牆震盪 | grid 太粗 | B2 膨脹策略 |

---

## 程式架構

```
mdp/path_planner/
├── pipeline_verification.py    # 驗收測試主模組 🆕
│   ├── TestStatus (Enum)       # 測試狀態: PASS/FAIL/WARN
│   ├── TestResult (dataclass)  # 單個測試結果
│   ├── VerificationResults     # 測試結果集合
│   ├── PipelineVerifier        # 驗收測試器
│   └── 8 個測試函數 (test_a1 ~ test_c4)
│
├── wall_geometry_sampler.py    # 牆壁幾何採樣
├── aitstar_adapter.py          # AIT* 規劃器
├── environment_map.py          # Occupancy grid
└── path_visualizer.py          # 路徑可視化
```

---

## 關鍵參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `sample_spacing` | 0.05m | 牆壁採樣間距（5cm） |
| `grid_resolution` | 0.1m | 網格解析度（10cm） |
| `robot_radius` | 0.3m | 機器人半徑 |
| `obstacle_inflation` | 0.3m | 障礙物膨脹半徑 |
| `inflation_cells` | 3 cells | 膨脹半徑（網格單位） |

---

## 故障排除

### 問題：沒有檢測到牆壁

**症狀**：
```
✗ FAIL A1. 牆點數量統計
    沒有檢測到任何牆壁！
```

**可能原因**：
1. 場景中沒有牆壁 asset
2. 牆壁命名不符合規範
3. `env.scene` 沒有正確加載牆壁

**解決方法**：
- 檢查場景配置中的牆壁 `prim_path`
- 確認牆壁名稱包含關鍵詞：`Wall_`, `U_Wall_`, `Corridor_`, `Partition_`, `Tee_`, `Narrow`

### 問題：門口被封死

**症狀**：
```
✗ FAIL B1. Doorway Probe
    門口被封死！8/10 個門口 probe 點被標記為障礙
```

**可能原因**：
1. bbox 採樣把門口也當牆
2. 膨脹半徑過大
3. 門口太窄

**解決方法**：
- 檢查 `wall_geometry_sampler.py` 中的 bbox 尺寸
- 減小 `obstacle_inflation`
- 確認門口寬度 > 機器人寬度

### 問題：座標轉換錯誤

**症狀**：
```
✗ FAIL A2. 範圍檢查
    有 123 個點的 grid index 越界！
```

**可能原因**：
1. `env_origin` 計算錯誤
2. `map_origin` 計算錯誤

**解決方法**：
- 檢查 `env.scene.env_origins` 是否正確
- 檢查 `planner.map.map_origin` 是否正確

---

## 總結

這 8 個驗收測試涵蓋了 USD → AIT* Pipeline 的所有關鍵點：

1. **資料層**：確保牆壁幾何正確進入 grid
2. **幾何層**：確保門口可通行、膨脹合理
3. **行為層**：確保 AIT* 路徑正確繞行

通過這些測試，你可以**確信** AIT* 規劃器「真的看到」了牆壁！
