# USD → AIT* Pipeline 驗證指南

## 目標

確保 AIT* 規劃器「真正看到」USD stage 中的牆壁拓撲結構，而不是抽象的邊界矩形。

## 完整 Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USD Stage                                     │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌───────────┐                   │
│  │Wall_North│  │U_Wall   │  │Corridor  │  │Partition │                   │
│  │         │  │         │  │          │  │           │                   │
│  └─────────┘  └─────────┘  └──────────┘  └───────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│              wall_geometry_sampler.sample_wall_geometry_from_usd()    │
│                                                                       │
│  1. 查找所有牆壁 prims (Wall_.*, U_Wall_.*, Corridor_.*, 等)      │
│  2. 獲取 bounding box (ComputeWorldBound)                         │
│  3. 在牆壁表面採樣點（每 5cm 一點）                                │
│  4. 返回 (sampled_points_local, sampled_points_world)               │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      座標轉換                                     │
│                                                                       │
│  sampled_points_local = sampled_points_world - env_origin            │
│                                                                       │
│  例子:                                                                 │
│    env_origin[0] = (9, -9)                                          │
│    sampled_world[0] = (12, -11)                                       │
│    sampled_local[0] = (12-9, -11+9) = (3, -2)                       │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│          add_wall_points_to_occupancy_grid()                        │
│                                                                       │
│  將採樣點轉換為網格坐標                                               │
│  標記 occupancy_grid[x, y] = 1                                       │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   AIT* Planner.is_state_valid()                     │
│                                                                       │
│  def is_state_valid(state):                                          │
│      grid_x, grid_y = world_to_grid(state.position)                 │
│      return not occupancy_grid[grid_x, grid_y]                        │
│                                                                       │
│  AIT* 現在「看到」USD 中的牆壁！                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## 調試輸出

運行 Phase 1/2/3 時，你會看到：

```
======================================================================
USD → AIT* Pipeline 驗證
======================================================================
  env_0:
    USD 牆壁採樣點: 1234 個
    添加到 grid: 456 個網格
    採樣範圍 local: x=[-7.50, 6.80] y=[-7.20, 7.10]
  env_1:
    USD 牆壁採樣點: 1456 個
    添加到 grid: 567 個網格
    採樣範圍 local: x=[-7.30, 7.00] y=[-7.50, 6.90]
  ...
======================================================================
總計: 5678 個牆壁採樣點 → AIT* occupancy grid
AIT* 現在「看到」牆壁拓撲結構！
======================================================================
```

## 可視化驗證

在 Isaac Sim 視口中：

| 元素 | 顏色 | 說明 |
|------|------|------|
| 牆壁實體 | 灰色 | USD stage 中的牆壁 |
| 綠色線條/點 | 綠色 | AIT* 規劃的路徑 |
| **紫色小球** | 紫色 | **採樣的牆壁點** |
| 紅色箭頭 | 紅色 | 目標位置 |

**驗證對齊：**
- 紫色小球應該貼在牆壁表面
- 綠色路徑應該繞過紫色小球區域
- 如果紫色小球沒有對齊牆壁 → USD 採樣有問題
- 如果綠色路徑穿過紫色小球 → 座標轉換有問題

## 測試命令

```bash
# 測試 Phase 1（U型牆、隔間）
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase1 \
    --num_envs 4

# 測試 Phase 2（走廊、窄通道）
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Phase2 \
    --num_envs 4
```

## 檢案結構

```
mdp/path_planner/
├── wall_geometry_sampler.py    # USD 採樣器
├── environment_map.py             # Occupancy grid
├── aitstar_adapter.py              # AIT* 適配器
└── path_visualizer.py             # 可視化器
```
