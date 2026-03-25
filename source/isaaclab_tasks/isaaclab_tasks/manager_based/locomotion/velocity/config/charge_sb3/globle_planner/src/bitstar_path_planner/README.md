# BIT* Path Planner

Batch Informed Trees (BIT*) 路徑規劃演算法實作，適用於 ROS2。

## 專案簡介

BIT* 是一個漸進最優的採樣式路徑規劃演算法，結合了 RRT* 的優點與批次處理和啟發式採樣。本專案提供完整的 ROS2 實作，可用於機器人路徑規劃任務。

## 專案結構

```
bitstar_ws/
└── src/
    └── bitstar_path_planner/
        ├── bitstar_path_planner/      # Python 套件
        │   ├── __init__.py
        │   ├── bitstar.py             # BIT* 核心演算法
        │   ├── bitstar_planner_node.py # ROS2 規劃節點
        │   └── bitstar_visualizer.py   # 可視化節點
        ├── launch/                     # Launch 檔案
        │   └── bitstar_planner.launch.py
        ├── config/                     # 參數設定檔
        │   └── bitstar_params.yaml
        ├── test/                       # 測試檔案
        ├── package.xml
        ├── setup.py
        ├── CMakeLists.txt
        └── README.md
```

## 功能特點

- ✅ **漸進最優性**：保證在無限時間內找到最優解
- ✅ **批次採樣**：使用批次處理提高效率
- ✅ **啟發式搜索**：使用 informed set 加速收斂
- ✅ **雙向搜索**：同時維護前向和反向搜索樹
- ✅ **ROS2 整合**：完整的 ROS2 節點和服務介面
- ✅ **可視化支援**：提供路徑和標記的可視化

## 依賴需求

- ROS2 (Humble/Iron 或更新版本)
- Python 3.8+
- NumPy
- rclpy
- geometry_msgs
- nav_msgs
- visualization_msgs

## 安裝與建置

### 1. 進入 workspace

```bash
cd ~/Ros/bitstar_ws
```

### 2. 安裝依賴

```bash
# 確保已安裝 ROS2
source /opt/ros/<your-ros2-distro>/setup.bash

# 安裝 Python 依賴（通常已包含在 ROS2 中）
```

### 3. 建置專案

```bash
colcon build --packages-select bitstar_path_planner
```

### 4. Source workspace

```bash
source install/setup.bash
```

## 使用方法

### 啟動規劃節點

```bash
# 使用 launch 檔案（推薦）
ros2 launch bitstar_path_planner bitstar_planner.launch.py

# 或直接執行節點
ros2 run bitstar_path_planner bitstar_planner_node
```

### 參數設定

可以透過 launch 參數或 YAML 檔案設定：

```bash
ros2 launch bitstar_path_planner bitstar_planner.launch.py \
    max_iterations:=10000 \
    max_batch_size:=200 \
    goal_radius:=0.3 \
    rewire_radius:=3.0
```

### 訂閱地圖

節點會訂閱 `/map` topic（`nav_msgs/OccupancyGrid`），確保地圖已發布：

```bash
# 檢查地圖是否發布
ros2 topic echo /map --once
```

### 規劃路徑

目前節點會自動規劃（未來將提供服務介面）。路徑會發布到：

- `/bitstar_path` (nav_msgs/Path) - 規劃的路徑
- `/bitstar_markers` (visualization_msgs/MarkerArray) - 可視化標記

### 可視化

在 RViz2 中訂閱以下 topics：

- `/bitstar_path` - 路徑線條
- `/bitstar_markers` - 起點（綠色）、終點（紅色）、路徑（藍色）

## 演算法參數說明

### max_iterations
- **類型**：整數
- **預設值**：5000
- **說明**：BIT* 演算法的最大迭代次數。增加此值可提高找到解的概率，但會增加計算時間。

### max_batch_size
- **類型**：整數
- **預設值**：100
- **說明**：每批次採樣的樣本數量。較大的批次可提高效率，但會增加記憶體使用。

### goal_radius
- **類型**：浮點數（米）
- **預設值**：0.5
- **說明**：目標區域的半徑。當節點在此半徑內時，視為到達目標。

### rewire_radius
- **類型**：浮點數（米）
- **預設值**：2.0
- **說明**：重新連接的最大半徑。較大的值可找到更好的路徑，但計算成本更高。

### goal_bias
- **類型**：浮點數（0.0-1.0）
- **預設值**：0.05
- **說明**：直接採樣目標的概率。增加此值可加快找到初始解，但可能減少探索。

## 演算法原理

BIT* 演算法結合了以下技術：

1. **批次採樣**：每次迭代生成多個樣本，提高效率
2. **Informed Set**：在當前最佳解的橢圓區域內採樣，加速收斂
3. **雙向搜索**：同時維護從起點和終點出發的搜索樹
4. **Rewiring**：優化樹結構以找到更短的路徑

## 使用範例

### Python API 使用

```python
from bitstar_path_planner.bitstar import BITStar
import numpy as np

# 定義起點和終點
start = np.array([0.0, 0.0])
goal = np.array([10.0, 10.0])

# 定義邊界
bounds_min = np.array([-5.0, -5.0])
bounds_max = np.array([15.0, 15.0])

# 定義碰撞檢查函數
def collision_checker(position):
    # 簡單範例：避開圓形障礙物
    obstacles = [(5.0, 5.0, 2.0)]  # (x, y, radius)
    for ox, oy, r in obstacles:
        dist = np.sqrt((position[0] - ox)**2 + (position[1] - oy)**2)
        if dist < r:
            return False
    return True

# 創建規劃器
planner = BITStar(
    start=start,
    goal=goal,
    bounds=(bounds_min, bounds_max),
    collision_checker=collision_checker,
    max_iterations=5000,
    max_batch_size=100,
)

# 執行規劃
success, path, cost = planner.plan()

if success:
    print(f"找到路徑！成本: {cost:.2f}")
    print(f"路徑點數: {len(path)}")
else:
    print("規劃失敗")
```

## 測試

執行測試（未來將添加）：

```bash
colcon test --packages-select bitstar_path_planner
```

## 注意事項

1. **地圖要求**：節點需要有效的 `nav_msgs/OccupancyGrid` 地圖才能進行規劃
2. **計算資源**：BIT* 演算法計算密集，複雜環境可能需要較多時間
3. **參數調整**：根據環境複雜度調整參數以平衡速度和品質
4. **ROS2 版本**：確保使用相容的 ROS2 版本

## 未來改進

- [ ] 添加 ROS2 Action 介面
- [ ] 支援 3D 路徑規劃
- [ ] 添加動態障礙物處理
- [ ] 性能優化（C++ 實作）
- [ ] 單元測試和整合測試
- [ ] 更多可視化選項

## 參考文獻

- Gammell, J. D., et al. "Batch Informed Trees (BIT*): Sampling-based optimal planning via the heuristically guided search of implicit random geometric graphs." *2015 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2015.

## 授權

MIT License

## 聯絡資訊

如有問題或建議，請聯繫專案維護者。
