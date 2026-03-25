# AIT* Path Planner

Adaptively Informed Trees (AIT*) 路徑規劃演算法實作，適用於 ROS2。

## 專案簡介

AIT* 是 BIT* 的改進版本，使用自適應的 informed set 和更好的啟發式搜索策略，能夠更快地收斂到最優解。本專案提供完整的 ROS2 實作，可用於機器人路徑規劃任務。

## 專案結構

```
aitstar_path_planner/
├── aitstar_path_planner/      # Python 套件
│   ├── __init__.py
│   ├── aitstar.py             # AIT* 核心演算法
│   ├── aitstar_planner_node.py # ROS2 規劃節點
│   └── aitstar_visualizer.py   # 可視化節點
├── launch/                     # Launch 檔案
│   └── aitstar_planner.launch.py
├── config/                     # 參數設定檔
│   └── aitstar_params.yaml
├── test/                       # 測試檔案
│   └── test_aitstar.py
├── package.xml
├── setup.py
├── CMakeLists.txt
└── README.md
```

## 功能特點

- ✅ **漸進最優性**：保證在無限時間內找到最優解
- ✅ **自適應 Informed Set**：根據搜索進度動態調整採樣區域
- ✅ **混合採樣策略**：結合探索和利用（exploration + exploitation）
- ✅ **改進的啟發式**：使用更精確的啟發式函數加速收斂
- ✅ **雙向搜索**：同時維護前向和反向搜索樹
- ✅ **ROS2 整合**：完整的 ROS2 節點和服務介面
- ✅ **可視化支援**：提供路徑和標記的可視化

## AIT* vs BIT* 的改進

1. **自適應 Informed Set**：根據搜索進度動態調整採樣區域大小
2. **混合採樣策略**：結合 informed sampling 和 near-tree sampling
3. **改進的啟發式**：使用更精確的 f-value 計算
4. **優先級隊列**：使用優先級隊列管理搜索節點
5. **更好的收斂性**：通常比 BIT* 更快找到最優解

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
cd ~/Ros/globle_planner
```

### 2. 安裝依賴

```bash
# 確保已安裝 ROS2
source /opt/ros/<your-ros2-distro>/setup.bash
```

### 3. 建置專案

```bash
colcon build --packages-select aitstar_path_planner
```

### 4. Source workspace

```bash
source install/setup.bash
```

## 使用方法

### 啟動規劃節點

```bash
# 使用 launch 檔案（推薦）
ros2 launch aitstar_path_planner aitstar_planner.launch.py

# 或直接執行節點
ros2 run aitstar_path_planner aitstar_planner_node
```

### 參數設定

可以透過 launch 參數或 YAML 檔案設定：

```bash
ros2 launch aitstar_path_planner aitstar_planner.launch.py \
    max_iterations:=10000 \
    max_batch_size:=200 \
    adaptive_factor:=1.5 \
    min_informed_ratio:=0.15
```

### 訂閱地圖

節點會訂閱 `/map` topic（`nav_msgs/OccupancyGrid`），確保地圖已發布：

```bash
# 檢查地圖是否發布
ros2 topic echo /map --once
```

### 規劃路徑

路徑會發布到：

- `/aitstar_path` (nav_msgs/Path) - 規劃的路徑
- `/aitstar_markers` (visualization_msgs/MarkerArray) - 可視化標記

### 可視化

在 RViz2 中訂閱以下 topics：

- `/aitstar_path` - 路徑線條（青色）
- `/aitstar_markers` - 起點（綠色）、終點（紅色）

## 演算法參數說明

### max_iterations
- **類型**：整數
- **預設值**：5000
- **說明**：AIT* 演算法的最大迭代次數。

### max_batch_size
- **類型**：整數
- **預設值**：100
- **說明**：每批次採樣的樣本數量。AIT* 會根據搜索進度動態調整批次大小。

### goal_radius
- **類型**：浮點數（米）
- **預設值**：0.5
- **說明**：目標區域的半徑。

### rewire_radius
- **類型**：浮點數（米）
- **預設值**：2.0
- **說明**：重新連接的最大半徑。

### goal_bias
- **類型**：浮點數（0.0-1.0）
- **預設值**：0.05
- **說明**：直接採樣目標的概率。

### adaptive_factor
- **類型**：浮點數
- **預設值**：1.2
- **說明**：自適應 informed set 的縮放因子。較大的值會擴大 informed set，加快收斂但可能降低最優性。

### min_informed_ratio
- **類型**：浮點數（0.0-1.0）
- **預設值**：0.1
- **說明**：informed set 相對於總空間的最小比例。較小的值允許更多探索。

## 演算法原理

AIT* 演算法的主要改進：

1. **自適應 Informed Set**：
   - 根據當前最佳解和搜索進度動態調整
   - 隨著搜索進行，逐漸縮小 informed set 以找到更優解

2. **混合採樣策略**：
   - 70% 探索：使用自適應 informed set 採樣
   - 30% 利用：在現有樹節點附近採樣

3. **改進的啟發式**：
   - 使用 f = g + h 值進行優先級排序
   - 更精確的成本估計

4. **優先級隊列**：
   - 使用堆積（heap）管理搜索節點
   - 優先處理最有希望的節點

## 使用範例

### Python API 使用

```python
from aitstar_path_planner.aitstar import AITStar
import numpy as np

# 定義起點和終點
start = np.array([0.0, 0.0])
goal = np.array([10.0, 10.0])

# 定義邊界
bounds_min = np.array([-5.0, -5.0])
bounds_max = np.array([15.0, 15.0])

# 定義碰撞檢查函數
def collision_checker(position):
    obstacles = [(5.0, 5.0, 2.0)]  # (x, y, radius)
    for ox, oy, r in obstacles:
        dist = np.sqrt((position[0] - ox)**2 + (position[1] - oy)**2)
        if dist < r:
            return False
    return True

# 創建 AIT* 規劃器
planner = AITStar(
    start=start,
    goal=goal,
    bounds=(bounds_min, bounds_max),
    collision_checker=collision_checker,
    max_iterations=5000,
    max_batch_size=100,
    adaptive_factor=1.2,
    min_informed_ratio=0.1,
)

# 執行規劃
success, path, cost = planner.plan()

if success:
    print(f"找到路徑！成本: {cost:.2f}")
    print(f"路徑點數: {len(path)}")
    print(f"搜索進度: {planner.search_progress*100:.1f}%")
else:
    print("規劃失敗")
```

## 測試

執行測試：

```bash
cd ~/Ros/globle_planner
python3 src/aitstar_path_planner/test/test_aitstar.py
```

## 性能比較

AIT* 相比 BIT* 的優勢：

- **更快的初始解**：自適應策略幫助更快找到第一個解
- **更好的收斂性**：混合採樣策略平衡探索和利用
- **更優的結果**：自適應 informed set 幫助找到更優解

## 注意事項

1. **地圖要求**：節點需要有效的 `nav_msgs/OccupancyGrid` 地圖
2. **計算資源**：AIT* 計算密集，複雜環境需要較多時間
3. **參數調整**：根據環境複雜度調整 `adaptive_factor` 和 `min_informed_ratio`
4. **ROS2 版本**：確保使用相容的 ROS2 版本

## 未來改進

- [ ] 添加 ROS2 Action 介面
- [ ] 支援 3D 路徑規劃
- [ ] 添加動態障礙物處理
- [ ] 性能優化（C++ 實作）
- [ ] 單元測試和整合測試
- [ ] 更多可視化選項

## 參考文獻

- Gammell, J. D., et al. "Adaptively Informed Trees (AIT*): Fast Asymptotically Optimal Path Planning through Adaptive Heuristics." *2020 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2020.

## 授權

MIT License

## 聯絡資訊

如有問題或建議，請聯繫專案維護者。
