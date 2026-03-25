# Global Path Planner Workspace

全域路徑規劃演算法 workspace，包含多種漸進最優的採樣式路徑規劃演算法實作。

## Workspace 結構

```
globle_planner/
├── src/
│   ├── bitstar_path_planner/    # BIT* (Batch Informed Trees) 路徑規劃
│   └── aitstar_path_planner/    # AIT* (Adaptively Informed Trees) 路徑規劃
├── build/                        # 建置目錄
├── install/                      # 安裝目錄
├── log/                          # 日誌目錄
└── README.md                     # 本文件
```

## 包含的演算法

### 1. BIT* (Batch Informed Trees)
- **位置**：`src/bitstar_path_planner/`
- **說明**：批次採樣的漸進最優路徑規劃演算法
- **特點**：使用 informed set 和批次處理提高效率
- **文檔**：參見 `src/bitstar_path_planner/README.md`

### 2. AIT* (Adaptively Informed Trees)
- **位置**：`src/aitstar_path_planner/`
- **說明**：BIT* 的改進版本，使用自適應 informed set
- **特點**：更快的收斂速度和更好的最優性
- **文檔**：參見 `src/aitstar_path_planner/README.md`

## 快速開始

### 建置所有套件

```bash
cd ~/Ros/globle_planner
source /opt/ros/<your-ros2-distro>/setup.bash
colcon build
source install/setup.bash
```

### 建置單一套件

```bash
# 建置 BIT*
colcon build --packages-select bitstar_path_planner

# 建置 AIT*
colcon build --packages-select aitstar_path_planner
```

### 測試演算法

```bash
# 測試 BIT*
python3 src/bitstar_path_planner/test/test_bitstar.py

# 測試 AIT*
python3 src/aitstar_path_planner/test/test_aitstar.py
```

## 演算法比較

| 特性 | BIT* | AIT* |
|------|------|------|
| 漸進最優性 | ✅ | ✅ |
| Informed Set | 固定 | 自適應 |
| 採樣策略 | Informed | 混合（探索+利用） |
| 收斂速度 | 中等 | 較快 |
| 計算複雜度 | 中等 | 中等-高 |
| 適用場景 | 一般路徑規劃 | 需要快速收斂的場景 |

## 使用建議

- **BIT***：適合一般路徑規劃任務，參數調整簡單
- **AIT***：適合需要快速找到高品質解的場景，參數調整更靈活

## 依賴需求

- ROS2 (Humble/Iron 或更新版本)
- Python 3.8+
- NumPy
- rclpy
- geometry_msgs
- nav_msgs
- visualization_msgs

## 授權

MIT License

## 參考文獻

- BIT*: Gammell, J. D., et al. "Batch Informed Trees (BIT*): Sampling-based optimal planning via the heuristically guided search of implicit random geometric graphs." *2015 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2015.

- AIT*: Gammell, J. D., et al. "Adaptively Informed Trees (AIT*): Fast Asymptotically Optimal Path Planning through Adaptive Heuristics." *2020 IEEE International Conference on Robotics and Automation (ICRA)*. IEEE, 2020.
