# BIT* Path Planner - 快速開始指南

## 專案位置

專案位於：`~/Ros/bitstar_ws/src/bitstar_path_planner`

## 快速建置步驟

### 1. 進入 workspace

```bash
cd ~/Ros/bitstar_ws
```

### 2. Source ROS2 環境

```bash
# 根據您的 ROS2 版本選擇（例如：humble, iron）
source /opt/ros/humble/setup.bash
```

### 3. 建置專案

```bash
colcon build --packages-select bitstar_path_planner
```

### 4. Source workspace

```bash
source install/setup.bash
```

## 測試演算法（無需 ROS）

```bash
cd ~/Ros/bitstar_ws
python3 src/bitstar_path_planner/test/test_bitstar.py
```

## 啟動 ROS2 節點

```bash
# 啟動規劃節點
ros2 launch bitstar_path_planner bitstar_planner.launch.py
```

## 檢查節點狀態

```bash
# 查看節點列表
ros2 node list

# 查看 topic 列表
ros2 topic list

# 查看路徑（如果有規劃結果）
ros2 topic echo /bitstar_path
```

## 注意事項

1. **地圖需求**：節點需要訂閱 `/map` topic 才能進行規劃
2. **首次使用**：建議先執行測試腳本確認演算法正常運作
3. **參數調整**：根據環境複雜度調整 `config/bitstar_params.yaml` 中的參數

## 下一步

- 閱讀 `src/bitstar_path_planner/README.md` 了解詳細使用說明
- 查看 `config/bitstar_params.yaml` 調整規劃參數
- 在 RViz2 中可視化規劃結果
