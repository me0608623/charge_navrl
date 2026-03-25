# Ray Caster Monitor 使用說明

## 概述

Ray Caster Monitor 用於監控訓練過程中 Ray Caster 感測器的運行狀態和數據。

## 文件說明

1. **`ray_caster_monitor.py`** (核心類別)
   - 位置：`source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_monitor.py`
   - 功能：監測類別定義，用於記錄和統計 ray_caster 數據

2. **`monitor_ray_caster_live.py`** (實時監控腳本) ⭐ 推薦
   - 位置：`scripts/tools/monitor_ray_caster_live.py`
   - 功能：獨立運行的監控腳本，無需修改訓練腳本，創建獨立環境來檢查 ray_caster

3. **`monitor_ray_caster.py`** (數據監控腳本)
   - 位置：`scripts/tools/monitor_ray_caster.py`
   - 功能：讀取已記錄的訓練數據（需要訓練腳本中集成監測器）

## 使用方法

### 方式 1：實時監控（推薦，無需修改訓練腳本）

在訓練過程中，在另一個終端執行：

**方式 1：使用包裝腳本（推薦，最簡單）**

```bash
# 基本檢查（單次）
./scripts/tools/run_monitor_ray_caster.sh \
    --task Isaac-Navigation-Carter-v0 \
    --num_envs 1

# 持續監控（每 2 秒更新一次）
./scripts/tools/run_monitor_ray_caster.sh \
    --task Isaac-Navigation-Carter-v0 \
    --num_envs 1 \
    --watch \
    --interval 2

# 指定感測器名稱
./scripts/tools/run_monitor_ray_caster.sh \
    --task Isaac-Navigation-Carter-v0 \
    --num_envs 1 \
    --sensor_name lidar \
    --watch
```

**方式 2：使用 isaaclab.sh**

```bash
# 基本檢查（單次）
./isaaclab.sh -p scripts/tools/monitor_ray_caster_live.py \
    --task Isaac-Navigation-Carter-v0 \
    --num_envs 1

# 持續監控
./isaaclab.sh -p scripts/tools/monitor_ray_caster_live.py \
    --task Isaac-Navigation-Carter-v0 \
    --num_envs 1 \
    --watch \
    --interval 2
```

**重要提示**：

1. **對於 pip 安裝的 Isaac Sim**：必須正確設置環境變數才能運行腳本。推薦使用 `run_monitor_ray_caster.sh` 包裝腳本，它會自動設置所有必要的環境變數。

2. **如果直接使用 `python` 命令**：會出現 `ModuleNotFoundError: No module named 'pxr'` 或 `ModuleNotFoundError: No module named 'omni.kit'` 錯誤。

3. **解決方案**：
   - **推薦**：使用 `./scripts/tools/run_monitor_ray_caster.sh` 運行腳本（最簡單）
   - **替代**：使用 `./isaaclab.sh -p` 運行腳本
   - **進階**：如果 conda 環境已正確設置 Isaac Sim 環境變數，可以直接使用 `python` 命令

此方式會：
- 創建一個獨立的環境實例（不影響訓練）
- 檢查 ray_caster 感測器配置
- 執行幾步以獲取感測器數據
- 顯示感測器狀態和統計資訊

## 監控腳本輸出示例

```
======================================================================
Ray Caster 監控統計
======================================================================

總記錄步數: 500
最新步數: 500

感測器資訊:
  感測器數量: 1
  每感測器射線數: 360
  總射線數: 360

碰撞統計:
  平均有效碰撞數: 320.5
  平均有效碰撞比例: 89.03%
  最新有效碰撞數: 325
  最新有效碰撞比例: 90.28%

  ✅ 正常: 有效碰撞比例正常 (90.28%)

距離統計:
  平均距離: 2.345 m
  最小距離: 0.123 m
  最大距離: 8.567 m

感測器位置 (最新):
  X: 1.234 m
  Y: 2.345 m
  Z: 0.200 m
======================================================================
```

## 檢查 ray_caster 是否正常運行

監控腳本會自動檢查：
- ✅ **正常**：有效碰撞比例 > 90%
- ⚠️ **注意**：有效碰撞比例 1-90%
- ⚠️ **警告**：有效碰撞比例 < 1%（可能未正常工作）

## 注意事項

1. 訓練腳本必須使用 `train_with_ray_caster_monitor.py` 才能記錄數據
2. 監控腳本需要安裝 pandas：`pip install pandas`
3. 實時監控模式會持續更新，按 Ctrl+C 停止

