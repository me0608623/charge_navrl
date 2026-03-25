# 模組化工作訓練測試說明

## 概述

此測試腳本用於驗證模組化後的環境是否可以正常進行訓練。測試包括：

1. **環境創建測試** - 驗證環境配置和環境實例是否可以正常創建
2. **環境重置測試** - 驗證環境重置功能是否正常，觀測格式是否正確
3. **訓練步驟測試** - 執行多個訓練步驟，驗證環境動態是否正常
4. **模組化組件檢查** - 驗證所有模組化組件（觀測、獎勵、終止條件、事件）是否正常

## 使用方法

### 方法 1: 使用 Isaac Lab 啟動器（推薦）

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/training_test.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 4 \
    --max_iterations 10 \
    --headless
```

### 方法 2: 直接運行（需要先啟動 Isaac Sim）

```bash
cd ~/IsaacLab
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/training_test.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 4 \
    --max_iterations 10 \
    --device cuda:0
```

## 參數說明

- `--task`: 環境任務名稱
  - `Isaac-Navigation-Charge-v0` - Phase 1 配置
  - `Isaac-Navigation-Charge-v1` - Phase 2 配置
  - `Isaac-Navigation-Charge-v2.5` - Phase 2.5 配置（自適應課程學習）
  
- `--num_envs`: 並行環境數量（測試用較少數量，建議 4-8）
- `--max_iterations`: 最大訓練迭代次數（測試用較少次數，建議 10-50）
- `--headless`: 無頭模式（不顯示 GUI，適合服務器環境）
- `--device`: 設備（`cuda:0` 或 `cpu`）

## 測試輸出

測試會輸出以下信息：

1. **環境創建測試**
   - 環境配置解析結果
   - 環境數量、設備信息
   - 動作空間和觀測空間信息

2. **環境重置測試**
   - 觀測類型和形狀
   - Info 字典的鍵

3. **訓練步驟測試**
   - 每 5 步的進度報告
   - 平均獎勵
   - 終止環境數
   - 成功和碰撞統計

4. **模組化組件檢查**
   - 觀測配置項數量
   - 獎勵配置項數量
   - 終止條件數量
   - 事件數量
   - 動作配置項數量

## 預期結果

如果模組化工作正確，應該看到：

```
✅ 環境創建成功
✅ 環境重置成功
✅ 訓練步驟測試完成
✅ 所有模組化組件檢查完成

總計: 4/4 通過
🎉 所有測試通過！模組化工作驗證成功！
```

## 故障排除

### 問題 1: 環境創建失敗

**錯誤**: `ImportError: cannot import name 'charge_mdp'`

**解決方案**: 
- 檢查 `cfg/charge_env_cfg.py` 中的導入路徑是否正確
- 確認 `charge_mdp.py` 文件存在於正確位置

### 問題 2: 觀測形狀不匹配

**錯誤**: `RuntimeError: shape mismatch`

**解決方案**:
- 檢查 `mdp/observations/` 中的觀測函數是否正確
- 確認觀測維度是否與配置一致

### 問題 3: 獎勵計算錯誤

**錯誤**: `RuntimeError: reward contains NaN`

**解決方案**:
- 檢查 `mdp/rewards/` 中的獎勵函數
- 確認獎勵計算邏輯是否正確

### 問題 4: 終止條件不工作

**錯誤**: 環境不會終止

**解決方案**:
- 檢查 `mdp/terminations/` 中的終止條件函數
- 確認終止條件配置是否正確

## 與完整訓練的區別

此測試腳本與完整訓練腳本的區別：

1. **環境數量**: 使用較少的環境數量（4-8 個）以加快測試速度
2. **迭代次數**: 只執行少量迭代（10-50 步）以驗證基本功能
3. **不保存模型**: 不保存訓練檢查點
4. **不記錄日誌**: 不記錄詳細的訓練日誌

## 下一步

測試通過後，可以使用完整的訓練腳本進行實際訓練：

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 128 \
    --headless
```

## 更新日期

2024-01 - 創建訓練測試腳本
