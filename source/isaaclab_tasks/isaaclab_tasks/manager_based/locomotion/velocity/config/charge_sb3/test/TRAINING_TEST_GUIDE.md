# 模組化工作訓練測試指南

## 快速開始

### 方法 1: 使用快速啟動腳本（最簡單）

```bash
cd ~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test
./run_training_test.sh
```

或者指定參數：

```bash
./run_training_test.sh Isaac-Navigation-Charge-v0 4 10 true
```

參數說明：
1. 任務名稱（默認: `Isaac-Navigation-Charge-v0`）
2. 環境數量（默認: `4`）
3. 最大迭代次數（默認: `10`）
4. 無頭模式（默認: `true`）

### 方法 2: 使用 Isaac Lab 啟動器

```bash
cd ~/IsaacLab
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/training_test.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 4 \
    --max_iterations 10 \
    --headless
```

### 方法 3: 直接運行（需要先啟動 Isaac Sim）

```bash
cd ~/IsaacLab
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge/test/training_test.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 4 \
    --max_iterations 10 \
    --device cuda:0
```

## 可用的測試任務

- `Isaac-Navigation-Charge-v0` - Phase 1 配置（3個靜態障礙物）
- `Isaac-Navigation-Charge-v1` - Phase 2 配置（5個靜態障礙物）
- `Isaac-Navigation-Charge-v2.5` - Phase 2.5 配置（自適應課程學習）

## 測試內容

測試腳本會執行以下測試：

1. **環境創建測試**
   - 驗證環境配置解析
   - 驗證環境實例創建
   - 檢查動作空間和觀測空間

2. **環境重置測試**
   - 驗證環境重置功能
   - 檢查觀測格式和形狀
   - 檢查 info 字典

3. **訓練步驟測試**
   - 執行多個訓練步驟
   - 驗證環境動態
   - 統計獎勵、成功、碰撞等信息

4. **模組化組件檢查**
   - 檢查觀測配置
   - 檢查獎勵配置
   - 檢查終止條件配置
   - 檢查事件配置
   - 檢查動作配置

## 預期輸出

成功的測試應該顯示：

```
======================================================================
模組化工作訓練測試
======================================================================
任務: Isaac-Navigation-Charge-v0
環境數量: 4
最大迭代次數: 10
設備: cuda:0
======================================================================

======================================================================
測試 1: 環境創建測試 - Isaac-Navigation-Charge-v0
======================================================================
✅ 環境配置解析成功
   - 環境數量: 4
   - 設備: cuda:0
✅ 環境創建成功
   - 動作空間: Box(-1.0, 1.0, (2,), float32)
   - 觀測空間: Box(-inf, inf, (131,), float32)

======================================================================
測試 2: 環境重置測試
======================================================================
✅ 環境重置成功
   - 觀測類型: Tensor
     - shape: (4, 131)
     - dtype: torch.float32
   - Info 鍵: ['time_outs', 'episode_extra_stats']

======================================================================
測試 3: 訓練步驟測試（10 步）
======================================================================
   ✅ 步驟 5/10 完成
      - 平均獎勵: 0.1234
      - 終止環境數: 0
   ✅ 步驟 10/10 完成
      - 平均獎勵: 0.2345
      - 終止環境數: 1

✅ 訓練步驟測試完成
   - 總步數: 10
   - 平均總獎勵: 2.3450
   - 成功次數: 0
   - 碰撞次數: 1
   - 平均 episode 長度: 10.00

======================================================================
測試 4: 模組化組件檢查
======================================================================
✅ 觀測配置存在
   - 觀測項數量: 8
✅ 獎勵配置存在
   - 獎勵項數量: 8
✅ 終止條件配置存在
   - 終止條件數量: 5
✅ 事件配置存在
   - 事件數量: 2
✅ 動作配置存在
   - 動作項數量: 1

✅ 所有模組化組件檢查完成

======================================================================
測試總結
======================================================================
環境創建: ✅ 通過
環境重置: ✅ 通過
訓練步驟: ✅ 通過
模組化組件: ✅ 通過
----------------------------------------------------------------------
總計: 4/4 通過
======================================================================

🎉 所有測試通過！模組化工作驗證成功！
```

## 故障排除

### 問題 1: 找不到模組

**錯誤**: `ModuleNotFoundError: No module named 'charge.mdp'`

**解決方案**:
- 確認 `mdp/` 目錄存在且包含 `__init__.py`
- 檢查 `cfg/charge_env_cfg.py` 中的導入路徑

### 問題 2: 環境創建失敗

**錯誤**: `ImportError: cannot import name 'ChargeNavigationEnvCfg'`

**解決方案**:
- 確認 `cfg/__init__.py` 正確導出所有配置類
- 檢查 `__init__.py` 中的環境註冊路徑

### 問題 3: CUDA 錯誤

**錯誤**: `RuntimeError: CUDA out of memory`

**解決方案**:
- 減少環境數量：`--num_envs 2`
- 使用 CPU：`--device cpu`

### 問題 4: 觀測形狀不匹配

**錯誤**: `RuntimeError: shape mismatch`

**解決方案**:
- 檢查 `mdp/observations/functions.py` 中的觀測函數
- 確認觀測維度與配置一致

## 測試參數建議

### 快速測試（驗證基本功能）
```bash
--num_envs 2 --max_iterations 5
```

### 標準測試（推薦）
```bash
--num_envs 4 --max_iterations 10
```

### 完整測試（驗證穩定性）
```bash
--num_envs 8 --max_iterations 50
```

## 下一步

測試通過後，可以進行完整訓練：

```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 128 \
    --headless
```

## 相關文檔

- [README_TRAINING_TEST.md](README_TRAINING_TEST.md) - 詳細測試說明
- [modularization_verification_test.py](modularization_verification_test.py) - 模組化驗證測試
- [MODULARIZATION_TEST_SUMMARY.md](MODULARIZATION_TEST_SUMMARY.md) - 模組化測試總結

## 更新日期

2024-01 - 創建訓練測試指南
