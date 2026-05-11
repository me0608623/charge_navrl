# 測試 AIT* 路徑視覺化

## 快速測試

### 方法 1：使用層級式導航環境

```bash
./isaaclab.sh -p scripts/demos/test_hierarchical_env.py
```

這將啟動層級式導航環境，你應該能看到：
- **綠色小球** - AIT* 規劃的路徑點
- **綠色線條** - 連接路徑點的線
- **綠色大球** - 機器人起點位置
- **紅色大球** - 目標位置

### 方法 2：使用完整的視覺化測試

```bash
./isaaclab.sh -p scripts/demos/test_aitstar_visualization.py --test hierarchical
```

### 方法 3：在訓練中使用

```bash
python scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-Hierarchical-v0 \
    --num_envs 5
```

（移除 --headless 參數以查看視覺化）

## 預期結果

當環境啟動後，你應該看到：

1. **綠色小球和線條** - AIT* 規劃的路徑
2. 綠色大球標示機器人位置
3. 紅色大球標示目標位置

## 故障排除

### 如果看不到路徑：

1. **檢查控制台輸出**
   - 應該看到 `[HierarchicalChargeNavigationEnv] AIT* 規劃器和視覺化器已初始化`
   - 應該看到 `[AIT*] 初始規劃成功: N 個路徑點`

2. **檢查相機角度**
   - 有時路徑在相機視角之外，嘗試旋轉視角

3. **檢查環境**
   - 確認使用的是 `Isaac-Navigation-Charge-Hierarchical-v0` 而不是 `v0`

4. **降低環境數量**
   - 使用 `--num_envs 1` 來簡化調試

## 調試輸出

環境會在控制台輸出調試信息：

```
[HierarchicalChargeNavigationEnv] AIT* 規劃器和視覺化器已初始化
[AIT*] Reset後初始規劃成功: 20 個路徑點
步驟 0: 獎勵 = -0.123
步驟 25: 獎勵 = 0.456
...
```

## 視覺化元素說明

| 元素 | 顏色 | 尺寸 | 說明 |
|------|------|------|------|
| 路徑點 | 綠色 | r=0.1m | AIT* 規劃的路徑點 |
| 連接線 | 綠色 | r=0.05m | 連接相鄰路徑點 |
| 起點 | 綠色 | r=0.15m | 機器人當前位置 |
| 終點 | 紅色 | r=0.15m | 目標位置 |
