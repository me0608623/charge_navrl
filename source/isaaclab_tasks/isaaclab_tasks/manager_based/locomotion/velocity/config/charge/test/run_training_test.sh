#!/bin/bash
# 模組化工作訓練測試快速啟動腳本

# 設置默認參數
TASK="${1:-Isaac-Navigation-Charge-v0}"
NUM_ENVS="${2:-4}"
MAX_ITERATIONS="${3:-10}"
HEADLESS="${4:-true}"

# 獲取腳本所在目錄
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_DIR="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"

echo "=========================================="
echo "模組化工作訓練測試"
echo "=========================================="
echo "任務: $TASK"
echo "環境數量: $NUM_ENVS"
echo "最大迭代次數: $MAX_ITERATIONS"
echo "無頭模式: $HEADLESS"
echo "=========================================="
echo ""

# 檢查 Isaac Lab 目錄是否存在
if [ ! -f "$ISAACLAB_DIR/isaaclab.sh" ]; then
    echo "❌ 錯誤: 找不到 isaaclab.sh"
    echo "   請確認 Isaac Lab 安裝路徑正確"
    exit 1
fi

# 運行測試
cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$SCRIPT_DIR/training_test.py" \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    --headless
