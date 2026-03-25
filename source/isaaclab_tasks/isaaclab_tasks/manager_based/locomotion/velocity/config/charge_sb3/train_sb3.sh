#!/bin/bash
# ==============================================================================
# Stable Baselines 3 PPO 訓練腳本 - Charge 導航任務
# ==============================================================================
# 使用方法：
#   ./train_sb3.sh <phase> [options]
#
# 參數說明：
#   <phase>: 訓練階段（v0, v1, v2, v3）
#   options:
#     --num-envs <num>: 並行環境數量（默認：128）
#     --seed <num>: 隨機種子（默認：42）
#     --headless: 無頭模式（默認：啟用）
#     --video: 啟用錄像
#     --resume: 從檢查點恢復
#
# 示例：
#   ./train_sb3.sh v1                    # Phase 1 訓練
#   ./train_sb3.sh v2 --num-envs 64     # Phase 2 訓練（64 環境）
#   ./train_sb3.sh v3 --seed 123        # Phase 3 訓練（不同種子）
# ==============================================================================

set -e  # 遇到錯誤立即退出

# 預設參數
PHASE=""
NUM_ENVS=128
SEED=42
HEADLESS="--headless"
VIDEO=""
RESUME=""
CHECKPOINT=""

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    case $1 in
        v0|v1|v2|v3)
            PHASE="$1"
            shift
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --no-headless)
            HEADLESS=""
            shift
            ;;
        --video)
            VIDEO="--video"
            shift
            ;;
        --resume)
            RESUME="--checkpoint"
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            RESUME="--checkpoint"
            shift 2
            ;;
        *)
            echo "未知參數：$1"
            echo "使用方法：$0 <phase> [options]"
            exit 1
            ;;
    esac
done

# 檢查必要參數
if [ -z "$PHASE" ]; then
    echo "錯誤：必須指定訓練階段（v0, v1, v2, v3）"
    echo ""
    echo "使用方法：$0 <phase> [options]"
    echo ""
    echo "階段說明："
    echo "  v0 - 無障礙物（基礎導航）"
    echo "  v1 - Phase 1（3 個障礙物）"
    echo "  v2 - Phase 2（5 個障礙物）"
    echo "  v3 - Phase 3（目標距離課程）"
    echo ""
    echo "選項："
    echo "  --num-envs <num>   並行環境數量（默認：128）"
    echo "  --seed <num>       隨機種子（默認：42）"
    echo "  --no-headless       顯示 GUI（默認：無頭模式）"
    echo "  --video            啟用錄像"
    echo "  --resume           從最新檢查點恢復"
    echo "  --checkpoint <path> 從指定檢查點恢復"
    exit 1
fi

# 構建環境 ID
ENV_ID="Isaac-Navigation-Charge-SB3-${PHASE}"
AGENT="sb3_cfg_entry_point"

# 打印配置信息
echo "========================================"
echo "  Stable Baselines 3 PPO 訓練配置"
echo "========================================"
echo "環境 ID: $ENV_ID"
echo "並行環境數量: $NUM_ENVS"
echo "隨機種子: $SEED"
echo "無頭模式: ${HEADLESS:-False}"
echo "錄像: ${VIDEO:-False}"
echo "恢復訓練: ${RESUME:-False}"
if [ -n "$CHECKPOINT" ]; then
    echo "檢查點: $CHECKPOINT"
fi
echo "========================================"
echo ""

# 構建訓練命令
TRAIN_CMD="./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py"
TRAIN_CMD="$TRAIN_CMD --task $ENV_ID"
TRAIN_CMD="$TRAIN_CMD --num_envs $NUM_ENVS"
TRAIN_CMD="$TRAIN_CMD --seed $SEED"
TRAIN_CMD="$TRAIN_CMD --agent $AGENT"
if [ -n "$HEADLESS" ]; then
    TRAIN_CMD="$TRAIN_CMD $HEADLESS"
fi
if [ -n "$VIDEO" ]; then
    TRAIN_CMD="$TRAIN_CMD $VIDEO"
fi
if [ -n "$RESUME" ] && [ -n "$CHECKPOINT" ]; then
    TRAIN_CMD="$TRAIN_CMD $RESUME $CHECKPOINT"
fi

# 執行訓練
echo "執行命令："
echo "$TRAIN_CMD"
echo ""
echo "開始訓練..."
echo ""

eval $TRAIN_CMD

echo ""
echo "========================================"
echo "  訓練完成！"
echo "========================================"
echo "查看 TensorBoard："
echo "  tensorboard --logdir logs/sb3/$ENV_ID/"
echo ""
