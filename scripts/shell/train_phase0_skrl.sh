#!/bin/bash
# ==============================================================================
# SKRL Training Script for Charge Navigation - Phase 0
# ==============================================================================
#
# 使用 SKRL 訓練 Phase 0 (車輛動力學校準)
#
# 環境規格:
#   - 觀測空間: 131 維
#   - 動作空間: Box(-1, 1, (2,)) - [前進速度, 旋轉速度]
#
# 使用方法:
#   # 訓練 (headless 模式)
#   ./scripts/train_phase0_skrl.sh
#
#   # 訓練 (GUI 模式 - 用於調試)
#   ./scripts/train_phase0_skrl.sh --gui
#
#   # 自定義環境數量
#   ./scripts/train_phase0_skrl.sh --num_envs 512
#
# ==============================================================================

# 默認參數
TASK="Isaac-Navigation-Charge-Phase0"
NUM_ENVS=256
HEADLESS="--headless"
DEVICE="cuda"
VIDEO=""
NUM_ENVS_ARG=""

# 解析命令行參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --gui)
            HEADLESS=""
            shift
            ;;
        --num_envs)
            NUM_ENVS="$2"
            NUM_ENVS_ARG="--num_envs $2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --video)
            VIDEO="--video"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gui] [--num_envs N] [--task NAME] [--device DEVICE] [--video]"
            exit 1
            ;;
    esac
done

# 構建命令
CMD="./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge.py"
CMD="$CMD --task $TASK"
CMD="$CMD --agent skrl_cfg_entry_point"
CMD="$CMD --ml_framework torch"

if [ -n "$NUM_ENVS_ARG" ]; then
    CMD="$CMD $NUM_ENVS_ARG"
fi

if [ -n "$HEADLESS" ]; then
    CMD="$CMD $HEADLESS"
fi

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$VIDEO" ]; then
    CMD="$CMD $VIDEO"
fi

# 打印信息
echo "=============================================================================="
echo "SKRL Training - Charge Navigation Phase 0"
echo "=============================================================================="
echo "Task:           $TASK"
echo "Num Envs:       $NUM_ENVS"
echo "Device:         $DEVICE"
echo "Headless:       $([ -z "$HEADLESS" ] && echo "No (GUI mode)" || echo "Yes")"
echo "Video:          $([ -z "$VIDEO" ] && echo "No" || echo "Yes")"
echo "=============================================================================="
echo ""
echo "Running command:"
echo "$CMD"
echo ""
echo "=============================================================================="
echo ""

# 執行訓練
eval $CMD

# 訓練完成
echo ""
echo "=============================================================================="
echo "Training completed!"
echo "Logs saved in: logs/skrl/$TASK/"
echo "=============================================================================="
