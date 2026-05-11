#!/bin/bash
# ==============================================================================
# Phase 0 突破 13% 瓶頸訓練腳本
# ==============================================================================
# 🚀 使用三管齊下策略突破成功率瓶頸：
#
# 行動 A：更高學習率（1e-4 → 3e-4）
# 行動 B：增加到達獎勵（reaching_goal: 8.0 → 30.0）
# 行動 C：降低動作隨機性（log_std_init: -1.0）
#
# 預期效果：
#   - Success Rate: 13% → 50%+
#   - Entropy Loss: 開始下降
#   - Episode Reward: 穩定上升
# ==============================================================================

# 訓練參數
NUM_ENVS=256
HEADLESS="--headless"

# 🚀 使用新的突破配置環境
TASK="Isaac-Navigation-Charge-Phase0-Breakthrough"

echo "======================================"
echo "🚀 Phase 0 突破瓶頸訓練"
echo "======================================"
echo "配置：sb3_ppo_cfg_breakthrough.yaml"
echo "環境：$TASK"
echo "環境數：$NUM_ENVS"
echo ""
echo "三管齊下策略："
echo "  • 學習率：3e-4（從 1e-4 提升）"
echo "  • 到達獎勵：30.0（從 8.0 提升）"
echo "  • 動作 std：exp(-1.0) ≈ 0.37"
echo "======================================"
echo ""

./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task "$TASK" \
    --num_envs $NUM_ENVS \
    $HEADLESS

echo ""
echo "======================================"
echo "訓練完成！"
echo "======================================"
echo "查看 WandB："
echo "  https://wandb.ai/your-username/charge_sb3"
echo ""
echo "關鍵監控指標："
echo "  • success_rate > 0.5 (目標)"
echo "  • train/ep_rew_mean (應上升)"
echo "  • train/policy_entropy (應下降)"
echo "======================================"
