#!/bin/bash
# ==============================================================================
# NavRL Curriculum Training — VLP16 + Dense Rewards + 5-Stage Curriculum
# ==============================================================================
#
# 使用方法:
#   # 正式訓練 (6144 envs, headless)
#   ./scripts/train_curriculum_navrl.sh
#
#   # 小規模驗證 (6 envs)
#   ./scripts/train_curriculum_navrl.sh --num_envs 6
#
#   # GUI 模式 (調試)
#   ./scripts/train_curriculum_navrl.sh --gui --num_envs 6
#
#   # 自訂 timesteps
#   ./scripts/train_curriculum_navrl.sh --timesteps 500000
#
# ==============================================================================

TASK="Isaac-Navigation-Charge-VLP16-Curriculum-NavRL"
NUM_ENVS=6144
HEADLESS="--headless"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --gui)
            HEADLESS=""
            shift
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --timesteps)
            EXTRA_ARGS="$EXTRA_ARGS --timesteps $2"
            shift 2
            ;;
        --checkpoint)
            EXTRA_ARGS="$EXTRA_ARGS --checkpoint $2"
            shift 2
            ;;
        --seed)
            EXTRA_ARGS="$EXTRA_ARGS --seed $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gui] [--num_envs N] [--timesteps N] [--checkpoint PATH] [--seed N]"
            exit 1
            ;;
    esac
done

CMD="./isaaclab.sh -p scripts/reinforcement_learning/skrl/train_charge_ac.py"
CMD="$CMD --task $TASK"
CMD="$CMD --num_envs $NUM_ENVS"
[ -n "$HEADLESS" ] && CMD="$CMD $HEADLESS"
[ -n "$EXTRA_ARGS" ] && CMD="$CMD $EXTRA_ARGS"

echo "=============================================================================="
echo "NavRL Curriculum Training"
echo "=============================================================================="
echo "Task:       $TASK"
echo "Num Envs:   $NUM_ENVS"
echo "Headless:   $([ -z "$HEADLESS" ] && echo "No (GUI)" || echo "Yes")"
echo "Extra Args: ${EXTRA_ARGS:-none}"
echo "=============================================================================="
echo ""
echo "$CMD"
echo ""

eval $CMD
