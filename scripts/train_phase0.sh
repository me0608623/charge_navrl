#!/bin/bash
# ============================================================================
# Phase 訓練/播放腳本 (互動式選單)
# 支援 GUI/Headless 模式切換、環境數量配置、Checkpoint 載入
# ============================================================================

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ============================================================================
# 配置區
# ============================================================================

LOG_DIR="/home/aa/IsaacLab/logs/sb3"

# ============================================================================
# 顯示 Banner
# ============================================================================

print_banner() {
    clear
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                        ║"
    echo "║         ${BOLD}Isaac Lab Charge 導航訓練腳本${NC}                                   ║"
    echo "║         ${BOLD}122 維固定拓撲觀測系統${NC}                                         ║"
    echo "║                                                                        ║"
    echo "╚════════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# ============================================================================
# 顯示主選單
# ============================================================================

show_main_menu() {
    echo -e "${BOLD}${BLUE}選擇操作:${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC}. 訓練 (Train)"
    echo -e "  ${GREEN}2${NC}. 播放 (Play)"
    echo -e "  ${GREEN}3${NC}. 列出 Checkpoints"
    echo -e "  ${RED}0${NC}. 退出"
    echo ""
    echo -ne "${YELLOW}請選擇 [0-3]: ${NC}"
    read -r choice
    echo ""
}

# ============================================================================
# 選擇模式
# ============================================================================

select_mode() {
    echo -e "${BOLD}${BLUE}選擇顯示模式:${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC}. GUI 模式 (有視窗，適合測試)"
    echo -e "  ${GREEN}2${NC}. Headless 模式 (無視窗，適合訓練)"
    echo ""
    echo -ne "${YELLOW}請選擇 [1-2]: ${NC}"
    read -r choice
    case $choice in
        1) echo "gui" ;;
        2) echo "headless" ;;
        *) echo "gui" ;;
    esac
}

# ============================================================================
# 選擇環境數量
# ============================================================================

select_num_envs() {
    echo -e "${BOLD}${BLUE}選擇環境數量:${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC}. 5 個環境 (測試用)"
    echo -e "  ${GREEN}2${NC}. 10 個環境 (小規模)"
    echo -e "  ${GREEN}3${NC}. 50 個環境 (中等規模)"
    echo -e "  ${GREEN}4${NC}. 100 個環境 (較大規模)"
    echo -e "  ${GREEN}5${NC}. 256 個環境 (大規模訓練)"
    echo -e "  ${GREEN}6${NC}. 512 個環境 (超大规模)"
    echo -e "  ${GREEN}7${NC}. 自訂"
    echo ""
    echo -ne "${YELLOW}請選擇 [1-7]: ${NC}"
    read -r choice
    case $choice in
        1) echo "5" ;;
        2) echo "10" ;;
        3) echo "50" ;;
        4) echo "100" ;;
        5) echo "256" ;;
        6) echo "512" ;;
        7)
            echo -ne "${YELLOW}請輸入環境數量: ${NC}"
            read -r num
            echo "$num"
            ;;
        *) echo "5" ;;
    esac
}

# ============================================================================
# 選擇 Phase
# ============================================================================

select_phase() {
    echo -e "${BOLD}${BLUE}選擇 Phase:${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC}. Phase 0 (基礎運動學校準 - 空地)"
    echo -e "  ${GREEN}2${NC}. Phase 1 (靜態障礙物導航)"
    echo -e "  ${GREEN}3${NC}. Phase 2 (動態障礙物導航)"
    echo ""
    echo -ne "${YELLOW}請選擇 [1-3]: ${NC}"
    read -r choice
    case $choice in
        1) echo "0" ;;
        2) echo "1" ;;
        3) echo "2" ;;
        *) echo "0" ;;
    esac
}

# ============================================================================
# 選擇 Checkpoint
# ============================================================================

select_checkpoint() {
    local phase=$1
    local task_name="Isaac-Navigation-Charge-Phase${phase}"
    local log_dir="$LOG_DIR/$task_name"

    echo -e "${BOLD}${BLUE}選擇 Checkpoint (Phase ${phase}):${NC}"
    echo ""

    if [ ! -d "$log_dir" ]; then
        echo -e "${RED}找不到 $task_name 的 log 目錄${NC}"
        echo ""
        return 1
    fi

    # 尋找 checkpoints
    local checkpoints=($(find "$log_dir" -name "model_*.zip" -type f 2>/dev/null | sort))

    if [ ${#checkpoints[@]} -eq 0 ]; then
        echo -e "${RED}沒有找到 checkpoints${NC}"
        echo ""
        return 1
    fi

    # 顯示選項
    local idx=1
    for ckpt in "${checkpoints[@]}"; do
        local basename=$(basename "$ckpt")
        echo -e "  ${GREEN}${idx}${NC}. $basename"
        idx=$((idx + 1))
    done

    echo ""
    echo -ne "${YELLOW}請選擇 [1-${#checkpoints[@]}]，或按 Enter 跳過: ${NC}"
    read -r choice

    if [ -z "$choice" ]; then
        return 1
    fi

    local idx=$((choice - 1))
    if [ $idx -ge 0 ] && [ $idx -lt ${#checkpoints[@]} ]; then
        echo "${checkpoints[$idx]}"
        return 0
    fi

    return 1
}

# ============================================================================
# 選擇來源 Phase (用於載入 checkpoint)
# ============================================================================

select_source_phase() {
    local current_phase=$1

    echo -e "${BOLD}${BLUE}是否載入前一 Phase 的 Checkpoint?${NC}"
    echo ""
    echo -e "  ${GREEN}1${NC}. 是，載入 Phase $((current_phase - 1)) 的最佳模型"
    echo -e "  ${GREEN}2${NC}. 否，從頭訓練"
    echo ""
    echo -ne "${YELLOW}請選擇 [1-2]: ${NC}"
    read -r choice

    case $choice in
        1)
            # 嘗試獲取前一 phase 的最佳 checkpoint
            local prev_phase=$((current_phase - 1))
            local task_name="Isaac-Navigation-Charge-Phase${prev_phase}"
            local best="$LOG_DIR/$task_name/best_model.zip"

            if [ -f "$best" ]; then
                echo "$best"
                echo -e "${GREEN}✓ 找到前一 Phase 的最佳模型${NC}"
            else
                # 尋找最新的 model
                local latest=$(find "$LOG_DIR/$task_name" -name "model_*.zip" -type f 2>/dev/null | sort | tail -1)
                if [ -n "$latest" ]; then
                    echo "$latest"
                    echo -e "${GREEN}✓ 找到前一 Phase 的最新模型${NC}"
                else
                    echo -e "${YELLOW}⚠ 找不到前一 Phase 的 checkpoint，將從頭訓練${NC}"
                    echo ""
                fi
            fi
            ;;
        *)
            echo ""
            ;;
    esac
}

# ============================================================================
# 列出所有 Checkpoints
# ============================================================================

list_all_checkpoints() {
    clear
    print_banner
    echo -e "${BOLD}${BLUE}可用的 Checkpoints:${NC}"
    echo ""

    # 遍歷所有 Phase 目錄
    found=false
    for phase_dir in "$LOG_DIR"/Isaac-Navigation-Charge-Phase*; do
        if [ -d "$phase_dir" ]; then
            found=true
            phase_name=$(basename "$phase_dir")
            echo -e "${GREEN}$phase_name${NC}"

            # 尋找 best_model.zip
            if [ -f "$phase_dir/best_model.zip" ]; then
                echo -e "  ${CYAN}★ best_model.zip${NC}"
            fi

            # 列出所有 model
            find "$phase_dir" -name "model_*.zip" -type f 2>/dev/null | sort | while read f; do
                if [ "$(basename "$f")" != "best_model.zip" ]; then
                    echo "  - $(basename "$f")"
                fi
            done
            echo ""
        fi
    done

    if [ "$found" = false ]; then
        echo -e "${YELLOW}沒有找到任何 checkpoints${NC}"
        echo ""
    fi

    echo -ne "${YELLOW}按 Enter 返回主選單...${NC}"
    read -r
}

# ============================================================================
# 執行訓練/播放
# ============================================================================

run_training() {
    local mode=$1
    local num_envs=$2
    local phase=$3
    local action=$4
    local checkpoint=$5

    local task_name="Isaac-Navigation-Charge-Phase${phase}"

    # 建構命令
    cd /home/aa/IsaacLab

    if [ "$action" = "train" ]; then
        CMD="./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py"
    else
        # Play 模式使用 -Play 任務
        task_name="${task_name}-Play"
        CMD="./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py"
    fi

    # 添加參數（直接添加，不需要 -- 分隔符）
    CMD="$CMD --task $task_name"
    CMD="$CMD --num_envs $num_envs"

    # 添加 headless 參數
    if [ "$mode" = "headless" ]; then
        CMD="$CMD --headless"
    fi

    # 添加 checkpoint (如果指定)
    if [ -n "$checkpoint" ]; then
        CMD="$CMD --checkpoint $checkpoint"
    fi

    # 顯示配置
    clear
    print_banner
    echo -e "${BOLD}${BLUE}配置確認:${NC}"
    echo ""
    echo -e "  模式:       ${GREEN}$mode${NC}"
    echo -e "  動作:       ${GREEN}$action${NC}"
    echo -e "  任務:       ${GREEN}$task_name${NC}"
    echo -e "  環境數量:   ${GREEN}$num_envs${NC}"
    if [ -n "$checkpoint" ]; then
        echo -e "  Checkpoint: ${GREEN}$checkpoint${NC}"
    fi
    echo ""

    echo -e "${YELLOW}執行命令:${NC}"
    echo -e "${CYAN}$CMD${NC}"
    echo ""
    echo -e "${YELLOW}按 Ctrl+C 停止訓練...${NC}"
    echo ""
    echo -ne "${YELLOW}按 Enter 開始執行...${NC}"
    read -r

    # 執行
    eval $CMD
}

# ============================================================================
# 訓練流程
# ============================================================================

training_flow() {
    clear
    print_banner
    echo -e "${BOLD}${MAGENTA}>>> 訓練模式 <<<${NC}"
    echo ""

    # 選擇 Phase
    local phase=$(select_phase)
    echo ""

    # 選擇模式
    local mode=$(select_mode)
    echo ""

    # 選擇環境數量
    local num_envs=$(select_num_envs)
    echo ""

    # 詢問是否載入 checkpoint
    local checkpoint=""
    if [ "$phase" -gt 0 ]; then
        checkpoint=$(select_source_phase $phase)
    else
        echo -e "${YELLOW}提示: Phase 0 從頭訓練${NC}"
        echo ""
    fi

    # 執行
    run_training "$mode" "$num_envs" "$phase" "train" "$checkpoint"
}

# ============================================================================
# 播放流程
# ============================================================================

play_flow() {
    clear
    print_banner
    echo -e "${BOLD}${MAGENTA}>>> 播放模式 <<<${NC}"
    echo ""

    # 選擇 Phase
    local phase=$(select_phase)
    echo ""

    # 選擇模式
    local mode=$(select_mode)
    echo ""

    # 選擇環境數量
    local num_envs=$(select_num_envs)
    echo ""

    # 選擇 checkpoint
    local checkpoint=""
    if select_checkpoint "$phase"; then
        checkpoint=$(select_checkpoint "$phase")
    fi

    # 如果沒有選擇，嘗試使用 best_model
    if [ -z "$checkpoint" ]; then
        local task_name="Isaac-Navigation-Charge-Phase${phase}"
        local best="$LOG_DIR/$task_name/best_model.zip"
        if [ -f "$best" ]; then
            checkpoint="$best"
            echo -e "${GREEN}✓ 使用最佳模型: $checkpoint${NC}"
            echo ""
        else
            echo -e "${RED}✗ 沒有找到 checkpoint，請先訓練模型${NC}"
            echo ""
            echo -ne "${YELLOW}按 Enter 返回...${NC}"
            read -r
            return
        fi
    fi

    # 執行
    run_training "$mode" "$num_envs" "$phase" "play" "$checkpoint"
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    while true; do
        print_banner
        show_main_menu

        case $choice in
            1)
                training_flow
                ;;
            2)
                play_flow
                ;;
            3)
                list_all_checkpoints
                ;;
            0)
                echo -e "${GREEN}再見！${NC}"
                exit 0
                ;;
            *)
                echo -e "${RED}無效選擇，請重試${NC}"
                sleep 1
                ;;
        esac
    done
}

# 執行主程序
main
