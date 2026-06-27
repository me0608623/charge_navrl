#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════
# setup_machine_b.sh — 第二台機 (PC-B) 一鍵設定 charge_skrl 訓練環境
#
# 用途：把本 repo 在另一台「路徑 / 版本 / 位置都相同」的電腦上設定好，
#       讓該機(或該機的 Claude Code)直接執行即可開始平行訓練。
#
# 前置 (PC-B 需已具備，因「環境都一樣」假設成立)：
#   - conda env: env_isaaclab (Python 3.11 + Isaac Sim 5.1 + CUDA 13.0)
#   - GitHub SSH key (能 git clone git@github.com:me0608623/...)
#   - 目標路徑 /home/aa/IsaacLab 可寫
#
# 首次取得本腳本 (repo 還沒 clone 時) — 在 PC-B 擇一：
#   A. 已有 repo：cd /home/aa/IsaacLab && git pull && bash scripts/setup_machine_b.sh
#   B. 全新機  ：把這支腳本 scp/貼到 PC-B 任一處再 `bash setup_machine_b.sh`
#               (腳本偵測到 repo 不存在會自動 clone)
#
# 用法：
#   bash setup_machine_b.sh             # clone(若缺)+ checkout + pull + 驗證
#   bash setup_machine_b.sh --smoke     # 額外跑 10-step 煙霧測試
#   bash setup_machine_b.sh --reinstall # 強制重跑 ./isaaclab.sh -i
# ════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ---- 可調參數 ----
REPO_URL="git@github.com:me0608623/charge_navrl.git"
BRANCH="wdclean-repro-20260429"
DEST="/home/aa/IsaacLab"
CONDA_ENV="env_isaaclab"

SMOKE=0; REINSTALL=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --reinstall) REINSTALL=1 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    *) echo "未知參數: $a (見 --help)"; exit 2 ;;
  esac
done

log()  { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }

# ---- 1. 取得 / 更新 repo ----
if [ -d "$DEST/.git" ]; then
  log "repo 已存在，更新中: $DEST"
  git -C "$DEST" fetch --all --quiet
  git -C "$DEST" checkout "$BRANCH"
  git -C "$DEST" pull --ff-only
else
  log "clone repo → $DEST"
  git clone "$REPO_URL" "$DEST"
  git -C "$DEST" checkout "$BRANCH"
fi
ok "repo @ $(git -C "$DEST" rev-parse --short HEAD) ($BRANCH)"

# ---- 2. conda env ----
log "啟用 conda env: $CONDA_ENV"
CONDA_BASE="$(conda info --base 2>/dev/null || /home/aa/miniconda3/bin/conda info --base 2>/dev/null || true)"
[ -n "$CONDA_BASE" ] && source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { warn "找不到 env '$CONDA_ENV' — 請先建好(版本須與 PC-A 相同)"; exit 3; }
ok "Python: $(python -V 2>&1)"

# ---- 3. editable 安裝 (僅在 import 失敗時，省去不必要的重裝) ----
cd "$DEST"
if [ "$REINSTALL" = "1" ] || ! python -c "import isaaclab_tasks" 2>/dev/null; then
  log "isaaclab_tasks 不可 import → 跑 ./isaaclab.sh -i"
  ./isaaclab.sh -i
else
  ok "isaaclab_tasks 已可 import (略過安裝)"
fi

# ---- 4. USD 驗證 (隨 repo 走，不需另搬) ----
USD="$DEST/assets/usd/charge/charge.usd"
if [ -f "$USD" ]; then
  ok "USD OK: $USD ($(du -h "$USD" | cut -f1))  ← charge_cfg.py 用 _REPO_ROOT 相對路徑自動定位"
else
  warn "USD 不在 $USD — 檢查 clone 是否完整 / .gitignore 是否誤擋"
fi

# ---- 4b. Claude Code 記憶同步 (PC-A 研究記憶 → 本機 ~/.claude) ----
MEM_SRC="$DEST/.claude-memory"
MEM_DST="$HOME/.claude/projects/-home-aa-IsaacLab/memory"
if [ -d "$MEM_SRC" ]; then
  mkdir -p "$MEM_DST"
  rsync -a "$MEM_SRC/" "$MEM_DST/"   # 不加 --delete：保留本機既有記憶，只新增/更新
  ok "Claude 記憶同步: $(ls "$MEM_DST"/*.md 2>/dev/null | wc -l) 個 .md → $MEM_DST"
else
  warn "repo 內無 .claude-memory/ (略過記憶同步)"
fi

# ---- 5. (可選) 煙霧測試：4 envs × 10 steps ----
if [ "$SMOKE" = "1" ]; then
  log "煙霧測試: wd_sa1_v3f_vaux, 4 envs × 10 steps"
  PYTHONUNBUFFERED=1 CHARGE_USE_ACT_HIST=0 ./isaaclab.sh -p \
    scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
    --experiment_config wd_sa1_v3f_vaux --headless --feat_norm --aux_epochs 8 \
    --num_envs 4 --timesteps 10 --run_name smoke_pcB \
    && ok "煙霧測試通過" || warn "煙霧測試失敗，看上面錯誤"
fi

# ---- 6. 平行訓練提醒 ----
cat <<'EOF'

────────────────────────────────────────────────
✅ 設定完成。開始平行訓練前請注意：
  • 這台機若要 commit 訓練改動，先開自己的 branch，避免和 PC-A 互蓋：
      git checkout -b wdclean-repro-20260429-pcB
  • push 前一定先 `git fetch`。
  • logs/ 和 wandb/ 各機獨立產生，不需互搬 (wandb 各自 sync 雲端)。
  • 正式訓練 (從頭訓 SA1 vaux)：
      PYTHONUNBUFFERED=1 CHARGE_USE_ACT_HIST=0 ./isaaclab.sh -p \
        scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
        --experiment_config wd_sa1_v3f_vaux --headless --feat_norm --aux_epochs 8 \
        --run_name sa1_v3f_vaux_ne1024_s42
────────────────────────────────────────────────
EOF
