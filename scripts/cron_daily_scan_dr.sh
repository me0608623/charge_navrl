#!/bin/bash
# 每日訓練掃描 + 開發日記 (cron job)
# Taipei 00:00 = UTC 16:00
# crontab: 0 16 * * * /home/aa/IsaacLab/scripts/cron_daily_scan_dr.sh

set -e

DATE=$(date +%Y%m%d)
LOG_DIR="/home/aa/Documents/dr/cron-logs"
REPO_DIR="/home/aa/IsaacLab"
CLAUDE="/home/aa/.npm-global/bin/claude"

# 確保 conda 環境可用
source /home/aa/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

cd "$REPO_DIR"

echo "=== Daily Scan+DR started: $(date) ===" >> "$LOG_DIR/cron-$DATE.log"

# Step 1: /scan
echo "[$(date)] Running /scan ..." >> "$LOG_DIR/cron-$DATE.log"
$CLAUDE -p "請執行 /scan skill，分析所有訓練進度" \
  --allowedTools "Bash,Read,Glob,Grep" \
  --max-turns 30 \
  --output-format text \
  >> "$LOG_DIR/scan-$DATE.log" 2>&1 || true

echo "[$(date)] /scan completed" >> "$LOG_DIR/cron-$DATE.log"

# Step 2: /dr
echo "[$(date)] Running /dr ..." >> "$LOG_DIR/cron-$DATE.log"
$CLAUDE -p "請執行 /dr skill，產生今天的開發日記" \
  --allowedTools "Bash,Read,Write,Glob,Grep" \
  --max-turns 30 \
  --output-format text \
  >> "$LOG_DIR/dr-$DATE.log" 2>&1 || true

echo "[$(date)] /dr completed" >> "$LOG_DIR/cron-$DATE.log"

# Step 3: /note (自動偵測最新 run，同步至 Obsidian)
echo "[$(date)] Running /note ..." >> "$LOG_DIR/cron-$DATE.log"
$CLAUDE -p "請執行 /note skill，將最新訓練進度同步至 Obsidian 筆記庫" \
  --allowedTools "Bash,Read,Write,Glob,Grep" \
  --max-turns 40 \
  --output-format text \
  >> "$LOG_DIR/note-$DATE.log" 2>&1 || true

echo "[$(date)] /note completed" >> "$LOG_DIR/cron-$DATE.log"
echo "=== Daily Scan+DR+Note finished: $(date) ===" >> "$LOG_DIR/cron-$DATE.log"
