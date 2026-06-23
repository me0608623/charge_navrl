#!/bin/bash
# SA4 chain: wait for SA4 → continue 500 iter → run ent_coeff ×1.25 (1400 iter)
set -e

SA4_PID=1206361
SA4_CKPT="logs/rnn_car/sa4_a2c_aux_extractor_ne1024_s42/checkpoint_270000.pt"
SA3_CKPT="logs/rnn_car/sa3_a2c_aux_extractor_ne1024_s42_lowent/checkpoint_270000.pt"
cd /home/aa/IsaacLab

echo "[$(date '+%H:%M:%S')] Waiting for SA4 (PID $SA4_PID) to finish..."
while kill -0 $SA4_PID 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%H:%M:%S')] SA4 finished."

# Verify checkpoint exists
if [ ! -f "$SA4_CKPT" ]; then
    echo "[ERROR] SA4 checkpoint not found: $SA4_CKPT"
    echo "Available checkpoints:"
    ls logs/rnn_car/sa4_a2c_aux_extractor_ne1024_s42/checkpoint_*.pt 2>/dev/null
    exit 1
fi

# --- Step 2: Continue SA4 for 500 more iters ---
echo "[$(date '+%H:%M:%S')] Starting SA4 continuation (+500 iter)..."
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
    --experiment_config wd_sa4_a2c_aux \
    --checkpoint "$SA4_CKPT" \
    --timesteps 150000 \
    --headless \
    --run_name sa4_a2c_aux_extractor_ne1024_s42_cont500 \
    > /tmp/sa4_cont500_train.log 2>&1

echo "[$(date '+%H:%M:%S')] SA4 continuation finished."

# --- Step 3: ent_coeff ×1.25 version (1400 iter from SA3 checkpoint) ---
echo "[$(date '+%H:%M:%S')] Starting SA4 ent_coeff ×1.25 (1400 iter)..."
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train/train_rnn_car_wdclip.py \
    --experiment_config wd_sa4_a2c_aux \
    --checkpoint "$SA3_CKPT" \
    --timesteps 420000 \
    --ent_coeff_linear 0.00625 \
    --ent_coeff_angular 0.0125 \
    --headless \
    --run_name sa4_a2c_aux_ent125_ne1024_s42 \
    > /tmp/sa4_ent125_train.log 2>&1

echo "[$(date '+%H:%M:%S')] SA4 ent_coeff ×1.25 finished. All done."
