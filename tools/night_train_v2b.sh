#!/usr/bin/env bash
# Storage-stable launcher: survives tmux death; heartbeats to file.
set -uo pipefail
cd /home/harshil/kaggle/rsna-knee
mkdir -p logs runs/v2b
S=logs/TRAIN_STATUS
echo "$$ $(date '+%H:%M:%S') v2b-launch" > logs/v2b.pid

echo "RUNNING train fold0 v2b $(date '+%H:%M:%S')" > "$S"
( setsid nohup python3 -m src.train --fold 0 --epochs 3 --no-gold-only \
  --labels /mnt/ssd/rsna-knee/data/train_weak_v2_img.csv > logs/train_fold0_v2b.log 2>&1 \
  && echo "train-rc-ok" || echo "train-rc-FAIL" ) > logs/v2b_train_rc.txt &
TPID=$!
echo "train-pid $TPID" >> logs/v2b.pid
wait $TPID

grep -q "train-rc-ok" logs/v2b_train_rc.txt || { echo "FAIL train v2b $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING infer v2b $(date '+%H:%M:%S')" > "$S"
python3 -m src.infer --ckpt checkpoints/best.pt --out runs/v2b/preds.csv > logs/infer_v2b.log 2>&1 \
  || { echo "FAIL infer v2b $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING score v2b $(date '+%H:%M:%S')" > "$S"
python3 score_gold.py --train /mnt/ssd/rsna-knee/data/train.csv --pred runs/v2b/preds.csv > logs/score_v2b.log 2>&1 \
  || { echo "FAIL score v2b $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "DONE v2b $(date '+%H:%M:%S')" > "$S"