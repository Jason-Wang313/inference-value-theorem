#!/bin/bash
cd "C:/Users/wangz/Downloads/inference-value-theorem"
LOG_DIR="C:/Users/wangz/Downloads/inference-value-theorem/logs"
mkdir -p "$LOG_DIR"

echo "=== TURBO collection: n=16 batch mode ===" | tee "$LOG_DIR/master.log"
echo "500 problems × 16 samples × 3 models = 1500 API calls" | tee -a "$LOG_DIR/master.log"
echo "Started at $(date)" | tee -a "$LOG_DIR/master.log"

# All 3 models in parallel — only 500 calls each
python experiments/01_collect.py --model 3B --n_samples 16 > "$LOG_DIR/3B.log" 2>&1 &
PID_3B=$!
python experiments/01_collect.py --model 8B --n_samples 16 > "$LOG_DIR/8B.log" 2>&1 &
PID_8B=$!
python experiments/01_collect.py --model 70B --n_samples 16 > "$LOG_DIR/70B.log" 2>&1 &
PID_70B=$!

echo "PIDs: 3B=$PID_3B 8B=$PID_8B 70B=$PID_70B" | tee -a "$LOG_DIR/master.log"

wait $PID_3B; echo "3B done at $(date) (exit: $?)" | tee -a "$LOG_DIR/master.log"
wait $PID_8B; echo "8B done at $(date) (exit: $?)" | tee -a "$LOG_DIR/master.log"
wait $PID_70B; echo "70B done at $(date) (exit: $?)" | tee -a "$LOG_DIR/master.log"

echo "=== ALL COMPLETE at $(date) ===" | tee -a "$LOG_DIR/master.log"
