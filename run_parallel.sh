#!/bin/bash
cd "C:/Users/wangz/Downloads/inference-value-theorem"
LOG_DIR="C:/Users/wangz/Downloads/inference-value-theorem/logs"
mkdir -p "$LOG_DIR"

echo "=== Parallel collection started at $(date) ===" | tee "$LOG_DIR/master.log"
echo "Models: 3B, 8B, 70B | Samples: 32 | Problems: 500" | tee -a "$LOG_DIR/master.log"

# Launch all 3 models simultaneously
python experiments/01_collect.py --model 3B --n_samples 32 > "$LOG_DIR/3B.log" 2>&1 &
PID_3B=$!

python experiments/01_collect.py --model 8B --n_samples 32 > "$LOG_DIR/8B.log" 2>&1 &
PID_8B=$!

python experiments/01_collect.py --model 70B --n_samples 32 > "$LOG_DIR/70B.log" 2>&1 &
PID_70B=$!

echo "PIDs: 3B=$PID_3B 8B=$PID_8B 70B=$PID_70B" | tee -a "$LOG_DIR/master.log"

# Wait for all to finish
wait $PID_3B
echo "3B finished at $(date) (exit: $?)" | tee -a "$LOG_DIR/master.log"

wait $PID_8B
echo "8B finished at $(date) (exit: $?)" | tee -a "$LOG_DIR/master.log"

wait $PID_70B
echo "70B finished at $(date) (exit: $?)" | tee -a "$LOG_DIR/master.log"

echo "=== ALL COMPLETE at $(date) ===" | tee -a "$LOG_DIR/master.log"
