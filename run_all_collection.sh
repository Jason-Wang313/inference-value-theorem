#!/bin/bash
cd "C:/Users/wangz/Downloads/inference-value-theorem"
LOG="C:/Users/wangz/Downloads/inference-value-theorem/collection.log"

echo "=== Collection started at $(date) ===" > "$LOG"

for MODEL in 3B 8B 70B 405B; do
    echo "" >> "$LOG"
    echo "============================================================" >> "$LOG"
    echo "STARTING: $MODEL at $(date)" >> "$LOG"
    echo "============================================================" >> "$LOG"
    python experiments/01_collect.py --model "$MODEL" >> "$LOG" 2>&1
    RC=$?
    echo "FINISHED: $MODEL at $(date) (exit code: $RC)" >> "$LOG"
    if [ $RC -ne 0 ]; then
        echo "ERROR: $MODEL failed, continuing to next model" >> "$LOG"
    fi
done

echo "" >> "$LOG"
echo "=== ALL COLLECTIONS COMPLETE at $(date) ===" >> "$LOG"
