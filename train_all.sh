#!/bin/bash
# Train POCO (deterministic) and POCO_prob (probabilistic) sequentially.
# Each model logs to results/logs/<name>.log
# Usage: bash train_all.sh

PYTHON="python3 -u"
mkdir -p results/logs

run() {
    local name=$1
    local script=$2
    echo "========================================"
    echo "Training: $name"
    echo "Log:      results/logs/${name}.log"
    echo "========================================"
    $PYTHON "$script" 2>&1 | tee "results/logs/${name}.log"
    echo "Done: $name"
    echo ""
}

run poco      POCO.py
run poco_prob poco_src/POCO_prob.py

echo "========================================"
echo "All training complete."
echo "Logs saved to results/logs/"
echo "========================================"
