#!/bin/bash
# Train all models sequentially, one at a time.
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

run ar              baseline_models/AR.py
run rnn             baseline_models/RNN.py
run lstm            baseline_models/LSTM.py
run nlinear         baseline_models/NLinear.py
run dlinear         baseline_models/DLinear.py
run tsmixer         baseline_models/TSMixer.py
run poco            POCO.py
run poco_prob       poco_src/POCO_prob.py
run poco_highdrop   poco_src/POCO_prob_highdrop.py
run poco_multi      poco_src/POCO_multisession.py
run poco_prob_multi poco_src/POCO_prob_multisession.py

echo "========================================"
echo "All training complete."
echo "Logs saved to results/logs/"
echo "========================================"
