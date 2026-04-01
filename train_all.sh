#!/bin/bash
# Train all models sequentially, one at a time.
# Each model logs to results/logs/<name>.log
# Usage: bash train_all.sh

PYTHON="/home/jiongx/micromamba/envs/poco/bin/python3 -u"
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

run ar              AR.py
run rnn             RNN.py
run lstm            LSTM.py
run nlinear         NLinear.py
run dlinear         DLinear.py
run tsmixer         TSMixer.py
run poco            POCO.py
run poco_prob       POCO_prob.py
run poco_highdrop   POCO_prob_highdrop.py
run poco_multi      POCO_multisession.py
run poco_prob_multi POCO_prob_multisession.py

echo "========================================"
echo "All training complete."
echo "Logs saved to results/logs/"
echo "========================================"
