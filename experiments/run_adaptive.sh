#!/bin/bash
# Adaptive participation experiment

echo "Running Adaptive Participation Experiment (E3)"
python main.py \
    --experiment single \
    --name "E3_Adaptive_Participation" \
    --dataset fakedata \
    --rounds 30 \
    --clients 8

echo "Adaptive participation experiment completed!"
