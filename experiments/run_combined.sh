#!/bin/bash
# Combined adaptive + clustering experiment

echo "Running Combined Approach Experiment (E5)"
python main.py \
    --experiment single \
    --name "E5_Combined_Approach" \
    --dataset fakedata \
    --rounds 30 \
    --clients 8

echo "Combined approach experiment completed!"
