#!/bin/bash
# Run full experiment suite

echo "Running Full Experiment Suite"
python main.py \
    --experiment suite \
    --dataset fakedata \
    --rounds 25 \
    --clients 8

echo "Full experiment suite completed!"
