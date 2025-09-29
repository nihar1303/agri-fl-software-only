#!/bin/bash
# Clustered personalization experiment

echo "Running Clustered Personalization Experiment (E4)"
python main.py \
    --experiment single \
    --name "E4_Clustered_Personalization" \
    --dataset fakedata \
    --rounds 30 \
    --clients 8

echo "Clustered personalization experiment completed!"
