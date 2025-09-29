#!/bin/bash
# Baseline FedAvg experiment on IID data

echo "Running Baseline FedAvg Experiment (E0)"
python main.py \
    --experiment single \
    --name "E0_Baseline_IID" \
    --dataset fakedata \
    --rounds 30 \
    --clients 8

echo "Baseline experiment completed!"
