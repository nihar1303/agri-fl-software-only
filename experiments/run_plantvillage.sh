#!/bin/bash
# Run with PlantVillage dataset (requires manual download)

echo "Running with PlantVillage Dataset"
python main.py \
    --experiment single \
    --name "PlantVillage_Combined" \
    --dataset plantvillage \
    --rounds 40 \
    --clients 10

echo "PlantVillage experiment completed!"
