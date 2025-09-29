#!/bin/bash
# Quick start script for immediate testing

echo "🚀 Adaptive Federated Learning for Agricultural IoT - Quick Start"
echo "=============================================================="

# Check if virtual environment exists
if [ ! -d "agri_fl_env" ]; then
    echo "Setting up environment..."
    bash setup.sh
fi

# Activate environment
echo "Activating environment..."
source agri_fl_env/bin/activate

# Run quick test
echo "Running setup test..."
python test_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎯 Running quick demo experiment..."
    python main.py --experiment single --name "QuickDemo" --dataset fakedata --rounds 5 --clients 3

    echo ""
    echo "✅ Quick start completed!"
    echo ""
    echo "Results saved in: results/"
    echo "Next: jupyter notebook notebooks/analysis.ipynb"
else
    echo "❌ Setup test failed. Please check the installation."
fi
