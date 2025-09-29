#!/bin/bash
# Setup script for Adaptive Federated Learning for Agricultural IoT

echo "🌾 Setting up Adaptive Federated Learning for Agricultural IoT"
echo "============================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\.[0-9]' | head -1)
echo "Python version: $python_version"

if (( $(echo "$python_version < 3.8" | bc -l) )); then
    echo "❌ Error: Python 3.8+ required"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv agri_fl_env
source agri_fl_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p results
mkdir -p data/plantvillage
mkdir -p logs

# Run quick test
echo "🧪 Running quick test..."
python -c "
import torch
import flwr as fl
import numpy as np
import sklearn
print('✅ All major dependencies installed successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'Flower version: {fl.__version__}')
print(f'NumPy version: {np.__version__}')
print(f'Scikit-learn version: {sklearn.__version__}')
"

echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source agri_fl_env/bin/activate"
echo "2. Run quick test: python test_setup.py"
echo "3. Run experiments: bash experiments/run_baseline.sh"
echo "4. View results: jupyter notebook notebooks/analysis.ipynb"
