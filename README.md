# Adaptive Federated Learning for Agricultural IoT

A comprehensive software-only federated learning framework designed for Agricultural IoT systems that handles unreliable clients with clustered personalization.

## 🌾 Project Overview

This project implements and evaluates a federated learning (FL) framework tailored to Agricultural IoT constraints. It addresses two critical challenges:

1. **Unreliable client participation** - IoT devices frequently dropout due to connectivity issues, battery depletion, or network instability
2. **Heterogeneous farm data** - Different farms have varying soil conditions, crops, climate patterns requiring personalized models

### Key Contributions

- **Adaptive Client Participation**: Priority-based scheduling that considers device reliability and data importance
- **Clustered Personalization**: k-means grouping of similar farms for semi-personalized model training  
- **Communication Efficiency**: Lightweight model compression with Top-K sparsification and quantization
- **Comprehensive Evaluation**: Comparison of FedAvg, FedProx, adaptive methods, and clustering approaches

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Farm Client   │    │  Aggregation     │    │   Farm Client   │
│   IoT Devices   │◄──►│     Server       │◄──►│   IoT Devices   │
│                 │    │                  │    │                 │
│ • Sensors       │    │ • Adaptive       │    │ • Cameras       │
│ • Local Model   │    │   Selection      │    │ • Local Model   │  
│ • Data Privacy  │    │ • Clustering     │    │ • Data Privacy  │
└─────────────────┘    │ • Personalization│    └─────────────────┘
                       └──────────────────┘
```

## 📁 Project Structure

```
agri-fl-software-only/
├── configs/
│   └── config.py              # Experiment configurations
├── data/
│   └── prepare_data.py        # Dataset preparation & non-IID partitioning
├── models/
│   └── models.py              # CNN and ResNet implementations
├── utils/
│   ├── clustering.py          # Client clustering for personalization
│   └── compress.py            # Model compression utilities
├── client/
│   └── client.py              # Federated learning client implementation
├── server/
│   └── server.py              # Adaptive FL server with clustering
├── experiments/
│   ├── run_baseline.sh        # Baseline FedAvg experiment
│   ├── run_adaptive.sh        # Adaptive participation experiment
│   ├── run_clustered.sh       # Clustered personalization experiment
│   ├── run_combined.sh        # Combined adaptive + clustering
│   └── run_suite.sh           # Full experiment suite
├── notebooks/
│   └── analysis.ipynb         # Results analysis and visualization
├── results/                   # Experiment outputs
├── main.py                    # Main experiment runner
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone <your-repo-url>
cd agri-fl-software-only

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Basic Experiment

```bash
# Quick test with synthetic data
python main.py --experiment single --dataset fakedata --rounds 10 --clients 5

# Run baseline experiment
bash experiments/run_baseline.sh

# Run full experiment suite
bash experiments/run_suite.sh
```

### 3. View Results

```bash
# Start Jupyter notebook for analysis
jupyter notebook notebooks/analysis.ipynb

# Check results directory
ls results/
```

## 🧪 Experiments

The project includes 7 comprehensive experiments:

| Experiment | Description | Key Features |
|------------|-------------|--------------|
| **E0** | Baseline IID | Standard FedAvg on IID data |
| **E1** | Non-IID FedAvg | FedAvg with label-skewed data |
| **E2** | FedProx | Proximal regularization (μ=0.01) |
| **E3** | Adaptive Participation | Smart client selection |
| **E4** | Clustered Personalization | K-means clustering (k=3) |
| **E5** | Combined Approach | Adaptive + Clustering |
| **E6** | Compression | Communication efficiency |

### Running Individual Experiments

```bash
# Adaptive participation
bash experiments/run_adaptive.sh

# Clustered personalization  
bash experiments/run_clustered.sh

# Combined approach
bash experiments/run_combined.sh

# With real PlantVillage dataset (requires manual download)
bash experiments/run_plantvillage.sh
```

### Custom Experiments

```bash
python main.py \
    --experiment single \
    --name "MyExperiment" \
    --dataset fakedata \
    --rounds 30 \
    --clients 8
```

## 📊 Datasets

### Synthetic Data (Default)
- **FakeData**: Generated agricultural-like image data (3x32x32, 10 classes)
- Perfect for testing and development
- Non-IID partitioning with Dirichlet distribution (α=0.1)

### Real Data
- **PlantVillage**: Plant disease detection dataset
- Download from [Kaggle PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  
- Place in `data/plantvillage/` directory
- Supports crop disease classification scenarios

## ⚙️ Configuration

Key configuration parameters in `configs/config.py`:

```python
# Experiment Settings
EXPERIMENT_CONFIG = {
    "num_clients": 10,          # Number of federated clients
    "num_rounds": 50,           # Training rounds
    "local_epochs": 3,          # Local training epochs
    "fraction_fit": 0.6,        # Client participation fraction
}

# Adaptive Participation
ADAPTIVE_CONFIG = {
    "enable_adaptive": True,
    "reliability_weight": 0.6,   # Weight for device reliability
    "importance_weight": 0.4,    # Weight for data importance
}

# Clustering Personalization  
CLUSTER_CONFIG = {
    "enable_clustering": True,
    "num_clusters": 3,           # Number of farm clusters
    "cluster_method": "kmeans",
}
```

## 🔬 Methodology

### Adaptive Client Participation
- **Reliability Score**: Simulated device uptime probability (0.3-0.95)
- **Importance Score**: Based on data entropy and class rarity  
- **Participation Score**: Weighted combination with fairness adjustment
- **Selection**: Probabilistic sampling favoring high-scoring clients

### Clustered Personalization
- **Feature Extraction**: Statistical properties (mean, std, class distribution)
- **Clustering**: K-means on client feature vectors (k=3)
- **Aggregation**: Within-cluster FedAvg, then cluster-level personalization
- **Adaptation**: Local fine-tuning on cluster-specific models

### Model Compression
- **Top-K Sparsification**: Send only top 5% of parameters by magnitude
- **8-bit Quantization**: Uniform quantization for communication efficiency
- **Hybrid Approach**: Top-K for large layers, quantization for others

## 📈 Expected Results

Based on agricultural federated learning literature, you should observe:

1. **FedProx vs FedAvg**: ~2-5% accuracy improvement under non-IID conditions
2. **Adaptive Participation**: ~15-25% reduction in convergence variance
3. **Clustered Personalization**: ~10-20% improvement in client-specific accuracy
4. **Compression**: 80-90% communication reduction with <2% accuracy loss
5. **Combined Approach**: Best overall performance with improved fairness

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+
- Flower 1.5+
- scikit-learn 1.3+
- matplotlib, seaborn, pandas
- Jupyter notebook (optional, for analysis)

## 🤝 Usage for Research

### Citing This Work
If you use this framework in your research, please cite:

```bibtex
@article{agri_fl_2024,
  title={Adaptive Federated Learning for Agricultural IoT: Handling Unreliable Clients with Clustered Personalization},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

### Extending the Framework

1. **New Algorithms**: Add implementations in `server/server.py`
2. **Custom Datasets**: Extend `data/prepare_data.py`  
3. **Different Models**: Add architectures to `models/models.py`
4. **Novel Compression**: Implement in `utils/compress.py`
5. **Advanced Clustering**: Extend `utils/clustering.py`

## 🛠️ Development

### Running Tests

```bash
# Test individual components
python -m data.prepare_data
python -m models.models  
python -m client.client
python -m server.server

# Test full pipeline
python main.py --experiment single --rounds 5 --clients 3
```

### Docker Support (Optional)

```bash
# Build container
docker build -t agri-fl .

# Run experiments
docker run -v $(pwd)/results:/app/results agri-fl python main.py --experiment suite
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or number of clients
2. **Slow Training**: Use FakeData instead of PlantVillage for testing
3. **Import Errors**: Ensure all dependencies are installed correctly
4. **No GPU**: Framework runs on CPU automatically

### Debug Mode

```bash
# Enable detailed logging
export PYTHONPATH=$PYTHONPATH:$(pwd)
python main.py --experiment single --rounds 5 --clients 2
```

## 🎯 Practical Deployment

### For Real Agricultural IoT Systems

1. **Device Requirements**:
   - ARM-based edge devices (Raspberry Pi 4+)
   - 2GB+ RAM for model training
   - WiFi/4G connectivity (intermittent OK)

2. **Recommended Setup**:
   - 10-20 farms per cluster
   - Weekly aggregation rounds
   - Seasonal model updates
   - Edge-cloud hybrid architecture

3. **Security Considerations**:
   - TLS for communication
   - Differential privacy (optional)
   - Secure aggregation protocols

## 📧 Support

For questions, issues, or contributions:
- Open GitHub issues for bugs
- Discussions for research questions  
- Pull requests for improvements

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

---

**🌱 Built for sustainable agriculture through privacy-preserving AI collaboration**
