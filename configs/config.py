"""
Configuration file for Adaptive Federated Learning for Agricultural IoT
"""

# Experiment Configuration
EXPERIMENT_CONFIG = {
    "num_clients": 5,
    "num_rounds": 50,
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.01,
    "fraction_fit": 0.6,  # Fraction of clients to sample each round
    "fraction_eval": 0.1,
    "min_fit_clients": 3,
    "min_eval_clients": 1,
    "seed": 42
}

# Model Configuration
MODEL_CONFIG = {
    "model_type": "cnn",
    "num_classes": 10,  # Will be updated based on dataset
    "input_channels": 3,
    "hidden_size": 64
}

# Dataset Configuration
DATASET_CONFIG = {
    "dataset_name": "fakedata",  # Use FakeData for quick testing
    "data_path": "./data",
    "non_iid": True,
    "alpha": 1.0,  # Dirichlet distribution parameter for non-IID
    "label_skew_clients": 2
}

# Adaptive Participation Configuration
ADAPTIVE_CONFIG = {
    "enable_adaptive": True,
    "reliability_weight": 0.6,
    "importance_weight": 0.4,
    "min_reliability": 0.3,
    "max_reliability": 0.95
}

# Clustered Personalization Configuration
CLUSTER_CONFIG = {
    "enable_clustering": True,
    "num_clusters": 3,
    "cluster_method": "kmeans",
    "feature_dim": 10,  # Dimension of client feature vectors
    "personalization_rounds": 5
}

# FedProx Configuration
FEDPROX_CONFIG = {
    "enable_fedprox": True,
    "mu": 0.01  # Proximal term coefficient
}

# Compression Configuration
COMPRESSION_CONFIG = {
    "enable_compression": True,
    "compression_type": "topk",  # "topk" or "quantization"
    "topk_ratio": 0.05,  # For top-k sparsification
    "quantization_bits": 8  # For quantization
}

# Logging and Results
LOGGING_CONFIG = {
    "log_level": "INFO",
    "save_results": True,
    "results_dir": "./results",
    "plot_results": True
}
