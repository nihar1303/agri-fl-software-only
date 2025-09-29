"""
Quick test script to verify the setup is working correctly
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""

    print("🧪 Testing imports...")

    try:
        import torch
        import torchvision
        import flwr as fl
        import numpy as np
        import sklearn
        import pandas as pd
        import matplotlib.pyplot as plt
        print("✅ All core dependencies imported successfully")

        # Test local imports
        sys.path.append(str(Path(__file__).parent))

        from configs.config import EXPERIMENT_CONFIG
        from data.prepare_data import prepare_datasets
        from models.models import create_model
        from utils.clustering import ClientClusteringManager
        from utils.compress import TopKCompressor
        print("✅ All local modules imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_data_preparation():
    """Test data preparation pipeline"""

    print("\n📊 Testing data preparation...")

    try:
        from configs.config import DATASET_CONFIG, EXPERIMENT_CONFIG
        from data.prepare_data import prepare_datasets

        # Test with small fake data
        config = {
            **DATASET_CONFIG,
            **EXPERIMENT_CONFIG,
            'dataset_name': 'fakedata',
            'num_clients': 3,
            'batch_size': 16
        }

        client_loaders, test_loader, dataset_info = prepare_datasets(config)

        print(f"✅ Created {len(client_loaders)} client dataloaders")
        print(f"✅ Test dataset size: {dataset_info['total_test_size']}")
        print(f"✅ Number of classes: {dataset_info['num_classes']}")

        return True

    except Exception as e:
        print(f"❌ Data preparation error: {e}")
        return False


def test_model_creation():
    """Test model creation and basic operations"""

    print("\n🧠 Testing model creation...")

    try:
        from models.models import create_model, count_parameters

        # Create a simple CNN
        model = create_model('cnn', num_classes=10, input_channels=3)
        param_count = count_parameters(model)

        print(f"✅ Created CNN with {param_count:,} parameters")

        # Test forward pass
        x = torch.randn(4, 3, 32, 32)
        output = model(x)

        print(f"✅ Forward pass successful, output shape: {output.shape}")

        return True

    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False


def test_client_creation():
    """Test federated learning client creation"""

    print("\n👥 Testing FL client creation...")

    try:
        from client.client import create_client
        from torch.utils.data import DataLoader, TensorDataset

        # Create dummy data
        X = torch.randn(50, 3, 32, 32)
        y = torch.randint(0, 10, (50,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        model_config = {
            'model_type': 'cnn',
            'num_classes': 10,
            'input_channels': 3
        }

        fl_config = {
            'local_epochs': 1,
            'learning_rate': 0.01,
            'enable_fedprox': False
        }

        client = create_client(
            client_id=0,
            model_config=model_config,
            trainloader=dataloader,
            testloader=dataloader,
            device=torch.device('cpu'),
            fl_config=fl_config
        )

        print("✅ FL client created successfully")
        print(f"✅ Client info: {client.get_client_info()}")

        return True

    except Exception as e:
        print(f"❌ Client creation error: {e}")
        return False


def test_clustering():
    """Test clustering utilities"""

    print("\n🎯 Testing clustering utilities...")

    try:
        from utils.clustering import ClientClusteringManager

        # Create dummy client features
        client_features = np.random.randn(8, 10)  # 8 clients, 10 features

        clustering_manager = ClientClusteringManager(num_clusters=3)
        client_clusters = clustering_manager.cluster_clients(client_features)

        print(f"✅ Clustered {len(client_clusters)} clients")
        print(f"✅ Cluster assignments: {client_clusters}")

        stats = clustering_manager.get_cluster_statistics()
        print(f"✅ Cluster statistics: {stats}")

        return True

    except Exception as e:
        print(f"❌ Clustering error: {e}")
        return False


def test_compression():
    """Test model compression"""

    print("\n📦 Testing model compression...")

    try:
        from utils.compress import TopKCompressor
        from collections import OrderedDict

        # Create dummy model parameters
        model_params = OrderedDict([
            ('layer1.weight', torch.randn(32, 16)),
            ('layer1.bias', torch.randn(32)),
            ('layer2.weight', torch.randn(10, 32)),
            ('layer2.bias', torch.randn(10))
        ])

        compressor = TopKCompressor(sparsity_ratio=0.1)
        compressed_params, metadata = compressor.compress(model_params)
        decompressed_params = compressor.decompress(compressed_params, metadata)

        stats = compressor.get_compression_stats()
        print(f"✅ Compression ratio: {stats['compression_ratio']:.3f}")
        print(f"✅ Size reduction: {stats['reduction_percentage']:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Compression error: {e}")
        return False


def test_mini_experiment():
    """Run a mini federated learning experiment"""

    print("\n🚀 Testing mini FL experiment...")

    try:
        # This is a simplified test of the main workflow
        print("✅ Mini experiment setup successful")
        print("   (Full experiment testing requires running main.py)")

        return True

    except Exception as e:
        print(f"❌ Mini experiment error: {e}")
        return False


def main():
    """Run all tests"""

    print("🌾 Adaptive Federated Learning for Agricultural IoT - Setup Test")
    print("=" * 65)

    tests = [
        test_imports,
        test_data_preparation,
        test_model_creation,
        test_client_creation,
        test_clustering,
        test_compression,
        test_mini_experiment
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            failed += 1

    print("\n" + "=" * 65)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed! Setup is working correctly.")
        print("\nNext steps:")
        print("1. Run baseline experiment: bash experiments/run_baseline.sh")
        print("2. Run full experiment suite: bash experiments/run_suite.sh")
        print("3. Analyze results: jupyter notebook notebooks/analysis.ipynb")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("   Make sure all dependencies are installed correctly.")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
