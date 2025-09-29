"""
Federated Learning Client Implementation
Supports FedAvg, FedProx, and adaptive participation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, OrderedDict
import flwr as fl
import copy
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.models import create_model, get_model_parameters, set_model_parameters
from utils.compress import get_compressor, ModelCompressor


class AgricultureFLClient(fl.client.NumPyClient):
    """Federated Learning Client for Agricultural IoT"""

    def __init__(self, client_id: int, model: nn.Module, trainloader: DataLoader, 
                 testloader: DataLoader, device: torch.device, config: Dict):
        self.client_id = client_id
        self.model = model.to(device)
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.config = config

        # FedProx parameters
        self.mu = config.get('mu', 0.01)
        self.enable_fedprox = config.get('enable_fedprox', False)
        self.global_params = None

        # Compression
        self.compression_config = config.get('compression', {})
        self.compressor = None
        if self.compression_config.get('enable_compression', False):
            self.compressor = get_compressor(
                self.compression_config.get('compression_type', 'topk'),
                **self.compression_config
            )

        # Client characteristics for adaptive participation
        self.reliability = np.random.uniform(0.3, 0.95)  # Simulated device reliability
        self.importance_score = 1.0  # Will be updated based on data importance
        self.participation_score = 0.5

        # Training history
        self.training_history = {
            'losses': [],
            'accuracies': [],
            'communication_bytes': 0
        }

        self._compute_data_importance()

    def _compute_data_importance(self):
        """Compute data importance score for adaptive participation"""

        if not self.trainloader:
            self.importance_score = 0.1
            return

        # Count samples per class
        class_counts = {}
        total_samples = 0

        for _, targets in self.trainloader:
            for target in targets:
                target_item = target.item()
                class_counts[target_item] = class_counts.get(target_item, 0) + 1
                total_samples += 1

        if total_samples == 0:
            self.importance_score = 0.1
            return

        # Calculate entropy-based importance
        class_probs = [count / total_samples for count in class_counts.values()]
        entropy = -sum(p * np.log(p) for p in class_probs if p > 0)

        # Calculate rarity score (higher for clients with rare classes)
        num_unique_classes = len(class_counts)
        rarity_score = num_unique_classes / 10.0  # Normalize by max expected classes

        # Combine entropy and rarity
        self.importance_score = 0.5 * entropy + 0.5 * rarity_score

        # Update participation score
        reliability_weight = self.config.get('reliability_weight', 0.6)
        importance_weight = self.config.get('importance_weight', 0.4)

        self.participation_score = (reliability_weight * self.reliability + 
                                  importance_weight * self.importance_score)

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Get model parameters"""
        parameters = [val.cpu().numpy() for _, val in self.model.state_dict().items()]

        # Apply compression if enabled
        if self.compressor and config.get('compress_upload', False):
            model_params = OrderedDict(self.model.named_parameters())
            compressed_params, metadata = self.compressor.compress(model_params)

            # Store metadata for decompression (in practice, this would be handled differently)
            self.compression_metadata = metadata

            # Convert compressed parameters to numpy arrays
            # This is a simplified approach - in practice, you'd implement proper serialization
            parameters = self._serialize_compressed_params(compressed_params)

            # Update communication bytes
            compressed_size = self.compressor.compressed_size * 4  # Rough estimate in bytes
            self.training_history['communication_bytes'] += compressed_size
        else:
            # Standard communication
            param_size = sum(param.nbytes for param in parameters)
            self.training_history['communication_bytes'] += param_size

        return parameters

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters"""
        # Handle compressed parameters
        if hasattr(self, 'compression_metadata'):
            # Decompress parameters (simplified implementation)
            # In practice, this would be more sophisticated
            pass

        # Standard parameter setting
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # Store global parameters for FedProx
        if self.enable_fedprox:
            self.global_params = copy.deepcopy(self.model.state_dict())

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model"""

        # Set global parameters
        self.set_parameters(parameters)

        # Training configuration
        epochs = config.get('local_epochs', self.config.get('local_epochs', 3))
        learning_rate = config.get('learning_rate', self.config.get('learning_rate', 0.01))

        # Setup optimizer
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_idx, (data, targets) in enumerate(self.trainloader):
                data, targets = data.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, targets)

                # Add proximal term for FedProx
                if self.enable_fedprox and self.global_params is not None:
                    proximal_loss = 0.0
                    for param_name, param in self.model.named_parameters():
                        if param_name in self.global_params:
                            proximal_loss += torch.norm(param - self.global_params[param_name].to(self.device)) ** 2

                    loss += (self.mu / 2.0) * proximal_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            total_loss += epoch_loss

            # Log progress
            if epoch % max(1, epochs // 2) == 0:
                print(f"Client {self.client_id}, Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        avg_loss = total_loss / max(1, num_batches)
        self.training_history['losses'].append(avg_loss)

        # Return updated parameters
        updated_parameters = self.get_parameters(config)

        # Training metrics
        metrics = {
            'loss': avg_loss,
            'client_id': self.client_id,
            'participation_score': self.participation_score,
            'reliability': self.reliability,
            'importance_score': self.importance_score
        }

        return updated_parameters, len(self.trainloader.dataset), metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        """Evaluate the model"""

        # Set parameters
        self.set_parameters(parameters)

        # Evaluation
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, targets in self.testloader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.testloader) if len(self.testloader) > 0 else 0.0

        self.training_history['accuracies'].append(accuracy)

        metrics = {
            'accuracy': accuracy,
            'client_id': self.client_id,
            'correct': correct,
            'total': total
        }

        return avg_loss, total, metrics

    def _serialize_compressed_params(self, compressed_params: Dict) -> List[np.ndarray]:
        """Serialize compressed parameters to numpy arrays (simplified)"""
        # This is a simplified implementation
        # In practice, you'd implement proper serialization for different compression types
        serialized = []

        for param_name, param_data in compressed_params.items():
            if isinstance(param_data, dict) and 'values' in param_data:
                # Top-K compression
                serialized.append(param_data['values'].cpu().numpy())
                serialized.append(param_data['indices'].cpu().numpy().astype(np.float32))
            else:
                # Quantization or other
                serialized.append(param_data.cpu().numpy())

        return serialized

    def get_client_info(self) -> Dict:
        """Get client information for server-side decisions"""
        return {
            'client_id': self.client_id,
            'reliability': self.reliability,
            'importance_score': self.importance_score,
            'participation_score': self.participation_score,
            'dataset_size': len(self.trainloader.dataset),
            'communication_bytes': self.training_history['communication_bytes']
        }


def create_client(client_id: int, model_config: Dict, trainloader: DataLoader,
                 testloader: DataLoader, device: torch.device, 
                 fl_config: Dict) -> AgricultureFLClient:
    """Factory function to create FL client"""

    # Create model
    model = create_model(
        model_type=model_config['model_type'],
        num_classes=model_config['num_classes'],
        input_channels=model_config['input_channels']
    )

    # Create client
    client = AgricultureFLClient(
        client_id=client_id,
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        device=device,
        config=fl_config
    )

    return client


if __name__ == "__main__":
    # Test client functionality
    print("Testing FL Client...")

    # Create dummy data
    from torch.utils.data import TensorDataset

    # Dummy data
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 10, (100,))

    train_dataset = TensorDataset(X[:80], y[:80])
    test_dataset = TensorDataset(X[80:], y[80:])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model config
    model_config = {
        'model_type': 'cnn',
        'num_classes': 10,
        'input_channels': 3
    }

    # FL config
    fl_config = {
        'local_epochs': 2,
        'learning_rate': 0.01,
        'enable_fedprox': True,
        'mu': 0.01,
        'compression': {
            'enable_compression': False
        }
    }

    device = torch.device('cpu')

    # Create client
    client = create_client(
        client_id=0,
        model_config=model_config,
        trainloader=train_loader,
        testloader=test_loader,
        device=device,
        fl_config=fl_config
    )

    # Test methods
    print(f"Client info: {client.get_client_info()}")

    # Get initial parameters
    initial_params = client.get_parameters({})
    print(f"Model has {len(initial_params)} parameter tensors")

    # Test training
    updated_params, num_samples, metrics = client.fit(initial_params, {'local_epochs': 1})
    print(f"Training completed. Metrics: {metrics}")

    # Test evaluation
    loss, eval_samples, eval_metrics = client.evaluate(updated_params, {})
    print(f"Evaluation completed. Loss: {loss:.4f}, Metrics: {eval_metrics}")
