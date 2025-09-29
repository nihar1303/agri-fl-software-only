"""
Federated Learning Server Implementation
Supports adaptive client selection, clustered personalization, and various aggregation strategies
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, OrderedDict
import flwr as fl
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy
from flwr.common import Parameters, FitRes, EvaluateRes, parameters_to_ndarrays, ndarrays_to_parameters
import copy
import sys
import os
from collections import defaultdict
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.clustering import ClientClusteringManager, ClusterPersonalizationAggregator
from models.models import create_model


class AdaptiveFederatedStrategy(Strategy):
    """
    Custom Federated Learning Strategy with Adaptive Client Selection and Clustering
    """

    def __init__(self, model_config: Dict, fl_config: Dict, device: torch.device):
        super().__init__()

        self.model_config = model_config
        self.fl_config = fl_config
        self.device = device

        # Initialize global model
        self.global_model = create_model(
            model_type=model_config['model_type'],
            num_classes=model_config['num_classes'],
            input_channels=model_config['input_channels']
        ).to(device)

        # Adaptive participation
        self.enable_adaptive = fl_config.get('enable_adaptive', True)
        self.client_scores = {}
        self.client_info = {}
        self.participation_history = defaultdict(list)

        # Clustering for personalization
        self.enable_clustering = fl_config.get('enable_clustering', True)
        self.clustering_manager = None
        self.personalization_aggregator = None
        self.cluster_models = {}
        self.personalization_round = 0

        if self.enable_clustering:
            cluster_config = fl_config.get('cluster_config', {})
            self.clustering_manager = ClientClusteringManager(
                num_clusters=cluster_config.get('num_clusters', 3),
                method=cluster_config.get('cluster_method', 'kmeans'),
                feature_dim=cluster_config.get('feature_dim', 10)
            )
            self.personalization_aggregator = ClusterPersonalizationAggregator(self.clustering_manager)

        # Training history
        self.training_history = {
            'global_accuracy': [],
            'global_loss': [],
            'client_accuracies': [],
            'participation_counts': defaultdict(int),
            'cluster_assignments': {},
            'communication_rounds': 0
        }

        # Current round
        self.current_round = 0

        print(f"Initialized Adaptive Federated Strategy:")
        print(f"  - Adaptive participation: {self.enable_adaptive}")
        print(f"  - Clustered personalization: {self.enable_clustering}")
        print(f"  - Global model parameters: {self._count_parameters():,}")

    def _count_parameters(self) -> int:
        """Count model parameters"""
        return sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize global model parameters"""
        parameters = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        return ndarrays_to_parameters(parameters)

    def configure_fit(self, server_round: int, parameters: Parameters, 
                     client_manager) -> List[Tuple[ClientProxy, Dict]]:
        """Configure clients for training round"""

        self.current_round = server_round
        available_clients = list(client_manager.all().values())

        if not available_clients:
            return []

        # Update client information (simulated)
        self._update_client_info(available_clients)

        # Select clients adaptively
        if self.enable_adaptive and server_round > 1:
            selected_clients = self._adaptive_client_selection(available_clients)
        else:
            # Standard random selection for first round or when adaptive is disabled
            fraction_fit = self.fl_config.get('fraction_fit', 0.6)
            min_fit_clients = self.fl_config.get('min_fit_clients', 3)

            num_clients = max(min_fit_clients, int(len(available_clients) * fraction_fit))
            selected_clients = np.random.choice(available_clients, 
                                              size=min(num_clients, len(available_clients)), 
                                              replace=False).tolist()

        # Update participation history
        for client in selected_clients:
            client_id = int(client.cid)
            self.training_history['participation_counts'][client_id] += 1
            self.participation_history[client_id].append(server_round)

        # Configuration for selected clients
        config = {
            'server_round': server_round,
            'local_epochs': self.fl_config.get('local_epochs', 3),
            'learning_rate': self._adaptive_learning_rate(server_round),
            'compress_upload': server_round > 10  # Enable compression after initial rounds
        }

        client_configs = [(client, config) for client in selected_clients]

        print(f"Round {server_round}: Selected {len(selected_clients)} clients")
        if self.enable_adaptive:
            selected_ids = [int(c.cid) for c in selected_clients]
            selected_scores = [self.client_scores.get(cid, 0.5) for cid in selected_ids]
            print(f"  Selected client IDs: {selected_ids}")
            print(f"  Average participation score: {np.mean(selected_scores):.3f}")

        return client_configs

    def _update_client_info(self, available_clients: List[ClientProxy]):
        """Update client information for adaptive selection"""

        # In practice, this would come from client reports
        # For simulation, we'll generate realistic client characteristics

        for client in available_clients:
            client_id = int(client.cid)

            if client_id not in self.client_info:
                # Initialize client info
                self.client_info[client_id] = {
                    'reliability': np.random.uniform(0.3, 0.95),
                    'importance_score': np.random.uniform(0.1, 1.0),
                    'dataset_size': np.random.randint(50, 500),
                    'last_seen': self.current_round
                }
            else:
                # Update last seen
                self.client_info[client_id]['last_seen'] = self.current_round

                # Simulate reliability changes over time
                current_reliability = self.client_info[client_id]['reliability']
                noise = np.random.normal(0, 0.05)  # Small random changes
                new_reliability = np.clip(current_reliability + noise, 0.1, 1.0)
                self.client_info[client_id]['reliability'] = new_reliability

            # Update participation score
            info = self.client_info[client_id]
            reliability_weight = self.fl_config.get('reliability_weight', 0.6)
            importance_weight = self.fl_config.get('importance_weight', 0.4)

            participation_score = (reliability_weight * info['reliability'] + 
                                 importance_weight * info['importance_score'])

            self.client_scores[client_id] = participation_score

    def _adaptive_client_selection(self, available_clients: List[ClientProxy]) -> List[ClientProxy]:
        """Select clients based on participation scores and fairness"""

        fraction_fit = self.fl_config.get('fraction_fit', 0.6)
        min_fit_clients = self.fl_config.get('min_fit_clients', 3)
        num_clients = max(min_fit_clients, int(len(available_clients) * fraction_fit))

        if num_clients >= len(available_clients):
            return available_clients

        # Get client scores
        client_ids = [int(client.cid) for client in available_clients]
        scores = [self.client_scores.get(cid, 0.5) for cid in client_ids]

        # Add fairness component (favor less frequently selected clients)
        fairness_scores = []
        for cid in client_ids:
            participation_count = self.training_history['participation_counts'][cid]
            fairness_score = 1.0 / (1.0 + participation_count * 0.1)  # Diminishing returns
            fairness_scores.append(fairness_score)

        # Combine scores
        combined_scores = np.array(scores) * 0.7 + np.array(fairness_scores) * 0.3

        # Probabilistic selection based on scores
        probabilities = combined_scores / np.sum(combined_scores)

        selected_indices = np.random.choice(
            len(available_clients), 
            size=num_clients, 
            replace=False, 
            p=probabilities
        )

        selected_clients = [available_clients[i] for i in selected_indices]

        return selected_clients

    def _adaptive_learning_rate(self, server_round: int) -> float:
        """Adaptive learning rate schedule"""

        base_lr = self.fl_config.get('learning_rate', 0.01)

        # Cosine annealing
        max_rounds = self.fl_config.get('num_rounds', 50)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * server_round / max_rounds))

        return max(lr, base_lr * 0.1)  # Minimum learning rate

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], 
                     failures: List[Tuple[ClientProxy, Exception]]) -> Tuple[Optional[Parameters], Dict]:
        """Aggregate client updates"""

        if not results:
            return None, {}

        # Extract client updates and metrics
        client_updates = {}
        client_metrics = {}
        total_samples = 0

        for client, fit_res in results:
            client_id = int(client.cid)
            parameters = parameters_to_ndarrays(fit_res.parameters)
            num_samples = fit_res.num_examples
            metrics = fit_res.metrics

            client_updates[client_id] = parameters
            client_metrics[client_id] = metrics
            total_samples += num_samples

        # Perform clustering-based personalization
        if (self.enable_clustering and server_round > 5 and 
            server_round % self.fl_config.get('personalization_rounds', 5) == 0):

            aggregated_parameters = self._clustered_aggregation(client_updates, client_metrics)

        else:
            # Standard FedAvg aggregation
            aggregated_parameters = self._federated_averaging(client_updates, client_metrics)

        # Update global model
        aggregated_parameters_fl = ndarrays_to_parameters(aggregated_parameters)

        # Collect aggregation metrics
        aggregation_metrics = {
            'participating_clients': len(results),
            'total_samples': total_samples,
            'failed_clients': len(failures),
            'average_client_loss': np.mean([m.get('loss', 0) for m in client_metrics.values()]),
        }

        # Add clustering information if available
        if self.clustering_manager and self.clustering_manager.client_clusters:
            aggregation_metrics['cluster_info'] = self.clustering_manager.get_cluster_statistics()

        self.training_history['communication_rounds'] += 1

        print(f"Round {server_round} aggregation completed:")
        print(f"  - Participating clients: {len(results)}")
        print(f"  - Total samples: {total_samples}")
        print(f"  - Average client loss: {aggregation_metrics['average_client_loss']:.4f}")

        return aggregated_parameters_fl, aggregation_metrics

    def _federated_averaging(self, client_updates: Dict[int, List], 
                           client_metrics: Dict[int, Dict]) -> List[np.ndarray]:
        """Standard FedAvg aggregation"""

        # Calculate weights based on number of samples
        weights = []
        total_samples = 0

        for client_id in client_updates.keys():
            num_samples = client_metrics[client_id].get('num_samples', 1)
            weights.append(num_samples)
            total_samples += num_samples

        # Normalize weights
        weights = [w / total_samples for w in weights]

        # Aggregate parameters
        aggregated_params = []
        client_params_list = list(client_updates.values())

        for layer_idx in range(len(client_params_list[0])):
            layer_params = [client_params[layer_idx] for client_params in client_params_list]

            # Weighted average
            weighted_layer = sum(w * params for w, params in zip(weights, layer_params))
            aggregated_params.append(weighted_layer)

        return aggregated_params

    def _clustered_aggregation(self, client_updates: Dict[int, List], 
                              client_metrics: Dict[int, Dict]) -> List[np.ndarray]:
        """Clustered personalization aggregation"""

        print(f"  Performing clustered personalization...")

        # Extract client features (simplified version)
        client_ids = list(client_updates.keys())
        client_features = np.random.randn(len(client_ids), 10)  # Placeholder

        # In practice, client features would be computed from actual client data
        # For simulation, we use metrics as features
        for i, client_id in enumerate(client_ids):
            metrics = client_metrics[client_id]
            # Use available metrics as features
            client_features[i, 0] = metrics.get('loss', 0.5)
            client_features[i, 1] = metrics.get('importance_score', 0.5)
            client_features[i, 2] = metrics.get('reliability', 0.5)
            # Add random features for other dimensions
            client_features[i, 3:] = np.random.randn(7) * 0.1

        # Perform clustering
        client_cluster_mapping = self.clustering_manager.cluster_clients(client_features)

        # Update training history
        self.training_history['cluster_assignments'][self.current_round] = client_cluster_mapping

        # Convert client updates to appropriate format for clustering
        converted_updates = {}
        for client_id, params in client_updates.items():
            # Convert numpy arrays to OrderedDict format expected by clustering
            param_dict = {}
            param_names = [f'param_{i}' for i in range(len(params))]
            for name, param in zip(param_names, params):
                param_dict[name] = torch.from_numpy(param)
            converted_updates[client_id] = param_dict

        # Get global model parameters as baseline
        global_params = {}
        param_names = [f'param_{i}' for i in range(len(client_updates[client_ids[0]]))]
        for i, name in enumerate(param_names):
            global_params[name] = torch.zeros_like(torch.from_numpy(client_updates[client_ids[0]][i]))

        # Aggregate within clusters
        cluster_models = self.personalization_aggregator.aggregate_cluster_updates(
            converted_updates, global_params
        )

        # For simplicity, return standard FedAvg for global model
        # In practice, you might want to aggregate cluster models differently
        global_aggregated = self._federated_averaging(client_updates, client_metrics)

        # Store cluster models for future personalization
        self.cluster_models = cluster_models

        print(f"  Clustered {len(client_ids)} clients into {len(cluster_models)} clusters")

        return global_aggregated

    def configure_evaluate(self, server_round: int, parameters: Parameters, 
                          client_manager) -> List[Tuple[ClientProxy, Dict]]:
        """Configure clients for evaluation"""

        # Evaluate on a subset of clients
        fraction_eval = self.fl_config.get('fraction_eval', 0.1)
        min_eval_clients = self.fl_config.get('min_eval_clients', 1)

        available_clients = list(client_manager.all().values())
        num_clients = max(min_eval_clients, int(len(available_clients) * fraction_eval))

        if num_clients >= len(available_clients):
            selected_clients = available_clients
        else:
            selected_clients = np.random.choice(available_clients, 
                                              size=num_clients, 
                                              replace=False).tolist()

        config = {'server_round': server_round}
        return [(client, config) for client in selected_clients]

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], 
                          failures: List[Tuple[ClientProxy, Exception]]) -> Tuple[Optional[float], Dict]:
        """Aggregate evaluation results"""

        if not results:
            return None, {}

        # Calculate weighted average loss and accuracy
        total_samples = 0
        total_loss = 0.0
        total_correct = 0

        client_accuracies = []

        for client, eval_res in results:
            client_id = int(client.cid)
            loss = eval_res.loss
            num_samples = eval_res.num_examples
            metrics = eval_res.metrics

            accuracy = metrics.get('accuracy', 0.0)
            correct = metrics.get('correct', 0)

            total_loss += loss * num_samples
            total_samples += num_samples
            total_correct += correct
            client_accuracies.append(accuracy)

        # Global metrics
        global_loss = total_loss / total_samples if total_samples > 0 else 0.0
        global_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        # Update training history
        self.training_history['global_accuracy'].append(global_accuracy)
        self.training_history['global_loss'].append(global_loss)
        self.training_history['client_accuracies'].append(client_accuracies)

        evaluation_metrics = {
            'global_accuracy': global_accuracy,
            'accuracy_std': np.std(client_accuracies),
            'participating_clients': len(results),
            'total_samples': total_samples
        }

        print(f"Round {server_round} evaluation:")
        print(f"  - Global accuracy: {global_accuracy:.4f}")
        print(f"  - Global loss: {global_loss:.4f}")
        print(f"  - Accuracy std: {evaluation_metrics['accuracy_std']:.4f}")

        return global_loss, evaluation_metrics

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict]]:
        """Server-side evaluation (optional)"""
        # This could implement centralized evaluation on a held-out test set
        # For simplicity, we'll skip server-side evaluation
        return None

    def get_training_history(self) -> Dict:
        """Get complete training history"""
        return self.training_history

    def save_results(self, filepath: str):
        """Save training results"""
        results = {
            'training_history': self.training_history,
            'client_info': self.client_info,
            'client_scores': self.client_scores,
            'config': {
                'model_config': self.model_config,
                'fl_config': self.fl_config
            }
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"Training results saved to {filepath}")


if __name__ == "__main__":
    # Test server strategy
    print("Testing Adaptive Federated Strategy...")

    model_config = {
        'model_type': 'cnn',
        'num_classes': 10,
        'input_channels': 3
    }

    fl_config = {
        'num_rounds': 10,
        'fraction_fit': 0.6,
        'fraction_eval': 0.1,
        'min_fit_clients': 2,
        'min_eval_clients': 1,
        'local_epochs': 3,
        'learning_rate': 0.01,
        'enable_adaptive': True,
        'enable_clustering': True,
        'cluster_config': {
            'num_clusters': 3,
            'cluster_method': 'kmeans',
            'feature_dim': 10
        }
    }

    device = torch.device('cpu')

    strategy = AdaptiveFederatedStrategy(model_config, fl_config, device)

    print(f"Strategy initialized successfully")
    print(f"Global model parameters: {strategy._count_parameters():,}")
