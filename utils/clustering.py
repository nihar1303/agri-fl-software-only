"""
Clustering utilities for federated personalization
"""

import numpy as np
import torch
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any
import pickle
import os
from collections import defaultdict


class ClientClusteringManager:
    """Manages client clustering for personalized federated learning"""

    def __init__(self, num_clusters: int = 3, method: str = 'kmeans', 
                 feature_dim: int = 10):
        self.num_clusters = num_clusters
        self.method = method
        self.feature_dim = feature_dim
        self.clusterer = None
        self.client_clusters = {}
        self.cluster_centers = None

    def extract_client_features(self, client_data: List[Any], 
                               client_models: Dict[int, Any] = None) -> np.ndarray:
        """
        Extract feature vectors for each client

        Args:
            client_data: List of client datasets or data loaders
            client_models: Optional client models for model-based features

        Returns:
            numpy array of client features (num_clients x feature_dim)
        """

        num_clients = len(client_data)
        features = np.zeros((num_clients, self.feature_dim))

        for client_id in range(num_clients):
            dataloader = client_data[client_id]

            # Extract statistical features from data
            all_data = []
            all_labels = []

            for batch_x, batch_y in dataloader:
                all_data.append(batch_x.numpy())
                all_labels.append(batch_y.numpy())

            if all_data:
                data = np.concatenate(all_data, axis=0)
                labels = np.concatenate(all_labels, axis=0)

                # Feature 1-3: Mean, std, variance of pixel intensities
                features[client_id, 0] = np.mean(data)
                features[client_id, 1] = np.std(data)
                features[client_id, 2] = np.var(data)

                # Feature 4-6: RGB channel means
                for c in range(min(3, data.shape[1])):
                    features[client_id, 3 + c] = np.mean(data[:, c, :, :])

                # Feature 7: Number of unique classes
                features[client_id, 6] = len(np.unique(labels))

                # Feature 8: Label distribution entropy
                label_counts = np.bincount(labels)
                label_probs = label_counts / len(labels)
                label_probs = label_probs[label_probs > 0]  # Remove zeros
                features[client_id, 7] = -np.sum(label_probs * np.log(label_probs))

                # Feature 9-10: Dataset size and dominant class ratio
                features[client_id, 8] = len(data)
                if len(label_counts) > 0:
                    features[client_id, 9] = np.max(label_counts) / len(labels)

        return features

    def cluster_clients(self, client_features: np.ndarray) -> Dict[int, int]:
        """
        Cluster clients based on their feature vectors

        Args:
            client_features: Feature matrix (num_clients x feature_dim)

        Returns:
            Dictionary mapping client_id to cluster_id
        """

        if self.method == 'kmeans':
            self.clusterer = KMeans(n_clusters=self.num_clusters, 
                                   random_state=42, n_init=10)
        else:
            raise ValueError(f"Unsupported clustering method: {self.method}")

        # Perform clustering
        cluster_labels = self.clusterer.fit_predict(client_features)
        self.cluster_centers = self.clusterer.cluster_centers_

        # Create client-to-cluster mapping
        self.client_clusters = {client_id: int(cluster_labels[client_id]) 
                               for client_id in range(len(cluster_labels))}

        return self.client_clusters

    def get_cluster_members(self, cluster_id: int) -> List[int]:
        """Get list of client IDs in a specific cluster"""
        return [client_id for client_id, cid in self.client_clusters.items() 
                if cid == cluster_id]

    def get_cluster_statistics(self) -> Dict[str, Any]:
        """Get clustering statistics"""
        if not self.client_clusters:
            return {}

        cluster_sizes = defaultdict(int)
        for cluster_id in self.client_clusters.values():
            cluster_sizes[cluster_id] += 1

        return {
            'num_clusters': self.num_clusters,
            'cluster_sizes': dict(cluster_sizes),
            'total_clients': len(self.client_clusters),
            'clustering_method': self.method
        }

    def save_clustering(self, filepath: str):
        """Save clustering results to file"""
        clustering_data = {
            'client_clusters': self.client_clusters,
            'cluster_centers': self.cluster_centers,
            'num_clusters': self.num_clusters,
            'method': self.method,
            'feature_dim': self.feature_dim
        }

        with open(filepath, 'wb') as f:
            pickle.dump(clustering_data, f)

    def load_clustering(self, filepath: str):
        """Load clustering results from file"""
        with open(filepath, 'rb') as f:
            clustering_data = pickle.load(f)

        self.client_clusters = clustering_data['client_clusters']
        self.cluster_centers = clustering_data['cluster_centers']
        self.num_clusters = clustering_data['num_clusters']
        self.method = clustering_data['method']
        self.feature_dim = clustering_data['feature_dim']


class ClusterPersonalizationAggregator:
    """Aggregates model updates within clusters for personalization"""

    def __init__(self, clustering_manager: ClientClusteringManager):
        self.clustering_manager = clustering_manager
        self.cluster_models = {}

    def aggregate_cluster_updates(self, client_updates: Dict[int, Any], 
                                 global_model_params: Any) -> Dict[int, Any]:
        """
        Aggregate updates within each cluster

        Args:
            client_updates: Dictionary mapping client_id to model parameters
            global_model_params: Global model parameters as baseline

        Returns:
            Dictionary mapping cluster_id to aggregated parameters
        """

        cluster_updates = defaultdict(list)
        cluster_weights = defaultdict(list)

        # Group updates by cluster
        for client_id, params in client_updates.items():
            cluster_id = self.clustering_manager.client_clusters.get(client_id, 0)
            cluster_updates[cluster_id].append(params)
            cluster_weights[cluster_id].append(1.0)  # Equal weighting for now

        # Aggregate within each cluster
        aggregated_clusters = {}

        for cluster_id, updates in cluster_updates.items():
            if updates:
                # Simple FedAvg within cluster
                total_weight = sum(cluster_weights[cluster_id])

                aggregated_params = {}
                for param_name in updates[0].keys():
                    weighted_sum = sum(weight * params[param_name] 
                                     for params, weight in zip(updates, cluster_weights[cluster_id]))
                    aggregated_params[param_name] = weighted_sum / total_weight

                aggregated_clusters[cluster_id] = aggregated_params
            else:
                # Fallback to global model if no updates in cluster
                aggregated_clusters[cluster_id] = global_model_params

        return aggregated_clusters

    def get_personalized_model(self, client_id: int, 
                              cluster_models: Dict[int, Any],
                              global_model: Any) -> Any:
        """
        Get personalized model for a specific client

        Args:
            client_id: Client identifier
            cluster_models: Dictionary of cluster-specific model parameters
            global_model: Global model as fallback

        Returns:
            Personalized model parameters for the client
        """

        cluster_id = self.clustering_manager.client_clusters.get(client_id, 0)

        if cluster_id in cluster_models:
            return cluster_models[cluster_id]
        else:
            return global_model


def visualize_clustering(client_features: np.ndarray, 
                        client_clusters: Dict[int, int],
                        save_path: str = None):
    """
    Visualize client clustering results

    Args:
        client_features: Feature matrix
        client_clusters: Client-to-cluster mapping
        save_path: Optional path to save the plot
    """

    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        # Reduce dimensionality for visualization
        if client_features.shape[1] > 2:
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(client_features)
        else:
            features_2d = client_features

        # Create scatter plot
        plt.figure(figsize=(10, 8))

        unique_clusters = set(client_clusters.values())
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_clusters)))

        for i, cluster_id in enumerate(unique_clusters):
            cluster_clients = [cid for cid, cluster in client_clusters.items() 
                             if cluster == cluster_id]
            cluster_features = features_2d[cluster_clients]

            plt.scatter(cluster_features[:, 0], cluster_features[:, 1], 
                       c=[colors[i]], label=f'Cluster {cluster_id}', 
                       s=100, alpha=0.7)

        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Client Clustering Visualization')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    except ImportError:
        print("Matplotlib not available. Skipping visualization.")


if __name__ == "__main__":
    # Test clustering functionality
    print("Testing clustering utilities...")

    # Generate fake client features
    num_clients = 8
    feature_dim = 10
    np.random.seed(42)

    # Create clusters in feature space
    cluster_centers = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]])

    client_features = []
    for i in range(num_clients):
        cluster_id = i % 3
        noise = np.random.normal(0, 0.1, feature_dim)
        feature = cluster_centers[cluster_id] + noise
        client_features.append(feature)

    client_features = np.array(client_features)

    # Test clustering
    clustering_manager = ClientClusteringManager(num_clusters=3)
    client_clusters = clustering_manager.cluster_clients(client_features)

    print(f"Client clusters: {client_clusters}")
    print(f"Clustering statistics: {clustering_manager.get_cluster_statistics()}")

    # Test visualization
    visualize_clustering(client_features, client_clusters)
