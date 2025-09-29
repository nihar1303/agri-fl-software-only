"""
Main Experiment Runner for Adaptive Federated Learning for Agricultural IoT
"""

import torch
import numpy as np
import flwr as fl
from flwr.simulation import start_simulation
import sys
import os
import argparse
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import *
from data.prepare_data import prepare_datasets
from client.client import create_client
from server.server import AdaptiveFederatedStrategy
from models.models import create_model


class FederatedExperiment:
    """Main experiment coordinator"""

    def __init__(self, config_name: str = "default"):
        self.config_name = config_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load configurations
        self.experiment_config = EXPERIMENT_CONFIG.copy()
        self.model_config = MODEL_CONFIG.copy()
        self.dataset_config = DATASET_CONFIG.copy()
        self.adaptive_config = ADAPTIVE_CONFIG.copy()
        self.cluster_config = CLUSTER_CONFIG.copy()
        self.fedprox_config = FEDPROX_CONFIG.copy()
        self.compression_config = COMPRESSION_CONFIG.copy()
        self.logging_config = LOGGING_CONFIG.copy()

        print(f"Initialized experiment '{config_name}' on device: {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"Flower version: {fl.__version__}")

    def prepare_data(self):
        """Prepare federated datasets"""

        print("\n=== Data Preparation ===")

        # Combine configs for data preparation
        data_config = {**self.dataset_config, **self.experiment_config}

        # Prepare datasets
        self.client_dataloaders, self.test_dataloader, self.dataset_info = prepare_datasets(data_config)

        # Update model config with dataset info
        self.model_config['num_classes'] = self.dataset_info['num_classes']

        print(f"Data preparation completed:")
        print(f"  - {len(self.client_dataloaders)} client dataloaders created")
        print(f"  - {self.dataset_info['num_classes']} classes detected")
        print(f"  - Test dataset: {self.dataset_info['total_test_size']} samples")

    def create_client_fn(self):
        """Create client function for Flower simulation"""

        def client_fn(cid: str):
            """Create a client instance"""

            client_id = int(cid)

            # Get client's dataloader
            if client_id < len(self.client_dataloaders):
                trainloader = self.client_dataloaders[client_id]
            else:
                # Fallback for extra clients
                trainloader = self.client_dataloaders[client_id % len(self.client_dataloaders)]

            # Use shared test dataloader (in practice, each client might have their own test set)
            testloader = self.test_dataloader

            # Create FL config for client
            fl_config = {
                **self.experiment_config,
                **self.fedprox_config,
                **self.adaptive_config,
                'compression': self.compression_config
            }

            # Create client
            client = create_client(
                client_id=client_id,
                model_config=self.model_config,
                trainloader=trainloader,
                testloader=testloader,
                device=self.device,
                fl_config=fl_config
            )

            return client

        return client_fn

    def create_strategy(self):
        """Create federated learning strategy"""

        print("\n=== Strategy Configuration ===")

        # Combine FL configurations
        fl_config = {
            **self.experiment_config,
            **self.adaptive_config,
            **self.fedprox_config,
            'enable_clustering': self.cluster_config['enable_clustering'],
            'cluster_config': self.cluster_config,
            'compression_config': self.compression_config
        }

        # Create strategy
        strategy = AdaptiveFederatedStrategy(
            model_config=self.model_config,
            fl_config=fl_config,
            device=self.device
        )

        return strategy

    def run_experiment(self, experiment_name: str = None):
        """Run the federated learning experiment"""

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"agri_fl_{timestamp}"

        print(f"\n=== Running Experiment: {experiment_name} ===")

        # Prepare data
        self.prepare_data()

        # Create strategy
        strategy = self.create_strategy()

        # Create client function
        client_fn = self.create_client_fn()

        # Configure simulation
        num_clients = self.experiment_config['num_clients']
        num_rounds = self.experiment_config['num_rounds']

        print(f"Starting federated learning simulation:")
        print(f"  - Clients: {num_clients}")
        print(f"  - Rounds: {num_rounds}")
        print(f"  - Strategy: Adaptive FL with Clustering")
        print(f"  - Device: {self.device}")

        # Run simulation
        try:
            history = start_simulation(
                client_fn=client_fn,
                num_clients=num_clients,
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
                client_resources={"num_cpus": 1, "num_gpus": 0.0}  # Resources per client
            )

            print("\n=== Experiment Completed Successfully ===")

            # Save results
            if self.logging_config['save_results']:
                self.save_experiment_results(experiment_name, strategy, history)

            # Plot results
            if self.logging_config['plot_results']:
                self.plot_results(strategy, history)

            return strategy, history

        except Exception as e:
            print(f"\nExperiment failed with error: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def save_experiment_results(self, experiment_name: str, strategy, history):
        """Save experiment results"""

        print("\n=== Saving Results ===")

        results_dir = self.logging_config['results_dir']
        os.makedirs(results_dir, exist_ok=True)

        # Save strategy results
        strategy_file = os.path.join(results_dir, f"{experiment_name}_strategy.json")
        strategy.save_results(strategy_file)

        # Save simulation history
        history_file = os.path.join(results_dir, f"{experiment_name}_history.json")

        # Convert history to serializable format
        history_data = {
            'losses_distributed': history.losses_distributed,
            'losses_centralized': history.losses_centralized,
            'metrics_distributed': history.metrics_distributed,
            'metrics_centralized': history.metrics_centralized
        }

        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2, default=str)

        # Save experiment configuration
        config_file = os.path.join(results_dir, f"{experiment_name}_config.json")
        experiment_config = {
            'experiment_config': self.experiment_config,
            'model_config': self.model_config,
            'dataset_config': self.dataset_config,
            'adaptive_config': self.adaptive_config,
            'cluster_config': self.cluster_config,
            'fedprox_config': self.fedprox_config,
            'compression_config': self.compression_config,
            'dataset_info': self.dataset_info
        }

        with open(config_file, 'w') as f:
            json.dump(experiment_config, f, indent=2, default=str)

        print(f"Results saved to {results_dir}/")

    def plot_results(self, strategy, history):
        """Plot experiment results"""

        print("\n=== Generating Plots ===")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create subplot figure
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Federated Learning Results', fontsize=16)

            # Plot 1: Global accuracy over rounds
            if history.metrics_distributed:
                rounds = list(range(1, len(history.metrics_distributed) + 1))
                accuracies = [metrics.get('global_accuracy', 0) for _, metrics in history.metrics_distributed]

                axes[0, 0].plot(rounds, accuracies, 'b-', linewidth=2, marker='o')
                axes[0, 0].set_title('Global Accuracy vs Rounds')
                axes[0, 0].set_xlabel('Round')
                axes[0, 0].set_ylabel('Accuracy')
                axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: Global loss over rounds
            if history.losses_distributed:
                rounds = list(range(1, len(history.losses_distributed) + 1))
                losses = [loss for loss, _ in history.losses_distributed]

                axes[0, 1].plot(rounds, losses, 'r-', linewidth=2, marker='s')
                axes[0, 1].set_title('Global Loss vs Rounds')
                axes[0, 1].set_xlabel('Round')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Client participation distribution
            training_history = strategy.get_training_history()
            participation_counts = training_history['participation_counts']

            if participation_counts:
                clients = list(participation_counts.keys())
                counts = list(participation_counts.values())

                axes[1, 0].bar(clients, counts, alpha=0.7)
                axes[1, 0].set_title('Client Participation Distribution')
                axes[1, 0].set_xlabel('Client ID')
                axes[1, 0].set_ylabel('Participation Count')
                axes[1, 0].grid(True, alpha=0.3, axis='y')

            # Plot 4: Accuracy variance over rounds
            if history.metrics_distributed:
                rounds = list(range(1, len(history.metrics_distributed) + 1))
                acc_stds = [metrics.get('accuracy_std', 0) for _, metrics in history.metrics_distributed]

                axes[1, 1].plot(rounds, acc_stds, 'g-', linewidth=2, marker='^')
                axes[1, 1].set_title('Accuracy Standard Deviation vs Rounds')
                axes[1, 1].set_xlabel('Round')
                axes[1, 1].set_ylabel('Accuracy Std')
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            # Save plot
            results_dir = self.logging_config['results_dir']
            plot_file = os.path.join(results_dir, f"experiment_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"Plots saved to {plot_file}")

        except ImportError:
            print("Matplotlib/Seaborn not available. Skipping plots.")
        except Exception as e:
            print(f"Error generating plots: {e}")

    def run_experiment_suite(self):
        """Run a suite of experiments for comparison"""

        print("\n=== Running Experiment Suite ===")

        experiments = [
            ("E0_Baseline_IID", {"non_iid": False, "enable_adaptive": False, "enable_clustering": False}),
            ("E1_NonIID_FedAvg", {"non_iid": True, "enable_adaptive": False, "enable_clustering": False}),
            ("E2_NonIID_FedProx", {"non_iid": True, "enable_adaptive": False, "enable_clustering": False, "enable_fedprox": True}),
            ("E3_Adaptive_Participation", {"non_iid": True, "enable_adaptive": True, "enable_clustering": False}),
            ("E4_Clustered_Personalization", {"non_iid": True, "enable_adaptive": False, "enable_clustering": True}),
            ("E5_Combined_Approach", {"non_iid": True, "enable_adaptive": True, "enable_clustering": True}),
        ]

        results = {}

        for exp_name, config_updates in experiments:
            print(f"\n{'='*50}")
            print(f"Running {exp_name}")
            print(f"{'='*50}")

            # Update configurations
            original_configs = {}
            for key, value in config_updates.items():
                if key in self.dataset_config:
                    original_configs[key] = self.dataset_config[key]
                    self.dataset_config[key] = value
                elif key in self.adaptive_config:
                    original_configs[key] = self.adaptive_config[key]
                    self.adaptive_config[key] = value
                elif key in self.cluster_config:
                    original_configs[key] = self.cluster_config[key]
                    self.cluster_config[key] = value
                elif key in self.fedprox_config:
                    original_configs[key] = self.fedprox_config[key]
                    self.fedprox_config[key] = value

            # Run experiment
            strategy, history = self.run_experiment(exp_name)

            # Store results
            if strategy and history:
                results[exp_name] = {
                    'strategy': strategy,
                    'history': history,
                    'config': config_updates
                }

            # Restore original configurations
            for key, value in original_configs.items():
                if key in self.dataset_config:
                    self.dataset_config[key] = value
                elif key in self.adaptive_config:
                    self.adaptive_config[key] = value
                elif key in self.cluster_config:
                    self.cluster_config[key] = value
                elif key in self.fedprox_config:
                    self.fedprox_config[key] = value

        print(f"\n=== Experiment Suite Completed ===")
        print(f"Completed {len(results)} experiments successfully")

        return results


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(description="Adaptive Federated Learning for Agricultural IoT")
    parser.add_argument("--config", type=str, default="default", help="Configuration name")
    parser.add_argument("--experiment", type=str, default="single", 
                       choices=["single", "suite"], help="Run single experiment or full suite")
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument("--dataset", type=str, default="fakedata", 
                       choices=["fakedata", "plantvillage"], help="Dataset to use")
    parser.add_argument("--rounds", type=int, default=None, help="Number of rounds")
    parser.add_argument("--clients", type=int, default=None, help="Number of clients")

    args = parser.parse_args()

    # Create experiment
    experiment = FederatedExperiment(args.config)

    # Update configurations based on arguments
    if args.dataset:
        experiment.dataset_config['dataset_name'] = args.dataset
    if args.rounds:
        experiment.experiment_config['num_rounds'] = args.rounds
    if args.clients:
        experiment.experiment_config['num_clients'] = args.clients

    # Run experiment(s)
    if args.experiment == "suite":
        results = experiment.run_experiment_suite()
    else:
        strategy, history = experiment.run_experiment(args.name)

    print("\n=== Execution Completed ===")


if __name__ == "__main__":
    main()
