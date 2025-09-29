"""
Neural network models for federated learning experiments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import OrderedDict


class SimpleCNN(nn.Module):
    """Simple CNN for agricultural image classification"""

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Calculate the size after convolutions for 32x32 input
        # After 3 pooling operations: 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ResNetBlock(nn.Module):
    """Basic ResNet block"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = F.relu(out)

        return out


class SimpleResNet(nn.Module):
    """Lightweight ResNet for federated learning"""

    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        super(SimpleResNet, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))

        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def create_model(model_type: str, num_classes: int, input_channels: int = 3):
    """Factory function to create models"""

    if model_type.lower() == 'cnn':
        return SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    elif model_type.lower() == 'resnet':
        return SimpleResNet(num_classes=num_classes, input_channels=input_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_parameters(model: nn.Module) -> OrderedDict:
    """Get model parameters as OrderedDict"""
    return OrderedDict(model.named_parameters())


def set_model_parameters(model: nn.Module, parameters: OrderedDict):
    """Set model parameters from OrderedDict"""
    model.load_state_dict(parameters, strict=False)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing models...")

    # Test SimpleCNN
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"SimpleCNN output shape: {out.shape}")
    print(f"SimpleCNN parameters: {count_parameters(model):,}")

    # Test SimpleResNet
    model = SimpleResNet(num_classes=10)
    out = model(x)
    print(f"SimpleResNet output shape: {out.shape}")
    print(f"SimpleResNet parameters: {count_parameters(model):,}")
