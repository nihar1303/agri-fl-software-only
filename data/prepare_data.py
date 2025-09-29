"""
Data preparation for Agricultural Federated Learning
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from collections import defaultdict


class FakeDataset(Dataset):
    def __init__(self, size=1000, image_shape=(3, 32, 32), num_classes=10, transform=None):
        self.size = size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.transform = transform
        np.random.seed(42)
        self.data = torch.randn(size, *image_shape)
        self.targets = torch.randint(0, num_classes, (size,)).tolist()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform:
            x = transforms.ToPILImage()(x)
            x = self.transform(x)
        return x, y


class PlantVillageDataset(Dataset):
    def __init__(self, root, train=True, transform=None, download=True):
        self.root = root
        self.train = train
        self.transform = transform
        if download:
            self._download()
        self._load()

    def _download(self):
        data_dir = os.path.join(self.root, 'plantville')
        if os.path.exists(data_dir):
            return
        print("PlantVillage dataset not found. Creating fake data for demo.")
        os.makedirs(data_dir, exist_ok=True)
        classes = ['healthy', 'disease_1', 'disease_2', 'disease_3']
        for cls in classes:
            cls_dir = os.path.join(data_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(os.path.join(cls_dir, f'fake_{i}.jpg'))

    def _load(self):
        data_dir = os.path.join(self.root, 'plantville')
        self.data = []
        self.targets = []
        self.classes = []
        if os.path.exists(data_dir):
            self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            class_to_idx = {c: i for i, c in enumerate(self.classes)}
            for c in self.classes:
                c_dir = os.path.join(data_dir, c)
                for fname in os.listdir(c_dir):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.data.append(os.path.join(c_dir, fname))
                        self.targets.append(class_to_idx[c])
        if not self.
            print("No data found. Falling back to fake data.")
            fake_data = FakeDataset(size=1000, image_shape=(3, 224, 224), num_classes=4)
            self.data = fake_data.data
            self.targets = fake_data.targets
            self.classes = [f'class_{i}' for i in range(4)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(self.data[0], str):
            img = Image.open(self.data[idx]).convert('RGB')
        else:
            img = transforms.ToPILImage()(self.data[idx])

        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


def create_noniid_split(dataset, num_clients=10, alpha=0.3, min_samples_per_client=1):
    targets = dataset.targets if hasattr(dataset, 'targets') else [dataset[i][1] for i in range(len(dataset))]
    if type(targets[0]) is torch.Tensor:
        targets = [int(t.item()) for t in targets]
    classes = set(targets)
    idx_by_class = {cls: [] for cls in classes}
    for idx, label in enumerate(targets):
        idx_by_class[label].append(idx)

    np.random.seed(42)
    client_idx = {i: [] for i in range(num_clients)}

    for c in classes:
        c_idxs = idx_by_class[c]
        np.random.shuffle(c_idxs)
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        proportions = (proportions * len(c_idxs)).astype(int)
        # Adjust last to correct rounding error
        proportions[-1] = len(c_idxs) - np.sum(proportions[:-1])

        pointer = 0
        for i, count in enumerate(proportions):
            if count > 0:
                client_idx[i].extend(c_idxs[pointer:pointer + count])
                pointer += count

    # Ensure no empty client
    for i in range(num_clients):
        if len(client_idx[i]) < min_samples_per_client:
            donor = max(client_idx, key=lambda k: len(client_idx[k]))
            if donor != i and len(client_idx[donor]) > min_samples_per_client:
                sample = client_idx[donor].pop()
                client_idx[i].append(sample)

    return [client_idx[i] for i in range(num_clients)]


def prepare_datasets(config):
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    ds_name = config.get('dataset', 'fake')
    data_root = config.get('data_path', './data')

    if ds_name == 'fake':
        train_ds = FakeDataset(size=5000, transform=train_transform)
        test_ds = FakeDataset(size=1000, transform=test_transform)
        num_classes = 10
    elif ds_name == 'plantvilllage':
        train_ds = PlantVillageDataset(data_root, train=True, transform=train_transform)
        test_ds = PlantVillageDataset(data_root, train=False, transform=test_transform)
        num_classes = len(train_ds.classes)
    else:
        raise ValueError(f"Unknown dataset {ds_name}")

    num_clients = config.get('num_clients', 10)
    batch_size = config.get('batch_size', 32)
    alpha = config.get('alpha', 0.3)
    noniid = config.get('non_iid', True)

    if noniid:
        print(f"Using non-iid split with alpha={alpha}")
        client_splits = create_noniid_split(train_ds, num_clients, alpha)
    else:
        idxs = np.arange(len(train_ds))
        np.random.shuffle(idxs)
        client_splits = np.array_split(idxs, num_clients)

    client_loaders = []
    for client_idxs in client_splits:
        subset = Subset(train_ds, list(client_idxs))
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))

    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    info = {
        'num_classes': num_classes,
        'num_clients': num_clients,
        'client_data_sizes': [len(c) for c in client_splits],
        'train_size': len(train_ds),
        'test_size': len(test_ds)
    }

    print(f"Prepared dataset with {num_clients} clients")
    print(f"Training samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    return client_loaders, test_loader, info


if __name__ == "__main__":
    cfg = {
        'dataset': 'fake',
        'num_clients': 5,
        'batch_size': 32,
        'non_iid': True,
        'alpha': 0.3,
        'data_path': './data'
    }
    clients, test, metadata = prepare_datasets(cfg)
    print(f"Data prepared: {len(clients)} clients, {len(test.dataset)} test samples")
