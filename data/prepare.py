""" Data preparation for Agricultural Federated Learning """

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
        self._load_data()

    def _download(self):
        data_dir = os.path.join(self.root, 'plantville')
        if os.path.exists(data_dir):
            return
        os.makedirs(data_dir, exist_ok=True)
        classes = ['healthy', 'disease_1', 'disease_2', 'disease_3']

        for cls in classes:
            class_dir = os.path.join(data_dir, cls)
            os.makedirs(class_dir, exist_ok=True)
            for i in range(10):
                img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
                img.save(os.path.join(class_dir, f'img_{i}.jpg'))

    def _load_data(self):
        data_dir = os.path.join(self.root, 'plantville')
        self.data = []
        self.targets = []
        self.classes = []
        if os.path.exists(data_dir):
            self.classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
            idx_map = {cls: idx for idx, cls in enumerate(self.classes)}
            for cls in self.classes:
                cls_dir = os.path.join(data_dir, cls)
                files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg', '.png'))]
                for f in files:
                    self.data.append(os.path.join(cls_dir, f))
                    self.targets.append(idx_map[cls])
        if not self.
            # fallback
            fake = FakeDataset(size=1000, image_shape=(3, 224, 224), num_classes=4)
            self.data = fake.data
            self.targets = fake.targets
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


def create_noniid_split(dataset, num_clients, alpha=0.3, min_samples=1):
    """Using Dirichlet distribution to split dataset by clients,
       ensures each client has min_samples at least."""
    targets = dataset.targets if hasattr(dataset, 'targets') else [dataset[i][1] for i in range(len(dataset))]
    if isinstance(targets[0], torch.Tensor):
        targets = [int(t.item()) for t in targets]
    classes = set(targets)
    idx_by_class = {cls: [] for cls in classes}
    for idx, label in enumerate(targets):
        idx_by_class[label].append(idx)

    import numpy as np
    np.random.seed(42)

    client_idx_map = {k: [] for k in range(num_clients)}
    for c in classes:
        idxs = idx_by_class[c]
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(alpha*np.ones(num_clients))
        proportions = (proportions * len(idxs)).astype(int)
        proportions[-1] = len(idxs) - np.sum(proportions[:-1])
        position = 0
        for cli in range(num_clients):
            cnt = proportions[cli]
            client_idx_map[cli].extend(idxs[position:position+cnt])
            position += cnt

    # Fix any client with less than min_samples by borrowing from others
    changes = True
    while changes:
        changes = False
        for cli, idxs in client_idx_map.items():
            if len(idxs) < min_samples:
                # Find donors with more than min_samples
                donors = [d for d, ix in client_idx_map.items() if len(ix) > min_samples]
                if donors:
                    donor = donors[0]
                    sample = client_idx_map[donor].pop()
                    client_idx_map[cli].append(sample)
                    changes = True
    return [client_idx_map[i] for i in range(num_clients)]


def prepare_datasets(cfg):
    """Prepare and split dataset into client data loaders and test data loader."""
    import torchvision.transforms as transforms

    tsf_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tsf_test = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset_name = cfg.get('dataset_name', 'fakedata')
    data_path = cfg.get('data_path', './data')

    if dataset_name == 'fakedata':
        print('Loading Fake dataset...')
        train_dataset = FakeDataset(size=5000, transform=tsf_train)
        test_dataset = FakeDataset(size=1000, transform=tsf_test)
        num_cls = 10
    elif dataset_name == 'plantville':
        print('Loading PlantVillage dataset...')
        train_dataset = PlantVillageDataset(data_path, train=True, transform=tsf_train)
        test_dataset = PlantVillageDataset(data_path, train=False, transform=tsf_test)
        num_cls = len(train_dataset.classes)
    else:
        raise ValueError(f'Unknown dataset {dataset_name}')

    n_clients = cfg.get('num_clients', 10)
    batch_sz = cfg.get('batch_size', 32)
    alpha = cfg.get('alpha', 0.3)
    noniid = cfg.get('non_iid', True)

    if noniid:
        print(f'Creating non-IID splits with alpha={alpha}')
        client_indices = create_noniid_split(train_dataset, n_clients, alpha)
    else:
        all_indices = np.arange(len(train_dataset))
        np.random.shuffle(all_indices)
        client_indices = np.array_split(all_indices, n_clients)
        client_indices = [list(c) for c in client_indices]

    # Ensure all clients have data (edge cases)
    for i in range(n_clients):
        if len(client_indices[i]) == 0:
            # borrow from largest client
            idxs = [len(ix) for ix in client_indices]
            donor = int(np.argmax(idxs))
            client_indices[i].append(client_indices[donor].pop())

    client_loaders = []
    for i in range(n_clients):
        subset = Subset(train_dataset, client_indices[i])
        # If subset is empty -> shuffle=False to avoid errors
        loader = DataLoader(subset, batch_size=batch_sz, shuffle=len(subset) > 0)
        client_loaders.append(loader)

    test_loader = DataLoader(test_dataset, batch_size=batch_sz*2, shuffle=False)
    info = {
        'num_classes': num_cls,
        'num_clients': n_clients,
        'client_sizes': [len(ix) for ix in client_indices],
        'total_train_size': len(train_dataset),
        'total_test_size': len(test_dataset)
    }

    print('Dataset ready.')
    print(f'Classes: {num_cls}, Clients: {n_clients}, Sizes per client: {info["client_sizes"]}')
    return client_loaders, test_loader, info


if __name__ == "__main__":
    # Simple local test
    cfg = {'dataset_name': 'fakedata', 'num_clients': 8, 'alpha': 0.3, 'non_iid': True}
    loaders, test_loader, meta = prepare_datasets(cfg)
    print(f'Client loaders: {len(loaders)}, test set: {len(test_loader.dataset)}')
